import torch
import torch.nn.functional as F
from torch import nn
from util.pose_utils import (
    global_pose_from_rotation_translation,
    global_pose_to_rotation_translation,
)

from .point_lifter import FourierPositionEncoder, make_uniform_grid


def _depthwise_conv3d_3x3x3(
    inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # Expose the 27 weighted neighbors so torch.compile can fuse the convolution.
    dtype = (
        torch.get_autocast_dtype(inputs.device.type)
        if torch.is_autocast_enabled(inputs.device.type)
        else inputs.dtype
    )
    values = F.pad(inputs.to(dtype), (1, 1, 1, 1, 1, 1))
    weights = weight.to(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        values = values.float()
        weights = weights.float()
    depth, height, width = inputs.shape[-3:]
    output = 0
    for z in range(3):
        for y in range(3):
            for x in range(3):
                output = output + (
                    values[..., z : z + depth, y : y + height, x : x + width]
                    * weights[:, 0, z, y, x][None, :, None, None, None]
                )
    output = output.to(dtype)
    return output + bias.to(dtype)[None, :, None, None, None]


class GlobalResponseNorm3d(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()

        self.gamma = nn.Parameter(torch.zeros(width))
        self.beta = nn.Parameter(torch.zeros(width))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(inputs, ord=2, dim=(1, 2, 3), keepdim=True)
        response = response / (response.mean(dim=-1, keepdim=True) + 1.0e-6)
        return inputs + self.gamma * (inputs * response) + self.beta


class ConvNeXtV2Block3d(nn.Module):
    def __init__(self, width: int, kernel_size: int, mlp_ratio: float) -> None:
        super().__init__()

        hidden_dim = round(width * mlp_ratio)
        self.depthwise_conv = nn.Conv3d(
            width,
            width,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=width,
        )
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            GlobalResponseNorm3d(hidden_dim),
            nn.Linear(hidden_dim, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if (
            inputs.is_cuda
            and torch.compiler.is_compiling()
            and self.depthwise_conv.kernel_size == (3, 3, 3)
        ):
            features = _depthwise_conv3d_3x3x3(
                inputs, self.depthwise_conv.weight, self.depthwise_conv.bias
            )
        else:
            features = self.depthwise_conv(inputs)
        features = features.permute(0, 2, 3, 4, 1)
        features = self.mlp(self.norm(features)).permute(0, 4, 1, 2, 3)
        return inputs + features


class SpatialVolumeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        kernel_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Conv3d(input_dim, width, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                ConvNeXtV2Block3d(
                    width,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input_projection(inputs))


class WorldKeypointPosterior(nn.Module):
    def __init__(self, width: int, num_keypoints: int, grid_size: int) -> None:
        super().__init__()

        self.num_keypoints = num_keypoints
        self.offset_scale = 0.5 / grid_size
        self.logit_head = nn.Conv3d(width, num_keypoints, kernel_size=1)
        self.offset_head = nn.Conv3d(width, 3 * num_keypoints, kernel_size=1)
        nn.init.trunc_normal_(self.logit_head.weight, std=0.02)
        nn.init.zeros_(self.logit_head.bias)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def forward(
        self,
        volume_features: torch.Tensor,
        normalized_cell_coordinates: torch.Tensor,
        roi_centers: torch.Tensor,
        roi_size: float,
    ) -> torch.Tensor:
        batch_size = len(volume_features)
        logits = self.logit_head(volume_features).flatten(2)
        offsets = self.offset_head(volume_features).reshape(
            batch_size, self.num_keypoints, 3, -1
        )
        with torch.autocast(device_type=volume_features.device.type, enabled=False):
            probabilities = logits.float().softmax(dim=-1)
            offsets = self.offset_scale * torch.tanh(offsets.float()).transpose(-1, -2)
            normalized_points = (
                normalized_cell_coordinates.float()[None, None] + offsets
            )
            normalized_keypoints = (probabilities[..., None] * normalized_points).sum(
                dim=-2
            )
            return roi_centers.float()[:, None] + roi_size * normalized_keypoints


class RigidPalmPoseEstimator(nn.Module):
    def __init__(
        self,
        palm_joint_indices: list[int],
        canonical_palm_template: torch.Tensor,
        centering: str = "wrist",
    ) -> None:
        super().__init__()

        if centering not in ("wrist", "centroid"):
            raise ValueError(f"Unknown palm-pose centering: {centering}")

        self.register_buffer(
            "palm_joint_indices", torch.tensor(palm_joint_indices, dtype=torch.long)
        )
        self.register_buffer("canonical_palm_template", canonical_palm_template)
        self.centering = centering

    @torch.compiler.disable
    def forward(self, world_keypoints: torch.Tensor) -> torch.Tensor:
        world_palm_joints = world_keypoints.index_select(1, self.palm_joint_indices)
        if self.centering == "wrist":
            world_center = world_keypoints[:, :1]
            canonical_center = self.canonical_palm_template[:1]
        else:
            world_center = world_palm_joints.mean(dim=1, keepdim=True)
            canonical_center = self.canonical_palm_template.mean(dim=0, keepdim=True)
        world_vectors = world_palm_joints - world_center
        canonical_vectors = self.canonical_palm_template - canonical_center
        covariance = canonical_vectors.transpose(-1, -2) @ world_vectors
        u, _, vh = torch.linalg.svd(covariance)
        correction = torch.ones(len(world_keypoints), 3, device=u.device, dtype=u.dtype)
        correction[:, -1] = torch.where(torch.linalg.det(u @ vh) < 0.0, -1.0, 1.0)
        canonical_to_world_row = (u * correction[:, None]) @ vh
        rotation = canonical_to_world_row.transpose(-1, -2)
        translation = world_keypoints[:, 0]
        return global_pose_from_rotation_translation(rotation, translation)


class QueryDecoderBlock(nn.Module):
    def __init__(self, width: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()

        self.self_attention_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.cross_attention_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(width)
        hidden_dim = round(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            # Preserve checkpoint indices and the original compiled inference graph.
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, width),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.self_attention_norm(queries)
        attended, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        queries = queries + attended
        normalized_queries = self.cross_attention_norm(queries)
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(
            normalized_queries,
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.mlp(self.output_norm(queries))


class CanonicalTokenDecoder(nn.Module):
    def __init__(
        self,
        num_pose_params: int,
        volume_width: int,
        width: int,
        grid_size: int,
        world_roi_size: float,
        canonical_roi_size: float,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()

        self.pose_tokens = nn.Parameter(torch.empty(1, num_pose_params, width))
        self.shape_token = nn.Parameter(torch.empty(1, 1, width))
        self.memory_projection = (
            nn.Identity() if volume_width == width else nn.Linear(volume_width, width)
        )
        self.position_encoder = FourierPositionEncoder(width)
        self.world_roi_size = world_roi_size
        self.canonical_roi_size = canonical_roi_size
        self.register_buffer(
            "world_grid", make_uniform_grid(grid_size), persistent=False
        )
        self.blocks = nn.ModuleList(
            [QueryDecoderBlock(width, num_heads, mlp_ratio) for _ in range(depth)]
        )
        nn.init.trunc_normal_(self.pose_tokens, std=0.02)
        nn.init.trunc_normal_(self.shape_token, std=0.02)

    def forward(
        self,
        world_volume: torch.Tensor,
        roi_centers: torch.Tensor,
        global_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = world_volume.flatten(2).transpose(1, 2)
        memory = self.memory_projection(memory)
        with torch.autocast(device_type=world_volume.device.type, enabled=False):
            rotation, translation = global_pose_to_rotation_translation(
                global_pose.detach().float()
            )
            world_points = roi_centers.float()[:, None] + (
                self.world_roi_size * self.world_grid[None]
            )
            canonical_points = (world_points - translation[:, None]) @ rotation
            canonical_coordinates = canonical_points / self.canonical_roi_size
        position_features = self.position_encoder(canonical_coordinates)
        memory = memory + position_features.to(dtype=memory.dtype)
        queries = torch.cat(
            [
                self.pose_tokens.expand(len(world_volume), -1, -1),
                self.shape_token.expand(len(world_volume), -1, -1),
            ],
            dim=1,
        )
        for block in self.blocks:
            queries = block(queries, memory)
        return queries[:, :-1], queries[:, -1]
