import math

import torch
import torch.nn.functional as F
from torch import nn


def make_uniform_grid(grid_size: int) -> torch.Tensor:
    coordinates = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
    coordinates = coordinates - 0.5
    return torch.stack(
        torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), dim=-1
    ).flatten(0, 2)


class FourierPositionEncoder(nn.Module):
    def __init__(self, output_dim: int, num_frequencies: int = 4) -> None:
        super().__init__()

        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_frequencies, dtype=torch.float32),
            persistent=False,
        )
        input_dim = 3 + 6 * num_frequencies
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * coordinates[..., None] * self.frequencies
        encoding = torch.cat(
            [coordinates, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1
        )
        return self.net(encoding)


class PointViewCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        geometry_dim: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()

        if width % num_heads != 0:
            raise ValueError("width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = width // num_heads

        self.point_semantic_norm = nn.LayerNorm(width)
        self.point_geometry_norm = nn.LayerNorm(width)
        self.semantic_norm = nn.LayerNorm(width)
        self.semantic_query = nn.Linear(width, width)
        self.geometry_query = nn.Linear(width, width)
        self.semantic_key = nn.Linear(width, width)
        self.geometry_key = nn.Linear(geometry_dim, width)
        self.semantic_value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

        self.output_norm = nn.LayerNorm(width)
        hidden_dim = round(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            # Preserve checkpoint indices and the original compiled inference graph.
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, width),
        )

    def forward(
        self,
        point_features: torch.Tensor,
        point_context: torch.Tensor,
        semantic_features: torch.Tensor,
        geometry_features: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_views, num_points, _ = semantic_features.shape
        normalized_semantics = self.semantic_norm(semantic_features)

        semantic_query = self.semantic_query(
            self.point_semantic_norm(point_features + point_context)
        ).reshape(batch_size, num_points, self.num_heads, self.head_dim)
        geometry_query = self.geometry_query(
            self.point_geometry_norm(point_context)
        ).reshape(batch_size, num_points, self.num_heads, self.head_dim)
        semantic_key = self.semantic_key(normalized_semantics).reshape(
            batch_size, num_views, num_points, self.num_heads, self.head_dim
        )
        geometry_key = self.geometry_key(geometry_features).reshape_as(semantic_key)
        values = self.semantic_value(normalized_semantics).reshape_as(semantic_key)

        semantic_scores = (semantic_query[:, None] * semantic_key).sum(dim=-1)
        geometry_scores = (geometry_query[:, None] * geometry_key).sum(dim=-1)
        scores = (semantic_scores + geometry_scores) / math.sqrt(2.0 * self.head_dim)
        scores = scores.masked_fill(
            ~view_valid[..., None], -torch.finfo(scores.dtype).max
        )
        # Keep the original zero-probability op for compiled bf16 numerical parity.
        weights = F.dropout(scores.softmax(dim=1), p=0.0, training=False)
        pooled = (weights[..., None] * values).sum(dim=1).flatten(-2)

        point_features = point_features + self.output(pooled)
        point_features = point_features + self.mlp(self.output_norm(point_features))

        return point_features


class PointViewCrossAttention(nn.Module):
    def __init__(
        self,
        width: int,
        geometry_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                PointViewCrossAttentionBlock(
                    width,
                    geometry_dim=geometry_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.null_observation = nn.Parameter(torch.empty(1, 1, 1, width))
        nn.init.trunc_normal_(self.null_observation, std=0.02)

    def forward(
        self,
        point_context: torch.Tensor,
        semantic_features: torch.Tensor,
        geometry_features: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, num_points, _ = semantic_features.shape
        null_observation = self.null_observation.expand(batch_size, -1, num_points, -1)
        semantic_features = torch.cat([semantic_features, null_observation], dim=1)
        null_geometry = geometry_features.new_zeros(
            batch_size, 1, num_points, geometry_features.shape[-1]
        )
        geometry_features = torch.cat([geometry_features, null_geometry], dim=1)
        view_valid = torch.cat(
            [
                view_valid,
                torch.ones(
                    batch_size,
                    1,
                    num_points,
                    device=view_valid.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )

        point_features = torch.zeros_like(point_context)
        for block in self.blocks:
            point_features = block(
                point_features,
                point_context,
                semantic_features,
                geometry_features,
                view_valid,
            )

        return point_features


class PointLifter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        geometry_feature_dim: int,
        point_grid_size: int,
        roi_size: float,
        view_num_heads: int,
        view_depth: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()

        self.point_grid_size = point_grid_size
        self.roi_size = roi_size
        self.register_buffer(
            "point_grid", make_uniform_grid(point_grid_size), persistent=False
        )
        self.position_encoder = FourierPositionEncoder(feature_dim)
        self.view_fusion = PointViewCrossAttention(
            feature_dim,
            geometry_feature_dim,
            view_num_heads,
            depth=view_depth,
            mlp_ratio=mlp_ratio,
        )

    def _project_points(
        self,
        points: torch.Tensor,
        cam_extrinsics: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_points = (
            points[:, None] @ cam_extrinsics[..., :3, :3].transpose(-1, -2)
            + cam_extrinsics[..., None, :3, 3]
        )
        homogeneous_pixels = camera_points @ cam_intrinsics.transpose(-1, -2)
        depth = camera_points[..., 2]
        denominator = depth.clamp_min(torch.finfo(depth.dtype).eps)
        pixels_x = homogeneous_pixels[..., 0] / denominator
        pixels_y = homogeneous_pixels[..., 1] / denominator
        grid_x = 2.0 * (pixels_x + 0.5) / image_width - 1.0
        grid_y = 2.0 * (pixels_y + 0.5) / image_height - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        inside_image = (
            (depth > 0.0)
            & (grid_x >= -1.0)
            & (grid_x <= 1.0)
            & (grid_y >= -1.0)
            & (grid_y <= 1.0)
        )
        return grid, inside_image

    def _sample_images(self, images: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, channels = images.shape[:3]
        num_points = grid.shape[2]
        flat_images = images.flatten(0, 1)
        flat_grid = grid.flatten(0, 1).to(dtype=images.dtype)
        sampled = F.grid_sample(
            flat_images,
            flat_grid.unsqueeze(-2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[..., 0]
        return sampled.reshape(batch_size, num_views, channels, num_points).permute(
            0, 1, 3, 2
        )

    def forward(
        self,
        image_features: torch.Tensor,
        ray_features: torch.Tensor,
        roi_centers: torch.Tensor,
        valid_views: torch.Tensor,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        masks = inputs["masks"]
        batch_size, _, image_height, image_width = masks.shape
        cam_extrinsics = inputs["cam_extrinsics"].float()
        cam_intrinsics = inputs["cam_intrinsics"].float()

        with torch.autocast(device_type=masks.device.type, enabled=False):
            points = roi_centers[:, None] + self.roi_size * self.point_grid[None]
            grid, inside_image = self._project_points(
                points, cam_extrinsics, cam_intrinsics, image_height, image_width
            )

        semantic_features = self._sample_images(image_features, grid)
        geometry_features = self._sample_images(ray_features, grid)
        point_context = self.position_encoder(
            self.point_grid.to(dtype=semantic_features.dtype)
        )
        point_context = point_context[None].expand(batch_size, -1, -1)
        view_valid = valid_views[..., None] & inside_image
        point_features = self.view_fusion(
            point_context, semantic_features, geometry_features, view_valid
        )

        volume_shape = (
            batch_size,
            self.point_grid_size,
            self.point_grid_size,
            self.point_grid_size,
        )
        feature_volume = point_features.reshape(*volume_shape, -1).permute(
            0, 4, 1, 2, 3
        )

        return feature_volume
