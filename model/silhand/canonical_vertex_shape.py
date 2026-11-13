from pathlib import Path

import torch
from torch import nn
from util.amano_hand_utils import AMANO
from util.pose_utils import identity_global_pose


def _make_neutral_shape_system(
    hand_side: str, mano_assets_root_dir: str | Path
) -> tuple[torch.Tensor, torch.Tensor]:
    amano = AMANO(hand_side, mano_assets_root_dir).to(dtype=torch.float64)
    num_shape_params = amano.num_shape_parameters
    num_probes = num_shape_params + 1
    pose_params = torch.zeros(
        num_probes, amano.num_pose_parameters, dtype=torch.float64
    )
    shape_params = torch.zeros(num_probes, num_shape_params, dtype=torch.float64)
    shape_params[1:] = torch.eye(num_shape_params, dtype=torch.float64)
    global_pose = identity_global_pose(num_probes, pose_params)
    with torch.no_grad():
        vertices, _ = amano(pose_params, shape_params, global_pose)
    mean_vertices = vertices[0]
    vertex_basis = (vertices[1:] - mean_vertices).permute(1, 2, 0)
    return mean_vertices, vertex_basis


class MANOShapeProjector(nn.Module):
    def __init__(
        self, hand_side: str, mano_assets_root_dir: str | Path, ridge: float = 0.0
    ) -> None:
        super().__init__()

        mean_vertices, vertex_basis = _make_neutral_shape_system(
            hand_side, mano_assets_root_dir
        )
        centered_vertex_basis = vertex_basis - vertex_basis.mean(dim=0, keepdim=True)
        centered_basis = centered_vertex_basis.flatten(0, 1)
        if ridge > 0.0:
            normal_matrix = centered_basis.mT @ centered_basis
            identity = torch.eye(len(normal_matrix), dtype=normal_matrix.dtype)
            projection_matrix = torch.linalg.solve(
                normal_matrix + ridge * identity, centered_basis.mT
            )
        else:
            projection_matrix = torch.linalg.pinv(centered_basis)

        self.register_buffer(
            "mean_vertices", mean_vertices.to(torch.float32), persistent=False
        )
        self.register_buffer(
            "projection_matrix", projection_matrix.to(torch.float32), persistent=False
        )

    @torch.no_grad()
    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        displacements = vertices - self.mean_vertices
        centered_displacements = displacements - displacements.mean(dim=1, keepdim=True)
        return centered_displacements.flatten(1) @ self.projection_matrix.mT


class CanonicalVertexShapeDecoder(nn.Module):
    def __init__(
        self,
        hand_side: str,
        mano_assets_root_dir: str | Path,
        shape_latent_dim: int,
        width: int,
        displacement_scale: float,
    ) -> None:
        super().__init__()

        mean_vertices, _ = _make_neutral_shape_system(hand_side, mano_assets_root_dir)
        self.displacement_scale = displacement_scale
        self.num_vertices = len(mean_vertices)
        self.register_buffer(
            "mean_vertices", mean_vertices.to(torch.float32), persistent=False
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(shape_latent_dim),
            nn.Linear(shape_latent_dim, width),
            nn.GELU(),
        )
        self.output = nn.Linear(width, 3 * self.num_vertices)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, shape_latent: torch.Tensor) -> torch.Tensor:
        displacements = self.displacement_scale * self.output(
            self.decoder(shape_latent)
        ).reshape(len(shape_latent), self.num_vertices, 3)
        return self.mean_vertices + displacements
