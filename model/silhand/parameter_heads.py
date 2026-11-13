import torch
import torch.nn.functional as F
from torch import nn


class PoseParamHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        pose_param_limits: list[list[float]],
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.feature_norm = nn.LayerNorm(latent_dim)
        self.feature_proj = (
            nn.Identity()
            if feature_dim == latent_dim
            else nn.Linear(latent_dim, feature_dim)
        )
        self.output_norm = nn.LayerNorm(feature_dim)

        layers: list[nn.Module] = []
        input_dim = feature_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.GELU()])
            input_dim = hidden_dim
        output = nn.Linear(input_dim, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.output = nn.Sequential(*layers)

        limits = torch.tensor(pose_param_limits, dtype=torch.float32)
        centers = 0.5 * (limits[:, 0] + limits[:, 1])
        half_ranges = 0.5 * (limits[:, 1] - limits[:, 0])
        neutral_unit_pose = -centers / half_ranges
        self.register_buffer("pose_centers", centers)
        self.register_buffer("pose_half_ranges", half_ranges)
        self.register_buffer(
            "neutral_pose_logits", torch.atanh(neutral_unit_pose), persistent=False
        )

    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        features = self.feature_proj(self.feature_norm(pose_features))
        unit_pose_params = self.output(self.output_norm(features)).squeeze(-1)
        return self.pose_centers[None] + self.pose_half_ranges[None] * F.tanh(
            unit_pose_params + self.neutral_pose_logits[None]
        )
