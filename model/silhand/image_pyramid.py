import torch
import torch.nn.functional as F
from torch import nn


def _upsample_bilinear_2x(inputs: torch.Tensor) -> torch.Tensor:
    # For align_corners=False, 2x interpolation uses clamped 1/4 and 3/4
    # weights. Separable interpolation avoids the deterministic index_put backward.
    values = (
        inputs.float() if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
    )
    left = torch.cat((values[..., :1], values[..., :-1]), dim=-1)
    right = torch.cat((values[..., 1:], values[..., -1:]), dim=-1)
    values = torch.stack(
        (0.25 * left + 0.75 * values, 0.75 * values + 0.25 * right), dim=-1
    ).flatten(-2)
    above = torch.cat((values[..., :1, :], values[..., :-1, :]), dim=-2)
    below = torch.cat((values[..., 1:, :], values[..., -1:, :]), dim=-2)
    values = torch.stack(
        (0.25 * above + 0.75 * values, 0.75 * values + 0.25 * below), dim=-2
    ).flatten(-3, -2)
    return values.to(inputs.dtype)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GlobalResponseNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()

        self.gamma = nn.Parameter(torch.zeros(width))
        self.beta = nn.Parameter(torch.zeros(width))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(inputs, ord=2, dim=(1, 2), keepdim=True)
        response = response / (response.mean(dim=-1, keepdim=True) + 1.0e-6)
        return inputs + self.gamma * (inputs * response) + self.beta


class ConvNeXtV2Block(nn.Module):
    def __init__(self, width: int, mlp_ratio: float) -> None:
        super().__init__()

        hidden_dim = round(width * mlp_ratio)
        self.depthwise_conv = nn.Conv2d(
            width, width, kernel_size=7, padding=3, groups=width
        )
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            GlobalResponseNorm(hidden_dim),
            nn.Linear(hidden_dim, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.depthwise_conv(inputs).permute(0, 2, 3, 1)
        features = self.mlp(self.norm(features)).permute(0, 3, 1, 2)
        return inputs + features


class ImageFeaturePyramid(nn.Module):
    def __init__(
        self,
        input_dim: int,
        stage_dims: tuple[int, int, int, int],
        stage_depths: tuple[int, int, int, int],
        output_dim: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()

        if len(stage_dims) != 4 or len(stage_depths) != 4:
            raise ValueError("ImageFeaturePyramid requires four stages")

        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, stage_dims[0], kernel_size=4, stride=4),
            LayerNorm2d(stage_dims[0]),
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm2d(stage_dims[index - 1]),
                    nn.Conv2d(
                        stage_dims[index - 1],
                        stage_dims[index],
                        kernel_size=2,
                        stride=2,
                    ),
                )
                for index in range(1, 4)
            ]
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[ConvNeXtV2Block(width, mlp_ratio=mlp_ratio) for _ in range(depth)]
                )
                for width, depth in zip(stage_dims, stage_depths)
            ]
        )
        self.lateral_projections = nn.ModuleList(
            [nn.Conv2d(width, output_dim, kernel_size=1) for width in stage_dims]
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        stage_features = []
        features = self.stages[0](self.stem(inputs))
        stage_features.append(features)
        for downsample, stage in zip(self.downsamples, self.stages[1:]):
            features = stage(downsample(features))
            stage_features.append(features)

        pyramid = [
            projection(features)
            for projection, features in zip(self.lateral_projections, stage_features)
        ]
        for index in range(len(pyramid) - 2, -1, -1):
            features = pyramid[index + 1]
            target_size = pyramid[index].shape[-2:]
            if target_size == (2 * features.shape[-2], 2 * features.shape[-1]):
                features = _upsample_bilinear_2x(features)
            else:
                features = F.interpolate(
                    features, size=target_size, mode="bilinear", align_corners=False
                )
            pyramid[index] = pyramid[index] + features

        return self.output_projection(pyramid[0])
