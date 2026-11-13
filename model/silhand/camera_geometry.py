import torch
import torch.nn.functional as F
from torch import nn
from util.roi_utils import triangulate_mask_centroid_rays


class PluckerRayMapGenerator(nn.Module):
    def __init__(
        self,
        image_size: int,
        ray_consensus_min_distance: float,
        ray_consensus_inlier_scale: float,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.ray_consensus_min_distance = ray_consensus_min_distance
        self.ray_consensus_inlier_scale = ray_consensus_inlier_scale

        coordinates = torch.arange(image_size, dtype=torch.float32)
        pixels_y, pixels_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        homogeneous_pixels = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_x)], dim=0
        ).flatten(1)
        self.register_buffer("homogeneous_pixels", homogeneous_pixels, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        masks: torch.Tensor,
        cam_extrinsics: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        valid_views: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=masks.device.type, enabled=False):
            if masks.shape[-2:] != (self.image_size, self.image_size):
                raise ValueError(
                    f"Expected image size {self.image_size}, got {masks.shape[-2:]}"
                )

            masks = masks.float()
            cam_extrinsics = cam_extrinsics.float()
            cam_intrinsics = cam_intrinsics.float()
            valid_views = valid_views.bool() & masks.flatten(2).any(dim=-1)
            roi_centers = triangulate_mask_centroid_rays(
                masks,
                cam_extrinsics,
                cam_intrinsics,
                valid_views,
                min_inlier_distance=self.ray_consensus_min_distance,
                inlier_scale=self.ray_consensus_inlier_scale,
            )

            batch_size, num_views = masks.shape[:2]
            homogeneous_pixels = self.homogeneous_pixels.expand(
                batch_size, num_views, -1, -1
            )
            directions_x = (
                homogeneous_pixels[..., 0, :] - cam_intrinsics[..., 0, 2, None]
            ) / cam_intrinsics[..., 0, 0, None]
            directions_y = (
                homogeneous_pixels[..., 1, :] - cam_intrinsics[..., 1, 2, None]
            ) / cam_intrinsics[..., 1, 1, None]
            directions_z = torch.ones_like(directions_x)
            camera_directions = torch.stack(
                [directions_x, directions_y, directions_z], dim=-2
            )
            rotations_cam_to_world = cam_extrinsics[..., :3, :3].transpose(-1, -2)
            world_directions = rotations_cam_to_world @ camera_directions
            world_directions = F.normalize(world_directions, p=2, dim=-2)

            camera_origins = -(
                rotations_cam_to_world @ cam_extrinsics[..., :3, 3:4]
            ).squeeze(-1)
            relative_origins = camera_origins - roi_centers[:, None]
            moments = torch.cross(
                relative_origins[..., None].expand_as(world_directions),
                world_directions,
                dim=-2,
            )
            ray_maps = torch.cat([world_directions, moments], dim=-2).reshape(
                batch_size,
                num_views,
                6,
                self.image_size,
                self.image_size,
            )
            ray_maps = ray_maps * valid_views[..., None, None, None]

            return roi_centers, ray_maps, valid_views


class PluckerRayEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(6, width, kernel_size=4, stride=4),
            nn.GELU(),
        )
        self.block = nn.Sequential(
            nn.Conv2d(
                width,
                width,
                kernel_size=3,
                padding=1,
                groups=width,
            ),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )

    def forward(self, ray_maps: torch.Tensor) -> torch.Tensor:
        features = self.stem(ray_maps)
        return features + self.block(features)
