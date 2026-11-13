import torch
from torch import nn

from .camera_geometry import PluckerRayEncoder, PluckerRayMapGenerator
from .canonical_vertex_shape import (
    CanonicalVertexShapeDecoder,
    MANOShapeProjector,
)
from .image_pyramid import ImageFeaturePyramid
from .parameter_heads import PoseParamHead
from .point_lifter import PointLifter
from .volume_decoder import (
    CanonicalTokenDecoder,
    RigidPalmPoseEstimator,
    SpatialVolumeEncoder,
    WorldKeypointPosterior,
)


class SilHand(nn.Module):
    def __init__(
        self,
        pose_param_limits: list[list[float]],
        num_joint_states: int,
        canonical_palm_template: list[list[float]],
        palm_joint_indices: list[int],
        hand_side: str,
        mano_assets_root_dir: str,
        palm_pose_centering: str = "wrist",
        image_size: int = 224,
        image_stage_dims: tuple[int, int, int, int] = (24, 48, 96, 192),
        image_stage_depths: tuple[int, int, int, int] = (1, 1, 2, 2),
        image_feature_dim: int = 64,
        image_mlp_ratio: float = 4.0,
        ray_feature_dim: int = 16,
        point_grid_size: int = 16,
        roi_size: float = 0.25,
        ray_consensus_min_distance: float = 0.02,
        ray_consensus_inlier_scale: float = 2.5,
        view_num_heads: int = 4,
        view_depth: int = 2,
        view_mlp_ratio: float = 4.0,
        volume_feature_dim: int = 64,
        world_volume_depth: int = 3,
        volume_kernel_size: int = 3,
        volume_mlp_ratio: float = 2.0,
        canonical_roi_size: float = 0.25,
        query_width: int = 128,
        query_depth: int = 3,
        query_num_heads: int = 8,
        query_mlp_ratio: float = 4.0,
        pose_param_feature_dim: int = 128,
        pose_param_decoder_width: int = 128,
        pose_param_decoder_layers: int = 4,
        vertex_shape_width: int = 128,
        vertex_shape_displacement_scale: float = 0.01,
        shape_projection_ridge: float = 0.0,
    ) -> None:
        super().__init__()

        canonical_palm_template_tensor = torch.tensor(
            canonical_palm_template, dtype=torch.float32
        )
        if canonical_palm_template_tensor.shape != (len(palm_joint_indices), 3):
            raise ValueError(
                "canonical_palm_template and palm_joint_indices must have matching joint counts"
            )
        if num_joint_states != len(pose_param_limits):
            raise ValueError(
                "The model requires one pose token per pose parameter and joint state"
            )
        self.image_size = image_size

        self.image_encoder = ImageFeaturePyramid(
            input_dim=1,
            stage_dims=tuple(image_stage_dims),
            stage_depths=tuple(image_stage_depths),
            output_dim=image_feature_dim,
            mlp_ratio=image_mlp_ratio,
        )
        self.camera_geometry = PluckerRayMapGenerator(
            image_size=image_size,
            ray_consensus_min_distance=ray_consensus_min_distance,
            ray_consensus_inlier_scale=ray_consensus_inlier_scale,
        )
        self.ray_encoder = PluckerRayEncoder(width=ray_feature_dim)
        self.point_lifter = PointLifter(
            feature_dim=image_feature_dim,
            geometry_feature_dim=ray_feature_dim,
            point_grid_size=point_grid_size,
            roi_size=roi_size,
            view_num_heads=view_num_heads,
            view_depth=view_depth,
            mlp_ratio=view_mlp_ratio,
        )
        self.world_volume_encoder = SpatialVolumeEncoder(
            image_feature_dim,
            volume_feature_dim,
            depth=world_volume_depth,
            kernel_size=volume_kernel_size,
            mlp_ratio=volume_mlp_ratio,
        )
        self.world_keypoint_head = WorldKeypointPosterior(
            volume_feature_dim, num_joint_states, point_grid_size
        )
        self.global_pose_estimator = RigidPalmPoseEstimator(
            palm_joint_indices,
            canonical_palm_template_tensor,
            centering=palm_pose_centering,
        )
        self.canonical_decoder = CanonicalTokenDecoder(
            num_pose_params=len(pose_param_limits),
            volume_width=volume_feature_dim,
            width=query_width,
            grid_size=point_grid_size,
            world_roi_size=roi_size,
            canonical_roi_size=canonical_roi_size,
            depth=query_depth,
            num_heads=query_num_heads,
            mlp_ratio=query_mlp_ratio,
        )
        self.pose_param_head = PoseParamHead(
            latent_dim=query_width,
            feature_dim=pose_param_feature_dim,
            pose_param_limits=pose_param_limits,
            hidden_dim=pose_param_decoder_width,
            num_layers=pose_param_decoder_layers,
        )
        self.canonical_vertex_shape_decoder = CanonicalVertexShapeDecoder(
            hand_side=hand_side,
            mano_assets_root_dir=mano_assets_root_dir,
            shape_latent_dim=query_width,
            width=vertex_shape_width,
            displacement_scale=vertex_shape_displacement_scale,
        )
        self.mano_shape_projector = MANOShapeProjector(
            hand_side, mano_assets_root_dir, ridge=shape_projection_ridge
        )

    def _make_image_features(
        self, inputs: dict[str, torch.Tensor], valid_views: torch.Tensor
    ) -> torch.Tensor:
        masks = inputs["masks"]
        batch_size, num_views, height, width = masks.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}, got {(height, width)}"
            )
        images = masks.unsqueeze(2)
        images = images * valid_views[..., None, None, None]
        flat_features = self.image_encoder(images.flatten(0, 1))
        return flat_features.reshape(
            batch_size,
            num_views,
            flat_features.shape[1],
            *flat_features.shape[-2:],
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        roi_centers, ray_maps, valid_views = self.camera_geometry(
            inputs["masks"],
            inputs["cam_extrinsics"],
            inputs["cam_intrinsics"],
            inputs["valid_views"],
        )
        image_features = self._make_image_features(inputs, valid_views)
        batch_size, num_views = ray_maps.shape[:2]
        flat_ray_features = self.ray_encoder(ray_maps.flatten(0, 1))
        ray_features = flat_ray_features.reshape(
            batch_size,
            num_views,
            flat_ray_features.shape[1],
            *flat_ray_features.shape[-2:],
        )
        feature_volume = self.point_lifter(
            image_features, ray_features, roi_centers, valid_views, inputs
        )
        world_volume = self.world_volume_encoder(feature_volume)
        world_keypoints = self.world_keypoint_head(
            world_volume,
            self.point_lifter.point_grid,
            roi_centers,
            self.point_lifter.roi_size,
        )
        with torch.autocast(device_type=world_volume.device.type, enabled=False):
            global_pose = self.global_pose_estimator(world_keypoints.float())

        pose_features, shape_feature = self.canonical_decoder(
            world_volume, roi_centers, global_pose
        )
        with torch.autocast(device_type=pose_features.device.type, enabled=False):
            pose_params = self.pose_param_head(pose_features.float())
            canonical_shape_vertices = self.canonical_vertex_shape_decoder(
                shape_feature.float()
            )
            shape_params = self.mano_shape_projector(
                canonical_shape_vertices
            )

        return {
            "pose_params": pose_params,
            "shape_params": shape_params,
            "global_pose": global_pose,
        }
