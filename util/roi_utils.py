import torch
import torch.nn.functional as F


def _masked_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = torch.broadcast_to(mask, values.shape)
    sorted_values = values.masked_fill(~mask, torch.inf).sort(dim=-1).values
    median_indices = (mask.sum(dim=-1) - 1) // 2
    return sorted_values.gather(-1, median_indices[..., None]).squeeze(-1)


def _point_to_ray_distance(
    points: torch.Tensor, ray_origins: torch.Tensor, ray_directions: torch.Tensor
) -> torch.Tensor:
    offsets = points[..., None, :] - ray_origins[:, None]
    longitudinal = (offsets * ray_directions[:, None]).sum(dim=-1)
    perpendicular_squared = (
        offsets.square().sum(dim=-1) - longitudinal.square()
    ).clamp_min(0.0)
    behind_origin = longitudinal.clamp_max(0.0)
    return (perpendicular_squared + behind_origin.square()).sqrt()


def robust_ray_consensus(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    valid_rays: torch.Tensor,
    min_inlier_distance: float = 0.02,
    inlier_scale: float = 2.5,
    regularization: float = 1.0e-6,
) -> torch.Tensor:
    """Triangulate one point per batch from a consensus of directed rays.

    Every non-parallel ray pair proposes the midpoint of its closest points. The
    proposal with the smallest median point-to-ray distance defines the consensus.
    Its inliers are then refit by least-squares intersection of their supporting
    lines.
    """

    ray_directions = F.normalize(ray_directions, p=2, dim=-1)
    pair_indices = torch.triu_indices(
        valid_rays.shape[1], valid_rays.shape[1], offset=1, device=valid_rays.device
    )
    first_indices, second_indices = pair_indices
    first_origins = ray_origins[:, first_indices]
    second_origins = ray_origins[:, second_indices]
    first_directions = ray_directions[:, first_indices]
    second_directions = ray_directions[:, second_indices]

    origin_offsets = first_origins - second_origins
    direction_dot = (first_directions * second_directions).sum(dim=-1)
    first_offset_dot = (first_directions * origin_offsets).sum(dim=-1)
    second_offset_dot = (second_directions * origin_offsets).sum(dim=-1)
    denominator = 1.0 - direction_dot.square()
    safe_denominator = denominator.clamp_min(1.0e-6)
    first_depth = (
        direction_dot * second_offset_dot - first_offset_dot
    ) / safe_denominator
    second_depth = (
        second_offset_dot - direction_dot * first_offset_dot
    ) / safe_denominator
    first_points = first_origins + first_depth[..., None] * first_directions
    second_points = second_origins + second_depth[..., None] * second_directions
    hypotheses = 0.5 * (first_points + second_points)

    valid_pairs = (
        valid_rays[:, first_indices]
        & valid_rays[:, second_indices]
        & (denominator > 1.0e-6)
        & (first_depth > 0.0)
        & (second_depth > 0.0)
    )
    # Keep the original uninterrupted graph when compiling for benchmark parity.
    if not torch.compiler.is_compiling() and not valid_pairs.any(dim=-1).all():
        raise ValueError(
            "No usable camera-ray pair: check world-to-camera extrinsics, "
            "camera baselines, and masks of the same hand"
        )
    residuals = _point_to_ray_distance(hypotheses, ray_origins, ray_directions)
    scores = _masked_median(residuals, valid_rays[:, None])
    scores = scores.masked_fill(~valid_pairs, torch.inf)

    best_pair_indices = scores.argmin(dim=-1)
    batch_indices = torch.arange(len(ray_origins), device=ray_origins.device)
    consensus_seeds = hypotheses[batch_indices, best_pair_indices]
    seed_residuals = residuals[batch_indices, best_pair_indices]
    thresholds = torch.maximum(
        inlier_scale * scores[batch_indices, best_pair_indices],
        scores.new_tensor(min_inlier_distance),
    )
    inliers = valid_rays & (seed_residuals <= thresholds[:, None])
    seed_pair_rays = pair_indices[:, best_pair_indices].transpose(0, 1)
    inliers.scatter_(dim=1, index=seed_pair_rays, value=True)

    identity = torch.eye(3, device=ray_origins.device, dtype=ray_origins.dtype)
    line_projectors = identity - (
        ray_directions[..., :, None] * ray_directions[..., None, :]
    )
    line_projectors = line_projectors * inliers[..., None, None]
    system = line_projectors.sum(dim=1)
    right_hand_side = (line_projectors @ ray_origins[..., None]).sum(dim=1).squeeze(-1)

    system = system + regularization * identity
    right_hand_side = right_hand_side + regularization * consensus_seeds

    return torch.linalg.solve(system, right_hand_side)


@torch.no_grad()
def triangulate_mask_centroid_rays(
    masks: torch.Tensor,
    cam_extrinsics: torch.Tensor,
    cam_intrinsics: torch.Tensor,
    valid_views: torch.Tensor,
    min_inlier_distance: float = 0.02,
    inlier_scale: float = 2.5,
) -> torch.Tensor:
    height, width = masks.shape[-2:]
    masses = masks.sum(dim=(-2, -1)).clamp_min(1.0)
    pixels_x = torch.arange(width, device=masks.device, dtype=masks.dtype)
    pixels_y = torch.arange(height, device=masks.device, dtype=masks.dtype)
    centroids_x = (masks.sum(dim=-2) * pixels_x).sum(dim=-1) / masses
    centroids_y = (masks.sum(dim=-1) * pixels_y).sum(dim=-1) / masses
    homogeneous_centroids = torch.stack(
        [centroids_x, centroids_y, torch.ones_like(centroids_x)], dim=-1
    )

    rotations_cam_to_world = cam_extrinsics[..., :3, :3].transpose(-1, -2)
    cam_directions = torch.linalg.solve(
        cam_intrinsics, homogeneous_centroids[..., None]
    ).squeeze(-1)
    world_directions = (rotations_cam_to_world @ cam_directions[..., None]).squeeze(-1)
    world_directions = F.normalize(world_directions, p=2, dim=-1)
    cam_origins = -(rotations_cam_to_world @ cam_extrinsics[..., :3, 3:4]).squeeze(-1)

    return robust_ray_consensus(
        cam_origins,
        world_directions,
        valid_views,
        min_inlier_distance=min_inlier_distance,
        inlier_scale=inlier_scale,
    )
