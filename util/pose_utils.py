import torch


def global_pose_from_rotation_translation(
    rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    transform_rows = torch.cat([rotation, translation[..., None]], dim=-1)
    bottom_row = rotation.new_zeros(*rotation.shape[:-2], 1, 4)
    bottom_row[..., 0, 3] = 1.0
    return torch.cat([transform_rows, bottom_row], dim=-2)


def global_pose_to_rotation_translation(
    global_pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if global_pose.shape[-2:] != (4, 4):
        raise ValueError(
            f"global_pose must end with shape (4, 4), got {tuple(global_pose.shape)}"
        )
    return global_pose[..., :3, :3], global_pose[..., :3, 3]


def identity_global_pose(batch_size: int, reference: torch.Tensor) -> torch.Tensor:
    rotation = torch.eye(3, dtype=reference.dtype, device=reference.device).expand(
        batch_size, -1, -1
    )
    translation = reference.new_zeros(batch_size, 3)
    return global_pose_from_rotation_translation(rotation, translation)
