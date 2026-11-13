import inspect
from pathlib import Path

import numpy as np
import torch

# chumpy is imported while loading the legacy MANO pickle. These aliases were
# removed from Python 3.11 and NumPy 1.24, while MANO v1.2 still depends on them.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _name, _type in {
    "bool": np.bool_,
    "int": np.int_,
    "float": np.float64,
    "complex": np.complex128,
    "object": np.object_,
    "str": np.str_,
    "unicode": np.str_,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _type)

# isort: off
from manotorch.axislayer import AxisLayerFK  # noqa: E402
from manotorch.manolayer import ManoLayer  # noqa: E402
from manotorch.utils.geometry import euler_angles_to_matrix  # noqa: E402
# isort: on


AMANO_DOF_NAMES = (
    "thumb1_rx",
    "thumb1_ry",
    "thumb1_rz",
    "thumb2_rz",
    "thumb3_rz",
    "index1_ry",
    "index1_rz",
    "index2_rz",
    "index3_rz",
    "middle1_ry",
    "middle1_rz",
    "middle2_rz",
    "middle3_rz",
    "ring1_ry",
    "ring1_rz",
    "ring2_rz",
    "ring3_rz",
    "pinky1_ry",
    "pinky1_rz",
    "pinky2_rz",
    "pinky3_rz",
)
_MANO_FINGER_JOINT_OFFSETS = {
    "thumb": 12,
    "index": 0,
    "middle": 3,
    "ring": 9,
    "pinky": 6,
}
_MANO_FINGERTIP_VERTEX_INDICES = (745, 317, 444, 556, 673)


def _dof_indices(name: str) -> tuple[int, int]:
    joint, axis = name.rsplit("_r", 1)
    joint_index = _MANO_FINGER_JOINT_OFFSETS[joint[:-1]] + int(joint[-1]) - 1
    return joint_index, "xyz".index(axis)


AMANO_DOF_JOINT_INDICES, AMANO_DOF_AXIS_INDICES = zip(
    *map(_dof_indices, AMANO_DOF_NAMES)
)


class AMANO(torch.nn.Module):
    num_vertices = 778
    num_pose_parameters = len(AMANO_DOF_NAMES)
    num_shape_parameters = 10

    def __init__(self, side: str, mano_assets_root_dir: str | Path) -> None:
        super().__init__()

        if side not in {"right", "left"}:
            raise ValueError(f"side must be 'right' or 'left', got {side}")
        mano_assets_root = str(mano_assets_root_dir)
        self.mano_layer = ManoLayer(
            side=side,
            mano_assets_root=mano_assets_root,
            use_pca=False,
            flat_hand_mean=False,
        )
        self.axis_layer = AxisLayerFK(side=side, mano_assets_root=mano_assets_root)
        self.register_buffer("faces", self.mano_layer.th_faces.clone().to(torch.int32))
        self._fingertip_vertex_indices = _MANO_FINGERTIP_VERTEX_INDICES
        self.register_buffer(
            "dof_joint_indices", torch.tensor(AMANO_DOF_JOINT_INDICES).to(torch.int32)
        )
        self.register_buffer(
            "dof_axis_indices", torch.tensor(AMANO_DOF_AXIS_INDICES).to(torch.int32)
        )
        with torch.no_grad():
            neutral_mano_output = self.mano_layer(
                torch.zeros(1, 48), torch.zeros(1, 10)
            )
            neutral_angles = self.axis_layer(neutral_mano_output.transforms_abs)[2]
        self.register_buffer(
            "neutral_anatomy_angles",
            neutral_angles.to(torch.float32),
        )

    def forward(
        self,
        pose_params: torch.Tensor,
        shape_params: torch.Tensor,
        global_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pose_params: [B, 21]
            shape_params: [B, 10] or [10,]
            global_pose: [B, 4, 4] wrist-to-world transform
        Returns:
            vertices: [B, 778, 3]
            joint_frames: [B, 21, 4, 4] joint-to-world transforms
        """

        shape_params = shape_params.expand(len(pose_params), -1)
        mano_vertices, mano_joint_frames = self._skin(
            self.pose_params_to_rotation_matrices(pose_params), shape_params
        )
        if global_pose.shape[-2:] != (4, 4):
            raise ValueError(
                f"global_pose must have shape [B, 4, 4], got {tuple(global_pose.shape)}"
            )
        global_rotation = global_pose[:, :3, :3]
        global_translation = global_pose[:, :3, 3]
        wrist_position = mano_joint_frames[:, 0, :3, 3]
        canonical_joint_positions = (
            mano_joint_frames[..., :3, 3] - wrist_position[:, None]
        )
        transformed_rotation = global_rotation[:, None] @ mano_joint_frames[..., :3, :3]
        transformed_rotation = transformed_rotation @ self.axis_layer.TMPL_R_p_a
        transformed_translation = (
            global_rotation[:, None] @ canonical_joint_positions[..., None]
        ).squeeze(-1) + global_translation[:, None]
        transform_rows = torch.cat(
            (transformed_rotation, transformed_translation[..., None]), dim=-1
        )
        transforms_abs = torch.cat(
            (transform_rows, mano_joint_frames[..., 3:, :]), dim=-2
        )
        canonical_vertices = mano_vertices - wrist_position[:, None]
        vertices = (global_rotation[:, None] @ canonical_vertices[..., None]).squeeze(
            -1
        ) + global_translation[:, None]
        joint_frames = self.add_fingertip_frames(vertices, transforms_abs)
        return vertices, joint_frames

    def _skin(
        self, rotations: torch.Tensor, shape_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(rotations)
        shape_offsets = (
            self.mano_layer.th_shapedirs @ shape_params.transpose(0, 1)
        ).permute(2, 0, 1)
        rest_vertices = self.mano_layer.th_v_template + shape_offsets
        rest_joints = self.mano_layer.th_J_regressor @ rest_vertices
        identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
        pose_features = (rotations[:, 1:] - identity).flatten(1)
        pose_offsets = (
            self.mano_layer.th_posedirs @ pose_features.transpose(0, 1)
        ).permute(2, 0, 1)
        posed_vertices = rest_vertices + pose_offsets

        bottom_row = rotations.new_zeros(batch_size, 1, 1, 4)
        bottom_row[..., 0, 3] = 1.0
        root_transform = torch.cat(
            (
                torch.cat((rotations[:, :1], rest_joints[:, :1, :, None]), dim=-1),
                bottom_row,
            ),
            dim=-2,
        )
        parent_transform = root_transform
        parent_joints = rest_joints[:, :1]
        finger_transforms = []
        for level in range(3):
            joint_positions = rest_joints[:, level + 1 :: 3]
            relative_positions = joint_positions - parent_joints
            relative_transform = torch.cat(
                (
                    torch.cat(
                        (rotations[:, level + 1 :: 3], relative_positions[..., None]),
                        dim=-1,
                    ),
                    bottom_row.expand(-1, 5, -1, -1),
                ),
                dim=-2,
            )
            parent_transform = parent_transform @ relative_transform
            finger_transforms.append(parent_transform)
            parent_joints = joint_positions
        joint_frames = torch.cat(
            (root_transform, torch.stack(finger_transforms, dim=2).flatten(1, 2)), dim=1
        )

        # MANO linear blend skinning, with the rest joint offsets removed.
        rest_joint_vectors = torch.cat(
            (rest_joints, rest_joints.new_zeros(batch_size, 16, 1)), dim=-1
        )
        joint_offsets = joint_frames @ rest_joint_vectors[..., None]
        skinning_transforms = (
            joint_frames
            - torch.cat(
                (joint_offsets.new_zeros(batch_size, 16, 4, 3), joint_offsets), dim=-1
            )
        ).permute(0, 2, 3, 1)
        vertex_transforms = skinning_transforms @ self.mano_layer.th_weights.transpose(
            0, 1
        )
        homogeneous_vertices = torch.cat(
            (
                posed_vertices.transpose(1, 2),
                posed_vertices.new_ones(batch_size, 1, self.num_vertices),
            ),
            dim=1,
        )
        vertices = (
            (vertex_transforms[:, :3] * homogeneous_vertices[:, None])
            .sum(2)
            .transpose(1, 2)
            .contiguous()
        )
        return vertices, joint_frames

    def pose_params_to_rotation_matrices(
        self, pose_params: torch.Tensor
    ) -> torch.Tensor:
        anatomy_angles = self.neutral_anatomy_angles.expand(
            len(pose_params), -1, -1
        ).clone()
        anatomy_angles[:, self.dof_joint_indices + 1, self.dof_axis_indices] += (
            pose_params
        )
        # Parent transforms cancel when expressing each joint relative to its
        # parent. Conjugation maps anatomical rotations into the MANO frame.
        anatomical_axes = self.axis_layer.TMPL_R_p_a
        rotations = (
            anatomical_axes
            @ euler_angles_to_matrix(anatomy_angles, convention="XYZ")
            @ anatomical_axes.transpose(-1, -2)
        )
        root_rotation = torch.eye(
            3, device=pose_params.device, dtype=pose_params.dtype
        ).expand(len(pose_params), 1, 3, 3)
        return torch.cat((root_rotation, rotations[:, 1:]), dim=1)

    def add_fingertip_frames(
        self, vertices: torch.Tensor, joint_frames: torch.Tensor
    ) -> torch.Tensor:
        frames = [joint_frames[:, :1]]
        for frame_start, fingertip_vertex_index in zip(
            (13, 1, 4, 10, 7), self._fingertip_vertex_indices
        ):
            finger_frames = joint_frames[:, frame_start : frame_start + 3]
            fingertip_frame = finger_frames[:, -1:].clone()
            fingertip_frame[..., :3, 3] = vertices[:, fingertip_vertex_index, None]
            frames.extend((finger_frames, fingertip_frame))
        return torch.cat(frames, dim=1)
