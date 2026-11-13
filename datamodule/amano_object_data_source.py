from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CAM_EXTRINSIC_SLICE = slice(0, 16)
CAM_INTRINSIC_SLICE = slice(16, 25)


@dataclass(frozen=True)
class PackedAMANOSample:
    vertices: torch.Tensor
    joint_positions: torch.Tensor
    reference: str


@dataclass(frozen=True)
class PackedCameraSample:
    cam_extrinsics: torch.Tensor
    cam_intrinsics: torch.Tensor
    source_image_sizes: torch.Tensor
    camera_valid_views: torch.Tensor
    observation_valid_views: torch.Tensor
    reference: str


@dataclass(frozen=True)
class PackedObjectSample:
    object_mesh_keys: list[str]
    object_local_vertices: list[torch.Tensor]
    object_faces: list[torch.Tensor]
    tfs_object_to_world: torch.Tensor
    object_scales: torch.Tensor


def _load_array(root_dir: Path, name: str) -> np.ndarray:
    path = (root_dir / f"{name}.npy").resolve()
    return np.load(path, mmap_mode="r")


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array).copy()).to(torch.float32)


class _PackedAMANOObjectSource:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.vertices = _load_array(root_dir, "vertices")
        self.joint_frames_12d = _load_array(root_dir, "joint_frames_12d")

        self.sample_indices = np.asarray(
            np.load(root_dir / "test_indices.npy"), dtype=np.int64
        )
        self.camera_params = _load_array(root_dir, "camera_params")
        self.source_image_sizes = _load_array(root_dir, "source_image_sizes")
        camera_valid_mask = _load_array(root_dir, "camera_valid_mask")
        test_view_masks = np.asarray(
            np.load(root_dir / "test_view_mask.npy"), dtype=np.bool_
        )
        self.camera_valid_view_masks = np.asarray(
            camera_valid_mask[self.sample_indices], dtype=np.bool_
        )
        self.observation_valid_view_masks = (
            self.camera_valid_view_masks & test_view_masks
        )

        self.object_mesh_cache = {}
        self.object_offsets = _load_array(root_dir, "object_offsets")
        self.object_mesh_indices = _load_array(root_dir, "object_mesh_indices")
        self.tfs_object_to_world = _load_array(root_dir, "tfs_object_to_world")
        self.object_scales = _load_array(root_dir, "object_scales")
        self.object_mesh_vertices = _load_array(root_dir, "object_mesh_vertices")
        self.object_mesh_faces = _load_array(root_dir, "object_mesh_faces")
        self.object_mesh_vertex_offsets = _load_array(
            root_dir, "object_mesh_vertex_offsets"
        )
        self.object_mesh_face_offsets = _load_array(
            root_dir, "object_mesh_face_offsets"
        )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _packed_index(self, index: int) -> int:
        return int(self.sample_indices[index])

    def load_amano_sample(self, index: int) -> PackedAMANOSample:
        packed_index = self._packed_index(index)
        vertices = _to_tensor(self.vertices[packed_index])
        # The packed 12D frames store translation before the nine rotation entries.
        joint_positions = _to_tensor(self.joint_frames_12d[packed_index, ..., :3])
        return PackedAMANOSample(
            vertices=vertices,
            joint_positions=joint_positions,
            reference=f"{self.root_dir}:{packed_index}",
        )

    def load_camera(self, index: int) -> PackedCameraSample:
        packed_index = self._packed_index(index)
        camera_params = _to_tensor(self.camera_params[packed_index])
        source_image_sizes = torch.from_numpy(
            self.source_image_sizes[packed_index].copy()
        )
        return PackedCameraSample(
            cam_extrinsics=camera_params[:, CAM_EXTRINSIC_SLICE].reshape(-1, 4, 4),
            cam_intrinsics=camera_params[:, CAM_INTRINSIC_SLICE].reshape(-1, 3, 3),
            source_image_sizes=source_image_sizes,
            camera_valid_views=torch.from_numpy(
                self.camera_valid_view_masks[index].copy()
            ),
            observation_valid_views=torch.from_numpy(
                self.observation_valid_view_masks[index].copy()
            ),
            reference=f"{self.root_dir}:{packed_index}",
        )

    def load_object_sample(self, index: int) -> PackedObjectSample:
        packed_index = self._packed_index(index)
        start = int(self.object_offsets[packed_index])
        end = int(self.object_offsets[packed_index + 1])
        mesh_indices = self.object_mesh_indices[start:end]
        object_mesh_keys = []
        object_local_vertices = []
        object_faces = []
        for mesh_index_value in mesh_indices:
            mesh_index = int(mesh_index_value)
            mesh = self.object_mesh_cache.get(mesh_index)
            if mesh is None:
                vertex_start = int(self.object_mesh_vertex_offsets[mesh_index])
                vertex_end = int(self.object_mesh_vertex_offsets[mesh_index + 1])
                face_start = int(self.object_mesh_face_offsets[mesh_index])
                face_end = int(self.object_mesh_face_offsets[mesh_index + 1])
                mesh = (
                    _to_tensor(self.object_mesh_vertices[vertex_start:vertex_end]),
                    torch.from_numpy(
                        np.asarray(
                            self.object_mesh_faces[face_start:face_end], dtype=np.int32
                        ).copy()
                    ),
                )
                self.object_mesh_cache[mesh_index] = mesh
            object_mesh_keys.append(f"{self.root_dir}:{mesh_index}")
            object_local_vertices.append(mesh[0])
            object_faces.append(mesh[1])
        return PackedObjectSample(
            object_mesh_keys=object_mesh_keys,
            object_local_vertices=object_local_vertices,
            object_faces=object_faces,
            tfs_object_to_world=_to_tensor(self.tfs_object_to_world[start:end]),
            object_scales=_to_tensor(self.object_scales[start:end]),
        )


class PackedAMANOObjectData:
    """Read geometry, calibration, and objects for the packed real test split."""

    def __init__(self, dataset_dirs: list[str]) -> None:
        self.sources = [
            _PackedAMANOObjectSource(Path(root_dir)) for root_dir in dataset_dirs
        ]
        self.offsets = np.cumsum([0, *(len(source) for source in self.sources)])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _resolve(self, index: int) -> tuple[_PackedAMANOObjectSource, int]:
        source_index = int(np.searchsorted(self.offsets[1:], index, side="right"))
        return self.sources[source_index], index - int(self.offsets[source_index])

    def load_amano_sample(self, index: int) -> PackedAMANOSample:
        source, local_index = self._resolve(index)
        return source.load_amano_sample(local_index)

    def load_camera(self, index: int) -> PackedCameraSample:
        source, local_index = self._resolve(index)
        return source.load_camera(local_index)

    def load_object_sample(self, index: int) -> PackedObjectSample:
        source, local_index = self._resolve(index)
        return source.load_object_sample(local_index)
