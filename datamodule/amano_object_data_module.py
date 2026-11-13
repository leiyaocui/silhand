from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, default_collate
from util.inference_utils import crop_masks

from .amano_object_data_source import (
    PackedAMANOObjectData,
    PackedCameraSample,
)


def collate_amano_object_samples(
    batch: list[tuple[dict, dict]],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str] | torch.Tensor]]:
    input_samples, other_samples = zip(*batch)
    inputs = {
        name: default_collate([sample[name] for sample in input_samples])
        for name in input_samples[0]
        if name
        not in {
            "object_mesh_keys",
            "object_local_vertices",
            "object_faces",
            "tfs_object_to_world",
            "object_scales",
        }
    }
    object_meshes_by_key = {}
    object_mesh_index_by_key = {}
    object_mesh_indices = []
    for sample in input_samples:
        for key, vertices, faces in zip(
            sample["object_mesh_keys"],
            sample["object_local_vertices"],
            sample["object_faces"],
            strict=True,
        ):
            if key not in object_meshes_by_key:
                object_meshes_by_key[key] = (vertices, faces)
                object_mesh_index_by_key[key] = len(object_mesh_index_by_key)
            object_mesh_indices.append(object_mesh_index_by_key[key])
    if object_meshes_by_key:
        object_local_vertices, object_faces = zip(
            *object_meshes_by_key.values(), strict=True
        )
        inputs["object_local_vertices"] = pad_sequence(
            object_local_vertices, batch_first=True, padding_value=0.0
        )
        inputs["object_faces"] = pad_sequence(
            object_faces, batch_first=True, padding_value=0
        )
        inputs["tfs_object_to_world"] = torch.cat(
            [sample["tfs_object_to_world"] for sample in input_samples]
        )
        inputs["object_scales"] = torch.cat(
            [sample["object_scales"] for sample in input_samples]
        )
        inputs["object_mesh_indices"] = torch.tensor(
            object_mesh_indices, dtype=torch.int64
        )
        inputs["object_batch_indices"] = torch.repeat_interleave(
            torch.arange(len(input_samples), dtype=torch.int64),
            torch.tensor(
                [len(sample["object_local_vertices"]) for sample in input_samples],
                dtype=torch.int64,
            ),
        )
    else:
        inputs["object_local_vertices"] = torch.empty((0, 0, 3), dtype=torch.float32)
        inputs["object_faces"] = torch.empty((0, 0, 3), dtype=torch.int32)
        inputs["tfs_object_to_world"] = torch.empty((0, 4, 4), dtype=torch.float32)
        inputs["object_scales"] = torch.empty((0, 3), dtype=torch.float32)
        inputs["object_mesh_indices"] = torch.empty(0, dtype=torch.int64)
        inputs["object_batch_indices"] = torch.empty(0, dtype=torch.int64)
    return inputs, default_collate(other_samples)


class AMANOObjectDataset(Dataset):
    def __init__(
        self,
        real_dataset_dirs: list[str],
        image_size=(224, 224),
        num_views=8,
        mask_crop_bbox_area_fraction=0.25,
    ):
        if not real_dataset_dirs:
            raise ValueError(
                "Set datamodule.real_dataset_dirs to packed real dataset directories"
            )
        if not 0 < mask_crop_bbox_area_fraction <= 1:
            raise ValueError("mask_crop_bbox_area_fraction must lie in (0,1]")
        if len(image_size) != 2 or image_size[0] != image_size[1]:
            raise ValueError("SilHand expects square input images")
        self.real_data = PackedAMANOObjectData(real_dataset_dirs)
        if not len(self.real_data):
            raise ValueError("No real test samples found")
        self.faces, self.cap_face_indices = self._load_hand_faces(real_dataset_dirs)
        self.image_size = tuple(image_size)
        self.num_views = num_views
        self.mask_crop_bbox_area_fraction = mask_crop_bbox_area_fraction

    def __len__(self):
        return len(self.real_data)

    def __getitem__(self, index):
        return self._load_real_sample(index)

    def _load_hand_faces(
        self, dataset_dirs: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for dataset_dir in dataset_dirs:
            face_data_file = Path(dataset_dir) / "face_data.npz"
            if not face_data_file.is_file():
                continue
            with np.load(face_data_file) as face_data:
                return (
                    torch.from_numpy(face_data["faces"].astype(np.int32)),
                    torch.from_numpy(face_data["cap_face_indices"].astype(np.int64)),
                )
        raise FileNotFoundError("No face_data.npz was found in the configured datasets")

    def _pad_camera(self, camera: PackedCameraSample) -> PackedCameraSample:
        source_num_views = len(camera.observation_valid_views)
        if source_num_views > self.num_views:
            raise ValueError(
                f"{camera.reference} has {source_num_views} views, "
                f"but num_views={self.num_views}"
            )

        cam_extrinsics = torch.eye(4)[None].repeat(self.num_views, 1, 1)
        cam_intrinsics = torch.eye(3)[None].repeat(self.num_views, 1, 1)
        source_image_sizes = torch.ones(self.num_views, 2, dtype=torch.int32)
        camera_valid_views = torch.zeros(self.num_views, dtype=torch.bool)
        observation_valid_views = torch.zeros(self.num_views, dtype=torch.bool)
        cam_extrinsics[:source_num_views] = camera.cam_extrinsics
        cam_intrinsics[:source_num_views] = camera.cam_intrinsics
        source_image_sizes[:source_num_views] = camera.source_image_sizes
        camera_valid_views[:source_num_views] = camera.camera_valid_views
        observation_valid_views[:source_num_views] = camera.observation_valid_views
        cam_extrinsics, cam_intrinsics = self.set_invalid_cameras_to_identity(
            cam_extrinsics, cam_intrinsics, camera_valid_views
        )

        return PackedCameraSample(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            source_image_sizes=source_image_sizes,
            camera_valid_views=camera_valid_views,
            observation_valid_views=observation_valid_views,
            reference=camera.reference,
        )

    def _load_real_sample(self, index: int) -> tuple[dict, dict]:
        amano_sample = self.real_data.load_amano_sample(index)
        camera = self._pad_camera(self.real_data.load_camera(index))
        object_sample = self.real_data.load_object_sample(index)
        inputs = {
            "gt_vertices": amano_sample.vertices,
            "gt_joint_positions": amano_sample.joint_positions,
            "cam_extrinsics": camera.cam_extrinsics.to(torch.float32),
            "cam_intrinsics": camera.cam_intrinsics.to(torch.float32),
            "source_image_sizes": camera.source_image_sizes,
            "valid_views": camera.observation_valid_views.to(torch.bool),
            "object_mesh_keys": object_sample.object_mesh_keys,
            "object_local_vertices": object_sample.object_local_vertices,
            "object_faces": object_sample.object_faces,
            "tfs_object_to_world": object_sample.tfs_object_to_world,
            "object_scales": object_sample.object_scales,
        }
        return inputs, {"sample_file": amano_sample.reference}

    def iter_samples(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> Iterator[dict[str, torch.Tensor]]:
        object_names = {
            "object_mesh_indices",
            "object_scales",
            "tfs_object_to_world",
            "object_batch_indices",
            "object_local_vertices",
            "object_faces",
        }
        for sample_index in range(len(inputs["gt_vertices"])):
            sample_inputs = {
                name: value[sample_index : sample_index + 1]
                for name, value in inputs.items()
                if name not in object_names
            }
            object_rows = inputs["object_batch_indices"] == sample_index
            for name in ("object_mesh_indices", "object_scales", "tfs_object_to_world"):
                sample_inputs[name] = inputs[name][object_rows]
            sample_inputs["object_batch_indices"] = torch.zeros_like(
                inputs["object_batch_indices"][object_rows]
            )
            for name in ("object_local_vertices", "object_faces"):
                sample_inputs[name] = inputs[name]
            yield sample_inputs

    def set_invalid_cameras_to_identity(
        self,
        cam_extrinsics: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        valid_views: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        extrinsic_identity = torch.eye(
            4, device=cam_extrinsics.device, dtype=cam_extrinsics.dtype
        )
        intrinsic_identity = torch.eye(
            3, device=cam_intrinsics.device, dtype=cam_intrinsics.dtype
        )
        return (
            torch.where(
                valid_views[..., None, None], cam_extrinsics, extrinsic_identity
            ),
            torch.where(
                valid_views[..., None, None], cam_intrinsics, intrinsic_identity
            ),
        )

    @torch.no_grad()
    def render_depth(
        self,
        vertices,
        faces,
        cam_extrinsics,
        cam_intrinsics,
        image_size,
        source_image_sizes,
        render_views,
        excluded_face_indices=None,
    ):
        import drtk

        batch_size, num_vertices = vertices.shape[:2]
        num_views = cam_extrinsics.shape[1]
        height, width = image_size
        scene_vertices = (
            vertices[:, None]
            .expand(batch_size, num_views, num_vertices, 3)
            .flatten(0, 1)
            .contiguous()
        )
        if faces.ndim == 3:
            faces = (
                faces[:, None]
                .expand(batch_size, num_views, *faces.shape[1:])
                .flatten(0, 1)
                .contiguous()
            )
        depth = vertices.new_full((batch_size * num_views, height, width), torch.inf)
        excluded = torch.zeros(
            faces.shape[-2], dtype=torch.bool, device=vertices.device
        )
        if excluded_face_indices is not None:
            excluded[excluded_face_indices] = True
        for source_height, source_width in torch.unique(
            source_image_sizes.flatten(0, 1), dim=0
        ).tolist():
            selected = (
                (source_image_sizes[..., 0] == source_height)
                & (source_image_sizes[..., 1] == source_width)
                & render_views
            )
            view_indices = torch.where(selected.flatten())[0]
            chunk_size = max(1, 64_000_000 // (source_height * source_width))
            for start in range(0, len(view_indices), chunk_size):
                indices = view_indices[start : start + chunk_size]
                image_vertices = drtk.transform(
                    scene_vertices[indices].contiguous(),
                    K=cam_intrinsics.flatten(0, 1)[indices].contiguous(),
                    Rt=cam_extrinsics.flatten(0, 1)[indices].contiguous(),
                )
                values, face_image = drtk.rasterize_with_depth(
                    image_vertices,
                    faces[indices].contiguous()
                    if faces.ndim == 3
                    else faces.contiguous(),
                    source_height,
                    source_width,
                )
                visible = (face_image >= 0) & ~excluded[face_image.clamp_min(0).long()]
                depth[indices, :source_height, :source_width] = torch.where(
                    visible, values, torch.inf
                )
        return depth.reshape(batch_size, num_views, height, width)

    @torch.no_grad()
    def render_observations(self, inputs):
        vertices = inputs["gt_vertices"].float()
        if not vertices.is_cuda:
            raise RuntimeError(
                "Real test-set inference requires CUDA for DRTK rendering"
            )
        device = vertices.device
        self.faces = self.faces.to(device)
        self.cap_face_indices = self.cap_face_indices.to(device)
        cam_extrinsics = inputs["cam_extrinsics"].float()
        cam_intrinsics = inputs["cam_intrinsics"].float()
        sizes = inputs["source_image_sizes"]
        image_size = tuple(int(value) for value in sizes.amax((0, 1)).tolist())
        valid_views = inputs["valid_views"].bool()
        object_batch_indices = inputs["object_batch_indices"].long()
        hand_depth = self.render_depth(
            vertices,
            self.faces,
            cam_extrinsics,
            cam_intrinsics,
            image_size,
            sizes,
            valid_views,
            self.cap_face_indices,
        )
        masks = torch.isfinite(hand_depth)
        if len(object_batch_indices):
            sample_indices = object_batch_indices
            mesh_indices = inputs["object_mesh_indices"]
            local_vertices = inputs["object_local_vertices"][mesh_indices]
            transforms = inputs["tfs_object_to_world"]
            scaled_vertices = local_vertices * inputs["object_scales"][:, None]
            world_vertices = (
                scaled_vertices @ transforms[:, :3, :3].transpose(-1, -2)
                + transforms[:, None, :3, 3]
            )
            object_depth = self.render_depth(
                world_vertices,
                inputs["object_faces"][mesh_indices].int(),
                cam_extrinsics[sample_indices],
                cam_intrinsics[sample_indices],
                image_size,
                sizes[sample_indices],
                valid_views[sample_indices],
            )
            for row, sample in enumerate(sample_indices.tolist()):
                masks[sample] &= hand_depth[sample] < object_depth[row]
        height, width = image_size
        source_pixels = (
            (
                torch.arange(height, device=device)[None, None, :, None]
                < sizes[..., 0, None, None]
            )
            & (
                torch.arange(width, device=device)[None, None, None, :]
                < sizes[..., 1, None, None]
            )
            & valid_views[..., None, None]
        )
        masks &= source_pixels
        return {
            "gt_vertices": inputs["gt_vertices"],
            "gt_joint_positions": inputs["gt_joint_positions"],
            "masks": masks.float(),
            "cam_extrinsics": cam_extrinsics,
            "cam_intrinsics": cam_intrinsics,
            "source_image_sizes": sizes,
            "valid_views": valid_views,
        }

    @torch.no_grad()
    def prepare_model_inputs(self, inputs):
        outputs = dict(inputs)
        outputs.update(
            crop_masks(
                inputs["masks"],
                inputs["cam_intrinsics"],
                inputs["source_image_sizes"],
                inputs["valid_views"],
                self.image_size[0],
                self.mask_crop_bbox_area_fraction,
            )
        )
        valid = outputs["valid_views"] & outputs["masks"].flatten(2).any(-1)
        outputs["valid_views"] = valid
        outputs["cam_extrinsics"], outputs["cam_intrinsics"] = (
            self.set_invalid_cameras_to_identity(
                outputs["cam_extrinsics"], outputs["cam_intrinsics"], valid
            )
        )
        batch_size, num_views = valid.shape
        batch_indices = torch.arange(batch_size, device=valid.device)[:, None]
        # Stable partition: preserve the test view order within each validity group.
        partition_keys = (~valid).long() * num_views + torch.arange(
            num_views, device=valid.device
        )
        indices = partition_keys.argsort(dim=1)
        for name in (
            "masks",
            "cam_extrinsics",
            "cam_intrinsics",
            "valid_views",
            "source_image_sizes",
            "crop_boxes",
            "crop_affines",
        ):
            outputs[name] = outputs[name][batch_indices, indices]
        return outputs


class AMANOObjectDatamodule:
    """Original packed real-data interface, restricted to the test split."""

    def __init__(
        self,
        real_dataset_dirs,
        image_size=(224, 224),
        num_views=8,
        mask_crop_bbox_area_fraction=0.25,
    ):
        self.test_dataset = AMANOObjectDataset(
            real_dataset_dirs,
            image_size=image_size,
            num_views=num_views,
            mask_crop_bbox_area_fraction=mask_crop_bbox_area_fraction,
        )

    def test_dataloader(self, **kwargs):
        if kwargs.get("shuffle", False) or kwargs.get("drop_last", False):
            raise ValueError(
                "Test inference requires shuffle=False and drop_last=False"
            )
        if kwargs.get("num_workers", 0) == 0:
            kwargs.pop("prefetch_factor", None)
        return DataLoader(
            self.test_dataset, collate_fn=collate_amano_object_samples, **kwargs
        )

    @torch.no_grad()
    def postprocess(self, inputs, others):
        # Preserve the original per-sample rendering and cropping order.
        samples = []
        for sample in self.test_dataset.iter_samples(inputs):
            observations = self.test_dataset.render_observations(sample)
            samples.append(self.test_dataset.prepare_model_inputs(observations))
        return {
            key: torch.cat([sample[key] for sample in samples]) for key in samples[0]
        }, others
