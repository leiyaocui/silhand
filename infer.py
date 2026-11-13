import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from util.amano_hand_utils import AMANO


def load_model(cfg):
    device = torch.device(cfg.device)
    if device.type != "cuda":
        raise ValueError("Real-world test inference requires CUDA/DRTK")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; real-world test inference requires CUDA/DRTK")
    torch.cuda.set_device(device)
    if cfg.precision not in {"32-true", "bf16-mixed"}:
        raise ValueError("precision must be 32-true or bf16-mixed")
    if cfg.precision == "bf16-mixed":
        if not torch.cuda.is_bf16_supported():
            raise ValueError(
                "bf16-mixed needs a compatible CUDA GPU; use precision=32-true"
            )
    if cfg.hand_side != "right" or cfg.model.hand_side != "right":
        raise ValueError("This checkpoint supports right hands only")
    if cfg.model._target_ != "model.silhand.SilHand":
        raise ValueError("Only the main SilHand architecture is included")
    mano_assets = Path(cfg.mano_assets_root_dir).expanduser().resolve()
    if not (mano_assets / "models/MANO_RIGHT.pkl").is_file():
        raise FileNotFoundError(
            f"Missing {mano_assets / 'models/MANO_RIGHT.pkl'}. "
            "Download MANO v1.2 from https://mano.is.tue.mpg.de/ as described in README.md."
        )
    model = hydra.utils.instantiate(cfg.model, mano_assets_root_dir=str(mano_assets))
    weights = torch.load(cfg.ckpt_file, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()
    if cfg.compile_model:
        model = torch.compile(model)
    hand = AMANO(cfg.hand_side, mano_assets).to(device).eval()
    return model, hand


@torch.inference_mode()
def predict_model_inputs(model, hand, inputs, *, precision="32-true"):
    """Forward already-cropped test inputs without applying a second crop."""
    device = next(model.parameters()).device
    # Fabric's mixed-precision wrapper casts floating inputs before autocast.
    # This includes camera calibration; autocast alone does not reproduce it.
    # Keep the original inputs in FP32 for evaluation and crop metadata.
    model_inputs = (
        {
            key: value.to(torch.bfloat16) if value.is_floating_point() else value
            for key, value in inputs.items()
        }
        if precision == "bf16-mixed"
        else inputs
    )
    with torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=precision == "bf16-mixed"
    ):
        outputs = model(model_inputs)
    vertices, frames = hand(
        outputs["pose_params"].float(),
        outputs["shape_params"].float(),
        outputs["global_pose"].float(),
    )
    result = {
        **outputs,
        "vertices_world": vertices,
        "joints_world": frames[..., :3, 3],
        "joint_frames": frames,
        "faces": hand.faces,
        "valid_views": inputs["valid_views"],
        "crop_affines": inputs["crop_affines"],
        "crop_boxes": inputs["crop_boxes"],
    }
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise RuntimeError(
            "Inference produced nonfinite values; check calibration and checkpoint"
        )
    return {
        key: (value.float() if value.is_floating_point() else value).cpu()
        for key, value in result.items()
    }


def main_test_infer(cfg, model, hand):
    from util.evaluation_utils import benchmark_metrics

    device = next(model.parameters()).device
    if cfg.test.max_batches is not None and cfg.test.max_batches < 1:
        raise ValueError("test.max_batches must be positive or null")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    if datamodule.test_dataset.image_size != (model.image_size, model.image_size):
        raise ValueError("Datamodule and model image sizes must match")
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = datamodule.test_dataloader(generator=generator, **cfg.test.dataloader)
    output_dir = Path(cfg.log_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Choose an empty log_dir to avoid mixing test runs: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, sums = defaultdict(int), defaultdict(lambda: defaultdict(float))
    writer = None
    processed_batches = 0
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        for batch_index, (inputs, others) in enumerate(loader):
            if cfg.test.max_batches is not None and batch_index >= cfg.test.max_batches:
                break
            inputs = {
                name: value.to(device, non_blocking=True)
                for name, value in inputs.items()
            }
            inputs, others = datamodule.postprocess(inputs, others)
            predictions = predict_model_inputs(
                model, hand, inputs, precision=cfg.precision
            )
            metrics = benchmark_metrics(inputs, predictions)
            sample_files = others["sample_file"]
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=["sample_file", *metrics])
                writer.writeheader()
            for index, reference in enumerate(sample_files):
                row = {name: float(values[index]) for name, values in metrics.items()}
                writer.writerow({"sample_file": reference, **row})
                dataset = str(Path(reference.rsplit(":", 1)[0]))
                for group in ("all", dataset):
                    counts[group] += 1
                    for name, value in row.items():
                        sums[group][name] += value
            if cfg.test.save_predictions:
                np.savez_compressed(
                    output_dir / f"batch_{batch_index:06d}.npz",
                    sample_file=np.asarray(sample_files),
                    **{name: value.numpy() for name, value in predictions.items()},
                )
            processed_batches += 1
            if processed_batches == 1 or processed_batches % 10 == 0:
                print(
                    f"Infer: {counts['all']}/{len(loader.dataset)} samples", flush=True
                )
    summary = {
        "checkpoint": str(Path(cfg.ckpt_file).resolve()),
        "precision": cfg.precision,
        "model_input_dtype": "bfloat16" if cfg.precision == "bf16-mixed" else "float32",
        "compile_model": bool(cfg.compile_model),
        "batch_size": cfg.test.dataloader.batch_size,
        "protocol": "real test split, clean masks with object occlusion; no mask stress augmentations",
        "num_test_samples": len(loader.dataset),
        "num_processed_samples": counts["all"],
        "num_processed_batches": processed_batches,
        "complete_test_set": counts["all"] == len(loader.dataset),
        "metrics": {name: total / counts["all"] for name, total in sums["all"].items()},
        "datasets": {
            group: {
                "num_samples": counts[group],
                "metrics": {
                    name: total / counts[group] for name, total in values.items()
                },
            }
            for group, values in sums.items()
            if group != "all"
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Saved test results to {output_dir}\n{json.dumps(summary['metrics'], indent=2)}"
    )


@hydra.main(version_base=None, config_path="config", config_name="infer")
def main(cfg):
    if not cfg.ckpt_file:
        raise ValueError("Set ckpt_file=<checkpoint.pth>")
    if cfg.num_threads < 1:
        raise ValueError("num_threads must be positive")
    torch.set_num_threads(cfg.num_threads)
    # Match the original inference numeric settings.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg.seed)
    model, hand = load_model(cfg)
    main_test_infer(cfg, model, hand)


if __name__ == "__main__":
    main()
