#!/usr/bin/env python3
"""Dataset-only evaluation for inspecting model inputs and outputs.

This script does not require Isaac Sim. It loads samples directly from the
training dataset, runs policy inference on each sample, dumps input/output
snapshots, and reports simple action error metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

import openpi.transforms as _transforms
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def _scalar_to_str(value: Any) -> str:
    arr = _to_numpy(value)
    if arr.size == 0:
        return ""
    return str(arr.reshape(-1)[0])


def _scalar_to_int(value: Any, default: int = -1) -> int:
    try:
        arr = _to_numpy(value)
        if arr.size == 0:
            return default
        return int(arr.reshape(-1)[0])
    except Exception:
        return default


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip()) or "unknown"


def _shape_dtype_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {k: _shape_dtype_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value), "items": [_shape_dtype_summary(v) for v in value]}

    arr = _to_numpy(value)

    min_val = None
    max_val = None
    preview = None

    is_numeric = np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)
    if arr.size and is_numeric:
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
    elif arr.size:
        # For string/object-like arrays, expose a small preview and skip numeric reductions.
        flat = arr.reshape(-1)
        preview = [str(x) for x in flat[:3]]

    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": min_val,
        "max": max_val,
        "preview": preview,
    }


def _image_to_uint8_hwc(image: Any) -> np.ndarray:
    arr = _to_numpy(image)

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            vmax = float(np.max(arr)) if arr.size else 1.0
            scale = 255.0 if vmax <= 1.5 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _align_action_arrays(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if gt.ndim != 2 or pred.ndim != 2:
        raise ValueError(f"Expected 2D action arrays, got gt={gt.shape}, pred={pred.shape}")

    t = min(gt.shape[0], pred.shape[0])
    d = min(gt.shape[1], pred.shape[1])
    return gt[:t, :d], pred[:t, :d]


def _extract_reference_actions(sample: dict[str, Any], data_cfg: _config.DataConfig) -> np.ndarray:
    """Project raw sample actions into the same action space as policy outputs.

    This applies data input/output transforms (Go2 slicing, ACOT horizon split,
    optional delta<->absolute transforms, output dimensional projection).
    """
    transformed = _transforms.compose(data_cfg.data_transforms.inputs)(copy.deepcopy(sample))
    payload: dict[str, Any] = {
        "actions": transformed["actions"],
    }
    if "state" in transformed:
        payload["state"] = transformed["state"]

    projected = _transforms.compose(data_cfg.data_transforms.outputs)(payload)
    if "actions" not in projected:
        raise ValueError("Failed to derive transformed reference actions from sample")
    return _to_numpy(projected["actions"])


def _episode_identity(sample: dict[str, Any]) -> tuple[str, int]:
    task = _scalar_to_str(sample.get("task", "unknown"))
    episode_index = _scalar_to_int(sample.get("episode_index", -1), default=-1)
    return task, episode_index


def _collect_episode_indices(dataset: _data_loader.Dataset, any_index: int) -> tuple[list[int], dict[str, Any]]:
    n = len(dataset)
    if any_index < 0 or any_index >= n:
        raise ValueError(f"start_index out of range: {any_index}, dataset_len={n}")

    center_sample = dataset[any_index]
    target_task, target_episode = _episode_identity(center_sample)

    start = any_index
    while start > 0:
        prev_task, prev_episode = _episode_identity(dataset[start - 1])
        if prev_task != target_task or prev_episode != target_episode:
            break
        start -= 1

    end = any_index
    while end + 1 < n:
        next_task, next_episode = _episode_identity(dataset[end + 1])
        if next_task != target_task or next_episode != target_episode:
            break
        end += 1

    indices = list(range(start, end + 1))
    info = {
        "task": target_task,
        "episode_index": target_episode,
        "start_frame_index": start,
        "end_frame_index": end,
        "num_frames": len(indices),
        "query_index": any_index,
    }
    return indices, info


@dataclass
class RunningStats:
    count: int = 0
    mse_sum: float = 0.0
    mae_sum: float = 0.0

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        diff = pred - gt
        self.mse_sum += float(np.mean(np.square(diff)))
        self.mae_sum += float(np.mean(np.abs(diff)))
        self.count += 1

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            return {"num_samples": 0, "mse": math.nan, "mae": math.nan}
        return {
            "num_samples": self.count,
            "mse": self.mse_sum / self.count,
            "mae": self.mae_sum / self.count,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trainset samples and inspect model I/O without simulator")
    parser.add_argument(
        "--config",
        default="acot_icra_simulation_challenge_reasoning_to_action",
        help="Training config name",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints/acot_icra_simulation_challenge_reasoning_to_action/v03/20000",
        help="Checkpoint step directory",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Dataset start index")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to inspect")
    parser.add_argument(
        "--mode",
        choices=["samples", "episode"],
        default="samples",
        help="samples: fixed number of independent indices; episode: evaluate all frames in one episode",
    )
    parser.add_argument(
        "--max-episode-frames",
        type=int,
        default=0,
        help="If >0, truncate evaluated episode frames to this number (0 means full episode)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/trainset_eval_io",
        help="Directory to save per-sample dumps",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save input camera frames as PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _config.get_config(args.config)
    ckpt_dir = pathlib.Path(args.checkpoint_dir)

    policy = _policy_config.create_trained_policy(cfg, ckpt_dir)

    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    base_dataset = _data_loader.create_torch_dataset(data_cfg, cfg.model)
    repacked_dataset = _data_loader.TransformedDataset(base_dataset, [*data_cfg.repack_transforms.inputs])

    dataset_len = len(repacked_dataset)
    run_output_dir = output_dir

    if args.mode == "samples":
        end_index = min(args.start_index + args.num_samples, dataset_len)
        if args.start_index >= end_index:
            raise ValueError(
                f"Invalid range: start_index={args.start_index}, num_samples={args.num_samples}, "
                f"dataset_len={dataset_len}"
            )
        indices = list(range(args.start_index, end_index))
        episode_info = None
    else:
        indices, episode_info = _collect_episode_indices(repacked_dataset, args.start_index)
        if args.max_episode_frames > 0:
            indices = indices[: args.max_episode_frames]

        task_safe = _safe_name(str(episode_info["task"]))
        run_output_dir = output_dir / (
            f"episode_{task_safe}_ep{episode_info['episode_index']}_"
            f"{episode_info['start_frame_index']}_{episode_info['end_frame_index']}"
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)

        print(
            "[episode mode] "
            f"task={episode_info['task']}, episode_index={episode_info['episode_index']}, "
            f"frames={episode_info['num_frames']}, range=[{episode_info['start_frame_index']}, {episode_info['end_frame_index']}], "
            f"evaluating={len(indices)}"
        )

    metrics = RunningStats()
    run_manifest: dict[str, Any] = {
        "mode": args.mode,
        "config": args.config,
        "checkpoint_dir": str(ckpt_dir),
        "dataset_len": dataset_len,
        "start_index": args.start_index,
        "num_samples": len(indices),
        "samples": [],
    }
    if episode_info is not None:
        run_manifest["episode_info"] = episode_info

    for i in indices:
        sample = repacked_dataset[i]
        pred = policy.infer(sample)

        sample_dir = run_output_dir / f"sample_{i:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        gt_actions = _extract_reference_actions(sample, data_cfg)
        pred_actions = _to_numpy(pred["actions"])
        gt_aligned, pred_aligned = _align_action_arrays(gt_actions, pred_actions)
        metrics.update(gt_aligned, pred_aligned)

        np.save(sample_dir / "gt_actions.npy", gt_aligned)
        np.save(sample_dir / "pred_actions.npy", pred_aligned)
        np.save(sample_dir / "action_diff.npy", pred_aligned - gt_aligned)

        if "tokenized_prompt" in sample:
            np.save(sample_dir / "tokenized_prompt.npy", _to_numpy(sample["tokenized_prompt"]))
        if "tokenized_prompt_mask" in sample:
            np.save(sample_dir / "tokenized_prompt_mask.npy", _to_numpy(sample["tokenized_prompt_mask"]))
        if "state" in sample:
            np.save(sample_dir / "state.npy", _to_numpy(sample["state"]))
        if "subtask_logits" in pred:
            np.save(sample_dir / "subtask_logits.npy", _to_numpy(pred["subtask_logits"]))

        if args.save_images and "images" in sample and isinstance(sample["images"], dict):
            for cam_name, cam_img in sample["images"].items():
                img = _image_to_uint8_hwc(cam_img)
                Image.fromarray(img).save(sample_dir / f"input_{cam_name}.png")

        sample_summary = {
            "index": i,
            "input_summary": _shape_dtype_summary(sample),
            "output_summary": _shape_dtype_summary(pred),
            "aligned_action_shape": list(gt_aligned.shape),
            "sample_mse": float(np.mean(np.square(pred_aligned - gt_aligned))),
            "sample_mae": float(np.mean(np.abs(pred_aligned - gt_aligned))),
        }
        with (sample_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(sample_summary, f, indent=2, ensure_ascii=False)

        run_manifest["samples"].append(
            {
                "index": i,
                "sample_dir": str(sample_dir),
                "sample_mse": sample_summary["sample_mse"],
                "sample_mae": sample_summary["sample_mae"],
            }
        )

        print(
            f"[sample {i}] action_shape={gt_aligned.shape}, "
            f"mse={sample_summary['sample_mse']:.6f}, mae={sample_summary['sample_mae']:.6f}"
        )

    run_manifest["metrics"] = metrics.summary()
    with (run_output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2, ensure_ascii=False)

    print("\n[done] Dataset-only eval finished.")
    print(json.dumps(run_manifest["metrics"], indent=2, ensure_ascii=False))
    print(f"Saved artifacts under: {run_output_dir}")


if __name__ == "__main__":
    main()
