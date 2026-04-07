
import random

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import torch
from openpi.shared import task_stage

def get_base_dataset(ds):
    if hasattr(ds, "_dataset"):
        return get_base_dataset(ds._dataset)
    return ds


def _normalize_task_name(name: str) -> str:
    return task_stage.normalize_task_name(name)


def _infer_task_name_from_instruction(instruction: str) -> str:
    return task_stage.infer_task_name(instruction)


def _extract_task_name(subtask: dict) -> str:
    for key in ("task", "task_name", "taskName", "name"):
        if key in subtask and isinstance(subtask[key], str):
            return _normalize_task_name(subtask[key])

    instruction = str(subtask.get("instruction", ""))
    if instruction:
        return _infer_task_name_from_instruction(instruction)

    return "unknown"

def sample_subtask(dataset):
    valid_intervals = []
    base_ds = get_base_dataset(dataset)
    
    sub_datasets = []
    
    if isinstance(base_ds, lerobot_dataset.MultiLeRobotDataset):
        for sub_ds in base_ds._datasets:
            sub_datasets.append(sub_ds)
    else:
        sub_datasets.append(dataset)

    current_global_offset = 0
    total_episodes_processed = 0

    print(f"Processing {len(sub_datasets)} sub-datasets...")

    for sub_ds in sub_datasets:
        inner_ds = get_base_dataset(sub_ds)
        
        instruction_segment = inner_ds.meta.info.get('instruction_segments', {})
        episode_data_index = inner_ds.episode_data_index
        num_episodes = len(episode_data_index['from'])
        
        for ep_idx in range(num_episodes):
            local_episode_start = episode_data_index['from'][ep_idx].item()
            
            if str(ep_idx) not in instruction_segment:
                continue

            tasks = instruction_segment[str(ep_idx)]
            for subtask in tasks:
                local_start = subtask["start_frame_index"] + local_episode_start
                local_end = subtask["success_frame_index"] + local_episode_start
                
                instruction = subtask["instruction"].lower()
                is_reset = any(k in instruction for k in ['reset', 'return', 'default'])
                
                if is_reset:
                    if local_end - local_start > 90:
                        local_end = local_start + 45
                
                global_start = local_start + current_global_offset
                global_end = local_end + current_global_offset
                
                valid_intervals.append((global_start, global_end))
        
        current_global_offset += len(sub_ds)
        total_episodes_processed += num_episodes

    print(f"Total {len(valid_intervals)} valid intervals from {total_episodes_processed} episodes.")
    return valid_intervals


def sample_task_weighted(dataset, task_weights: dict[str, float], task_ignore: set[str]):
    weighted_intervals = []
    base_ds = get_base_dataset(dataset)

    sub_datasets = []
    if isinstance(base_ds, lerobot_dataset.MultiLeRobotDataset):
        for sub_ds in base_ds._datasets:
            sub_datasets.append(sub_ds)
    else:
        sub_datasets.append(dataset)

    current_global_offset = 0
    total_episodes_processed = 0

    normalized_weights = {_normalize_task_name(k): float(v) for k, v in task_weights.items()}
    normalized_ignore = {_normalize_task_name(k) for k in task_ignore}

    print(f"Processing {len(sub_datasets)} sub-datasets with task-weighted sampler...")
    print(f"Configured task weights: {sorted(normalized_weights.items(), key=lambda x: x[0])}")
    if normalized_ignore:
        print(f"Tasks forced to neutral weight 1.0: {sorted(normalized_ignore)}")

    per_task_counts: dict[str, int] = {}

    for sub_ds in sub_datasets:
        inner_ds = get_base_dataset(sub_ds)

        instruction_segment = inner_ds.meta.info.get("instruction_segments", {})
        episode_data_index = inner_ds.episode_data_index
        num_episodes = len(episode_data_index["from"])

        for ep_idx in range(num_episodes):
            local_episode_start = episode_data_index["from"][ep_idx].item()

            if str(ep_idx) not in instruction_segment:
                continue

            tasks = instruction_segment[str(ep_idx)]
            for subtask in tasks:
                local_start = subtask["start_frame_index"] + local_episode_start
                local_end = subtask["success_frame_index"] + local_episode_start

                task_name = _extract_task_name(subtask)
                base_weight = normalized_weights.get(task_name, 1.0)
                if task_name in normalized_ignore:
                    base_weight = 1.0

                # Keep reset/return/default segments from dominating effective training frames.
                instruction = str(subtask.get("instruction", "")).lower()
                is_reset = any(k in instruction for k in ["reset", "return", "default"])
                if is_reset and local_end - local_start > 90:
                    local_end = local_start + 45

                global_start = local_start + current_global_offset
                global_end = local_end + current_global_offset
                weighted_intervals.append((global_start, global_end, float(base_weight), task_name))
                per_task_counts[task_name] = per_task_counts.get(task_name, 0) + 1

        current_global_offset += len(sub_ds)
        total_episodes_processed += num_episodes

    print(f"Total {len(weighted_intervals)} weighted intervals from {total_episodes_processed} episodes.")
    print(f"Task coverage in sampler: {sorted(per_task_counts.items(), key=lambda x: x[0])}")
    return weighted_intervals


class FrameSampler(torch.utils.data.Sampler):
    """
    Custom sampler that only samples data indices falling within specified intervals
    """
    def __init__(
        self,
        dataset,
        sampler_type,
        *,
        task_sampling_weights: dict[str, float] | None = None,
        task_sampling_ignore: tuple[str, ...] = (),
        sampling_seed: int = 0,
    ):
        parsed = self.parse_dataset(
            dataset,
            sampler_type,
            task_sampling_weights=task_sampling_weights,
            task_sampling_ignore=task_sampling_ignore,
        )
        self.sample_frames(parsed, len(dataset), sampler_type=sampler_type, sampling_seed=sampling_seed)

    def parse_dataset(self, dataset, sampler_type, *, task_sampling_weights, task_sampling_ignore):
        """
        Args:
            intervals: List of (start_index, end_index) tuples
        """
        if sampler_type == 'subtask':
            return sample_subtask(dataset)
        if sampler_type == "task_weighted":
            if not task_sampling_weights:
                raise ValueError("task_weighted sampler requires non-empty task_sampling_weights")
            return sample_task_weighted(dataset, task_sampling_weights, set(task_sampling_ignore))
        else:
            raise ValueError(f"Invalid sampler type: {sampler_type}")

    def sample_frames(self, intervals, dataset_size, *, sampler_type: str, sampling_seed: int):
        """
        Args:
            intervals: List of (start_index, end_index) tuples
            dataset_size: Total size of the dataset
        """
        self.intervals = intervals
        self.dataset_size = dataset_size

        if sampler_type == "task_weighted":
            candidates: list[int] = []
            weights: list[float] = []
            for start_idx, end_idx, interval_weight, _task_name in intervals:
                start_idx = max(0, start_idx)
                end_idx = min(dataset_size - 1, end_idx)
                if start_idx > end_idx:
                    continue
                for idx in range(start_idx, end_idx + 1):
                    candidates.append(idx)
                    weights.append(max(float(interval_weight), 1e-6))

            if not candidates:
                raise ValueError("task_weighted sampler produced no valid candidate frames")

            rng = random.Random(sampling_seed)
            self.valid_indices = rng.choices(candidates, weights=weights, k=len(candidates))
            print(f"Total {len(self.valid_indices)} weighted sampled indices, original: {dataset_size}")
            return

        # subtask sampler path
        self.valid_indices = []
        for start_idx, end_idx in intervals:
            # Ensure indices are within dataset bounds
            start_idx = max(0, start_idx)
            end_idx = min(dataset_size - 1, end_idx)

            # Add all indices within the interval
            self.valid_indices.extend(range(start_idx, end_idx + 1))

        # Remove duplicates and sort
        self.valid_indices = sorted(list(set(self.valid_indices)))
        print(f"Total {len(self.valid_indices)} valid indices,", "original:", dataset_size)

        rng = random.Random(sampling_seed)
        rng.shuffle(self.valid_indices)
    
    def __iter__(self):
        return iter(self.valid_indices)
    
    def __len__(self):
        return len(self.valid_indices)
