from collections import Counter

from openpi.training.sampler import FrameSampler, _infer_task_name_from_instruction, _normalize_task_name


def test_task_name_normalization_and_inference():
    assert _normalize_task_name("Sort packages") == "sorting packages"
    assert _normalize_task_name("sorting_packages_continuous") == "sorting packages"
    assert _infer_task_name_from_instruction("Turn the doorknob and push the door") == "open door"
    assert _infer_task_name_from_instruction("Pick and place package with barcode") == "sorting packages"


def test_subtask_sampler_frame_construction():
    sampler = FrameSampler.__new__(FrameSampler)
    sampler.sample_frames([(0, 2), (2, 4)], dataset_size=5, sampler_type="subtask", sampling_seed=0)

    assert len(sampler) == 5
    assert set(sampler.valid_indices) == {0, 1, 2, 3, 4}


def test_task_weighted_sampler_biases_harder_task():
    sampler = FrameSampler.__new__(FrameSampler)

    weighted_intervals = [
        (0, 49, 1.0, "easy"),
        (50, 99, 4.0, "hard"),
    ]
    sampler.sample_frames(weighted_intervals, dataset_size=100, sampler_type="task_weighted", sampling_seed=42)

    counts = Counter(0 if idx < 50 else 1 for idx in sampler.valid_indices)

    assert len(sampler.valid_indices) == 100
    assert counts[1] > counts[0]
