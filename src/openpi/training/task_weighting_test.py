from openpi.training import task_weighting


def test_compute_average_scores_has_all_tasks():
    averages = task_weighting.compute_average_scores()

    assert set(averages.keys()) == set(task_weighting.TASK_NAMES)
    assert all(0.0 <= score <= 1.0 for score in averages.values())


def test_generate_weights_mean_gives_higher_weight_to_harder_tasks():
    weights = task_weighting.generate_task_sampling_weights(method="mean", min_weight=0.75, max_weight=3.0, gamma=1.0)

    assert weights["sorting packages"] > weights["open door"]
    assert weights["stock and straighten shelf"] > weights["hold pot"]


def test_generate_weights_quantile_respects_bounds_and_ignore():
    weights = task_weighting.generate_task_sampling_weights(
        method="quantile",
        min_weight=0.5,
        max_weight=2.5,
        gamma=1.2,
        ignore_tasks=["clean the desktop"],
        neutral_weight=1.0,
    )

    assert all(0.5 <= w <= 2.5 or w == 1.0 for w in weights.values())
    assert weights["clean the desktop"] == 1.0
