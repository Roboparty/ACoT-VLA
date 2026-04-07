from collections.abc import Sequence
import math

TASK_NAMES: tuple[str, ...] = (
    "sorting packages",
    "pour workpiece",
    "take wrong item shelf",
    "stock and straighten shelf",
    "scoop popcorn",
    "open door",
    "place block into box",
    "hold pot",
    "clean the desktop",
)

# 27-team leaderboard collected by the user (April 2026).
# Each item stores per-task score in [0, 1].
LEADERBOARD_ROWS: tuple[dict[str, float], ...] = (
    {
        "sorting packages": 0.92,
        "sorting packages continuous": 0.13,
        "pour workpiece": 0.57,
        "take wrong item shelf": 0.98,
        "stock and straighten shelf": 0.97,
        "scoop popcorn": 1.0,
        "open door": 1.0,
        "place block into box": 0.53,
        "hold pot": 1.0,
        "clean the desktop": 0.38,
    },
    {
        "sorting packages": 0.87,
        "sorting packages continuous": 0.01,
        "pour workpiece": 0.95,
        "take wrong item shelf": 1.0,
        "stock and straighten shelf": 0.31,
        "scoop popcorn": 1.0,
        "open door": 1.0,
        "place block into box": 0.39,
        "hold pot": 1.0,
        "clean the desktop": 0.65,
    },
    {
        "sorting packages": 0.84,
        "sorting packages continuous": 0.05,
        "pour workpiece": 0.63,
        "take wrong item shelf": 0.98,
        "stock and straighten shelf": 0.86,
        "scoop popcorn": 1.0,
        "open door": 0.7,
        "place block into box": 0.49,
        "hold pot": 0.9,
        "clean the desktop": 0.48,
    },
    {
        "sorting packages": 0.81,
        "sorting packages continuous": 0.01,
        "pour workpiece": 0.89,
        "take wrong item shelf": 0.87,
        "stock and straighten shelf": 0.57,
        "scoop popcorn": 1.0,
        "open door": 0.72,
        "place block into box": 0.63,
        "hold pot": 0.94,
        "clean the desktop": 0.41,
    },
    {
        "sorting packages": 0.81,
        "sorting packages continuous": 0.04,
        "pour workpiece": 0.93,
        "take wrong item shelf": 0.73,
        "stock and straighten shelf": 0.56,
        "scoop popcorn": 1.0,
        "open door": 0.77,
        "place block into box": 0.68,
        "hold pot": 0.77,
        "clean the desktop": 0.45,
    },
    {
        "sorting packages": 0.64,
        "sorting packages continuous": 0.02,
        "pour workpiece": 0.91,
        "take wrong item shelf": 0.92,
        "stock and straighten shelf": 0.37,
        "scoop popcorn": 1.0,
        "open door": 1.0,
        "place block into box": 0.5,
        "hold pot": 0.91,
        "clean the desktop": 0.42,
    },
    {
        "sorting packages": 0.59,
        "sorting packages continuous": 0.01,
        "pour workpiece": 0.94,
        "take wrong item shelf": 0.9,
        "stock and straighten shelf": 0.37,
        "scoop popcorn": 1.0,
        "open door": 1.0,
        "place block into box": 0.48,
        "hold pot": 0.96,
        "clean the desktop": 0.44,
    },
    {
        "sorting packages": 0.65,
        "sorting packages continuous": 0.0,
        "pour workpiece": 0.86,
        "take wrong item shelf": 0.93,
        "stock and straighten shelf": 0.37,
        "scoop popcorn": 1.0,
        "open door": 1.0,
        "place block into box": 0.55,
        "hold pot": 0.97,
        "clean the desktop": 0.34,
    },
    {
        "sorting packages": 0.57,
        "sorting packages continuous": 0.01,
        "pour workpiece": 0.9,
        "take wrong item shelf": 0.87,
        "stock and straighten shelf": 0.64,
        "scoop popcorn": 0.9,
        "open door": 0.8,
        "place block into box": 0.55,
        "hold pot": 0.9,
        "clean the desktop": 0.43,
    },
    {
        "sorting packages": 0.65,
        "sorting packages continuous": 0.03,
        "pour workpiece": 0.87,
        "take wrong item shelf": 0.92,
        "stock and straighten shelf": 0.49,
        "scoop popcorn": 0.9,
        "open door": 1.0,
        "place block into box": 0.45,
        "hold pot": 0.9,
        "clean the desktop": 0.34,
    },
    {
        "sorting packages": 0.6,
        "sorting packages continuous": 0.0,
        "pour workpiece": 0.93,
        "take wrong item shelf": 0.91,
        "stock and straighten shelf": 0.45,
        "scoop popcorn": 0.9,
        "open door": 0.72,
        "place block into box": 0.56,
        "hold pot": 1.0,
        "clean the desktop": 0.48,
    },
    {
        "sorting packages": 0.62,
        "sorting packages continuous": 0.01,
        "pour workpiece": 0.88,
        "take wrong item shelf": 0.93,
        "stock and straighten shelf": 0.37,
        "scoop popcorn": 1.0,
        "open door": 1.0,
        "place block into box": 0.44,
        "hold pot": 0.95,
        "clean the desktop": 0.33,
    },
    {
        "sorting packages": 0.37,
        "sorting packages continuous": 0.02,
        "pour workpiece": 0.93,
        "take wrong item shelf": 0.97,
        "stock and straighten shelf": 0.49,
        "scoop popcorn": 1.0,
        "open door": 0.85,
        "place block into box": 0.61,
        "hold pot": 0.8,
        "clean the desktop": 0.44,
    },
    {
        "sorting packages": 0.48,
        "sorting packages continuous": 0.01,
        "pour workpiece": 0.86,
        "take wrong item shelf": 0.97,
        "stock and straighten shelf": 0.39,
        "scoop popcorn": 1.0,
        "open door": 1.0,
        "place block into box": 0.48,
        "hold pot": 0.89,
        "clean the desktop": 0.37,
    },
    {
        "sorting packages": 0.57,
        "sorting packages continuous": 0.02,
        "pour workpiece": 0.81,
        "take wrong item shelf": 0.93,
        "stock and straighten shelf": 0.37,
        "scoop popcorn": 1.0,
        "open door": 0.9,
        "place block into box": 0.5,
        "hold pot": 0.94,
        "clean the desktop": 0.37,
    },
    {
        "sorting packages": 0.67,
        "sorting packages continuous": 0.01,
        "pour workpiece": 0.9,
        "take wrong item shelf": 0.98,
        "stock and straighten shelf": 0.38,
        "scoop popcorn": 1.0,
        "open door": 0.67,
        "place block into box": 0.54,
        "hold pot": 0.96,
        "clean the desktop": 0.3,
    },
    {
        "sorting packages": 0.63,
        "sorting packages continuous": 0.01,
        "pour workpiece": 0.83,
        "take wrong item shelf": 0.9,
        "stock and straighten shelf": 0.43,
        "scoop popcorn": 0.9,
        "open door": 0.87,
        "place block into box": 0.43,
        "hold pot": 0.95,
        "clean the desktop": 0.35,
    },
    {
        "sorting packages": 0.59,
        "sorting packages continuous": 0.06,
        "pour workpiece": 0.67,
        "take wrong item shelf": 0.97,
        "stock and straighten shelf": 0.25,
        "scoop popcorn": 1.0,
        "open door": 0.97,
        "place block into box": 0.43,
        "hold pot": 0.9,
        "clean the desktop": 0.44,
    },
    {
        "sorting packages": 0.59,
        "sorting packages continuous": 0.03,
        "pour workpiece": 0.87,
        "take wrong item shelf": 0.91,
        "stock and straighten shelf": 0.34,
        "scoop popcorn": 1.0,
        "open door": 0.75,
        "place block into box": 0.41,
        "hold pot": 0.9,
        "clean the desktop": 0.32,
    },
    {
        "sorting packages": 0.62,
        "sorting packages continuous": 0.03,
        "pour workpiece": 0.9,
        "take wrong item shelf": 0.9,
        "stock and straighten shelf": 0.41,
        "scoop popcorn": 1.0,
        "open door": 0.55,
        "place block into box": 0.5,
        "hold pot": 0.85,
        "clean the desktop": 0.34,
    },
    {
        "sorting packages": 0.66,
        "sorting packages continuous": 0.04,
        "pour workpiece": 0.68,
        "take wrong item shelf": 0.92,
        "stock and straighten shelf": 0.21,
        "scoop popcorn": 1.0,
        "open door": 0.8,
        "place block into box": 0.51,
        "hold pot": 0.95,
        "clean the desktop": 0.3,
    },
    {
        "sorting packages": 0.65,
        "sorting packages continuous": 0.04,
        "pour workpiece": 0.78,
        "take wrong item shelf": 0.92,
        "stock and straighten shelf": 0.38,
        "scoop popcorn": 1.0,
        "open door": 0.47,
        "place block into box": 0.45,
        "hold pot": 0.91,
        "clean the desktop": 0.38,
    },
    {
        "sorting packages": 0.66,
        "sorting packages continuous": 0.06,
        "pour workpiece": 0.68,
        "take wrong item shelf": 0.85,
        "stock and straighten shelf": 0.23,
        "scoop popcorn": 0.9,
        "open door": 0.65,
        "place block into box": 0.48,
        "hold pot": 0.95,
        "clean the desktop": 0.46,
    },
    {
        "sorting packages": 0.62,
        "sorting packages continuous": 0.03,
        "pour workpiece": 0.63,
        "take wrong item shelf": 0.83,
        "stock and straighten shelf": 0.21,
        "scoop popcorn": 1.0,
        "open door": 0.75,
        "place block into box": 0.49,
        "hold pot": 0.95,
        "clean the desktop": 0.34,
    },
    {
        "sorting packages": 0.64,
        "sorting packages continuous": 0.06,
        "pour workpiece": 0.71,
        "take wrong item shelf": 0.91,
        "stock and straighten shelf": 0.22,
        "scoop popcorn": 1.0,
        "open door": 0.45,
        "place block into box": 0.46,
        "hold pot": 0.95,
        "clean the desktop": 0.4,
    },
    {
        "sorting packages": 0.72,
        "sorting packages continuous": 0.06,
        "pour workpiece": 0.7,
        "take wrong item shelf": 0.95,
        "stock and straighten shelf": 0.22,
        "scoop popcorn": 1.0,
        "open door": 0.45,
        "place block into box": 0.4,
        "hold pot": 0.97,
        "clean the desktop": 0.32,
    },
    {
        "sorting packages": 0.5,
        "sorting packages continuous": 0.04,
        "pour workpiece": 0.65,
        "take wrong item shelf": 0.88,
        "stock and straighten shelf": 0.22,
        "scoop popcorn": 1.0,
        "open door": 0.3,
        "place block into box": 0.44,
        "hold pot": 0.92,
        "clean the desktop": 0.34,
    },
)


def compute_average_scores(rows: Sequence[dict[str, float]] = LEADERBOARD_ROWS) -> dict[str, float]:
    if not rows:
        raise ValueError("Leaderboard rows must not be empty")

    averages: dict[str, float] = {}
    for task in TASK_NAMES:
        if task == "sorting packages":
            vals = [
                (float(row.get("sorting packages", 0.0)) + float(row.get("sorting packages continuous", 0.0))) / 2.0
                for row in rows
            ]
        else:
            vals = [float(row[task]) for row in rows]
        averages[task] = sum(vals) / len(vals)
    return averages


def _min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    low = min(values.values())
    high = max(values.values())
    if math.isclose(high, low):
        return {k: 0.0 for k in values}
    return {k: (v - low) / (high - low) for k, v in values.items()}


def _quantile_normalize(values: dict[str, float]) -> dict[str, float]:
    sorted_items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(sorted_items)
    if n == 1:
        return {sorted_items[0][0]: 0.0}

    normalized: dict[str, float] = {}
    for rank, (task, _) in enumerate(sorted_items):
        normalized[task] = rank / (n - 1)
    return normalized


def generate_task_sampling_weights(
    *,
    method: str = "mean",
    min_weight: float = 0.75,
    max_weight: float = 3.0,
    gamma: float = 1.2,
    neutral_weight: float = 1.0,
    ignore_tasks: Sequence[str] = (),
    rows: Sequence[dict[str, float]] = LEADERBOARD_ROWS,
) -> dict[str, float]:
    if min_weight <= 0:
        raise ValueError("min_weight must be positive")
    if max_weight < min_weight:
        raise ValueError("max_weight must be >= min_weight")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    avg_scores = compute_average_scores(rows)
    difficulty = {task: max(0.0, min(1.0, 1.0 - score)) for task, score in avg_scores.items()}

    if method == "mean":
        normalized_difficulty = _min_max_normalize(difficulty)
    elif method == "quantile":
        normalized_difficulty = _quantile_normalize(difficulty)
    else:
        raise ValueError(f"Unknown method: {method}. Expected 'mean' or 'quantile'.")

    ignore_set = {task.strip().lower() for task in ignore_tasks}

    weights: dict[str, float] = {}
    for task in TASK_NAMES:
        if task in ignore_set:
            weights[task] = float(neutral_weight)
            continue
        scaled = normalized_difficulty[task] ** gamma
        weights[task] = float(min_weight + (max_weight - min_weight) * scaled)

    return weights
