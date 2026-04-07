import argparse
import json

from openpi.training import task_weighting


def main():
    parser = argparse.ArgumentParser(description="Generate task sampling weights from the 27-team leaderboard.")
    parser.add_argument("--method", choices=["mean", "quantile"], default="mean")
    parser.add_argument("--min-weight", type=float, default=0.75)
    parser.add_argument("--max-weight", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument(
        "--ignore-tasks",
        type=str,
        default="",
        help="Comma separated task names to force neutral weight 1.0",
    )
    args = parser.parse_args()

    ignore_tasks = [item.strip().lower() for item in args.ignore_tasks.split(",") if item.strip()]

    avg_scores = task_weighting.compute_average_scores()
    weights = task_weighting.generate_task_sampling_weights(
        method=args.method,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        gamma=args.gamma,
        ignore_tasks=ignore_tasks,
    )

    print("Average task scores (across 27 teams):")
    print(json.dumps(avg_scores, indent=2, sort_keys=True))
    print()
    print("Generated task_sampling_weights:")
    print(json.dumps(weights, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
