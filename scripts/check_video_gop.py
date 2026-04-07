#!/usr/bin/env python3
"""
Check video GOP structure and B-frames in datasets.
Videos with large GOP or B-frames may cause timestamp alignment issues during training.
"""

import subprocess
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASETS = [
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/clean_the_desktop_part_1",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/clean_the_desktop_part_2",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/clean_the_desktop_addition",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/hold_pot",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/open_door",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/place_block_into_box",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/pour_workpiece",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/scoop_popcorn",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/scoop_popcorn_part_2",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/sorting_packages_part_1",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/sorting_packages_part_2",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/sorting_packages_part_3",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/stock_and_straighten_shelf",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/stock_and_straighten_shelf_part_2",
    "/mnt/h20_1data0/ygx_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/dataset_without_depth/take_wrong_item_shelf",
]

MAX_GOP_THRESHOLD = 30  # GOP > 30 frames (~1 second at 30fps) is considered problematic
MAX_VIDEOS_PER_DATASET = 10  # Check first N videos per dataset for speed


def check_video_gop(video_path: str) -> dict:
    """Check video GOP structure and B-frames."""
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pict_type',
        '-of', 'json', video_path
    ], capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        return {"error": result.stderr, "video": video_path}

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "Failed to parse ffprobe output", "video": video_path}

    frames = data.get('frames', [])
    if not frames:
        return {"error": "No frames found", "video": video_path}

    # Find keyframe positions
    keyframes = [i for i, f in enumerate(frames) if f.get('pict_type') == 'I']

    if not keyframes:
        return {"error": "No keyframes found", "video": video_path}

    # Calculate GOP sizes
    gop_sizes = [keyframes[i+1] - keyframes[i] for i in range(len(keyframes)-1)]
    max_gop = max(gop_sizes) if gop_sizes else len(frames)
    avg_gop = sum(gop_sizes) / len(gop_sizes) if gop_sizes else len(frames)

    # Check for B-frames
    b_frame_count = sum(1 for f in frames if f.get('pict_type') == 'B')
    has_b_frames = b_frame_count > 0

    return {
        "video": video_path,
        "total_frames": len(frames),
        "keyframe_count": len(keyframes),
        "max_gop": max_gop,
        "avg_gop": round(avg_gop, 1),
        "has_b_frames": has_b_frames,
        "b_frame_count": b_frame_count,
        "b_frame_ratio": round(b_frame_count / len(frames), 2) if frames else 0,
        "is_problematic": max_gop > MAX_GOP_THRESHOLD or has_b_frames,
    }


def check_dataset(dataset_path: str) -> list:
    """Check all videos in a dataset."""
    video_dir = Path(dataset_path) / "videos" / "chunk-000" / "observation.images.top_head"
    if not video_dir.exists():
        print(f"  [SKIP] No video dir: {dataset_path.split('/')[-1]}")
        return []

    results = []
    video_files = sorted(video_dir.glob("*.mp4"))[:MAX_VIDEOS_PER_DATASET]

    for video_file in video_files:
        info = check_video_gop(str(video_file))
        info["dataset"] = dataset_path.split('/')[-1]
        results.append(info)

    return results


def main():
    print("=" * 80)
    print("Video GOP Structure Checker")
    print(f"Checking {len(DATASETS)} datasets, {MAX_VIDEOS_PER_DATASET} videos per dataset")
    print(f"Problematic if: GOP > {MAX_GOP_THRESHOLD} frames OR has B-frames")
    print("=" * 80)
    print()

    all_results = []
    problematic_count = 0

    for dataset_path in DATASETS:
        dataset_name = dataset_path.split('/')[-1]
        print(f"Checking: {dataset_name}")

        results = check_dataset(dataset_path)

        for r in results:
            if "error" in r:
                print(f"  [ERROR] {Path(r['video']).name}: {r['error']}")
            elif r["is_problematic"]:
                problematic_count += 1
                print(f"  [PROBLEM] {Path(r['video']).name}: GOP={r['max_gop']}, B-frames={r['has_b_frames']} ({r['b_frame_ratio']:.0%})")
            all_results.append(r)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Count by dataset
    dataset_stats = {}
    for r in all_results:
        if "error" not in r:
            ds = r["dataset"]
            if ds not in dataset_stats:
                dataset_stats[ds] = {"total": 0, "problematic": 0, "max_gop": 0, "has_b_frames": False}
            dataset_stats[ds]["total"] += 1
            if r["is_problematic"]:
                dataset_stats[ds]["problematic"] += 1
            dataset_stats[ds]["max_gop"] = max(dataset_stats[ds]["max_gop"], r["max_gop"])
            dataset_stats[ds]["has_b_frames"] = dataset_stats[ds]["has_b_frames"] or r["has_b_frames"]

    print(f"\n{'Dataset':<45} {'Videos':<10} {'Problems':<10} {'Max GOP':<10} {'B-frames':<10}")
    print("-" * 85)
    for ds, stats in dataset_stats.items():
        print(f"{ds:<45} {stats['total']:<10} {stats['problematic']:<10} {stats['max_gop']:<10} {'Yes' if stats['has_b_frames'] else 'No':<10}")

    print()
    print(f"Total videos checked: {len(all_results)}")
    print(f"Problematic videos: {problematic_count}")

    if problematic_count > 0:
        print()
        print("RECOMMENDATION:")
        print("  Option 1: Increase tolerance_s in data_loader.py to 0.04")
        print("  Option 2: Re-encode videos with: ffmpeg -i input.mp4 -g 10 -bf 0 output.mp4")


if __name__ == "__main__":
    main()
