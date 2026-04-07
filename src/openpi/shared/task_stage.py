from __future__ import annotations

import re

# Canonical challenge task names. Index is used as task_id.
TASKS: tuple[str, ...] = (
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

TASK_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(TASKS)}

_TASK_ALIASES: dict[str, str] = {
    "sort packages": "sorting packages",
    "sorting package": "sorting packages",
    "sorting packages continuous": "sorting packages",
    "sorting_packages_continuous": "sorting packages",
    "sorting packages part 2": "sorting packages",
    "sorting packages part2": "sorting packages",
    "sorting_packages_part_2": "sorting packages",
    "unload workpiece": "pour workpiece",
    "unload workpiece icra sim": "pour workpiece",
    "pour the workpiece into the box": "pour workpiece",
    "remove misplaced beverages from shelves": "take wrong item shelf",
    "take wrong item shelf": "take wrong item shelf",
    "stock supermarket shelves straighten products attend icra conference operate sim card": "stock and straighten shelf",
    "stock and straighten shelf": "stock and straighten shelf",
    "stock and straighten shelf part 2": "stock and straighten shelf",
    "stock_and_straighten_shelf_part_2": "stock and straighten shelf",
    "hold the tilted wei chuan grape juice upright with right arm": "stock and straighten shelf",
    "make popcorn": "scoop popcorn",
    "scoop popcorn": "scoop popcorn",
    "turn the doorknob": "open door",
    "turn the doorknob and push the door": "open door",
    "open door": "open door",
    "insert building block holes 2 sim": "place block into box",
    "insert building block holes_2_sim": "place block into box",
    "place block into box": "place block into box",
    "carry the pot": "hold pot",
    "hold pot": "hold pot",
    "clear the desktop": "clean the desktop",
    "clean the desktop": "clean the desktop",
}


def normalize_task_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return _TASK_ALIASES.get(normalized, normalized)


def infer_task_name(text: str) -> str:
    normalized = normalize_task_name(text)
    if normalized in TASK_NAME_TO_ID:
        return normalized

    if "barcode" in normalized or "package" in normalized:
        # Sorting variants share color distributions in training data, so avoid color heuristics.
        return "sorting packages"
    if "workpiece" in normalized:
        return "pour workpiece"
    if "misplaced" in normalized or ("shelf" in normalized and "basket" in normalized):
        return "take wrong item shelf"
    if "tilted" in normalized or "upright" in normalized:
        return "stock and straighten shelf"
    if "straighten" in normalized or ("stock" in normalized and "shelf" in normalized):
        return "stock and straighten shelf"
    if "popcorn" in normalized:
        return "scoop popcorn"
    if "doorknob" in normalized or "door" in normalized:
        return "open door"
    if "block" in normalized and "box" in normalized:
        return "place block into box"
    if "pot" in normalized:
        return "hold pot"
    if "desktop" in normalized or "laptop" in normalized or "pen holder" in normalized:
        return "clean the desktop"

    return normalized


def task_name_to_id(name: str | None, default_id: int = 0) -> int:
    if not name:
        return default_id
    canonical = infer_task_name(name)
    return TASK_NAME_TO_ID.get(canonical, default_id)
