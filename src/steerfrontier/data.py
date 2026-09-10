import json
from pathlib import Path
from typing import Any, Dict, List, Union

PathLike = Union[str, Path]

# Behavior datasets may use arbitrary multiple-choice labels. The CAA
# datasets currently use choices up to (H).
VALID_BEHAVIOR_ANSWERS = {f"({chr(code)})" for code in range(ord("A"), ord("Z") + 1)}

# The current MMLU subset is explicitly binary.
VALID_BINARY_ANSWERS = {"(A)", "(B)"}


def load_json(path: PathLike) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {path}, found {type(data).__name__}."
        )

    return data


def _validate_behavior_examples(data, source: PathLike):
    required = {
        "question",
        "answer_matching_behavior",
        "answer_not_matching_behavior",
    }

    for i, item in enumerate(data):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Example {i} in {source} is missing fields: {missing}")

        matching = item["answer_matching_behavior"]
        nonmatching = item["answer_not_matching_behavior"]

        if matching not in VALID_BEHAVIOR_ANSWERS:
            raise ValueError(
                f"Example {i} in {source} has invalid matching answer: {matching!r}."
            )

        if nonmatching not in VALID_BEHAVIOR_ANSWERS:
            raise ValueError(
                f"Example {i} in {source} has invalid non-matching answer: "
                f"{nonmatching!r}."
            )

        if matching == nonmatching:
            raise ValueError(
                f"Example {i} in {source} has identical matching/non-matching answers."
            )


def load_behavior_data(path: PathLike) -> List[Dict[str, Any]]:
    """Load and validate one contrastive behavior dataset split."""

    data = load_json(path)
    _validate_behavior_examples(data, path)
    return data


def load_vector_data(path: PathLike) -> List[Dict[str, Any]]:
    """Load examples reserved for steering-vector extraction."""
    return load_behavior_data(path)


def load_validation_data(path: PathLike) -> List[Dict[str, Any]]:
    """Load examples reserved for layer and coefficient selection."""
    return load_behavior_data(path)


def load_test_data(path: PathLike) -> List[Dict[str, Any]]:
    """Load examples reserved for final behavioral reporting."""
    return load_behavior_data(path)


def load_mmlu_data(path: PathLike) -> List[Dict[str, Any]]:
    """Load the fixed binary MMLU capability subset."""

    data = load_json(path)
    required = {"prompt", "correct", "incorrect", "category"}

    for i, item in enumerate(data):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"MMLU example {i} is missing fields: {missing}")

        if item["correct"] not in VALID_BINARY_ANSWERS:
            raise ValueError(f"Unexpected correct answer: {item['correct']}")

        if item["incorrect"] not in VALID_BINARY_ANSWERS:
            raise ValueError(f"Unexpected incorrect answer: {item['incorrect']}")

        if item["correct"] == item["incorrect"]:
            raise ValueError(
                f"MMLU example {i} has identical correct/incorrect answers."
            )

    return data


def discover_behaviors(data_root: PathLike = "data") -> List[str]:
    """Discover behavior directories containing all Phase-2 splits."""

    root = Path(data_root)

    if not root.exists():
        return []

    required_files = (
        "vector.json",
        "validation.json",
        "test.json",
    )

    behaviors = []

    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name in {"mmlu", "source"}:
            continue

        if all((path / filename).exists() for filename in required_files):
            behaviors.append(path.name)

    return behaviors
