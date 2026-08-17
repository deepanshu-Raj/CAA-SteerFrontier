import json
from pathlib import Path
from typing import Any, Dict, List, Union

PathLike = Union[str, Path]
VALID_ANSWERS = {"(A)", "(B)"}


def load_json(path: PathLike) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, found {type(data).__name__}.")
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

        if matching not in VALID_ANSWERS or nonmatching not in VALID_ANSWERS:
            raise ValueError(
                f"Example {i} in {source} must use '(A)'/'(B)' answers; "
                f"found {matching!r}, {nonmatching!r}."
            )

        if matching == nonmatching:
            raise ValueError(
                f"Example {i} in {source} has identical matching/non-matching answers."
            )


def load_generation_data(path: PathLike) -> List[Dict[str, Any]]:
    """Load contrastive examples used to construct a behavior steering vector."""

    data = load_json(path)
    _validate_behavior_examples(data, path)
    return data


def load_ab_test_data(path: PathLike) -> List[Dict[str, Any]]:
    """Load held-out binary behavioral evaluation examples."""

    data = load_json(path)
    _validate_behavior_examples(data, path)
    return data


def load_mmlu_data(path: PathLike) -> List[Dict[str, Any]]:
    """Load the fixed binary MMLU capability subset."""

    data = load_json(path)
    required = {"prompt", "correct", "incorrect", "category"}

    for i, item in enumerate(data):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"MMLU example {i} is missing fields: {missing}")

        if item["correct"] not in VALID_ANSWERS:
            raise ValueError(f"Unexpected correct answer: {item['correct']}")
        if item["incorrect"] not in VALID_ANSWERS:
            raise ValueError(f"Unexpected incorrect answer: {item['incorrect']}")
        if item["correct"] == item["incorrect"]:
            raise ValueError(f"MMLU example {i} has identical correct/incorrect answers.")

    return data


def discover_behaviors(data_root: PathLike = "data") -> List[str]:
    """Discover behavior directories containing both generation and test splits."""

    root = Path(data_root)
    if not root.exists():
        return []

    behaviors = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name == "mmlu":
            continue
        if (path / "generate.json").exists() and (path / "test_ab.json").exists():
            behaviors.append(path.name)

    return behaviors
