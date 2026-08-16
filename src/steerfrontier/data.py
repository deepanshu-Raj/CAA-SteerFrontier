import json
from pathlib import Path
from typing import Any, Dict, List, Union

PathLike = Union[str, Path]


def load_json(path: PathLike) -> List[Dict[str, Any]]:
    path = Path(path)

    with path.open("r") as f:
        return json.load(f)


def load_generation_data(path: PathLike) -> List[Dict[str, Any]]:
    """Load contrastive examples used to construct steering vectors."""
    data = load_json(path)

    required = {
        "question",
        "answer_matching_behavior",
        "answer_not_matching_behavior",
    }

    for i, item in enumerate(data):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Generation example {i} is missing fields: {missing}")

    return data


def load_ab_test_data(path: PathLike) -> List[Dict[str, Any]]:
    """Load held-out A/B behavioral evaluation examples."""
    return load_generation_data(path)


def load_mmlu_data(path: PathLike) -> List[Dict[str, Any]]:
    """Load the fixed binary MMLU capability subset."""
    data = load_json(path)

    required = {
        "prompt",
        "correct",
        "incorrect",
        "category",
    }

    for i, item in enumerate(data):
        missing = required - item.keys()

        if missing:
            raise ValueError(f"MMLU example {i} is missing fields: {missing}")

        if item["correct"] not in {"(A)", "(B)"}:
            raise ValueError(f"Unexpected correct answer: {item['correct']}")

    return data
