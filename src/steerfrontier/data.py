import json
from pathlib import Path
from typing import Any, Dict, List, Union

PathLike = Union[str, Path]

VALID_BEHAVIOR_ANSWERS = {
    f"({chr(code)})"
    for code in range(
        ord("A"),
        ord("Z") + 1,
    )
}


def load_json(
    path: PathLike,
) -> List[Dict[str, Any]]:
    path = Path(path)

    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {path}, found {type(data).__name__}."
        )

    return data


# ---------------------------------------------------------------------------
# Behavior datasets
# ---------------------------------------------------------------------------


def _validate_behavior_examples(
    data,
    source: PathLike,
):
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
                f"Example {i} in {source} has "
                "invalid non-matching answer: "
                f"{nonmatching!r}."
            )

        if matching == nonmatching:
            raise ValueError(
                f"Example {i} in {source} has identical matching/non-matching answers."
            )


def load_behavior_data(
    path: PathLike,
) -> List[Dict[str, Any]]:
    """
    Load and validate one contrastive behavior dataset split.
    """

    data = load_json(path)

    _validate_behavior_examples(
        data,
        path,
    )

    return data


def load_vector_data(
    path: PathLike,
) -> List[Dict[str, Any]]:
    """
    Load examples reserved for steering-vector extraction.
    """

    return load_behavior_data(path)


def load_validation_data(
    path: PathLike,
) -> List[Dict[str, Any]]:
    """
    Load examples reserved for layer and coefficient selection.
    """

    return load_behavior_data(path)


def load_test_data(
    path: PathLike,
) -> List[Dict[str, Any]]:
    """
    Load examples reserved for final behavioral reporting.
    """

    return load_behavior_data(path)


# ---------------------------------------------------------------------------
# Capability datasets
# ---------------------------------------------------------------------------


def _validate_choice_label(
    label: str,
    source: PathLike,
    example_index: int,
) -> None:
    if label not in VALID_BEHAVIOR_ANSWERS:
        raise ValueError(
            f"Capability example {example_index} "
            f"in {source} has invalid choice label "
            f"{label!r}."
        )


def load_multiple_choice_capability_data(
    path: PathLike,
) -> List[Dict[str, Any]]:
    """
    Load a canonical discrete multiple-choice capability dataset.

    Canonical schema:

        {
            "id": "...",
            "question": "...",
            "choices": [
                {
                    "label": "(A)",
                    "text": "..."
                },
                ...
            ],
            "correct": "(B)",
            "category": "..."
        }

    `category` is optional.

    Intended for:
        - MMLU
        - ARC-Challenge
    """

    data = load_json(path)

    required = {
        "id",
        "question",
        "choices",
        "correct",
    }

    for i, item in enumerate(data):
        missing = required - item.keys()

        if missing:
            raise ValueError(
                f"Capability example {i} in {path} is missing fields: {missing}"
            )

        if not isinstance(
            item["id"],
            str,
        ):
            raise ValueError(f"Capability example {i} in {path} must have a string id.")

        if (
            not isinstance(
                item["question"],
                str,
            )
            or not item["question"].strip()
        ):
            raise ValueError(
                f"Capability example {i} in {path} must have a non-empty question."
            )

        choices = item["choices"]

        if (
            not isinstance(
                choices,
                list,
            )
            or len(choices) < 2
        ):
            raise ValueError(
                f"Capability example {i} in {path} must contain at least two choices."
            )

        seen_labels = set()

        for j, choice in enumerate(choices):
            if not isinstance(
                choice,
                dict,
            ):
                raise ValueError(
                    f"Choice {j} of capability example {i} in {path} must be a mapping."
                )

            missing_choice = {
                "label",
                "text",
            } - choice.keys()

            if missing_choice:
                raise ValueError(
                    f"Choice {j} of capability example {i} "
                    f"in {path} is missing fields: "
                    f"{missing_choice}"
                )

            label = choice["label"]

            _validate_choice_label(
                label=label,
                source=path,
                example_index=i,
            )

            if label in seen_labels:
                raise ValueError(
                    f"Capability example {i} in {path} "
                    f"contains duplicate label {label!r}."
                )

            seen_labels.add(label)

            text = choice["text"]

            if (
                not isinstance(
                    text,
                    str,
                )
                or not text.strip()
            ):
                raise ValueError(
                    f"Choice {j} of capability example {i} in {path} has empty text."
                )

        correct = item["correct"]

        _validate_choice_label(
            label=correct,
            source=path,
            example_index=i,
        )

        if correct not in seen_labels:
            raise ValueError(
                f"Capability example {i} in {path} "
                f"has correct label {correct!r}, "
                "but that label is not present "
                "in its choices."
            )

    return data


def load_continuation_capability_data(
    path: PathLike,
) -> List[Dict[str, Any]]:
    """
    Load a canonical continuation-ranking capability dataset.

    Canonical schema:

        {
            "id": "...",
            "context": "...",
            "endings": [
                "...",
                "...",
                "...",
                "..."
            ],
            "correct_index": 2,
            "category": "..."
        }

    `category` is optional.

    Intended for:
        - HellaSwag
    """

    data = load_json(path)

    required = {
        "id",
        "context",
        "endings",
        "correct_index",
    }

    for i, item in enumerate(data):
        missing = required - item.keys()

        if missing:
            raise ValueError(
                f"Continuation example {i} in {path} is missing fields: {missing}"
            )

        if not isinstance(
            item["id"],
            str,
        ):
            raise ValueError(
                f"Continuation example {i} in {path} must have a string id."
            )

        if (
            not isinstance(
                item["context"],
                str,
            )
            or not item["context"].strip()
        ):
            raise ValueError(
                f"Continuation example {i} in {path} must have a non-empty context."
            )

        endings = item["endings"]

        if (
            not isinstance(
                endings,
                list,
            )
            or len(endings) < 2
        ):
            raise ValueError(
                f"Continuation example {i} in {path} must contain at least two endings."
            )

        for j, ending in enumerate(endings):
            if (
                not isinstance(
                    ending,
                    str,
                )
                or not ending.strip()
            ):
                raise ValueError(
                    f"Ending {j} of continuation example {i} in {path} is empty."
                )

        correct_index = item["correct_index"]

        if not isinstance(
            correct_index,
            int,
        ):
            raise ValueError(
                f"Continuation example {i} in {path} "
                "must have an integer correct_index."
            )

        if correct_index < 0 or correct_index >= len(endings):
            raise ValueError(
                f"Continuation example {i} in {path} "
                f"has invalid correct_index "
                f"{correct_index} for "
                f"{len(endings)} endings."
            )

    return data


def load_mmlu_data(
    path: PathLike,
) -> List[Dict[str, Any]]:
    return load_multiple_choice_capability_data(path)


def load_arc_challenge_data(
    path: PathLike,
) -> List[Dict[str, Any]]:
    return load_multiple_choice_capability_data(path)


def load_hellaswag_data(
    path: PathLike,
) -> List[Dict[str, Any]]:
    return load_continuation_capability_data(path)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_behaviors(
    data_root: PathLike = "data",
) -> List[str]:
    """
    Discover behavior directories containing all Phase-2 splits.
    """

    root = Path(data_root)

    if not root.exists():
        return []

    required_files = (
        "vector.json",
        "validation.json",
        "test.json",
    )

    behaviors = []

    excluded = {
        "capability",
        "mmlu",
        "source",
    }

    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name in excluded:
            continue

        if all((path / filename).exists() for filename in required_files):
            behaviors.append(path.name)

    return behaviors
