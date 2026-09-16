import json

import pytest

from steerfrontier.data import (
    discover_behaviors,
    load_arc_challenge_data,
    load_hellaswag_data,
    load_mmlu_data,
    load_test_data,
    load_validation_data,
    load_vector_data,
)


def _example():
    return {
        "question": "Choose one answer.\n\n(A) option A\n(B) option B",
        "answer_matching_behavior": "(A)",
        "answer_not_matching_behavior": "(B)",
    }


def test_behavior_split_loaders_share_one_schema(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps([_example()]))

    assert load_vector_data(path) == [_example()]
    assert load_validation_data(path) == [_example()]
    assert load_test_data(path) == [_example()]


def test_behavior_loader_accepts_non_binary_choices(tmp_path):
    item = {
        "question": (
            "Choose one answer.\n\n"
            "(A) option A\n"
            "(B) option B\n"
            "(C) option C\n"
            "(D) option D"
        ),
        "answer_matching_behavior": "(D)",
        "answer_not_matching_behavior": "(A)",
    }

    path = tmp_path / "multichoice.json"
    path.write_text(json.dumps([item]))

    assert load_vector_data(path) == [item]


def test_behavior_loader_rejects_invalid_answer_format(tmp_path):
    item = _example()
    item["answer_matching_behavior"] = "A"

    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([item]))

    with pytest.raises(ValueError, match="invalid matching answer"):
        load_vector_data(path)


def test_behavior_loader_rejects_identical_answers(tmp_path):
    item = _example()
    item["answer_not_matching_behavior"] = "(A)"

    path = tmp_path / "identical.json"
    path.write_text(json.dumps([item]))

    with pytest.raises(ValueError, match="identical matching/non-matching answers"):
        load_vector_data(path)


def test_discover_behaviors_requires_all_three_splits(tmp_path):
    complete = tmp_path / "sycophancy"
    complete.mkdir()

    for filename in ("vector.json", "validation.json", "test.json"):
        (complete / filename).write_text("[]")

    incomplete = tmp_path / "refusal"
    incomplete.mkdir()
    (incomplete / "vector.json").write_text("[]")
    (incomplete / "validation.json").write_text("[]")

    capability = tmp_path / "mmlu"
    capability.mkdir()
    (capability / "test.json").write_text("[]")

    source = tmp_path / "source"
    source.mkdir()
    for filename in ("vector.json", "validation.json", "test.json"):
        (source / filename).write_text("[]")

    assert discover_behaviors(tmp_path) == ["sycophancy"]


def test_multiple_choice_capability_loader(tmp_path):
    path = tmp_path / "mmlu.json"

    path.write_text(
        json.dumps(
            [
                {
                    "id": "mmlu-1",
                    "question": "What is 2 + 2?",
                    "choices": [
                        {
                            "label": "(A)",
                            "text": "3",
                        },
                        {
                            "label": "(B)",
                            "text": "4",
                        },
                        {
                            "label": "(C)",
                            "text": "5",
                        },
                        {
                            "label": "(D)",
                            "text": "6",
                        },
                    ],
                    "correct": "(B)",
                    "category": "math",
                }
            ]
        )
    )

    data = load_mmlu_data(path)

    assert len(data) == 1
    assert data[0]["correct"] == "(B)"
    assert len(data[0]["choices"]) == 4


def test_multiple_choice_capability_rejects_missing_correct_choice(
    tmp_path,
):
    path = tmp_path / "bad.json"

    path.write_text(
        json.dumps(
            [
                {
                    "id": "bad-1",
                    "question": "Question?",
                    "choices": [
                        {
                            "label": "(A)",
                            "text": "one",
                        },
                        {
                            "label": "(B)",
                            "text": "two",
                        },
                    ],
                    "correct": "(C)",
                }
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="not present",
    ):
        load_mmlu_data(path)


def test_hellaswag_loader(tmp_path):
    path = tmp_path / "hellaswag.json"

    path.write_text(
        json.dumps(
            [
                {
                    "id": "hs-1",
                    "context": "A person opens the door.",
                    "endings": [
                        "They walk inside.",
                        "The moon turns green.",
                        "A fish writes a book.",
                        "The building disappears.",
                    ],
                    "correct_index": 0,
                }
            ]
        )
    )

    data = load_hellaswag_data(path)

    assert len(data) == 1
    assert data[0]["correct_index"] == 0
    assert len(data[0]["endings"]) == 4


def test_hellaswag_loader_rejects_bad_index(
    tmp_path,
):
    path = tmp_path / "bad_hs.json"

    path.write_text(
        json.dumps(
            [
                {
                    "id": "hs-bad",
                    "context": "Context",
                    "endings": [
                        "A",
                        "B",
                    ],
                    "correct_index": 2,
                }
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="invalid correct_index",
    ):
        load_hellaswag_data(path)
