import json

import pytest

from steerfrontier.data import (
    discover_behaviors,
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
