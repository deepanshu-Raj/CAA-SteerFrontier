"""Prepare frozen out-of-distribution datasets for generalization evaluation.

Sycophancy is split into NLP and political domains, while hallucination uses a
deterministically sampled TruthfulQA multiple-choice set.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# =============================================================================
# Frozen protocol
# =============================================================================

SEED = 20260919

TRUTHFULQA_SAMPLE_SIZE = 150

OUTPUT_ROOT = Path("data/ood")

SYCOPHANCY_ROOT = OUTPUT_ROOT / "sycophancy"
HALLUCINATION_ROOT = OUTPUT_ROOT / "hallucination"

OUTPUTS = {
    "sycophancy_nlp_vector": (SYCOPHANCY_ROOT / "nlp_vector.json"),
    "sycophancy_nlp_test": (SYCOPHANCY_ROOT / "nlp_test.json"),
    "sycophancy_political_vector": (SYCOPHANCY_ROOT / "political_vector.json"),
    "sycophancy_political_test": (SYCOPHANCY_ROOT / "political_test.json"),
    "hallucination_truthfulqa_test": (HALLUCINATION_ROOT / "truthfulqa_test.json"),
}

MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"


# These are the counts we just audited from the frozen canonical data.
EXPECTED_SYCOPHANCY_COUNTS = {
    "vector": {
        "nlp": 447,
        "political": 453,
    },
    "test": {
        "nlp": 164,
        "political": 136,
    },
}


# =============================================================================
# Helpers
# =============================================================================


def stable_seed(
    *parts,
):
    payload = "|".join(str(part) for part in parts).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def write_json(
    path: Path,
    value,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            value,
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.write("\n")


def sha256_file(
    path: Path,
):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_answer(
    answer,
):
    answer = str(answer).strip()

    if answer not in {
        "(A)",
        "(B)",
    }:
        raise ValueError(f"Unexpected answer label: {answer!r}")

    return answer


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def answer_balance(
    examples,
):
    return dict(Counter(item["answer_matching_behavior"] for item in examples))


# =============================================================================
# Sycophancy domain split
# =============================================================================


def classify_sycophancy_domain(
    item,
):
    """
    Operational domain definition frozen before OOD evaluation.

    Political:
        Existing CAA example carries user_affiliation metadata.

    NLP:
        Existing CAA example does not carry user_affiliation metadata.

    This reproduces the two visibly distinct domains in our canonical
    sycophancy pool. The upstream PhilPapers mirror is not used because
    its records were found to be identical to the NLP source mirror.
    """

    affiliation = item.get("user_affiliation")

    if affiliation is not None and str(affiliation).strip():
        return "political"

    return "nlp"


def prepare_sycophancy_split(
    split: str,
):
    source_path = Path("data") / "sycophancy" / f"{split}.json"

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    data = load_json(source_path)

    grouped = {
        "nlp": [],
        "political": [],
    }

    for source_index, item in enumerate(data):
        matching = normalize_answer(item["answer_matching_behavior"])

        nonmatching = normalize_answer(item["answer_not_matching_behavior"])

        if matching == nonmatching:
            raise RuntimeError(
                f"{source_path}: identical behavior labels at index {source_index}."
            )

        domain = classify_sycophancy_domain(item)

        example = dict(item)

        example["answer_matching_behavior"] = matching

        example["answer_not_matching_behavior"] = nonmatching

        example["ood_domain"] = domain

        example["canonical_split"] = split

        example["canonical_source_index"] = int(source_index)

        grouped[domain].append(example)

    expected = EXPECTED_SYCOPHANCY_COUNTS[split]

    for domain in [
        "nlp",
        "political",
    ]:
        observed = len(grouped[domain])

        if observed != expected[domain]:
            raise RuntimeError(
                f"Sycophancy {split}/{domain}: "
                f"expected "
                f"{expected[domain]} examples, "
                f"found {observed}."
            )

    if len(grouped["nlp"]) + len(grouped["political"]) != len(data):
        raise RuntimeError(
            f"Sycophancy {split}: domain partition does not cover all examples exactly."
        )

    return (
        grouped,
        source_path,
    )


# =============================================================================
# TruthfulQA OOD set
# =============================================================================


def prepare_truthfulqa():
    print("Loading TruthfulQA multiple-choice validation split...")

    dataset = load_dataset(
        "truthful_qa",
        "multiple_choice",
        split="validation",
    )

    if len(dataset) < TRUTHFULQA_SAMPLE_SIZE:
        raise RuntimeError(
            f"TruthfulQA contains fewer than {TRUTHFULQA_SAMPLE_SIZE} examples."
        )

    # ----------------------------------------------------------------
    # Deterministically choose 150 unique questions.
    # No model outputs are involved.
    # ----------------------------------------------------------------

    sample_rng = random.Random(
        stable_seed(
            SEED,
            "truthfulqa",
            "sample",
        )
    )

    all_indices = list(range(len(dataset)))

    sample_rng.shuffle(all_indices)

    selected_indices = all_indices[:TRUTHFULQA_SAMPLE_SIZE]

    # Keep stable source-order presentation in the frozen file.
    selected_indices = sorted(selected_indices)

    # ----------------------------------------------------------------
    # Freeze exact 75 / 75 behavior-label position balance.
    #
    # "matching behavior" here means choosing the incorrect answer,
    # i.e. the hallucination-compatible response.
    # ----------------------------------------------------------------

    label_positions = ["(A)"] * (TRUTHFULQA_SAMPLE_SIZE // 2) + ["(B)"] * (
        TRUTHFULQA_SAMPLE_SIZE - TRUTHFULQA_SAMPLE_SIZE // 2
    )

    position_rng = random.Random(
        stable_seed(
            SEED,
            "truthfulqa",
            "answer_position",
        )
    )

    position_rng.shuffle(label_positions)

    examples = []

    for local_index, source_index in enumerate(selected_indices):
        item = dataset[source_index]

        question = str(item["question"]).strip()

        mc1 = item["mc1_targets"]

        choices = [str(choice).strip() for choice in mc1["choices"]]

        labels = [int(label) for label in mc1["labels"]]

        if len(choices) != len(labels):
            raise RuntimeError(
                f"TruthfulQA item {source_index}: choices/labels length mismatch."
            )

        correct_indices = [i for i, label in enumerate(labels) if label == 1]

        incorrect_indices = [i for i, label in enumerate(labels) if label == 0]

        if len(correct_indices) != 1:
            raise RuntimeError(
                f"TruthfulQA item "
                f"{source_index}: expected "
                "exactly one MC1 correct "
                f"answer, found "
                f"{len(correct_indices)}."
            )

        if not incorrect_indices:
            raise RuntimeError(
                f"TruthfulQA item {source_index}: no incorrect MC1 answers."
            )

        correct_answer = choices[correct_indices[0]]

        # One wrong answer is chosen deterministically per question.
        distractor_rng = random.Random(
            stable_seed(
                SEED,
                "truthfulqa",
                source_index,
                "incorrect_answer",
            )
        )

        chosen_incorrect_index = distractor_rng.choice(incorrect_indices)

        incorrect_answer = choices[chosen_incorrect_index]

        if correct_answer == incorrect_answer:
            raise RuntimeError(
                f"TruthfulQA item "
                f"{source_index}: chosen "
                "correct/incorrect answers "
                "are text-identical."
            )

        matching_label = label_positions[local_index]

        if matching_label == "(A)":
            # A = hallucination-compatible incorrect answer
            answer_a = incorrect_answer

            answer_b = correct_answer

            nonmatching_label = "(B)"

        elif matching_label == "(B)":
            # B = hallucination-compatible incorrect answer
            answer_a = correct_answer

            answer_b = incorrect_answer

            nonmatching_label = "(A)"

        else:
            raise RuntimeError("Unexpected matching label.")

        prompt = f"{question}\n\nChoices:\n (A) {answer_a}\n (B) {answer_b}\n\nAnswer:"

        example = {
            "question": prompt,
            "answer_matching_behavior": (matching_label),
            "answer_not_matching_behavior": (nonmatching_label),
            "ood_source": ("truthfulqa_mc1"),
            "source_split": ("validation"),
            "source_index": int(source_index),
            "truthfulqa_category": (
                str(
                    item.get(
                        "category",
                        "",
                    )
                )
            ),
            "truthful_answer": (correct_answer),
            "incorrect_answer": (incorrect_answer),
            "incorrect_choice_source_index": int(chosen_incorrect_index),
        }

        examples.append(example)

    if len(examples) != TRUTHFULQA_SAMPLE_SIZE:
        raise RuntimeError("Unexpected TruthfulQA sample size.")

    balance = answer_balance(examples)

    expected_half = TRUTHFULQA_SAMPLE_SIZE // 2

    if balance.get(
        "(A)",
        0,
    ) != expected_half or balance.get(
        "(B)",
        0,
    ) != (TRUTHFULQA_SAMPLE_SIZE - expected_half):
        raise RuntimeError("TruthfulQA matching-answer positions are not balanced.")

    return (
        examples,
        len(dataset),
    )


# =============================================================================
# Leakage / overlap checks
# =============================================================================


def canonical_questions(
    behavior: str,
):
    questions = set()

    for split in [
        "vector",
        "validation",
        "test",
    ]:
        path = Path("data") / behavior / f"{split}.json"

        if not path.is_file():
            raise FileNotFoundError(path)

        data = load_json(path)

        for item in data:
            questions.add(str(item["question"]).strip())

    return questions


def exact_question_overlap(
    examples,
    reference_questions,
):
    return sum(
        str(item["question"]).strip() in reference_questions for item in examples
    )


def assert_disjoint_source_indices(
    a,
    b,
):
    a_ids = {int(item["canonical_source_index"]) for item in a}

    b_ids = {int(item["canonical_source_index"]) for item in b}

    overlap = a_ids & b_ids

    if overlap:
        raise RuntimeError("Expected source-domain partitions to be disjoint.")


# =============================================================================
# Manifest helpers
# =============================================================================


def file_record(
    path: Path,
    examples,
):
    return {
        "path": str(path),
        "num_examples": int(len(examples)),
        "matching_answer_balance": (answer_balance(examples)),
        "sha256": (sha256_file(path)),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 96)
    print("PREPARING FROZEN OOD / GENERALIZATION DATA")
    print("=" * 96)

    # ----------------------------------------------------------------
    # Sycophancy
    # ----------------------------------------------------------------

    print()
    print("Preparing sycophancy domains...")

    (
        vector_domains,
        vector_source_path,
    ) = prepare_sycophancy_split("vector")

    (
        test_domains,
        test_source_path,
    ) = prepare_sycophancy_split("test")

    assert_disjoint_source_indices(
        vector_domains["nlp"],
        vector_domains["political"],
    )

    assert_disjoint_source_indices(
        test_domains["nlp"],
        test_domains["political"],
    )

    write_json(
        OUTPUTS["sycophancy_nlp_vector"],
        vector_domains["nlp"],
    )

    write_json(
        OUTPUTS["sycophancy_nlp_test"],
        test_domains["nlp"],
    )

    write_json(
        OUTPUTS["sycophancy_political_vector"],
        vector_domains["political"],
    )

    write_json(
        OUTPUTS["sycophancy_political_test"],
        test_domains["political"],
    )

    # ----------------------------------------------------------------
    # Hallucination / TruthfulQA
    # ----------------------------------------------------------------

    print("Preparing TruthfulQA hallucination OOD set...")

    (
        truthfulqa,
        truthfulqa_source_size,
    ) = prepare_truthfulqa()

    canonical_hall_questions = canonical_questions("hallucination")

    truthfulqa_overlap = exact_question_overlap(
        truthfulqa,
        canonical_hall_questions,
    )

    if truthfulqa_overlap != 0:
        raise RuntimeError(
            "Unexpected exact prompt "
            "overlap between TruthfulQA "
            "OOD and canonical "
            "hallucination data."
        )

    write_json(
        OUTPUTS["hallucination_truthfulqa_test"],
        truthfulqa,
    )

    # ----------------------------------------------------------------
    # Manifest
    # ----------------------------------------------------------------

    manifest = {
        "protocol": ("ood_generalization_v1"),
        "seed": int(SEED),
        "frozen_before_model_evaluation": True,
        "model_outcomes_used_for_data_selection": False,
        "ood_validation_split_used": False,
        "intervention_policy": {
            "epsilon": 0.03,
            "layer_selection": ("Reuse frozen layer selected by the main experiment."),
            "coefficient_selection": (
                "Reuse frozen epsilon=0.03 "
                "operating-point strength; "
                "no OOD coefficient tuning."
            ),
            "target_test_used_for_tuning": False,
        },
        "sycophancy": {
            "design": ("Cross-domain transfer."),
            "domains": {
                "nlp": (
                    "Canonical sycophancy examples without user_affiliation metadata."
                ),
                "political": (
                    "Canonical sycophancy examples with user_affiliation metadata."
                ),
            },
            "transfers": [
                {
                    "source": "nlp",
                    "target": "political",
                },
                {
                    "source": "political",
                    "target": "nlp",
                },
            ],
            "vector_source": str(vector_source_path),
            "test_source": str(test_source_path),
            "upstream_note": (
                "The Anthropic PhilPapers "
                "mirror was not used as a "
                "distinct OOD domain because "
                "its 9,984 labeled records "
                "were found to be identical "
                "to the mirrored NLP source."
            ),
            "files": {
                "nlp_vector": (
                    file_record(
                        OUTPUTS["sycophancy_nlp_vector"],
                        vector_domains["nlp"],
                    )
                ),
                "nlp_test": (
                    file_record(
                        OUTPUTS["sycophancy_nlp_test"],
                        test_domains["nlp"],
                    )
                ),
                "political_vector": (
                    file_record(
                        OUTPUTS["sycophancy_political_vector"],
                        vector_domains["political"],
                    )
                ),
                "political_test": (
                    file_record(
                        OUTPUTS["sycophancy_political_test"],
                        test_domains["political"],
                    )
                ),
            },
        },
        "hallucination": {
            "design": ("Independent-dataset generalization."),
            "source_vector": ("Existing frozen canonical CAA hallucination vector."),
            "target_dataset": ("TruthfulQA multiple-choice validation split."),
            "target_source_size": int(truthfulqa_source_size),
            "sample_size": int(len(truthfulqa)),
            "sampling": ("Uniform deterministic sample without replacement."),
            "binary_construction": (
                "Each TruthfulQA MC1 correct "
                "answer is paired with one "
                "deterministically selected "
                "incorrect answer. The "
                "incorrect answer is defined "
                "as matching the hallucination "
                "behavior. A/B answer position "
                "is exactly balanced."
            ),
            "exact_prompt_overlap_with_core": int(truthfulqa_overlap),
            "file": (
                file_record(
                    OUTPUTS["hallucination_truthfulqa_test"],
                    truthfulqa,
                )
            ),
        },
    }

    write_json(
        MANIFEST_PATH,
        manifest,
    )

    # ----------------------------------------------------------------
    # Console audit
    # ----------------------------------------------------------------

    print()
    print("=" * 96)
    print("SYCOPHANCY CROSS-DOMAIN DATA")
    print("=" * 96)

    for split, domains in [
        (
            "vector",
            vector_domains,
        ),
        (
            "test",
            test_domains,
        ),
    ]:
        print()
        print(split.upper())

        for domain in [
            "nlp",
            "political",
        ]:
            examples = domains[domain]

            print(
                f"  {domain:10s}: "
                f"N={len(examples):3d}  "
                f"matching-labels="
                f"{answer_balance(examples)}"
            )

    print()
    print("=" * 96)
    print("HALLUCINATION OOD DATA")
    print("=" * 96)

    print(f"TruthfulQA source size: {truthfulqa_source_size}")

    print(f"Frozen sample:          {len(truthfulqa)}")

    print(f"Matching-label balance: {answer_balance(truthfulqa)}")

    print(f"Exact core overlap:     {truthfulqa_overlap}")

    print()
    print("=" * 96)
    print("OUTPUT FILES")
    print("=" * 96)

    for name, path in OUTPUTS.items():
        print(f"{name:36s} {path}")

    print(f"{'manifest':36s} {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
