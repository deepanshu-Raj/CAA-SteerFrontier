"""Prepare deterministic capability-selection and holdout benchmark datasets.

Selection and holdout splits are frozen separately so holdout performance cannot
influence steering-budget selection.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

# =============================================================================
# Configuration
# =============================================================================

BASE_SEED = 42

MMLU_SELECTION_SIZE = 1500
MMLU_HOLDOUT_SIZE = 1500

HELLASWAG_SELECTION_SIZE = 1500
HELLASWAG_HOLDOUT_SIZE = 1500


def stable_seed(
    base_seed: int,
    name: str,
) -> int:
    payload = (f"{base_seed}:{name}").encode()

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def label_for_index(
    index: int,
) -> str:
    if index < 0 or index >= 26:
        raise ValueError(f"Invalid choice index: {index}")

    return f"({chr(ord('A') + index)})"


def deterministic_sample(
    rows,
    size: int,
    seed: int,
):
    rows = list(rows)

    if size <= 0:
        raise ValueError("Sample size must be positive.")

    if size > len(rows):
        raise ValueError(f"Requested {size} examples from pool of only {len(rows)}.")

    if size == len(rows):
        return rows

    rng = random.Random(seed)

    indices = list(range(len(rows)))

    rng.shuffle(indices)

    selected = sorted(indices[:size])

    return [rows[i] for i in selected]


# =============================================================================
# MMLU
# =============================================================================


def canonicalize_mmlu(
    dataset,
    source_split: str,
):
    rows = []

    for index, item in enumerate(dataset):
        choices = list(item["choices"])

        answer_index = int(item["answer"])

        canonical_choices = [
            {
                "label": (label_for_index(choice_index)),
                "text": str(text).strip(),
            }
            for choice_index, text in enumerate(choices)
        ]

        rows.append(
            {
                "id": (f"mmlu-{source_split}-{index}"),
                "question": str(item["question"]).strip(),
                "choices": (canonical_choices),
                "correct": (label_for_index(answer_index)),
                "category": str(
                    item.get(
                        "subject",
                        "unknown",
                    )
                ),
                "source_split": (source_split),
            }
        )

    return rows


def prepare_mmlu(
    base_seed: int,
):
    validation = load_dataset(
        "cais/mmlu",
        "all",
        split="validation",
    )

    test = load_dataset(
        "cais/mmlu",
        "all",
        split="test",
    )

    validation_rows = canonicalize_mmlu(
        validation,
        source_split="validation",
    )

    test_rows = canonicalize_mmlu(
        test,
        source_split="test",
    )

    selection = deterministic_sample(
        rows=validation_rows,
        size=MMLU_SELECTION_SIZE,
        seed=stable_seed(
            base_seed,
            "mmlu_selection",
        ),
    )

    holdout = deterministic_sample(
        rows=test_rows,
        size=MMLU_HOLDOUT_SIZE,
        seed=stable_seed(
            base_seed,
            "mmlu_holdout",
        ),
    )

    return selection, holdout


# =============================================================================
# ARC-Challenge
# =============================================================================


def canonicalize_arc(
    dataset,
    source_split: str,
):
    rows = []

    for index, item in enumerate(dataset):
        choices_obj = item["choices"]

        source_labels = [str(x).strip() for x in choices_obj["label"]]

        texts = [str(x).strip() for x in choices_obj["text"]]

        if len(source_labels) != len(texts):
            raise ValueError(
                f"ARC {source_split} item {index}: labels/texts differ in length."
            )

        source_answer = str(item["answerKey"]).strip()

        try:
            correct_index = source_labels.index(source_answer)

        except ValueError as exc:
            raise ValueError(
                f"ARC {source_split} item "
                f"{index}: answerKey "
                f"{source_answer!r} not "
                f"present in {source_labels!r}."
            ) from exc

        canonical_choices = [
            {
                "label": (label_for_index(choice_index)),
                "text": text,
            }
            for choice_index, text in enumerate(texts)
        ]

        rows.append(
            {
                "id": (f"arc-challenge-{source_split}-{item.get('id', index)}"),
                "question": str(item["question"]).strip(),
                "choices": (canonical_choices),
                "correct": (label_for_index(correct_index)),
                "category": ("arc_challenge"),
                "source_split": (source_split),
            }
        )

    return rows


def prepare_arc_challenge():
    train = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split="train",
    )

    validation = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split="validation",
    )

    test = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split="test",
    )

    selection = canonicalize_arc(
        train,
        source_split="train",
    ) + canonicalize_arc(
        validation,
        source_split="validation",
    )

    holdout = canonicalize_arc(
        test,
        source_split="test",
    )

    return selection, holdout


# =============================================================================
# HellaSwag
# =============================================================================


def clean_hellaswag_text(
    text: str,
) -> str:
    return " ".join(str(text).split())


def canonicalize_hellaswag(
    dataset,
    source_split: str,
):
    rows = []

    for index, item in enumerate(dataset):
        ctx_a = clean_hellaswag_text(
            item.get(
                "ctx_a",
                "",
            )
        )

        ctx_b = clean_hellaswag_text(
            item.get(
                "ctx_b",
                "",
            )
        )

        context = " ".join(
            part
            for part in [
                ctx_a,
                ctx_b,
            ]
            if part
        ).strip()

        endings = [clean_hellaswag_text(ending) for ending in item["endings"]]

        correct_index = int(item["label"])

        if correct_index < 0 or correct_index >= len(endings):
            raise ValueError(
                f"HellaSwag {source_split} item {index}: invalid label {correct_index}."
            )

        rows.append(
            {
                "id": (f"hellaswag-{source_split}-{index}"),
                "context": context,
                "endings": endings,
                "correct_index": (correct_index),
                "category": str(
                    item.get(
                        "activity_label",
                        "unknown",
                    )
                ),
                "source_split": (source_split),
            }
        )

    return rows


def prepare_hellaswag(
    base_seed: int,
):
    train = load_dataset(
        "Rowan/hellaswag",
        split="train",
    )

    validation = load_dataset(
        "Rowan/hellaswag",
        split="validation",
    )

    train_rows = canonicalize_hellaswag(
        train,
        source_split="train",
    )

    validation_rows = canonicalize_hellaswag(
        validation,
        source_split="validation",
    )

    selection = deterministic_sample(
        rows=train_rows,
        size=(HELLASWAG_SELECTION_SIZE),
        seed=stable_seed(
            base_seed,
            "hellaswag_selection",
        ),
    )

    holdout = deterministic_sample(
        rows=validation_rows,
        size=(HELLASWAG_HOLDOUT_SIZE),
        seed=stable_seed(
            base_seed,
            "hellaswag_holdout",
        ),
    )

    return selection, holdout


# =============================================================================
# Output
# =============================================================================


def write_json(
    path: Path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w") as f:
        json.dump(
            rows,
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_dataset(
    *,
    root: Path,
    role: str,
    name: str,
    rows,
):
    path = root / role / f"{name}.json"

    write_json(
        path,
        rows,
    )

    return path


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=("Prepare deterministic capability-selection and holdout datasets.")
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/capability"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
    )

    args = parser.parse_args()

    root = args.output_root

    print("=" * 76)

    print("PREPARING CAPABILITY DATA")

    print("=" * 76)

    (
        mmlu_selection,
        mmlu_holdout,
    ) = prepare_mmlu(
        base_seed=args.seed,
    )

    (
        arc_selection,
        arc_holdout,
    ) = prepare_arc_challenge()

    (
        hellaswag_selection,
        hellaswag_holdout,
    ) = prepare_hellaswag(
        base_seed=args.seed,
    )

    datasets = {
        "selection": {
            "mmlu": (mmlu_selection),
            "arc_challenge": (arc_selection),
            "hellaswag": (hellaswag_selection),
        },
        "holdout": {
            "mmlu": (mmlu_holdout),
            "arc_challenge": (arc_holdout),
            "hellaswag": (hellaswag_holdout),
        },
    }

    manifest = {
        "protocol": ("capability_selection_holdout"),
        "seed": int(args.seed),
        "roles": {
            "selection": (
                "Used to determine capability-feasible "
                "steering coefficients. These examples "
                "may influence alpha selection."
            ),
            "holdout": (
                "Untouched until steering layer and "
                "coefficient are frozen. Must not be "
                "used to reselect alpha."
            ),
        },
        "datasets": {},
    }

    for role, role_datasets in datasets.items():
        manifest["datasets"][role] = {}

        for name, rows in role_datasets.items():
            path = write_dataset(
                root=root,
                role=role,
                name=name,
                rows=rows,
            )

            manifest["datasets"][role][name] = {
                "num_examples": len(rows),
                "path": str(path),
                "sha256": (sha256_file(path)),
                "source_splits": sorted({row["source_split"] for row in rows}),
            }

            print(f"{role:10s} {name:15s} {len(rows):5d} -> {path}")

    manifest_path = root / "manifest.json"

    with manifest_path.open("w") as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        "Manifest ->",
        manifest_path,
    )


if __name__ == "__main__":
    main()
