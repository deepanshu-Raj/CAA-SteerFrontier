"""Prepare the frozen GSM8K subset and few-shot demonstrations.

The script also audits tokenizer context lengths for every evaluated model so
the shared generation protocol fits each model consistently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from steerfrontier.config import load_experiment_config
from steerfrontier.prompts import get_prompt_adapter

# =============================================================================
# Frozen protocol
# =============================================================================

TEST_SIZE = 64
TEST_SEED = 20260919

GENERATION_RESERVE = 2048

SHOT_COUNTS = [
    4,
    5,
    8,
]

MODEL_ORDER = [
    "llama2-7b-chat",
    "llama3.1-8b-instruct",
    "mistral-7b-instruct-v0.3",
    "qwen2.5-7b-instruct",
]

OUTPUT_ROOT = Path("data/secondary")

TEST_PATH = OUTPUT_ROOT / "gsm8k_test_64.json"

DEMOS_PATH = OUTPUT_ROOT / "gsm8k_fewshot_8.json"

AUDIT_PATH = OUTPUT_ROOT / "gsm8k_fewshot_context_audit.csv"

MANIFEST_PATH = OUTPUT_ROOT / "gsm8k_secondary_data_manifest.json"


# These were the original eight randomly drawn training examples.
ORIGINAL_DEMO_INDICES = [
    7396,
    3892,
    6043,
    2784,
    693,
    3512,
    7083,
    326,
]

# We reject this exemplar before running the final few-shot experiment because
# its source reasoning is malformed as a demonstration.
REJECTED_DEMOS = {
    6043: (
        "Source reasoning is mathematically awkward as a few-shot exemplar: "
        "the quarter-to-ounce explanation is not expressed as a clean chain."
    ),
}

# Replacement is chosen deterministically without consulting model accuracy
# or steering outcomes.
DEMO_REPLACEMENT_SEED = 20260921


ANNOTATION_PATTERN = re.compile(r"<<[^<>]*>>")

FINAL_ANSWER_PATTERN = re.compile(r"(?m)^####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$")


# =============================================================================
# General helpers
# =============================================================================


def discover_configs():
    configs = {}

    for path in sorted(Path("configs").glob("*.yaml")):
        try:
            config = load_experiment_config(str(path))
        except Exception:
            continue

        if config.model_slug in MODEL_ORDER:
            configs[config.model_slug] = path

    missing = [slug for slug in MODEL_ORDER if slug not in configs]

    if missing:
        raise RuntimeError(f"Missing configs: {missing}")

    return configs


def clean_gsm8k_solution(
    solution: str,
):
    """
    Mechanical cleanup only.

    1. Remove GSM8K calculator annotations such as:
         <<16*8=128>>
       while retaining the already-rendered result that follows them.

    2. Fix the known tokenization/typing artifact:
         to.school -> to school

    3. Normalize horizontal whitespace without rewriting reasoning.
    """

    cleaned = ANNOTATION_PATTERN.sub(
        "",
        solution,
    )

    cleaned = cleaned.replace(
        "to.school",
        "to school",
    )

    normalized_lines = []

    for line in cleaned.splitlines():
        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        if line:
            normalized_lines.append(line)

    cleaned = "\n".join(normalized_lines).strip()

    return cleaned


def extract_final_answer(
    solution: str,
):
    matches = list(FINAL_ANSWER_PATTERN.finditer(solution))

    if not matches:
        raise ValueError("Could not find final GSM8K answer marker.")

    return (
        matches[-1]
        .group(1)
        .replace(
            ",",
            "",
        )
    )


# =============================================================================
# Prompt
# =============================================================================


def build_fewshot_prompt(
    *,
    demonstrations,
    question: str,
):
    parts = [
        (
            "Solve the final grade-school math word problem. "
            "Use the worked examples below to understand the task "
            "and answer format."
        )
    ]

    for index, demo in enumerate(
        demonstrations,
        start=1,
    ):
        parts.append(
            (
                f"Worked Example {index}\n\n"
                f"Problem:\n"
                f"{demo['question'].strip()}\n\n"
                f"Solution:\n"
                f"{demo['solution_clean'].strip()}"
            )
        )

    parts.append(
        (
            "Now solve the following problem yourself. "
            "Reason step by step and perform the arithmetic carefully.\n\n"
            "First write your reasoning and complete all calculations. "
            "Do not write the final-answer line until your reasoning "
            "is finished.\n\n"
            "After the reasoning, write exactly one final-answer line "
            "using the same four-hash format shown in the worked examples. "
            "The final-answer line must be the last non-empty line of your "
            "response. It must contain only the four hash characters, one "
            "space, and the numerical answer. Do not include units, commas, "
            "currency symbols, or any text after it.\n\n"
            f"Problem:\n"
            f"{question.strip()}\n\n"
            "Solution:"
        )
    )

    return "\n\n".join(parts)


# =============================================================================
# Test subset
# =============================================================================


def prepare_test_subset(
    test,
):
    rng = np.random.default_rng(TEST_SEED)

    indices = rng.choice(
        len(test),
        size=TEST_SIZE,
        replace=False,
    )

    # Sorting does not affect the random sample itself.
    # It only gives stable/readable output order.
    indices = np.sort(indices)

    rows = []

    for subset_index, source_index in enumerate(indices.tolist()):
        item = test[int(source_index)]

        gold = extract_final_answer(str(item["answer"]))

        rows.append(
            {
                "subset_index": int(subset_index),
                "source_index": int(source_index),
                "question": str(item["question"]),
                "gold_answer": (gold),
            }
        )

    return rows


# =============================================================================
# Few-shot demonstrations
# =============================================================================


def choose_replacement_demo(
    *,
    train,
):
    used = set(ORIGINAL_DEMO_INDICES)

    rejected = set(REJECTED_DEMOS)

    rng = np.random.default_rng(DEMO_REPLACEMENT_SEED)

    candidates = rng.permutation(len(train))

    for candidate in candidates.tolist():
        candidate = int(candidate)

        if candidate in used:
            continue

        if candidate in rejected:
            continue

        item = train[candidate]

        raw_solution = str(item["answer"])

        try:
            extract_final_answer(raw_solution)
        except ValueError:
            continue

        return candidate

    raise RuntimeError("Could not find deterministic replacement exemplar.")


def prepare_demonstrations(
    train,
):
    replacement_index = choose_replacement_demo(train=train)

    resolved_indices = []

    for source_index in ORIGINAL_DEMO_INDICES:
        if source_index in REJECTED_DEMOS:
            resolved_indices.append(replacement_index)
        else:
            resolved_indices.append(source_index)

    if len(set(resolved_indices)) != 8:
        raise RuntimeError("Few-shot demonstration indices are not unique.")

    rows = []

    for demo_order, source_index in enumerate(resolved_indices):
        item = train[int(source_index)]

        raw_solution = str(item["answer"])

        clean_solution = clean_gsm8k_solution(raw_solution)

        if "<<" in clean_solution or ">>" in clean_solution:
            raise RuntimeError("Calculator annotation survived cleanup.")

        raw_gold = extract_final_answer(raw_solution)

        clean_gold = extract_final_answer(clean_solution)

        if raw_gold != clean_gold:
            raise RuntimeError("Cleaning changed final answer.")

        was_replacement = source_index == replacement_index

        rows.append(
            {
                "demo_order": int(demo_order),
                "source_index": int(source_index),
                "was_replacement": bool(was_replacement),
                "question": str(item["question"]),
                "solution_raw": (raw_solution),
                "solution_clean": (clean_solution),
                "gold_answer": (clean_gold),
            }
        )

    return (
        rows,
        replacement_index,
    )


# =============================================================================
# Context-window audit
# =============================================================================


def infer_context_window(
    config_object,
    tokenizer,
):
    candidates = []

    for attribute in [
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
    ]:
        value = getattr(
            config_object,
            attribute,
            None,
        )

        if (
            value is not None
            and isinstance(
                value,
                int,
            )
            and value > 0
        ):
            candidates.append(int(value))

    tokenizer_limit = getattr(
        tokenizer,
        "model_max_length",
        None,
    )

    if (
        isinstance(
            tokenizer_limit,
            int,
        )
        and tokenizer_limit > 0
        and tokenizer_limit < 1_000_000
    ):
        candidates.append(int(tokenizer_limit))

    if not candidates:
        raise ValueError("Could not infer finite context window.")

    return min(candidates)


def audit_context_lengths(
    *,
    test_rows,
    demo_rows,
):
    configs = discover_configs()

    rows = []

    print()
    print("=" * 88)
    print("FEW-SHOT CONTEXT AUDIT")
    print("=" * 88)

    for slug in MODEL_ORDER:
        config = load_experiment_config(str(configs[slug]))

        print()
        print(f"Loading tokenizer: {config.model_label}")

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        model_config = AutoConfig.from_pretrained(config.model_name)

        context_window = infer_context_window(
            model_config,
            tokenizer,
        )

        adapter = get_prompt_adapter(
            model=None,
            tokenizer=tokenizer,
            model_name=(config.model_name),
            prompt_family=(config.prompt_family),
        )

        for num_shots in SHOT_COUNTS:
            demonstrations = demo_rows[:num_shots]

            lengths = []

            for item in test_rows:
                context = build_fewshot_prompt(
                    demonstrations=(demonstrations),
                    question=(item["question"]),
                )

                probe = adapter.build_continuation_prompt(
                    tokenizer,
                    context,
                    "0",
                )

                prompt_tokens = int(probe.continuation_start_position)

                lengths.append(prompt_tokens)

            array = np.asarray(
                lengths,
                dtype=int,
            )

            max_prompt = int(array.max())

            headroom = int(context_window - max_prompt - GENERATION_RESERVE)

            fits = headroom >= 0

            row = {
                "model_slug": (config.model_slug),
                "model_label": (config.model_label),
                "prompt_family": (config.prompt_family),
                "shots": int(num_shots),
                "context_window": int(context_window),
                "generation_reserve": int(GENERATION_RESERVE),
                "prompt_tokens_min": int(array.min()),
                "prompt_tokens_median": float(np.median(array)),
                "prompt_tokens_p95": float(
                    np.quantile(
                        array,
                        0.95,
                    )
                ),
                "prompt_tokens_max": int(max_prompt),
                "headroom_after_2048_at_max_prompt": int(headroom),
                "fits_with_2048_reserve": bool(fits),
            }

            rows.append(row)

            print(
                f"{num_shots}-shot | "
                f"context={context_window:6d} | "
                f"median={np.median(array):7.1f} | "
                f"p95={np.quantile(array, 0.95):7.1f} | "
                f"max={max_prompt:5d} | "
                f"headroom={headroom:6d} | "
                f"fits={fits}"
            )

    audit = pd.DataFrame(rows)

    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )

    return audit


# =============================================================================
# Main
# =============================================================================


def main():
    print("Loading GSM8K...")

    train = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
    )

    test = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
    )

    test_rows = prepare_test_subset(test)

    (
        demo_rows,
        replacement_index,
    ) = prepare_demonstrations(train)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            test_rows,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with DEMOS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            demo_rows,
            f,
            indent=2,
            ensure_ascii=False,
        )

    manifest = {
        "dataset": ("openai/gsm8k"),
        "config": ("main"),
        "test": {
            "split": ("test"),
            "num_examples": int(TEST_SIZE),
            "seed": int(TEST_SEED),
            "sampling": (
                "uniform without replacement; sampled indices sorted before saving"
            ),
        },
        "fewshot": {
            "split": ("train"),
            "num_shots": 8,
            "original_random_indices": (ORIGINAL_DEMO_INDICES),
            "rejected_indices": (
                {str(key): value for key, value in REJECTED_DEMOS.items()}
            ),
            "replacement_seed": int(DEMO_REPLACEMENT_SEED),
            "replacement_source_index": int(replacement_index),
            "cleaning": [
                ("Remove GSM8K <<...>> calculator annotations."),
                ("Normalize horizontal whitespace."),
                ("Replace known typo 'to.school' with 'to school'."),
            ],
            "selection_note": (
                "No model accuracy or steering outcome "
                "was used to select or replace demonstrations."
            ),
        },
        "candidate_shot_counts": (SHOT_COUNTS),
        "shot_selection_rule": (
            "Choose the largest candidate shot count "
            "that preserves a 2048-token generation reserve "
            "for every model."
        ),
        "selected_shot_count": 8,
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 88)
    print("FROZEN DATA")
    print("=" * 88)
    print(f"Test subset: {TEST_PATH}")
    print(f"Test examples: {len(test_rows)}")
    print(f"Demo pool:   {DEMOS_PATH}")
    print(f"Demo count:  {len(demo_rows)}")
    print(f"Replacement source index: train[{replacement_index}]")

    print()
    print("=" * 88)
    print("FINAL DEMONSTRATIONS")
    print("=" * 88)

    for demo in demo_rows:
        marker = " [REPLACEMENT]" if demo["was_replacement"] else ""

        print()
        print(f"DEMO {demo['demo_order'] + 1}: train[{demo['source_index']}]{marker}")
        print()
        print(demo["question"])
        print()
        print(demo["solution_clean"])

    audit = audit_context_lengths(
        test_rows=test_rows,
        demo_rows=demo_rows,
    )

    print()
    print("=" * 88)
    print("COMPACT AUDIT")
    print("=" * 88)

    print(
        audit[
            [
                "model_slug",
                "shots",
                "context_window",
                "prompt_tokens_max",
                "headroom_after_2048_at_max_prompt",
                "fits_with_2048_reserve",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Audit saved: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
