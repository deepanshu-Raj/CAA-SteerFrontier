"""Prepare frozen free-form prompts for open-ended steering evaluation.

Forced-choice scaffolding is removed while preserving the underlying behavior
content, with deterministic sampling across the evaluated behaviors.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

SEED = 20260919

OUTPUT_ROOT = Path("data") / "open_ended"

SYC_NLP_PATH = Path("data/ood/sycophancy/nlp_test.json")

SYC_POLITICAL_PATH = Path("data/ood/sycophancy/political_test.json")

REFUSAL_PATH = Path("data/refusal/test.json")

HALLUCINATION_PATH = Path("data/ood/hallucination/truthfulqa_test.json")


# =============================================================================
# General helpers
# =============================================================================


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_json(
    path: Path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    temporary.replace(path)


def sha256_file(
    path: Path,
):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# =============================================================================
# Forced-choice -> open-ended conversion
# =============================================================================


def strip_forced_choice_scaffold(
    question: str,
):
    """
    Keep the natural-language user prompt and remove the
    trailing A/B forced-choice block.

    Handles both forms seen in the project:

        question...
        Choices:
        (A) ...
        (B) ...

        Answer:

    and:

        question...
        (A) ...
        (B) ...

        Answer:
    """

    if not isinstance(
        question,
        str,
    ):
        raise TypeError("Question must be a string.")

    text = question.replace(
        "\r\n",
        "\n",
    ).strip()

    lines = text.splitlines()

    cut_index = None

    # Preferred explicit Choices: marker.
    for i, line in enumerate(lines):
        if line.strip().lower() == "choices:":
            cut_index = i
            break

    # Some sycophancy political prompts do not
    # contain an explicit "Choices:" line.
    if cut_index is None:
        for i, line in enumerate(lines):
            if re.match(
                r"^\s*\(A\)\s+",
                line,
            ):
                later_has_b = any(
                    re.match(
                        r"^\s*\(B\)\s+",
                        later,
                    )
                    for later in lines[i + 1 :]
                )

                if later_has_b:
                    cut_index = i
                    break

    if cut_index is None:
        raise RuntimeError(
            f"Could not locate trailing forced-choice scaffold:\n{question}"
        )

    prompt = "\n".join(lines[:cut_index]).strip()

    if not prompt:
        raise RuntimeError("Prompt became empty after removing choices.")

    # Audit that the trailing choice scaffold
    # really disappeared.
    if re.search(
        r"(?m)^\s*Choices:\s*$",
        prompt,
        flags=re.IGNORECASE,
    ):
        raise RuntimeError("Choices marker remained.")

    if re.search(
        r"(?m)^\s*\(A\)\s+",
        prompt,
    ):
        raise RuntimeError("(A) choice remained.")

    if re.search(
        r"(?m)^\s*\(B\)\s+",
        prompt,
    ):
        raise RuntimeError("(B) choice remained.")

    if re.search(
        r"(?m)^\s*Answer:\s*$",
        prompt,
        flags=re.IGNORECASE,
    ):
        raise RuntimeError("Answer marker remained.")

    return prompt


# =============================================================================
# Deterministic sampling
# =============================================================================


def balanced_sample(
    data,
    *,
    field: str,
    values,
    per_value: int,
    rng: random.Random,
):
    chosen = []

    for value in values:
        candidates = [
            (
                source_index,
                item,
            )
            for source_index, item in enumerate(data)
            if item.get(field) == value
        ]

        if len(candidates) < per_value:
            raise RuntimeError(
                f"Need {per_value} rows "
                f"with {field}={value!r}; "
                f"found {len(candidates)}."
            )

        picked = rng.sample(
            candidates,
            per_value,
        )

        chosen.extend(picked)

    # Shuffle the final balanced selection
    # deterministically so strata are not grouped.
    rng.shuffle(chosen)

    return chosen


# =============================================================================
# Sycophancy
# =============================================================================


def prepare_sycophancy(
    rng: random.Random,
):
    nlp = load_json(SYC_NLP_PATH)

    political = load_json(SYC_POLITICAL_PATH)

    # 12 NLP: balanced across the original
    # A/B matching-answer position.
    nlp_sample = balanced_sample(
        nlp,
        field="answer_matching_behavior",
        values=[
            "(A)",
            "(B)",
        ],
        per_value=6,
        rng=rng,
    )

    # 12 political: explicitly balance the
    # user-affiliation domains.
    political_sample = balanced_sample(
        political,
        field="user_affiliation",
        values=[
            "liberal",
            "conservative",
        ],
        per_value=6,
        rng=rng,
    )

    rows = []

    for domain, path, sample in [
        (
            "nlp",
            SYC_NLP_PATH,
            nlp_sample,
        ),
        (
            "political",
            SYC_POLITICAL_PATH,
            political_sample,
        ),
    ]:
        for source_index, item in sample:
            prompt = strip_forced_choice_scaffold(item["question"])

            rows.append(
                {
                    "behavior": ("sycophancy"),
                    "domain": domain,
                    "prompt": prompt,
                    "source_path": str(path),
                    "source_file_index": int(source_index),
                    "canonical_source_index": (item.get("canonical_source_index")),
                    "answer_matching_behavior": (item["answer_matching_behavior"]),
                    "answer_not_matching_behavior": (
                        item["answer_not_matching_behavior"]
                    ),
                    "user_affiliation": (item.get("user_affiliation")),
                }
            )

    return rows


# =============================================================================
# Refusal
# =============================================================================


def prepare_refusal(
    rng: random.Random,
):
    data = load_json(REFUSAL_PATH)

    sample = balanced_sample(
        data,
        field="answer_matching_behavior",
        values=[
            "(A)",
            "(B)",
        ],
        per_value=12,
        rng=rng,
    )

    rows = []

    for source_index, item in sample:
        prompt = strip_forced_choice_scaffold(item["question"])

        rows.append(
            {
                "behavior": ("refusal"),
                "domain": ("canonical_test"),
                "prompt": prompt,
                "source_path": str(REFUSAL_PATH),
                "source_file_index": int(source_index),
                "answer_matching_behavior": (item["answer_matching_behavior"]),
                "answer_not_matching_behavior": (item["answer_not_matching_behavior"]),
            }
        )

    return rows


# =============================================================================
# Hallucination
# =============================================================================


def prepare_hallucination(
    rng: random.Random,
):
    data = load_json(HALLUCINATION_PATH)

    sample = balanced_sample(
        data,
        field="answer_matching_behavior",
        values=[
            "(A)",
            "(B)",
        ],
        per_value=12,
        rng=rng,
    )

    rows = []

    for source_index, item in sample:
        prompt = strip_forced_choice_scaffold(item["question"])

        rows.append(
            {
                "behavior": ("hallucination"),
                "domain": ("truthfulqa"),
                "prompt": prompt,
                "source_path": str(HALLUCINATION_PATH),
                "source_file_index": int(source_index),
                "truthfulqa_source_index": (item.get("source_index")),
                "truthful_answer": (item["truthful_answer"]),
                "incorrect_answer": (item["incorrect_answer"]),
                "answer_matching_behavior": (item["answer_matching_behavior"]),
                "answer_not_matching_behavior": (item["answer_not_matching_behavior"]),
            }
        )

    return rows


# =============================================================================
# Audit
# =============================================================================


def audit_rows(
    rows,
):
    if len(rows) != 72:
        raise RuntimeError(f"Expected 72 total prompts, found {len(rows)}.")

    expected = {
        "sycophancy": 24,
        "refusal": 24,
        "hallucination": 24,
    }

    counts = Counter(row["behavior"] for row in rows)

    if dict(counts) != expected:
        raise RuntimeError(f"Unexpected behavior counts: {dict(counts)}")

    syc_domains = Counter(
        row["domain"] for row in rows if row["behavior"] == "sycophancy"
    )

    if syc_domains != {
        "nlp": 12,
        "political": 12,
    }:
        raise RuntimeError(f"Unexpected sycophancy domains: {dict(syc_domains)}")

    prompts = [row["prompt"] for row in rows]

    if len(prompts) != len(set(prompts)):
        raise RuntimeError("Duplicate open-ended prompts found.")

    for row in rows:
        prompt = row["prompt"]

        forbidden = [
            "\nChoices:",
            "\n(A)",
            "\n(B)",
            "\nAnswer:",
        ]

        if any(marker in prompt for marker in forbidden):
            raise RuntimeError(f"Forced-choice scaffold remained in prompt:\n{prompt}")


# =============================================================================
# Main
# =============================================================================


def main():
    rng = random.Random(SEED)

    sycophancy = prepare_sycophancy(rng)

    refusal = prepare_refusal(rng)

    hallucination = prepare_hallucination(rng)

    rows = sycophancy + refusal + hallucination

    audit_rows(rows)

    # Add stable IDs only after final sample order
    # has been frozen.
    behavior_counters = Counter()

    for row in rows:
        behavior = row["behavior"]

        index = behavior_counters[behavior]

        behavior_counters[behavior] += 1

        row["prompt_id"] = f"{behavior}_{index:02d}"

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "sycophancy": (OUTPUT_ROOT / "sycophancy.json"),
        "refusal": (OUTPUT_ROOT / "refusal.json"),
        "hallucination": (OUTPUT_ROOT / "hallucination.json"),
    }

    grouped = {
        behavior: [row for row in rows if row["behavior"] == behavior]
        for behavior in outputs
    }

    for behavior, path in outputs.items():
        write_json(
            path,
            grouped[behavior],
        )

    combined_path = OUTPUT_ROOT / "prompts.json"

    write_json(
        combined_path,
        rows,
    )

    manifest = {
        "protocol": ("open_ended_generation_prompts_v1"),
        "seed": int(SEED),
        "num_prompts": int(len(rows)),
        "behaviors": {
            "sycophancy": {
                "num_prompts": 24,
                "domains": {
                    "nlp": 12,
                    "political": 12,
                },
                "selection": (
                    "NLP balanced 6/6 by original "
                    "matching-answer position; "
                    "political balanced 6 liberal / "
                    "6 conservative."
                ),
            },
            "refusal": {
                "num_prompts": 24,
                "selection": ("Balanced 12/12 by original matching-answer position."),
            },
            "hallucination": {
                "num_prompts": 24,
                "domain": ("TruthfulQA"),
                "selection": ("Balanced 12/12 by original matching-answer position."),
            },
        },
        "prompt_transformation": (
            "Remove only the trailing forced-choice "
            "scaffold. Preserve the original natural-"
            "language user statement and question. "
            "No additional behavioral instruction "
            "is appended."
        ),
        "evaluation_role": (
            "Supporting open-ended modality-transfer "
            "validation; not an independent benchmark."
        ),
        "source_files": {
            str(SYC_NLP_PATH): sha256_file(SYC_NLP_PATH),
            str(SYC_POLITICAL_PATH): sha256_file(SYC_POLITICAL_PATH),
            str(REFUSAL_PATH): sha256_file(REFUSAL_PATH),
            str(HALLUCINATION_PATH): sha256_file(HALLUCINATION_PATH),
        },
    }

    write_json(
        OUTPUT_ROOT / "manifest.json",
        manifest,
    )

    print("=" * 100)
    print("OPEN-ENDED GENERATION DATA FROZEN")
    print("=" * 100)

    print(f"Seed:        {SEED}")
    print(f"Total:       {len(rows)}")
    print(f"Sycophancy:  {len(sycophancy)} (12 NLP + 12 political)")
    print(f"Refusal:     {len(refusal)}")
    print(f"Hallucination: {len(hallucination)}")

    for behavior in [
        "sycophancy",
        "refusal",
        "hallucination",
    ]:
        print()
        print("=" * 100)
        print(behavior.upper())
        print("=" * 100)

        for row in grouped[behavior][:3]:
            print()
            print(f"[{row['prompt_id']}] domain={row['domain']}")
            print(row["prompt"])

    print()
    print("=" * 100)
    print("FILES")
    print("=" * 100)

    for behavior, path in outputs.items():
        print(f"{behavior:15s} {path}")

    print(f"{'combined':15s} {combined_path}")

    print(f"{'manifest':15s} {OUTPUT_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
