"""Judge blinded open-ended response triplets with a structured scoring rubric.

The judge receives randomized A/B/C labels only; true steering conditions are
kept outside the judging loop until downstream unblinding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

# =============================================================================
# Frozen judge protocol
# =============================================================================

MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 1200

PACKETS_PATH = (
    Path("results")
    / "cross_model"
    / "open_ended_generation"
    / "judging"
    / "judge_packets.jsonl"
)

RUBRICS_PATH = (
    Path("results")
    / "cross_model"
    / "open_ended_generation"
    / "judging"
    / "rubrics.json"
)

DEFAULT_OUTPUT_ROOT = (
    Path("results")
    / "cross_model"
    / "open_ended_generation"
    / "judging"
    / "judge_results"
)

EXPECTED_CASES = 276

# Standard GPT-5.6 Sol pricing observed/frozen for this experiment.
# USD per 1M tokens.
INPUT_PRICE = 4.00
CACHED_INPUT_PRICE = 0.40
OUTPUT_PRICE = 20.00


# =============================================================================
# Structured output
# =============================================================================


class ResponseAssessment(BaseModel):
    score: int = Field(
        ge=0,
        le=4,
    )

    rationale: str = Field(
        description=(
            "Brief justification based only on the response and the supplied rubric."
        )
    )


class JudgeOutput(BaseModel):
    A: ResponseAssessment
    B: ResponseAssessment
    C: ResponseAssessment


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Blindly judge open-ended CAA generations using GPT-5.6 Sol.")
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Judge only the first N frozen blinded cases. "
            "Use with a separate --output-root for smoke tests."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_jsonl(
    path: Path,
):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


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


def write_csv(
    path: Path,
    frame: pd.DataFrame,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    frame.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(path)


def estimated_cost(
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
):
    uncached = max(
        0,
        int(input_tokens) - int(cached_input_tokens),
    )

    return float(
        (
            uncached * INPUT_PRICE
            + cached_input_tokens * CACHED_INPUT_PRICE
            + output_tokens * OUTPUT_PRICE
        )
        / 1_000_000.0
    )


# =============================================================================
# Prompt construction
# =============================================================================

SYSTEM_INSTRUCTION = """
You are an evaluator for a controlled language-model research experiment.

You will receive:
1. one user prompt,
2. three anonymized responses labeled A, B, and C,
3. a behavior-specific scoring rubric.

The responses come from the same underlying model under hidden experimental
conditions. You do not know those conditions, and you must not attempt to
infer or guess them.

Judge the three responses jointly for consistency, but assign an independent
integer target-behavior score from 0 through 4 to each response.

Important:
- Follow the supplied behavior-specific rubric exactly.
- Score the TARGET BEHAVIOR, not general response quality.
- Do not reward eloquence, verbosity, confidence, or writing style.
- Do not punish a response merely because you disagree with it.
- Ties are explicitly allowed.
- Use any supplied reference only for the purpose described by the rubric.
- Do not rank A/B/C unless their actual behavior scores differ.
- Give a brief rationale for each score.
""".strip()


def make_case_input(
    packet,
    rubric_payload,
):
    behavior = str(packet["behavior"])

    rubrics = rubric_payload["rubrics"]

    if behavior not in rubrics:
        raise RuntimeError(f"Missing rubric for {behavior}.")

    rubric = rubrics[behavior]

    scale_lines = []

    for score in [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]:
        scale_lines.append(f"{score}: {rubric['scale'][score]}")

    reference = packet.get("reference") or {}

    reference_text = (
        json.dumps(
            reference,
            ensure_ascii=False,
            indent=2,
        )
        if reference
        else "None supplied."
    )

    responses = packet["responses"]

    return f"""
CASE ID
{packet["case_id"]}

TARGET BEHAVIOR
{behavior}

BEHAVIOR DEFINITION
{rubric["target_behavior"]}

IMPORTANT RUBRIC NOTE
{rubric["important_note"]}

SCORING SCALE
{chr(10).join(scale_lines)}

USER PROMPT
--- BEGIN USER PROMPT ---
{packet["prompt"]}
--- END USER PROMPT ---

REFERENCE
--- BEGIN REFERENCE ---
{reference_text}
--- END REFERENCE ---

RESPONSE A
--- BEGIN RESPONSE A ---
{responses["A"]}
--- END RESPONSE A ---

RESPONSE B
--- BEGIN RESPONSE B ---
{responses["B"]}
--- END RESPONSE B ---

RESPONSE C
--- BEGIN RESPONSE C ---
{responses["C"]}
--- END RESPONSE C ---

Return independent 0-4 scores and concise rationales for A, B, and C.
""".strip()


# =============================================================================
# API usage extraction
# =============================================================================


def extract_usage(
    response,
):
    usage = response.usage

    if usage is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

    input_tokens = int(
        getattr(
            usage,
            "input_tokens",
            0,
        )
        or 0
    )

    output_tokens = int(
        getattr(
            usage,
            "output_tokens",
            0,
        )
        or 0
    )

    cached_input_tokens = 0

    input_details = getattr(
        usage,
        "input_tokens_details",
        None,
    )

    if input_details is not None:
        cached_input_tokens = int(
            getattr(
                input_details,
                "cached_tokens",
                0,
            )
            or 0
        )

    reasoning_tokens = 0

    output_details = getattr(
        usage,
        "output_tokens_details",
        None,
    )

    if output_details is not None:
        reasoning_tokens = int(
            getattr(
                output_details,
                "reasoning_tokens",
                0,
            )
            or 0
        )

    return {
        "input_tokens": (input_tokens),
        "cached_input_tokens": (cached_input_tokens),
        "output_tokens": (output_tokens),
        "reasoning_tokens": (reasoning_tokens),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    packets = load_jsonl(PACKETS_PATH)

    rubrics = load_json(RUBRICS_PATH)

    if len(packets) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_CASES} blinded cases, found {len(packets)}."
        )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1.")

        packets = packets[: args.limit]

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = args.output_root / "judge_results.csv"

    manifest_path = args.output_root / "manifest.json"

    if results_path.is_file():
        existing = pd.read_csv(results_path)
    else:
        existing = pd.DataFrame()

    completed = (
        set(existing["case_id"].astype(str))
        if (not existing.empty and "case_id" in existing.columns)
        else set()
    )

    print("=" * 100)
    print("BLINDED OPEN-ENDED JUDGE")
    print("=" * 100)

    print(f"Model:             {MODEL}")

    print(f"Reasoning effort:  {REASONING_EFFORT}")

    print(f"Cases requested:   {len(packets)}")

    print(f"Already complete:  {len(completed)}")

    print(f"Output:            {args.output_root}")

    print()

    client = OpenAI()

    rows = existing.to_dict(orient="records") if not existing.empty else []

    for index, packet in enumerate(
        packets,
        start=1,
    ):
        case_id = str(packet["case_id"])

        if case_id in completed:
            print(f"[{index}/{len(packets)}] {case_id}: already complete")
            continue

        print(f"[{index}/{len(packets)}] {case_id}: judging...")

        user_input = make_case_input(
            packet,
            rubrics,
        )

        response = client.responses.parse(
            model=MODEL,
            reasoning={
                "effort": (REASONING_EFFORT),
            },
            instructions=(SYSTEM_INSTRUCTION),
            input=user_input,
            text_format=JudgeOutput,
            max_output_tokens=(MAX_OUTPUT_TOKENS),
        )

        parsed = response.output_parsed

        if parsed is None:
            raise RuntimeError(
                f"{case_id}: model returned "
                "no parsed structured output. "
                "No further cases were submitted; "
                "rerun to resume."
            )

        usage = extract_usage(response)

        cost = estimated_cost(
            input_tokens=(usage["input_tokens"]),
            cached_input_tokens=(usage["cached_input_tokens"]),
            output_tokens=(usage["output_tokens"]),
        )

        row = {
            "case_id": (case_id),
            "behavior": (packet["behavior"]),
            "A_score": int(parsed.A.score),
            "A_rationale": (parsed.A.rationale),
            "B_score": int(parsed.B.score),
            "B_rationale": (parsed.B.rationale),
            "C_score": int(parsed.C.score),
            "C_rationale": (parsed.C.rationale),
            "judge_model": (MODEL),
            "reasoning_effort": (REASONING_EFFORT),
            "input_tokens": (usage["input_tokens"]),
            "cached_input_tokens": (usage["cached_input_tokens"]),
            "output_tokens": (usage["output_tokens"]),
            "reasoning_tokens": (usage["reasoning_tokens"]),
            "estimated_cost_usd": (cost),
            "response_id": (response.id),
        }

        rows.append(row)

        completed.add(case_id)

        frame = (
            pd.DataFrame(rows)
            .drop_duplicates(
                subset=["case_id"],
                keep="last",
            )
            .sort_values("case_id")
            .reset_index(drop=True)
        )

        write_csv(
            results_path,
            frame,
        )

        running_cost = float(frame["estimated_cost_usd"].sum())

        print(f"    scores: A={parsed.A.score}, B={parsed.B.score}, C={parsed.C.score}")

        print(
            f"    tokens: "
            f"in={usage['input_tokens']}, "
            f"cached={usage['cached_input_tokens']}, "
            f"out={usage['output_tokens']}, "
            f"reasoning={usage['reasoning_tokens']}"
        )

        print(f"    estimated request cost: ${cost:.4f}")

        print(f"    running estimated cost: ${running_cost:.4f}")

    final = pd.read_csv(results_path)

    total_input = int(final["input_tokens"].sum())

    total_cached = int(final["cached_input_tokens"].sum())

    total_output = int(final["output_tokens"].sum())

    total_reasoning = int(final["reasoning_tokens"].sum())

    total_cost = float(final["estimated_cost_usd"].sum())

    mean_cost = float(final["estimated_cost_usd"].mean())

    projected_276 = mean_cost * EXPECTED_CASES

    manifest = {
        "protocol": ("open_ended_blinded_judge_v1"),
        "judge_model": (MODEL),
        "reasoning_effort": (REASONING_EFFORT),
        "max_output_tokens": int(MAX_OUTPUT_TOKENS),
        "requested_case_limit": (args.limit),
        "completed_cases": int(len(final)),
        "pricing_used_usd_per_million_tokens": {
            "input": (INPUT_PRICE),
            "cached_input": (CACHED_INPUT_PRICE),
            "output": (OUTPUT_PRICE),
        },
        "usage": {
            "input_tokens": (total_input),
            "cached_input_tokens": (total_cached),
            "output_tokens": (total_output),
            "reasoning_tokens": (total_reasoning),
        },
        "estimated_cost_usd": (total_cost),
        "mean_estimated_cost_per_case_usd": (mean_cost),
        "projected_276_case_cost_usd": (projected_276),
        "condition_key_accessed": (False),
    }

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 100)
    print("JUDGE RUN COMPLETE")
    print("=" * 100)

    print(f"Completed cases:      {len(final)}")

    print(f"Input tokens:         {total_input:,}")

    print(f"Cached input tokens:  {total_cached:,}")

    print(f"Output tokens:        {total_output:,}")

    print(f"Reasoning tokens:     {total_reasoning:,}")

    print(f"Estimated cost:       ${total_cost:.4f}")

    print(f"Mean cost / case:     ${mean_cost:.4f}")

    print(f"Projected 276 cost:   ${projected_276:.2f}")

    print()
    print(f"Results:  {results_path}")

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
