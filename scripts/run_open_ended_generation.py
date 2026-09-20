"""Generate free-form responses under frozen negative, baseline, and positive steering.

The same prompts and deterministic decoding protocol are used across conditions;
condition outputs are checkpointed for safe resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse the exact generation + batched steering implementation
# already exercised by the GSM8K experiment.
from run_gsm8k_secondary import generate_batch

from steerfrontier.config import load_experiment_config
from steerfrontier.models import load_model
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.workflow import load_normalized_vectors

# =============================================================================
# Frozen protocol
# =============================================================================

BEHAVIORS = (
    "sycophancy",
    "refusal",
    "hallucination",
)

CONDITIONS = (
    "baseline",
    "negative",
    "positive",
)

EPSILON = 0.03
PROMPTS_PER_BEHAVIOR = 24
EXPECTED_CONDITIONS = 9

DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 256

DATA_ROOT = Path("data/open_ended")


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen open-ended generation under "
            "baseline and CAA steering conditions."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-prompts-per-behavior",
        type=int,
        default=None,
        help=("Optional prefix truncation for smoke tests. Requires --output-root."),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
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
    frame: pd.DataFrame,
    path: Path,
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
# Frozen operating points
# =============================================================================


def load_frozen_points(
    config,
):
    path = config.results_root / "final_evaluation" / "frontier_test_results.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "behavior",
        "direction",
        "epsilon",
        "layer",
        "alpha_star",
        "eta_star",
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

    frame = frame[
        frame["behavior"].isin(BEHAVIORS)
        & np.isclose(
            frame["epsilon"].astype(float),
            EPSILON,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

    for behavior in BEHAVIORS:
        rows = frame[frame["behavior"] == behavior]

        if len(rows) != 2:
            raise RuntimeError(
                f"{behavior}: expected "
                f"two epsilon={EPSILON} "
                f"frontier rows, found "
                f"{len(rows)}."
            )

        if set(rows["direction"]) != {
            "negative",
            "positive",
        }:
            raise RuntimeError(
                f"{behavior}: expected exactly negative and positive directions."
            )

        if rows["layer"].astype(int).nunique() != 1:
            raise RuntimeError(
                f"{behavior}: frozen positive and negative rows use different layers."
            )

    return (
        frame.reset_index(drop=True),
        path,
    )


def frozen_point(
    points: pd.DataFrame,
    *,
    behavior: str,
    direction: str,
):
    rows = points[(points["behavior"] == behavior) & (points["direction"] == direction)]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one frozen row for {behavior}/{direction}; found {len(rows)}."
        )

    return rows.iloc[0].to_dict()


# =============================================================================
# Frozen prompts
# =============================================================================


def load_behavior_prompts(
    behavior: str,
    *,
    max_prompts=None,
):
    path = DATA_ROOT / f"{behavior}.json"

    if not path.is_file():
        raise FileNotFoundError(path)

    rows = load_json(path)

    if len(rows) != PROMPTS_PER_BEHAVIOR:
        raise RuntimeError(
            f"{path}: expected {PROMPTS_PER_BEHAVIOR} prompts, found {len(rows)}."
        )

    required = {
        "prompt_id",
        "behavior",
        "domain",
        "prompt",
    }

    for index, row in enumerate(rows):
        missing = required - set(row)

        if missing:
            raise RuntimeError(f"{path}: item {index} missing {sorted(missing)}.")

        if row["behavior"] != behavior:
            raise RuntimeError(
                f"{path}: item {index} has behavior={row['behavior']!r}."
            )

    ids = [row["prompt_id"] for row in rows]

    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{path}: duplicate prompt IDs.")

    full_size = len(rows)

    if max_prompts is not None:
        if max_prompts < 1:
            raise ValueError("--max-prompts-per-behavior must be >= 1.")

        rows = rows[:max_prompts]

    return (
        rows,
        full_size,
        path,
    )


# =============================================================================
# Durable condition checkpointing
# =============================================================================


def load_partial_condition(
    path: Path,
    expected_prompt_ids,
    *,
    resume: bool,
):
    if not resume or not path.is_file():
        return pd.DataFrame()

    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    required = {
        "prompt_id",
        "response",
        "condition",
        "behavior",
    }

    if not required.issubset(frame.columns):
        return pd.DataFrame()

    if frame["prompt_id"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate prompt IDs.")

    expected = set(expected_prompt_ids)

    found = set(frame["prompt_id"].astype(str))

    if not found.issubset(expected):
        raise RuntimeError(f"{path}: contains unexpected prompt IDs.")

    return frame


def append_checkpoint(
    existing: pd.DataFrame,
    new_rows,
    path: Path,
):
    new_frame = pd.DataFrame(new_rows)

    if existing.empty:
        combined = new_frame
    else:
        combined = pd.concat(
            [
                existing,
                new_frame,
            ],
            ignore_index=True,
        )

    if combined["prompt_id"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate prompt ID during checkpoint.")

    combined = combined.sort_values("prompt_id").reset_index(drop=True)

    write_csv(
        combined,
        path,
    )

    return combined


# =============================================================================
# One condition
# =============================================================================


def run_condition(
    *,
    model,
    tokenizer,
    adapter,
    model_slug: str,
    behavior: str,
    condition: str,
    dataset,
    layer: int,
    vector,
    alpha: float,
    eta: float,
    batch_size: int,
    max_new_tokens: int,
    device: str,
    path: Path,
    resume: bool,
):
    expected_ids = [row["prompt_id"] for row in dataset]

    frame = load_partial_condition(
        path,
        expected_ids,
        resume=resume,
    )

    completed_ids = set(frame["prompt_id"].astype(str)) if not frame.empty else set()

    remaining = [row for row in dataset if row["prompt_id"] not in completed_ids]

    print(
        f"{behavior:14s} "
        f"{condition:8s} "
        f"L={layer:2d} "
        f"alpha={alpha:+.3f} "
        f"eta={eta:+.6f} "
        f"completed={len(completed_ids):2d}/"
        f"{len(dataset):2d}"
    )

    if not remaining:
        return frame

    for start in range(
        0,
        len(remaining),
        batch_size,
    ):
        batch_rows = remaining[start : start + batch_size]

        prompts = [row["prompt"] for row in batch_rows]

        outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            prompts=prompts,
            device=device,
            max_new_tokens=(max_new_tokens),
            layer=(layer if condition != "baseline" else None),
            vector=(vector if condition != "baseline" else None),
            alpha=(alpha if condition != "baseline" else 0.0),
        )

        if len(outputs) != len(batch_rows):
            raise RuntimeError("Generation output count does not match batch size.")

        checkpoint_rows = []

        for source, output in zip(
            batch_rows,
            outputs,
        ):
            checkpoint_rows.append(
                {
                    "model_slug": (model_slug),
                    "behavior": (behavior),
                    "condition": (condition),
                    "direction": (condition),
                    "prompt_id": (source["prompt_id"]),
                    "domain": (source["domain"]),
                    "layer": int(layer),
                    "alpha": float(alpha if condition != "baseline" else 0.0),
                    "eta": float(eta if condition != "baseline" else 0.0),
                    "prompt": (source["prompt"]),
                    "response": (output["response"]),
                    "prompt_token_count": int(output["prompt_token_count"]),
                    "new_token_count": int(output["new_token_count"]),
                    "ended_with_eos": bool(output["ended_with_eos"]),
                    "hit_max_tokens": bool(output["hit_max_tokens"]),
                }
            )

        frame = append_checkpoint(
            frame,
            checkpoint_rows,
            path,
        )

        completed = len(frame)

        print(f"    checkpoint {completed:2d}/{len(dataset):2d}")

    if len(frame) != len(dataset):
        raise RuntimeError(
            f"{behavior}/{condition}: expected {len(dataset)} rows, found {len(frame)}."
        )

    return frame


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.max_prompts_per_behavior is not None and args.output_root is None:
        raise ValueError(
            "--max-prompts-per-behavior "
            "is for smoke testing and "
            "requires --output-root."
        )

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    config = load_experiment_config(args.config)

    output_root = (
        args.output_root
        if args.output_root is not None
        else (config.results_root / "open_ended_generation")
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    resume = not (args.no_resume)

    # ----------------------------------------------------------------
    # Frozen prompts
    # ----------------------------------------------------------------

    datasets = {}
    full_sizes = {}
    data_paths = {}

    for behavior in BEHAVIORS:
        (
            rows,
            full_size,
            path,
        ) = load_behavior_prompts(
            behavior,
            max_prompts=(args.max_prompts_per_behavior),
        )

        datasets[behavior] = rows

        full_sizes[behavior] = full_size

        data_paths[behavior] = path

    # ----------------------------------------------------------------
    # Frozen frontier
    # ----------------------------------------------------------------

    (
        points,
        frontier_path,
    ) = load_frozen_points(config)

    vectors = load_normalized_vectors(config)

    for behavior in BEHAVIORS:
        if behavior not in vectors:
            raise RuntimeError(f"Canonical vector missing for {behavior}.")

    # ----------------------------------------------------------------
    # Print frozen plan
    # ----------------------------------------------------------------

    print("=" * 100)
    print("OPEN-ENDED GENERATION")
    print("=" * 100)

    print(f"Model:          {config.model_label}")
    print(f"Model slug:     {config.model_slug}")
    print(f"Epsilon:        {EPSILON}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")

    if args.max_prompts_per_behavior is not None:
        print(f"SMOKE prompts:  {args.max_prompts_per_behavior}/behavior")

    print()

    frozen_specs = {}

    for behavior in BEHAVIORS:
        negative = frozen_point(
            points,
            behavior=behavior,
            direction="negative",
        )

        positive = frozen_point(
            points,
            behavior=behavior,
            direction="positive",
        )

        layer = int(negative["layer"])

        if layer != int(positive["layer"]):
            raise RuntimeError(f"{behavior}: frozen layer mismatch.")

        frozen_specs[behavior] = {
            "layer": layer,
            "negative": {
                "alpha": float(negative["alpha_star"]),
                "eta": float(negative["eta_star"]),
            },
            "positive": {
                "alpha": float(positive["alpha_star"]),
                "eta": float(positive["eta_star"]),
            },
        }

        print(
            f"{behavior:14s} "
            f"L={layer:2d}  "
            f"negative "
            f"alpha="
            f"{frozen_specs[behavior]['negative']['alpha']:+.3f} "
            f"eta="
            f"{frozen_specs[behavior]['negative']['eta']:+.6f}  "
            f"positive "
            f"alpha="
            f"{frozen_specs[behavior]['positive']['alpha']:+.3f} "
            f"eta="
            f"{frozen_specs[behavior]['positive']['eta']:+.6f}"
        )

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------

    print()
    print("Loading model...")

    (
        model,
        tokenizer,
    ) = load_model(
        model_name=(config.model_name),
        device=args.device,
        dtype=(config.torch_dtype),
    )

    model.eval()

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=(config.model_name),
        prompt_family=(config.prompt_family),
    )

    print(f"Prompt adapter: {adapter.family}")

    # ----------------------------------------------------------------
    # Generation
    # ----------------------------------------------------------------

    all_frames = []

    for behavior in BEHAVIORS:
        spec = frozen_specs[behavior]

        layer = int(spec["layer"])

        tensor = vectors[behavior]

        if not (0 <= layer < tensor.shape[0]):
            raise RuntimeError(
                f"{behavior}: invalid "
                f"layer {layer} for vector "
                f"shape {tuple(tensor.shape)}."
            )

        vector = tensor[layer]

        print()
        print("=" * 100)
        print(behavior.upper())
        print("=" * 100)

        for condition in CONDITIONS:
            if condition == "baseline":
                alpha = 0.0
                eta = 0.0

            else:
                alpha = float(spec[condition]["alpha"])

                eta = float(spec[condition]["eta"])

            path = output_root / "conditions" / f"{behavior}__{condition}.csv"

            frame = run_condition(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                model_slug=(config.model_slug),
                behavior=behavior,
                condition=condition,
                dataset=(datasets[behavior]),
                layer=layer,
                vector=vector,
                alpha=alpha,
                eta=eta,
                batch_size=(args.batch_size),
                max_new_tokens=(args.max_new_tokens),
                device=args.device,
                path=path,
                resume=resume,
            )

            all_frames.append(frame)

    # ----------------------------------------------------------------
    # Combined output
    # ----------------------------------------------------------------

    combined = pd.concat(
        all_frames,
        ignore_index=True,
    )

    expected_rows = sum(len(datasets[behavior]) for behavior in BEHAVIORS) * len(
        CONDITIONS
    )

    if len(combined) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} generated responses, found {len(combined)}."
        )

    if (
        combined[
            [
                "behavior",
                "condition",
                "prompt_id",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError("Duplicate generation keys.")

    combined = combined.sort_values(
        [
            "behavior",
            "prompt_id",
            "condition",
        ]
    ).reset_index(drop=True)

    combined_path = output_root / "generations.csv"

    write_csv(
        combined,
        combined_path,
    )

    # ----------------------------------------------------------------
    # Mechanical summary only
    # ----------------------------------------------------------------

    summary = combined.groupby(
        [
            "behavior",
            "condition",
        ],
        as_index=False,
    ).agg(
        num_prompts=(
            "prompt_id",
            "count",
        ),
        mean_prompt_tokens=(
            "prompt_token_count",
            "mean",
        ),
        mean_new_tokens=(
            "new_token_count",
            "mean",
        ),
        eos_rate=(
            "ended_with_eos",
            "mean",
        ),
        truncation_rate=(
            "hit_max_tokens",
            "mean",
        ),
    )

    summary_path = output_root / "generation_summary.csv"

    write_csv(
        summary,
        summary_path,
    )

    # ----------------------------------------------------------------
    # Manifest
    # ----------------------------------------------------------------

    manifest = {
        "protocol": ("open_ended_generation_v1"),
        "model_name": (config.model_name),
        "model_label": (config.model_label),
        "model_slug": (config.model_slug),
        "behaviors": list(BEHAVIORS),
        "conditions": list(CONDITIONS),
        "epsilon": float(EPSILON),
        "generation": {
            "do_sample": False,
            "max_new_tokens": int(args.max_new_tokens),
            "batch_size": int(args.batch_size),
            "temperature": 1.0,
            "top_p": 1.0,
        },
        "prompt_policy": (
            "Use frozen natural-language "
            "open-ended prompts exactly as "
            "stored. No additional behavioral "
            "instruction is appended."
        ),
        "vector_policy": ("Use canonical normalized CAA vector for each behavior."),
        "intervention_policy": (
            "Use frozen behavior-selected "
            "layer and frozen epsilon=0.03 "
            "positive/negative coefficients. "
            "Baseline uses alpha=0."
        ),
        "no_generation_tuning": True,
        "no_output_based_prompt_selection": True,
        "smoke_prefix": bool(args.max_prompts_per_behavior is not None),
        "max_prompts_per_behavior": (args.max_prompts_per_behavior),
        "full_prompt_counts": {
            behavior: int(full_sizes[behavior]) for behavior in BEHAVIORS
        },
        "data_files": {
            behavior: {
                "path": str(data_paths[behavior]),
                "sha256": sha256_file(data_paths[behavior]),
            }
            for behavior in BEHAVIORS
        },
        "frozen_frontier_path": str(frontier_path),
        "frozen_frontier_sha256": (sha256_file(frontier_path)),
        "frozen_specs": (frozen_specs),
        "outputs": {
            "combined": str(combined_path),
            "summary": str(summary_path),
        },
    }

    write_json(
        output_root / "manifest.json",
        manifest,
    )

    # ----------------------------------------------------------------
    # Final report
    # ----------------------------------------------------------------

    print()
    print("=" * 100)
    print("OPEN-ENDED GENERATION COMPLETE")
    print("=" * 100)

    print(f"Responses: {len(combined)}")

    print()
    print(summary.to_string(index=False))

    print()
    print(f"Generations: {combined_path}")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()
