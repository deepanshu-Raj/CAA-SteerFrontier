"""Run the frozen GSM8K secondary-capability evaluation across all models.

The launcher resumes completed model outputs and then builds the cross-model
summary from the per-model result artifacts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from steerfrontier.config import (
    load_experiment_config,
)

# =============================================================================
# Configuration
# =============================================================================

MODEL_ORDER = [
    "llama2-7b-chat",
    "llama3.1-8b-instruct",
    "mistral-7b-instruct-v0.3",
    "qwen2.5-7b-instruct",
]

EXPECTED_POINTS_PER_MODEL = 14

EXPECTED_TOTAL_POINTS = EXPECTED_POINTS_PER_MODEL * len(MODEL_ORDER)

DEFAULT_BATCH_SIZE = 4


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen epsilon=0.03 "
            "64-example 8-shot GSM8K "
            "secondary capability evaluation "
            "for all four models."
        )
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/secondary/gsm8k_test_64.json"),
    )

    parser.add_argument(
        "--demos-path",
        type=Path,
        default=Path("data/secondary/gsm8k_fewshot_8.json"),
    )

    return parser.parse_args()


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


def summary_path_for(
    config_path: Path,
):
    config = load_experiment_config(str(config_path))

    return config.results_root / "secondary_capability" / "gsm8k" / "gsm8k_summary.csv"


def completed_count(
    config_path: Path,
):
    path = summary_path_for(config_path)

    if not path.is_file():
        return 0

    return len(pd.read_csv(path))


def aggregate_results(
    configs,
):
    output_root = Path("results") / "cross_model" / "secondary_capability" / "gsm8k"

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = []

    for slug in MODEL_ORDER:
        path = summary_path_for(configs[slug])

        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)

        if len(frame) != EXPECTED_POINTS_PER_MODEL:
            raise RuntimeError(
                f"{slug}: expected "
                f"{EXPECTED_POINTS_PER_MODEL} "
                f"rows, found "
                f"{len(frame)}."
            )

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "model_slug",
            "behavior",
            "direction",
        ]
    ).reset_index(drop=True)

    combined_path = output_root / "gsm8k_summary.csv"

    combined.to_csv(
        combined_path,
        index=False,
    )

    model_summary = combined.groupby(
        [
            "model_slug",
            "model_label",
        ],
        as_index=False,
    ).agg(
        num_operating_points=(
            "delta_accuracy_flexible",
            "size",
        ),
        baseline_accuracy_flexible=(
            "baseline_accuracy_flexible",
            "first",
        ),
        mean_accuracy_flexible=(
            "accuracy_flexible",
            "mean",
        ),
        mean_delta_accuracy_flexible=(
            "delta_accuracy_flexible",
            "mean",
        ),
        median_delta_accuracy_flexible=(
            "delta_accuracy_flexible",
            "median",
        ),
        worst_delta_accuracy_flexible=(
            "delta_accuracy_flexible",
            "min",
        ),
        best_delta_accuracy_flexible=(
            "delta_accuracy_flexible",
            "max",
        ),
        baseline_accuracy_strict=(
            "baseline_accuracy_strict",
            "first",
        ),
        mean_accuracy_strict=(
            "accuracy_strict",
            "mean",
        ),
        mean_delta_accuracy_strict=(
            "delta_accuracy_strict",
            "mean",
        ),
        mean_parse_rate_flexible=(
            "parse_rate_flexible",
            "mean",
        ),
        mean_marker_rate=(
            "marker_rate",
            "mean",
        ),
        mean_truncation_rate=(
            "truncation_rate",
            "mean",
        ),
        mean_generated_tokens=(
            "mean_generated_tokens",
            "mean",
        ),
    )

    model_summary_path = output_root / "gsm8k_model_summary.csv"

    model_summary.to_csv(
        model_summary_path,
        index=False,
    )

    return (
        combined_path,
        model_summary_path,
        combined,
    )


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    configs = discover_configs()

    print("=" * 80)
    print("ALL-MODEL GSM8K SECONDARY CAPABILITY")
    print("=" * 80)
    print(f"Epsilon:         {args.epsilon}")
    print("Examples/model:  64")
    print("Few-shot:        8")
    print(f"Points/model:    {EXPECTED_POINTS_PER_MODEL}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Max new tokens:  {args.max_new_tokens}")
    print(f"Expected total:  {EXPECTED_TOTAL_POINTS}")

    for index, slug in enumerate(
        MODEL_ORDER,
        start=1,
    ):
        config_path = configs[slug]

        before = completed_count(config_path)

        print()
        print("=" * 80)
        print(f"[{index}/{len(MODEL_ORDER)}] {slug}")
        print("=" * 80)
        print(f"Config:            {config_path}")
        print(f"Already completed: {before}/{EXPECTED_POINTS_PER_MODEL}")

        command = [
            sys.executable,
            "-u",
            "scripts/run_gsm8k_secondary.py",
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--epsilon",
            str(args.epsilon),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--batch-size",
            str(args.batch_size),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--data-path",
            str(args.data_path),
            "--demos-path",
            str(args.demos_path),
        ]

        subprocess.run(
            command,
            check=True,
        )

        after = completed_count(config_path)

        print()
        print(f"{slug}: {after}/{EXPECTED_POINTS_PER_MODEL} complete")

        if after != EXPECTED_POINTS_PER_MODEL:
            raise RuntimeError(
                f"{slug}: expected "
                f"{EXPECTED_POINTS_PER_MODEL} "
                f"completed points, "
                f"found {after}."
            )

    (
        combined_path,
        model_summary_path,
        combined,
    ) = aggregate_results(configs)

    print()
    print("=" * 80)
    print("ALL GSM8K EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Combined rows: {len(combined)}")
    print(f"Expected:      {EXPECTED_TOTAL_POINTS}")
    print(f"Combined:      {combined_path}")
    print(f"Model summary: {model_summary_path}")

    if len(combined) != EXPECTED_TOTAL_POINTS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_POINTS} rows, found {len(combined)}."
        )


if __name__ == "__main__":
    main()
