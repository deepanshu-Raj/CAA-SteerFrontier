"""Run WikiText-2 perplexity evaluation across all configured models.

The launcher resumes complete model outputs and preserves the frozen set of
steering operating points.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from steerfrontier.config import load_experiment_config

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


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run WikiText-2 perplexity secondary-capability "
            "evaluation across all four models."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
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
        raise RuntimeError(f"Could not discover configs for: {missing}")

    return configs


def completed_points(
    model_slug: str,
):
    path = (
        Path("results")
        / model_slug
        / "secondary_capability"
        / "perplexity"
        / "perplexity_summary.csv"
    )

    if not path.is_file():
        return 0

    frame = pd.read_csv(path)

    if frame.empty:
        return 0

    if not {
        "behavior",
        "direction",
    }.issubset(frame.columns):
        raise ValueError(f"Unexpected summary schema: {path}")

    return int(
        frame[
            [
                "behavior",
                "direction",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    configs = discover_configs()

    print("=" * 88)
    print("WIKITEXT-2 PERPLEXITY: ALL MODELS")
    print("=" * 88)
    print(f"epsilon:           {args.epsilon}")
    print(f"batch size:        {args.batch_size}")
    print(f"bootstrap samples: {args.bootstrap_samples}")
    print()

    for model_slug in MODEL_ORDER:
        config_path = configs[model_slug]

        already_done = completed_points(model_slug)

        print()
        print("=" * 88)
        print(model_slug)
        print("=" * 88)
        print(f"Config:    {config_path}")
        print(f"Completed: {already_done}/{EXPECTED_POINTS_PER_MODEL}")

        if already_done >= EXPECTED_POINTS_PER_MODEL and not args.no_resume:
            print("Already complete; skipping model.")
            continue

        command = [
            sys.executable,
            "-u",
            "scripts/run_perplexity_secondary.py",
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--epsilon",
            str(args.epsilon),
            "--batch-size",
            str(args.batch_size),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
        ]

        if args.no_resume:
            command.append("--no-resume")

        print()
        print("Running:")
        print(" ".join(command))
        print()

        subprocess.run(
            command,
            check=True,
        )

        finished = completed_points(model_slug)

        if finished != EXPECTED_POINTS_PER_MODEL:
            raise RuntimeError(
                f"{model_slug}: expected "
                f"{EXPECTED_POINTS_PER_MODEL} "
                f"completed points, found "
                f"{finished}."
            )

        print()
        print(f"{model_slug}: {finished}/{EXPECTED_POINTS_PER_MODEL} complete.")

    print()
    print("=" * 88)
    print("ALL PERPLEXITY RUNS COMPLETE")
    print("=" * 88)

    for model_slug in MODEL_ORDER:
        count = completed_points(model_slug)

        print(f"{model_slug:28s} {count}/{EXPECTED_POINTS_PER_MODEL}")


if __name__ == "__main__":
    main()
