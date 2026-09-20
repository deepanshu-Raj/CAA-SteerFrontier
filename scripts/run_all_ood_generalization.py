"""Run the frozen OOD/generalization evaluation across all configured models.

Completed per-model evaluations are reused so the launcher can safely resume
without altering the protocol.
"""

from __future__ import annotations

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

EXPECTED_RESULTS_PER_MODEL = 10


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


def completed_count(
    model_slug,
):
    path = Path("results") / model_slug / "ood_generalization" / "ood_results.csv"

    if not path.is_file():
        return 0

    frame = pd.read_csv(path)

    required = {
        "experiment",
        "source_domain",
        "test_domain",
        "direction",
    }

    if not required.issubset(frame.columns):
        return 0

    return int(
        frame[
            [
                "experiment",
                "source_domain",
                "test_domain",
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
    configs = discover_configs()

    print("=" * 92)
    print("ALL-MODEL OOD / GENERALIZATION")
    print("=" * 92)

    for index, slug in enumerate(
        MODEL_ORDER,
        start=1,
    ):
        done = completed_count(slug)

        print()
        print("=" * 92)
        print(f"[{index}/{len(MODEL_ORDER)}] {slug}")
        print("=" * 92)
        print(f"Already complete: {done}/{EXPECTED_RESULTS_PER_MODEL}")

        if done >= EXPECTED_RESULTS_PER_MODEL:
            print("Skipping completed model.")
            continue

        command = [
            sys.executable,
            "-u",
            "scripts/run_ood_generalization.py",
            "--config",
            str(configs[slug]),
        ]

        print()
        print("Running:")
        print(" ".join(command))
        print()

        subprocess.run(
            command,
            check=True,
        )

        finished = completed_count(slug)

        if finished != EXPECTED_RESULTS_PER_MODEL:
            raise RuntimeError(
                f"{slug}: expected {EXPECTED_RESULTS_PER_MODEL}, found {finished}."
            )

        print()
        print(f"{slug}: {finished}/{EXPECTED_RESULTS_PER_MODEL} complete.")

    print()
    print("=" * 92)
    print("ALL OOD / GENERALIZATION RUNS COMPLETE")
    print("=" * 92)

    for slug in MODEL_ORDER:
        print(f"{slug:28s} {completed_count(slug)}/{EXPECTED_RESULTS_PER_MODEL}")


if __name__ == "__main__":
    main()
