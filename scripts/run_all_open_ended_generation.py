"""Run frozen open-ended generation conditions across all configured models.

Generation uses the shared prompt set and fixed condition protocol, with safe
resume based on completed response counts.
"""

from __future__ import annotations

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

EXPECTED_RESPONSES = 216

# Frozen open-ended generation protocol.
MAX_NEW_TOKENS = 2056
BATCH_SIZE = 4


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
    model_slug: str,
):
    path = Path("results") / model_slug / "open_ended_generation" / "generations.csv"

    if not path.is_file():
        return 0

    try:
        frame = pd.read_csv(path)
    except Exception:
        return 0

    required = {
        "behavior",
        "condition",
        "prompt_id",
        "response",
    }

    if not required.issubset(frame.columns):
        return 0

    return int(
        frame[
            [
                "behavior",
                "condition",
                "prompt_id",
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

    print("=" * 100)
    print("ALL-MODEL OPEN-ENDED GENERATION")
    print("=" * 100)

    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Batch size:     {BATCH_SIZE}")

    for index, slug in enumerate(
        MODEL_ORDER,
        start=1,
    ):
        done = completed_count(slug)

        print()
        print("=" * 100)
        print(f"[{index}/{len(MODEL_ORDER)}] {slug}")
        print("=" * 100)

        print(f"Completed responses: {done}/{EXPECTED_RESPONSES}")

        if done >= EXPECTED_RESPONSES:
            print("Skipping completed model.")
            continue

        command = [
            sys.executable,
            "-u",
            "scripts/run_open_ended_generation.py",
            "--config",
            str(configs[slug]),
            "--batch-size",
            str(BATCH_SIZE),
            "--max-new-tokens",
            str(MAX_NEW_TOKENS),
        ]

        print()
        print("Running:")
        print(" ".join(command))
        print()

        subprocess.run(
            command,
            check=True,
        )

        final_count = completed_count(slug)

        if final_count != EXPECTED_RESPONSES:
            raise RuntimeError(
                f"{slug}: expected {EXPECTED_RESPONSES} responses, found {final_count}."
            )

        print()
        print(f"{slug}: {final_count}/{EXPECTED_RESPONSES} complete.")

    print()
    print("=" * 100)
    print("ALL OPEN-ENDED GENERATION COMPLETE")
    print("=" * 100)

    for slug in MODEL_ORDER:
        print(f"{slug:28s} {completed_count(slug)}/{EXPECTED_RESPONSES}")


if __name__ == "__main__":
    main()
