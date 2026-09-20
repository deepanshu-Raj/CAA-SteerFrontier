"""Run the fixed-strength causal-transfer protocol sequentially across all models.

Per-model outputs are resumable and aggregated only after each model's frozen
transfer conditions are evaluated.
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


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-eta ordered causal-transfer experiments "
            "sequentially for all four models."
        )
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260918,
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
        raise RuntimeError(f"Could not find configs for: {missing}")

    return configs


def completed_count(
    config_path: Path,
):
    config = load_experiment_config(str(config_path))

    path = config.results_root / "causal_transfer" / "transfer_results.csv"

    if not path.is_file():
        return 0

    return len(pd.read_csv(path))


def aggregate_results(
    configs,
):
    output_root = Path("results") / "cross_model" / "causal_transfer"

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_frames = []
    summary_frames = []

    for slug in MODEL_ORDER:
        config = load_experiment_config(str(configs[slug]))

        root = config.results_root / "causal_transfer"

        results_path = root / "transfer_results.csv"

        summary_path = root / "transfer_layer_summary.csv"

        if results_path.is_file():
            result_frames.append(pd.read_csv(results_path))

        if summary_path.is_file():
            summary_frames.append(pd.read_csv(summary_path))

    if result_frames:
        combined = pd.concat(
            result_frames,
            ignore_index=True,
        )

        combined = combined.sort_values(
            [
                "model_slug",
                "layer",
                "source_behavior",
                "target_behavior",
            ]
        ).reset_index(drop=True)

        combined.to_csv(
            output_root / "transfer_results.csv",
            index=False,
        )

    if summary_frames:
        combined_summary = pd.concat(
            summary_frames,
            ignore_index=True,
        )

        combined_summary = combined_summary.sort_values(
            [
                "model_slug",
                "requested_fractional_depth",
            ]
        ).reset_index(drop=True)

        combined_summary.to_csv(
            output_root / "transfer_layer_summary.csv",
            index=False,
        )

    return output_root


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    configs = discover_configs()

    print("=" * 76)
    print("ALL-MODEL CAUSAL TRANSFER")
    print("=" * 76)
    print(f"Frozen eta: {args.eta:.4f}")
    print("Expected/model: 294")
    print(f"Expected total: {294 * len(MODEL_ORDER)}")

    for index, slug in enumerate(
        MODEL_ORDER,
        start=1,
    ):
        config_path = configs[slug]

        before = completed_count(config_path)

        print()
        print("=" * 76)
        print(f"[{index}/{len(MODEL_ORDER)}] {slug}")
        print("=" * 76)
        print(f"Config: {config_path}")
        print(f"Already completed: {before}/294")

        command = [
            sys.executable,
            "-u",
            "scripts/run_causal_transfer.py",
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--eta",
            str(args.eta),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--base-seed",
            str(args.base_seed),
        ]

        subprocess.run(
            command,
            check=True,
        )

        after = completed_count(config_path)

        print()
        print(f"{slug}: {after}/294 complete")

        if after != 294:
            raise RuntimeError(
                f"{slug}: expected 294 completed conditions, found {after}."
            )

    output_root = aggregate_results(configs)

    total = 0

    combined_path = output_root / "transfer_results.csv"

    if combined_path.is_file():
        combined = pd.read_csv(combined_path)
        total = len(combined)

    print()
    print("=" * 76)
    print("ALL CAUSAL TRANSFER COMPLETE")
    print("=" * 76)
    print(f"Combined conditions: {total}")
    print(f"Expected:            {294 * len(MODEL_ORDER)}")
    print(f"Combined output:     {output_root}")

    if total != (294 * len(MODEL_ORDER)):
        raise RuntimeError(f"Combined result count is {total}, expected 1176.")


if __name__ == "__main__":
    main()
