"""Run label-shuffled CAA specificity controls across all configured models.

The launcher preserves fixed seed ranges and resumes already-completed control
evaluations without changing frozen intervention points.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
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

NONZERO_TOL = 1e-12


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run label-shuffled CAA specificity controls "
            "sequentially across all four models."
        )
    )

    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs"),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260917,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )

    return parser.parse_args()


def discover_configs(
    configs_dir: Path,
):
    discovered = {}

    candidates = sorted(
        list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    )

    if not candidates:
        raise FileNotFoundError(f"No YAML configs found in {configs_dir}")

    for path in candidates:
        try:
            config = load_experiment_config(str(path))
        except Exception:
            continue

        slug = getattr(
            config,
            "model_slug",
            None,
        )

        if slug in MODEL_ORDER:
            if slug in discovered:
                raise ValueError(
                    f"Multiple configs found for {slug}:\n"
                    f"  {discovered[slug]}\n"
                    f"  {path}"
                )

            discovered[slug] = path

    missing = [slug for slug in MODEL_ORDER if slug not in discovered]

    if missing:
        raise ValueError(f"Missing experiment configs for: {missing}")

    return discovered


def load_unique_operating_points(
    final_path: Path,
):
    if not final_path.is_file():
        raise FileNotFoundError(final_path)

    frame = pd.read_csv(final_path)

    required = {
        "behavior",
        "layer",
        "direction",
        "alpha_star",
        "eta_star",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"{final_path} missing columns: {sorted(missing)}")

    frame = frame.loc[frame["alpha_star"].abs() > NONZERO_TOL].copy()

    key = [
        "behavior",
        "layer",
        "direction",
        "alpha_star",
        "eta_star",
    ]

    return frame.drop_duplicates(key).reset_index(drop=True)


def completed_rows_for_seed_range(
    results_path: Path,
    *,
    seed_start: int,
    num_seeds: int,
):
    if not results_path.is_file():
        return 0

    frame = pd.read_csv(results_path)

    if frame.empty:
        return 0

    required = {
        "control_seed_index",
        "behavior",
        "layer",
        "direction",
        "alpha_star",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"{results_path} missing columns: {sorted(missing)}")

    seed_end = seed_start + num_seeds

    frame = frame.loc[
        (frame["control_seed_index"] >= seed_start)
        & (frame["control_seed_index"] < seed_end)
    ].copy()

    key = [
        "control_seed_index",
        "behavior",
        "layer",
        "direction",
        "alpha_star",
    ]

    return int(len(frame.drop_duplicates(key)))


def format_duration(
    seconds: float,
):
    seconds = int(round(seconds))

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"

    return f"{minutes:d}m {seconds:02d}s"


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1")

    configs = discover_configs(args.configs_dir)

    script_path = Path("scripts/run_label_shuffled_controls.py")

    if not script_path.is_file():
        raise FileNotFoundError(script_path)

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "CAA-STEERFRONTIER SEQUENTIAL LABEL-SHUFFLED CONTROLS",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Seeds:       {args.seed_start} .. {args.seed_start + args.num_seeds - 1}",
        flush=True,
    )

    print(
        f"Num seeds:   {args.num_seeds}",
        flush=True,
    )

    print(
        f"Bootstrap B: {args.bootstrap_samples}",
        flush=True,
    )

    print(
        f"Device:      {args.device}",
        flush=True,
    )

    print(
        "\nConfigs:",
        flush=True,
    )

    for slug in MODEL_ORDER:
        print(
            f"  {slug:28s} -> {configs[slug]}",
            flush=True,
        )

    overall_start = time.time()

    completed_models = []

    for model_index, slug in enumerate(
        MODEL_ORDER,
        start=1,
    ):
        config_path = configs[slug]

        config = load_experiment_config(str(config_path))

        final_path = (
            config.results_root / "final_evaluation" / "frontier_test_results.csv"
        )

        operating_points = load_unique_operating_points(final_path)

        n_operating_points = int(len(operating_points))

        expected = n_operating_points * args.num_seeds

        results_path = (
            config.results_root
            / "controls"
            / "label_shuffled"
            / "label_shuffled_results.csv"
        )

        completed = completed_rows_for_seed_range(
            results_path,
            seed_start=args.seed_start,
            num_seeds=args.num_seeds,
        )

        print()
        print(
            "=" * 80,
            flush=True,
        )

        print(
            f"[{model_index}/{len(MODEL_ORDER)}] {config.model_label}",
            flush=True,
        )

        print(
            "=" * 80,
            flush=True,
        )

        print(
            f"Operating points: {n_operating_points}",
            flush=True,
        )

        print(
            f"Expected evals:    {expected}",
            flush=True,
        )

        print(
            f"Already complete:  {completed}",
            flush=True,
        )

        print(
            f"Remaining:         {max(expected - completed, 0)}",
            flush=True,
        )

        if completed >= expected:
            print(
                "Status: COMPLETE — skipping model.",
                flush=True,
            )

            completed_models.append(slug)

            continue

        command = [
            sys.executable,
            "-u",
            str(script_path),
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--num-seeds",
            str(args.num_seeds),
            "--seed-start",
            str(args.seed_start),
            "--base-seed",
            str(args.base_seed),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
        ]

        print()
        print(
            "Launching:",
            flush=True,
        )

        print(
            " ".join(command),
            flush=True,
        )

        print(
            "\nResume remains ENABLED.",
            flush=True,
        )

        model_start = time.time()

        result = subprocess.run(
            command,
            check=False,
        )

        elapsed = time.time() - model_start

        if result.returncode != 0:
            print()
            print(
                "!" * 80,
                flush=True,
            )

            print(
                f"MODEL FAILED / INTERRUPTED: {config.model_label}",
                flush=True,
            )

            print(
                f"Exit code: {result.returncode}",
                flush=True,
            )

            print(
                f"Elapsed:   {format_duration(elapsed)}",
                flush=True,
            )

            print(
                "Completed control rows and extraction caches remain on disk.",
                flush=True,
            )

            print(
                "Rerun this exact wrapper command to resume.",
                flush=True,
            )

            print(
                "!" * 80,
                flush=True,
            )

            sys.exit(result.returncode)

        completed_after = completed_rows_for_seed_range(
            results_path,
            seed_start=args.seed_start,
            num_seeds=args.num_seeds,
        )

        if completed_after < expected:
            raise RuntimeError(
                f"{slug}: child exited successfully "
                f"but only {completed_after}/{expected} "
                "expected evaluations were found."
            )

        print()
        print(
            f"Completed {config.model_label} in {format_duration(elapsed)}.",
            flush=True,
        )

        completed_models.append(slug)

    overall_elapsed = time.time() - overall_start

    print()
    print(
        "=" * 80,
        flush=True,
    )

    print(
        "ALL REQUESTED LABEL-SHUFFLED CONTROLS COMPLETE",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Models complete: {len(completed_models)}/{len(MODEL_ORDER)}",
        flush=True,
    )

    print(
        f"Elapsed:         {format_duration(overall_elapsed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
