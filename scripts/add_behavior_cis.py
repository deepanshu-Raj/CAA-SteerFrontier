"""Add paired-bootstrap confidence intervals to behavior sweep artifacts.

Resampling uses deterministic condition-level seeds and preserves the existing
experimental outputs apart from the added uncertainty columns.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from steerfrontier.statistics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_mean_ci,
)


def stable_seed(
    base_seed: int,
    behavior: str,
    layer: int,
    alpha: float,
) -> int:
    """
    Derive a deterministic seed for one experimental condition.
    """

    payload = (f"{base_seed}:{behavior}:{layer}:{alpha:.12g}").encode()

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
    )


def add_bootstrap_cis(
    summary_path: Path,
    items_path: Path,
    n_bootstrap: int,
    base_seed: int,
):
    summary = pd.read_csv(summary_path)

    items = pd.read_csv(items_path)

    required_summary = {
        "behavior",
        "layer",
        "alpha",
        "delta_from_baseline",
    }

    required_items = {
        "behavior",
        "layer",
        "alpha",
        "item_index",
        "delta_from_baseline",
    }

    missing_summary = required_summary - set(summary.columns)

    missing_items = required_items - set(items.columns)

    if missing_summary:
        raise ValueError(f"{summary_path} missing columns: {sorted(missing_summary)}")

    if missing_items:
        raise ValueError(f"{items_path} missing columns: {sorted(missing_items)}")

    statistics = []

    grouped = items.groupby(
        [
            "behavior",
            "layer",
            "alpha",
        ],
        sort=True,
    )

    for (
        behavior,
        layer,
        alpha,
    ), group in grouped:
        behavior = str(behavior)
        layer = int(layer)
        alpha = float(alpha)

        seed = stable_seed(
            base_seed=base_seed,
            behavior=behavior,
            layer=layer,
            alpha=alpha,
        )

        mean, ci_low, ci_high = bootstrap_mean_ci(
            group["delta_from_baseline"].to_numpy(),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

        statistics.append(
            {
                "behavior": behavior,
                "layer": layer,
                "alpha": alpha,
                "bootstrap_delta_mean": mean,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "bootstrap_samples": (n_bootstrap),
            }
        )

    statistics = pd.DataFrame(statistics)

    merged = summary.merge(
        statistics,
        on=[
            "behavior",
            "layer",
            "alpha",
        ],
        how="left",
        validate="one_to_one",
    )

    if merged["delta_ci_low"].isna().any():
        raise RuntimeError(
            "Some summary rows did not receive bootstrap confidence intervals."
        )

    # Sanity check: aggregate mean delta and mean item-level
    # paired delta should agree up to numerical precision.
    difference = (merged["delta_from_baseline"] - merged["bootstrap_delta_mean"]).abs()

    if difference.max() > 1e-8:
        raise RuntimeError(
            "Aggregate and item-level mean deltas disagree. "
            f"Maximum difference: {difference.max()}"
        )

    merged.to_csv(
        summary_path,
        index=False,
    )

    return merged


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add paired-bootstrap confidence intervals to a behavior sweep CSV."
        )
    )

    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--items",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )

    args = parser.parse_args()

    items_path = args.items

    if items_path is None:
        items_path = args.summary.with_name(f"{args.summary.stem}_items.csv")

    if not args.summary.exists():
        raise FileNotFoundError(args.summary)

    if not items_path.exists():
        raise FileNotFoundError(items_path)

    result = add_bootstrap_cis(
        summary_path=args.summary,
        items_path=items_path,
        n_bootstrap=(args.bootstrap_samples),
        base_seed=args.seed,
    )

    print()
    print("=" * 76)
    print("PAIRED BOOTSTRAP COMPLETE")
    print("=" * 76)

    print(
        "Summary:",
        args.summary,
    )

    print(
        "Items:",
        items_path,
    )

    print(
        "Conditions:",
        len(result),
    )

    print(
        "Bootstrap samples:",
        args.bootstrap_samples,
    )

    print()

    columns = [
        "behavior",
        "layer",
        "alpha",
        "delta_from_baseline",
        "delta_ci_low",
        "delta_ci_high",
    ]

    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
