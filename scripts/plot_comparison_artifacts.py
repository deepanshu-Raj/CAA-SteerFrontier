"""Render per-model summary figures for frozen steering and capability results.

The figures compare selected depths, steering budgets, held-out survival, and
validation-to-test agreement without performing new selection.
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

from steerfrontier.config import load_experiment_config
from steerfrontier.plot_style import (
    BEHAVIOR_BLUE,
    CAPABILITY_ORANGE,
    OTHER_GRAY,
    SELECTED_PURPLE,
    ZERO_GRAY,
    configure_matplotlib,
    get_heatmap_cmap,
    pretty_behavior,
    save_figure,
)

# =============================================================================
# Constants
# =============================================================================

BENCHMARK_LABELS = {
    "mmlu": "MMLU",
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
}


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-model steering comparison / summary "
            "artifacts for CAA-SteerFrontier."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("plots/per_model"),
    )

    parser.add_argument(
        "--heatmap-cmap",
        choices=[
            "diverging",
            "sequential",
        ],
        default="diverging",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def require_columns(
    frame: pd.DataFrame,
    required,
    *,
    source: Path,
):
    missing = set(required) - set(frame.columns)

    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")


def as_bool(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    result = normalized.map(mapping)

    if result.isna().any():
        bad = sorted(normalized[result.isna()].unique())

        raise ValueError(f"Could not parse boolean values: {bad}")

    return result.astype(bool)


def load_frontiers(
    results_root: Path,
) -> pd.DataFrame:
    path = results_root / "steering_frontiers.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    require_columns(
        frame,
        [
            "behavior",
            "layer",
            "epsilon",
            "direction",
            "alpha_star",
            "eta_star",
            "behavior_gain",
            "limiting_benchmark",
            "nonzero_feasible",
        ],
        source=path,
    )

    return frame


def load_final_results(
    results_root: Path,
) -> pd.DataFrame:
    path = results_root / "final_evaluation" / "frontier_test_results.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    require_columns(
        frame,
        [
            "behavior",
            "layer",
            "epsilon",
            "direction",
            "alpha_star",
            "eta_star",
            "selection_behavior_gain",
            "test_directional_gain",
            "test_direction_supported",
            "capability_holdout_pass",
            "worst_holdout_lcb",
            "limiting_holdout_benchmark",
        ],
        source=path,
    )

    frame["test_direction_supported"] = as_bool(frame["test_direction_supported"])

    frame["capability_holdout_pass"] = as_bool(frame["capability_holdout_pass"])

    frame["joint_pass"] = (
        frame["test_direction_supported"] & frame["capability_holdout_pass"]
    )

    return frame


def selected_layer_for_behavior(
    frontiers: pd.DataFrame,
    behavior: str,
) -> int:
    values = sorted(
        set(
            frontiers.loc[
                frontiers["behavior"] == behavior,
                "layer",
            ].astype(int)
        )
    )

    if len(values) != 1:
        raise ValueError(
            f"{behavior}: expected exactly one selected layer, found {values}"
        )

    return int(values[0])


def fractional_depth_for_behavior(
    *,
    results_root: Path,
    behavior: str,
    selected_layer: int,
) -> float:
    path = results_root / behavior / "layer_sweep.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    require_columns(
        frame,
        [
            "layer",
            "fractional_depth",
        ],
        source=path,
    )

    rows = frame.loc[frame["layer"].astype(int) == int(selected_layer)]

    if rows.empty:
        raise ValueError(
            f"{behavior}: selected layer {selected_layer} is absent from {path}"
        )

    values = rows["fractional_depth"].astype(float).drop_duplicates().to_numpy()

    if len(values) != 1:
        raise ValueError(
            f"{behavior}: selected layer "
            f"{selected_layer} has multiple "
            "fractional depths."
        )

    return float(values[0])


# =============================================================================
# Plot 1: selected intervention depth
# =============================================================================


def plot_selected_intervention_depth(
    *,
    frontiers: pd.DataFrame,
    behavior_order,
    model_label: str,
    model_slug: str,
    results_root: Path,
    output_dir: Path,
    dpi: int,
):
    rows = []

    for behavior in behavior_order:
        layer = selected_layer_for_behavior(
            frontiers,
            behavior,
        )

        depth = fractional_depth_for_behavior(
            results_root=results_root,
            behavior=behavior,
            selected_layer=layer,
        )

        rows.append(
            {
                "behavior": behavior,
                "layer": layer,
                "fractional_depth": depth,
            }
        )

    frame = pd.DataFrame(rows)

    frame["behavior"] = pd.Categorical(
        frame["behavior"],
        categories=behavior_order,
        ordered=True,
    )

    frame = frame.sort_values("behavior").reset_index(drop=True)

    spacing = 0.25
    x = np.arange(len(frame)) * spacing

    depths = frame["fractional_depth"].astype(float).to_numpy()

    layers = frame["layer"].astype(int).to_numpy()

    min_depth = float(np.min(depths))

    max_depth = float(np.max(depths))

    min_layer = int(np.min(layers))

    max_layer = int(np.max(layers))

    fig, ax = plt.subplots(
        figsize=(12.0, 6.0),
        constrained_layout=True,
    )

    # ------------------------------------------------------------
    # Selected-layer range across behaviors
    # ------------------------------------------------------------

    ax.axhspan(
        min_depth,
        max_depth,
        xmin=0.0,
        xmax=1.0,
        facecolor=SELECTED_PURPLE,
        edgecolor=SELECTED_PURPLE,
        linewidth=1.5,
        alpha=0.09,
        zorder=0,
    )

    # ------------------------------------------------------------
    # Per-behavior depth stems
    # ------------------------------------------------------------

    ax.vlines(
        x=x,
        ymin=0.0,
        ymax=depths,
        color=OTHER_GRAY,
        linewidth=1.8,
        alpha=0.75,
        zorder=1,
    )

    ax.scatter(
        x,
        depths,
        s=85,
        color=SELECTED_PURPLE,
        zorder=3,
    )

    for xi, layer, depth in zip(
        x,
        layers,
        depths,
    ):
        ax.annotate(
            rf"$\ell_b^*={layer}$",
            xy=(
                xi,
                depth,
            ),
            xytext=(
                0,
                9,
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.0,
        )

    # ------------------------------------------------------------
    # Range annotation
    # ------------------------------------------------------------

    range_label = (
        rf"$\ell_b^*\in[{min_layer},{max_layer}]$"
        "\n"
        rf"$d_b^*\in[{min_depth:.2f},{max_depth:.2f}]$"
    )

    ax.text(
        0.985,
        max_depth + 0.035,
        range_label,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=SELECTED_PURPLE,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": SELECTED_PURPLE,
            "alpha": 0.90,
        },
    )

    # ------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        [pretty_behavior(str(behavior)) for behavior in frame["behavior"]],
        rotation=0,
        ha="center",
        fontsize=9.2,
    )

    ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel("Behavior")

    ax.set_ylabel(
        r"Relative depth, "
        r"$d_b^*=\ell_b^*/(L-1)$"
    )

    ax.set_title(
        r"Selected intervention depths "
        r"$d_b^*$"
        f" — {model_label}"
    )

    ax.grid(
        axis="y",
        alpha=0.4,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    return save_figure(
        fig,
        output_dir / (f"selected_intervention_depth_{model_slug}"),
        dpi=dpi,
    )


# =============================================================================
# Plot 2: frozen steering budgets
# =============================================================================


def plot_frozen_frontier_heatmap(
    *,
    frontiers: pd.DataFrame,
    behavior_order,
    model_label: str,
    model_slug: str,
    output_dir: Path,
    dpi: int,
    heatmap_cmap: str,
):
    epsilons = sorted(frontiers["epsilon"].astype(float).unique())

    signed_epsilons = sorted(
        [-float(epsilon) for epsilon in epsilons]
        + [float(epsilon) for epsilon in epsilons]
    )

    eta_matrix = np.full(
        (
            len(behavior_order),
            len(signed_epsilons),
        ),
        np.nan,
        dtype=float,
    )

    alpha_matrix = np.full_like(
        eta_matrix,
        np.nan,
    )

    for i, behavior in enumerate(behavior_order):
        for j, signed_epsilon in enumerate(signed_epsilons):
            epsilon = abs(float(signed_epsilon))

            direction = "negative" if signed_epsilon < 0 else "positive"

            rows = frontiers.loc[
                (frontiers["behavior"] == behavior)
                & np.isclose(
                    frontiers["epsilon"].astype(float),
                    epsilon,
                )
                & (frontiers["direction"] == direction)
            ]

            if len(rows) != 1:
                raise ValueError(
                    f"{behavior}, epsilon={epsilon}, "
                    f"direction={direction}: expected "
                    f"one row, found {len(rows)}"
                )

            row = rows.iloc[0]

            eta_matrix[
                i,
                j,
            ] = float(row["eta_star"])

            alpha_matrix[
                i,
                j,
            ] = float(row["alpha_star"])

    cmap = get_heatmap_cmap(heatmap_cmap)

    if heatmap_cmap == "diverging":
        plotted_values = eta_matrix

        maximum = float(np.nanmax(np.abs(plotted_values)))

        maximum = max(
            maximum,
            1e-12,
        )

        norm = TwoSlopeNorm(
            vmin=-maximum,
            vcenter=0.0,
            vmax=maximum,
        )

        colorbar_label = (
            r"Signed steering budget, "
            r"$\eta_b^*(\epsilon,s)$"
        )

    else:
        plotted_values = np.abs(eta_matrix)

        maximum = float(np.nanmax(plotted_values))

        maximum = max(
            maximum,
            1e-12,
        )

        norm = Normalize(
            vmin=0.0,
            vmax=maximum,
        )

        colorbar_label = (
            r"Steering-budget magnitude, "
            r"$|\eta_b^*(\epsilon,s)|$"
        )

    x = np.asarray(
        signed_epsilons,
        dtype=float,
    )

    # The signed epsilon values are evenly spaced:
    # -0.05, -0.03, -0.01, +0.01, +0.03, +0.05.
    spacing = float(np.min(np.diff(x)))

    x_edges = np.concatenate(
        [
            [x[0] - spacing / 2.0],
            (x[:-1] + x[1:]) / 2.0,
            [x[-1] + spacing / 2.0],
        ]
    )

    y_edges = np.arange(
        len(behavior_order) + 1,
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(11.0, 6.4),
        constrained_layout=True,
    )

    image = ax.pcolormesh(
        x_edges,
        y_edges,
        plotted_values,
        cmap=cmap,
        norm=norm,
        shading="flat",
    )

    ax.grid(False)

    ax.axvline(
        0.0,
        color="black",
        linewidth=1.5,
        alpha=0.8,
    )

    ax.set_xticks(x)

    ax.set_xticklabels([f"{value:+.2f}" for value in x])

    ax.set_yticks(np.arange(len(behavior_order)) + 0.5)

    ax.set_yticklabels([pretty_behavior(behavior) for behavior in behavior_order])

    ax.invert_yaxis()

    for i in range(len(behavior_order)):
        for j, signed_epsilon in enumerate(signed_epsilons):
            eta = float(
                eta_matrix[
                    i,
                    j,
                ]
            )

            alpha = float(
                alpha_matrix[
                    i,
                    j,
                ]
            )

            ax.text(
                signed_epsilon,
                i + 0.5,
                (
                    rf"$\eta^*={eta:.2f}$"
                    "\n"
                    rf"$\alpha^*={alpha:g}$"
                ),
                ha="center",
                va="center",
                fontsize=8.6,
            )

    ax.set_xlabel(
        r"Signed capability tolerance - "
        r"$(s, \epsilon)$"
    )

    ax.set_ylabel("Behavior")

    ax.set_title(
        r"Frozen steering budgets "
        r"$\eta_b^*(\epsilon,s)$"
        f" — {model_label}"
    )

    ax.text(
        0.25,
        -0.13,
        r"$s=-1$: counter-steering",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
    )

    ax.text(
        0.75,
        -0.13,
        r"$s=+1$: in-direction steering",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.015,
    )

    colorbar.set_label(colorbar_label)

    return save_figure(
        fig,
        output_dir / (f"frozen_steering_budgets_{model_slug}"),
        dpi=dpi,
    )


# =============================================================================
# Plot 3: validation-test agreement
# =============================================================================


def plot_validation_test_agreement(
    *,
    final_results: pd.DataFrame,
    model_label: str,
    model_slug: str,
    output_dir: Path,
    dpi: int,
):
    frame = final_results.loc[final_results["alpha_star"].abs() > 1e-12].copy()

    physical_key = [
        "behavior",
        "layer",
        "direction",
        "alpha_star",
        "eta_star",
    ]

    frame = frame.drop_duplicates(physical_key).reset_index(drop=True)

    x = frame["selection_behavior_gain"].astype(float).to_numpy()

    y = frame["test_directional_gain"].astype(float).to_numpy()

    if len(frame) < 2:
        raise ValueError("Need at least two nonzero interventions for correlation.")

    pearson = float(
        pd.Series(x).corr(
            pd.Series(y),
            method="pearson",
        )
    )

    spearman = float(
        pd.Series(x).corr(
            pd.Series(y),
            method="spearman",
        )
    )

    fig, ax = plt.subplots(
        figsize=(7.1, 6.2),
        constrained_layout=True,
    )

    negative = frame.loc[frame["direction"] == "negative"]

    positive = frame.loc[frame["direction"] == "positive"]

    ax.scatter(
        negative["selection_behavior_gain"],
        negative["test_directional_gain"],
        color=BEHAVIOR_BLUE,
        marker="o",
        s=65,
        alpha=0.88,
        label=r"Counter-steering, $-$",
        zorder=3,
    )

    ax.scatter(
        positive["selection_behavior_gain"],
        positive["test_directional_gain"],
        color=CAPABILITY_ORANGE,
        marker="s",
        s=62,
        alpha=0.88,
        label=r"In-direction, $+$",
        zorder=3,
    )

    low = float(
        min(
            np.min(x),
            np.min(y),
            0.0,
        )
    )

    high = float(
        max(
            np.max(x),
            np.max(y),
            0.0,
        )
    )

    span = max(
        high - low,
        1e-6,
    )

    padding = 0.06 * span

    low -= padding
    high += padding

    ax.plot(
        [
            low,
            high,
        ],
        [
            low,
            high,
        ],
        color=ZERO_GRAY,
        linestyle="--",
        linewidth=1.4,
        alpha=0.75,
        label=r"$y=x$",
        zorder=1,
    )

    ax.set_xlim(
        low,
        high,
    )

    ax.set_ylim(
        low,
        high,
    )

    ax.set_xlabel(
        r"Validation directional gain, "
        r"$\Delta B_b^{\mathrm{val}}$"
    )

    ax.set_ylabel(
        r"Test directional gain, "
        r"$\Delta B_b^{\mathrm{test}}$"
    )

    ax.set_title(f"Validation–test agreement — {model_label}")

    ax.text(
        0.04,
        0.96,
        (
            rf"$r={pearson:.3f}$"
            "\n"
            rf"$\rho={spearman:.3f}$"
            "\n"
            rf"$N={len(frame)}$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": OTHER_GRAY,
            "alpha": 0.9,
        },
    )

    ax.legend(
        loc="lower right",
        frameon=False,
    )

    return save_figure(
        fig,
        output_dir / (f"validation_test_agreement_{model_slug}"),
        dpi=dpi,
    )


# =============================================================================
# Plot 4: held-out frontier survival
# =============================================================================


def plot_heldout_frontier_survival(
    *,
    final_results: pd.DataFrame,
    model_label: str,
    model_slug: str,
    output_dir: Path,
    dpi: int,
):
    frame = final_results.loc[final_results["alpha_star"].abs() > 1e-12].copy()

    rows = []

    for epsilon, group in frame.groupby(
        "epsilon",
        sort=True,
    ):
        n = int(len(group))

        if n == 0:
            continue

        behavior_supported = int(group["test_direction_supported"].sum())

        capability_pass = int(group["capability_holdout_pass"].sum())

        joint_pass = int(group["joint_pass"].sum())

        rows.append(
            {
                "epsilon": float(epsilon),
                "n": n,
                "behavior_rate": (behavior_supported / n),
                "capability_rate": (capability_pass / n),
                "joint_rate": (joint_pass / n),
                "behavior_count": (behavior_supported),
                "capability_count": (capability_pass),
                "joint_count": (joint_pass),
            }
        )

    summary = pd.DataFrame(rows).sort_values("epsilon")

    x = np.arange(len(summary))

    width = 0.23

    fig, ax = plt.subplots(
        figsize=(9.2, 5.8),
        constrained_layout=True,
    )

    common_bar_style = {
        "edgecolor": "black",
        "linewidth": 0.9,
        "alpha": 0.70,
    }

    behavior_bars = ax.bar(
        x - width,
        summary["behavior_rate"],
        width=width,
        color=BEHAVIOR_BLUE,
        label=r"Behavior support, $R_B$",
        **common_bar_style,
    )

    capability_bars = ax.bar(
        x,
        summary["capability_rate"],
        width=width,
        color=CAPABILITY_ORANGE,
        label=r"Capability holdout, $R_C$",
        **common_bar_style,
    )

    joint_bars = ax.bar(
        x + width,
        summary["joint_rate"],
        width=width,
        color=SELECTED_PURPLE,
        label=r"Joint survival, $R_J$",
        **common_bar_style,
    )

    ax.set_xticks(x)

    ax.set_xticklabels([rf"${epsilon:.2f}$" for epsilon in summary["epsilon"]])

    ax.set_ylim(0.0, 1.14)

    ax.set_ylabel(r"Survival rate, $R(\epsilon)$")

    ax.set_xlabel(r"Capability tolerance, $\epsilon$")

    ax.set_title(
        r"Held-out frontier survival "
        r"$R_B(\epsilon),\,R_C(\epsilon),\,R_J(\epsilon)$"
        f" — {model_label}"
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor=OTHER_GRAY,
        framealpha=0.94,
    )

    ax.grid(
        axis="y",
        alpha=0.4,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    for bars, count_column in [
        (
            behavior_bars,
            "behavior_count",
        ),
        (
            capability_bars,
            "capability_count",
        ),
        (
            joint_bars,
            "joint_count",
        ),
    ]:
        for bar, count, n in zip(
            bars,
            summary[count_column],
            summary["n"],
        ):
            height = float(bar.get_height())

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.022,
                f"{int(count)}/{int(n)}",
                ha="center",
                va="bottom",
                fontsize=9.5,
            )

    return save_figure(
        fig,
        output_dir / (f"heldout_frontier_survival_{model_slug}"),
        dpi=dpi,
    )


# =============================================================================
# Plot 5: held-out limiting benchmark
# =============================================================================


def plot_limiting_capability_benchmark(
    *,
    final_results: pd.DataFrame,
    model_label: str,
    model_slug: str,
    output_dir: Path,
    dpi: int,
):
    frame = final_results.loc[final_results["alpha_star"].abs() > 1e-12].copy()

    benchmark_order = [
        "mmlu",
        "arc_challenge",
        "hellaswag",
    ]

    epsilons = sorted(frame["epsilon"].astype(float).unique())

    counts = np.zeros(
        (
            len(benchmark_order),
            len(epsilons),
        ),
        dtype=float,
    )

    for i, benchmark in enumerate(benchmark_order):
        for j, epsilon in enumerate(epsilons):
            counts[
                i,
                j,
            ] = float(
                (
                    (frame["limiting_holdout_benchmark"] == benchmark)
                    & np.isclose(
                        frame["epsilon"].astype(float),
                        epsilon,
                    )
                ).sum()
            )

    x = np.arange(len(benchmark_order))

    width = 0.72 / len(epsilons)

    fig, ax = plt.subplots(
        figsize=(8.7, 5.7),
        constrained_layout=True,
    )

    epsilon_colors = [
        BEHAVIOR_BLUE,
        CAPABILITY_ORANGE,
        SELECTED_PURPLE,
    ]

    for j, epsilon in enumerate(epsilons):
        offset = (j - (len(epsilons) - 1) / 2.0) * width

        bars = ax.bar(
            x + offset,
            counts[
                :,
                j,
            ],
            width=width,
            color=epsilon_colors[j % len(epsilon_colors)],
            edgecolor="black",
            linewidth=0.9,
            alpha=0.70,
            label=(rf"$\epsilon={epsilon:.2f}$"),
        )

        for bar in bars:
            height = float(bar.get_height())

            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.08,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                )

    ax.set_xticks(x)

    ax.set_xticklabels([BENCHMARK_LABELS[benchmark] for benchmark in benchmark_order])

    ax.set_ylabel(
        r"Frozen frontier claims, "
        r"$N_{k^*}(\epsilon)$"
    )

    ax.set_xlabel(r"Limiting capability benchmark, $k^*$")

    ax.set_title(
        r"Limiting benchmark counts "
        r"$N_{k^*}(\epsilon)$"
        f" — {model_label}"
    )

    ax.legend(
        loc="upper left",
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.4,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    return save_figure(
        fig,
        output_dir / (f"limiting_capability_benchmark_{model_slug}"),
        dpi=dpi,
    )


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    configure_matplotlib()

    config = load_experiment_config(args.config)

    results_root = Path(config.results_root)

    behavior_order = list(config.behaviors)

    frontiers = load_frontiers(results_root)

    final_results = load_final_results(results_root)

    output_dir = args.output_root / config.model_slug / "summary"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)

    print("PER-MODEL STEERING SUMMARY ARTIFACTS")

    print("=" * 76)

    print(f"Model:        {config.model_label}")

    print(f"Model slug:   {config.model_slug}")

    print(f"Results root: {results_root}")

    print(f"Output root:  {output_dir}")

    print(f"Heatmap cmap: {args.heatmap_cmap}")

    outputs = []

    outputs.extend(
        plot_selected_intervention_depth(
            frontiers=frontiers,
            behavior_order=behavior_order,
            model_label=config.model_label,
            model_slug=config.model_slug,
            results_root=results_root,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    outputs.extend(
        plot_frozen_frontier_heatmap(
            frontiers=frontiers,
            behavior_order=behavior_order,
            model_label=config.model_label,
            model_slug=config.model_slug,
            output_dir=output_dir,
            dpi=args.dpi,
            heatmap_cmap=args.heatmap_cmap,
        )
    )

    outputs.extend(
        plot_validation_test_agreement(
            final_results=final_results,
            model_label=config.model_label,
            model_slug=config.model_slug,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    outputs.extend(
        plot_heldout_frontier_survival(
            final_results=final_results,
            model_label=config.model_label,
            model_slug=config.model_slug,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    outputs.extend(
        plot_limiting_capability_benchmark(
            final_results=final_results,
            model_label=config.model_label,
            model_slug=config.model_slug,
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    print()
    print("Generated:")

    for path in outputs:
        print(f"  {path}")

    print()
    print(f"Artifacts: {len(outputs)} files (5 figures × PNG/PDF)")


if __name__ == "__main__":
    main()
