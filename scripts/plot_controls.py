"""Render cross-model specificity-control figures from frozen control summaries.

Paired bootstrap intervals are computed over frozen intervention points; the
plotting layer never changes the underlying control evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260919

MODEL_ORDER = [
    "llama2-7b-chat",
    "llama3.1-8b-instruct",
    "mistral-7b-instruct-v0.3",
    "qwen2.5-7b-instruct",
]

MODEL_LABELS = {
    "llama2-7b-chat": "Llama 2",
    "llama3.1-8b-instruct": "Llama 3.1",
    "mistral-7b-instruct-v0.3": "Mistral",
    "qwen2.5-7b-instruct": "Qwen 2.5",
}

RESULTS_ROOT = Path("results")
OUTPUT_ROOT = Path("plots") / "controls"

CAA_COLUMN = "caa_test_directional_gain"


@dataclass(frozen=True)
class ControlSpec:
    key: str
    relative_path: Path
    control_column: str
    q975_column: str
    title: str
    control_label: str
    output_name: str
    match_rate_column: str | None = None
    match_count_column: str | None = None


CONTROL_SPECS = [
    ControlSpec(
        key="norm_matched_random",
        relative_path=(
            Path("controls") / "norm_matched_random" / "random_control_summary.csv"
        ),
        control_column="random_gain_mean",
        q975_column="random_gain_q975",
        title="Norm-Matched Random Directions",
        control_label="Random",
        output_name="norm_matched_random",
    ),
    ControlSpec(
        key="label_shuffled",
        relative_path=(
            Path("controls") / "label_shuffled" / "label_shuffled_summary.csv"
        ),
        control_column="shuffled_gain_mean",
        q975_column="shuffled_gain_q975",
        title="Label-Shuffled CAA Directions",
        control_label="Shuffled",
        output_name="label_shuffled",
    ),
    ControlSpec(
        key="off_target_kl_matched",
        relative_path=(
            Path("controls") / "off_target_kl_matched" / "kl_matched_summary.csv"
        ),
        control_column="matched_gain_mean",
        q975_column="matched_gain_q975",
        title="Off-Target-KL-Matched Random Directions",
        control_label="KL-matched",
        output_name="off_target_kl_matched",
        match_rate_column="kl_match_success_rate",
        match_count_column="num_kl_matched_random_directions",
    ),
]


# =============================================================================
# Plot style
# =============================================================================


def configure_matplotlib() -> None:
    """
    Use a clean sans-serif figure style that pairs with the serif manuscript
    body while preserving mathematical notation in axes and annotations.
    """

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Avenir",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "dejavusans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# Loading and validation
# =============================================================================


def load_control_data(
    spec: ControlSpec,
) -> pd.DataFrame:
    frames = []

    for model_slug in MODEL_ORDER:
        path = RESULTS_ROOT / model_slug / spec.relative_path

        if not path.is_file():
            raise FileNotFoundError(f"Missing control summary:\n{path}")

        frame = pd.read_csv(path)

        required = {
            "model_slug",
            "behavior",
            "direction",
            "layer",
            "alpha_star",
            "eta_star",
            CAA_COLUMN,
            spec.control_column,
            spec.q975_column,
        }

        if spec.match_rate_column is not None:
            required.add(spec.match_rate_column)

        if spec.match_count_column is not None:
            required.add(spec.match_count_column)

        missing = required - set(frame.columns)

        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")

        observed_models = set(frame["model_slug"].astype(str))

        if observed_models != {model_slug}:
            raise RuntimeError(
                f"{path}: unexpected model_slug values {sorted(observed_models)}"
            )

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined.empty:
        raise RuntimeError(f"No data loaded for {spec.key}.")

    return combined


# =============================================================================
# Paired bootstrap
# =============================================================================


def bootstrap_mean_pair(
    frame: pd.DataFrame,
    *,
    control_column: str,
    seed: int,
) -> dict:
    """
    Resample frozen intervention points with replacement while keeping each
    CAA/control observation paired within every bootstrap draw.
    """

    caa = frame[CAA_COLUMN].to_numpy(dtype=float)

    control = frame[control_column].to_numpy(dtype=float)

    if len(caa) != len(control):
        raise RuntimeError("CAA/control length mismatch.")

    n = len(caa)

    if n < 1:
        raise ValueError("Cannot summarize an empty frame.")

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        n,
        size=(
            BOOTSTRAP_SAMPLES,
            n,
        ),
    )

    caa_boot = caa[indices].mean(axis=1)

    control_boot = control[indices].mean(axis=1)

    diff_boot = (caa[indices] - control[indices]).mean(axis=1)

    caa_low, caa_high = np.quantile(
        caa_boot,
        [
            0.025,
            0.975,
        ],
    )

    control_low, control_high = np.quantile(
        control_boot,
        [
            0.025,
            0.975,
        ],
    )

    diff_low, diff_high = np.quantile(
        diff_boot,
        [
            0.025,
            0.975,
        ],
    )

    return {
        "n": int(n),
        "caa_mean": float(caa.mean()),
        "caa_ci_low": float(caa_low),
        "caa_ci_high": float(caa_high),
        "control_mean": float(control.mean()),
        "control_ci_low": float(control_low),
        "control_ci_high": float(control_high),
        "difference_mean": float((caa - control).mean()),
        "difference_ci_low": float(diff_low),
        "difference_ci_high": float(diff_high),
    }


# =============================================================================
# Summary construction
# =============================================================================


def summarize_control(
    frame: pd.DataFrame,
    spec: ControlSpec,
) -> pd.DataFrame:
    rows = []

    groups = [
        (
            model_slug,
            frame[frame["model_slug"] == model_slug],
        )
        for model_slug in MODEL_ORDER
    ]

    groups.append(
        (
            "overall",
            frame,
        )
    )

    for group_index, (
        group_name,
        group_frame,
    ) in enumerate(groups):
        summary = bootstrap_mean_pair(
            group_frame,
            control_column=(spec.control_column),
            seed=(BOOTSTRAP_SEED + group_index),
        )

        q975_rate = float(
            (group_frame[CAA_COLUMN] > group_frame[spec.q975_column]).mean()
        )

        row = {
            "control": (spec.key),
            "group": (group_name),
            **summary,
            "caa_above_control_q975_rate": (q975_rate),
        }

        if spec.match_rate_column is not None:
            if spec.match_count_column is not None:
                weights = group_frame[spec.match_count_column].to_numpy(dtype=float)

                match_rates = group_frame[spec.match_rate_column].to_numpy(dtype=float)

                match_rate = float(
                    np.average(
                        match_rates,
                        weights=weights,
                    )
                )
            else:
                match_rate = float(group_frame[spec.match_rate_column].mean())

            row["kl_match_success_rate"] = match_rate

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Plotting helpers
# =============================================================================


def asymmetric_error(
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        [
            mean - low,
            high - mean,
        ]
    )


def format_plot_value(
    value: float,
) -> str:
    """
    Display means to three decimals while suppressing negative zero created
    purely by decimal rounding.
    """

    if abs(value) < 0.0005:
        value = 0.0

    return f"{value:.3f}"


def add_value_labels(
    ax,
    *,
    x_positions: np.ndarray,
    values: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    y_span: float,
) -> None:
    """
    Place each mean just beyond the outer end of its bootstrap confidence
    interval so labels remain clear of the error-bar stems and caps.
    """

    offset = 0.018 * y_span

    for (
        x_position,
        value,
        low,
        high,
    ) in zip(
        x_positions,
        values,
        ci_low,
        ci_high,
    ):
        if value >= 0:
            y = high + offset
            vertical_alignment = "bottom"
        else:
            y = low - offset
            vertical_alignment = "top"

        ax.text(
            x_position,
            y,
            format_plot_value(value),
            ha="center",
            va=vertical_alignment,
            fontsize=7.0,
            zorder=5,
        )


# =============================================================================
# Plotting
# =============================================================================


def plot_control(
    summary: pd.DataFrame,
    spec: ControlSpec,
) -> None:
    groups = MODEL_ORDER + ["overall"]

    ordered = summary.set_index("group").loc[groups].reset_index()

    labels = []

    for _, row in ordered.iterrows():
        group = row["group"]

        if group == "overall":
            name = "Overall"
        else:
            name = MODEL_LABELS[group]

        labels.append(f"{name}\n$n={int(row['n'])}$")

    x = np.arange(
        len(ordered),
        dtype=float,
    )

    width = 0.34

    caa_x = x - width / 2

    control_x = x + width / 2

    caa_mean = ordered["caa_mean"].to_numpy(dtype=float)

    caa_low = ordered["caa_ci_low"].to_numpy(dtype=float)

    caa_high = ordered["caa_ci_high"].to_numpy(dtype=float)

    control_mean = ordered["control_mean"].to_numpy(dtype=float)

    control_low = ordered["control_ci_low"].to_numpy(dtype=float)

    control_high = ordered["control_ci_high"].to_numpy(dtype=float)

    all_high = np.concatenate(
        [
            caa_high,
            control_high,
        ]
    )

    all_low = np.concatenate(
        [
            caa_low,
            control_low,
            np.asarray([0.0]),
        ]
    )

    y_min = float(np.min(all_low))

    y_max = float(np.max(all_high))

    y_span = max(
        0.05,
        y_max - y_min,
    )

    fig, ax = plt.subplots(
        figsize=(
            5.2,
            3.35,
        )
    )

    caa_color = "#4C72B0"
    control_color = "#B8B8B8"

    error_style = {
        "elinewidth": 0.9,
        "capsize": 2.5,
        "capthick": 0.9,
    }

    ax.bar(
        caa_x,
        caa_mean,
        width=width,
        label="CAA",
        color=caa_color,
        edgecolor="black",
        linewidth=0.5,
        yerr=asymmetric_error(
            caa_mean,
            caa_low,
            caa_high,
        ),
        error_kw=error_style,
        zorder=3,
    )

    ax.bar(
        control_x,
        control_mean,
        width=width,
        label=(spec.control_label),
        color=control_color,
        edgecolor="black",
        linewidth=0.5,
        yerr=asymmetric_error(
            control_mean,
            control_low,
            control_high,
        ),
        error_kw=error_style,
        zorder=3,
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.65,
        zorder=2,
    )

    add_value_labels(
        ax,
        x_positions=caa_x,
        values=caa_mean,
        ci_low=caa_low,
        ci_high=caa_high,
        y_span=y_span,
    )

    add_value_labels(
        ax,
        x_positions=control_x,
        values=control_mean,
        ci_low=control_low,
        ci_high=control_high,
        y_span=y_span,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    # g denotes intervention-level directional behavior gain.
    ax.set_ylabel(r"Mean directional gain $\mathbb{E}[g]$")

    ax.set_title(
        spec.title,
        pad=8,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.7,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="upper left",
        ncols=2,
        columnspacing=1.2,
        handlelength=1.2,
    )

    overall = ordered[ordered["group"] == "overall"].iloc[0]

    annotation = (
        r"CAA $>$ null $q_{.975}$: "
        f"{100.0 * overall['caa_above_control_q975_rate']:.1f}%"
    )

    if spec.key == "off_target_kl_matched":
        annotation += (
            "\n"
            r"KL match: "
            f"{100.0 * overall['kl_match_success_rate']:.1f}%"
        )

    ax.text(
        0.99,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        linespacing=1.25,
    )

    ax.set_ylim(
        y_min - 0.16 * y_span,
        y_max + 0.30 * y_span,
    )

    fig.tight_layout(pad=0.7)

    pdf_path = OUTPUT_ROOT / f"{spec.output_name}.pdf"

    png_path = OUTPUT_ROOT / f"{spec.output_name}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {pdf_path}")

    print(f"Saved: {png_path}")


# =============================================================================
# Console audit
# =============================================================================


def print_summary(
    summary: pd.DataFrame,
    spec: ControlSpec,
) -> None:
    columns = [
        "group",
        "n",
        "caa_mean",
        "control_mean",
        "difference_mean",
        "difference_ci_low",
        "difference_ci_high",
        "caa_above_control_q975_rate",
    ]

    if "kl_match_success_rate" in summary.columns:
        columns.append("kl_match_success_rate")

    print()
    print("=" * 110)
    print(spec.title.upper())
    print("=" * 110)

    print(summary[columns].to_string(index=False))


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    configure_matplotlib()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = []

    for spec in CONTROL_SPECS:
        frame = load_control_data(spec)

        summary = summarize_control(
            frame,
            spec,
        )

        all_summaries.append(summary)

        print_summary(
            summary,
            spec,
        )

        plot_control(
            summary,
            spec,
        )

    combined_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    summary_path = OUTPUT_ROOT / "control_plot_summary.csv"

    combined_summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("CONTROL PLOTTING COMPLETE")
    print("=" * 110)

    print(f"Summary: {summary_path}")

    print(f"Plots:   {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
