"""Evaluate norm-matched random directions as a steering-specificity control.

Random vectors match the CAA direction norm at each frozen operating point and
use deterministic seeds for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import torch

# Reuse the exact final-evaluation behavior path.
from run_final_evaluation import (
    behavior_statistics,
    evaluate_behavior_condition,
)

from steerfrontier.config import load_experiment_config
from steerfrontier.data import load_test_data
from steerfrontier.models import load_model
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.workflow import load_normalized_vectors

# =============================================================================
# Constants
# =============================================================================

CONTROL_NAME = "norm_matched_random"

DEFAULT_NUM_SEEDS = 20
DEFAULT_BASE_SEED = 20260917
DEFAULT_BOOTSTRAP_SAMPLES = 10_000

NONZERO_TOL = 1e-12


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate norm-matched random steering directions at the "
            "already-frozen CAA operating points."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Experiment YAML for one model.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help=(
            "Number of deterministic random directions per behavior. "
            "Proposal target is approximately 20."
        ),
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help=("First control-seed index. Useful for extending or splitting runs."),
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )

    parser.add_argument(
        "--behaviors",
        nargs="*",
        default=None,
        help=(
            "Optional subset of configured behaviors. "
            "Default: all configured behaviors."
        ),
    )

    parser.add_argument(
        "--frontiers",
        type=Path,
        default=None,
        help=(
            "Frozen steering_frontiers.csv. "
            "Default: <results_root>/steering_frontiers.csv"
        ),
    )

    parser.add_argument(
        "--final-results",
        type=Path,
        default=None,
        help=(
            "Final frontier_test_results.csv. "
            "Default: <results_root>/final_evaluation/"
            "frontier_test_results.csv"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=("Output directory. Default: <results_root>/controls/norm_matched_random"),
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Print the frozen operating points and planned evaluation count "
            "without loading the model."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing partial results CSV.",
    )

    return parser.parse_args()


# =============================================================================
# Deterministic seeds
# =============================================================================


def stable_seed(
    base_seed: int,
    *parts,
) -> int:
    """
    Deterministically map arbitrary experiment identifiers to a 32-bit seed.

    This makes each random vector independent of iteration order and allows
    exact resume / chunked execution.
    """

    payload = "|".join(
        [
            str(base_seed),
            *[str(part) for part in parts],
        ]
    )

    digest = hashlib.sha256(payload.encode("utf-8")).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )


# =============================================================================
# File helpers
# =============================================================================


def sha256_file(
    path: Path,
) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def write_csv(
    frame: pd.DataFrame,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
    )


def as_dataframe(
    value,
) -> pd.DataFrame:
    if isinstance(
        value,
        pd.DataFrame,
    ):
        return value.copy()

    if isinstance(
        value,
        pd.Series,
    ):
        return pd.DataFrame([value.to_dict()])

    return pd.DataFrame(value)


def as_dict(
    value: Any,
) -> Dict[str, Any]:
    if isinstance(
        value,
        Mapping,
    ):
        return dict(value)

    if isinstance(
        value,
        pd.Series,
    ):
        return value.to_dict()

    if isinstance(
        value,
        pd.DataFrame,
    ):
        if len(value) != 1:
            raise ValueError(
                f"Expected one-row statistics DataFrame, got {len(value)} rows."
            )

        return value.iloc[0].to_dict()

    if hasattr(
        value,
        "_asdict",
    ):
        return dict(value._asdict())

    raise TypeError(f"Unsupported statistics return type: {type(value).__name__}")


# =============================================================================
# Frozen operating points
# =============================================================================


def load_operating_points(
    frozen_path: Path,
    final_path: Path,
    behaviors,
) -> pd.DataFrame:
    """
    Audit frozen and final artifacts, then deduplicate repeated interventions.

    One physical intervention can appear under multiple epsilon values.
    We evaluate that intervention only once per random seed.
    """

    if not frozen_path.is_file():
        raise FileNotFoundError(frozen_path)

    if not final_path.is_file():
        raise FileNotFoundError(final_path)

    frozen = pd.read_csv(frozen_path)

    final = pd.read_csv(final_path)

    frozen_required = {
        "behavior",
        "layer",
        "epsilon",
        "direction",
        "alpha_star",
        "eta_star",
    }

    final_required = {
        "behavior",
        "layer",
        "epsilon",
        "direction",
        "alpha_star",
        "eta_star",
        "selection_behavior_gain",
        "test_behavior_baseline",
        "test_behavior_score",
        "test_behavior_delta",
        "test_behavior_delta_ci_low",
        "test_behavior_delta_ci_high",
        "test_directional_gain",
        "test_directional_gain_ci_low",
        "test_directional_gain_ci_high",
        "test_direction_supported",
    }

    missing = frozen_required - set(frozen.columns)

    if missing:
        raise ValueError(f"{frozen_path} missing columns: {sorted(missing)}")

    missing = final_required - set(final.columns)

    if missing:
        raise ValueError(f"{final_path} missing columns: {sorted(missing)}")

    frozen = frozen.loc[frozen["behavior"].isin(behaviors)].copy()

    final = final.loc[final["behavior"].isin(behaviors)].copy()

    key = [
        "behavior",
        "layer",
        "epsilon",
        "direction",
    ]

    audit = final.merge(
        frozen[
            key
            + [
                "alpha_star",
                "eta_star",
            ]
        ],
        on=key,
        how="outer",
        validate="one_to_one",
        suffixes=(
            "_final",
            "_frozen",
        ),
        indicator=True,
    )

    if not (audit["_merge"] == "both").all():
        raise ValueError(
            "Frozen frontier keys do not exactly match "
            "the final-evaluation frontier keys."
        )

    for column in (
        "alpha_star",
        "eta_star",
    ):
        left = audit[f"{column}_final"].to_numpy(dtype=float)

        right = audit[f"{column}_frozen"].to_numpy(dtype=float)

        if not np.allclose(
            left,
            right,
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError(f"Frozen/final mismatch for {column}.")

    final = final.loc[final["alpha_star"].abs() > NONZERO_TOL].copy()

    intervention_key = [
        "behavior",
        "layer",
        "direction",
        "alpha_star",
        "eta_star",
    ]

    # Verify repeated frontier claims really are the same physical
    # intervention and held-out CAA result.
    consistency_columns = [
        "selection_behavior_gain",
        "test_behavior_baseline",
        "test_behavior_score",
        "test_behavior_delta",
        "test_behavior_delta_ci_low",
        "test_behavior_delta_ci_high",
        "test_directional_gain",
        "test_directional_gain_ci_low",
        "test_directional_gain_ci_high",
        "test_direction_supported",
    ]

    for _, group in final.groupby(
        intervention_key,
        dropna=False,
    ):
        for column in consistency_columns:
            values = group[column]

            if pd.api.types.is_numeric_dtype(values):
                array = values.to_numpy(dtype=float)

                if not np.allclose(
                    array,
                    array[0],
                    atol=1e-10,
                    rtol=0.0,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"Repeated intervention has inconsistent {column}."
                    )

            elif values.nunique(dropna=False) != 1:
                raise ValueError(f"Repeated intervention has inconsistent {column}.")

    epsilon_map = (
        final.groupby(
            intervention_key,
            dropna=False,
        )["epsilon"]
        .apply(lambda x: ";".join(f"{float(v):.2f}" for v in sorted(set(x))))
        .rename("frontier_epsilons")
        .reset_index()
    )

    operating = (
        final.sort_values("epsilon")
        .drop_duplicates(
            intervention_key,
            keep="first",
        )
        .merge(
            epsilon_map,
            on=intervention_key,
            how="left",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    return operating


# =============================================================================
# Random control construction
# =============================================================================


def make_norm_matched_random_vector(
    learned_vector: torch.Tensor,
    *,
    seed: int,
):
    """
    Sample r ~ N(0, I) and rescale it so:

        ||r_tilde||_2 = ||v_tilde||_2

    No orthogonalization is performed: this is a true isotropic
    norm-matched random-direction null.
    """

    learned = learned_vector.detach().cpu().float().reshape(-1)

    learned_norm = float(torch.linalg.vector_norm(learned).item())

    if not np.isfinite(learned_norm) or learned_norm <= 0.0:
        raise ValueError(f"Learned steering vector has invalid norm: {learned_norm}")

    generator = torch.Generator(device="cpu")

    generator.manual_seed(int(seed))

    random_vector = torch.randn(
        learned.shape,
        generator=generator,
        dtype=torch.float32,
    )

    random_norm = float(torch.linalg.vector_norm(random_vector).item())

    if not np.isfinite(random_norm) or random_norm <= 0.0:
        raise ValueError("Random steering vector has invalid norm.")

    random_vector = random_vector * (learned_norm / random_norm)

    matched_norm = float(torch.linalg.vector_norm(random_vector).item())

    cosine = float(
        torch.dot(
            learned,
            random_vector,
        ).item()
        / (learned_norm * matched_norm)
    )

    return (
        random_vector,
        {
            "learned_vector_norm": (learned_norm),
            "random_vector_norm": (matched_norm),
            "norm_ratio": (matched_norm / learned_norm),
            "cosine_with_caa": (cosine),
        },
    )


# =============================================================================
# Statistics normalization
# =============================================================================


def canonical_behavior_statistics(
    stats,
    *,
    direction: str,
):
    """
    Convert behavior_statistics() output into the fields required by
    the specificity-control analysis.
    """

    row = as_dict(stats)

    required = [
        "baseline_behavior_score",
        "behavior_score",
        "delta_behavior",
        "delta_ci_low",
        "delta_ci_high",
        "num_examples",
        "bootstrap_samples",
    ]

    missing = [column for column in required if column not in row]

    if missing:
        raise ValueError(
            "behavior_statistics() returned an unexpected schema. "
            f"Missing: {missing}; "
            f"available: {sorted(row)}"
        )

    delta = float(row["delta_behavior"])

    delta_low = float(row["delta_ci_low"])

    delta_high = float(row["delta_ci_high"])

    if direction == "positive":
        directional_gain = delta

        directional_low = delta_low

        directional_high = delta_high

    elif direction == "negative":
        directional_gain = -delta

        directional_low = -delta_high

        directional_high = -delta_low

    else:
        raise ValueError(f"Unexpected direction: {direction}")

    return {
        "random_behavior_baseline": float(row["baseline_behavior_score"]),
        "random_behavior_score": float(row["behavior_score"]),
        "random_behavior_delta": (delta),
        "random_behavior_delta_ci_low": (delta_low),
        "random_behavior_delta_ci_high": (delta_high),
        "random_directional_gain": (directional_gain),
        "random_directional_gain_ci_low": (directional_low),
        "random_directional_gain_ci_high": (directional_high),
        "random_direction_supported": bool(directional_low > 0.0),
        "num_examples": int(row["num_examples"]),
        "bootstrap_samples": int(row["bootstrap_samples"]),
    }


# =============================================================================
# Resume
# =============================================================================


def result_key(
    row,
):
    return (
        int(row["control_seed_index"]),
        str(row["behavior"]),
        int(row["layer"]),
        str(row["direction"]),
        round(
            float(row["alpha_star"]),
            12,
        ),
    )


def load_existing_results(
    path: Path,
    *,
    resume: bool,
):
    if not resume or not path.is_file():
        return (
            pd.DataFrame(),
            set(),
        )

    frame = pd.read_csv(path)

    keys = {result_key(row) for row in (frame.to_dict(orient="records"))}

    print(f"Resume: loaded {len(frame)} completed random-control evaluations.")

    return (
        frame,
        keys,
    )


# =============================================================================
# Summary
# =============================================================================


def summarize_random_controls(
    results: pd.DataFrame,
):
    if results.empty:
        return pd.DataFrame()

    intervention_key = [
        "model_slug",
        "model_label",
        "behavior",
        "layer",
        "direction",
        "alpha_star",
        "eta_star",
        "frontier_epsilons",
    ]

    rows = []

    for key_values, group in results.groupby(
        intervention_key,
        dropna=False,
        sort=False,
    ):
        base = dict(
            zip(
                intervention_key,
                key_values,
            )
        )

        random_gain = group["random_directional_gain"].astype(float).to_numpy()

        caa_gain_values = group["caa_test_directional_gain"].astype(float).unique()

        if len(caa_gain_values) != 1:
            raise ValueError("CAA held-out gain differs across random seeds.")

        caa_gain = float(caa_gain_values[0])

        n = int(len(random_gain))

        # One-sided Monte Carlo p-value:
        # how often does a random direction achieve at least
        # the held-out directional gain of the actual CAA vector?
        empirical_p = float((1 + np.sum(random_gain >= caa_gain)) / (n + 1))

        percentile = float(
            (np.sum(random_gain < caa_gain) + 0.5 * np.sum(random_gain == caa_gain)) / n
        )

        rows.append(
            {
                **base,
                "num_random_directions": (n),
                "caa_test_directional_gain": (caa_gain),
                "random_gain_mean": float(np.mean(random_gain)),
                "random_gain_std": (
                    float(
                        np.std(
                            random_gain,
                            ddof=1,
                        )
                    )
                    if n > 1
                    else np.nan
                ),
                "random_gain_q025": float(
                    np.quantile(
                        random_gain,
                        0.025,
                    )
                ),
                "random_gain_median": float(
                    np.quantile(
                        random_gain,
                        0.50,
                    )
                ),
                "random_gain_q975": float(
                    np.quantile(
                        random_gain,
                        0.975,
                    )
                ),
                "caa_minus_random_mean": float(caa_gain - np.mean(random_gain)),
                "caa_gain_random_percentile": (percentile),
                "empirical_p_random_ge_caa": (empirical_p),
                "random_direction_support_rate": float(
                    group["random_direction_supported"].mean()
                ),
                "mean_abs_cosine_with_caa": float(
                    group["cosine_with_caa"].abs().mean()
                ),
                "max_abs_cosine_with_caa": float(group["cosine_with_caa"].abs().max()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Plan
# =============================================================================


def print_plan(
    *,
    config,
    behaviors,
    operating_points,
    num_seeds,
    seed_start,
    frozen_path,
    final_path,
    output_root,
):
    print("=" * 76)

    print("NORM-MATCHED RANDOM SPECIFICITY CONTROL")

    print("=" * 76)

    print(f"Model:          {config.model_label}")

    print(f"Model slug:     {config.model_slug}")

    print(f"Behaviors:      {len(behaviors)}")

    print(f"Seeds:          {seed_start} .. {seed_start + num_seeds - 1}")

    print(f"Random draws:   {num_seeds} / behavior")

    print(f"Operating pts:  {len(operating_points)}")

    print(f"Steered evals:  {len(operating_points) * num_seeds}")

    print(f"Baseline evals: {len(behaviors)}")

    print(f"Frozen input:   {frozen_path}")

    print(f"Final input:    {final_path}")

    print(f"Output:         {output_root}")

    print()

    display = operating_points[
        [
            "behavior",
            "layer",
            "direction",
            "alpha_star",
            "eta_star",
            "frontier_epsilons",
            "test_directional_gain",
        ]
    ].copy()

    print(display.to_string(index=False))


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1.")

    config = load_experiment_config(args.config)

    behaviors = (
        list(config.behaviors) if args.behaviors is None else list(args.behaviors)
    )

    unknown = set(behaviors) - set(config.behaviors)

    if unknown:
        raise ValueError(f"Requested behaviors are not in config: {sorted(unknown)}")

    frozen_path = (
        args.frontiers
        if args.frontiers is not None
        else (config.results_root / "steering_frontiers.csv")
    )

    final_path = (
        args.final_results
        if args.final_results is not None
        else (config.results_root / "final_evaluation" / "frontier_test_results.csv")
    )

    output_root = (
        args.output_root
        if args.output_root is not None
        else (config.results_root / "controls" / CONTROL_NAME)
    )

    operating_points = load_operating_points(
        frozen_path,
        final_path,
        behaviors,
    )

    print_plan(
        config=config,
        behaviors=behaviors,
        operating_points=operating_points,
        num_seeds=args.num_seeds,
        seed_start=args.seed_start,
        frozen_path=frozen_path,
        final_path=final_path,
        output_root=output_root,
    )

    if args.plan_only:
        print()
        print("Plan only — model not loaded.")
        return

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectors_root = output_root / "vectors"

    vectors_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = output_root / "random_control_results.csv"

    summary_path = output_root / "random_control_summary.csv"

    resume = not args.no_resume

    existing_df, completed = load_existing_results(
        results_path,
        resume=resume,
    )

    print()
    print("Loading normalized CAA vectors...")

    normalized_vectors = load_normalized_vectors(config)

    # Validate selected layers before GPU/model loading.
    for behavior in behaviors:
        rows = operating_points.loc[operating_points["behavior"] == behavior]

        if rows.empty:
            print(f"WARNING: {behavior}: no nonzero frozen operating points.")

            continue

        layers = set(rows["layer"].astype(int))

        if len(layers) != 1:
            raise ValueError(
                f"{behavior}: expected one frozen selected layer, "
                f"found {sorted(layers)}"
            )

        layer = int(next(iter(layers)))

        vectors = normalized_vectors[behavior]

        if not (0 <= layer < vectors.shape[0]):
            raise ValueError(
                f"{behavior}: selected layer {layer} "
                f"outside vector tensor shape "
                f"{tuple(vectors.shape)}"
            )

    print("Loading model...")

    model, tokenizer = load_model(
        model_name=config.model_name,
        device=args.device,
        dtype=config.torch_dtype,
    )

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=config.model_name,
        prompt_family=config.prompt_family,
    )

    print(
        "Prompt adapter:",
        adapter.family,
    )

    # ------------------------------------------------------------------
    # Baseline test evaluation once per behavior
    # ------------------------------------------------------------------

    baselines = {}
    test_datasets = {}

    print()
    print("=" * 76)

    print("TEST BASELINES")

    print("=" * 76)

    for behavior in behaviors:
        points = operating_points.loc[operating_points["behavior"] == behavior]

        if points.empty:
            continue

        layer = int(points["layer"].iloc[0])

        learned_vector = normalized_vectors[behavior][layer]

        dataset_path = config.layout.test_data_path(behavior)

        dataset = load_test_data(dataset_path)

        test_datasets[behavior] = dataset

        print(f"{behavior:18s} layer={layer:2d} N={len(dataset)}")

        baseline_result, baseline_items = evaluate_behavior_condition(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            dataset=dataset,
            layer=layer,
            vector=learned_vector,
            alpha=0.0,
            device=args.device,
        )

        baselines[behavior] = as_dataframe(baseline_items)

    # ------------------------------------------------------------------
    # Random controls
    # ------------------------------------------------------------------

    all_rows = existing_df.to_dict(orient="records") if not existing_df.empty else []

    seed_indices = range(
        args.seed_start,
        args.seed_start + args.num_seeds,
    )

    total = len(operating_points) * args.num_seeds

    done_this_run = 0

    print()
    print("=" * 76)

    print("RANDOM-DIRECTION EVALUATION")

    print("=" * 76)

    for seed_index in seed_indices:
        seed_vector_path = vectors_root / (f"seed_{seed_index:03d}.pt")

        random_vectors = {}
        vector_metadata = {}

        for behavior in behaviors:
            points = operating_points.loc[operating_points["behavior"] == behavior]

            if points.empty:
                continue

            layer = int(points["layer"].iloc[0])

            learned_vector = normalized_vectors[behavior][layer]

            vector_seed = stable_seed(
                args.base_seed,
                CONTROL_NAME,
                config.model_slug,
                behavior,
                seed_index,
            )

            (
                random_vector,
                metadata,
            ) = make_norm_matched_random_vector(
                learned_vector,
                seed=vector_seed,
            )

            random_vectors[behavior] = random_vector

            vector_metadata[behavior] = {
                "layer": layer,
                "vector_seed": int(vector_seed),
                **metadata,
            }

        # Save the actual vectors, not just their seeds.
        torch.save(
            {
                "control": (CONTROL_NAME),
                "model_slug": (config.model_slug),
                "control_seed_index": (seed_index),
                "base_seed": (args.base_seed),
                "vectors": (random_vectors),
                "metadata": (vector_metadata),
            },
            seed_vector_path,
        )

        print()
        print(f"Seed {seed_index:03d} -> {seed_vector_path}")

        for behavior in behaviors:
            points = operating_points.loc[
                operating_points["behavior"] == behavior
            ].sort_values(
                [
                    "direction",
                    "alpha_star",
                ]
            )

            if points.empty:
                continue

            vector = random_vectors[behavior]

            metadata = vector_metadata[behavior]

            baseline_df = baselines[behavior]

            dataset = test_datasets[behavior]

            for point in points.to_dict(orient="records"):
                layer = int(point["layer"])

                alpha = float(point["alpha_star"])

                direction = str(point["direction"])

                candidate_key = (
                    seed_index,
                    behavior,
                    layer,
                    direction,
                    round(
                        alpha,
                        12,
                    ),
                )

                if candidate_key in (completed):
                    continue

                print(
                    f"[{done_this_run + 1:4d}/{total}] "
                    f"seed={seed_index:03d} "
                    f"{behavior:18s} "
                    f"L={layer:2d} "
                    f"alpha={alpha:+.2f}"
                )

                steered_result, steered_items = evaluate_behavior_condition(
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    dataset=dataset,
                    layer=layer,
                    vector=vector,
                    alpha=alpha,
                    device=args.device,
                )

                steered_df = as_dataframe(steered_items)

                bootstrap_seed = stable_seed(
                    args.base_seed,
                    CONTROL_NAME,
                    "bootstrap",
                    config.model_slug,
                    behavior,
                    seed_index,
                    layer,
                    direction,
                    alpha,
                )

                stats = behavior_statistics(
                    baseline_df=baseline_df,
                    steered_df=steered_df,
                    behavior=behavior,
                    alpha=alpha,
                    bootstrap_samples=(args.bootstrap_samples),
                    bootstrap_seed=(bootstrap_seed),
                )

                canonical = canonical_behavior_statistics(
                    stats,
                    direction=direction,
                )

                row = {
                    "control": (CONTROL_NAME),
                    "model_name": (config.model_name),
                    "model_label": (config.model_label),
                    "model_slug": (config.model_slug),
                    "behavior": (behavior),
                    "layer": (layer),
                    "direction": (direction),
                    "alpha_star": (alpha),
                    "eta_star": float(point["eta_star"]),
                    "frontier_epsilons": (point["frontier_epsilons"]),
                    "control_seed_index": (seed_index),
                    "vector_seed": int(metadata["vector_seed"]),
                    "bootstrap_seed": int(bootstrap_seed),
                    "learned_vector_norm": float(metadata["learned_vector_norm"]),
                    "random_vector_norm": float(metadata["random_vector_norm"]),
                    "norm_ratio": float(metadata["norm_ratio"]),
                    "cosine_with_caa": float(metadata["cosine_with_caa"]),
                    "selection_behavior_gain": float(point["selection_behavior_gain"]),
                    "caa_test_behavior_delta": float(point["test_behavior_delta"]),
                    "caa_test_behavior_delta_ci_low": float(
                        point["test_behavior_delta_ci_low"]
                    ),
                    "caa_test_behavior_delta_ci_high": float(
                        point["test_behavior_delta_ci_high"]
                    ),
                    "caa_test_directional_gain": float(point["test_directional_gain"]),
                    "caa_test_directional_gain_ci_low": float(
                        point["test_directional_gain_ci_low"]
                    ),
                    "caa_test_directional_gain_ci_high": float(
                        point["test_directional_gain_ci_high"]
                    ),
                    "caa_test_direction_supported": bool(
                        point["test_direction_supported"]
                    ),
                    **canonical,
                }

                all_rows.append(row)

                completed.add(candidate_key)

                done_this_run += 1

                # Persist after every condition so a preemption loses
                # at most one evaluation.
                result_frame = (
                    pd.DataFrame(all_rows)
                    .sort_values(
                        [
                            "control_seed_index",
                            "behavior",
                            "layer",
                            "alpha_star",
                        ]
                    )
                    .reset_index(drop=True)
                )

                write_csv(
                    result_frame,
                    results_path,
                )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    results = pd.DataFrame(all_rows)

    if not results.empty:
        results = results.sort_values(
            [
                "behavior",
                "layer",
                "direction",
                "alpha_star",
                "control_seed_index",
            ]
        ).reset_index(drop=True)

        write_csv(
            results,
            results_path,
        )

    summary = summarize_random_controls(results)

    write_csv(
        summary,
        summary_path,
    )

    manifest = {
        "control": (CONTROL_NAME),
        "model_name": (config.model_name),
        "model_label": (config.model_label),
        "model_slug": (config.model_slug),
        "config": str(args.config),
        "device": (args.device),
        "behaviors": (behaviors),
        "num_random_directions": (args.num_seeds),
        "seed_start": (args.seed_start),
        "base_seed": (args.base_seed),
        "bootstrap_samples": (args.bootstrap_samples),
        "frozen_frontiers": str(frozen_path),
        "frozen_frontiers_sha256": (sha256_file(frozen_path)),
        "final_results": str(final_path),
        "final_results_sha256": (sha256_file(final_path)),
        "num_unique_frozen_nonzero_operating_points": int(len(operating_points)),
        "num_completed_control_evaluations": int(len(results)),
        "vector_definition": (
            "isotropic Gaussian random direction, "
            "rescaled to exactly match the L2 norm of the "
            "normalized CAA vector at the frozen selected layer"
        ),
        "selection_policy": (
            "No control-based selection or reselection. "
            "CAA layers and coefficients remain frozen."
        ),
        "evaluation_split": ("behavior test"),
    }

    manifest_path = output_root / "manifest.json"

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 76)

    print("NORM-MATCHED RANDOM CONTROL COMPLETE")

    print("=" * 76)

    print(f"Control evaluations: {len(results)}")

    print(f"Unique operating pts: {len(summary)}")

    print(f"Results:              {results_path}")

    print(f"Summary:              {summary_path}")

    print(f"Manifest:             {manifest_path}")

    if not summary.empty:
        print()
        print(
            summary[
                [
                    "behavior",
                    "layer",
                    "direction",
                    "alpha_star",
                    "caa_test_directional_gain",
                    "random_gain_mean",
                    "random_gain_q975",
                    "empirical_p_random_ge_caa",
                    "mean_abs_cosine_with_caa",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
