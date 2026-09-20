"""Evaluate ordered causal transfer between behavior steering directions.

Transfer strength is fixed in residual-relative units at predetermined depth
anchors, preventing target-specific tuning of intervention magnitude.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_final_evaluation import (
    behavior_statistics,
    evaluate_behavior_condition,
)
from run_specificity_controls import (
    as_dataframe,
    as_dict,
    stable_seed,
    write_csv,
)

from steerfrontier.config import load_experiment_config
from steerfrontier.data import load_test_data
from steerfrontier.models import load_model
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.workflow import load_normalized_vectors

# =============================================================================
# Constants
# =============================================================================

DEFAULT_TRANSFER_ETA = 0.20
DEFAULT_BOOTSTRAP_SAMPLES = 10_000

NONZERO_TOL = 1e-12


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure ordered cross-behavior causal transfer at fixed "
            "relative perturbation strength eta."
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
        "--eta",
        type=float,
        default=DEFAULT_TRANSFER_ETA,
        help=("Fixed relative perturbation magnitude. Frozen default: eta=0.20."),
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260918,
    )

    parser.add_argument(
        "--vectors-root",
        type=Path,
        default=Path("vectors"),
    )

    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=None,
        help=("Default: <results_root>/geometry"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=("Default: <results_root>/causal_transfer"),
    )

    parser.add_argument(
        "--source-behaviors",
        nargs="*",
        default=None,
        help=(
            "Optional subset of injected source behaviors. "
            "Default: all configured behaviors."
        ),
    )

    parser.add_argument(
        "--target-behaviors",
        nargs="*",
        default=None,
        help=(
            "Optional subset of evaluated target behaviors. "
            "Default: all configured behaviors."
        ),
    )

    parser.add_argument(
        "--depths",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional subset of requested fractional-depth anchors, "
            "e.g. --depths 0.42. Matching uses np.isclose."
        ),
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def sha256_file(
    path: Path,
):
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def result_key(
    row,
):
    return (
        str(row["source_behavior"]),
        str(row["target_behavior"]),
        int(row["layer"]),
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

    keys = {result_key(row) for row in frame.to_dict(orient="records")}

    print(f"Resume: loaded {len(frame)} completed causal-transfer conditions.")

    return (
        frame,
        keys,
    )


def canonical_transfer_statistics(
    stats,
):
    """
    For cross-behavior transfer we preserve the signed target-behavior
    change directly:

        T_{i->j}(l) = B_j(steered with +v_i) - B_j(baseline)

    No source-direction sign correction is applied.
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
            "behavior_statistics() returned "
            "an unexpected schema. "
            f"Missing: {missing}; "
            f"available: {sorted(row)}"
        )

    return {
        "target_behavior_baseline": float(row["baseline_behavior_score"]),
        "target_behavior_score": float(row["behavior_score"]),
        "transfer_effect": float(row["delta_behavior"]),
        "transfer_ci_low": float(row["delta_ci_low"]),
        "transfer_ci_high": float(row["delta_ci_high"]),
        "transfer_supported_positive": bool(float(row["delta_ci_low"]) > 0.0),
        "transfer_supported_negative": bool(float(row["delta_ci_high"]) < 0.0),
        "num_examples": int(row["num_examples"]),
        "bootstrap_samples": int(row["bootstrap_samples"]),
    }


# =============================================================================
# Geometry / anchors
# =============================================================================


def load_anchor_layers(
    geometry_root: Path,
    *,
    requested_depths=None,
):
    path = geometry_root / "anchor_layers.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "requested_fractional_depth",
        "layer",
        "fractional_depth",
        "depth_error",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if requested_depths is not None:
        keep = np.zeros(
            len(frame),
            dtype=bool,
        )

        for depth in requested_depths:
            keep |= np.isclose(
                frame["requested_fractional_depth"].astype(float),
                float(depth),
                atol=1e-8,
                rtol=0.0,
            )

        frame = frame.loc[keep].copy()

        if len(frame) != len(requested_depths):
            found = frame["requested_fractional_depth"].tolist()

            raise ValueError(
                "Could not uniquely match all "
                f"requested depths. "
                f"Requested={requested_depths}; "
                f"found={found}"
            )

    return frame.sort_values("requested_fractional_depth").reset_index(drop=True)


def load_geometry_map(
    geometry_root: Path,
):
    path = geometry_root / "pairwise_cosine_anchor_layers.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "layer",
        "behavior_i",
        "behavior_j",
        "cosine_similarity",
        "abs_cosine_similarity",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    mapping = {}

    for row in frame.to_dict(orient="records"):
        pair = tuple(
            sorted(
                [
                    str(row["behavior_i"]),
                    str(row["behavior_j"]),
                ]
            )
        )

        key = (
            int(row["layer"]),
            pair[0],
            pair[1],
        )

        if key in mapping:
            raise ValueError(f"Duplicate geometry key: {key}")

        mapping[key] = {
            "cosine_similarity": float(row["cosine_similarity"]),
            "abs_cosine_similarity": float(row["abs_cosine_similarity"]),
        }

    return (
        mapping,
        path,
    )


def geometry_for_pair(
    mapping,
    *,
    layer: int,
    source_behavior: str,
    target_behavior: str,
):
    pair = tuple(
        sorted(
            [
                source_behavior,
                target_behavior,
            ]
        )
    )

    key = (
        int(layer),
        pair[0],
        pair[1],
    )

    if key not in mapping:
        raise KeyError(f"Missing geometry for layer={layer}, pair={pair}")

    return mapping[key]


# =============================================================================
# Eta calibration
# =============================================================================


def load_eta_calibration(
    *,
    vectors_root: Path,
    model_slug: str,
    num_layers: int,
):
    root = vectors_root / model_slug / "normalization"

    residual_path = root / "residual_scale.pt"

    reference_path = root / "reference_layer_norms.pt"

    ratio_path = root / "perturbation_ratio.pt"

    metadata_path = root / "residual_scale_metadata.json"

    for path in [
        residual_path,
        reference_path,
        ratio_path,
        metadata_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    residual_scale = (
        torch.load(
            residual_path,
            map_location="cpu",
        )
        .float()
        .reshape(-1)
    )

    reference_norm = (
        torch.load(
            reference_path,
            map_location="cpu",
        )
        .float()
        .reshape(-1)
    )

    perturbation_ratio = (
        torch.load(
            ratio_path,
            map_location="cpu",
        )
        .float()
        .reshape(-1)
    )

    for name, tensor in [
        (
            "residual_scale",
            residual_scale,
        ),
        (
            "reference_layer_norms",
            reference_norm,
        ),
        (
            "perturbation_ratio",
            perturbation_ratio,
        ),
    ]:
        if tensor.numel() != num_layers:
            raise ValueError(
                f"{name}: expected {num_layers} layers, found {tensor.numel()}."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values.")

    if (residual_scale <= 0.0).any():
        raise ValueError("Residual scale must be strictly positive.")

    if (reference_norm <= 0.0).any():
        raise ValueError("Reference vector norm must be strictly positive.")

    if (perturbation_ratio <= 0.0).any():
        raise ValueError("Perturbation ratio must be strictly positive.")

    recomputed_ratio = reference_norm / residual_scale

    if not torch.allclose(
        perturbation_ratio,
        recomputed_ratio,
        atol=1e-6,
        rtol=1e-5,
    ):
        maximum_error = float(
            torch.max(torch.abs(perturbation_ratio - recomputed_ratio)).item()
        )

        raise ValueError(
            "Saved perturbation_ratio "
            "does not match "
            "reference_layer_norms / "
            "residual_scale. "
            f"max abs error="
            f"{maximum_error:.6g}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    expected_definition = (
        "eta(alpha,l) = |alpha| * reference_vector_norm_l / residual_scale_l"
    )

    actual_definition = metadata.get("eta_definition")

    if actual_definition != expected_definition:
        print("WARNING: eta_definition metadata differs from the expected string:")
        print(f"  {actual_definition}")

    return {
        "root": root,
        "residual_scale": residual_scale,
        "reference_norm": reference_norm,
        "perturbation_ratio": (perturbation_ratio),
        "metadata": metadata,
        "residual_path": residual_path,
        "reference_path": reference_path,
        "ratio_path": ratio_path,
        "metadata_path": metadata_path,
    }


def alpha_for_eta(
    *,
    eta: float,
    perturbation_ratio: float,
):
    """
    eta(alpha,l)
        = |alpha| * perturbation_ratio_l

    For the positive/source-vector intervention used in causal transfer:

        alpha_l = eta / perturbation_ratio_l
    """

    if eta <= 0.0:
        raise ValueError("Transfer eta must be > 0.")

    if perturbation_ratio <= 0.0:
        raise ValueError("Perturbation ratio must be > 0.")

    return float(eta / perturbation_ratio)


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.eta <= 0.0:
        raise ValueError("--eta must be > 0.")

    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be >= 1.")

    config = load_experiment_config(args.config)

    all_behaviors = list(config.behaviors)

    sources = (
        all_behaviors if args.source_behaviors is None else list(args.source_behaviors)
    )

    targets = (
        all_behaviors if args.target_behaviors is None else list(args.target_behaviors)
    )

    unknown_sources = set(sources) - set(all_behaviors)

    unknown_targets = set(targets) - set(all_behaviors)

    if unknown_sources:
        raise ValueError(f"Unknown source behaviors: {sorted(unknown_sources)}")

    if unknown_targets:
        raise ValueError(f"Unknown target behaviors: {sorted(unknown_targets)}")

    geometry_root = (
        args.geometry_root
        if args.geometry_root is not None
        else (config.results_root / "geometry")
    )

    output_root = (
        args.output_root
        if args.output_root is not None
        else (config.results_root / "causal_transfer")
    )

    anchors = load_anchor_layers(
        geometry_root,
        requested_depths=(args.depths),
    )

    (
        geometry_map,
        geometry_path,
    ) = load_geometry_map(geometry_root)

    print()
    print("Loading normalized steering vectors...")

    normalized_vectors = load_normalized_vectors(config)

    first_behavior = all_behaviors[0]

    first_tensor = normalized_vectors[first_behavior]

    if first_tensor.ndim != 2:
        raise ValueError("Expected steering-vector tensor shape [L,H].")

    num_layers = int(first_tensor.shape[0])

    hidden_size = int(first_tensor.shape[1])

    for behavior in all_behaviors:
        tensor = normalized_vectors[behavior]

        if tuple(tensor.shape) != (
            num_layers,
            hidden_size,
        ):
            raise ValueError(
                f"{behavior}: inconsistent vector shape {tuple(tensor.shape)}."
            )

    calibration = load_eta_calibration(
        vectors_root=(args.vectors_root),
        model_slug=(config.model_slug),
        num_layers=(num_layers),
    )

    # ------------------------------------------------------------
    # Build frozen intervention table
    # ------------------------------------------------------------

    intervention_rows = []

    for anchor in anchors.to_dict(orient="records"):
        layer = int(anchor["layer"])

        ratio = float(calibration["perturbation_ratio"][layer].item())

        residual_scale = float(calibration["residual_scale"][layer].item())

        reference_norm = float(calibration["reference_norm"][layer].item())

        alpha = alpha_for_eta(
            eta=(args.eta),
            perturbation_ratio=(ratio),
        )

        for source in sources:
            for target in targets:
                if source == target:
                    continue

                geometry = geometry_for_pair(
                    geometry_map,
                    layer=layer,
                    source_behavior=(source),
                    target_behavior=(target),
                )

                intervention_rows.append(
                    {
                        "source_behavior": (source),
                        "target_behavior": (target),
                        "layer": (layer),
                        "requested_fractional_depth": float(
                            anchor["requested_fractional_depth"]
                        ),
                        "fractional_depth": float(anchor["fractional_depth"]),
                        "depth_error": float(anchor["depth_error"]),
                        "eta_transfer": float(args.eta),
                        "perturbation_ratio": (ratio),
                        "residual_scale": (residual_scale),
                        "reference_vector_norm": (reference_norm),
                        "alpha_transfer": (alpha),
                        "cosine_similarity": float(geometry["cosine_similarity"]),
                        "abs_cosine_similarity": float(
                            geometry["abs_cosine_similarity"]
                        ),
                    }
                )

    interventions = (
        pd.DataFrame(intervention_rows)
        .sort_values(
            [
                "layer",
                "source_behavior",
                "target_behavior",
            ]
        )
        .reset_index(drop=True)
    )

    expected = len(anchors) * sum(
        1 for source in sources for target in targets if source != target
    )

    if len(interventions) != expected:
        raise RuntimeError(
            f"Expected {expected} transfer conditions, found {len(interventions)}."
        )

    print()
    print("=" * 76)
    print("CAUSAL TRANSFER")
    print("=" * 76)
    print(f"Model:             {config.model_label}")
    print(f"Model slug:        {config.model_slug}")
    print(f"Layers:            {num_layers}")
    print(f"Hidden size:       {hidden_size}")
    print(f"Source behaviors:  {len(sources)}")
    print(f"Target behaviors:  {len(targets)}")
    print(f"Depth anchors:     {len(anchors)}")
    print(f"eta_transfer:      {args.eta:.4f}")
    print(f"Transfer evals:    {len(interventions)}")
    print(f"Output:            {output_root}")

    print()
    print("Frozen layer-wise intervention strengths:")

    display = (
        interventions[
            [
                "requested_fractional_depth",
                "layer",
                "fractional_depth",
                "perturbation_ratio",
                "alpha_transfer",
            ]
        ]
        .drop_duplicates()
        .sort_values("requested_fractional_depth")
    )

    print(display.to_string(index=False))

    if args.plan_only:
        print()
        print("Plan only — model not loaded.")
        return

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    plan_path = output_root / "transfer_plan.csv"

    results_path = output_root / "transfer_results.csv"

    summary_path = output_root / "transfer_layer_summary.csv"

    interventions.to_csv(
        plan_path,
        index=False,
    )

    resume = not args.no_resume

    (
        existing_results,
        completed,
    ) = load_existing_results(
        results_path,
        resume=resume,
    )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    print()
    print("Loading model...")

    model, tokenizer = load_model(
        model_name=(config.model_name),
        device=(args.device),
        dtype=(config.torch_dtype),
    )

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=(config.model_name),
        prompt_family=(config.prompt_family),
    )

    print(
        "Prompt adapter:",
        adapter.family,
    )

    # ------------------------------------------------------------
    # Target datasets + baselines
    # ------------------------------------------------------------

    target_datasets = {}
    baseline_items = {}

    first_anchor_layer = int(anchors["layer"].iloc[0])

    print()
    print("=" * 76)
    print("TARGET TEST BASELINES")
    print("=" * 76)

    for target in targets:
        # Skip targets for which no actual off-diagonal
        # condition exists after source/target filtering.
        if not (interventions["target_behavior"] == target).any():
            continue

        dataset = load_test_data(config.layout.test_data_path(target))

        target_datasets[target] = dataset

        print(f"{target:18s} N={len(dataset)}")

        # alpha=0, so neither layer nor vector changes the model.
        # We use a valid target vector at the first anchor merely
        # to exercise exactly the same evaluation path.
        (
            _,
            items,
        ) = evaluate_behavior_condition(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            dataset=dataset,
            layer=(first_anchor_layer),
            vector=(normalized_vectors[target][first_anchor_layer]),
            alpha=0.0,
            device=(args.device),
        )

        baseline_items[target] = as_dataframe(items)

    # ------------------------------------------------------------
    # Transfer evaluations
    # ------------------------------------------------------------

    all_rows = (
        existing_results.to_dict(orient="records") if not existing_results.empty else []
    )

    total = len(interventions)

    print()
    print("=" * 76)
    print("ORDERED CAUSAL TRANSFER")
    print("=" * 76)
    print(f"Expected:  {total}")
    print(f"Completed: {len(completed)}")
    print(f"Remaining: {max(total - len(completed), 0)}")

    for condition in interventions.to_dict(orient="records"):
        key = result_key(condition)

        if key in completed:
            continue

        source = str(condition["source_behavior"])

        target = str(condition["target_behavior"])

        layer = int(condition["layer"])

        alpha = float(condition["alpha_transfer"])

        print(
            f"[{len(completed) + 1:4d}/"
            f"{total}] "
            f"d="
            f"{float(condition['requested_fractional_depth']):.2f} "
            f"L={layer:2d} "
            f"{source:18s} -> "
            f"{target:18s} "
            f"eta={args.eta:.2f} "
            f"alpha={alpha:+.4f}"
        )

        (
            _,
            steered_items,
        ) = evaluate_behavior_condition(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            dataset=(target_datasets[target]),
            layer=(layer),
            vector=(normalized_vectors[source][layer]),
            alpha=(alpha),
            device=(args.device),
        )

        bootstrap_seed = stable_seed(
            args.base_seed,
            "causal_transfer",
            config.model_slug,
            source,
            target,
            layer,
            args.eta,
        )

        stats = behavior_statistics(
            baseline_df=(baseline_items[target]),
            steered_df=(as_dataframe(steered_items)),
            behavior=(target),
            alpha=(alpha),
            bootstrap_samples=(args.bootstrap_samples),
            bootstrap_seed=(bootstrap_seed),
        )

        canonical = canonical_transfer_statistics(stats)

        row = {
            "model_name": (config.model_name),
            "model_label": (config.model_label),
            "model_slug": (config.model_slug),
            **condition,
            "source_intervention_sign": ("positive"),
            "bootstrap_seed": int(bootstrap_seed),
            **canonical,
        }

        all_rows.append(row)

        completed.add(key)

        frame = (
            pd.DataFrame(all_rows)
            .sort_values(
                [
                    "layer",
                    "source_behavior",
                    "target_behavior",
                ]
            )
            .reset_index(drop=True)
        )

        # Persist after every condition.
        write_csv(
            frame,
            results_path,
        )

    # ------------------------------------------------------------
    # Final results / summary
    # ------------------------------------------------------------

    results = pd.DataFrame(all_rows)

    if not results.empty:
        results = results.sort_values(
            [
                "layer",
                "source_behavior",
                "target_behavior",
            ]
        ).reset_index(drop=True)

        write_csv(
            results,
            results_path,
        )

    layer_summary = results.groupby(
        [
            "model_slug",
            "model_label",
            "requested_fractional_depth",
            "layer",
            "fractional_depth",
        ],
        as_index=False,
    ).agg(
        num_transfers=(
            "transfer_effect",
            "size",
        ),
        mean_signed_transfer=(
            "transfer_effect",
            "mean",
        ),
        mean_abs_transfer=(
            "transfer_effect",
            lambda x: float(
                np.mean(
                    np.abs(
                        np.asarray(
                            x,
                            dtype=float,
                        )
                    )
                )
            ),
        ),
        max_abs_transfer=(
            "transfer_effect",
            lambda x: float(
                np.max(
                    np.abs(
                        np.asarray(
                            x,
                            dtype=float,
                        )
                    )
                )
            ),
        ),
        positive_supported_rate=(
            "transfer_supported_positive",
            "mean",
        ),
        negative_supported_rate=(
            "transfer_supported_negative",
            "mean",
        ),
        mean_cosine=(
            "cosine_similarity",
            "mean",
        ),
        mean_abs_cosine=(
            "abs_cosine_similarity",
            "mean",
        ),
    )

    write_csv(
        layer_summary,
        summary_path,
    )

    manifest = {
        "analysis": ("ordered_cross_behavior_causal_transfer"),
        "model_name": (config.model_name),
        "model_label": (config.model_label),
        "model_slug": (config.model_slug),
        "source_behaviors": (sources),
        "target_behaviors": (targets),
        "num_layers": (num_layers),
        "hidden_size": (hidden_size),
        "eta_transfer": float(args.eta),
        "eta_status": ("frozen before causal-transfer outcomes were observed"),
        "eta_selection_basis": (
            "eta=0.20 chosen conservatively from the "
            "already-completed positive-direction "
            "epsilon=0.03 frozen frontier distribution; "
            "below its first quartile."
        ),
        "intervention_definition": (
            "Inject +alpha_l * normalized CAA source vector "
            "at layer l, where "
            "alpha_l = eta_transfer / perturbation_ratio_l."
        ),
        "eta_definition": (
            "eta(alpha,l) = |alpha| * reference_vector_norm_l / residual_scale_l"
        ),
        "calibration_note": (
            "reference_vector_norm_l is shared across behaviors "
            "within a model/layer, so alpha_l depends on "
            "model/layer but not source behavior."
        ),
        "diagonal_policy": (
            "Excluded. Only ordered source != target transfers are evaluated."
        ),
        "target_split": ("behavior test"),
        "selection_policy": (
            "No transfer-result-based tuning. Depth anchors and eta are frozen."
        ),
        "bootstrap_samples": (args.bootstrap_samples),
        "base_seed": (args.base_seed),
        "num_depth_anchors": int(len(anchors)),
        "depth_anchors": (anchors.to_dict(orient="records")),
        "num_transfer_conditions": int(len(interventions)),
        "num_completed_transfer_conditions": int(len(results)),
        "geometry_file": str(geometry_path),
        "geometry_file_sha256": (sha256_file(geometry_path)),
        "residual_scale_file": str(calibration["residual_path"]),
        "reference_layer_norms_file": str(calibration["reference_path"]),
        "perturbation_ratio_file": str(calibration["ratio_path"]),
        "residual_scale_metadata_file": str(calibration["metadata_path"]),
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
    print("CAUSAL TRANSFER COMPLETE")
    print("=" * 76)
    print(f"Conditions: {len(results)}")
    print(f"Plan:       {plan_path}")
    print(f"Results:    {results_path}")
    print(f"Summary:    {summary_path}")
    print(f"Manifest:   {manifest_path}")

    if not layer_summary.empty:
        print()
        print(
            layer_summary[
                [
                    "requested_fractional_depth",
                    "layer",
                    "num_transfers",
                    "mean_signed_transfer",
                    "mean_abs_transfer",
                    "max_abs_transfer",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
