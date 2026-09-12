import argparse
import hashlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

from steerfrontier.config import load_experiment_config
from steerfrontier.models import load_model
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.statistics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_mean_ci,
)
from steerfrontier.workflow import (
    build_shared_normalization,
    compute_layer_selection_metrics,
    extract_or_load_raw_vectors,
    infer_candidate_layers,
    load_behavior_validation,
    parse_layer_specs,
    print_selection_metrics,
    run_coefficient_sweep,
    run_layer_sweep,
    write_run_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phase-2 CAA analysis runner. For every configured behavior, this "
            "runs or resumes the coarse layer sweep, paired-bootstrap CIs, "
            "candidate-layer inference, dense coefficient sweep, paired-bootstrap "
            "CIs, and directional layer-selection diagnostics."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--phase",
        choices=["analyze", "finalize"],
        default="analyze",
        help=(
            "Use 'analyze' for the current Phase-2 pipeline. "
            "'finalize' is intentionally disabled until the Phase-2 capability "
            "suite and budget-selection protocol are implemented."
        ),
    )

    parser.add_argument(
        "--extract-mode",
        choices=["auto", "always", "never"],
        default="auto",
        help="Extract missing, all, or no raw steering vectors.",
    )

    parser.add_argument(
        "--candidate-layer-spec",
        nargs="*",
        default=[],
        metavar="BEHAVIOR=START-END",
        help=(
            "Optional dense-sweep regions. Behaviors not specified here use a "
            "fixed-width window centered on the strongest coarse layer."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse each compatible completed sweep independently. Missing, "
            "incomplete, or incompatible outputs are recomputed."
        ),
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Number of paired bootstrap resamples per steering condition.",
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=("Base seed used to derive deterministic per-condition bootstrap seeds."),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    return parser.parse_args()


def validate_candidate_specs(
    config,
    candidate_specs,
):
    known = set(config.behaviors)

    unknown = sorted(set(candidate_specs) - known)

    if unknown:
        raise ValueError(f"Candidate-layer specs contain unknown behaviors: {unknown}")


def load_residual_scale(config):
    """
    Load the precomputed unsteered residual-stream scale r_l.

    Generate this once per model with:
        scripts/measure_residual_scale.py
    """

    path = config.layout.normalization_dir() / "residual_scale.pt"

    if not path.exists():
        raise FileNotFoundError(
            "Missing residual-scale calibration: "
            f"{path}\n"
            "Run scripts/measure_residual_scale.py "
            "for this model before analysis."
        )

    residual_scale = torch.load(
        path,
        map_location="cpu",
    ).float()

    if residual_scale.ndim != 1:
        raise ValueError(
            "Residual-scale calibration must be rank-1, "
            f"got shape {tuple(residual_scale.shape)}."
        )

    return residual_scale


def _items_path(
    summary_path: Path,
) -> Path:
    return summary_path.with_name(f"{summary_path.stem}_items.csv")


def _stable_bootstrap_seed(
    base_seed: int,
    behavior: str,
    layer: int,
    alpha: float,
) -> int:
    """
    Derive a deterministic seed for one behavior/layer/alpha condition.
    """

    payload = (f"{base_seed}:{behavior}:{layer}:{alpha:.12g}").encode()

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
    )


def _add_bootstrap_cis(
    summary_path: Path,
    n_bootstrap: int,
    base_seed: int,
) -> pd.DataFrame:
    """
    Add paired-bootstrap 95% CIs for delta_from_baseline.

    The item-level CSV already contains paired per-example deltas:

        steered_matching_prob - baseline_matching_prob

    Bootstrapping those deltas therefore preserves the pairing.
    """

    if n_bootstrap <= 0:
        raise ValueError("bootstrap-samples must be positive.")

    items_path = _items_path(summary_path)

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    if not items_path.exists():
        raise FileNotFoundError(items_path)

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

    # These are cheap CPU-side statistics, so recompute them whenever
    # the runner resumes. That guarantees one consistent definition.
    summary = summary.drop(
        columns=[
            "bootstrap_delta_mean",
            "delta_ci_low",
            "delta_ci_high",
            "bootstrap_samples",
            "bootstrap_base_seed",
        ],
        errors="ignore",
    )

    rows = []

    grouped = items.groupby(
        [
            "behavior",
            "layer",
            "alpha",
        ],
        sort=False,
    )

    for (
        behavior,
        layer,
        alpha,
    ), group in grouped:
        behavior = str(behavior)

        layer = int(layer)

        alpha = float(alpha)

        if group["item_index"].duplicated().any():
            raise RuntimeError(
                "Duplicate item_index values found within "
                f"{behavior}, layer={layer}, alpha={alpha}."
            )

        seed = _stable_bootstrap_seed(
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

        rows.append(
            {
                "behavior": behavior,
                "layer": layer,
                "alpha": alpha,
                "bootstrap_delta_mean": mean,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "bootstrap_samples": int(n_bootstrap),
                "bootstrap_base_seed": int(base_seed),
            }
        )

    statistics = pd.DataFrame(rows)

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
            "Some sweep rows did not receive bootstrap confidence intervals."
        )

    difference = (merged["delta_from_baseline"] - merged["bootstrap_delta_mean"]).abs()

    if len(difference) and difference.max() > 1e-8:
        raise RuntimeError(
            "Aggregate and item-level mean deltas disagree. "
            f"Maximum difference: {difference.max()}"
        )

    merged.to_csv(
        summary_path,
        index=False,
    )

    print(f"Paired bootstrap ({n_bootstrap:,} resamples) -> {summary_path}")

    return merged


def _same_float_values(
    actual: Iterable[float],
    expected: Iterable[float],
) -> bool:
    actual = sorted(float(x) for x in actual)

    expected = sorted(float(x) for x in expected)

    if len(actual) != len(expected):
        return False

    return bool(
        np.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=1e-12,
        )
    )


def _load_reusable_sweep(
    summary_path: Path,
    behavior: str,
    expected_layers: Iterable[int],
    expected_alphas: Iterable[float],
    expected_examples: int,
) -> tuple[
    Optional[pd.DataFrame],
    str,
]:
    """
    Reuse a sweep only if both its summary and item-level outputs are
    complete and match the current experimental grid.
    """

    expected_layers = sorted(int(x) for x in expected_layers)

    expected_alphas = [float(x) for x in expected_alphas]

    items_path = _items_path(summary_path)

    if not summary_path.exists():
        return (
            None,
            "summary CSV is missing",
        )

    if not items_path.exists():
        return (
            None,
            "item-level CSV is missing",
        )

    try:
        summary = pd.read_csv(summary_path)

        items = pd.read_csv(items_path)

    except Exception as exc:
        return (
            None,
            f"could not read saved CSVs: {exc}",
        )

    required_summary = {
        "behavior",
        "layer",
        "alpha",
        "num_examples",
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
        return (
            None,
            f"summary CSV is missing columns {sorted(missing_summary)}",
        )

    if missing_items:
        return (
            None,
            f"item-level CSV is missing columns {sorted(missing_items)}",
        )

    if set(summary["behavior"].astype(str)) != {behavior}:
        return (
            None,
            "summary CSV has the wrong behavior",
        )

    if set(items["behavior"].astype(str)) != {behavior}:
        return (
            None,
            "item-level CSV has the wrong behavior",
        )

    actual_layers = sorted(summary["layer"].astype(int).unique().tolist())

    if actual_layers != expected_layers:
        return (
            None,
            f"layer grid differs: saved={actual_layers}, expected={expected_layers}",
        )

    if not _same_float_values(
        summary["alpha"].unique(),
        expected_alphas,
    ):
        return (
            None,
            "alpha grid differs",
        )

    expected_conditions = len(expected_layers) * len(expected_alphas)

    if len(summary) != expected_conditions:
        return (
            None,
            f"summary has {len(summary)} rows; expected {expected_conditions}",
        )

    if not (summary["num_examples"].astype(int) == int(expected_examples)).all():
        return (
            None,
            "saved validation-set size differs",
        )

    if items.duplicated(
        subset=[
            "layer",
            "alpha",
            "item_index",
        ]
    ).any():
        return (
            None,
            "duplicate item rows detected",
        )

    if (
        not items["item_index"]
        .between(
            0,
            expected_examples - 1,
        )
        .all()
    ):
        return (
            None,
            "item_index values are outside the expected range",
        )

    condition_sizes = items.groupby(
        [
            "layer",
            "alpha",
        ]
    ).size()

    if len(condition_sizes) != expected_conditions:
        return (
            None,
            "item-level condition grid is incomplete",
        )

    if not (condition_sizes == expected_examples).all():
        return (
            None,
            "one or more conditions have missing item rows",
        )

    return (
        summary,
        "compatible saved sweep",
    )


def _run_or_resume_layer_sweep(
    *,
    model,
    tokenizer,
    adapter,
    behavior,
    dataset,
    vectors,
    residual_scale,
    alphas,
    output_path,
    resume,
    device,
) -> pd.DataFrame:
    expected_layers = list(range(vectors.shape[0]))

    if resume:
        saved, reason = _load_reusable_sweep(
            summary_path=output_path,
            behavior=behavior,
            expected_layers=expected_layers,
            expected_alphas=alphas,
            expected_examples=len(dataset),
        )

        if saved is not None:
            print(f"RESUME — reusing layer sweep for {behavior}: {output_path}")

            return saved

        print(f"RESUME — layer sweep for {behavior} will run ({reason}).")

    return run_layer_sweep(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        behavior=behavior,
        dataset=dataset,
        vectors=vectors,
        residual_scale=residual_scale,
        alphas=alphas,
        output_path=output_path,
        device=device,
    )


def _run_or_resume_coefficient_sweep(
    *,
    model,
    tokenizer,
    adapter,
    behavior,
    dataset,
    vectors,
    residual_scale,
    candidate_layers,
    alphas,
    output_path,
    resume,
    device,
) -> pd.DataFrame:
    if resume:
        saved, reason = _load_reusable_sweep(
            summary_path=output_path,
            behavior=behavior,
            expected_layers=candidate_layers,
            expected_alphas=alphas,
            expected_examples=len(dataset),
        )

        if saved is not None:
            print(f"RESUME — reusing coefficient sweep for {behavior}: {output_path}")

            return saved

        print(f"RESUME — coefficient sweep for {behavior} will run ({reason}).")

    return run_coefficient_sweep(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        behavior=behavior,
        dataset=dataset,
        vectors=vectors,
        residual_scale=residual_scale,
        candidate_layers=candidate_layers,
        alphas=alphas,
        output_path=output_path,
        device=device,
    )


def run_analysis(
    model,
    tokenizer,
    adapter,
    config,
    extract_mode,
    candidate_specs,
    resume,
    bootstrap_samples,
    bootstrap_seed,
    device,
):
    raw_vectors = extract_or_load_raw_vectors(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        config=config,
        extract_mode=extract_mode,
        device=device,
    )

    _, normalized_vectors = build_shared_normalization(
        raw_vectors=raw_vectors,
        config=config,
    )

    residual_scale = load_residual_scale(config)

    num_layers = next(iter(normalized_vectors.values())).shape[0]

    if residual_scale.shape[0] != num_layers:
        raise ValueError(
            "Residual-scale layer count does not "
            "match steering vectors: "
            f"{residual_scale.shape[0]} "
            f"!= {num_layers}"
        )

    print()

    print(
        "Residual-scale calibration:",
        (config.layout.normalization_dir() / "residual_scale.pt"),
    )

    print(
        "Layers calibrated:",
        residual_scale.shape[0],
    )

    print(
        "Bootstrap resamples:",
        f"{bootstrap_samples:,}",
    )

    print(
        "Bootstrap base seed:",
        bootstrap_seed,
    )

    candidate_map = {}

    for (
        behavior_index,
        behavior,
    ) in enumerate(
        config.behaviors,
        start=1,
    ):
        print()

        print("#" * 76)

        print(f"BEHAVIOR {behavior_index}/{len(config.behaviors)} — {behavior}")

        print("#" * 76)

        result_dir = config.layout.results_dir(behavior)

        result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset = load_behavior_validation(
            config,
            behavior,
        )

        vectors = normalized_vectors[behavior]

        if vectors.shape[0] != residual_scale.shape[0]:
            raise ValueError(
                f"{behavior}: vector layer count "
                f"{vectors.shape[0]} does not match "
                "residual-scale layer count "
                f"{residual_scale.shape[0]}."
            )

        # ============================================================
        # 1. Coarse layer sweep
        # ============================================================

        layer_path = result_dir / "layer_sweep.csv"

        layer_df = _run_or_resume_layer_sweep(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            behavior=behavior,
            dataset=dataset,
            vectors=vectors,
            residual_scale=residual_scale,
            alphas=(config.layer_sweep_alphas),
            output_path=layer_path,
            resume=resume,
            device=device,
        )

        layer_df = _add_bootstrap_cis(
            summary_path=layer_path,
            n_bootstrap=(bootstrap_samples),
            base_seed=(bootstrap_seed),
        )

        # ============================================================
        # 2. Candidate-layer window
        # ============================================================

        candidate_layers = candidate_specs.get(behavior)

        if candidate_layers is None:
            candidate_layers = infer_candidate_layers(
                layer_sweep_df=layer_df,
                num_layers=(vectors.shape[0]),
                width=(config.candidate_width),
            )

            print(f"Auto candidate window for {behavior}: {candidate_layers}")

        else:
            candidate_layers = [int(x) for x in candidate_layers]

            print(f"Configured candidate window for {behavior}: {candidate_layers}")

        invalid = [
            layer
            for layer in candidate_layers
            if (layer < 0 or layer >= vectors.shape[0])
        ]

        if invalid:
            raise ValueError(f"Invalid candidate layers for {behavior}: {invalid}")

        candidate_map[behavior] = candidate_layers

        # ============================================================
        # 3. Dense coefficient sweep
        # ============================================================

        coefficient_path = result_dir / "coefficient_layer_sweep.csv"

        coefficient_df = _run_or_resume_coefficient_sweep(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            behavior=behavior,
            dataset=dataset,
            vectors=vectors,
            residual_scale=residual_scale,
            candidate_layers=(candidate_layers),
            alphas=config.alphas,
            output_path=(coefficient_path),
            resume=resume,
            device=device,
        )

        coefficient_df = _add_bootstrap_cis(
            summary_path=(coefficient_path),
            n_bootstrap=(bootstrap_samples),
            base_seed=(bootstrap_seed),
        )

        # ============================================================
        # 4. Directional layer-selection diagnostics
        #
        # Always recompute because this is cheap and ensures resumed
        # runs use the current metric definition and current CIs.
        # ============================================================

        metrics_path = result_dir / "layer_selection_metrics.csv"

        metrics_df = compute_layer_selection_metrics(
            behavior=behavior,
            coefficient_df=(coefficient_df),
            output_path=(metrics_path),
        )

        print()

        print("Layer-selection metrics:")

        print_selection_metrics(metrics_df)

        print()

        print(f"COMPLETE — {behavior}")

    manifest = write_run_manifest(
        config=config,
        model=model,
        prompt_family=adapter.family,
        phase="analysis",
        extra={
            "candidate_layers": (candidate_map),
            "residual_scale_path": str(
                (config.layout.normalization_dir() / "residual_scale.pt")
            ),
            "eta_definition": ("|alpha| * vector_norm / residual_scale"),
            "signed_eta_definition": ("alpha * vector_norm / residual_scale"),
            "bootstrap": {
                "method": (
                    "paired percentile bootstrap over per-example behavior deltas"
                ),
                "samples": int(bootstrap_samples),
                "confidence_level": 0.95,
                "base_seed": int(bootstrap_seed),
            },
            "resume_enabled": bool(resume),
        },
    )

    print()

    print("=" * 76)

    print("PHASE-2 BEHAVIOR ANALYSIS COMPLETE")

    print("=" * 76)

    print("All configured behaviors now have:")

    print("  layer_sweep.csv")

    print("  layer_sweep_items.csv")

    print("  coefficient_layer_sweep.csv")

    print("  coefficient_layer_sweep_items.csv")

    print("  layer_selection_metrics.csv")

    print()

    print(
        "Manifest:",
        manifest,
    )

    print()

    print(
        "Next protocol step: review/freeze the "
        "behavior-selected layers, then implement "
        "the Phase-2 capability suite and usable-budget "
        "selection. Do not run the old MMLU-only "
        "finalization path."
    )


def main():
    args = parse_args()

    if args.phase == "finalize":
        raise RuntimeError(
            "Phase-2 finalization is intentionally "
            "disabled for now. The old MMLU-only "
            "finalization does not implement the new "
            "MMLU + ARC-Challenge + HellaSwag "
            "capability constraint or validation-selected "
            "usable steering budgets."
        )

    config = load_experiment_config(args.config)

    candidate_specs = parse_layer_specs(args.candidate_layer_spec)

    validate_candidate_specs(
        config,
        candidate_specs,
    )

    print("=" * 76)

    print("CAA STEERFRONTIER — PHASE 2")

    print("=" * 76)

    print(
        "Phase:",
        args.phase,
    )

    print(
        "Model:",
        config.model_label,
    )

    print(
        "HF model:",
        config.model_name,
    )

    print(
        "Behaviors:",
        config.behaviors,
    )

    print(
        "Prompt family:",
        config.prompt_family,
    )

    print("Normalization: shared mean per-layer vector norm")

    print("Calibration: residual-relative eta")

    print(
        "Vector root:",
        config.vector_root,
    )

    print(
        "Results root:",
        config.results_root,
    )

    print(
        "Resume:",
        args.resume,
    )

    print(
        "Device:",
        args.device,
    )

    print()

    model, tokenizer = load_model(
        model_name=config.model_name,
        device=args.device,
        dtype=config.torch_dtype,
    )

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=config.model_name,
        prompt_family=(config.prompt_family),
    )

    print(
        "Resolved prompt family:",
        adapter.family,
    )

    run_analysis(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        config=config,
        extract_mode=(args.extract_mode),
        candidate_specs=(candidate_specs),
        resume=args.resume,
        bootstrap_samples=(args.bootstrap_samples),
        bootstrap_seed=(args.bootstrap_seed),
        device=args.device,
    )


if __name__ == "__main__":
    main()
