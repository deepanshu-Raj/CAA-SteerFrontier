import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from steerfrontier.data import load_ab_test_data, load_generation_data
from steerfrontier.evaluation import evaluate_ab, evaluate_mmlu_detailed
from steerfrontier.models import load_model
from steerfrontier.vectors import extract_all_layer_vectors

DEFAULT_BEHAVIORS = [
    "sycophancy",
    "refusal",
    "hallucination",
]

DEFAULT_ALPHAS = [
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full CAA behavior pipeline: optional raw-vector extraction, "
            "shared behavior-space normalization, layer sweep, coefficient sweep, "
            "layer selection, MMLU capability sweep, and summary generation."
        )
    )

    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=DEFAULT_BEHAVIORS,
    )

    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-2-7b-chat-hf",
    )

    parser.add_argument(
        "--model-label",
        default="Llama 2 7B Chat",
    )

    parser.add_argument(
        "--model-slug",
        default="llama2-7b-chat",
    )

    parser.add_argument(
        "--data-root",
        default="data",
    )

    parser.add_argument(
        "--vector-root",
        default="vectors",
        help=(
            "Root containing <behavior>/raw_vectors.pt. For the existing Llama-2 "
            "run, use 'vectors'. For another model, use a model-specific root."
        ),
    )

    parser.add_argument(
        "--results-root",
        default="results/normalization_3beh",
        help="Output root for this pipeline run; existing primary results are untouched.",
    )

    parser.add_argument(
        "--normalization-tag",
        default="3beh",
        help="Tag used in normalized-vector filenames.",
    )

    parser.add_argument(
        "--extract-mode",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "auto: extract only missing raw vectors; always: re-extract all; "
            "never: require existing raw vectors."
        ),
    )

    parser.add_argument(
        "--candidate-layer-spec",
        nargs="*",
        default=[],
        metavar="BEHAVIOR=START-END",
        help=(
            "Optional per-behavior coefficient-sweep ranges, e.g. "
            "sycophancy=10-18 refusal=9-16 hallucination=10-15. "
            "If omitted for a behavior, a window is inferred around the strongest "
            "alpha=+1 vs -1 layer-sweep span."
        ),
    )

    parser.add_argument(
        "--auto-candidate-width",
        type=int,
        default=9,
        help="Number of candidate layers if a range is inferred automatically.",
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
    )

    parser.add_argument(
        "--selection-mode",
        choices=["manual", "auto"],
        default="manual",
        help=(
            "Layer-selection policy. 'manual' (default) computes and saves all "
            "layer-selection metrics, then stops before MMLU unless every behavior "
            "has a --selected-layer-spec. 'auto' uses the heuristic selector for "
            "behaviors without an explicit override."
        ),
    )

    parser.add_argument(
        "--selection-gain-fraction",
        type=float,
        default=0.50,
        help=(
            "Used only with --selection-mode auto. The heuristic first looks for "
            "directionally consistent layers whose bidirectional gain is at least "
            "this fraction of the maximum. Default: 0.50."
        ),
    )

    parser.add_argument(
        "--selected-layer-spec",
        nargs="*",
        default=[],
        metavar="BEHAVIOR=LAYER",
        help=(
            "Authoritative per-behavior layer choices, e.g. "
            "sycophancy=16 refusal=13 hallucination=13. In manual mode, MMLU "
            "runs only after a layer is supplied for every behavior."
        ),
    )

    parser.add_argument(
        "--reuse-analysis",
        action="store_true",
        help=(
            "Reuse existing layer_sweep.csv, coefficient_layer_sweep.csv, and "
            "layer_selection_metrics.csv from --results-root instead of rerunning "
            "those GPU sweeps. Intended for resuming after manual layer selection."
        ),
    )

    parser.add_argument(
        "--mmlu-path",
        default="data/mmlu/test.json",
    )

    parser.add_argument(
        "--skip-mmlu-items",
        action="store_true",
        help="Do not save per-example MMLU records.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------


def generation_path(data_root: Path, behavior: str) -> Path:
    return data_root / behavior / "generate.json"


def test_ab_path(data_root: Path, behavior: str) -> Path:
    return data_root / behavior / "test_ab.json"


def raw_vector_path(vector_root: Path, behavior: str) -> Path:
    return vector_root / behavior / "raw_vectors.pt"


def normalized_vector_path(
    vector_root: Path,
    behavior: str,
    normalization_tag: str,
) -> Path:
    return vector_root / behavior / f"normalized_vectors_{normalization_tag}.pt"


def behavior_result_dir(results_root: Path, behavior: str) -> Path:
    return results_root / behavior


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------


def parse_candidate_specs(specs: List[str]) -> Dict[str, List[int]]:
    parsed = {}

    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid candidate-layer spec '{spec}'. Expected behavior=start-end."
            )

        behavior, value = spec.split("=", 1)
        behavior = behavior.strip()
        value = value.strip()

        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid layer range '{value}'.")
            layers = list(range(start, end + 1))
        else:
            layers = [int(value)]

        parsed[behavior] = layers

    return parsed


def parse_selected_specs(specs: List[str]) -> Dict[str, int]:
    parsed = {}

    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid selected-layer spec '{spec}'. Expected behavior=layer."
            )

        behavior, value = spec.split("=", 1)
        parsed[behavior.strip()] = int(value.strip())

    return parsed


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------


def ensure_raw_vectors(
    model,
    tokenizer,
    behaviors: List[str],
    data_root: Path,
    vector_root: Path,
    model_name: str,
    extract_mode: str,
) -> Dict[str, torch.Tensor]:
    raw_vectors = {}

    for behavior in behaviors:
        output_path = raw_vector_path(vector_root, behavior)

        should_extract = extract_mode == "always" or (
            extract_mode == "auto" and not output_path.exists()
        )

        if extract_mode == "never" and not output_path.exists():
            raise FileNotFoundError(f"Missing raw vector for {behavior}: {output_path}")

        if should_extract:
            dataset_path = generation_path(data_root, behavior)
            dataset = load_generation_data(dataset_path)

            print("\n" + "=" * 72)
            print(f"EXTRACTION: {behavior}")
            print("=" * 72)
            print("Dataset:", dataset_path)
            print("Examples:", len(dataset))

            vectors = extract_all_layer_vectors(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(vectors, output_path)

            metadata = {
                "behavior": behavior,
                "model": model_name,
                "dataset": str(dataset_path),
                "num_examples": len(dataset),
                "num_layers": int(vectors.shape[0]),
                "hidden_size": int(vectors.shape[1]),
                "dtype": str(vectors.dtype),
                "vector_definition": "mean(h_matching - h_not_matching)",
            }

            with (output_path.parent / "metadata.json").open("w") as f:
                json.dump(metadata, f, indent=2)

            print("Saved:", output_path)
        else:
            print(f"Using existing raw vector: {output_path}")

        raw_vectors[behavior] = torch.load(
            output_path,
            map_location="cpu",
        ).float()

    shapes = {tuple(v.shape) for v in raw_vectors.values()}
    if len(shapes) != 1:
        raise ValueError(f"Raw-vector shapes differ across behaviors: {shapes}")

    return raw_vectors


# ---------------------------------------------------------------------
# Shared behavior-space normalization
# ---------------------------------------------------------------------


def build_shared_normalization(
    raw_vectors: Dict[str, torch.Tensor],
    behaviors: List[str],
    vector_root: Path,
    normalization_tag: str,
    model_slug: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    raw_norms = torch.stack(
        [raw_vectors[b].norm(dim=1) for b in behaviors],
        dim=0,
    )

    reference_norms = raw_norms.mean(dim=0)

    normalization_dir = vector_root / "normalization"
    normalization_dir.mkdir(parents=True, exist_ok=True)

    reference_path = (
        normalization_dir / f"{model_slug}_{normalization_tag}_reference_layer_norms.pt"
    )

    torch.save(reference_norms, reference_path)

    normalized = {}

    print("\n" + "=" * 72)
    print("SHARED NORMALIZATION")
    print("=" * 72)
    print("Behaviors:", ", ".join(behaviors))
    print("Reference:", reference_path)

    for behavior in behaviors:
        raw = raw_vectors[behavior]
        norms = raw.norm(dim=1, keepdim=True)

        if torch.any(norms == 0):
            raise ValueError(f"Zero-norm raw vector found for {behavior}.")

        normalized_behavior = raw * reference_norms.unsqueeze(1) / norms
        normalized[behavior] = normalized_behavior

        output_path = normalized_vector_path(
            vector_root,
            behavior,
            normalization_tag,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(normalized_behavior, output_path)

        print(f"{behavior:15s} -> {output_path}")

    return reference_norms, normalized


# ---------------------------------------------------------------------
# Layer sweep
# ---------------------------------------------------------------------


def run_layer_sweep(
    model,
    tokenizer,
    behavior: str,
    dataset,
    vectors: torch.Tensor,
    output_path: Path,
) -> pd.DataFrame:
    baseline_result = evaluate_ab(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=0,
        vector=vectors[0],
        coefficient=0.0,
    )

    baseline = baseline_result["behavior_score"]
    rows = []

    print("\n" + "=" * 72)
    print(f"LAYER SWEEP: {behavior}")
    print("=" * 72)
    print(f"Baseline B(0) = {baseline:.6f}")

    for layer in range(vectors.shape[0]):
        scores = {}

        for alpha in [-1.0, 1.0]:
            result = evaluate_ab(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                layer_idx=layer,
                vector=vectors[layer],
                coefficient=alpha,
            )

            score = result["behavior_score"]
            scores[alpha] = score

            rows.append(
                {
                    "behavior": behavior,
                    "layer": layer,
                    "alpha": alpha,
                    "behavior_score": score,
                    "baseline": baseline,
                    "delta_from_baseline": score - baseline,
                }
            )

        span = scores[1.0] - scores[-1.0]

        print(
            f"L{layer:02d} | -1={scores[-1.0]:.4f} | "
            f"0={baseline:.4f} | +1={scores[1.0]:.4f} | span={span:+.4f}"
        )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


# ---------------------------------------------------------------------
# Candidate-layer inference
# ---------------------------------------------------------------------


def infer_candidate_layers(
    layer_sweep_df: pd.DataFrame,
    num_layers: int,
    width: int,
) -> List[int]:
    if width < 1:
        raise ValueError("auto candidate width must be >= 1")

    pivot = layer_sweep_df.pivot(
        index="layer",
        columns="alpha",
        values="behavior_score",
    )

    pivot["span"] = pivot[1.0] - pivot[-1.0]
    best_layer = int(pivot["span"].idxmax())

    width = min(width, num_layers)
    start = best_layer - width // 2
    end = start + width

    if start < 0:
        start = 0
        end = width

    if end > num_layers:
        end = num_layers
        start = end - width

    return list(range(start, end))


# ---------------------------------------------------------------------
# Coefficient sweep
# ---------------------------------------------------------------------


def run_coefficient_sweep(
    model,
    tokenizer,
    behavior: str,
    dataset,
    vectors: torch.Tensor,
    candidate_layers: List[int],
    alphas: List[float],
    output_path: Path,
) -> pd.DataFrame:
    baseline_result = evaluate_ab(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=candidate_layers[0],
        vector=vectors[candidate_layers[0]],
        coefficient=0.0,
    )

    baseline = baseline_result["behavior_score"]
    rows = []

    print("\n" + "=" * 72)
    print(f"COEFFICIENT SWEEP: {behavior}")
    print("=" * 72)
    print("Candidate layers:", candidate_layers)
    print("Alphas:", alphas)
    print(f"Baseline B(0) = {baseline:.6f}")

    for layer in candidate_layers:
        print(f"\n--- Layer {layer} ---")

        for alpha in alphas:
            if alpha == 0.0:
                score = baseline
            else:
                result = evaluate_ab(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    layer_idx=layer,
                    vector=vectors[layer],
                    coefficient=alpha,
                )
                score = result["behavior_score"]

            rows.append(
                {
                    "behavior": behavior,
                    "layer": layer,
                    "alpha": alpha,
                    "behavior_score": score,
                    "baseline": baseline,
                    "delta_from_baseline": score - baseline,
                }
            )

            print(
                f"L{layer:02d} | alpha={alpha:+.1f} | "
                f"B={score:.4f} | delta={score - baseline:+.4f}"
            )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


# ---------------------------------------------------------------------
# Layer-selection metrics
# ---------------------------------------------------------------------


def compute_layer_selection_metrics(
    behavior: str,
    coefficient_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    rows = []

    for layer in sorted(coefficient_df["layer"].unique()):
        sub = (
            coefficient_df[coefficient_df["layer"] == layer]
            .sort_values("alpha")
            .reset_index(drop=True)
        )

        baseline = float(sub["baseline"].iloc[0])
        neg = sub[sub["alpha"] < 0]
        pos = sub[sub["alpha"] > 0]

        negative_gain = baseline - float(neg["behavior_score"].min())
        positive_gain = float(pos["behavior_score"].max()) - baseline
        bidirectional_gain = negative_gain + positive_gain

        mean_negative_gain = float((baseline - neg["behavior_score"]).mean())
        mean_positive_gain = float((pos["behavior_score"] - baseline).mean())

        negative_direction_fraction = float((neg["behavior_score"] < baseline).mean())
        positive_direction_fraction = float((pos["behavior_score"] > baseline).mean())

        rho, _ = spearmanr(sub["alpha"], sub["behavior_score"])

        scores = sub["behavior_score"].to_numpy()
        monotonic_violations = int(np.sum(np.diff(scores) < -1e-6))
        total_range = float(sub["behavior_score"].max() - sub["behavior_score"].min())

        rows.append(
            {
                "behavior": behavior,
                "layer": int(layer),
                "spearman_rho": float(rho),
                "monotonic_violations": monotonic_violations,
                "negative_gain": negative_gain,
                "positive_gain": positive_gain,
                "bidirectional_gain": bidirectional_gain,
                "mean_negative_gain": mean_negative_gain,
                "mean_positive_gain": mean_positive_gain,
                "negative_direction_fraction": negative_direction_fraction,
                "positive_direction_fraction": positive_direction_fraction,
                "total_range": total_range,
            }
        )

    metrics = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)

    print("\nLayer-selection metrics:")
    print(
        metrics.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return metrics


# ---------------------------------------------------------------------
# Auto layer selection
# ---------------------------------------------------------------------


def auto_select_layer(
    metrics: pd.DataFrame,
    gain_fraction: float,
) -> Tuple[int, pd.DataFrame]:
    max_gain = float(metrics["bidirectional_gain"].max())
    threshold = gain_fraction * max_gain

    eligible = metrics[
        (metrics["negative_direction_fraction"] >= 0.999)
        & (metrics["positive_direction_fraction"] >= 0.999)
        & (metrics["bidirectional_gain"] >= threshold)
    ].copy()

    if eligible.empty:
        eligible = metrics[metrics["bidirectional_gain"] >= threshold].copy()

    if eligible.empty:
        eligible = metrics.copy()

    eligible = eligible.sort_values(
        by=[
            "spearman_rho",
            "monotonic_violations",
            "bidirectional_gain",
        ],
        ascending=[False, True, False],
    )

    selected_layer = int(eligible.iloc[0]["layer"])
    return selected_layer, eligible


# ---------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------


def run_mmlu_sweep(
    model,
    tokenizer,
    behavior: str,
    layer: int,
    vector: torch.Tensor,
    mmlu_dataset,
    alphas: List[float],
    summary_path: Path,
    items_path: Optional[Path],
) -> pd.DataFrame:
    rows = []
    item_rows = []

    print("\n" + "=" * 72)
    print(f"MMLU SWEEP: {behavior}, layer {layer}")
    print("=" * 72)
    print(f"Vector norm: {vector.norm().item():.6f}")

    for alpha in alphas:
        result = evaluate_mmlu_detailed(
            model=model,
            tokenizer=tokenizer,
            dataset=mmlu_dataset,
            layer_idx=layer,
            vector=vector,
            coefficient=alpha,
        )

        capability = float(result["capability_score"])
        accuracy = float(result["accuracy"])

        rows.append(
            {
                "behavior": behavior,
                "layer": layer,
                "alpha": alpha,
                "capability_score": capability,
                "accuracy": accuracy,
            }
        )

        print(
            f"alpha={alpha:+.1f} | C(alpha)={capability:.6f} | accuracy={accuracy:.4f}"
        )

        if items_path is not None:
            records = result.get("records")
            if records is None:
                records = result.get("items")
            if records is None:
                raise KeyError(
                    "evaluate_mmlu_detailed() returned neither 'records' nor 'items'."
                )

            for index, record in enumerate(records):
                item_rows.append(
                    {
                        "behavior": behavior,
                        "layer": layer,
                        "alpha": alpha,
                        "item_index": index,
                        **record,
                    }
                )

    baseline_rows = [row for row in rows if row["alpha"] == 0.0]
    if len(baseline_rows) != 1:
        raise ValueError("Alpha list must contain exactly one 0.0 baseline.")

    baseline = baseline_rows[0]["capability_score"]

    for row in rows:
        absolute_drop = baseline - row["capability_score"]
        row["baseline"] = baseline
        row["absolute_drop"] = absolute_drop
        row["relative_drop"] = absolute_drop / baseline

    summary_df = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    if items_path is not None and item_rows:
        items_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(item_rows).to_csv(items_path, index=False)

    return summary_df


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def write_behavior_summary(
    behavior: str,
    model_name: str,
    model_label: str,
    normalization_tag: str,
    selected_layer: int,
    candidate_layers: List[int],
    coefficient_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    mmlu_df: pd.DataFrame,
    output_path: Path,
):
    selected_coeff = coefficient_df[coefficient_df["layer"] == selected_layer]
    selection_row = metrics_df[metrics_df["layer"] == selected_layer].iloc[0]

    baseline_behavior = float(
        selected_coeff[selected_coeff["alpha"] == 0.0]["behavior_score"].iloc[0]
    )

    baseline_capability = float(
        mmlu_df[mmlu_df["alpha"] == 0.0]["capability_score"].iloc[0]
    )

    summary = {
        "behavior": behavior,
        "model_name": model_name,
        "model_label": model_label,
        "normalization": normalization_tag,
        "selected_layer": selected_layer,
        "candidate_layers": candidate_layers,
        "baseline_behavior": baseline_behavior,
        "baseline_capability": baseline_capability,
        "alphas": sorted(coefficient_df["alpha"].unique().tolist()),
        "layer_selection": {
            "spearman_rho": float(selection_row["spearman_rho"]),
            "monotonic_violations": int(selection_row["monotonic_violations"]),
            "negative_gain": float(selection_row["negative_gain"]),
            "positive_gain": float(selection_row["positive_gain"]),
            "bidirectional_gain": float(selection_row["bidirectional_gain"]),
            "mean_negative_gain": float(selection_row["mean_negative_gain"]),
            "mean_positive_gain": float(selection_row["mean_positive_gain"]),
            "negative_direction_fraction": float(
                selection_row["negative_direction_fraction"]
            ),
            "positive_direction_fraction": float(
                selection_row["positive_direction_fraction"]
            ),
            "total_range": float(selection_row["total_range"]),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    vector_root = Path(args.vector_root)
    results_root = Path(args.results_root)
    mmlu_path = Path(args.mmlu_path)

    candidate_specs = parse_candidate_specs(args.candidate_layer_spec)
    selected_specs = parse_selected_specs(args.selected_layer_spec)

    unknown_selected = sorted(set(selected_specs) - set(args.behaviors))
    if unknown_selected:
        raise ValueError(
            "--selected-layer-spec contains behaviors not present in --behaviors: "
            f"{unknown_selected}"
        )

    if 0.0 not in args.alphas:
        raise ValueError("--alphas must include 0.0.")

    print("=" * 72)
    print("CAA FULL PIPELINE")
    print("=" * 72)
    print("Model:", args.model_label)
    print("Model name:", args.model_name)
    print("Behaviors:", args.behaviors)
    print("Normalization tag:", args.normalization_tag)
    print("Vector root:", vector_root)
    print("Results root:", results_root)
    print("Extraction mode:", args.extract_mode)
    print("Selection mode:", args.selection_mode)
    print("Reuse analysis:", args.reuse_analysis)
    print("Alphas:", args.alphas)
    print()

    # -------------------------------------------------------------
    # Load model once. Even a resume run needs it for MMLU.
    # -------------------------------------------------------------
    print("Loading model...")
    model, tokenizer = load_model(
        model_name=args.model_name,
        dtype=torch.float32,
    )

    # -------------------------------------------------------------
    # Stage 1: raw-vector extraction / reuse.
    # -------------------------------------------------------------
    raw_vectors = ensure_raw_vectors(
        model=model,
        tokenizer=tokenizer,
        behaviors=args.behaviors,
        data_root=data_root,
        vector_root=vector_root,
        model_name=args.model_name,
        extract_mode=args.extract_mode,
    )

    # -------------------------------------------------------------
    # Stage 2: shared N-behavior normalization.
    # -------------------------------------------------------------
    reference_norms, normalized_vectors = build_shared_normalization(
        raw_vectors=raw_vectors,
        behaviors=args.behaviors,
        vector_root=vector_root,
        normalization_tag=args.normalization_tag,
        model_slug=args.model_slug,
    )

    print("\nReference norms at selected diagnostic layers:")
    for layer in [10, 11, 12, 13, 14, 15, 16]:
        if layer < len(reference_norms):
            print(f"L{layer:02d}: {reference_norms[layer].item():.6f}")

    # -------------------------------------------------------------
    # Phase A: behavior analysis for ALL behaviors.
    #
    # We intentionally finish layer/coefficient analysis for every
    # behavior before deciding whether to proceed to MMLU. This lets
    # manual selection happen once, after reviewing the full set.
    # -------------------------------------------------------------
    analysis = {}

    for behavior in args.behaviors:
        behavior_dir = behavior_result_dir(results_root, behavior)
        behavior_dir.mkdir(parents=True, exist_ok=True)

        vectors = normalized_vectors[behavior]

        layer_sweep_path = behavior_dir / "layer_sweep.csv"
        coefficient_path = behavior_dir / "coefficient_layer_sweep.csv"
        metrics_path = behavior_dir / "layer_selection_metrics.csv"

        if args.reuse_analysis:
            missing = [
                path
                for path in [layer_sweep_path, coefficient_path, metrics_path]
                if not path.exists()
            ]
            if missing:
                raise FileNotFoundError(
                    "--reuse-analysis was requested, but the following files are "
                    f"missing for {behavior}: {missing}"
                )

            print("\n" + "=" * 72)
            print(f"REUSING SAVED ANALYSIS: {behavior}")
            print("=" * 72)

            layer_sweep_df = pd.read_csv(layer_sweep_path)
            coefficient_df = pd.read_csv(coefficient_path)
            metrics_df = pd.read_csv(metrics_path)
            candidate_layers = sorted(
                coefficient_df["layer"].astype(int).unique().tolist()
            )

            print("Layer sweep:", layer_sweep_path)
            print("Coefficient sweep:", coefficient_path)
            print("Selection metrics:", metrics_path)
            print("Candidate layers:", candidate_layers)

        else:
            test_dataset = load_ab_test_data(test_ab_path(data_root, behavior))

            # Stage 3: layer sweep.
            layer_sweep_df = run_layer_sweep(
                model=model,
                tokenizer=tokenizer,
                behavior=behavior,
                dataset=test_dataset,
                vectors=vectors,
                output_path=layer_sweep_path,
            )

            # Candidate region: explicit spec or inferred from layer sweep.
            candidate_layers = candidate_specs.get(behavior)

            if candidate_layers is None:
                candidate_layers = infer_candidate_layers(
                    layer_sweep_df=layer_sweep_df,
                    num_layers=vectors.shape[0],
                    width=args.auto_candidate_width,
                )
                print(f"\nAuto candidate layers for {behavior}: {candidate_layers}")
            else:
                print(
                    f"\nConfigured candidate layers for {behavior}: {candidate_layers}"
                )

            invalid_layers = [
                layer
                for layer in candidate_layers
                if layer < 0 or layer >= vectors.shape[0]
            ]
            if invalid_layers:
                raise ValueError(
                    f"Invalid candidate layers for {behavior}: {invalid_layers}"
                )

            # Stage 4: coefficient sweep.
            coefficient_df = run_coefficient_sweep(
                model=model,
                tokenizer=tokenizer,
                behavior=behavior,
                dataset=test_dataset,
                vectors=vectors,
                candidate_layers=candidate_layers,
                alphas=args.alphas,
                output_path=coefficient_path,
            )

            # Stage 5: layer-selection metrics.
            metrics_df = compute_layer_selection_metrics(
                behavior=behavior,
                coefficient_df=coefficient_df,
                output_path=metrics_path,
            )

        analysis[behavior] = {
            "candidate_layers": candidate_layers,
            "coefficient_df": coefficient_df,
            "metrics_df": metrics_df,
            "vectors": vectors,
            "behavior_dir": behavior_dir,
        }

    # -------------------------------------------------------------
    # Stage 5b: resolve layer choices.
    # -------------------------------------------------------------
    selected_layers = {}
    missing_manual = []

    for behavior in args.behaviors:
        metrics_df = analysis[behavior]["metrics_df"]

        if behavior in selected_specs:
            selected_layer = int(selected_specs[behavior])
            print(f"\nMANUAL SELECTED {behavior}: L{selected_layer}")

        elif args.selection_mode == "auto":
            selected_layer, eligible = auto_select_layer(
                metrics=metrics_df,
                gain_fraction=args.selection_gain_fraction,
            )

            print(f"\nAUTO SELECTED {behavior}: L{selected_layer}")
            print(
                "Auto-selection is a convenience heuristic; inspect the saved "
                "metrics before using it as a reported experimental choice."
            )
            print(
                eligible[
                    [
                        "layer",
                        "spearman_rho",
                        "monotonic_violations",
                        "bidirectional_gain",
                        "negative_direction_fraction",
                        "positive_direction_fraction",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

        else:
            missing_manual.append(behavior)
            continue

        if selected_layer not in set(metrics_df["layer"].astype(int)):
            raise ValueError(
                f"Selected layer L{selected_layer} for {behavior} was not included "
                "in the coefficient sweep."
            )

        selected_layers[behavior] = selected_layer

    # -------------------------------------------------------------
    # Manual mode deliberately stops here until all layer choices
    # have been supplied. No MMLU compute is spent on an accidental
    # auto-selected layer.
    # -------------------------------------------------------------
    if missing_manual:
        print("\n" + "=" * 72)
        print("ANALYSIS COMPLETE — MANUAL LAYER SELECTION REQUIRED")
        print("=" * 72)
        print("No MMLU sweep was run for behaviors without an explicit layer.")
        print("\nReview these files:")

        for behavior in missing_manual:
            behavior_dir = analysis[behavior]["behavior_dir"]
            print(f"  {behavior:15s} {behavior_dir / 'layer_selection_metrics.csv'}")
            print(f"  {'':15s} {behavior_dir / 'coefficient_layer_sweep.csv'}")

        print("\nThen resume without rerunning the sweeps:")
        print(
            "  python scripts/run_experiment_pipeline.py <same arguments> "
            "--reuse-analysis --selection-mode manual "
            "--selected-layer-spec " + " ".join(f"{b}=LAYER" for b in args.behaviors)
        )
        print()
        return

    # -------------------------------------------------------------
    # Phase B: MMLU + summary after layer choices are resolved.
    # -------------------------------------------------------------
    with mmlu_path.open() as f:
        mmlu_dataset = json.load(f)

    print("\nMMLU examples:", len(mmlu_dataset))

    for behavior in args.behaviors:
        selected_layer = selected_layers[behavior]
        record = analysis[behavior]
        behavior_dir = record["behavior_dir"]
        vectors = record["vectors"]
        coefficient_df = record["coefficient_df"]
        metrics_df = record["metrics_df"]
        candidate_layers = record["candidate_layers"]

        # Stage 6: MMLU at selected layer.
        mmlu_summary_path = behavior_dir / f"mmlu_layer{selected_layer}.csv"
        mmlu_items_path = None
        if not args.skip_mmlu_items:
            mmlu_items_path = behavior_dir / f"mmlu_layer{selected_layer}_items.csv"

        mmlu_df = run_mmlu_sweep(
            model=model,
            tokenizer=tokenizer,
            behavior=behavior,
            layer=selected_layer,
            vector=vectors[selected_layer],
            mmlu_dataset=mmlu_dataset,
            alphas=args.alphas,
            summary_path=mmlu_summary_path,
            items_path=mmlu_items_path,
        )

        # Stage 7: summary.
        write_behavior_summary(
            behavior=behavior,
            model_name=args.model_name,
            model_label=args.model_label,
            normalization_tag=args.normalization_tag,
            selected_layer=selected_layer,
            candidate_layers=candidate_layers,
            coefficient_df=coefficient_df,
            metrics_df=metrics_df,
            mmlu_df=mmlu_df,
            output_path=behavior_dir / "summary.json",
        )

        print(f"\nCompleted behavior: {behavior}")
        print("Results:", behavior_dir)

    # -------------------------------------------------------------
    # Run-level manifest.
    # -------------------------------------------------------------
    manifest = {
        "model_name": args.model_name,
        "model_label": args.model_label,
        "model_slug": args.model_slug,
        "behaviors": args.behaviors,
        "normalization_tag": args.normalization_tag,
        "normalization_reference": (
            "mean raw vector norm across behaviors at each layer"
        ),
        "selection_mode": args.selection_mode,
        "selected_layers": selected_layers,
        "alphas": args.alphas,
        "results_root": str(results_root),
        "vector_root": str(vector_root),
    }

    results_root.mkdir(parents=True, exist_ok=True)
    with (results_root / "run_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print("Selected layers:", selected_layers)
    print("Manifest:", results_root / "run_manifest.json")


if __name__ == "__main__":
    main()
