import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from steerfrontier.config import ExperimentConfig
from steerfrontier.data import (
    load_ab_test_data,
    load_generation_data,
    load_mmlu_data,
)
from steerfrontier.evaluation import evaluate_ab, evaluate_mmlu_detailed
from steerfrontier.models import model_summary
from steerfrontier.normalization import normalize_behavior_vectors
from steerfrontier.prompts import PromptAdapter
from steerfrontier.vectors import extract_all_layer_vectors


def parse_layer_specs(specs: Iterable[str]) -> Dict[str, List[int]]:
    """Parse ``behavior=10-18`` or ``behavior=13`` specifications."""

    parsed: Dict[str, List[int]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid layer spec {spec!r}; expected behavior=START-END."
            )

        behavior, value = spec.split("=", 1)
        behavior = behavior.strip()
        value = value.strip()

        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid layer range {value!r}.")
            parsed[behavior] = list(range(start, end + 1))
        else:
            parsed[behavior] = [int(value)]

    return parsed


def parse_selected_layers(specs: Iterable[str]) -> Dict[str, int]:
    parsed: Dict[str, int] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid selected-layer spec {spec!r}; expected behavior=LAYER."
            )
        behavior, value = spec.split("=", 1)
        parsed[behavior.strip()] = int(value.strip())
    return parsed


def write_run_manifest(
    config: ExperimentConfig,
    model,
    prompt_family: str,
    phase: str,
    extra: Optional[dict] = None,
) -> Path:
    output = config.results_root / f"{phase}_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "phase": phase,
        "model_name": config.model_name,
        "model_label": config.model_label,
        "model_slug": config.model_slug,
        "model_summary": model_summary(model),
        "prompt_family": prompt_family,
        "behaviors": config.behaviors,
        "normalization_tag": config.normalization_tag,
        "alphas": config.alphas,
        "layer_sweep_alphas": config.layer_sweep_alphas,
        "candidate_width": config.candidate_width,
        "data_root": str(config.data_root),
        "vector_root": str(config.vector_root),
        "results_root": str(config.results_root),
        "plots_root": str(config.plots_root),
        "dtype": config.dtype,
    }
    if extra:
        payload.update(extra)

    with output.open("w") as f:
        json.dump(payload, f, indent=2)

    return output


def extract_or_load_raw_vectors(
    model,
    tokenizer,
    adapter: PromptAdapter,
    config: ExperimentConfig,
    extract_mode: str = "auto",
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Extract missing raw vectors or reuse existing model-specific vectors."""

    if extract_mode not in {"auto", "always", "never"}:
        raise ValueError("extract_mode must be auto, always, or never.")

    layout = config.layout
    raw_vectors: Dict[str, torch.Tensor] = {}

    for behavior in config.behaviors:
        output_path = layout.raw_vectors_path(behavior)
        should_extract = extract_mode == "always" or (
            extract_mode == "auto" and not output_path.exists()
        )

        if extract_mode == "never" and not output_path.exists():
            raise FileNotFoundError(f"Missing raw vector for {behavior}: {output_path}")

        if should_extract:
            dataset_path = layout.generation_path(behavior)
            dataset = load_generation_data(dataset_path)

            print("\n" + "=" * 76)
            print(f"EXTRACTION — {behavior}")
            print("=" * 76)
            print("Dataset:", dataset_path)
            print("Examples:", len(dataset))

            vectors = extract_all_layer_vectors(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                device=device,
                prompt_adapter=adapter,
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(vectors, output_path)

            metadata = {
                "behavior": behavior,
                "model_name": config.model_name,
                "model_label": config.model_label,
                "model_slug": config.model_slug,
                "prompt_family": adapter.family,
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
            print("Using existing raw vector:", output_path)

        raw_vectors[behavior] = torch.load(
            output_path,
            map_location="cpu",
        ).float()

    shapes = {tuple(v.shape) for v in raw_vectors.values()}
    if len(shapes) != 1:
        raise ValueError(f"Raw-vector shapes differ across behaviors: {shapes}")

    return raw_vectors


def build_shared_normalization(
    raw_vectors: Dict[str, torch.Tensor],
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Normalize all configured behavior vectors to one mean per-layer norm."""

    normalized, reference_norms = normalize_behavior_vectors(raw_vectors)
    layout = config.layout

    reference_path = layout.reference_norms_path(config.normalization_tag)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reference_norms, reference_path)

    for behavior, vectors in normalized.items():
        output_path = layout.normalized_vectors_path(
            behavior,
            tag=config.normalization_tag,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vectors, output_path)
        print(f"Normalized {behavior:15s} -> {output_path}")

    print("Reference layer norms ->", reference_path)
    return reference_norms, normalized


def load_normalized_vectors(config: ExperimentConfig) -> Dict[str, torch.Tensor]:
    layout = config.layout
    vectors = {}
    for behavior in config.behaviors:
        path = layout.normalized_vectors_path(
            behavior,
            tag=config.normalization_tag,
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing normalized vector for {behavior}: {path}. "
                "Run the analysis phase first."
            )
        vectors[behavior] = torch.load(path, map_location="cpu").float()
    return vectors


def run_layer_sweep(
    model,
    tokenizer,
    adapter: PromptAdapter,
    behavior: str,
    dataset,
    vectors: torch.Tensor,
    alphas: List[float],
    output_path: Path,
    device: str = "cuda",
) -> pd.DataFrame:
    """Evaluate the two coarse steering coefficients at every decoder layer."""

    baseline = evaluate_ab(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=0,
        vector=vectors[0],
        coefficient=0.0,
        device=device,
        prompt_adapter=adapter,
    )["behavior_score"]

    rows = []
    print("\n" + "=" * 76)
    print(f"LAYER SWEEP — {behavior}")
    print("=" * 76)
    print(f"B_b(0) = {baseline:.6f}")

    for layer in range(vectors.shape[0]):
        scores = []
        for alpha in alphas:
            result = evaluate_ab(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                layer_idx=layer,
                vector=vectors[layer],
                coefficient=float(alpha),
                device=device,
                prompt_adapter=adapter,
            )
            score = float(result["behavior_score"])
            scores.append(score)
            rows.append(
                {
                    "behavior": behavior,
                    "layer": layer,
                    "alpha": float(alpha),
                    "behavior_score": score,
                    "baseline": float(baseline),
                    "delta_from_baseline": score - float(baseline),
                }
            )

        effect_span = max(scores) - min(scores)
        print(
            f"L{layer:02d} | "
            + " | ".join(f"alpha={a:+g}: {s:.4f}" for a, s in zip(alphas, scores))
            + f" | span={effect_span:.4f}"
        )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def infer_candidate_layers(
    layer_sweep_df: pd.DataFrame,
    num_layers: int,
    width: int,
) -> List[int]:
    """Choose a review window around the layer with the largest coarse effect."""

    if width < 1:
        raise ValueError("candidate width must be >= 1.")

    grouped = layer_sweep_df.groupby("layer")["behavior_score"]
    spans = grouped.max() - grouped.min()
    best_layer = int(spans.idxmax())

    width = min(int(width), int(num_layers))
    start = best_layer - width // 2
    end = start + width

    if start < 0:
        start = 0
        end = width
    if end > num_layers:
        end = num_layers
        start = end - width

    return list(range(start, end))


def run_coefficient_sweep(
    model,
    tokenizer,
    adapter: PromptAdapter,
    behavior: str,
    dataset,
    vectors: torch.Tensor,
    candidate_layers: List[int],
    alphas: List[float],
    output_path: Path,
    device: str = "cuda",
) -> pd.DataFrame:
    if not candidate_layers:
        raise ValueError("candidate_layers cannot be empty.")

    baseline = evaluate_ab(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=candidate_layers[0],
        vector=vectors[candidate_layers[0]],
        coefficient=0.0,
        device=device,
        prompt_adapter=adapter,
    )["behavior_score"]

    rows = []
    print("\n" + "=" * 76)
    print(f"COEFFICIENT SWEEP — {behavior}")
    print("=" * 76)
    print("Candidate layers:", candidate_layers)
    print("Alphas:", alphas)
    print(f"B_b(0) = {baseline:.6f}")

    for layer in candidate_layers:
        for alpha in alphas:
            if float(alpha) == 0.0:
                score = float(baseline)
            else:
                score = float(
                    evaluate_ab(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        layer_idx=layer,
                        vector=vectors[layer],
                        coefficient=float(alpha),
                        device=device,
                        prompt_adapter=adapter,
                    )["behavior_score"]
                )

            rows.append(
                {
                    "behavior": behavior,
                    "layer": int(layer),
                    "alpha": float(alpha),
                    "behavior_score": score,
                    "baseline": float(baseline),
                    "delta_from_baseline": score - float(baseline),
                }
            )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def compute_layer_selection_metrics(
    behavior: str,
    coefficient_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Compute descriptive layer-selection metrics without selecting a layer."""

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
        if neg.empty or pos.empty:
            raise ValueError(
                "Coefficient sweep must contain both negative and positive alphas."
            )

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
    return metrics


def print_selection_metrics(metrics: pd.DataFrame) -> None:
    print(
        metrics.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def run_mmlu_sweep(
    model,
    tokenizer,
    adapter: PromptAdapter,
    behavior: str,
    layer: int,
    vector: torch.Tensor,
    dataset,
    alphas: List[float],
    summary_path: Path,
    items_path: Optional[Path],
    device: str = "cuda",
) -> pd.DataFrame:
    rows = []
    item_rows = []

    print("\n" + "=" * 76)
    print(f"MMLU SWEEP — {behavior}, layer {layer}")
    print("=" * 76)
    print(f"Vector norm: {vector.norm().item():.6f}")

    for alpha in alphas:
        result = evaluate_mmlu_detailed(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            layer_idx=layer,
            vector=vector,
            coefficient=float(alpha),
            device=device,
            prompt_adapter=adapter,
        )

        capability = float(result["capability_score"])
        accuracy = float(result["accuracy"])
        rows.append(
            {
                "behavior": behavior,
                "layer": int(layer),
                "alpha": float(alpha),
                "capability_score": capability,
                "accuracy": accuracy,
            }
        )

        if items_path is not None:
            for index, record in enumerate(result["items"]):
                item_rows.append(
                    {
                        "behavior": behavior,
                        "layer": int(layer),
                        "alpha": float(alpha),
                        "item_index": index,
                        **record,
                    }
                )

        print(
            f"alpha={alpha:+g} | C_b(alpha)={capability:.6f} | accuracy={accuracy:.4f}"
        )

    baseline_rows = [row for row in rows if row["alpha"] == 0.0]
    if len(baseline_rows) != 1:
        raise ValueError("Alphas must contain exactly one 0.0 baseline.")

    baseline = float(baseline_rows[0]["capability_score"])
    for row in rows:
        absolute_drop = baseline - row["capability_score"]
        row["baseline"] = baseline
        row["absolute_drop"] = absolute_drop
        row["relative_drop"] = absolute_drop / baseline

    df = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)

    if items_path is not None:
        pd.DataFrame(item_rows).to_csv(items_path, index=False)

    return df


def write_behavior_summary(
    config: ExperimentConfig,
    behavior: str,
    selected_layer: int,
    candidate_layers: List[int],
    coefficient_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    mmlu_df: pd.DataFrame,
    output_path: Path,
) -> dict:
    selected_coeff = coefficient_df[coefficient_df["layer"] == selected_layer]
    selection_rows = metrics_df[metrics_df["layer"] == selected_layer]
    if selected_coeff.empty or selection_rows.empty:
        raise ValueError(
            f"Selected layer {selected_layer} is missing from saved analysis for {behavior}."
        )

    selection_row = selection_rows.iloc[0]
    behavior_curve = selected_coeff.sort_values("alpha")
    capability_curve = mmlu_df.sort_values("alpha")

    summary = {
        "behavior": behavior,
        "model_name": config.model_name,
        "model_label": config.model_label,
        "model_slug": config.model_slug,
        "normalization": config.normalization_tag,
        "selected_layer": int(selected_layer),
        "candidate_layers": [int(x) for x in candidate_layers],
        "baseline_behavior": float(
            behavior_curve[behavior_curve["alpha"] == 0.0]["behavior_score"].iloc[0]
        ),
        "baseline_capability": float(
            capability_curve[capability_curve["alpha"] == 0.0]["capability_score"].iloc[
                0
            ]
        ),
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
        "behavior_curve": behavior_curve[
            ["alpha", "behavior_score", "delta_from_baseline"]
        ].to_dict(orient="records"),
        "capability_curve": capability_curve[
            [
                "alpha",
                "capability_score",
                "accuracy",
                "absolute_drop",
                "relative_drop",
            ]
        ].to_dict(orient="records"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_saved_analysis(
    config: ExperimentConfig,
    behavior: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[int]]:
    result_dir = config.layout.results_dir(behavior)
    layer_path = result_dir / "layer_sweep.csv"
    coefficient_path = result_dir / "coefficient_layer_sweep.csv"
    metrics_path = result_dir / "layer_selection_metrics.csv"

    missing = [
        path
        for path in [layer_path, coefficient_path, metrics_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing saved analysis for {behavior}: {missing}. "
            "Run --phase analyze first."
        )

    layer_df = pd.read_csv(layer_path)
    coefficient_df = pd.read_csv(coefficient_path)
    metrics_df = pd.read_csv(metrics_path)
    candidate_layers = sorted(coefficient_df["layer"].astype(int).unique().tolist())
    return layer_df, coefficient_df, metrics_df, candidate_layers


def load_behavior_test(config: ExperimentConfig, behavior: str):
    return load_ab_test_data(config.layout.test_ab_path(behavior))


def load_mmlu(config: ExperimentConfig):
    return load_mmlu_data(config.layout.mmlu_path())
