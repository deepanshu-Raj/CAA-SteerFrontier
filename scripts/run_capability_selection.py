"""Evaluate frozen behavior-selected layers on capability-selection benchmarks.

Capability results determine feasible steering budgets but never behavior-layer
selection; holdout data are not consulted by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from steerfrontier.config import (
    load_experiment_config,
)
from steerfrontier.data import (
    load_arc_challenge_data,
    load_hellaswag_data,
    load_mmlu_data,
)
from steerfrontier.evaluation import (
    evaluate_arc_challenge,
    evaluate_hellaswag,
    evaluate_mmlu,
)
from steerfrontier.models import (
    load_model,
)
from steerfrontier.prompts import (
    get_prompt_adapter,
)
from steerfrontier.statistics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_mean_ci,
)
from steerfrontier.workflow import (
    load_normalized_vectors,
)

# =============================================================================
# Configuration
# =============================================================================

CORE_BENCHMARKS = (
    "mmlu",
    "arc_challenge",
    "hellaswag",
)

DEFAULT_TOLERANCES = (
    0.01,
    0.03,
    0.05,
)


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run capability-selection sweeps at frozen behavior-selected layers."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--selected-layers",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--capability-root",
        type=Path,
        default=Path("data/capability"),
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Inference batch size. "
            "Recommended: 8-16 on L40S, "
            "32 on H100 for FP32 Llama-2-7B."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    return parser.parse_args()


def stable_seed(
    base_seed: int,
    benchmark: str,
    behavior: str,
    alpha: float,
) -> int:
    payload = (f"{base_seed}:{benchmark}:{behavior}:{alpha:.12g}").encode()

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
    )


def load_selected_layers(
    path: Path,
    expected_behaviors,
):
    with path.open("r") as f:
        payload = json.load(f)

    if "selected_layers" not in payload:
        raise ValueError(f"{path} does not contain 'selected_layers'.")

    selected = {
        str(behavior): int(layer)
        for behavior, layer in payload["selected_layers"].items()
    }

    missing = [behavior for behavior in expected_behaviors if behavior not in selected]

    if missing:
        raise ValueError(f"Selected-layer file is missing behaviors: {missing}")

    return selected


def load_capability_datasets(
    root: Path,
):
    selection = root / "selection"

    return {
        "mmlu": load_mmlu_data(selection / "mmlu.json"),
        "arc_challenge": (load_arc_challenge_data(selection / "arc_challenge.json")),
        "hellaswag": (load_hellaswag_data(selection / "hellaswag.json")),
    }


def evaluator_for(
    benchmark: str,
):
    mapping = {
        "mmlu": evaluate_mmlu,
        "arc_challenge": (evaluate_arc_challenge),
        "hellaswag": (evaluate_hellaswag),
    }

    return mapping[benchmark]


def simplify_items(
    benchmark: str,
    result,
):
    rows = []

    for item in result["items"]:
        common = {
            "item_index": int(item["item_index"]),
            "id": str(item["id"]),
            "is_correct": int(bool(item["is_correct"])),
        }

        if benchmark in {
            "mmlu",
            "arc_challenge",
        }:
            rows.append(
                {
                    **common,
                    "correct": (item["correct"]),
                    "prediction": (item["prediction"]),
                    "correct_raw_prob": float(item["correct_raw_prob"]),
                    "correct_choice_prob": float(item["correct_choice_prob"]),
                }
            )

        elif benchmark == "hellaswag":
            rows.append(
                {
                    **common,
                    "correct_index": int(item["correct_index"]),
                    "prediction": int(item["prediction"]),
                    "raw_prediction": int(item["raw_prediction"]),
                    "raw_is_correct": int(bool(item["raw_is_correct"])),
                    "correct_num_tokens": int(item["correct_num_tokens"]),
                    "correct_sum_log_likelihood": float(
                        item["correct_sum_log_likelihood"]
                    ),
                    "correct_mean_log_likelihood": float(
                        item["correct_mean_log_likelihood"]
                    ),
                }
            )

        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    return rows


def evaluate_condition(
    *,
    model,
    tokenizer,
    adapter,
    benchmark,
    dataset,
    layer,
    vector,
    alpha,
    device,
    batch_size,
):
    evaluator = evaluator_for(benchmark)

    result = evaluator(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=int(layer),
        vector=vector,
        coefficient=float(alpha),
        device=device,
        prompt_adapter=adapter,
        batch_size=int(batch_size),
    )

    return (
        result,
        simplify_items(
            benchmark,
            result,
        ),
    )


def write_condition(
    path: Path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def read_condition(
    path: Path,
):
    return pd.read_csv(path)


def validate_condition_file(
    path: Path,
    expected_examples: int,
):
    if not path.exists():
        return False

    try:
        df = read_condition(path)

    except Exception:
        return False

    required = {
        "item_index",
        "id",
        "is_correct",
    }

    if not required.issubset(df.columns):
        return False

    if len(df) != expected_examples:
        return False

    if df["item_index"].duplicated().any():
        return False

    return True


def condition_path(
    output_root: Path,
    benchmark: str,
    behavior: str,
    alpha: float,
):
    alpha_text = (
        f"{alpha:+.1f}".replace(
            "+",
            "p",
        )
        .replace(
            "-",
            "m",
        )
        .replace(
            ".",
            "p",
        )
    )

    return output_root / benchmark / behavior / f"alpha_{alpha_text}.csv"


def baseline_path(
    output_root: Path,
    benchmark: str,
):
    return output_root / benchmark / "baseline_alpha_0.csv"


def paired_capability_statistics(
    *,
    baseline_df,
    steered_df,
    benchmark,
    behavior,
    alpha,
    bootstrap_samples,
    bootstrap_seed,
):
    baseline = baseline_df[
        [
            "id",
            "is_correct",
        ]
    ].rename(columns={"is_correct": ("baseline_correct")})

    steered = steered_df[
        [
            "id",
            "is_correct",
        ]
    ].rename(columns={"is_correct": ("steered_correct")})

    merged = baseline.merge(
        steered,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(baseline_df):
        raise RuntimeError(
            f"{benchmark}/{behavior}/"
            f"alpha={alpha}: baseline and "
            "steered items do not align."
        )

    merged["delta_correct"] = merged["steered_correct"] - merged["baseline_correct"]

    baseline_accuracy = float(merged["baseline_correct"].mean())

    steered_accuracy = float(merged["steered_correct"].mean())

    seed = stable_seed(
        base_seed=bootstrap_seed,
        benchmark=benchmark,
        behavior=behavior,
        alpha=alpha,
    )

    (
        delta_accuracy,
        ci_low,
        ci_high,
    ) = bootstrap_mean_ci(
        merged["delta_correct"].to_numpy(),
        n_bootstrap=(bootstrap_samples),
        seed=seed,
    )

    direct_delta = steered_accuracy - baseline_accuracy

    if abs(direct_delta - delta_accuracy) > 1e-10:
        raise RuntimeError("Paired item mean does not match aggregate accuracy change.")

    return {
        "benchmark": benchmark,
        "behavior": behavior,
        "alpha": float(alpha),
        "baseline_accuracy": (baseline_accuracy),
        "accuracy": (steered_accuracy),
        "delta_accuracy": float(delta_accuracy),
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "num_examples": int(len(merged)),
        "bootstrap_samples": int(bootstrap_samples),
    }


def add_feasibility(
    summary: pd.DataFrame,
):
    for epsilon in DEFAULT_TOLERANCES:
        name = f"feasible_eps_{int(epsilon * 100):02d}"

        summary[name] = summary["delta_ci_low"] >= -epsilon

    return summary


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    config = load_experiment_config(args.config)

    selected_layers = load_selected_layers(
        args.selected_layers,
        config.behaviors,
    )

    datasets = load_capability_datasets(args.capability_root)

    vectors = load_normalized_vectors(config)

    output_root = config.results_root / "capability_selection"

    print("=" * 76)

    print("CAPABILITY SELECTION")

    print("=" * 76)

    print(
        "Model:",
        config.model_label,
    )

    print(
        "Selected layers:",
        selected_layers,
    )

    print(
        "Alphas:",
        config.alphas,
    )

    print(
        "Core benchmarks:",
        CORE_BENCHMARKS,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Bootstrap samples:",
        f"{args.bootstrap_samples:,}",
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
        model_name=(config.model_name),
        prompt_family=(config.prompt_family),
    )

    summary_rows = []

    baselines = {}

    reference_behavior = config.behaviors[0]

    reference_layer = int(selected_layers[reference_behavior])

    reference_vector = vectors[reference_behavior][reference_layer]

    for benchmark in CORE_BENCHMARKS:
        dataset = datasets[benchmark]

        path = baseline_path(
            output_root,
            benchmark,
        )

        if args.resume and validate_condition_file(
            path,
            len(dataset),
        ):
            print(f"REUSE baseline — {benchmark}")

            baseline_df = read_condition(path)

        else:
            print()

            print("=" * 76)

            print(f"BASELINE — {benchmark}")

            print("=" * 76)

            (
                result,
                rows,
            ) = evaluate_condition(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                benchmark=benchmark,
                dataset=dataset,
                layer=reference_layer,
                vector=(reference_vector),
                alpha=0.0,
                device=args.device,
                batch_size=(args.batch_size),
            )

            write_condition(
                path,
                rows,
            )

            baseline_df = pd.DataFrame(rows)

            print(f"Baseline accuracy: {result['accuracy']:.6f}")

            print(
                "Saved:",
                path,
            )

        baselines[benchmark] = baseline_df

    nonzero_alphas = [float(alpha) for alpha in config.alphas if float(alpha) != 0.0]

    for behavior_index, behavior in enumerate(
        config.behaviors,
        start=1,
    ):
        layer = int(selected_layers[behavior])

        vector = vectors[behavior][layer]

        print()

        print("#" * 76)

        print(
            f"BEHAVIOR {behavior_index}/{len(config.behaviors)} — {behavior} @ L{layer}"
        )

        print("#" * 76)

        for benchmark in CORE_BENCHMARKS:
            baseline_df = baselines[benchmark]

            baseline_accuracy = float(baseline_df["is_correct"].mean())

            summary_rows.append(
                {
                    "benchmark": benchmark,
                    "behavior": behavior,
                    "layer": layer,
                    "alpha": 0.0,
                    "baseline_accuracy": (baseline_accuracy),
                    "accuracy": (baseline_accuracy),
                    "delta_accuracy": 0.0,
                    "delta_ci_low": 0.0,
                    "delta_ci_high": 0.0,
                    "num_examples": int(len(baseline_df)),
                    "bootstrap_samples": int(args.bootstrap_samples),
                }
            )

        for alpha in nonzero_alphas:
            print()

            print(f"--- {behavior} L{layer} alpha={alpha:+g} ---")

            for benchmark in CORE_BENCHMARKS:
                dataset = datasets[benchmark]

                path = condition_path(
                    output_root=(output_root),
                    benchmark=benchmark,
                    behavior=behavior,
                    alpha=alpha,
                )

                if args.resume and validate_condition_file(
                    path,
                    len(dataset),
                ):
                    print(
                        "REUSE —",
                        benchmark,
                        behavior,
                        f"alpha={alpha:+g}",
                    )

                    steered_df = read_condition(path)

                else:
                    (
                        result,
                        rows,
                    ) = evaluate_condition(
                        model=model,
                        tokenizer=tokenizer,
                        adapter=adapter,
                        benchmark=benchmark,
                        dataset=dataset,
                        layer=layer,
                        vector=vector,
                        alpha=alpha,
                        device=args.device,
                        batch_size=(args.batch_size),
                    )

                    write_condition(
                        path,
                        rows,
                    )

                    steered_df = pd.DataFrame(rows)

                    print(f"{benchmark}: accuracy={result['accuracy']:.6f}")

                stats = paired_capability_statistics(
                    baseline_df=(baselines[benchmark]),
                    steered_df=(steered_df),
                    benchmark=(benchmark),
                    behavior=behavior,
                    alpha=alpha,
                    bootstrap_samples=(args.bootstrap_samples),
                    bootstrap_seed=(args.bootstrap_seed),
                )

                stats["layer"] = layer

                summary_rows.append(stats)

                print(
                    f"{benchmark:15s} "
                    f"ΔC="
                    f"{stats['delta_accuracy']:+.4f} "
                    f"95% CI=["
                    f"{stats['delta_ci_low']:+.4f}, "
                    f"{stats['delta_ci_high']:+.4f}]"
                )

    summary = pd.DataFrame(summary_rows)

    summary = add_feasibility(summary)

    summary = summary[
        [
            "behavior",
            "layer",
            "benchmark",
            "alpha",
            "baseline_accuracy",
            "accuracy",
            "delta_accuracy",
            "delta_ci_low",
            "delta_ci_high",
            "feasible_eps_01",
            "feasible_eps_03",
            "feasible_eps_05",
            "num_examples",
            "bootstrap_samples",
        ]
    ]

    summary_path = output_root / "summary.csv"

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "model_name": (config.model_name),
        "model_label": (config.model_label),
        "role": ("capability_selection"),
        "selected_layers_file": str(args.selected_layers),
        "selected_layers": (selected_layers),
        "benchmarks": list(CORE_BENCHMARKS),
        "alphas": [float(x) for x in config.alphas],
        "tolerances": list(DEFAULT_TOLERANCES),
        "constraint": ("LCB95(delta_accuracy) >= -epsilon for every core benchmark"),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "batch_size": int(args.batch_size),
        "precision": str(config.torch_dtype),
    }

    with (output_root / "manifest.json").open("w") as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()

    print("=" * 76)

    print("CAPABILITY-SELECTION SWEEP COMPLETE")

    print("=" * 76)

    print(
        "Summary ->",
        summary_path,
    )


if __name__ == "__main__":
    main()
