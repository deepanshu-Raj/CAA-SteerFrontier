"""Evaluate frozen steering frontiers on untouched test and holdout datasets.

Layers, coefficients, and capability budgets are immutable inputs during this
confirmatory evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from steerfrontier.config import (
    load_experiment_config,
)
from steerfrontier.data import (
    load_arc_challenge_data,
    load_hellaswag_data,
    load_mmlu_data,
    load_test_data,
)
from steerfrontier.evaluation import (
    evaluate_arc_challenge,
    evaluate_behavior,
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


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen final confirmatory evaluation on "
            "behavior test and capability holdout data."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--selected-coefficients",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--behavior-data-root",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--capability-root",
        type=Path,
        default=Path("data/capability"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=(DEFAULT_BOOTSTRAP_SAMPLES),
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=(DEFAULT_BOOTSTRAP_SEED),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Print the frozen final-evaluation plan "
            "without loading test/holdout datasets "
            "or the model."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Reproducibility helpers
# =============================================================================


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def stable_seed(
    *,
    base_seed: int,
    domain: str,
    behavior: str,
    alpha: float,
    benchmark: str = "",
) -> int:
    payload = (f"{base_seed}:{domain}:{behavior}:{benchmark}:{alpha:.12g}").encode()

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
    )


def alpha_filename(
    alpha: float,
) -> str:
    text = f"{float(alpha):+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")

    return f"alpha_{text}.csv"


# =============================================================================
# Frozen frontier parsing
# =============================================================================


def load_frozen_frontiers(
    path: Path,
):
    with path.open("r") as f:
        payload = json.load(f)

    if "behaviors" not in payload:
        raise ValueError(f"{path} does not contain 'behaviors'.")

    if "tolerances" not in payload:
        raise ValueError(f"{path} does not contain 'tolerances'.")

    frontier_specs = []
    execution_plan = {}

    for behavior, behavior_payload in payload["behaviors"].items():
        layer = int(behavior_payload["layer"])

        tolerances = behavior_payload["tolerances"]

        unique_nonzero = set()

        for epsilon_key, epsilon in payload["tolerances"].items():
            if epsilon_key not in tolerances:
                raise ValueError(f"{behavior} is missing tolerance {epsilon_key}.")

            for direction in (
                "negative",
                "positive",
            ):
                if direction not in tolerances[epsilon_key]:
                    raise ValueError(
                        f"{behavior}/{epsilon_key} is missing {direction}."
                    )

                record = tolerances[epsilon_key][direction]

                alpha = float(record["alpha"])

                if alpha != 0.0:
                    unique_nonzero.add(alpha)

                frontier_specs.append(
                    {
                        "behavior": behavior,
                        "layer": layer,
                        "epsilon_key": (epsilon_key),
                        "epsilon": float(epsilon),
                        "direction": (direction),
                        "alpha": alpha,
                        "eta": record.get("eta"),
                        "selection_behavior_gain": float(
                            record.get(
                                "behavior_gain",
                                0.0,
                            )
                        ),
                    }
                )

        execution_plan[behavior] = {
            "layer": layer,
            "nonzero_alphas": sorted(unique_nonzero),
        }

    return (
        payload,
        frontier_specs,
        execution_plan,
    )


def print_plan(
    execution_plan,
    frontier_specs,
):
    print("=" * 76)

    print("FROZEN FINAL-EVALUATION PLAN")

    print("=" * 76)

    print()

    total_nonzero = 0

    for behavior, plan in execution_plan.items():
        alphas = plan["nonzero_alphas"]

        total_nonzero += len(alphas)

        print(f"{behavior:20s} L{plan['layer']:02d} alphas={alphas}")

    print()

    print(f"Unique nonzero behavior/intervention conditions: {total_nonzero}")

    print(
        "Frozen frontier claims:",
        len(frontier_specs),
    )

    print(
        "Behavior-test inference conditions:",
        total_nonzero + len(execution_plan),
        "(includes one baseline per behavior)",
    )

    print(
        "Capability-holdout steered conditions:",
        total_nonzero * len(CORE_BENCHMARKS),
    )

    print(
        "Capability-holdout shared baselines:",
        len(CORE_BENCHMARKS),
    )

    print()

    print("No coefficient will be selected or changed during this run.")


# =============================================================================
# Condition file helpers
# =============================================================================


def write_csv(
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


def read_csv(
    path: Path,
):
    return pd.read_csv(path)


def valid_behavior_condition(
    path: Path,
    expected_examples: int,
):
    if not path.exists():
        return False

    try:
        df = pd.read_csv(path)

    except Exception:
        return False

    required = {
        "item_index",
        "matching_prob",
    }

    if not required.issubset(df.columns):
        return False

    if len(df) != expected_examples:
        return False

    if df["item_index"].duplicated().any():
        return False

    return True


def valid_capability_condition(
    path: Path,
    expected_examples: int,
):
    if not path.exists():
        return False

    try:
        df = pd.read_csv(path)

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


# =============================================================================
# Behavior-test evaluation
# =============================================================================


def simplify_behavior_items(
    result,
):
    rows = []

    for item in result["items"]:
        rows.append(
            {
                "item_index": int(item["item_index"]),
                "matching_answer": (item["matching_answer"]),
                "nonmatching_answer": (item["nonmatching_answer"]),
                "matching_raw_prob": float(item["matching_raw_prob"]),
                "nonmatching_raw_prob": float(item["nonmatching_raw_prob"]),
                "matching_prob": float(item["matching_prob"]),
                "nonmatching_prob": float(item["nonmatching_prob"]),
            }
        )

    return rows


def evaluate_behavior_condition(
    *,
    model,
    tokenizer,
    adapter,
    dataset,
    layer,
    vector,
    alpha,
    device,
):
    result = evaluate_behavior(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=int(layer),
        vector=vector,
        coefficient=float(alpha),
        device=device,
        prompt_adapter=adapter,
    )

    return (
        result,
        simplify_behavior_items(result),
    )


def behavior_statistics(
    *,
    baseline_df,
    steered_df,
    behavior,
    alpha,
    bootstrap_samples,
    bootstrap_seed,
):
    baseline = baseline_df[
        [
            "item_index",
            "matching_prob",
        ]
    ].rename(columns={"matching_prob": ("baseline_matching_prob")})

    steered = steered_df[
        [
            "item_index",
            "matching_prob",
        ]
    ].rename(columns={"matching_prob": ("steered_matching_prob")})

    merged = baseline.merge(
        steered,
        on="item_index",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(baseline_df):
        raise RuntimeError(
            f"{behavior}, alpha={alpha}: "
            "behavior baseline and steered "
            "items do not align."
        )

    merged["delta_behavior"] = (
        merged["steered_matching_prob"] - merged["baseline_matching_prob"]
    )

    baseline_score = float(merged["baseline_matching_prob"].mean())

    behavior_score = float(merged["steered_matching_prob"].mean())

    if float(alpha) == 0.0:
        return {
            "behavior": behavior,
            "alpha": 0.0,
            "baseline_behavior_score": (baseline_score),
            "behavior_score": (baseline_score),
            "delta_behavior": 0.0,
            "delta_ci_low": 0.0,
            "delta_ci_high": 0.0,
            "num_examples": int(len(merged)),
            "bootstrap_samples": int(bootstrap_samples),
        }

    seed = stable_seed(
        base_seed=(bootstrap_seed),
        domain="behavior_test",
        behavior=behavior,
        alpha=alpha,
    )

    (
        delta,
        ci_low,
        ci_high,
    ) = bootstrap_mean_ci(
        merged["delta_behavior"].to_numpy(),
        n_bootstrap=(bootstrap_samples),
        seed=seed,
    )

    direct_delta = behavior_score - baseline_score

    if abs(direct_delta - delta) > 1e-10:
        raise RuntimeError(
            "Paired behavioral mean does not match aggregate behavior delta."
        )

    return {
        "behavior": behavior,
        "alpha": float(alpha),
        "baseline_behavior_score": (baseline_score),
        "behavior_score": (behavior_score),
        "delta_behavior": float(delta),
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "num_examples": int(len(merged)),
        "bootstrap_samples": int(bootstrap_samples),
    }


# =============================================================================
# Capability-holdout evaluation
# =============================================================================


def evaluator_for(
    benchmark: str,
):
    mapping = {
        "mmlu": evaluate_mmlu,
        "arc_challenge": (evaluate_arc_challenge),
        "hellaswag": (evaluate_hellaswag),
    }

    return mapping[benchmark]


def simplify_capability_items(
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


def evaluate_capability_condition(
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
        simplify_capability_items(
            benchmark,
            result,
        ),
    )


def capability_statistics(
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
            f"{benchmark}/{behavior}/alpha={alpha}: holdout items do not align."
        )

    merged["delta_correct"] = merged["steered_correct"] - merged["baseline_correct"]

    baseline_accuracy = float(merged["baseline_correct"].mean())

    accuracy = float(merged["steered_correct"].mean())

    if float(alpha) == 0.0:
        return {
            "benchmark": benchmark,
            "behavior": behavior,
            "alpha": 0.0,
            "baseline_accuracy": (baseline_accuracy),
            "accuracy": (baseline_accuracy),
            "delta_accuracy": 0.0,
            "delta_ci_low": 0.0,
            "delta_ci_high": 0.0,
            "num_examples": int(len(merged)),
            "bootstrap_samples": int(bootstrap_samples),
        }

    seed = stable_seed(
        base_seed=(bootstrap_seed),
        domain="capability_holdout",
        behavior=behavior,
        benchmark=benchmark,
        alpha=alpha,
    )

    (
        delta,
        ci_low,
        ci_high,
    ) = bootstrap_mean_ci(
        merged["delta_correct"].to_numpy(),
        n_bootstrap=(bootstrap_samples),
        seed=seed,
    )

    direct_delta = accuracy - baseline_accuracy

    if abs(direct_delta - delta) > 1e-10:
        raise RuntimeError(
            "Paired capability mean does not match aggregate accuracy change."
        )

    return {
        "benchmark": benchmark,
        "behavior": behavior,
        "alpha": float(alpha),
        "baseline_accuracy": (baseline_accuracy),
        "accuracy": accuracy,
        "delta_accuracy": float(delta),
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "num_examples": int(len(merged)),
        "bootstrap_samples": int(bootstrap_samples),
    }


# =============================================================================
# Frontier-level final report
# =============================================================================


def find_condition_row(
    df,
    *,
    behavior,
    alpha,
    benchmark=None,
):
    mask = (df["behavior"] == behavior) & ((df["alpha"] - float(alpha)).abs() < 1e-9)

    if benchmark is not None:
        mask = mask & (df["benchmark"] == benchmark)

    subset = df[mask]

    if len(subset) != 1:
        raise RuntimeError(
            "Expected exactly one matching "
            f"result for behavior={behavior}, "
            f"alpha={alpha}, "
            f"benchmark={benchmark}; "
            f"found {len(subset)}."
        )

    return subset.iloc[0]


def build_frontier_results(
    *,
    frontier_specs,
    behavior_summary,
    capability_summary,
):
    rows = []

    for spec in frontier_specs:
        behavior = spec["behavior"]

        alpha = float(spec["alpha"])

        epsilon = float(spec["epsilon"])

        direction = spec["direction"]

        behavior_row = find_condition_row(
            behavior_summary,
            behavior=behavior,
            alpha=alpha,
        )

        signed_delta = float(behavior_row["delta_behavior"])

        signed_low = float(behavior_row["delta_ci_low"])

        signed_high = float(behavior_row["delta_ci_high"])

        if direction == "positive":
            directional_gain = signed_delta

            directional_low = signed_low

            directional_high = signed_high

        elif direction == "negative":
            directional_gain = -signed_delta

            # Multiplication by -1 reverses
            # the interval.
            directional_low = -signed_high

            directional_high = -signed_low

        else:
            raise ValueError(f"Unknown direction: {direction}")

        capability_rows = []

        for benchmark in CORE_BENCHMARKS:
            capability_rows.append(
                find_condition_row(
                    capability_summary,
                    behavior=behavior,
                    alpha=alpha,
                    benchmark=benchmark,
                )
            )

        worst = min(
            capability_rows,
            key=lambda row: float(row["delta_ci_low"]),
        )

        capability_pass = all(
            float(row["delta_ci_low"]) >= -epsilon for row in capability_rows
        )

        rows.append(
            {
                "behavior": behavior,
                "layer": int(spec["layer"]),
                "epsilon": (epsilon),
                "direction": (direction),
                "alpha_star": (alpha),
                "eta_star": (spec["eta"]),
                "selection_behavior_gain": (spec["selection_behavior_gain"]),
                "test_behavior_baseline": float(
                    behavior_row["baseline_behavior_score"]
                ),
                "test_behavior_score": float(behavior_row["behavior_score"]),
                "test_behavior_delta": (signed_delta),
                "test_behavior_delta_ci_low": (signed_low),
                "test_behavior_delta_ci_high": (signed_high),
                "test_directional_gain": (directional_gain),
                "test_directional_gain_ci_low": (directional_low),
                "test_directional_gain_ci_high": (directional_high),
                "test_direction_supported": bool(directional_low > 0.0),
                "capability_holdout_pass": bool(capability_pass),
                "worst_holdout_lcb": float(worst["delta_ci_low"]),
                "limiting_holdout_benchmark": str(worst["benchmark"]),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    (
        frozen_payload,
        frontier_specs,
        execution_plan,
    ) = load_frozen_frontiers(args.selected_coefficients)

    print_plan(
        execution_plan,
        frontier_specs,
    )

    if args.plan_only:
        print()

        print("PLAN ONLY — no test or holdout data were loaded.")

        return

    config = load_experiment_config(args.config)

    expected_behaviors = set(config.behaviors)

    frozen_behaviors = set(execution_plan)

    if frozen_behaviors != expected_behaviors:
        raise ValueError(
            "Frozen behavior set does not match "
            "experiment config.\n"
            f"Frozen: "
            f"{sorted(frozen_behaviors)}\n"
            f"Config: "
            f"{sorted(expected_behaviors)}"
        )

    output_root = Path(config.results_root) / "final_evaluation"

    behavior_output_root = output_root / "behavior_test"

    capability_output_root = output_root / "capability_holdout"

    # ================================================================
    # This is the point at which final datasets are opened.
    # ================================================================

    behavior_datasets = {
        behavior: (load_test_data(args.behavior_data_root / behavior / "test.json"))
        for behavior in config.behaviors
    }

    holdout_root = args.capability_root / "holdout"

    capability_datasets = {
        "mmlu": load_mmlu_data(holdout_root / "mmlu.json"),
        "arc_challenge": (load_arc_challenge_data(holdout_root / "arc_challenge.json")),
        "hellaswag": (load_hellaswag_data(holdout_root / "hellaswag.json")),
    }

    vectors = load_normalized_vectors(config)

    model, tokenizer = load_model(
        model_name=(config.model_name),
        device=args.device,
        dtype=config.torch_dtype,
    )

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=(config.model_name),
        prompt_family=(config.prompt_family),
    )

    # ================================================================
    # BEHAVIOR TEST
    # ================================================================

    print()

    print("=" * 76)

    print("BEHAVIOR TEST EVALUATION")

    print("=" * 76)

    behavior_summary_rows = []

    for behavior in config.behaviors:
        dataset = behavior_datasets[behavior]

        layer = execution_plan[behavior]["layer"]

        vector = vectors[behavior][layer]

        behavior_dir = behavior_output_root / behavior

        baseline_path = behavior_dir / "baseline_alpha_0.csv"

        if args.resume and valid_behavior_condition(
            baseline_path,
            len(dataset),
        ):
            print(f"REUSE behavior baseline — {behavior}")

            baseline_df = read_csv(baseline_path)

        else:
            print(f"\nBehavior test baseline — {behavior} @ L{layer}")

            (
                result,
                rows,
            ) = evaluate_behavior_condition(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                dataset=dataset,
                layer=layer,
                vector=vector,
                alpha=0.0,
                device=args.device,
            )

            write_csv(
                baseline_path,
                rows,
            )

            baseline_df = pd.DataFrame(rows)

            print(f"baseline behavior score = {result['behavior_score']:.6f}")

        behavior_summary_rows.append(
            behavior_statistics(
                baseline_df=baseline_df,
                steered_df=baseline_df,
                behavior=behavior,
                alpha=0.0,
                bootstrap_samples=(args.bootstrap_samples),
                bootstrap_seed=(args.bootstrap_seed),
            )
        )

        for alpha in execution_plan[behavior]["nonzero_alphas"]:
            path = behavior_dir / alpha_filename(alpha)

            if args.resume and valid_behavior_condition(
                path,
                len(dataset),
            ):
                print(
                    "REUSE behavior —",
                    behavior,
                    f"alpha={alpha:+g}",
                )

                steered_df = read_csv(path)

            else:
                print(f"Behavior test — {behavior} alpha={alpha:+g}")

                (
                    result,
                    rows,
                ) = evaluate_behavior_condition(
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    dataset=dataset,
                    layer=layer,
                    vector=vector,
                    alpha=alpha,
                    device=args.device,
                )

                write_csv(
                    path,
                    rows,
                )

                steered_df = pd.DataFrame(rows)

                print(f"behavior score = {result['behavior_score']:.6f}")

            stats = behavior_statistics(
                baseline_df=baseline_df,
                steered_df=steered_df,
                behavior=behavior,
                alpha=alpha,
                bootstrap_samples=(args.bootstrap_samples),
                bootstrap_seed=(args.bootstrap_seed),
            )

            behavior_summary_rows.append(stats)

            print(
                f"  ΔB="
                f"{stats['delta_behavior']:+.4f} "
                f"95% CI=["
                f"{stats['delta_ci_low']:+.4f}, "
                f"{stats['delta_ci_high']:+.4f}]"
            )

    behavior_summary = pd.DataFrame(behavior_summary_rows)

    behavior_summary_path = output_root / "behavior_test_summary.csv"

    behavior_summary.to_csv(
        behavior_summary_path,
        index=False,
    )

    # ================================================================
    # CAPABILITY HOLDOUT
    # ================================================================

    print()

    print("=" * 76)

    print("CAPABILITY HOLDOUT EVALUATION")

    print("=" * 76)

    capability_baselines = {}

    reference_behavior = config.behaviors[0]

    reference_layer = execution_plan[reference_behavior]["layer"]

    reference_vector = vectors[reference_behavior][reference_layer]

    for benchmark in CORE_BENCHMARKS:
        dataset = capability_datasets[benchmark]

        path = capability_output_root / benchmark / "baseline_alpha_0.csv"

        if args.resume and valid_capability_condition(
            path,
            len(dataset),
        ):
            print(f"REUSE holdout baseline — {benchmark}")

            baseline_df = read_csv(path)

        else:
            print(f"\nHoldout baseline — {benchmark}")

            (
                result,
                rows,
            ) = evaluate_capability_condition(
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

            write_csv(
                path,
                rows,
            )

            baseline_df = pd.DataFrame(rows)

            print(f"baseline accuracy = {result['accuracy']:.6f}")

        capability_baselines[benchmark] = baseline_df

    capability_summary_rows = []

    for behavior in config.behaviors:
        layer = execution_plan[behavior]["layer"]

        vector = vectors[behavior][layer]

        # Logical alpha=0 rows for every behavior.
        for benchmark in CORE_BENCHMARKS:
            baseline_df = capability_baselines[benchmark]

            capability_summary_rows.append(
                capability_statistics(
                    baseline_df=(baseline_df),
                    steered_df=(baseline_df),
                    benchmark=(benchmark),
                    behavior=(behavior),
                    alpha=0.0,
                    bootstrap_samples=(args.bootstrap_samples),
                    bootstrap_seed=(args.bootstrap_seed),
                )
            )

        for alpha in execution_plan[behavior]["nonzero_alphas"]:
            print()

            print(f"Holdout — {behavior} @ L{layer}, alpha={alpha:+g}")

            for benchmark in CORE_BENCHMARKS:
                dataset = capability_datasets[benchmark]

                path = (
                    capability_output_root
                    / benchmark
                    / behavior
                    / alpha_filename(alpha)
                )

                if args.resume and valid_capability_condition(
                    path,
                    len(dataset),
                ):
                    print(
                        "REUSE —",
                        benchmark,
                    )

                    steered_df = read_csv(path)

                else:
                    (
                        result,
                        rows,
                    ) = evaluate_capability_condition(
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

                    write_csv(
                        path,
                        rows,
                    )

                    steered_df = pd.DataFrame(rows)

                    print(f"{benchmark}: accuracy={result['accuracy']:.6f}")

                stats = capability_statistics(
                    baseline_df=(capability_baselines[benchmark]),
                    steered_df=(steered_df),
                    benchmark=(benchmark),
                    behavior=behavior,
                    alpha=alpha,
                    bootstrap_samples=(args.bootstrap_samples),
                    bootstrap_seed=(args.bootstrap_seed),
                )

                capability_summary_rows.append(stats)

                print(
                    f"{benchmark:15s} "
                    f"ΔC="
                    f"{stats['delta_accuracy']:+.4f} "
                    f"95% CI=["
                    f"{stats['delta_ci_low']:+.4f}, "
                    f"{stats['delta_ci_high']:+.4f}]"
                )

    capability_summary = pd.DataFrame(capability_summary_rows)

    capability_summary_path = output_root / "capability_holdout_summary.csv"

    capability_summary.to_csv(
        capability_summary_path,
        index=False,
    )

    # ================================================================
    # FROZEN FRONTIER FINAL RESULTS
    # ================================================================

    frontier_results = build_frontier_results(
        frontier_specs=(frontier_specs),
        behavior_summary=(behavior_summary),
        capability_summary=(capability_summary),
    )

    frontier_path = output_root / "frontier_test_results.csv"

    frontier_results.to_csv(
        frontier_path,
        index=False,
    )

    manifest = {
        "protocol": ("final_confirmatory_evaluation"),
        "model_name": (config.model_name),
        "model_label": (config.model_label),
        "precision": str(config.torch_dtype),
        "batch_size": int(args.batch_size),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "frozen_coefficients_file": str(args.selected_coefficients),
        "frozen_coefficients_sha256": (sha256_file(args.selected_coefficients)),
        "behavior_split": ("test"),
        "capability_role": ("holdout"),
        "core_benchmarks": list(CORE_BENCHMARKS),
        "rule": (
            "Frozen layer/coefficient choices "
            "must not be changed after examining "
            "behavior-test or capability-holdout "
            "results."
        ),
    }

    manifest_path = output_root / "manifest.json"

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open("w") as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()

    print("=" * 76)

    print("FINAL CONFIRMATORY EVALUATION COMPLETE")

    print("=" * 76)

    print()

    print(
        frontier_results[
            [
                "behavior",
                "epsilon",
                "direction",
                "alpha_star",
                "selection_behavior_gain",
                "test_directional_gain",
                "test_directional_gain_ci_low",
                "test_direction_supported",
                "worst_holdout_lcb",
                "limiting_holdout_benchmark",
                "capability_holdout_pass",
            ]
        ].to_string(index=False)
    )

    print()

    print(
        "Behavior test ->",
        behavior_summary_path,
    )

    print(
        "Capability holdout ->",
        capability_summary_path,
    )

    print(
        "Final frontiers ->",
        frontier_path,
    )

    print(
        "Manifest ->",
        manifest_path,
    )


if __name__ == "__main__":
    main()
