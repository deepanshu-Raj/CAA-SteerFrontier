"""Evaluate random directions matched to CAA by off-target forward KL.

Random directions come from the saved norm-matched control and are matched on
off-target perturbation size before target-behavior evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from run_final_evaluation import (
    behavior_statistics,
    evaluate_behavior_condition,
)
from run_specificity_controls import (
    as_dataframe,
    as_dict,
    load_operating_points,
    sha256_file,
    stable_seed,
    write_csv,
)

from steerfrontier.config import load_experiment_config
from steerfrontier.data import (
    load_test_data,
    load_validation_data,
)
from steerfrontier.models import load_model
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.steering import SteeringIntervention
from steerfrontier.workflow import load_normalized_vectors

# =============================================================================
# Configuration
# =============================================================================

CONTROL_NAME = "off_target_kl_matched"
RANDOM_SOURCE_CONTROL = "norm_matched_random"

NONZERO_TOL = 1e-12
BETA_DIGITS = 10


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate off-target-KL-matched random controls "
            "at frozen CAA operating points."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260917,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--behaviors",
        nargs="*",
        default=None,
        help=(
            "Optional subset of target behaviors. "
            "Off-target calibration still uses every "
            "other configured behavior."
        ),
    )

    parser.add_argument(
        "--frontiers",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--final-results",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--random-control-root",
        type=Path,
        default=None,
        help=(
            "Completed norm-matched-random control directory "
            "containing vectors/seed_XXX.pt."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--offtarget-per-behavior",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--match-rtol",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--match-atol",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--max-refine",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-beta",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--anchor-magnitudes",
        type=float,
        nargs="*",
        default=[
            0.25,
            0.50,
            1.00,
            1.50,
            2.00,
            3.00,
            4.00,
            6.00,
        ],
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
# Keys / resume
# =============================================================================


def intervention_key(
    row,
):
    return (
        str(row["behavior"]),
        int(row["layer"]),
        str(row["direction"]),
        round(
            float(row["alpha_star"]),
            12,
        ),
    )


def matched_result_key(
    row,
):
    return (
        int(row["control_seed_index"]),
        *intervention_key(row),
    )


def curve_key(
    row,
):
    return (
        int(row["control_seed_index"]),
        str(row["behavior"]),
        str(row["direction"]),
        round(
            float(row["beta"]),
            BETA_DIGITS,
        ),
    )


def load_existing(
    path: Path,
    *,
    resume: bool,
    key_fn,
):
    if not resume or not path.is_file():
        return (
            pd.DataFrame(),
            set(),
        )

    frame = pd.read_csv(path)

    keys = {key_fn(row) for row in frame.to_dict(orient="records")}

    print(f"Resume: loaded {len(frame)} rows from {path}")

    return (
        frame,
        keys,
    )


# =============================================================================
# Fixed off-target validation subset
# =============================================================================


def text_sha256(
    text: str,
):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_offtarget_examples(
    config,
    *,
    per_behavior: int,
    base_seed: int,
):
    """
    Pick one deterministic subset from every behavior's validation split.

    The target behavior is excluded later when computing KL, so with seven
    configured behaviors and k=8 this gives 6*8=48 prompts per target.
    """

    selected = []

    for behavior in config.behaviors:
        path = config.layout.validation_data_path(behavior)

        dataset = load_validation_data(path)

        if len(dataset) < per_behavior:
            raise ValueError(
                f"{behavior}: validation has "
                f"{len(dataset)} examples; "
                f"need {per_behavior}."
            )

        # Deliberately omit model_slug so the same source indices
        # are selected across models.
        seed = stable_seed(
            base_seed,
            CONTROL_NAME,
            "offtarget_subset",
            behavior,
        )

        rng = np.random.default_rng(seed)

        indices = sorted(
            int(value)
            for value in rng.choice(
                len(dataset),
                size=per_behavior,
                replace=False,
            )
        )

        for index in indices:
            question = dataset[index]["question"]

            selected.append(
                {
                    "source_behavior": behavior,
                    "source_index": index,
                    "question": question,
                    "question_sha256": (text_sha256(question)),
                    "subset_seed": int(seed),
                }
            )

    return selected


# =============================================================================
# Prompt / baseline KL cache
# =============================================================================


@torch.no_grad()
def build_prompt_cache(
    model,
    tokenizer,
    adapter,
    selected_examples,
    device: str,
):
    """
    Tokenize every unique calibration prompt once and cache the unsteered
    full-vocabulary next-token distribution on GPU.
    """

    cache = {}

    print()
    print("=" * 76)
    print("OFF-TARGET BASELINE DISTRIBUTIONS")
    print("=" * 76)
    print(f"Unique calibration prompts: {len(selected_examples)}")

    for count, item in enumerate(
        selected_examples,
        start=1,
    ):
        scoring = adapter.build_scoring_prompt(
            tokenizer,
            item["question"],
        )

        input_ids = scoring.input_ids.to(device)

        outputs = model(
            input_ids=input_ids,
            use_cache=False,
        )

        logits = outputs.logits[
            0,
            -1,
            :,
        ].float()

        baseline_logp = F.log_softmax(
            logits,
            dim=-1,
        ).detach()

        key = (
            item["source_behavior"],
            int(item["source_index"]),
        )

        cache[key] = {
            "source_behavior": (item["source_behavior"]),
            "input_ids": (input_ids),
            "from_position": int(scoring.steering_start_position),
            "baseline_logp": (baseline_logp),
            "baseline_p": (baseline_logp.exp().detach()),
        }

        if count % 8 == 0 or count == len(selected_examples):
            print(f"  {count}/{len(selected_examples)}")

    return cache


# =============================================================================
# KL evaluation
# =============================================================================


@torch.no_grad()
def off_target_kl(
    model,
    cache,
    target_behavior: str,
    layer: int,
    vector: torch.Tensor,
    coefficient: float,
):
    """
    Mean full-vocabulary next-token

        D_KL(p_0 || p_intervened)

    over fixed validation prompts from every behavior except target_behavior.
    """

    entries = [
        item for item in cache.values() if item["source_behavior"] != target_behavior
    ]

    if abs(float(coefficient)) <= NONZERO_TOL:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": len(entries),
            "by_source": (
                json.dumps(
                    {},
                    sort_keys=True,
                )
            ),
        }

    values = []
    by_source = {}

    for item in entries:
        with SteeringIntervention(
            model=model,
            layer_idx=int(layer),
            vector=vector,
            coefficient=float(coefficient),
            from_position=(item["from_position"]),
        ):
            outputs = model(
                input_ids=(item["input_ids"]),
                use_cache=False,
            )

        steered_logp = F.log_softmax(
            outputs.logits[
                0,
                -1,
                :,
            ].float(),
            dim=-1,
        )

        kl = torch.sum(item["baseline_p"] * (item["baseline_logp"] - steered_logp))

        # Numerical roundoff can occasionally make a tiny KL negative.
        value = max(
            float(kl.item()),
            0.0,
        )

        values.append(value)

        by_source.setdefault(
            item["source_behavior"],
            [],
        ).append(value)

    if not values:
        raise ValueError(f"No off-target prompts available for {target_behavior}.")

    source_means = {
        behavior: float(np.mean(source_values))
        for behavior, source_values in sorted(by_source.items())
    }

    return {
        "mean": float(np.mean(values)),
        "std": (
            float(
                np.std(
                    values,
                    ddof=1,
                )
            )
            if len(values) > 1
            else 0.0
        ),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
        "by_source": (
            json.dumps(
                source_means,
                sort_keys=True,
            )
        ),
    }


# =============================================================================
# Behavior statistics
# =============================================================================


def canonical_matched_statistics(
    stats,
    *,
    direction: str,
):
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
            f"behavior_statistics() returned unexpected schema. Missing: {missing}"
        )

    delta = float(row["delta_behavior"])

    delta_low = float(row["delta_ci_low"])

    delta_high = float(row["delta_ci_high"])

    if direction == "positive":
        gain = delta
        gain_low = delta_low
        gain_high = delta_high

    elif direction == "negative":
        gain = -delta
        gain_low = -delta_high
        gain_high = -delta_low

    else:
        raise ValueError(f"Unexpected direction: {direction}")

    return {
        "matched_behavior_baseline": float(row["baseline_behavior_score"]),
        "matched_behavior_score": float(row["behavior_score"]),
        "matched_behavior_delta": (delta),
        "matched_behavior_delta_ci_low": (delta_low),
        "matched_behavior_delta_ci_high": (delta_high),
        "matched_directional_gain": (gain),
        "matched_directional_gain_ci_low": (gain_low),
        "matched_directional_gain_ci_high": (gain_high),
        "matched_direction_supported": bool(gain_low > 0.0),
        "num_examples": int(row["num_examples"]),
        "bootstrap_samples": int(row["bootstrap_samples"]),
    }


# =============================================================================
# Load exact saved random directions
# =============================================================================


def load_random_vector(
    root: Path,
    *,
    model_slug: str,
    behavior: str,
    seed_index: int,
):
    path = root / "vectors" / f"seed_{seed_index:03d}.pt"

    if not path.is_file():
        raise FileNotFoundError(f"Missing saved norm-matched random vector: {path}")

    payload = torch.load(
        path,
        map_location="cpu",
    )

    if payload.get("control") != RANDOM_SOURCE_CONTROL:
        raise ValueError(
            f"{path}: expected "
            f"control="
            f"{RANDOM_SOURCE_CONTROL!r}, "
            f"found "
            f"{payload.get('control')!r}."
        )

    if payload.get("model_slug") != model_slug:
        raise ValueError(f"{path}: model slug mismatch.")

    if int(payload.get("control_seed_index")) != int(seed_index):
        raise ValueError(f"{path}: seed index mismatch.")

    if behavior not in payload["vectors"]:
        raise KeyError(f"{path}: no vector for {behavior}.")

    vector = payload["vectors"][behavior].detach().cpu().float()

    metadata = dict(
        payload.get(
            "metadata",
            {},
        ).get(
            behavior,
            {},
        )
    )

    return (
        vector,
        metadata,
        path,
    )


def validate_random_vector(
    learned_vector,
    random_vector,
    metadata: Mapping[
        str,
        Any,
    ],
):
    learned = learned_vector.detach().cpu().float().reshape(-1)

    random = random_vector.detach().cpu().float().reshape(-1)

    if learned.shape != random.shape:
        raise ValueError("CAA/random vector shape mismatch.")

    learned_norm = float(torch.linalg.vector_norm(learned).item())

    random_norm = float(torch.linalg.vector_norm(random).item())

    if not np.isclose(
        learned_norm,
        random_norm,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "Saved random vector "
            "is not norm-matched: "
            f"CAA={learned_norm}, "
            f"random={random_norm}"
        )

    cosine = float(
        torch.dot(
            learned,
            random,
        ).item()
        / (learned_norm * random_norm)
    )

    if "cosine_with_caa" in metadata:
        if not np.isclose(
            cosine,
            float(metadata["cosine_with_caa"]),
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError("Saved cosine metadata does not match tensors.")

    return {
        "learned_norm": (learned_norm),
        "random_norm": (random_norm),
        "norm_ratio": (random_norm / learned_norm),
        "cosine": (cosine),
    }


# =============================================================================
# Random KL curves
# =============================================================================


def curve_rows_for(
    rows,
    *,
    seed_index: int,
    behavior: str,
    direction: str,
):
    return sorted(
        [
            row
            for row in rows
            if int(row["control_seed_index"]) == int(seed_index)
            and row["behavior"] == behavior
            and row["direction"] == direction
        ],
        key=lambda row: abs(float(row["beta"])),
    )


def monotone_curve(
    rows,
):
    """
    Return |beta|, raw KL, and cumulative-max KL.

    The cumulative maximum only stabilizes interpolation against small
    finite-sample non-monotonicities. Actual achieved KL is always measured
    at the final selected beta.
    """

    by_magnitude = {}

    for row in rows:
        magnitude = abs(float(row["beta"]))

        by_magnitude.setdefault(
            magnitude,
            [],
        ).append(float(row["random_offtarget_kl_mean"]))

    magnitudes = np.asarray(
        sorted(by_magnitude),
        dtype=float,
    )

    kls = np.asarray(
        [np.mean(by_magnitude[magnitude]) for magnitude in magnitudes],
        dtype=float,
    )

    envelope = np.maximum.accumulate(kls)

    return (
        magnitudes,
        kls,
        envelope,
    )


def propose_beta(
    rows,
    *,
    target_kl: float,
    sign: float,
):
    magnitudes, _, envelope = monotone_curve(rows)

    if target_kl <= 0.0:
        return (
            0.0,
            True,
        )

    if envelope[-1] + 1e-15 < target_kl:
        return (
            float(sign * magnitudes[-1]),
            False,
        )

    frontier_magnitudes = [float(magnitudes[0])]

    frontier_kls = [float(envelope[0])]

    running = frontier_kls[0]

    for magnitude, value in zip(
        magnitudes[1:],
        envelope[1:],
    ):
        value = float(value)

        if value > running + 1e-15:
            frontier_magnitudes.append(float(magnitude))

            frontier_kls.append(value)

            running = value

    if frontier_kls[0] > 0.0:
        frontier_magnitudes.insert(
            0,
            0.0,
        )

        frontier_kls.insert(
            0,
            0.0,
        )

    magnitude = float(
        np.interp(
            np.sqrt(target_kl),
            np.sqrt(
                np.maximum(
                    frontier_kls,
                    0.0,
                )
            ),
            frontier_magnitudes,
        )
    )

    return (
        float(sign * magnitude),
        True,
    )


def ensure_curve_point(
    *,
    rows,
    lookup,
    path: Path,
    model,
    cache,
    config,
    seed_index: int,
    behavior: str,
    layer: int,
    direction: str,
    beta: float,
    vector,
    metadata,
):
    key = (
        int(seed_index),
        behavior,
        direction,
        round(
            float(beta),
            BETA_DIGITS,
        ),
    )

    if key in lookup:
        return lookup[key]

    print(
        f"    KL curve: "
        f"seed={seed_index:03d} "
        f"{behavior:18s} "
        f"{direction:8s} "
        f"beta={beta:+.5f}"
    )

    stats = off_target_kl(
        model,
        cache,
        behavior,
        layer,
        vector,
        beta,
    )

    row = {
        "control": (CONTROL_NAME),
        "model_slug": (config.model_slug),
        "model_label": (config.model_label),
        "behavior": (behavior),
        "layer": int(layer),
        "direction": (direction),
        "control_seed_index": int(seed_index),
        "beta": float(beta),
        "beta_magnitude": abs(float(beta)),
        "random_offtarget_kl_mean": (stats["mean"]),
        "random_offtarget_kl_std": (stats["std"]),
        "random_offtarget_kl_min": (stats["min"]),
        "random_offtarget_kl_max": (stats["max"]),
        "num_offtarget_prompts": (stats["n"]),
        "source_behavior_kl_json": (stats["by_source"]),
        "learned_vector_norm": (metadata["learned_norm"]),
        "random_vector_norm": (metadata["random_norm"]),
        "norm_ratio": (metadata["norm_ratio"]),
        "cosine_with_caa": (metadata["cosine"]),
    }

    rows.append(row)

    lookup[key] = row

    frame = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "control_seed_index",
                "behavior",
                "direction",
                "beta_magnitude",
            ]
        )
        .reset_index(drop=True)
    )

    write_csv(
        frame,
        path,
    )

    return row


# =============================================================================
# Summary
# =============================================================================


def summarize(
    results: pd.DataFrame,
):
    if results.empty:
        return pd.DataFrame()

    group_columns = [
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

    for values, group in results.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        base = dict(
            zip(
                group_columns,
                values,
            )
        )

        gains = group["matched_directional_gain"].astype(float).to_numpy()

        caa_gain = float(group["caa_test_directional_gain"].iloc[0])

        target_kl = float(group["caa_offtarget_kl_mean"].iloc[0])

        n = len(group)

        percentile = float(
            (np.sum(gains < caa_gain) + 0.5 * np.sum(gains == caa_gain)) / n
        )

        empirical_p = float((1 + np.sum(gains >= caa_gain)) / (n + 1))

        rows.append(
            {
                **base,
                "num_kl_matched_random_directions": (n),
                "caa_test_directional_gain": (caa_gain),
                "caa_offtarget_kl_mean": (target_kl),
                "matched_gain_mean": float(np.mean(gains)),
                "matched_gain_std": (
                    float(
                        np.std(
                            gains,
                            ddof=1,
                        )
                    )
                    if n > 1
                    else np.nan
                ),
                "matched_gain_q025": float(
                    np.quantile(
                        gains,
                        0.025,
                    )
                ),
                "matched_gain_median": float(
                    np.quantile(
                        gains,
                        0.50,
                    )
                ),
                "matched_gain_q975": float(
                    np.quantile(
                        gains,
                        0.975,
                    )
                ),
                "caa_minus_matched_mean": float(caa_gain - np.mean(gains)),
                "caa_gain_matched_percentile": (percentile),
                "empirical_p_matched_ge_caa": (empirical_p),
                "matched_direction_support_rate": float(
                    group["matched_direction_supported"].mean()
                ),
                "achieved_offtarget_kl_mean": float(
                    group["matched_offtarget_kl_mean"].mean()
                ),
                "mean_abs_kl_error": float(group["kl_match_abs_error"].mean()),
                "mean_relative_kl_error": float(group["kl_match_rel_error"].mean()),
                "max_relative_kl_error": float(group["kl_match_rel_error"].max()),
                "kl_match_success_rate": float(group["kl_match_success"].mean()),
                "mean_abs_beta_star": float(group["beta_star"].abs().mean()),
                "mean_abs_cosine_with_caa": float(
                    group["cosine_with_caa"].abs().mean()
                ),
                "max_abs_cosine_with_caa": float(group["cosine_with_caa"].abs().max()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1.")

    if args.offtarget_per_behavior < 1:
        raise ValueError("--offtarget-per-behavior must be >= 1.")

    if (
        args.match_rtol < 0.0
        or args.match_atol < 0.0
        or args.max_refine < 0
        or args.max_beta <= 0.0
    ):
        raise ValueError("Invalid KL matching arguments.")

    anchor_magnitudes = sorted(
        {
            float(value)
            for value in args.anchor_magnitudes
            if (float(value) > 0.0 and float(value) <= args.max_beta)
        }
    )

    if not anchor_magnitudes:
        raise ValueError("No valid anchor magnitudes.")

    if anchor_magnitudes[-1] < args.max_beta:
        anchor_magnitudes.append(float(args.max_beta))

    config = load_experiment_config(args.config)

    behaviors = (
        list(config.behaviors) if args.behaviors is None else list(args.behaviors)
    )

    unknown = set(behaviors) - set(config.behaviors)

    if unknown:
        raise ValueError(f"Unknown behaviors: {sorted(unknown)}")

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

    random_control_root = (
        args.random_control_root
        if args.random_control_root is not None
        else (config.results_root / "controls" / RANDOM_SOURCE_CONTROL)
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

    off_target_count = (len(config.behaviors) - 1) * args.offtarget_per_behavior

    print("=" * 76)
    print("OFF-TARGET-KL-MATCHED SPECIFICITY CONTROL")
    print("=" * 76)
    print(f"Model:              {config.model_label}")
    print(
        f"Seeds:              "
        f"{args.seed_start} .. "
        f"{args.seed_start + args.num_seeds - 1}"
    )
    print(f"Operating points:   {len(operating_points)}")
    print(f"Matched test evals: {len(operating_points) * args.num_seeds}")
    print(f"Off-target prompts: {off_target_count} / target behavior")
    print(f"KL tolerance:       atol={args.match_atol:g}, rtol={args.match_rtol:g}")
    print(f"Random vectors:     {random_control_root}")
    print(f"Output:             {output_root}")

    if args.plan_only:
        print()
        print("Plan only — model not loaded.")
        return

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    resume = not args.no_resume

    subset_path = output_root / "offtarget_subset.csv"

    targets_path = output_root / "caa_kl_targets.csv"

    curve_path = output_root / "random_kl_curve.csv"

    results_path = output_root / "kl_matched_results.csv"

    summary_path = output_root / "kl_matched_summary.csv"

    # ------------------------------------------------------------------
    # Fixed off-target subset
    # ------------------------------------------------------------------

    selected_examples = choose_offtarget_examples(
        config,
        per_behavior=(args.offtarget_per_behavior),
        base_seed=(args.base_seed),
    )

    subset_frame = (
        pd.DataFrame(
            [
                {
                    key: item[key]
                    for key in (
                        "source_behavior",
                        "source_index",
                        "question_sha256",
                        "subset_seed",
                    )
                }
                for item in selected_examples
            ]
        )
        .sort_values(
            [
                "source_behavior",
                "source_index",
            ]
        )
        .reset_index(drop=True)
    )

    if resume and subset_path.is_file():
        previous_subset = pd.read_csv(subset_path).reset_index(drop=True)

        if not previous_subset.equals(subset_frame):
            raise ValueError(
                "Existing off-target subset "
                "does not match the current "
                "deterministic subset."
            )

    else:
        write_csv(
            subset_frame,
            subset_path,
        )

    # ------------------------------------------------------------------
    # Model / prompt cache
    # ------------------------------------------------------------------

    print()
    print("Loading normalized CAA vectors...")

    normalized_vectors = load_normalized_vectors(config)

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
        prompt_family=(config.prompt_family),
    )

    print(
        "Prompt adapter:",
        adapter.family,
    )

    prompt_cache = build_prompt_cache(
        model,
        tokenizer,
        adapter,
        selected_examples,
        args.device,
    )

    # ------------------------------------------------------------------
    # Measure frozen CAA KL targets
    # ------------------------------------------------------------------

    existing_targets, target_done = load_existing(
        targets_path,
        resume=resume,
        key_fn=(intervention_key),
    )

    target_rows = (
        existing_targets.to_dict(orient="records") if not existing_targets.empty else []
    )

    print()
    print("=" * 76)
    print("CAA OFF-TARGET KL TARGETS")
    print("=" * 76)

    for point in operating_points.to_dict(orient="records"):
        if intervention_key(point) in target_done:
            continue

        behavior = str(point["behavior"])

        layer = int(point["layer"])

        alpha = float(point["alpha_star"])

        print(f"{behavior:18s} L={layer:2d} {point['direction']:8s} alpha={alpha:+.2f}")

        stats = off_target_kl(
            model,
            prompt_cache,
            behavior,
            layer,
            normalized_vectors[behavior][layer],
            alpha,
        )

        row = {
            "model_slug": (config.model_slug),
            "model_label": (config.model_label),
            "behavior": (behavior),
            "layer": (layer),
            "direction": (point["direction"]),
            "alpha_star": (alpha),
            "eta_star": float(point["eta_star"]),
            "frontier_epsilons": (point["frontier_epsilons"]),
            "caa_test_directional_gain": float(point["test_directional_gain"]),
            "caa_offtarget_kl_mean": (stats["mean"]),
            "caa_offtarget_kl_std": (stats["std"]),
            "caa_offtarget_kl_min": (stats["min"]),
            "caa_offtarget_kl_max": (stats["max"]),
            "num_offtarget_prompts": (stats["n"]),
            "source_behavior_kl_json": (stats["by_source"]),
        }

        target_rows.append(row)

        target_done.add(intervention_key(point))

        write_csv(
            (
                pd.DataFrame(target_rows)
                .sort_values(
                    [
                        "behavior",
                        "direction",
                        "alpha_star",
                    ]
                )
                .reset_index(drop=True)
            ),
            targets_path,
        )

    target_map = {intervention_key(row): row for row in target_rows}

    # ------------------------------------------------------------------
    # Target behavior test baselines
    # ------------------------------------------------------------------

    baselines = {}
    test_datasets = {}

    print()
    print("=" * 76)
    print("TARGET-BEHAVIOR TEST BASELINES")
    print("=" * 76)

    for behavior in behaviors:
        points = operating_points.loc[operating_points["behavior"] == behavior]

        if points.empty:
            continue

        layer = int(points["layer"].iloc[0])

        dataset = load_test_data(config.layout.test_data_path(behavior))

        test_datasets[behavior] = dataset

        print(f"{behavior:18s} L={layer:2d} N={len(dataset)}")

        _, baseline_items = evaluate_behavior_condition(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            dataset=dataset,
            layer=layer,
            vector=(normalized_vectors[behavior][layer]),
            alpha=0.0,
            device=args.device,
        )

        baselines[behavior] = as_dataframe(baseline_items)

    # ------------------------------------------------------------------
    # Resume random KL curves + matched evaluations
    # ------------------------------------------------------------------

    existing_curve, _ = load_existing(
        curve_path,
        resume=resume,
        key_fn=curve_key,
    )

    curve_rows = (
        existing_curve.to_dict(orient="records") if not existing_curve.empty else []
    )

    curve_lookup = {curve_key(row): row for row in curve_rows}

    existing_results, result_done = load_existing(
        results_path,
        resume=resume,
        key_fn=(matched_result_key),
    )

    result_rows = (
        existing_results.to_dict(orient="records") if not existing_results.empty else []
    )

    total = len(operating_points) * args.num_seeds

    print()
    print("=" * 76)
    print("KL-MATCHED RANDOM CONTROL")
    print("=" * 76)
    print(f"Expected matched evaluations: {total}")
    print(f"Already completed: {len(result_done)}")

    # ------------------------------------------------------------------
    # Seed / behavior loop
    # ------------------------------------------------------------------

    for seed_index in range(
        args.seed_start,
        (args.seed_start + args.num_seeds),
    ):
        print()
        print("-" * 76)
        print(f"SEED {seed_index:03d}")
        print("-" * 76)

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

            layer = int(points["layer"].iloc[0])

            (
                random_vector,
                saved_metadata,
                vector_path,
            ) = load_random_vector(
                random_control_root,
                model_slug=(config.model_slug),
                behavior=(behavior),
                seed_index=(seed_index),
            )

            vector_metadata = validate_random_vector(
                normalized_vectors[behavior][layer],
                random_vector,
                saved_metadata,
            )

            for (
                direction,
                direction_points,
            ) in points.groupby(
                "direction",
                sort=False,
            ):
                alphas = direction_points["alpha_star"].astype(float).to_numpy()

                if direction == "positive":
                    if not np.all(alphas > 0.0):
                        raise ValueError(
                            f"{behavior}: "
                            "positive frontier "
                            "contains non-positive alpha."
                        )

                    sign = 1.0

                elif direction == "negative":
                    if not np.all(alphas < 0.0):
                        raise ValueError(
                            f"{behavior}: "
                            "negative frontier "
                            "contains non-negative alpha."
                        )

                    sign = -1.0

                else:
                    raise ValueError(f"Unexpected direction: {direction}")

                target_kls = [
                    float(target_map[intervention_key(point)]["caa_offtarget_kl_mean"])
                    for point in direction_points.to_dict(orient="records")
                ]

                max_target_kl = max(target_kls)

                # Analytical KL=0 anchor.
                ensure_curve_point(
                    rows=curve_rows,
                    lookup=curve_lookup,
                    path=curve_path,
                    model=model,
                    cache=prompt_cache,
                    config=config,
                    seed_index=seed_index,
                    behavior=behavior,
                    layer=layer,
                    direction=direction,
                    beta=0.0,
                    vector=random_vector,
                    metadata=(vector_metadata),
                )

                # Evaluate sign-specific anchors only until the
                # largest CAA target KL is bracketed.
                for magnitude in anchor_magnitudes:
                    ensure_curve_point(
                        rows=curve_rows,
                        lookup=curve_lookup,
                        path=curve_path,
                        model=model,
                        cache=prompt_cache,
                        config=config,
                        seed_index=(seed_index),
                        behavior=behavior,
                        layer=layer,
                        direction=(direction),
                        beta=(sign * magnitude),
                        vector=random_vector,
                        metadata=(vector_metadata),
                    )

                    _, _, envelope = monotone_curve(
                        curve_rows_for(
                            curve_rows,
                            seed_index=(seed_index),
                            behavior=behavior,
                            direction=(direction),
                        )
                    )

                    if envelope[-1] >= max_target_kl:
                        break

                # ------------------------------------------------------
                # Match every frozen intervention using only off-target KL
                # ------------------------------------------------------

                for point in direction_points.to_dict(orient="records"):
                    completed_key = (
                        int(seed_index),
                        *intervention_key(point),
                    )

                    if completed_key in result_done:
                        continue

                    target = target_map[intervention_key(point)]

                    target_kl = float(target["caa_offtarget_kl_mean"])

                    # Interpolate in sqrt(KL), evaluate the proposal,
                    # then optionally refine using the enlarged curve.
                    for _ in range(args.max_refine + 1):
                        current_curve = curve_rows_for(
                            curve_rows,
                            seed_index=(seed_index),
                            behavior=behavior,
                            direction=(direction),
                        )

                        (
                            proposed_beta,
                            bracketed,
                        ) = propose_beta(
                            current_curve,
                            target_kl=(target_kl),
                            sign=sign,
                        )

                        proposed_beta = float(
                            np.clip(
                                proposed_beta,
                                -args.max_beta,
                                args.max_beta,
                            )
                        )

                        if any(
                            np.isclose(
                                float(row["beta"]),
                                proposed_beta,
                                atol=1e-6,
                                rtol=0.0,
                            )
                            for row in current_curve
                        ):
                            break

                        candidate = ensure_curve_point(
                            rows=(curve_rows),
                            lookup=(curve_lookup),
                            path=(curve_path),
                            model=model,
                            cache=(prompt_cache),
                            config=config,
                            seed_index=(seed_index),
                            behavior=(behavior),
                            layer=(layer),
                            direction=(direction),
                            beta=(proposed_beta),
                            vector=(random_vector),
                            metadata=(vector_metadata),
                        )

                        error = abs(
                            float(candidate["random_offtarget_kl_mean"]) - target_kl
                        )

                        tolerance = max(
                            args.match_atol,
                            (args.match_rtol * target_kl),
                        )

                        if error <= tolerance:
                            break

                        if not bracketed and abs(proposed_beta) >= (
                            args.max_beta - 1e-8
                        ):
                            break

                    # Pick the actually evaluated coefficient with the
                    # closest measured KL.
                    current_curve = curve_rows_for(
                        curve_rows,
                        seed_index=(seed_index),
                        behavior=behavior,
                        direction=(direction),
                    )

                    best = min(
                        current_curve,
                        key=lambda row: abs(
                            float(row["random_offtarget_kl_mean"]) - target_kl
                        ),
                    )

                    beta_star = float(best["beta"])

                    achieved_kl = float(best["random_offtarget_kl_mean"])

                    absolute_error = abs(achieved_kl - target_kl)

                    relative_error = absolute_error / max(
                        target_kl,
                        1e-12,
                    )

                    tolerance = max(
                        args.match_atol,
                        (args.match_rtol * target_kl),
                    )

                    match_success = bool(absolute_error <= tolerance)

                    print(
                        f"[{len(result_done) + 1:4d}/"
                        f"{total}] "
                        f"seed={seed_index:03d} "
                        f"{behavior:18s} "
                        f"{direction:8s} "
                        f"alpha="
                        f"{float(point['alpha_star']):+.2f} "
                        f"beta*={beta_star:+.4f} "
                        f"KL={achieved_kl:.6g}/"
                        f"{target_kl:.6g} "
                        f"relerr="
                        f"{relative_error:.3f} "
                        f"{'OK' if match_success else 'MISS'}"
                    )

                    # --------------------------------------------------
                    # beta* is now frozen. Touch target test set only now.
                    # --------------------------------------------------

                    (
                        _,
                        matched_items,
                    ) = evaluate_behavior_condition(
                        model=model,
                        tokenizer=tokenizer,
                        adapter=adapter,
                        dataset=(test_datasets[behavior]),
                        layer=layer,
                        vector=(random_vector),
                        alpha=(beta_star),
                        device=(args.device),
                    )

                    bootstrap_seed = stable_seed(
                        args.base_seed,
                        CONTROL_NAME,
                        "bootstrap",
                        config.model_slug,
                        behavior,
                        seed_index,
                        layer,
                        direction,
                        float(point["alpha_star"]),
                    )

                    stats = behavior_statistics(
                        baseline_df=(baselines[behavior]),
                        steered_df=(as_dataframe(matched_items)),
                        behavior=(behavior),
                        alpha=(beta_star),
                        bootstrap_samples=(args.bootstrap_samples),
                        bootstrap_seed=(bootstrap_seed),
                    )

                    canonical = canonical_matched_statistics(
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
                        "alpha_star": float(point["alpha_star"]),
                        "eta_star": float(point["eta_star"]),
                        "frontier_epsilons": (point["frontier_epsilons"]),
                        "control_seed_index": (seed_index),
                        "random_vector_path": str(vector_path),
                        "bootstrap_seed": int(bootstrap_seed),
                        "learned_vector_norm": (vector_metadata["learned_norm"]),
                        "random_vector_norm": (vector_metadata["random_norm"]),
                        "norm_ratio": (vector_metadata["norm_ratio"]),
                        "cosine_with_caa": (vector_metadata["cosine"]),
                        "selection_behavior_gain": float(
                            point["selection_behavior_gain"]
                        ),
                        "caa_test_behavior_delta": float(point["test_behavior_delta"]),
                        "caa_test_behavior_delta_ci_low": float(
                            point["test_behavior_delta_ci_low"]
                        ),
                        "caa_test_behavior_delta_ci_high": float(
                            point["test_behavior_delta_ci_high"]
                        ),
                        "caa_test_directional_gain": float(
                            point["test_directional_gain"]
                        ),
                        "caa_test_directional_gain_ci_low": float(
                            point["test_directional_gain_ci_low"]
                        ),
                        "caa_test_directional_gain_ci_high": float(
                            point["test_directional_gain_ci_high"]
                        ),
                        "caa_test_direction_supported": bool(
                            point["test_direction_supported"]
                        ),
                        "caa_offtarget_kl_mean": (target_kl),
                        "caa_offtarget_kl_std": float(target["caa_offtarget_kl_std"]),
                        "beta_star": (beta_star),
                        "matched_offtarget_kl_mean": (achieved_kl),
                        "matched_offtarget_kl_std": float(
                            best["random_offtarget_kl_std"]
                        ),
                        "kl_match_abs_error": (absolute_error),
                        "kl_match_rel_error": (relative_error),
                        "kl_match_tolerance": (tolerance),
                        "kl_match_success": (match_success),
                        "num_offtarget_prompts": int(best["num_offtarget_prompts"]),
                        **canonical,
                    }

                    result_rows.append(row)

                    result_done.add(completed_key)

                    result_frame = (
                        pd.DataFrame(result_rows)
                        .sort_values(
                            [
                                "control_seed_index",
                                "behavior",
                                "direction",
                                "alpha_star",
                            ]
                        )
                        .reset_index(drop=True)
                    )

                    # Persist after every final matched behavior evaluation.
                    write_csv(
                        result_frame,
                        results_path,
                    )

    # ------------------------------------------------------------------
    # Summary / manifest
    # ------------------------------------------------------------------

    results = pd.DataFrame(result_rows)

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

    summary = summarize(results)

    write_csv(
        summary,
        summary_path,
    )

    random_manifest = random_control_root / "manifest.json"

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
        "random_control_root": str(random_control_root),
        "random_control_manifest_sha256": (
            sha256_file(random_manifest) if random_manifest.is_file() else None
        ),
        "random_vector_policy": (
            "Reuse exact saved norm-matched "
            "isotropic Gaussian directions from "
            "the completed norm_matched_random control."
        ),
        "offtarget_split": ("validation"),
        "offtarget_examples_per_source_behavior": (args.offtarget_per_behavior),
        "offtarget_prompts_per_target_behavior": (off_target_count),
        "offtarget_subset_policy": (
            "One deterministic fixed subset per source behavior; "
            "the target behavior is excluded during calibration. "
            "Selection seed does not include model slug."
        ),
        "kl_definition": (
            "Mean full-vocabulary next-token "
            "D_KL(p0 || p_intervened) at the model-family "
            "PromptAdapter scoring position."
        ),
        "matching_policy": (
            "For each saved random direction and frozen CAA operating "
            "point, beta has the same sign as alpha_star. "
            "A sign-specific random KL curve is evaluated only on "
            "off-target validation prompts; sqrt(KL) interpolation "
            "and optional refinement select beta. "
            "Target behavior test data is evaluated only after beta "
            "is frozen."
        ),
        "match_rtol": (args.match_rtol),
        "match_atol": (args.match_atol),
        "max_refine": (args.max_refine),
        "max_beta": (args.max_beta),
        "anchor_magnitudes": (anchor_magnitudes),
        "num_unique_frozen_nonzero_operating_points": int(len(operating_points)),
        "num_completed_control_evaluations": int(len(results)),
        "evaluation_split": ("behavior test after KL matching is frozen"),
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
    print("OFF-TARGET-KL-MATCHED CONTROL COMPLETE")
    print("=" * 76)
    print(f"Control evaluations: {len(results)}")
    print(f"Unique operating pts: {len(summary)}")
    print(f"CAA KL targets:       {targets_path}")
    print(f"Random KL curve:      {curve_path}")
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
                    "matched_gain_mean",
                    "matched_gain_q975",
                    "kl_match_success_rate",
                    "mean_relative_kl_error",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
