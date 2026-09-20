"""Evaluate frozen steering interventions on WikiText-2 perplexity.

A shared frozen corpus prefix and fixed target-token budget are used across all
conditions so NLL changes are directly comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from steerfrontier.config import load_experiment_config
from steerfrontier.models import (
    get_decoder_layers,
    load_model,
)
from steerfrontier.workflow import load_normalized_vectors

# =============================================================================
# Frozen protocol
# =============================================================================

DEFAULT_EPSILON = 0.03

DEFAULT_SCORED_TOKENS = 32_768

DEFAULT_TARGETS_PER_WINDOW = 256

DEFAULT_INPUT_TOKENS_PER_WINDOW = DEFAULT_TARGETS_PER_WINDOW + 1

DEFAULT_NUM_WINDOWS = DEFAULT_SCORED_TOKENS // DEFAULT_TARGETS_PER_WINDOW

DEFAULT_BATCH_SIZE = 4

DEFAULT_BOOTSTRAP_SAMPLES = 10_000

DEFAULT_DATA_PATH = Path("data/secondary/wikitext2_test.txt")

DEFAULT_DATA_MANIFEST_PATH = Path("data/secondary/wikitext2_test_manifest.json")


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen CAA frontier interventions "
            "on WikiText-2 next-token negative log-likelihood."
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
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )

    parser.add_argument(
        "--data-manifest-path",
        type=Path,
        default=DEFAULT_DATA_MANIFEST_PATH,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--behavior",
        default=None,
        help="Optional single behavior for smoke/targeted runs.",
    )

    parser.add_argument(
        "--direction",
        choices=[
            "positive",
            "negative",
        ],
        default=None,
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=(
            "Optional truncation for smoke testing. Requires a separate --output-root."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# Steering
# =============================================================================


class SustainedSteeringIntervention:
    """
    Add one frozen CAA vector to every token position at one decoder layer.

    This differs deliberately from instruction-generation evaluation:
    WikiText perplexity has no model-defined answer-generation boundary,
    so the intervention is applied persistently across the entire
    257-token evaluation window.
    """

    def __init__(
        self,
        *,
        model,
        layer_idx: int,
        vector: torch.Tensor,
        coefficient: float,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.vector = vector
        self.coefficient = float(coefficient)

        self.decoder_layers = get_decoder_layers(model)

        if self.layer_idx < 0 or self.layer_idx >= len(self.decoder_layers):
            raise ValueError(
                f"Invalid layer "
                f"{self.layer_idx}; "
                f"model has "
                f"{len(self.decoder_layers)} "
                f"layers."
            )

        self.handles = []

    def _hook(
        self,
        module,
        args,
        output,
    ):
        hidden_states = (
            output[0]
            if isinstance(
                output,
                tuple,
            )
            else output
        )

        vector = self.coefficient * self.vector.to(
            device=(hidden_states.device),
            dtype=(hidden_states.dtype),
        )

        steered_hidden = hidden_states + vector.view(
            1,
            1,
            -1,
        )

        if isinstance(
            output,
            tuple,
        ):
            return (steered_hidden,) + output[1:]

        return steered_hidden

    def __enter__(
        self,
    ):
        layer = self.decoder_layers[self.layer_idx]

        handle = layer.register_forward_hook(self._hook)

        self.handles.append(handle)

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()


# =============================================================================
# General helpers
# =============================================================================


def stable_seed(
    *parts,
):
    payload = "|".join(str(part) for part in parts).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    ) % (2**32)


def sha256_bytes(
    payload: bytes,
):
    return hashlib.sha256(payload).hexdigest()


def token_stream_sha256(
    token_ids,
):
    array = np.asarray(
        token_ids,
        dtype="<i8",
    )

    return sha256_bytes(array.tobytes())


def write_csv(
    frame: pd.DataFrame,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    frame.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(path)


def condition_name(
    behavior: str,
    direction: str,
):
    return f"{behavior}__{direction}"


# =============================================================================
# Frozen operating points
# =============================================================================


def load_operating_points(
    config,
    *,
    epsilon: float,
    behavior=None,
    direction=None,
):
    path = config.results_root / "final_evaluation" / "frontier_test_results.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "epsilon",
        "behavior",
        "direction",
        "layer",
        "alpha_star",
        "eta_star",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    frame = frame[
        np.isclose(
            frame["epsilon"].astype(float),
            float(epsilon),
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()

    if behavior is not None:
        frame = frame[frame["behavior"] == behavior].copy()

    if direction is not None:
        frame = frame[frame["direction"] == direction].copy()

    duplicated = frame.duplicated(
        [
            "behavior",
            "direction",
        ]
    )

    if duplicated.any():
        raise ValueError("Duplicate frozen operating points.")

    if frame.empty:
        raise ValueError("No frozen operating points matched.")

    return frame.sort_values(
        [
            "behavior",
            "direction",
        ]
    ).reset_index(drop=True)


# =============================================================================
# Frozen WikiText token stream
# =============================================================================


def load_frozen_corpus(
    *,
    data_path: Path,
    manifest_path: Path,
):
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    text = data_path.read_text(encoding="utf-8")

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    observed_hash = sha256_bytes(text.encode("utf-8"))

    expected_hash = manifest.get("text_sha256")

    if expected_hash is not None and observed_hash != expected_hash:
        raise RuntimeError("Frozen WikiText corpus hash does not match its manifest.")

    return (
        text,
        observed_hash,
        manifest,
    )


def build_token_windows(
    *,
    tokenizer,
    text: str,
    num_windows: int,
):
    required_tokens = num_windows * DEFAULT_TARGETS_PER_WINDOW + 1

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(token_ids) < required_tokens:
        raise RuntimeError(
            f"Tokenizer produced only "
            f"{len(token_ids):,} tokens; "
            f"need at least "
            f"{required_tokens:,}."
        )

    frozen_ids = token_ids[:required_tokens]

    windows = []

    for window_index in range(num_windows):
        start = window_index * DEFAULT_TARGETS_PER_WINDOW

        end = start + DEFAULT_INPUT_TOKENS_PER_WINDOW

        window = frozen_ids[start:end]

        if len(window) != DEFAULT_INPUT_TOKENS_PER_WINDOW:
            raise RuntimeError(f"Unexpected incomplete window {window_index}.")

        windows.append(window)

    tensor = torch.tensor(
        windows,
        dtype=torch.long,
    )

    if tensor.shape != (
        num_windows,
        DEFAULT_INPUT_TOKENS_PER_WINDOW,
    ):
        raise RuntimeError(f"Unexpected WikiText tensor shape {tuple(tensor.shape)}.")

    return {
        "windows": tensor,
        "frozen_token_ids": (frozen_ids),
        "total_tokenizer_tokens": int(len(token_ids)),
        "required_tokens": int(required_tokens),
        "token_stream_sha256": (token_stream_sha256(frozen_ids)),
    }


# =============================================================================
# NLL evaluation
# =============================================================================


@torch.inference_mode()
def evaluate_condition(
    *,
    model,
    windows: torch.Tensor,
    device,
    batch_size: int,
    description: str,
    layer=None,
    vector=None,
    alpha: float = 0.0,
):
    rows = []

    num_windows = int(windows.shape[0])

    progress = tqdm(
        total=num_windows,
        desc=description,
    )

    for start in range(
        0,
        num_windows,
        batch_size,
    ):
        end = min(
            start + batch_size,
            num_windows,
        )

        input_ids = windows[start:end].to(device)

        if vector is None or abs(float(alpha)) <= 1e-12:
            outputs = model(
                input_ids=input_ids,
                use_cache=False,
            )

        else:
            with SustainedSteeringIntervention(
                model=model,
                layer_idx=int(layer),
                vector=vector,
                coefficient=float(alpha),
            ):
                outputs = model(
                    input_ids=input_ids,
                    use_cache=False,
                )

        # Position t predicts token t+1.
        #
        # Each 257-token window therefore
        # contributes exactly 256 scored
        # next-token targets.
        logits = (
            outputs.logits[
                :,
                :-1,
                :,
            ]
            .float()
            .contiguous()
        )

        targets = input_ids[
            :,
            1:,
        ].contiguous()

        batch_size_actual = int(targets.shape[0])

        scored_tokens = int(targets.shape[1])

        if scored_tokens != DEFAULT_TARGETS_PER_WINDOW:
            raise RuntimeError(
                "Expected "
                f"{DEFAULT_TARGETS_PER_WINDOW} "
                "targets/window, found "
                f"{scored_tokens}."
            )

        token_nll = F.cross_entropy(
            logits.view(
                -1,
                logits.shape[-1],
            ),
            targets.view(-1),
            reduction="none",
        ).view(
            batch_size_actual,
            scored_tokens,
        )

        window_nll_sum = token_nll.sum(dim=1).detach().cpu().double().numpy()

        window_nll_mean = token_nll.mean(dim=1).detach().cpu().double().numpy()

        for local_index in range(batch_size_actual):
            window_index = start + local_index

            source_start = window_index * DEFAULT_TARGETS_PER_WINDOW

            rows.append(
                {
                    "window_index": int(window_index),
                    "source_start_token": int(source_start),
                    "source_end_token_exclusive": int(
                        source_start + DEFAULT_INPUT_TOKENS_PER_WINDOW
                    ),
                    "input_tokens": int(DEFAULT_INPUT_TOKENS_PER_WINDOW),
                    "scored_tokens": int(DEFAULT_TARGETS_PER_WINDOW),
                    "nll_sum": float(window_nll_sum[local_index]),
                    "mean_nll": float(window_nll_mean[local_index]),
                }
            )

        progress.update(batch_size_actual)

        del outputs
        del logits
        del targets
        del token_nll
        del input_ids

    progress.close()

    return pd.DataFrame(rows)


# =============================================================================
# Paired statistics
# =============================================================================


def validate_window_frame(
    frame: pd.DataFrame,
    *,
    expected_windows: int,
):
    required = {
        "window_index",
        "scored_tokens",
        "nll_sum",
        "mean_nll",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"Window result missing columns: {sorted(missing)}")

    if len(frame) != expected_windows:
        raise ValueError(f"Expected {expected_windows} windows, found {len(frame)}.")

    expected_indices = set(range(expected_windows))

    observed_indices = set(frame["window_index"].astype(int).tolist())

    if observed_indices != expected_indices:
        raise ValueError("Window indices do not match requested corpus prefix.")

    scored = frame["scored_tokens"].astype(int)

    if not (scored == DEFAULT_TARGETS_PER_WINDOW).all():
        raise ValueError("Not every window contains exactly 256 scored targets.")


def aggregate_nll(
    frame: pd.DataFrame,
):
    total_nll = float(frame["nll_sum"].sum())

    total_tokens = int(frame["scored_tokens"].sum())

    if total_tokens <= 0:
        raise ValueError("No scored tokens.")

    mean_nll = total_nll / total_tokens

    perplexity = math.exp(mean_nll)

    return {
        "total_nll": float(total_nll),
        "num_scored_tokens": int(total_tokens),
        "mean_nll": float(mean_nll),
        "perplexity": float(perplexity),
    }


def paired_nll_statistics(
    *,
    baseline: pd.DataFrame,
    steered: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
):
    merged = (
        baseline[
            [
                "window_index",
                "mean_nll",
            ]
        ]
        .merge(
            steered[
                [
                    "window_index",
                    "mean_nll",
                ]
            ],
            on="window_index",
            suffixes=(
                "_baseline",
                "_steered",
            ),
            validate="one_to_one",
        )
        .sort_values("window_index")
        .reset_index(drop=True)
    )

    baseline_values = merged["mean_nll_baseline"].astype(float).to_numpy()

    steered_values = merged["mean_nll_steered"].astype(float).to_numpy()

    differences = steered_values - baseline_values

    delta_nll = float(differences.mean())

    if np.allclose(
        differences,
        0.0,
        atol=0.0,
        rtol=0.0,
    ):
        ci_low = 0.0
        ci_high = 0.0

    else:
        rng = np.random.default_rng(seed)

        indices = rng.integers(
            0,
            len(differences),
            size=(
                bootstrap_samples,
                len(differences),
            ),
        )

        bootstrap = differences[indices].mean(axis=1)

        (
            ci_low,
            ci_high,
        ) = (
            np.quantile(
                bootstrap,
                [
                    0.025,
                    0.975,
                ],
            )
            .astype(float)
            .tolist()
        )

    baseline_stats = aggregate_nll(baseline)

    steered_stats = aggregate_nll(steered)

    # Since every window contains the
    # same 256 scored targets:
    #
    # mean(window-level delta NLL)
    # ==
    # full token-level delta NLL.
    exact_delta_nll = float(steered_stats["mean_nll"] - baseline_stats["mean_nll"])

    if not np.isclose(
        delta_nll,
        exact_delta_nll,
        atol=1e-8,
        rtol=0.0,
    ):
        raise RuntimeError("Window-paired and token-aggregated delta NLL disagree.")

    ppl_ratio = math.exp(exact_delta_nll)

    return {
        "num_windows": int(len(merged)),
        "num_scored_tokens": int(steered_stats["num_scored_tokens"]),
        "baseline_mean_nll": float(baseline_stats["mean_nll"]),
        "mean_nll": float(steered_stats["mean_nll"]),
        "delta_nll": float(exact_delta_nll),
        "delta_nll_ci_low": float(ci_low),
        "delta_nll_ci_high": float(ci_high),
        "baseline_perplexity": float(baseline_stats["perplexity"]),
        "perplexity": float(steered_stats["perplexity"]),
        "perplexity_ratio": float(ppl_ratio),
        "perplexity_percent_change": float(100.0 * (ppl_ratio - 1.0)),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be >= 1.")

    if args.max_windows is not None:
        if args.max_windows < 1 or args.max_windows > DEFAULT_NUM_WINDOWS:
            raise ValueError(f"--max-windows must be in [1, {DEFAULT_NUM_WINDOWS}].")

        if args.output_root is None:
            raise ValueError(
                "--max-windows is for "
                "smoke testing and "
                "requires a separate "
                "--output-root."
            )

    num_windows = (
        args.max_windows if args.max_windows is not None else DEFAULT_NUM_WINDOWS
    )

    config = load_experiment_config(args.config)

    output_root = (
        args.output_root
        if args.output_root is not None
        else (config.results_root / "secondary_capability" / "perplexity")
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_root = output_root / "windows"

    window_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        corpus_text,
        corpus_sha256,
        data_manifest,
    ) = load_frozen_corpus(
        data_path=(args.data_path),
        manifest_path=(args.data_manifest_path),
    )

    operating_points = load_operating_points(
        config,
        epsilon=(args.epsilon),
        behavior=(args.behavior),
        direction=(args.direction),
    )

    print("=" * 88)
    print("WIKITEXT-2 PERPLEXITY SECONDARY CAPABILITY")
    print("=" * 88)
    print(f"Model:              {config.model_label}")
    print(f"Model slug:         {config.model_slug}")
    print(f"Epsilon:            {args.epsilon}")
    print(f"Operating points:   {len(operating_points)}")
    print(f"Windows:            {num_windows}")
    print(f"Input tokens/window: {DEFAULT_INPUT_TOKENS_PER_WINDOW}")
    print(f"Targets/window:     {DEFAULT_TARGETS_PER_WINDOW}")
    print(f"Scored tokens:      {num_windows * DEFAULT_TARGETS_PER_WINDOW:,}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Corpus SHA256:      {corpus_sha256}")
    print(f"Output:             {output_root}")

    print()
    print("Loading normalized vectors...")

    vectors = load_normalized_vectors(config)

    print("Loading model...")

    (
        model,
        tokenizer,
    ) = load_model(
        model_name=(config.model_name),
        device=(args.device),
        dtype=(config.torch_dtype),
    )

    model.eval()

    print()
    print("Tokenizing frozen corpus...")

    token_data = build_token_windows(
        tokenizer=tokenizer,
        text=corpus_text,
        num_windows=(num_windows),
    )

    windows = token_data["windows"]

    token_stream_hash = token_data["token_stream_sha256"]

    print(f"Tokenizer tokens:   {token_data['total_tokenizer_tokens']:,}")
    print(f"Frozen prefix toks: {token_data['required_tokens']:,}")
    print(f"Token SHA256:       {token_stream_hash}")

    # ----------------------------------------------------------------
    # Baseline
    # ----------------------------------------------------------------

    baseline_path = output_root / "baseline_windows.csv"

    if baseline_path.is_file() and not args.no_resume:
        baseline = pd.read_csv(baseline_path)

        validate_window_frame(
            baseline,
            expected_windows=(num_windows),
        )

        print()
        print(f"Reusing baseline: {baseline_path}")

    else:
        print()
        print("=" * 88)
        print("BASELINE")
        print("=" * 88)

        baseline = evaluate_condition(
            model=model,
            windows=windows,
            device=(args.device),
            batch_size=(args.batch_size),
            description=("WikiText baseline"),
        )

        validate_window_frame(
            baseline,
            expected_windows=(num_windows),
        )

        write_csv(
            baseline,
            baseline_path,
        )

    baseline_stats = aggregate_nll(baseline)

    print()
    print(f"Baseline mean NLL:  {baseline_stats['mean_nll']:.6f}")
    print(f"Baseline perplexity: {baseline_stats['perplexity']:.4f}")

    # ----------------------------------------------------------------
    # Resume state
    # ----------------------------------------------------------------

    summary_path = output_root / "perplexity_summary.csv"

    if summary_path.is_file() and not args.no_resume:
        summary = pd.read_csv(summary_path)

        completed = {
            (
                str(row["behavior"]),
                str(row["direction"]),
            )
            for row in (summary.to_dict(orient="records"))
        }

        summary_rows = summary.to_dict(orient="records")

    else:
        completed = set()
        summary_rows = []

    # ----------------------------------------------------------------
    # Frozen interventions
    # ----------------------------------------------------------------

    for row in operating_points.to_dict(orient="records"):
        behavior = str(row["behavior"])

        direction = str(row["direction"])

        key = (
            behavior,
            direction,
        )

        if key in completed:
            print(f"Skipping completed: {behavior} / {direction}")
            continue

        layer = int(row["layer"])

        alpha = float(row["alpha_star"])

        eta = float(row["eta_star"])

        print()
        print("=" * 88)
        print(f"{behavior} / {direction}")
        print("=" * 88)
        print(f"Layer: {layer}")
        print(f"alpha: {alpha:+.6f}")
        print(f"eta:   {eta:+.6f}")

        condition_path = window_root / (
            condition_name(
                behavior,
                direction,
            )
            + ".csv"
        )

        # If the forward pass finished and
        # wrote its window file before a
        # later interruption, reuse it.
        if condition_path.is_file() and not args.no_resume:
            steered = pd.read_csv(condition_path)

            validate_window_frame(
                steered,
                expected_windows=(num_windows),
            )

            print(f"Reusing condition windows: {condition_path}")

        elif abs(alpha) <= 1e-12:
            # A frozen zero-strength
            # frontier point is exactly
            # the baseline condition.
            steered = baseline.copy()

            write_csv(
                steered,
                condition_path,
            )

        else:
            if behavior not in vectors:
                raise KeyError(f"No normalized vector for {behavior}.")

            behavior_vectors = vectors[behavior]

            if behavior_vectors.ndim != 2:
                raise ValueError(
                    f"Expected vectors[{behavior!r}] to have shape "
                    f"[num_layers, hidden_size], found "
                    f"{tuple(behavior_vectors.shape)}."
                )

            if layer < 0 or layer >= behavior_vectors.shape[0]:
                raise IndexError(
                    f"Layer {layer} is out of range for {behavior}; "
                    f"available layers: 0..{behavior_vectors.shape[0] - 1}."
                )

            steering_vector = behavior_vectors[layer]

            steered = evaluate_condition(
                model=model,
                windows=windows,
                device=args.device,
                batch_size=args.batch_size,
                description=f"WikiText {behavior} {direction}",
                layer=layer,
                vector=steering_vector,
                alpha=alpha,
            )

            validate_window_frame(
                steered,
                expected_windows=(num_windows),
            )

            write_csv(
                steered,
                condition_path,
            )

        bootstrap_seed = stable_seed(
            "wikitext2_perplexity",
            config.model_slug,
            behavior,
            direction,
            args.epsilon,
            num_windows,
        )

        stats = paired_nll_statistics(
            baseline=baseline,
            steered=steered,
            bootstrap_samples=(args.bootstrap_samples),
            seed=(bootstrap_seed),
        )

        summary_row = {
            "model_name": (config.model_name),
            "model_label": (config.model_label),
            "model_slug": (config.model_slug),
            "epsilon": float(args.epsilon),
            "behavior": (behavior),
            "direction": (direction),
            "layer": int(layer),
            "alpha_star": float(alpha),
            "eta_star": float(eta),
            **stats,
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
        }

        summary_rows.append(summary_row)

        completed.add(key)

        summary = (
            pd.DataFrame(summary_rows)
            .sort_values(
                [
                    "behavior",
                    "direction",
                ]
            )
            .reset_index(drop=True)
        )

        write_csv(
            summary,
            summary_path,
        )

        print(f"mean NLL:     {stats['mean_nll']:.6f}")
        print(
            f"delta NLL:    "
            f"{stats['delta_nll']:+.6f} "
            f"["
            f"{stats['delta_nll_ci_low']:+.6f}, "
            f"{stats['delta_nll_ci_high']:+.6f}"
            f"]"
        )
        print(f"perplexity:   {stats['perplexity']:.4f}")
        print(f"PPL ratio:    {stats['perplexity_ratio']:.6f}")
        print(f"PPL change:   {stats['perplexity_percent_change']:+.3f}%")

    # ----------------------------------------------------------------
    # Manifest
    # ----------------------------------------------------------------

    summary = pd.DataFrame(summary_rows)

    manifest = {
        "analysis": ("wikitext2_perplexity_secondary_capability"),
        "protocol_version": ("wikitext2_raw_test_32768_targets_v1"),
        "model_name": (config.model_name),
        "model_label": (config.model_label),
        "model_slug": (config.model_slug),
        "frontier_epsilon": float(args.epsilon),
        "dataset": (
            data_manifest.get(
                "dataset",
                "Salesforce/wikitext",
            )
        ),
        "dataset_configuration": (
            data_manifest.get(
                "configuration",
                "wikitext-2-raw-v1",
            )
        ),
        "dataset_split": (
            data_manifest.get(
                "split",
                "test",
            )
        ),
        "corpus_path": str(args.data_path),
        "corpus_sha256": (corpus_sha256),
        "model_token_stream_sha256": (token_stream_hash),
        "tokenization": ("Model-specific tokenizer; add_special_tokens=False."),
        "num_windows": int(num_windows),
        "input_tokens_per_window": int(DEFAULT_INPUT_TOKENS_PER_WINDOW),
        "targets_per_window": int(DEFAULT_TARGETS_PER_WINDOW),
        "num_scored_tokens": int(num_windows * DEFAULT_TARGETS_PER_WINDOW),
        "window_stride": int(DEFAULT_TARGETS_PER_WINDOW),
        "window_overlap_tokens": 1,
        "steering_scope": (
            "Frozen CAA vector added at every token position in each evaluation window."
        ),
        "primary_metric": (
            "Delta mean next-token NLL relative to the same model's unsteered baseline."
        ),
        "secondary_metric": ("Perplexity ratio exp(delta_NLL)."),
        "bootstrap_unit": (
            "257-token evaluation window; each window contributes 256 scored targets."
        ),
        "bootstrap_samples": int(args.bootstrap_samples),
        "absolute_ppl_note": (
            "Absolute perplexity should not "
            "be compared across models with "
            "different tokenizers; inference "
            "uses within-model delta NLL and "
            "perplexity ratios."
        ),
        "num_operating_points_requested": int(len(operating_points)),
        "num_operating_points_completed": int(len(summary)),
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
    print("=" * 88)
    print("PERPLEXITY EVALUATION COMPLETE")
    print("=" * 88)
    print(f"Baseline NLL:       {baseline_stats['mean_nll']:.6f}")
    print(f"Baseline PPL:       {baseline_stats['perplexity']:.4f}")
    print(f"Completed points:   {len(summary)}/{len(operating_points)}")
    print(f"Summary:            {summary_path}")
    print(f"Manifest:           {manifest_path}")


if __name__ == "__main__":
    main()
