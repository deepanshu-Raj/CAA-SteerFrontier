"""Evaluate frozen steering interventions on the GSM8K secondary benchmark.

Flexible numeric accuracy is primary; strict answer-marker parsing and generation
stability metrics are retained as diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from steerfrontier.config import load_experiment_config
from steerfrontier.models import (
    get_decoder_layers,
    load_model,
)
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.workflow import load_normalized_vectors

# =============================================================================
# Frozen protocol
# =============================================================================

PROTOCOL_VERSION = "gsm8k_secondary_64_8shot_clean_v1"

DEFAULT_EPSILON = 0.03
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_BATCH_SIZE = 4

EXPECTED_TEST_EXAMPLES = 64
EXPECTED_NUM_SHOTS = 8

DEFAULT_DATA_PATH = Path("data/secondary/gsm8k_test_64.json")

DEFAULT_DEMOS_PATH = Path("data/secondary/gsm8k_fewshot_8.json")


NUMBER_PATTERN = re.compile(
    r"[-+]?"
    r"(?:\d[\d,]*)"
    r"(?:\.\d+)?"
    r"(?:/\d[\d,]*)?"
)

# Strict score requires an actual #### answer line.
STRICT_FINAL_PATTERN = re.compile(
    r"####\s*"
    r"([-+]?"
    r"(?:\d[\d,]*)"
    r"(?:\.\d+)?"
    r"(?:/\d[\d,]*)?)"
    r"\s*\Z"
)


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen CAA frontier interventions on the "
            "fixed 64-example, 8-shot GSM8K secondary reasoning probe."
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
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )

    parser.add_argument(
        "--demos-path",
        type=Path,
        default=DEFAULT_DEMOS_PATH,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--behavior",
        default=None,
        help=("Optional single behavior for smoke/targeted runs."),
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
        "--max-examples",
        type=int,
        default=None,
        help=("Optional truncation for smoke testing. Use a separate --output-root."),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# Batched steering
# =============================================================================


class BatchedSteeringIntervention:
    """
    Apply one steering vector/coefficient to a batch while allowing
    every example to have its own logical steering-start position.

    With left padding, each prompt retains its original logical token
    positions. If a model exposes only physical cache_position, the
    left-padding offset is removed before comparing with from_positions.
    """

    def __init__(
        self,
        *,
        model,
        layer_idx: int,
        vector: torch.Tensor,
        coefficient: float,
        from_positions: torch.Tensor,
        left_padding: torch.Tensor,
    ):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.vector = vector
        self.coefficient = float(coefficient)

        self.from_positions = torch.as_tensor(
            from_positions,
            dtype=torch.long,
        ).reshape(-1)

        self.left_padding = torch.as_tensor(
            left_padding,
            dtype=torch.long,
        ).reshape(-1)

        if self.from_positions.shape != self.left_padding.shape:
            raise ValueError(
                "from_positions and left_padding must have the same shape."
            )

        self.decoder_layers = get_decoder_layers(model)

        if self.layer_idx < 0 or self.layer_idx >= len(self.decoder_layers):
            raise ValueError(
                f"Invalid layer {self.layer_idx}; "
                f"model has "
                f"{len(self.decoder_layers)} layers."
            )

        self.handles = []

    def _resolve_positions(
        self,
        hidden_states,
        kwargs,
    ):
        batch_size = int(hidden_states.shape[0])

        position_ids = kwargs.get("position_ids")

        if position_ids is not None:
            positions = position_ids

            if positions.ndim == 1:
                positions = positions.unsqueeze(0)

            if positions.shape[0] == 1 and batch_size > 1:
                positions = positions.expand(
                    batch_size,
                    -1,
                )

            if positions.shape[0] != batch_size:
                raise RuntimeError(
                    "Unexpected position_ids "
                    f"shape "
                    f"{tuple(positions.shape)} "
                    f"for batch size "
                    f"{batch_size}."
                )

            return positions

        cache_position = kwargs.get("cache_position")

        if cache_position is not None:
            positions = cache_position

            if positions.ndim == 1:
                positions = positions.unsqueeze(0)

            if positions.shape[0] == 1 and batch_size > 1:
                positions = positions.expand(
                    batch_size,
                    -1,
                )

            if positions.shape[0] != batch_size:
                raise RuntimeError(
                    "Unexpected cache_position "
                    f"shape "
                    f"{tuple(positions.shape)} "
                    f"for batch size "
                    f"{batch_size}."
                )

            padding = self.left_padding.to(device=(positions.device)).view(
                batch_size,
                1,
            )

            return positions - padding

        raise RuntimeError(
            "Batched cached generation exposed neither "
            "position_ids nor cache_position. "
            "Cannot safely determine per-example "
            "steering positions."
        )

    def _hook(
        self,
        module,
        args,
        kwargs,
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

        batch_size = int(hidden_states.shape[0])

        if batch_size != len(self.from_positions):
            raise RuntimeError(
                "Steering batch-size mismatch: "
                f"hidden batch={batch_size}, "
                f"thresholds="
                f"{len(self.from_positions)}."
            )

        positions = self._resolve_positions(
            hidden_states,
            kwargs,
        )

        thresholds = self.from_positions.to(device=(positions.device)).view(
            batch_size,
            1,
        )

        mask = (positions >= thresholds).unsqueeze(-1)

        vector = self.coefficient * self.vector.to(
            device=(hidden_states.device),
            dtype=(hidden_states.dtype),
        )

        steered_hidden = hidden_states + mask.to(hidden_states.dtype) * vector

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

        handle = layer.register_forward_hook(
            self._hook,
            with_kwargs=True,
        )

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


def file_sha256(
    path: Path,
):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


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


def normalize_eos_ids(
    model,
    tokenizer,
):
    eos = getattr(
        model.generation_config,
        "eos_token_id",
        None,
    )

    if eos is None:
        eos = tokenizer.eos_token_id

    if eos is None:
        return set()

    if isinstance(
        eos,
        int,
    ):
        return {int(eos)}

    return {int(token_id) for token_id in eos}


# =============================================================================
# Numeric answer handling
# =============================================================================


def numeric_fraction(
    value: str,
):
    value = (
        value.strip()
        .replace(
            ",",
            "",
        )
        .replace(
            "$",
            "",
        )
    )

    try:
        if "/" in value:
            (
                numerator,
                denominator,
            ) = value.split(
                "/",
                maxsplit=1,
            )

            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))

        return Fraction(Decimal(value))

    except (
        InvalidOperation,
        ValueError,
        ZeroDivisionError,
    ):
        return None


def fraction_string(
    value,
):
    if value is None:
        return None

    if value.denominator == 1:
        return str(value.numerator)

    return f"{value.numerator}/{value.denominator}"


def extract_strict_prediction(
    response: str,
):
    matches = list(STRICT_FINAL_PATTERN.finditer(response))

    if not matches:
        return None

    return numeric_fraction(matches[-1].group(1))


def extract_flexible_prediction(
    response: str,
    *,
    hit_max_tokens: bool,
):
    strict = extract_strict_prediction(response)

    if strict is not None:
        return (
            strict,
            "marker_line",
        )

    # If generation was cut off before a valid final marker,
    # do not infer an answer from unfinished reasoning.
    if hit_max_tokens:
        return (
            None,
            "truncated_unparsed",
        )

    candidates = list(NUMBER_PATTERN.finditer(response))

    for match in reversed(candidates):
        parsed = numeric_fraction(match.group(0))

        if parsed is not None:
            return (
                parsed,
                "fallback_last_number",
            )

    return (
        None,
        "unparsed",
    )


# =============================================================================
# Frozen GSM8K data
# =============================================================================


def load_frozen_dataset(
    path: Path,
):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if len(data) != EXPECTED_TEST_EXAMPLES:
        raise ValueError(
            f"{path} contains {len(data)} examples; expected {EXPECTED_TEST_EXAMPLES}."
        )

    expected_indices = list(range(EXPECTED_TEST_EXAMPLES))

    observed_indices = [int(row["subset_index"]) for row in data]

    if observed_indices != expected_indices:
        raise ValueError("Frozen GSM8K subset indices are not 0..63 in order.")

    return data


def load_frozen_demonstrations(
    path: Path,
):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        demos = json.load(f)

    if len(demos) != EXPECTED_NUM_SHOTS:
        raise ValueError(
            f"{path} contains "
            f"{len(demos)} demonstrations; "
            f"expected "
            f"{EXPECTED_NUM_SHOTS}."
        )

    orders = [int(demo["demo_order"]) for demo in demos]

    if orders != list(range(EXPECTED_NUM_SHOTS)):
        raise ValueError("Few-shot demo ordering is not 0..7.")

    for demo in demos:
        solution = str(demo["solution_clean"])

        if "<<" in solution or ">>" in solution:
            raise ValueError(
                "Calculator annotation found in frozen cleaned demonstration."
            )

        if extract_strict_prediction(solution) is None:
            raise ValueError("Frozen demonstration lacks valid #### answer line.")

    return demos


# =============================================================================
# Frozen frontier operating points
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
# Prompt construction
# =============================================================================


def gsm8k_prompt(
    *,
    question: str,
    demonstrations,
):
    parts = [
        (
            "Solve the final grade-school math word problem. "
            "Use the worked examples below to understand the task "
            "and answer format."
        )
    ]

    for index, demo in enumerate(
        demonstrations,
        start=1,
    ):
        parts.append(
            (
                f"Worked Example {index}\n\n"
                f"Problem:\n"
                f"{demo['question'].strip()}\n\n"
                f"Solution:\n"
                f"{demo['solution_clean'].strip()}"
            )
        )

    parts.append(
        (
            "Now solve the following problem yourself. "
            "Reason step by step and perform the arithmetic carefully.\n\n"
            "First write your reasoning and complete all calculations. "
            "Do not write the final-answer line until your reasoning "
            "is finished.\n\n"
            "After the reasoning, write exactly one final-answer line "
            "using the same four-hash format shown in the worked examples. "
            "The final-answer line must be the last non-empty line of your "
            "response. It must contain only the four hash characters, one "
            "space, and the numerical answer. Do not include units, commas, "
            "currency symbols, or any text after it.\n\n"
            f"Problem:\n"
            f"{question.strip()}\n\n"
            "Solution:"
        )
    )

    return "\n\n".join(parts)


def build_generation_prefix(
    *,
    adapter,
    tokenizer,
    context: str,
):
    probe = adapter.build_continuation_prompt(
        tokenizer,
        context,
        "0",
    )

    continuation_start = int(probe.continuation_start_position)

    prefix = (
        probe.input_ids[
            0,
            :continuation_start,
        ]
        .clone()
        .long()
    )

    return (
        prefix,
        int(probe.steering_start_position),
    )


def build_generation_batch(
    *,
    adapter,
    tokenizer,
    prompts,
    device,
):
    prefixes = []
    from_positions = []

    for prompt in prompts:
        (
            prefix,
            from_position,
        ) = build_generation_prefix(
            adapter=adapter,
            tokenizer=tokenizer,
            context=prompt,
        )

        prefixes.append(prefix)

        from_positions.append(from_position)

    lengths = torch.tensor(
        [int(prefix.numel()) for prefix in prefixes],
        dtype=torch.long,
    )

    max_length = int(lengths.max().item())

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

    batch_size = len(prefixes)

    input_ids = torch.full(
        (
            batch_size,
            max_length,
        ),
        int(pad_token_id),
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (
            batch_size,
            max_length,
        ),
        dtype=torch.long,
    )

    left_padding = max_length - lengths

    for (
        row_index,
        prefix,
    ) in enumerate(prefixes):
        length = int(prefix.numel())

        input_ids[
            row_index,
            max_length - length :,
        ] = prefix

        attention_mask[
            row_index,
            max_length - length :,
        ] = 1

    return {
        "input_ids": (input_ids.to(device)),
        "attention_mask": (attention_mask.to(device)),
        "from_positions": (
            torch.tensor(
                from_positions,
                dtype=torch.long,
            )
        ),
        "left_padding": (left_padding),
        "padded_prompt_length": int(max_length),
        "prompt_lengths": (lengths),
        "pad_token_id": int(pad_token_id),
    }


# =============================================================================
# Batched generation
# =============================================================================


def decode_generated_row(
    *,
    suffix,
    model,
    tokenizer,
    max_new_tokens,
):
    eos_ids = normalize_eos_ids(
        model,
        tokenizer,
    )

    tokens = [int(value) for value in suffix.tolist()]

    first_eos = None

    for (
        index,
        token_id,
    ) in enumerate(tokens):
        if token_id in eos_ids:
            first_eos = index
            break

    if first_eos is not None:
        effective_tokens = tokens[: first_eos + 1]

        ended_with_eos = True
        hit_max_tokens = False

    else:
        effective_tokens = tokens

        ended_with_eos = False

        hit_max_tokens = len(tokens) >= int(max_new_tokens)

    response = tokenizer.decode(
        effective_tokens,
        skip_special_tokens=True,
    ).strip()

    return {
        "response": (response),
        "new_token_count": int(len(effective_tokens)),
        "ended_with_eos": bool(ended_with_eos),
        "hit_max_tokens": bool(hit_max_tokens),
    }


@torch.inference_mode()
def generate_batch(
    *,
    model,
    tokenizer,
    adapter,
    prompts,
    device,
    max_new_tokens,
    layer=None,
    vector=None,
    alpha=0.0,
):
    batch = build_generation_batch(
        adapter=adapter,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )

    generation_kwargs = {
        "input_ids": (batch["input_ids"]),
        "attention_mask": (batch["attention_mask"]),
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "use_cache": True,
        "pad_token_id": (batch["pad_token_id"]),
    }

    if vector is None or abs(float(alpha)) <= 1e-12:
        generated = model.generate(**generation_kwargs)

    else:
        with BatchedSteeringIntervention(
            model=model,
            layer_idx=int(layer),
            vector=vector,
            coefficient=float(alpha),
            from_positions=(batch["from_positions"]),
            left_padding=(batch["left_padding"]),
        ):
            generated = model.generate(**generation_kwargs)

    prompt_width = int(batch["padded_prompt_length"])

    suffixes = generated[
        :,
        prompt_width:,
    ]

    outputs = []

    for row_index in range(suffixes.shape[0]):
        result = decode_generated_row(
            suffix=(suffixes[row_index].detach().cpu()),
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=(max_new_tokens),
        )

        result["prompt_token_count"] = int(batch["prompt_lengths"][row_index].item())

        outputs.append(result)

    return outputs


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_condition(
    *,
    model,
    tokenizer,
    adapter,
    dataset,
    demonstrations,
    device,
    max_new_tokens,
    batch_size,
    description,
    layer=None,
    vector=None,
    alpha=0.0,
):
    rows = []

    total_batches = math.ceil(len(dataset) / batch_size)

    progress = tqdm(
        total=len(dataset),
        desc=description,
    )

    for batch_index in range(total_batches):
        start = batch_index * batch_size

        end = min(
            start + batch_size,
            len(dataset),
        )

        items = dataset[start:end]

        prompts = [
            gsm8k_prompt(
                question=(item["question"]),
                demonstrations=(demonstrations),
            )
            for item in items
        ]

        generations = generate_batch(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            prompts=prompts,
            device=device,
            max_new_tokens=(max_new_tokens),
            layer=layer,
            vector=vector,
            alpha=alpha,
        )

        if len(generations) != len(items):
            raise RuntimeError("Generation batch returned wrong number of rows.")

        for (
            item,
            generation,
        ) in zip(
            items,
            generations,
        ):
            response = generation["response"]

            strict_prediction = extract_strict_prediction(response)

            (
                flexible_prediction,
                flexible_parse_method,
            ) = extract_flexible_prediction(
                response,
                hit_max_tokens=(generation["hit_max_tokens"]),
            )

            gold = numeric_fraction(item["gold_answer"])

            strict_correct = (
                strict_prediction is not None
                and gold is not None
                and strict_prediction == gold
            )

            flexible_correct = (
                flexible_prediction is not None
                and gold is not None
                and flexible_prediction == gold
            )

            rows.append(
                {
                    "protocol_version": (PROTOCOL_VERSION),
                    "subset_index": int(item["subset_index"]),
                    "source_index": int(item["source_index"]),
                    "question": str(item["question"]),
                    "gold_answer": (fraction_string(gold)),
                    "predicted_answer_strict": (fraction_string(strict_prediction)),
                    "predicted_answer_flexible": (fraction_string(flexible_prediction)),
                    "parse_method_flexible": (flexible_parse_method),
                    "used_required_marker": bool(strict_prediction is not None),
                    "is_correct_strict": bool(strict_correct),
                    "is_correct_flexible": bool(flexible_correct),
                    "prompt_token_count": int(generation["prompt_token_count"]),
                    "new_token_count": int(generation["new_token_count"]),
                    "ended_with_eos": bool(generation["ended_with_eos"]),
                    "hit_max_tokens": bool(generation["hit_max_tokens"]),
                    "response": (response),
                }
            )

        progress.update(len(items))

    progress.close()

    return pd.DataFrame(rows)


def paired_accuracy_statistics(
    *,
    baseline,
    steered,
    score_column,
    bootstrap_samples,
    seed,
):
    merged = baseline[
        [
            "subset_index",
            score_column,
        ]
    ].merge(
        steered[
            [
                "subset_index",
                score_column,
            ]
        ],
        on="subset_index",
        suffixes=(
            "_baseline",
            "_steered",
        ),
        validate="one_to_one",
    )

    baseline_values = merged[f"{score_column}_baseline"].astype(float).to_numpy()

    steered_values = merged[f"{score_column}_steered"].astype(float).to_numpy()

    difference = steered_values - baseline_values

    delta = float(difference.mean())

    if np.allclose(
        difference,
        0.0,
    ):
        ci_low = 0.0
        ci_high = 0.0

    else:
        rng = np.random.default_rng(seed)

        indices = rng.integers(
            0,
            len(difference),
            size=(
                bootstrap_samples,
                len(difference),
            ),
        )

        bootstrap = difference[indices].mean(axis=1)

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

    return {
        "num_examples": int(len(merged)),
        "baseline_accuracy": float(baseline_values.mean()),
        "accuracy": float(steered_values.mean()),
        "delta_accuracy": float(delta),
        "delta_accuracy_ci_low": float(ci_low),
        "delta_accuracy_ci_high": float(ci_high),
    }


def validate_resume_frame(
    frame: pd.DataFrame,
):
    if "protocol_version" not in frame.columns:
        raise ValueError(
            "Existing GSM8K output predates "
            "the frozen 64-example 8-shot protocol. "
            "Use a clean output directory."
        )

    versions = set(frame["protocol_version"].astype(str))

    if versions != {PROTOCOL_VERSION}:
        raise ValueError(
            "Existing GSM8K output uses "
            f"protocol versions {versions}; "
            f"expected {PROTOCOL_VERSION}."
        )


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    config = load_experiment_config(args.config)

    output_root = (
        args.output_root
        if args.output_root is not None
        else (config.results_root / "secondary_capability" / "gsm8k")
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    item_root = output_root / "items"

    item_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_frozen_dataset(args.data_path)

    demonstrations = load_frozen_demonstrations(args.demos_path)

    if args.max_examples is not None:
        if args.max_examples < 1:
            raise ValueError("--max-examples must be >= 1.")

        dataset = dataset[: args.max_examples]

    operating_points = load_operating_points(
        config,
        epsilon=(args.epsilon),
        behavior=(args.behavior),
        direction=(args.direction),
    )

    data_hash = file_sha256(args.data_path)

    demos_hash = file_sha256(args.demos_path)

    print("=" * 80)
    print("GSM8K SECONDARY CAPABILITY")
    print("=" * 80)
    print(f"Protocol:           {PROTOCOL_VERSION}")
    print(f"Model:              {config.model_label}")
    print(f"Model slug:         {config.model_slug}")
    print(f"Epsilon:            {args.epsilon}")
    print(f"Examples:           {len(dataset)}")
    print(f"Few-shot examples:  {len(demonstrations)}")
    print(f"Operating points:   {len(operating_points)}")
    print(
        f"Nonzero points:     "
        f"{int((operating_points['alpha_star'].abs() > 1e-12).sum())}"
    )
    print(f"Batch size:         {args.batch_size}")
    print(f"Generation calls/condition: {math.ceil(len(dataset) / args.batch_size)}")
    print(f"Max new tokens:     {args.max_new_tokens}")
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

    tokenizer.padding_side = "left"

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=(config.model_name),
        prompt_family=(config.prompt_family),
    )

    print(f"Prompt adapter:     {adapter.family}")

    # ----------------------------------------------------------------
    # Baseline
    # ----------------------------------------------------------------

    baseline_path = output_root / "baseline_items.csv"

    if baseline_path.is_file() and not args.no_resume:
        baseline = pd.read_csv(baseline_path)

        validate_resume_frame(baseline)

        expected_indices = {int(item["subset_index"]) for item in dataset}

        observed_indices = set(baseline["subset_index"].astype(int))

        if observed_indices != expected_indices:
            raise ValueError(
                "Existing baseline does not match requested smoke/full subset."
            )

        print()
        print(f"Reusing baseline: {baseline_path}")

    else:
        print()
        print("=" * 80)
        print("BASELINE")
        print("=" * 80)

        baseline = evaluate_condition(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            dataset=dataset,
            demonstrations=(demonstrations),
            device=(args.device),
            max_new_tokens=(args.max_new_tokens),
            batch_size=(args.batch_size),
            description=("GSM8K baseline"),
        )

        write_csv(
            baseline,
            baseline_path,
        )

    baseline_accuracy_flexible = float(
        baseline["is_correct_flexible"].astype(float).mean()
    )

    baseline_accuracy_strict = float(baseline["is_correct_strict"].astype(float).mean())

    baseline_marker_rate = float(baseline["used_required_marker"].astype(float).mean())

    baseline_parse_rate_flexible = float(
        (
            ~baseline["parse_method_flexible"].isin(
                [
                    "unparsed",
                    "truncated_unparsed",
                ]
            )
        )
        .astype(float)
        .mean()
    )

    baseline_truncation_rate = float(baseline["hit_max_tokens"].astype(float).mean())

    baseline_mean_tokens = float(baseline["new_token_count"].astype(float).mean())

    print()
    print(f"Baseline flexible:   {baseline_accuracy_flexible:.4f}")
    print(f"Baseline strict:     {baseline_accuracy_strict:.4f}")
    print(f"Marker rate:         {baseline_marker_rate:.4f}")
    print(f"Flexible parse rate: {baseline_parse_rate_flexible:.4f}")
    print(f"Truncation rate:     {baseline_truncation_rate:.4f}")
    print(f"Mean generated toks: {baseline_mean_tokens:.1f}")

    # ----------------------------------------------------------------
    # Resume summary
    # ----------------------------------------------------------------

    summary_path = output_root / "gsm8k_summary.csv"

    if summary_path.is_file() and not args.no_resume:
        summary = pd.read_csv(summary_path)

        validate_resume_frame(summary)

        completed = {
            (
                str(row["behavior"]),
                str(row["direction"]),
            )
            for row in summary.to_dict(orient="records")
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
        print("=" * 80)
        print(f"{behavior} / {direction}")
        print("=" * 80)
        print(f"Layer: {layer}")
        print(f"alpha: {alpha:+.6f}")
        print(f"eta:   {eta:.6f}")

        item_path = item_root / (
            condition_name(
                behavior,
                direction,
            )
            + ".csv"
        )

        if abs(alpha) <= 1e-12:
            steered = baseline.copy()

            write_csv(
                steered,
                item_path,
            )

        else:
            steered = evaluate_condition(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                dataset=dataset,
                demonstrations=(demonstrations),
                device=(args.device),
                max_new_tokens=(args.max_new_tokens),
                batch_size=(args.batch_size),
                description=(f"GSM8K {behavior} {direction}"),
                layer=(layer),
                vector=(vectors[behavior][layer]),
                alpha=(alpha),
            )

            write_csv(
                steered,
                item_path,
            )

        flexible_seed = stable_seed(
            PROTOCOL_VERSION,
            "flexible",
            config.model_slug,
            behavior,
            direction,
            args.epsilon,
        )

        strict_seed = stable_seed(
            PROTOCOL_VERSION,
            "strict",
            config.model_slug,
            behavior,
            direction,
            args.epsilon,
        )

        flexible_stats = paired_accuracy_statistics(
            baseline=baseline,
            steered=steered,
            score_column=("is_correct_flexible"),
            bootstrap_samples=(args.bootstrap_samples),
            seed=(flexible_seed),
        )

        strict_stats = paired_accuracy_statistics(
            baseline=baseline,
            steered=steered,
            score_column=("is_correct_strict"),
            bootstrap_samples=(args.bootstrap_samples),
            seed=(strict_seed),
        )

        marker_rate = float(steered["used_required_marker"].astype(float).mean())

        parse_rate_flexible = float(
            (
                ~steered["parse_method_flexible"].isin(
                    [
                        "unparsed",
                        "truncated_unparsed",
                    ]
                )
            )
            .astype(float)
            .mean()
        )

        truncation_rate = float(steered["hit_max_tokens"].astype(float).mean())

        mean_generated_tokens = float(steered["new_token_count"].astype(float).mean())

        summary_row = {
            "protocol_version": (PROTOCOL_VERSION),
            "model_name": (config.model_name),
            "model_label": (config.model_label),
            "model_slug": (config.model_slug),
            "epsilon": float(args.epsilon),
            "behavior": (behavior),
            "direction": (direction),
            "layer": (layer),
            "alpha_star": (alpha),
            "eta_star": (eta),
            "num_examples": (flexible_stats["num_examples"]),
            "baseline_accuracy_flexible": (flexible_stats["baseline_accuracy"]),
            "accuracy_flexible": (flexible_stats["accuracy"]),
            "delta_accuracy_flexible": (flexible_stats["delta_accuracy"]),
            "delta_accuracy_flexible_ci_low": (flexible_stats["delta_accuracy_ci_low"]),
            "delta_accuracy_flexible_ci_high": (
                flexible_stats["delta_accuracy_ci_high"]
            ),
            "baseline_accuracy_strict": (strict_stats["baseline_accuracy"]),
            "accuracy_strict": (strict_stats["accuracy"]),
            "delta_accuracy_strict": (strict_stats["delta_accuracy"]),
            "delta_accuracy_strict_ci_low": (strict_stats["delta_accuracy_ci_low"]),
            "delta_accuracy_strict_ci_high": (strict_stats["delta_accuracy_ci_high"]),
            "marker_rate": (marker_rate),
            "parse_rate_flexible": (parse_rate_flexible),
            "truncation_rate": (truncation_rate),
            "mean_generated_tokens": (mean_generated_tokens),
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed_flexible": int(flexible_seed),
            "bootstrap_seed_strict": int(strict_seed),
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

        print(f"flex accuracy: {flexible_stats['accuracy']:.4f}")
        print(
            f"flex delta:    "
            f"{flexible_stats['delta_accuracy']:+.4f} "
            f"["
            f"{flexible_stats['delta_accuracy_ci_low']:+.4f}, "
            f"{flexible_stats['delta_accuracy_ci_high']:+.4f}"
            f"]"
        )
        print(f"strict acc:    {strict_stats['accuracy']:.4f}")
        print(
            f"strict delta:  "
            f"{strict_stats['delta_accuracy']:+.4f} "
            f"["
            f"{strict_stats['delta_accuracy_ci_low']:+.4f}, "
            f"{strict_stats['delta_accuracy_ci_high']:+.4f}"
            f"]"
        )
        print(f"marker rate:   {marker_rate:.4f}")
        print(f"parse rate:    {parse_rate_flexible:.4f}")
        print(f"truncation:    {truncation_rate:.4f}")
        print(f"mean tokens:   {mean_generated_tokens:.1f}")

    # ----------------------------------------------------------------
    # Manifest
    # ----------------------------------------------------------------

    summary = pd.DataFrame(summary_rows)

    manifest = {
        "analysis": ("gsm8k_secondary_capability"),
        "protocol_version": (PROTOCOL_VERSION),
        "model_name": (config.model_name),
        "model_label": (config.model_label),
        "model_slug": (config.model_slug),
        "frontier_epsilon": float(args.epsilon),
        "dataset": ("openai/gsm8k"),
        "dataset_config": ("main"),
        "dataset_split": ("test"),
        "subset_file": str(args.data_path),
        "subset_sha256": (data_hash),
        "subset_size_frozen": int(EXPECTED_TEST_EXAMPLES),
        "subset_size_evaluated": int(len(dataset)),
        "fewshot": {
            "shots": int(EXPECTED_NUM_SHOTS),
            "demonstration_file": str(args.demos_path),
            "demonstration_sha256": (demos_hash),
            "demonstration_split": ("train"),
            "calculator_annotations_removed": True,
        },
        "generation": {
            "batch_size": int(args.batch_size),
            "padding_side": ("left"),
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": int(args.max_new_tokens),
            "use_cache": True,
            "per_example_steering_boundary": True,
        },
        "scoring": {
            "primary": ("flexible exact numerical equality"),
            "strict": (
                "Exact numerical equality when parsed "
                "from a standalone #### answer line."
            ),
            "flexible": (
                "Use standalone #### answer when present; "
                "otherwise use the final parseable number "
                "for complete, non-truncated generations."
            ),
            "truncation_rule": (
                "A generation that hits max_new_tokens "
                "without a valid #### answer is "
                "unparsed/incorrect."
            ),
        },
        "bootstrap": {
            "paired": True,
            "samples": int(args.bootstrap_samples),
            "confidence_interval": ("percentile 95%"),
        },
        "selection_note": (
            "GSM8K is secondary evaluation only. "
            "No GSM8K outcome is used for layer, coefficient, "
            "eta, or frontier selection."
        ),
        "num_operating_points_requested": int(len(operating_points)),
        "num_operating_points_completed": int(len(summary)),
    }

    with (output_root / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("GSM8K COMPLETE")
    print("=" * 80)
    print(f"Baseline flexible:   {baseline_accuracy_flexible:.4f}")
    print(f"Baseline strict:     {baseline_accuracy_strict:.4f}")
    print(f"Baseline truncation: {baseline_truncation_rate:.4f}")
    print(f"Completed points:    {len(summary)}/{len(operating_points)}")
    print(f"Summary:             {summary_path}")


if __name__ == "__main__":
    main()
