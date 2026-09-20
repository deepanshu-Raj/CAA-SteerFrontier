"""Forced-choice behavior and capability evaluation utilities.

Scoring compares only the designated answer options and supports optional steering
interventions at fixed decoder layers.
"""

from typing import Optional

import torch
from tqdm import tqdm

from steerfrontier.prompts import (
    VALID_ANSWERS,
    PromptAdapter,
    get_prompt_adapter,
)
from steerfrontier.steering import (
    SteeringIntervention,
)

# =============================================================================
# Shared helpers
# =============================================================================

def _validate_target(
    answer: str,
) -> str:
    answer = answer.strip()

    if answer not in VALID_ANSWERS:
        raise ValueError(f"Invalid answer label: {answer!r}.")

    return answer


def _validate_batch_size(
    batch_size: int,
) -> int:
    batch_size = int(batch_size)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    return batch_size


def _get_pad_token_id(
    tokenizer,
) -> int:
    """
    Pick a safe padding token ID.

    Padding is always accompanied by an attention mask,
    so its semantic identity is irrelevant.
    """

    for attr in (
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
    ):
        token_id = getattr(
            tokenizer,
            attr,
            None,
        )

        if token_id is not None:
            return int(token_id)

    return 0


def _right_pad_input_ids(
    sequences,
    *,
    tokenizer,
    device: str,
):
    """
    Right-pad a collection of [1, seq_len] tensors.

    Returns:
        input_ids
        attention_mask
        lengths
        has_padding
    """

    if len(sequences) == 0:
        raise ValueError("Cannot pad an empty batch.")

    flattened = [sequence.reshape(-1) for sequence in sequences]

    lengths = [int(sequence.numel()) for sequence in flattened]

    max_length = max(lengths)

    pad_token_id = _get_pad_token_id(tokenizer)

    dtype = flattened[0].dtype

    input_ids = torch.full(
        (
            len(flattened),
            max_length,
        ),
        fill_value=pad_token_id,
        dtype=dtype,
        device=device,
    )

    attention_mask = torch.zeros(
        (
            len(flattened),
            max_length,
        ),
        dtype=torch.long,
        device=device,
    )

    for row, sequence in enumerate(flattened):
        length = lengths[row]

        input_ids[
            row,
            :length,
        ] = sequence.to(device)

        attention_mask[
            row,
            :length,
        ] = 1

    has_padding = len(set(lengths)) > 1

    return (
        input_ids,
        attention_mask,
        lengths,
        has_padding,
    )


def _forward_model(
    model,
    *,
    input_ids,
    attention_mask,
    has_padding: bool,
):
    """
    Preserve the old single-example call path when no padding
    is needed. This also keeps lightweight test models working.

    Real padded batches receive an attention mask.
    """

    kwargs = {
        "input_ids": input_ids,
        "use_cache": False,
    }

    if has_padding:
        kwargs["attention_mask"] = attention_mask

    return model(**kwargs)


def _format_multiple_choice_question(
    question: str,
    choices,
) -> str:
    lines = [
        question.strip(),
        "",
    ]

    for choice in choices:
        label = _validate_target(choice["label"])

        text = str(choice["text"]).strip()

        lines.append(f"{label} {text}")

    lines.extend(
        [
            "",
            "Answer with only the option label.",
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Behavior evaluation
# =============================================================================

@torch.no_grad()
def _score_contrastive_item(
    model,
    tokenizer,
    question: str,
    first_answer: str,
    second_answer: str,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    adapter: PromptAdapter,
    device: str,
):
    first_answer = _validate_target(first_answer)

    second_answer = _validate_target(second_answer)

    if first_answer == second_answer:
        raise ValueError("Contrastive answers must be different.")

    scoring = adapter.build_scoring_prompt(
        tokenizer,
        question,
    )

    missing = [
        answer
        for answer in (
            first_answer,
            second_answer,
        )
        if answer not in scoring.answer_token_ids
    ]

    if missing:
        raise ValueError(
            "Prompt adapter cannot score required "
            f"answer labels {missing}. Available labels: "
            f"{sorted(scoring.answer_token_ids)}"
        )

    first_token = scoring.answer_token_ids[first_answer]

    second_token = scoring.answer_token_ids[second_answer]

    if first_token == second_token:
        raise ValueError(
            f"Answer labels {first_answer} and "
            f"{second_answer} map to the same "
            "scoring token."
        )

    input_ids = scoring.input_ids.to(device)

    with SteeringIntervention(
        model=model,
        layer_idx=layer_idx,
        vector=vector,
        coefficient=coefficient,
        from_position=(scoring.steering_start_position),
    ):
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
        )

    logits = outputs.logits[
        0,
        -1,
        :,
    ]

    probs = torch.softmax(
        logits,
        dim=-1,
    )

    first_raw_prob = float(probs[first_token].item())

    second_raw_prob = float(probs[second_token].item())

    denom = first_raw_prob + second_raw_prob

    if denom <= 0.0:
        raise RuntimeError("Contrastive probability mass is zero; cannot renormalize.")

    return {
        "first_answer": (first_answer),
        "second_answer": (second_answer),
        "first_raw_prob": (first_raw_prob),
        "second_raw_prob": (second_raw_prob),
        "first_prob": (first_raw_prob / denom),
        "second_prob": (second_raw_prob / denom),
    }


@torch.no_grad()
def evaluate_behavior(
    model,
    tokenizer,
    dataset,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    device: str = "cuda",
    prompt_adapter: Optional[PromptAdapter] = None,
    prompt_family: str = "auto",
):
    """Evaluate one behavior dataset using designated matching and nonmatching answers."""
    if len(dataset) == 0:
        raise ValueError("Behavior evaluation dataset is empty.")

    adapter = prompt_adapter or get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        prompt_family=prompt_family,
    )

    items = []

    for index, item in enumerate(
        tqdm(
            dataset,
            desc=(f"Layer {layer_idx}, alpha={coefficient}"),
        )
    ):
        matching = _validate_target(item["answer_matching_behavior"])

        nonmatching = _validate_target(item["answer_not_matching_behavior"])

        scores = _score_contrastive_item(
            model=model,
            tokenizer=tokenizer,
            question=item["question"],
            first_answer=matching,
            second_answer=nonmatching,
            layer_idx=layer_idx,
            vector=vector,
            coefficient=coefficient,
            adapter=adapter,
            device=device,
        )

        items.append(
            {
                "item_index": (index),
                "matching_answer": (matching),
                "nonmatching_answer": (nonmatching),
                "matching_raw_prob": (scores["first_raw_prob"]),
                "nonmatching_raw_prob": (scores["second_raw_prob"]),
                "matching_prob": (scores["first_prob"]),
                "nonmatching_prob": (scores["second_prob"]),
            }
        )

    behavior_score = sum(item["matching_prob"] for item in items) / len(items)

    return {
        "behavior_score": (behavior_score),
        "num_examples": len(items),
        "items": items,
    }


# =============================================================================
# Batched discrete multiple-choice evaluation
#
# MMLU / ARC-Challenge
# =============================================================================
def _prepare_multiple_choice_record(
    *,
    item,
    item_index: int,
    tokenizer,
    adapter: PromptAdapter,
):
    prompt = _format_multiple_choice_question(
        question=item["question"],
        choices=item["choices"],
    )

    scoring = adapter.build_scoring_prompt(
        tokenizer,
        prompt,
    )

    labels = [_validate_target(choice["label"]) for choice in item["choices"]]

    missing = [label for label in labels if label not in scoring.answer_token_ids]

    if missing:
        raise ValueError(
            "Prompt adapter cannot score "
            f"required labels {missing}. "
            "Available labels: "
            f"{sorted(scoring.answer_token_ids)}"
        )

    token_ids = [int(scoring.answer_token_ids[label]) for label in labels]

    if len(set(token_ids)) != len(token_ids):
        raise ValueError(
            "Two or more multiple-choice labels map to the same scoring token."
        )

    return {
        "item_index": int(item_index),
        "item": item,
        "input_ids": (scoring.input_ids),
        "length": int(scoring.input_ids.numel()),
        "steering_start_position": int(scoring.steering_start_position),
        "labels": labels,
        "token_ids": token_ids,
    }


@torch.no_grad()
def _score_multiple_choice_batch(
    *,
    model,
    tokenizer,
    records,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    device: str,
):
    sequences = [record["input_ids"] for record in records]

    (
        input_ids,
        attention_mask,
        lengths,
        has_padding,
    ) = _right_pad_input_ids(
        sequences,
        tokenizer=tokenizer,
        device=device,
    )

    steering_starts = [record["steering_start_position"] for record in records]

    with SteeringIntervention(
        model=model,
        layer_idx=layer_idx,
        vector=vector,
        coefficient=coefficient,
        from_position=(steering_starts),
    ):
        outputs = _forward_model(
            model,
            input_ids=input_ids,
            attention_mask=(attention_mask),
            has_padding=(has_padding),
        )

    results = []

    for row, record in enumerate(records):
        final_position = lengths[row] - 1

        logits = outputs.logits[
            row,
            final_position,
            :,
        ]

        full_probs = torch.softmax(
            logits,
            dim=-1,
        )

        token_ids = record["token_ids"]

        candidate_logits = logits[
            torch.tensor(
                token_ids,
                dtype=torch.long,
                device=(logits.device),
            )
        ]

        candidate_probs = torch.softmax(
            candidate_logits,
            dim=0,
        )

        scores = {}

        for choice_index, label in enumerate(record["labels"]):
            token_id = token_ids[choice_index]

            scores[label] = {
                "token_id": int(token_id),
                "raw_prob": float(full_probs[token_id].item()),
                "choice_prob": float(candidate_probs[choice_index].item()),
                "logit": float(candidate_logits[choice_index].item()),
            }

        prediction_index = int(torch.argmax(candidate_logits).item())

        prediction = record["labels"][prediction_index]

        results.append(
            {
                "item_index": (record["item_index"]),
                "prediction": (prediction),
                "scores": scores,
            }
        )

    return results


@torch.no_grad()
def evaluate_multiple_choice_capability(
    model,
    tokenizer,
    dataset,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    benchmark_name: str,
    device: str = "cuda",
    prompt_adapter: Optional[PromptAdapter] = None,
    prompt_family: str = "auto",
    batch_size: int = 1,
):
    """
    Native multiple-choice capability evaluation.

    Primary metric:
        accuracy

    Secondary diagnostic:
        normalized probability assigned to the correct choice.

    Examples are sorted by token length before batching to reduce
    padding overhead. Returned results preserve original dataset order.
    """

    if len(dataset) == 0:
        raise ValueError(f"{benchmark_name} dataset is empty.")

    batch_size = _validate_batch_size(batch_size)

    adapter = prompt_adapter or get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        prompt_family=prompt_family,
    )

    records = [
        _prepare_multiple_choice_record(
            item=item,
            item_index=index,
            tokenizer=tokenizer,
            adapter=adapter,
        )
        for index, item in enumerate(dataset)
    ]

    # Length bucketing.
    records.sort(
        key=lambda record: (
            record["length"],
            record["item_index"],
        )
    )

    scored = []

    progress = tqdm(
        total=len(records),
        desc=(
            f"{benchmark_name} L{layer_idx}, alpha={coefficient}, batch={batch_size}"
        ),
    )

    for start in range(
        0,
        len(records),
        batch_size,
    ):
        batch = records[start : start + batch_size]

        scored.extend(
            _score_multiple_choice_batch(
                model=model,
                tokenizer=tokenizer,
                records=batch,
                layer_idx=layer_idx,
                vector=vector,
                coefficient=coefficient,
                device=device,
            )
        )

        progress.update(len(batch))

    progress.close()

    scored_by_index = {result["item_index"]: result for result in scored}

    items = []

    for index, item in enumerate(dataset):
        correct = _validate_target(item["correct"])

        result = scored_by_index[index]

        prediction = result["prediction"]

        correct_scores = result["scores"][correct]

        items.append(
            {
                "item_index": (index),
                "id": item["id"],
                "category": (item.get("category")),
                "correct": (correct),
                "prediction": (prediction),
                "is_correct": (prediction == correct),
                "correct_raw_prob": float(correct_scores["raw_prob"]),
                "correct_choice_prob": float(correct_scores["choice_prob"]),
                "choice_scores": (result["scores"]),
            }
        )

    accuracy = sum(int(item["is_correct"]) for item in items) / len(items)

    mean_correct_choice_prob = sum(item["correct_choice_prob"] for item in items) / len(
        items
    )

    return {
        "benchmark": (benchmark_name),
        "accuracy": float(accuracy),
        "mean_correct_choice_prob": float(mean_correct_choice_prob),
        "num_examples": len(items),
        "items": items,
    }


# =============================================================================
# Batched continuation ranking
#
# HellaSwag
# =============================================================================
def _prepare_continuation_record(
    *,
    item,
    item_index: int,
    ending_index: int,
    ending: str,
    tokenizer,
    adapter: PromptAdapter,
):
    prompt = adapter.build_continuation_prompt(
        tokenizer=tokenizer,
        context=item["context"],
        continuation=ending,
    )

    return {
        "item_index": int(item_index),
        "ending_index": int(ending_index),
        "input_ids": (prompt.input_ids),
        "length": int(prompt.input_ids.numel()),
        "continuation_start_position": int(prompt.continuation_start_position),
        "steering_start_position": int(prompt.steering_start_position),
    }


@torch.no_grad()
def _score_continuation_batch(
    *,
    model,
    tokenizer,
    records,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    device: str,
):
    sequences = [record["input_ids"] for record in records]

    (
        input_ids,
        attention_mask,
        lengths,
        has_padding,
    ) = _right_pad_input_ids(
        sequences,
        tokenizer=tokenizer,
        device=device,
    )

    steering_starts = [record["steering_start_position"] for record in records]

    with SteeringIntervention(
        model=model,
        layer_idx=layer_idx,
        vector=vector,
        coefficient=coefficient,
        from_position=(steering_starts),
    ):
        outputs = _forward_model(
            model,
            input_ids=input_ids,
            attention_mask=(attention_mask),
            has_padding=(has_padding),
        )

    results = []

    for row, record in enumerate(records):
        seq_len = lengths[row]

        start = int(record["continuation_start_position"])

        if start <= 0:
            raise ValueError("Continuation must begin after at least one prefix token.")

        if start >= seq_len:
            raise ValueError("Continuation contains no tokens.")

        prediction_logits = outputs.logits[
            row,
            start - 1 : seq_len - 1,
            :,
        ]

        targets = input_ids[
            row,
            start:seq_len,
        ]

        if prediction_logits.shape[0] != targets.shape[0]:
            raise RuntimeError(
                "Continuation logits and targets do not have matching lengths."
            )

        log_probs = torch.log_softmax(
            prediction_logits,
            dim=-1,
        )

        token_log_probs = log_probs.gather(
            dim=-1,
            index=(targets.unsqueeze(-1)),
        ).squeeze(-1)

        num_tokens = int(token_log_probs.numel())

        if num_tokens == 0:
            raise RuntimeError("Continuation produced no scored tokens.")

        results.append(
            {
                "item_index": int(record["item_index"]),
                "ending_index": int(record["ending_index"]),
                "num_tokens": (num_tokens),
                "sum_log_likelihood": float(token_log_probs.sum().item()),
                "mean_log_likelihood": float(token_log_probs.mean().item()),
            }
        )

    return results


@torch.no_grad()
def evaluate_continuation_capability(
    model,
    tokenizer,
    dataset,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    benchmark_name: str,
    device: str = "cuda",
    prompt_adapter: Optional[PromptAdapter] = None,
    prompt_family: str = "auto",
    batch_size: int = 1,
):
    """
    Continuation-ranking evaluation.

    For HellaSwag, all candidate endings across all questions are
    flattened, length-bucketed, and processed in batches.

    Primary prediction:
        argmax mean token log-likelihood

    Secondary prediction:
        argmax summed log-likelihood
    """

    if len(dataset) == 0:
        raise ValueError(f"{benchmark_name} dataset is empty.")

    batch_size = _validate_batch_size(batch_size)

    adapter = prompt_adapter or get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        prompt_family=prompt_family,
    )

    records = []

    for item_index, item in enumerate(dataset):
        for ending_index, ending in enumerate(item["endings"]):
            records.append(
                _prepare_continuation_record(
                    item=item,
                    item_index=(item_index),
                    ending_index=(ending_index),
                    ending=ending,
                    tokenizer=tokenizer,
                    adapter=adapter,
                )
            )

    # Critical for HellaSwag: group similar sequence lengths
    # so short candidates are not padded to long outliers.
    records.sort(
        key=lambda record: (
            record["length"],
            record["item_index"],
            record["ending_index"],
        )
    )

    scored = []

    progress = tqdm(
        total=len(records),
        desc=(
            f"{benchmark_name} L{layer_idx}, alpha={coefficient}, batch={batch_size}"
        ),
    )

    for start in range(
        0,
        len(records),
        batch_size,
    ):
        batch = records[start : start + batch_size]

        scored.extend(
            _score_continuation_batch(
                model=model,
                tokenizer=tokenizer,
                records=batch,
                layer_idx=layer_idx,
                vector=vector,
                coefficient=coefficient,
                device=device,
            )
        )

        progress.update(len(batch))

    progress.close()

    scores_by_item = {item_index: {} for item_index in range(len(dataset))}

    for score in scored:
        scores_by_item[score["item_index"]][score["ending_index"]] = score

    items = []

    for item_index, item in enumerate(dataset):
        ending_scores = []

        for ending_index in range(len(item["endings"])):
            if ending_index not in scores_by_item[item_index]:
                raise RuntimeError(
                    "Missing continuation score for "
                    f"item {item_index}, "
                    f"ending {ending_index}."
                )

            ending_scores.append(scores_by_item[item_index][ending_index])

        normalized_values = torch.tensor(
            [score["mean_log_likelihood"] for score in ending_scores],
            dtype=torch.float64,
        )

        raw_values = torch.tensor(
            [score["sum_log_likelihood"] for score in ending_scores],
            dtype=torch.float64,
        )

        prediction = int(torch.argmax(normalized_values).item())

        raw_prediction = int(torch.argmax(raw_values).item())

        correct_index = int(item["correct_index"])

        correct_score = ending_scores[correct_index]

        items.append(
            {
                "item_index": (item_index),
                "id": item["id"],
                "category": (item.get("category")),
                "correct_index": (correct_index),
                "prediction": (prediction),
                "raw_prediction": (raw_prediction),
                "is_correct": (prediction == correct_index),
                "raw_is_correct": (raw_prediction == correct_index),
                "correct_num_tokens": int(correct_score["num_tokens"]),
                "correct_sum_log_likelihood": float(
                    correct_score["sum_log_likelihood"]
                ),
                "correct_mean_log_likelihood": float(
                    correct_score["mean_log_likelihood"]
                ),
                "ending_scores": (ending_scores),
            }
        )

    accuracy = sum(int(item["is_correct"]) for item in items) / len(items)

    raw_accuracy = sum(int(item["raw_is_correct"]) for item in items) / len(items)

    return {
        "benchmark": (benchmark_name),
        "accuracy": float(accuracy),
        "raw_accuracy": float(raw_accuracy),
        "num_examples": len(items),
        "items": items,
    }


# =============================================================================
# Benchmark-specific wrappers
# =============================================================================

def evaluate_mmlu(
    model,
    tokenizer,
    dataset,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    device: str = "cuda",
    prompt_adapter: Optional[PromptAdapter] = None,
    prompt_family: str = "auto",
    batch_size: int = 1,
):
    """Evaluate MMLU accuracy under an optional steering intervention."""
    return evaluate_multiple_choice_capability(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=layer_idx,
        vector=vector,
        coefficient=coefficient,
        benchmark_name="mmlu",
        device=device,
        prompt_adapter=(prompt_adapter),
        prompt_family=(prompt_family),
        batch_size=(batch_size),
    )


def evaluate_arc_challenge(
    model,
    tokenizer,
    dataset,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    device: str = "cuda",
    prompt_adapter: Optional[PromptAdapter] = None,
    prompt_family: str = "auto",
    batch_size: int = 1,
):
    """Evaluate ARC-Challenge accuracy under an optional steering intervention."""
    return evaluate_multiple_choice_capability(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=layer_idx,
        vector=vector,
        coefficient=coefficient,
        benchmark_name=("arc_challenge"),
        device=device,
        prompt_adapter=(prompt_adapter),
        prompt_family=(prompt_family),
        batch_size=(batch_size),
    )


def evaluate_hellaswag(
    model,
    tokenizer,
    dataset,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    device: str = "cuda",
    prompt_adapter: Optional[PromptAdapter] = None,
    prompt_family: str = "auto",
    batch_size: int = 1,
):
    """Evaluate HellaSwag accuracy under an optional steering intervention."""
    return evaluate_continuation_capability(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=layer_idx,
        vector=vector,
        coefficient=coefficient,
        benchmark_name="hellaswag",
        device=device,
        prompt_adapter=(prompt_adapter),
        prompt_family=(prompt_family),
        batch_size=(batch_size),
    )


# =============================================================================
# Temporary compatibility hook
#
# Remove together with the old Phase-1 finalize path in workflow.py.
# =============================================================================
def evaluate_mmlu_detailed(
    model,
    tokenizer,
    dataset,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    device: str = "cuda",
    prompt_adapter: Optional[PromptAdapter] = None,
    prompt_family: str = "auto",
    batch_size: int = 1,
):
    """Evaluate MMLU and return both aggregate accuracy and item-level outputs."""
    return evaluate_mmlu(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=layer_idx,
        vector=vector,
        coefficient=coefficient,
        device=device,
        prompt_adapter=(prompt_adapter),
        prompt_family=(prompt_family),
        batch_size=(batch_size),
    )
