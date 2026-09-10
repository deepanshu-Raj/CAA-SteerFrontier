from typing import Optional

import torch
from tqdm import tqdm

from steerfrontier.prompts import (
    VALID_ANSWERS,
    PromptAdapter,
    get_prompt_adapter,
)
from steerfrontier.steering import SteeringIntervention


def _validate_target(answer: str) -> str:
    answer = answer.strip()

    if answer not in VALID_ANSWERS:
        raise ValueError(f"Invalid answer label: {answer!r}.")

    return answer


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
    """
    Score two specified answer choices against one another.

    Probabilities are renormalized over only the two explicitly
    specified contrastive choices.
    """

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
            "Prompt adapter cannot score required answer labels "
            f"{missing}. Available labels: "
            f"{sorted(scoring.answer_token_ids)}"
        )

    first_token = scoring.answer_token_ids[first_answer]

    second_token = scoring.answer_token_ids[second_answer]

    if first_token == second_token:
        raise ValueError(
            f"Answer labels {first_answer} and {second_answer} "
            "map to the same scoring token."
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
        "first_answer": first_answer,
        "second_answer": second_answer,
        "first_raw_prob": first_raw_prob,
        "second_raw_prob": second_raw_prob,
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
    """
    Evaluate probability of the behavior-matching answer.

    Each example is evaluated contrastively using only:

        answer_matching_behavior
            vs.
        answer_not_matching_behavior

    Returns both the aggregate behavior score and item-level values,
    allowing paired bootstrap analysis later without rerunning the model.
    """

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
                "item_index": index,
                "matching_answer": matching,
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


@torch.no_grad()
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
):
    """
    Evaluate the existing contrastive MMLU capability subset.

    Each item contains one correct and one incorrect answer choice.
    """

    if len(dataset) == 0:
        raise ValueError("MMLU evaluation dataset is empty.")

    adapter = prompt_adapter or get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        prompt_family=prompt_family,
    )

    rows = []

    for item in tqdm(
        dataset,
        desc=(f"MMLU L{layer_idx}, alpha={coefficient}"),
    ):
        correct = _validate_target(item["correct"])

        incorrect = _validate_target(item["incorrect"])

        scores = _score_contrastive_item(
            model=model,
            tokenizer=tokenizer,
            question=item["prompt"],
            first_answer=correct,
            second_answer=incorrect,
            layer_idx=layer_idx,
            vector=vector,
            coefficient=coefficient,
            adapter=adapter,
            device=device,
        )

        correct_prob = scores["first_prob"]

        incorrect_prob = scores["second_prob"]

        prediction = correct if correct_prob >= incorrect_prob else incorrect

        rows.append(
            {
                "category": (item["category"]),
                "correct": correct,
                "incorrect": incorrect,
                "correct_raw_prob": (scores["first_raw_prob"]),
                "incorrect_raw_prob": (scores["second_raw_prob"]),
                "correct_prob": (correct_prob),
                "incorrect_prob": (incorrect_prob),
                "prediction": (prediction),
                "is_correct": (prediction == correct),
            }
        )

    capability_score = sum(row["correct_prob"] for row in rows) / len(rows)

    accuracy = sum(row["is_correct"] for row in rows) / len(rows)

    return {
        "capability_score": (capability_score),
        "accuracy": accuracy,
        "num_examples": len(rows),
        "items": rows,
    }


@torch.no_grad()
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
):
    result = evaluate_mmlu_detailed(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=layer_idx,
        vector=vector,
        coefficient=coefficient,
        device=device,
        prompt_adapter=prompt_adapter,
        prompt_family=prompt_family,
    )

    return {
        "capability_score": (result["capability_score"]),
        "num_examples": (result["num_examples"]),
    }
