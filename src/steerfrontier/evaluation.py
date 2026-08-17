from typing import Optional

import torch
from tqdm import tqdm

from steerfrontier.prompts import PromptAdapter, VALID_ANSWERS, get_prompt_adapter
from steerfrontier.steering import SteeringIntervention


def _validate_target(answer: str) -> str:
    if answer not in VALID_ANSWERS:
        raise ValueError(f"Expected binary target '(A)' or '(B)', got {answer!r}.")
    return answer


@torch.no_grad()
def _score_binary_item(
    model,
    tokenizer,
    question: str,
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    adapter: PromptAdapter,
    device: str,
):
    scoring = adapter.build_scoring_prompt(tokenizer, question)
    input_ids = scoring.input_ids.to(device)

    with SteeringIntervention(
        model=model,
        layer_idx=layer_idx,
        vector=vector,
        coefficient=coefficient,
        from_position=scoring.steering_start_position,
    ):
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
        )

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    a_prob = probs[scoring.answer_token_ids["(A)"]].item()
    b_prob = probs[scoring.answer_token_ids["(B)"]].item()
    denom = a_prob + b_prob

    if denom <= 0.0:
        raise RuntimeError("A/B probability mass is zero; cannot renormalize.")

    return {
        "a_prob": a_prob,
        "b_prob": b_prob,
        "a_ab_prob": a_prob / denom,
        "b_ab_prob": b_prob / denom,
    }


@torch.no_grad()
def evaluate_ab(
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
    """Evaluate mean probability of the answer matching the target behavior."""

    if len(dataset) == 0:
        raise ValueError("Behavior evaluation dataset is empty.")

    adapter = prompt_adapter or get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        prompt_family=prompt_family,
    )

    matching_probs = []

    for item in tqdm(dataset, desc=f"Layer {layer_idx}, alpha={coefficient}"):
        target = _validate_target(item["answer_matching_behavior"])
        scores = _score_binary_item(
            model=model,
            tokenizer=tokenizer,
            question=item["question"],
            layer_idx=layer_idx,
            vector=vector,
            coefficient=coefficient,
            adapter=adapter,
            device=device,
        )

        matching_probs.append(
            scores["a_ab_prob"] if target == "(A)" else scores["b_ab_prob"]
        )

    return {
        "behavior_score": sum(matching_probs) / len(matching_probs),
        "num_examples": len(matching_probs),
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
    """Evaluate binary MMLU correct-answer probability and A/B accuracy."""

    if len(dataset) == 0:
        raise ValueError("MMLU evaluation dataset is empty.")

    adapter = prompt_adapter or get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        prompt_family=prompt_family,
    )

    rows = []

    for item in tqdm(dataset, desc=f"MMLU L{layer_idx}, alpha={coefficient}"):
        correct = _validate_target(item["correct"])
        scores = _score_binary_item(
            model=model,
            tokenizer=tokenizer,
            question=item["prompt"],
            layer_idx=layer_idx,
            vector=vector,
            coefficient=coefficient,
            adapter=adapter,
            device=device,
        )

        correct_prob = (
            scores["a_ab_prob"] if correct == "(A)" else scores["b_ab_prob"]
        )
        prediction = (
            "(A)" if scores["a_ab_prob"] >= scores["b_ab_prob"] else "(B)"
        )

        rows.append(
            {
                "category": item["category"],
                "correct": correct,
                "incorrect": item["incorrect"],
                **scores,
                "correct_prob": correct_prob,
                "prediction": prediction,
                "is_correct": prediction == correct,
            }
        )

    capability_score = sum(row["correct_prob"] for row in rows) / len(rows)
    accuracy = sum(row["is_correct"] for row in rows) / len(rows)

    return {
        "capability_score": capability_score,
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
    """Lightweight MMLU wrapper retaining the original public API."""

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
        "capability_score": result["capability_score"],
        "num_examples": result["num_examples"],
    }
