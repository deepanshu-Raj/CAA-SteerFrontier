import torch
from tqdm import tqdm

from steerfrontier.prompts import (
    get_instruction_end_position,
    tokenize_llama2_chat,
)
from steerfrontier.steering import SteeringIntervention


@torch.no_grad()
def evaluate_ab(
    model,
    tokenizer,
    dataset: list[dict[str, str]],
    layer_idx: int,
    vector: torch.Tensor,
    coefficient: float,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Evaluate behavioral probability on held-out A/B questions.
    """

    a_token_id = tokenizer.convert_tokens_to_ids("A")
    b_token_id = tokenizer.convert_tokens_to_ids("B")

    matching_probs = []

    for item in tqdm(
        dataset,
        desc=f"Layer {layer_idx}, alpha={coefficient}",
    ):
        # As in original CAA evaluation:
        # the model continues after "(".
        input_ids = tokenize_llama2_chat(
            tokenizer,
            item["question"],
            "(",
        ).to(device)

        from_position = get_instruction_end_position(
            tokenizer,
            input_ids[0],
        )

        with SteeringIntervention(
            model=model,
            layer_idx=layer_idx,
            vector=vector,
            coefficient=coefficient,
            from_position=from_position,
        ):
            outputs = model(
                input_ids=input_ids,
                use_cache=False,
            )

        logits = outputs.logits[0, -1, :]

        probs = torch.softmax(logits, dim=-1)

        a_prob = probs[a_token_id].item()
        b_prob = probs[b_token_id].item()

        # Renormalize over A/B.
        denom = a_prob + b_prob

        if "(A)" == item["answer_matching_behavior"]:
            matching_prob = a_prob / denom
        elif "(B)" == item["answer_matching_behavior"]:
            matching_prob = b_prob / denom
        else:
            raise ValueError("Expected behavior-matching answer to be (A) or (B).")

        matching_probs.append(matching_prob)

    return {
        "behavior_score": sum(matching_probs) / len(matching_probs),
        "num_examples": len(matching_probs),
    }
