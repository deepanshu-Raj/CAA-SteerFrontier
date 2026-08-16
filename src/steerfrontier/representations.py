from typing import Dict, List

import torch
from tqdm import tqdm

from steerfrontier.activations import LayerActivationCapture
from steerfrontier.prompts import (
    get_answer_token_position,
    tokenize_llama2_chat,
)


@torch.no_grad()
def extract_contrastive_activations(
    model,
    tokenizer,
    dataset,
    layers: List[int],
    device: str = "cuda",
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Extract positive and negative answer-token activations for selected layers.

    Positive:
        activation corresponding to answer_matching_behavior

    Negative:
        activation corresponding to answer_not_matching_behavior

    Returns
    -------
    {
        layer_idx: {
            "positive": Tensor[N, hidden_size],
            "negative": Tensor[N, hidden_size],
        }
    }
    """

    num_layers = model.config.num_hidden_layers

    for layer in layers:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Invalid layer {layer}; model has {num_layers} layers.")

    positive = {layer: [] for layer in layers}

    negative = {layer: [] for layer in layers}

    with LayerActivationCapture(model) as capture:
        for item in tqdm(
            dataset,
            desc="Extracting contrastive activations",
        ):
            question = item["question"]

            pos_answer = item["answer_matching_behavior"]
            neg_answer = item["answer_not_matching_behavior"]

            # ----------------------------------------------------
            # Positive / behavior-matching activation
            # ----------------------------------------------------
            pos_tokens = tokenize_llama2_chat(
                tokenizer,
                question,
                pos_answer,
            ).to(device)

            pos_position = get_answer_token_position(
                tokenizer,
                question,
                pos_answer,
            )

            capture.clear()

            model(
                input_ids=pos_tokens,
                use_cache=False,
            )

            pos_all = capture.at_position(pos_position)

            for layer in layers:
                positive[layer].append(pos_all[layer])

            # ----------------------------------------------------
            # Negative / behavior-not-matching activation
            # ----------------------------------------------------
            neg_tokens = tokenize_llama2_chat(
                tokenizer,
                question,
                neg_answer,
            ).to(device)

            neg_position = get_answer_token_position(
                tokenizer,
                question,
                neg_answer,
            )

            capture.clear()

            model(
                input_ids=neg_tokens,
                use_cache=False,
            )

            neg_all = capture.at_position(neg_position)

            for layer in layers:
                negative[layer].append(neg_all[layer])

    results = {}

    for layer in layers:
        results[layer] = {
            "positive": torch.stack(positive[layer]),
            "negative": torch.stack(negative[layer]),
        }

    return results
