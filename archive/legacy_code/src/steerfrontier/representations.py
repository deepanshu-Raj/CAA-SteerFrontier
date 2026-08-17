from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from steerfrontier.activations import LayerActivationCapture
from steerfrontier.models import get_decoder_layers
from steerfrontier.prompts import PromptAdapter, get_prompt_adapter


@torch.no_grad()
def extract_contrastive_activations(
    model,
    tokenizer,
    dataset,
    layers: List[int],
    device: str = "cuda",
    prompt_adapter: Optional[PromptAdapter] = None,
    prompt_family: str = "auto",
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Extract positive/negative answer-token activations for selected layers.

    This module is optional diagnostic functionality; the primary CAA vector path
    lives in ``vectors.py``.
    """

    decoder_layers = get_decoder_layers(model)
    for layer in layers:
        if layer < 0 or layer >= len(decoder_layers):
            raise ValueError(
                f"Invalid layer {layer}; model has {len(decoder_layers)} layers."
            )

    adapter = prompt_adapter or get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        prompt_family=prompt_family,
    )

    positive = {layer: [] for layer in layers}
    negative = {layer: [] for layer in layers}

    with LayerActivationCapture(model, layers=layers) as capture:
        for item in tqdm(dataset, desc="Extracting contrastive activations"):
            question = item["question"]

            pos = adapter.build_contrastive_prompt(
                tokenizer,
                question,
                item["answer_matching_behavior"],
            )
            neg = adapter.build_contrastive_prompt(
                tokenizer,
                question,
                item["answer_not_matching_behavior"],
            )

            capture.clear()
            model(input_ids=pos.input_ids.to(device), use_cache=False)
            pos_all = capture.at_position(pos.answer_position)

            capture.clear()
            model(input_ids=neg.input_ids.to(device), use_cache=False)
            neg_all = capture.at_position(neg.answer_position)

            for local_idx, layer in enumerate(layers):
                positive[layer].append(pos_all[local_idx])
                negative[layer].append(neg_all[local_idx])

    return {
        layer: {
            "positive": torch.stack(positive[layer]),
            "negative": torch.stack(negative[layer]),
        }
        for layer in layers
    }
