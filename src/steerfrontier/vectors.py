"""Contrastive activation extraction for CAA steering directions.

Per-layer vectors are formed from paired positive-minus-negative representations
at model-specific contrastive answer positions.
"""

from typing import Optional

import torch
from tqdm import tqdm

from steerfrontier.activations import LayerActivationCapture
from steerfrontier.models import get_decoder_layers
from steerfrontier.prompts import PromptAdapter, get_prompt_adapter


# =============================================================================
# Vector extraction
# =============================================================================

@torch.no_grad()
def extract_all_layer_vectors(
    model,
    tokenizer,
    dataset,
    device: str = "cuda",
    prompt_adapter: Optional[PromptAdapter] = None,
    prompt_family: str = "auto",
) -> torch.Tensor:
    """Extract raw CAA steering vectors for every decoder layer.

    For each example and layer ``l``::

        delta_l = h_l(answer_matching_behavior) - h_l(answer_not_matching_behavior)

    The returned tensor has shape ``[num_layers, hidden_size]`` and is stored on
    CPU in float32.
    """

    if len(dataset) == 0:
        raise ValueError("Cannot extract a steering vector from an empty dataset.")

    adapter = prompt_adapter or get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        prompt_family=prompt_family,
    )

    num_layers = len(get_decoder_layers(model))
    hidden_size = int(model.config.hidden_size)

    running_sum = torch.zeros(
        num_layers,
        hidden_size,
        dtype=torch.float32,
        device="cpu",
    )

    with LayerActivationCapture(model) as capture:
        for item in tqdm(dataset, desc="Extracting steering vectors"):
            question = item["question"]

            positive = adapter.build_contrastive_prompt(
                tokenizer,
                question,
                item["answer_matching_behavior"],
            )
            negative = adapter.build_contrastive_prompt(
                tokenizer,
                question,
                item["answer_not_matching_behavior"],
            )

            capture.clear()
            model(
                input_ids=positive.input_ids.to(device),
                use_cache=False,
            )
            positive_activations = capture.at_position(positive.answer_position)

            capture.clear()
            model(
                input_ids=negative.input_ids.to(device),
                use_cache=False,
            )
            negative_activations = capture.at_position(negative.answer_position)

            running_sum += positive_activations - negative_activations

    return running_sum / float(len(dataset))
