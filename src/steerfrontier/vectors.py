import torch
from tqdm import tqdm

from steerfrontier.activations import LayerActivationCapture
from steerfrontier.prompts import (
    get_answer_token_position,
    tokenize_llama2_chat,
)


@torch.no_grad()
def extract_all_layer_vectors(
    model,
    tokenizer,
    dataset: list[dict[str, str]],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract raw CAA steering vectors for all transformer layers.

    For each contrastive example:

        delta_l = h_l(behavior_matching) - h_l(behavior_not_matching)

    The final steering vector for each layer is the mean delta over
    all examples.

    Returns
    -------
    torch.Tensor
        Shape [num_layers, hidden_size], stored on CPU in float32.
    """

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    running_sum = torch.zeros(
        num_layers,
        hidden_size,
        dtype=torch.float32,
        device="cpu",
    )

    with LayerActivationCapture(model) as capture:
        for item in tqdm(dataset, desc="Extracting steering vectors"):
            question = item["question"]

            positive_answer = item["answer_matching_behavior"]
            negative_answer = item["answer_not_matching_behavior"]

            # ---------------------------------------------------------
            # Positive / behavior-matching activation
            # ---------------------------------------------------------
            positive_tokens = tokenize_llama2_chat(
                tokenizer,
                question,
                positive_answer,
            ).to(device)

            positive_position = get_answer_token_position(
                tokenizer,
                question,
                positive_answer,
            )

            capture.clear()

            model(
                input_ids=positive_tokens,
                use_cache=False,
            )

            positive_activations = capture.at_position(positive_position)

            # ---------------------------------------------------------
            # Negative / behavior-not-matching activation
            # ---------------------------------------------------------
            negative_tokens = tokenize_llama2_chat(
                tokenizer,
                question,
                negative_answer,
            ).to(device)

            negative_position = get_answer_token_position(
                tokenizer,
                question,
                negative_answer,
            )

            capture.clear()

            model(
                input_ids=negative_tokens,
                use_cache=False,
            )

            negative_activations = capture.at_position(negative_position)

            # [num_layers, hidden_size]
            running_sum += positive_activations - negative_activations

    return running_sum / len(dataset)
