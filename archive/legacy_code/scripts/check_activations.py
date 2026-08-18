import torch

from steerfrontier.activations import LayerActivationCapture
from steerfrontier.data import load_generation_data
from steerfrontier.models import load_model
from steerfrontier.prompts import (
    get_answer_token_position,
    tokenize_llama2_chat,
)

DATA_PATH = "data/sycophancy/generate.json"


def main():
    print("Loading model...")

    model, tokenizer = load_model()

    item = load_generation_data(DATA_PATH)[0]

    question = item["question"]
    answer = item["answer_matching_behavior"]

    input_ids = tokenize_llama2_chat(
        tokenizer,
        question,
        answer,
    ).to("cuda")

    position = get_answer_token_position(
        tokenizer,
        question,
        answer,
    )

    print("Answer:", answer)
    print("Answer position:", position)
    print(
        "Answer token:",
        repr(tokenizer.decode([input_ids[0, position].item()])),
    )

    with LayerActivationCapture(model) as capture:
        with torch.no_grad():
            capture.clear()

            model(
                input_ids=input_ids,
                use_cache=False,
            )

            activations = capture.at_position(position)

    print("Activation shape:", activations.shape)
    print("dtype:", activations.dtype)

    print("\nLayer norms:")
    for layer in [0, 1, 13, 31]:
        print(f"Layer {layer:2d}: {activations[layer].norm().item():.6f}")


if __name__ == "__main__":
    main()
