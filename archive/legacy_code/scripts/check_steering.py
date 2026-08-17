import torch

from steerfrontier.data import load_ab_test_data
from steerfrontier.evaluation import evaluate_ab
from steerfrontier.models import load_model

DATA = "data/sycophancy/test_ab.json"

VECTOR = "../CAA/normalized_vectors/sycophancy/vec_layer_13_Llama-2-7b-chat-hf.pt"


def main():
    print("Loading model...")

    model, tokenizer = load_model(
        dtype=torch.float32,
    )

    dataset = load_ab_test_data(DATA)

    vector = torch.load(
        VECTOR,
        map_location="cpu",
    ).float()

    print("Vector norm:", vector.norm().item())

    for alpha in [-1.0, 0.0, 1.0]:
        result = evaluate_ab(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            layer_idx=13,
            vector=vector,
            coefficient=alpha,
        )

        print(f"alpha={alpha:+.1f} | P(behavior)={result['behavior_score']:.4f}")


if __name__ == "__main__":
    main()
