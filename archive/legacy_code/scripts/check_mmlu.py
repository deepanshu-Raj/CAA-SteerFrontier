import torch

from steerfrontier.data import load_mmlu_data
from steerfrontier.evaluation import evaluate_mmlu
from steerfrontier.models import load_model


def main():
    model, tokenizer = load_model(
        dtype=torch.float32,
    )

    dataset = load_mmlu_data("data/mmlu/test.json")

    vectors = torch.load(
        "vectors/sycophancy/normalized_vectors.pt",
        map_location="cpu",
    ).float()

    print("MMLU examples:", len(dataset))

    result = evaluate_mmlu(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=16,
        vector=vectors[16],
        coefficient=0.0,
    )

    print(f"MMLU capability = {result['capability_score']:.6f}")


if __name__ == "__main__":
    main()
