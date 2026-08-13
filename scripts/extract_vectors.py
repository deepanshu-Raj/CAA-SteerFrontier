import argparse
import json
from pathlib import Path

import torch

from steerfrontier.data import load_generation_data
from steerfrontier.models import load_model
from steerfrontier.vectors import extract_all_layer_vectors


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/sycophancy/generate.json",
    )

    parser.add_argument(
        "--output-dir",
        default="vectors/sycophancy",
    )

    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-chat-hf",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = load_generation_data(args.data)

    print("Examples:", len(dataset))

    print("Loading model...")

    # FP32 matches our reproduction setup.
    model, tokenizer = load_model(
        model_name=args.model,
        dtype=torch.float32,
    )

    print("Extracting vectors...")

    vectors = extract_all_layer_vectors(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    vector_path = output_dir / "raw_vectors.pt"

    torch.save(vectors, vector_path)

    metadata = {
        "model": args.model,
        "dataset": args.data,
        "num_examples": len(dataset),
        "num_layers": vectors.shape[0],
        "hidden_size": vectors.shape[1],
        "dtype": str(vectors.dtype),
        "vector_definition": "mean(h_matching - h_not_matching)",
    }

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("Saved:", vector_path)
    print("Shape:", vectors.shape)
    print("dtype:", vectors.dtype)

    print("\nVector norms:")
    for layer_idx, vector in enumerate(vectors):
        print(f"Layer {layer_idx:2d}: {vector.norm().item():.6f}")


if __name__ == "__main__":
    main()
