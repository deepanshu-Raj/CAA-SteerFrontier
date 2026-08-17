import argparse
import json
from pathlib import Path

import torch

from steerfrontier.data import load_generation_data
from steerfrontier.experiment import (
    MODEL,
    generation_path,
    raw_vectors_path,
)
from steerfrontier.models import load_model
from steerfrontier.vectors import extract_all_layer_vectors


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behavior",
        required=True,
    )

    parser.add_argument(
        "--model",
        default=MODEL,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = generation_path(args.behavior)
    vector_path = raw_vectors_path(args.behavior)

    vector_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Behavior:", args.behavior)
    print("Dataset:", data_path)

    dataset = load_generation_data(data_path)

    print("Examples:", len(dataset))
    print("Loading model...")

    model, tokenizer = load_model(
        model_name=args.model,
        dtype=torch.float32,
    )

    vectors = extract_all_layer_vectors(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    torch.save(
        vectors,
        vector_path,
    )

    metadata = {
        "behavior": args.behavior,
        "model": args.model,
        "dataset": str(data_path),
        "num_examples": len(dataset),
        "num_layers": vectors.shape[0],
        "hidden_size": vectors.shape[1],
        "dtype": str(vectors.dtype),
        "vector_definition": "mean(h_matching - h_not_matching)",
    }

    with (vector_path.parent / "metadata.json").open("w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("Saved:", vector_path)
    print("Shape:", vectors.shape)

    print("\nVector norms:")
    for layer, vector in enumerate(vectors):
        print(f"L{layer:02d}: {vector.norm().item():.6f}")


if __name__ == "__main__":
    main()
