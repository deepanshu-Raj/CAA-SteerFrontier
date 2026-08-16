import argparse
import json
from pathlib import Path

import torch

from steerfrontier.data import load_generation_data
from steerfrontier.models import load_model
from steerfrontier.representations import (
    extract_contrastive_activations,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behavior",
        required=True,
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-chat-hf",
    )

    parser.add_argument(
        "--data",
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = (
        args.data if args.data is not None else f"data/{args.behavior}/generate.json"
    )

    output_dir = Path(f"representations/{args.behavior}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Behavior:", args.behavior)
    print("Layers:", args.layers)
    print("Dataset:", data_path)

    dataset = load_generation_data(data_path)

    print("Examples:", len(dataset))
    print("Loading model...")

    model, tokenizer = load_model(
        model_name=args.model,
        dtype=torch.float32,
    )

    activations = extract_contrastive_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=args.layers,
    )

    for layer in args.layers:
        output_path = output_dir / f"layer_{layer}.pt"

        torch.save(
            activations[layer],
            output_path,
        )

        pos = activations[layer]["positive"]
        neg = activations[layer]["negative"]

        print(
            f"L{layer:02d} | positive={tuple(pos.shape)} | negative={tuple(neg.shape)}"
        )

        print("Saved:", output_path)

    metadata = {
        "behavior": args.behavior,
        "model": args.model,
        "dataset": data_path,
        "num_examples": len(dataset),
        "layers": args.layers,
        "activation_position": "answer-letter token",
    }

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
