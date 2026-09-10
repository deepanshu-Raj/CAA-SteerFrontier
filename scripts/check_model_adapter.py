import argparse

import torch

from steerfrontier.activations import LayerActivationCapture
from steerfrontier.config import load_experiment_config
from steerfrontier.data import load_vector_data
from steerfrontier.models import get_decoder_layers, load_model, model_summary
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.steering import SteeringIntervention


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test one configured model/prompt/steering adapter."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--behavior", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    behavior = args.behavior or config.behaviors[0]
    if behavior not in config.behaviors:
        raise ValueError(f"Behavior {behavior!r} is not configured.")

    model, tokenizer = load_model(
        model_name=config.model_name,
        device=args.device,
        dtype=config.torch_dtype,
    )
    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=config.model_name,
        prompt_family=config.prompt_family,
    )

    dataset = load_vector_data(config.layout.vector_data_path(behavior))
    item = dataset[0]
    layers = get_decoder_layers(model)
    layer_idx = args.layer if args.layer is not None else len(layers) // 2

    positive = adapter.build_contrastive_prompt(
        tokenizer, item["question"], item["answer_matching_behavior"]
    )
    negative = adapter.build_contrastive_prompt(
        tokenizer, item["question"], item["answer_not_matching_behavior"]
    )
    scoring = adapter.build_scoring_prompt(tokenizer, item["question"])

    print("Model summary:", model_summary(model))
    print("Prompt family:", adapter.family)
    print("Behavior:", behavior)
    print("Test layer:", layer_idx)
    print("Positive answer position:", positive.answer_position)
    print("Negative answer position:", negative.answer_position)
    print("Scoring steering start:", scoring.steering_start_position)
    print("Scoring length:", scoring.input_ids.shape[1])
    print(
        "A token:",
        scoring.answer_token_ids["(A)"],
        repr(tokenizer.decode([scoring.answer_token_ids["(A)"]])),
    )
    print(
        "B token:",
        scoring.answer_token_ids["(B)"],
        repr(tokenizer.decode([scoring.answer_token_ids["(B)"]])),
    )

    with LayerActivationCapture(model, layers=[layer_idx]) as capture:
        model(input_ids=positive.input_ids.to(args.device), use_cache=False)
        activation = capture.at_position(positive.answer_position)[0]
    print("Captured activation shape:", tuple(activation.shape))

    scoring_ids = scoring.input_ids.to(args.device)
    with torch.no_grad():
        baseline = model(input_ids=scoring_ids, use_cache=False).logits[0, -1, :]

    torch.manual_seed(0)
    direction = torch.randn(model.config.hidden_size, dtype=torch.float32)
    direction = direction / direction.norm()

    with SteeringIntervention(
        model=model,
        layer_idx=layer_idx,
        vector=direction,
        coefficient=1.0,
        from_position=scoring.steering_start_position,
    ):
        with torch.no_grad():
            steered = model(input_ids=scoring_ids, use_cache=False).logits[0, -1, :]

    change = (steered - baseline).norm().item()
    print("||steered logits - baseline logits||:", change)
    if change <= 0.0:
        raise RuntimeError("Steering hook produced no logit change.")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
