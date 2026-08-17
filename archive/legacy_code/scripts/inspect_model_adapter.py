import argparse

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )

    return parser.parse_args()


def find_subsequence(sequence, subsequence):
    matches = []

    n = len(subsequence)

    for i in range(len(sequence) - n + 1):
        if sequence[i : i + n] == subsequence:
            matches.append(i)

    return matches


def main():
    args = parse_args()

    assert torch.cuda.is_available()

    device = torch.device("cuda")

    print("=" * 70)
    print("Loading tokenizer")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Tokenizer:", type(tokenizer).__name__)
    print("Vocabulary size:", len(tokenizer))
    print("BOS:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
    print()

    print("=" * 70)
    print("Loading model")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
    )

    model.to(device)
    model.eval()

    print("Model class:", type(model).__name__)
    print("Config model_type:", model.config.model_type)
    print("Hidden size:", model.config.hidden_size)
    print("Layers:", model.config.num_hidden_layers)
    print(
        "Attention heads:",
        model.config.num_attention_heads,
    )
    print(
        "KV heads:",
        getattr(
            model.config,
            "num_key_value_heads",
            None,
        ),
    )
    print(
        "Intermediate size:",
        model.config.intermediate_size,
    )
    print(
        "Max positions:",
        model.config.max_position_embeddings,
    )
    print()

    # ------------------------------------------------------------
    # Verify decoder-layer access.
    # ------------------------------------------------------------
    layers = model.model.layers

    print("Decoder container:", type(layers).__name__)
    print("Decoder blocks:", len(layers))
    print("Layer 0:", type(layers[0]).__name__)
    print()

    # ============================================================
    # Prompt inspection
    # ============================================================

    question = (
        "Which statement is correct?\n"
        "(A) Paris is the capital of France.\n"
        "(B) Paris is the capital of Germany."
    )

    user_messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    # Prompt used when asking the model to produce an answer.
    generation_ids = tokenizer.apply_chat_template(
        user_messages,
        add_generation_prompt=True,
        tokenize=True,
    )

    print("=" * 70)
    print("GENERATION PROMPT")
    print("=" * 70)

    print(
        tokenizer.decode(
            generation_ids,
            skip_special_tokens=False,
        )
    )

    print("\nToken count:", len(generation_ids))

    # ------------------------------------------------------------
    # Find where the assistant generation header begins.
    #
    # Compare:
    # user-only chat
    # vs
    # user chat + generation prompt.
    # ------------------------------------------------------------
    user_only_ids = tokenizer.apply_chat_template(
        user_messages,
        add_generation_prompt=False,
        tokenize=True,
    )

    assistant_boundary = len(user_only_ids)

    print(
        "User-only tokens:",
        len(user_only_ids),
    )

    print(
        "Assistant-generation boundary:",
        assistant_boundary,
    )

    print("\nTokens around boundary:")

    left = max(
        0,
        assistant_boundary - 5,
    )

    for i in range(
        left,
        len(generation_ids),
    ):
        token_id = generation_ids[i]

        print(f"{i:4d} {token_id:8d} {repr(tokenizer.decode([token_id]))}")

    # ============================================================
    # Inspect A/B answer tokenization
    # ============================================================

    print()
    print("=" * 70)
    print("A/B TOKENIZATION")
    print("=" * 70)

    for text in [
        "A",
        "B",
        "(A)",
        "(B)",
        " A",
        " B",
    ]:
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        print(f"{text!r:8s} -> {ids} -> {[tokenizer.decode([x]) for x in ids]}")

    # ============================================================
    # Full positive / negative contrastive prompts
    # ============================================================

    positive_messages = [
        {
            "role": "user",
            "content": question,
        },
        {
            "role": "assistant",
            "content": "(A)",
        },
    ]

    negative_messages = [
        {
            "role": "user",
            "content": question,
        },
        {
            "role": "assistant",
            "content": "(B)",
        },
    ]

    positive_ids = tokenizer.apply_chat_template(
        positive_messages,
        add_generation_prompt=False,
        tokenize=True,
    )

    negative_ids = tokenizer.apply_chat_template(
        negative_messages,
        add_generation_prompt=False,
        tokenize=True,
    )

    print()
    print("=" * 70)
    print("CONTRASTIVE PROMPTS")
    print("=" * 70)

    print("\nPositive:")
    print(
        tokenizer.decode(
            positive_ids,
            skip_special_tokens=False,
        )
    )

    print("\nNegative:")
    print(
        tokenizer.decode(
            negative_ids,
            skip_special_tokens=False,
        )
    )

    # ------------------------------------------------------------
    # Locate the A/B token inside the complete prompt.
    #
    # Search for tokenized "(A)" and "(B)" subsequences.
    # ------------------------------------------------------------
    a_pattern = tokenizer.encode(
        "(A)",
        add_special_tokens=False,
    )

    b_pattern = tokenizer.encode(
        "(B)",
        add_special_tokens=False,
    )

    a_matches = find_subsequence(
        positive_ids,
        a_pattern,
    )

    b_matches = find_subsequence(
        negative_ids,
        b_pattern,
    )

    print("\n(A) token pattern:", a_pattern)
    print("(A) matches:", a_matches)

    print("(B) token pattern:", b_pattern)
    print("(B) matches:", b_matches)

    # ============================================================
    # Forward hook sanity check
    # ============================================================

    test_layer = model.config.num_hidden_layers // 2

    captured = {}

    def capture_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output

        captured["hidden"] = hidden.detach().cpu()

    handle = layers[test_layer].register_forward_hook(capture_hook)

    input_ids = torch.tensor(
        [positive_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    handle.remove()

    hidden = captured["hidden"]

    print()
    print("=" * 70)
    print("ACTIVATION CAPTURE")
    print("=" * 70)

    print("Test layer:", test_layer)
    print("Hidden shape:", tuple(hidden.shape))

    expected = (
        1,
        len(positive_ids),
        model.config.hidden_size,
    )

    print("Expected shape:", expected)

    assert tuple(hidden.shape) == expected

    # ============================================================
    # Intervention smoke test
    # ============================================================

    generation_input = torch.tensor(
        [generation_ids],
        dtype=torch.long,
        device=device,
    )

    generation_mask = torch.ones_like(generation_input)

    with torch.no_grad():
        baseline_logits = model(
            input_ids=generation_input,
            attention_mask=generation_mask,
        ).logits[:, -1, :]

    # Arbitrary tiny residual direction.
    torch.manual_seed(0)

    direction = torch.randn(
        model.config.hidden_size,
        device=device,
        dtype=torch.float32,
    )

    direction = direction / direction.norm()

    coefficient = 1.0

    def steering_hook(
        module,
        inputs,
        output,
    ):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        modified = hidden.clone()

        modified[
            :,
            assistant_boundary:,
            :,
        ] += coefficient * direction

        if isinstance(output, tuple):
            return (
                modified,
                *output[1:],
            )

        return modified

    handle = layers[test_layer].register_forward_hook(steering_hook)

    with torch.no_grad():
        steered_logits = model(
            input_ids=generation_input,
            attention_mask=generation_mask,
        ).logits[:, -1, :]

    handle.remove()

    logit_change = (steered_logits - baseline_logits).norm().item()

    print()
    print("=" * 70)
    print("STEERING HOOK")
    print("=" * 70)

    print(
        "Assistant boundary:",
        assistant_boundary,
    )

    print(
        "Last prompt position:",
        len(generation_ids) - 1,
    )

    print(
        "||steered logits - baseline logits||:",
        logit_change,
    )

    assert logit_change > 0

    print()
    print("=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
