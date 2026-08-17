from typing import Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """Load a tokenizer and causal language model for steering experiments.

    The loader intentionally avoids architecture- or memory-specific optimizations so
    that the Llama-2 and Llama-3 experiment paths remain as comparable as possible.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    model = model.to(device)
    model.eval()

    return model, tokenizer


def get_decoder_layers(model) -> Iterable[torch.nn.Module]:
    """Return the model's ordered decoder-block container.

    Llama-2 and Llama-3 expose ``model.model.layers``. A few common decoder-only
    layouts are supported as fallbacks so the core intervention code is not tied to
    a single Hugging Face class hierarchy.
    """

    candidates = [
        ("model", "layers"),       # Llama / Mistral / Qwen-style HF models
        ("transformer", "h"),      # GPT-2-style models
        ("gpt_neox", "layers"),    # GPT-NeoX-style models
    ]

    for parent_name, child_name in candidates:
        parent = getattr(model, parent_name, None)
        if parent is None:
            continue

        layers = getattr(parent, child_name, None)
        if layers is not None:
            return layers

    raise ValueError(
        "Could not locate decoder layers for model class "
        f"{type(model).__name__}. Add its layer path to get_decoder_layers()."
    )


def model_name_from_objects(model=None, tokenizer=None) -> Optional[str]:
    """Best-effort Hugging Face model identifier from loaded objects."""

    if tokenizer is not None:
        name = getattr(tokenizer, "name_or_path", None)
        if name:
            return str(name)

    if model is not None:
        config = getattr(model, "config", None)
        name = getattr(config, "_name_or_path", None) if config is not None else None
        if name:
            return str(name)

    return None


def model_summary(model) -> dict:
    """Return architecture metadata useful for experiment manifests."""

    config = model.config
    layers = get_decoder_layers(model)

    return {
        "model_class": type(model).__name__,
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": len(layers),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
    }
