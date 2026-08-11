import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load tokenizer and causal language model for steering experiments."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    model = model.to(device)
    model.eval()

    return model, tokenizer
