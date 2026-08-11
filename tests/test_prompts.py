from transformers import AutoTokenizer

from steerfrontier.prompts import (
    format_llama2_chat,
    get_answer_token_position,
    tokenize_llama2_chat,
)

MODEL = "meta-llama/Llama-2-7b-chat-hf"


def test_prompt_format():
    prompt = format_llama2_chat("Question?", "(A)")
    assert prompt == "[INST] Question? [/INST] (A)"


def test_answer_token_position():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    question = "Choose one:\n (A) Foo\n (B) Bar"

    tokens = tokenize_llama2_chat(
        tokenizer,
        question,
        "(A)",
    )[0]

    position = get_answer_token_position(
        tokenizer,
        question,
        "(A)",
    )

    assert tokenizer.decode([tokens[position]]).strip() == "A"
