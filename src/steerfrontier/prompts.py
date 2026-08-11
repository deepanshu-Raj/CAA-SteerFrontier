from typing import Optional

from transformers import PreTrainedTokenizer

B_INST = "[INST]"
E_INST = "[/INST]"


def format_llama2_chat(
    question: str,
    answer: Optional[str] = None,
) -> str:
    """Format a single-turn Llama-2 chat prompt."""

    text = f"{B_INST} {question.strip()} {E_INST}"

    if answer is not None:
        text += f" {answer.strip()}"

    return text


def tokenize_llama2_chat(
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: Optional[str] = None,
):
    text = format_llama2_chat(question, answer)
    return tokenizer.encode(text, return_tensors="pt")


def get_answer_token_position(
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: str,
) -> int:
    """
    Locate the A/B answer-letter token.

    CAA prompts end with:
        ... [/INST] (A)
        ... [/INST] (B)

    For the Llama-2 tokenizer, the final tokens are:
        '('  'A/B'  ')'

    Therefore the answer letter is the second-to-last token.
    """

    tokens = tokenize_llama2_chat(
        tokenizer,
        question,
        answer,
    )[0]

    answer_position = len(tokens) - 2
    decoded = tokenizer.decode([tokens[answer_position]])

    expected = answer.strip("()")

    if decoded.strip() != expected:
        raise ValueError(
            f"Expected answer token {expected!r}, "
            f"found {decoded!r} at position {answer_position}"
        )

    return answer_position
