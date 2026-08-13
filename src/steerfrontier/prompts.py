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


def find_last_subsequence(tokens, subsequence) -> int:
    """Return the starting index of the last occurrence of subsequence."""
    n = len(tokens)
    m = len(subsequence)

    for i in range(n - m, -1, -1):
        if tokens[i:i + m] == subsequence:
            return i

    raise ValueError("Subsequence not found in token sequence.")


def get_instruction_end_position(
    tokenizer: PreTrainedTokenizer,
    input_ids,
) -> int:
    """
    Return the position of the final token in the last [/INST] marker.

    Steering begins at this position to match the original CAA setup.
    """

    end_tokens = tokenizer.encode(
        E_INST,
        add_special_tokens=False,
    )

    tokens = input_ids.tolist()

    start = find_last_subsequence(
        tokens,
        end_tokens,
    )

    return start + len(end_tokens) - 1
