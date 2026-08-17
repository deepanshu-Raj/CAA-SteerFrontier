from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import PreTrainedTokenizer

from steerfrontier.models import model_name_from_objects

B_INST = "[INST]"
E_INST = "[/INST]"
VALID_ANSWERS = {"(A)", "(B)"}


@dataclass(frozen=True)
class ContrastivePrompt:
    """Tokenized prompt used for CAA vector extraction."""

    input_ids: torch.Tensor
    answer_position: int


@dataclass(frozen=True)
class ScoringPrompt:
    """Tokenized partial-answer prompt used for binary A/B scoring."""

    input_ids: torch.Tensor
    steering_start_position: int
    answer_token_ids: Dict[str, int]


class PromptAdapter(ABC):
    """Model-family-specific prompt/tokenization behavior.

    CAA extraction/evaluation code should depend on this interface rather than on
    concrete chat-template syntax.
    """

    family: str

    @abstractmethod
    def build_contrastive_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        question: str,
        answer: str,
    ) -> ContrastivePrompt:
        raise NotImplementedError

    @abstractmethod
    def build_scoring_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        question: str,
    ) -> ScoringPrompt:
        raise NotImplementedError


def _validate_answer(answer: str) -> str:
    answer = answer.strip()
    if answer not in VALID_ANSWERS:
        raise ValueError(f"Expected one of {sorted(VALID_ANSWERS)}, got {answer!r}.")
    return answer


def _encode_1d(tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return ids[0]


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Expected {text!r} to tokenize to one token, got {ids}. "
            "Binary A/B evaluation needs a model-specific scoring adapter."
        )
    return int(ids[0])


def find_last_subsequence(tokens, subsequence) -> int:
    """Return the starting index of the last occurrence of ``subsequence``."""

    tokens = list(tokens)
    subsequence = list(subsequence)

    n = len(tokens)
    m = len(subsequence)

    for i in range(n - m, -1, -1):
        if tokens[i : i + m] == subsequence:
            return i

    raise ValueError("Subsequence not found in token sequence.")


# -----------------------------------------------------------------------------
# Llama-2 Chat: exact legacy formatting used by the verified reproduction.
# -----------------------------------------------------------------------------


def format_llama2_chat(
    question: str,
    answer: Optional[str] = None,
) -> str:
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
    """Legacy Llama-2 answer-letter position helper."""

    answer = _validate_answer(answer)
    tokens = tokenize_llama2_chat(tokenizer, question, answer)[0]

    # Verified Llama-2 CAA prompt ending: '(' 'A/B' ')'.
    answer_position = len(tokens) - 2
    decoded = tokenizer.decode([tokens[answer_position]])
    expected = answer.strip("()")

    if decoded.strip() != expected:
        raise ValueError(
            f"Expected answer token {expected!r}, found {decoded!r} "
            f"at position {answer_position}."
        )

    return answer_position


def get_instruction_end_position(
    tokenizer: PreTrainedTokenizer,
    input_ids,
) -> int:
    """Legacy Llama-2 steering boundary: final token of the last ``[/INST]``."""

    end_tokens = tokenizer.encode(E_INST, add_special_tokens=False)
    start = find_last_subsequence(input_ids.tolist(), end_tokens)
    return start + len(end_tokens) - 1


class Llama2ChatAdapter(PromptAdapter):
    family = "llama2-chat"

    def build_contrastive_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        question: str,
        answer: str,
    ) -> ContrastivePrompt:
        answer = _validate_answer(answer)
        input_ids = tokenize_llama2_chat(tokenizer, question, answer)
        answer_position = get_answer_token_position(tokenizer, question, answer)
        return ContrastivePrompt(input_ids=input_ids, answer_position=answer_position)

    def build_scoring_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        question: str,
    ) -> ScoringPrompt:
        # Exact original CAA scoring convention: continue after "(".
        input_ids = tokenize_llama2_chat(tokenizer, question, "(")
        from_position = get_instruction_end_position(tokenizer, input_ids[0])

        return ScoringPrompt(
            input_ids=input_ids,
            steering_start_position=from_position,
            answer_token_ids={
                "(A)": _single_token_id(tokenizer, "A"),
                "(B)": _single_token_id(tokenizer, "B"),
            },
        )


# -----------------------------------------------------------------------------
# Llama-3-family Instruct: native Hugging Face chat template.
# -----------------------------------------------------------------------------


class Llama3InstructAdapter(PromptAdapter):
    family = "llama3-instruct"

    @staticmethod
    def _generation_prefix(
        tokenizer: PreTrainedTokenizer,
        question: str,
    ) -> torch.Tensor:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template().")

        messages = [{"role": "user", "content": question.strip()}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # HF returns [1, seq] when return_tensors='pt'.
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError(
                "Expected apply_chat_template(..., return_tensors='pt') "
                "to return a rank-2 tensor."
            )

        return input_ids

    def build_contrastive_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        question: str,
        answer: str,
    ) -> ContrastivePrompt:
        answer = _validate_answer(answer)

        prefix = self._generation_prefix(tokenizer, question)
        answer_ids = _encode_1d(tokenizer, answer)

        if answer_ids.numel() == 0:
            raise ValueError(f"Answer {answer!r} produced no tokens.")

        # The first appended token is the answer-bearing token. In the verified
        # Llama-3.1 tokenizer, '(A)' -> ['(A', ')'] and '(B)' -> ['(B', ')'].
        answer_position = prefix.shape[1]
        input_ids = torch.cat([prefix, answer_ids.unsqueeze(0)], dim=1)

        expected_letter = answer.strip("()")
        decoded_first = tokenizer.decode([int(answer_ids[0])])
        if expected_letter not in decoded_first:
            raise ValueError(
                f"Expected first answer token for {answer!r} to contain "
                f"{expected_letter!r}, found {decoded_first!r}."
            )

        return ContrastivePrompt(
            input_ids=input_ids,
            answer_position=answer_position,
        )

    def build_scoring_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        question: str,
    ) -> ScoringPrompt:
        prefix = self._generation_prefix(tokenizer, question)

        # Unlike Llama-2, the verified Llama-3.1 tokenizer merges the opening
        # parenthesis with the answer letter: '(A)' -> ['(A', ')'] and
        # '(B)' -> ['(B', ')']. Feeding a standalone '(' and then scoring
        # standalone 'A'/'B' would therefore create a non-canonical tokenization.
        # Score the first *native answer-bearing token* directly from the assistant
        # generation prefix instead.
        a_ids = tokenizer.encode("(A)", add_special_tokens=False)
        b_ids = tokenizer.encode("(B)", add_special_tokens=False)
        if not a_ids or not b_ids:
            raise ValueError("Llama-3 A/B labels produced no tokens.")
        if a_ids[0] == b_ids[0]:
            raise ValueError(
                "The first tokens of '(A)' and '(B)' are identical; "
                "a sequence-level scoring adapter is required."
            )

        # The last token of the assistant-generation prefix is the state from
        # which the first answer-bearing token is predicted.
        from_position = prefix.shape[1] - 1

        return ScoringPrompt(
            input_ids=prefix,
            steering_start_position=from_position,
            answer_token_ids={
                "(A)": int(a_ids[0]),
                "(B)": int(b_ids[0]),
            },
        )


_ADAPTERS = {
    "llama2": Llama2ChatAdapter,
    "llama2-chat": Llama2ChatAdapter,
    "llama-2": Llama2ChatAdapter,
    "llama3": Llama3InstructAdapter,
    "llama3-instruct": Llama3InstructAdapter,
    "llama31": Llama3InstructAdapter,
    "llama-3": Llama3InstructAdapter,
    "llama-3.1": Llama3InstructAdapter,
}


def infer_prompt_family(model=None, tokenizer=None, model_name: Optional[str] = None) -> str:
    """Infer the supported chat family from model/tokenizer metadata."""

    name = model_name or model_name_from_objects(model=model, tokenizer=tokenizer) or ""
    normalized = name.lower().replace("_", "-")

    if "llama-3" in normalized or "llama3" in normalized:
        return "llama3-instruct"

    if "llama-2" in normalized or "llama2" in normalized:
        return "llama2-chat"

    # Fallback to tokenizer vocabulary markers when the model path is local.
    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = {}

        if "<|start_header_id|>" in vocab and "<|eot_id|>" in vocab:
            return "llama3-instruct"

        chat_template = getattr(tokenizer, "chat_template", None) or ""
        if "[INST]" in chat_template or "[/INST]" in chat_template:
            return "llama2-chat"

    raise ValueError(
        "Could not infer prompt family. Pass prompt_family explicitly. "
        f"Observed model name: {name!r}."
    )


def get_prompt_adapter(
    model=None,
    tokenizer=None,
    model_name: Optional[str] = None,
    prompt_family: str = "auto",
) -> PromptAdapter:
    """Return a prompt adapter for the requested/loaded model family."""

    family = prompt_family
    if family == "auto":
        family = infer_prompt_family(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
        )

    key = family.lower()
    adapter_cls = _ADAPTERS.get(key)
    if adapter_cls is None:
        raise ValueError(
            f"Unsupported prompt family {family!r}. "
            f"Supported aliases: {sorted(_ADAPTERS)}"
        )

    return adapter_cls()
