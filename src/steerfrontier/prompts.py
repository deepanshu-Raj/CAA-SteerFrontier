from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import PreTrainedTokenizer

from steerfrontier.models import model_name_from_objects

B_INST = "[INST]"
E_INST = "[/INST]"

VALID_ANSWERS = {f"({chr(code)})" for code in range(ord("A"), ord("Z") + 1)}


@dataclass(frozen=True)
class ContrastivePrompt:
    """Tokenized prompt used for CAA vector extraction."""

    input_ids: torch.Tensor
    answer_position: int


@dataclass(frozen=True)
class ScoringPrompt:
    """Prompt state from which answer-choice tokens are scored."""

    input_ids: torch.Tensor
    steering_start_position: int
    answer_token_ids: Dict[str, int]


class PromptAdapter(ABC):
    """Model-family-specific prompt/tokenization behavior."""

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
        raise ValueError(
            f"Expected an answer label like '(A)' ... '(Z)', got {answer!r}."
        )

    return answer


def _answer_letter(answer: str) -> str:
    return _validate_answer(answer)[1]


def _encode_1d(tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )

    return ids[0]


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(ids) != 1:
        raise ValueError(f"Expected {text!r} to tokenize to one token, got {ids}.")

    return int(ids[0])


def find_last_subsequence(tokens, subsequence) -> int:
    """Return the starting index of the last occurrence of subsequence."""

    tokens = list(tokens)
    subsequence = list(subsequence)

    n = len(tokens)
    m = len(subsequence)

    for i in range(n - m, -1, -1):
        if tokens[i : i + m] == subsequence:
            return i

    raise ValueError("Subsequence not found in token sequence.")


# ---------------------------------------------------------------------------
# Llama-2 Chat
# ---------------------------------------------------------------------------


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
    text = format_llama2_chat(
        question,
        answer,
    )

    return tokenizer.encode(
        text,
        return_tensors="pt",
    )


def get_answer_token_position(
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: str,
) -> int:
    """
    Return the token position containing the answer letter.

    Verified Llama-2 formatting ends with:
        '(' 'A/B/C/...' ')'
    """

    answer = _validate_answer(answer)

    tokens = tokenize_llama2_chat(
        tokenizer,
        question,
        answer,
    )[0]

    answer_position = len(tokens) - 2

    decoded = tokenizer.decode([tokens[answer_position]])

    expected = _answer_letter(answer)

    if decoded.strip() != expected:
        raise ValueError(
            f"Expected answer token {expected!r}, "
            f"found {decoded!r} at position {answer_position}."
        )

    return answer_position


def get_instruction_end_position(
    tokenizer: PreTrainedTokenizer,
    input_ids,
) -> int:
    """Return the final token of the last [/INST] marker."""

    end_tokens = tokenizer.encode(
        E_INST,
        add_special_tokens=False,
    )

    start = find_last_subsequence(
        input_ids.tolist(),
        end_tokens,
    )

    return start + len(end_tokens) - 1


def _llama2_answer_token_ids(tokenizer) -> Dict[str, int]:
    """
    Build label -> answer-letter token mapping.

    Unsupported labels are omitted. Evaluation checks that the specific
    requested pair is available.
    """

    result = {}

    for answer in sorted(VALID_ANSWERS):
        letter = _answer_letter(answer)

        try:
            token_id = _single_token_id(
                tokenizer,
                letter,
            )
        except (KeyError, ValueError):
            continue

        decoded = tokenizer.decode([token_id]).strip()

        if decoded == letter:
            result[answer] = token_id

    return result


class Llama2ChatAdapter(PromptAdapter):
    family = "llama2-chat"

    def build_contrastive_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        question: str,
        answer: str,
    ) -> ContrastivePrompt:
        answer = _validate_answer(answer)

        input_ids = tokenize_llama2_chat(
            tokenizer,
            question,
            answer,
        )

        answer_position = get_answer_token_position(
            tokenizer,
            question,
            answer,
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
        # Continue immediately after the opening parenthesis.
        input_ids = tokenize_llama2_chat(
            tokenizer,
            question,
            "(",
        )

        from_position = get_instruction_end_position(
            tokenizer,
            input_ids[0],
        )

        answer_token_ids = _llama2_answer_token_ids(tokenizer)

        if len(answer_token_ids) < 2:
            raise ValueError(
                "Could not construct at least two answer-choice tokens "
                "for Llama-2 scoring."
            )

        return ScoringPrompt(
            input_ids=input_ids,
            steering_start_position=from_position,
            answer_token_ids=answer_token_ids,
        )


# ---------------------------------------------------------------------------
# Llama-3 Instruct
# ---------------------------------------------------------------------------


class Llama3InstructAdapter(PromptAdapter):
    family = "llama3-instruct"

    @staticmethod
    def _generation_prefix(
        tokenizer: PreTrainedTokenizer,
        question: str,
    ) -> torch.Tensor:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template().")

        messages = [
            {
                "role": "user",
                "content": question.strip(),
            }
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

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

        prefix = self._generation_prefix(
            tokenizer,
            question,
        )

        answer_ids = _encode_1d(
            tokenizer,
            answer,
        )

        if answer_ids.numel() == 0:
            raise ValueError(f"Answer {answer!r} produced no tokens.")

        answer_position = prefix.shape[1]

        input_ids = torch.cat(
            [
                prefix,
                answer_ids.unsqueeze(0),
            ],
            dim=1,
        )

        expected_letter = _answer_letter(answer)

        decoded_first = tokenizer.decode([int(answer_ids[0])])

        if expected_letter not in decoded_first:
            raise ValueError(
                f"Expected first answer token for {answer!r} "
                f"to contain {expected_letter!r}, "
                f"found {decoded_first!r}."
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
        prefix = self._generation_prefix(
            tokenizer,
            question,
        )

        answer_token_ids = {}
        token_to_answer = {}

        for answer in sorted(VALID_ANSWERS):
            try:
                ids = tokenizer.encode(
                    answer,
                    add_special_tokens=False,
                )
            except (KeyError, ValueError):
                continue

            if not ids:
                continue

            token_id = int(ids[0])

            decoded = tokenizer.decode([token_id])

            if _answer_letter(answer) not in decoded:
                continue

            # If two labels share the same first token, first-token scoring
            # cannot distinguish them safely.
            if token_id in token_to_answer:
                continue

            answer_token_ids[answer] = token_id
            token_to_answer[token_id] = answer

        if len(answer_token_ids) < 2:
            raise ValueError(
                "Could not construct at least two distinct answer-bearing "
                "tokens for Llama-3 scoring."
            )

        # State from which the first assistant answer token is predicted.
        from_position = prefix.shape[1] - 1

        return ScoringPrompt(
            input_ids=prefix,
            steering_start_position=from_position,
            answer_token_ids=answer_token_ids,
        )


# ---------------------------------------------------------------------------
# Adapter selection
# ---------------------------------------------------------------------------


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


def infer_prompt_family(
    model=None,
    tokenizer=None,
    model_name: Optional[str] = None,
) -> str:
    """Infer supported prompt family from model/tokenizer metadata."""

    name = (
        model_name
        or model_name_from_objects(
            model=model,
            tokenizer=tokenizer,
        )
        or ""
    )

    normalized = name.lower().replace("_", "-")

    if "llama-3" in normalized or "llama3" in normalized:
        return "llama3-instruct"

    if "llama-2" in normalized or "llama2" in normalized:
        return "llama2-chat"

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
        "Could not infer prompt family. "
        "Pass prompt_family explicitly. "
        f"Observed model name: {name!r}."
    )


def get_prompt_adapter(
    model=None,
    tokenizer=None,
    model_name: Optional[str] = None,
    prompt_family: str = "auto",
) -> PromptAdapter:
    """Return the prompt adapter for the requested model family."""

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
