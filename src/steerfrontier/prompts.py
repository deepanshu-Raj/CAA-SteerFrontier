from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import PreTrainedTokenizer

from steerfrontier.models import model_name_from_objects

B_INST = "[INST]"
E_INST = "[/INST]"

VALID_ANSWERS = {
    f"({chr(code)})"
    for code in range(
        ord("A"),
        ord("Z") + 1,
    )
}


@dataclass(frozen=True)
class ContrastivePrompt:
    """
    Tokenized prompt used for CAA vector extraction.
    """

    input_ids: torch.Tensor
    answer_position: int


@dataclass(frozen=True)
class ScoringPrompt:
    """
    Prompt state from which discrete answer-choice tokens are scored.
    """

    input_ids: torch.Tensor
    steering_start_position: int
    answer_token_ids: Dict[str, int]


@dataclass(frozen=True)
class ContinuationPrompt:
    """
    Full prompt + candidate continuation for sequence likelihood scoring.

    continuation_start_position:
        Index of the first continuation token inside input_ids.

    steering_start_position:
        First residual-stream position at which CAA is applied.

    For a continuation beginning at token position t, the model logits
    at position t-1 predict that first continuation token.
    """

    input_ids: torch.Tensor
    continuation_start_position: int
    steering_start_position: int


class PromptAdapter(ABC):
    """
    Model-family-specific prompt/tokenization behavior.
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

    def build_continuation_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        context: str,
        continuation: str,
    ) -> ContinuationPrompt:
        """
        Build a prompt for full candidate-continuation scoring.

        Kept non-abstract so lightweight test adapters that only exercise
        behavior scoring do not need to implement continuation scoring.
        """

        raise NotImplementedError(
            f"{self.family} does not implement continuation scoring."
        )


def _validate_answer(
    answer: str,
) -> str:
    answer = answer.strip()

    if answer not in VALID_ANSWERS:
        raise ValueError(
            f"Expected an answer label like '(A)' ... '(Z)', got {answer!r}."
        )

    return answer


def _answer_letter(
    answer: str,
) -> str:
    return _validate_answer(answer)[1]


def _validate_continuation(
    continuation: str,
) -> str:
    if not isinstance(
        continuation,
        str,
    ):
        raise TypeError("Continuation must be a string.")

    continuation = continuation.strip()

    if not continuation:
        raise ValueError("Continuation cannot be empty.")

    return continuation


def _encode_1d(
    tokenizer,
    text: str,
) -> torch.Tensor:
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )

    return ids[0]


def _single_token_id(
    tokenizer,
    text: str,
) -> int:
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(ids) != 1:
        raise ValueError(f"Expected {text!r} to tokenize to one token, got {ids}.")

    return int(ids[0])


def find_last_subsequence(
    tokens,
    subsequence,
) -> int:
    """
    Return the starting index of the last occurrence of subsequence.
    """

    tokens = list(tokens)

    subsequence = list(subsequence)

    n = len(tokens)

    m = len(subsequence)

    for i in range(
        n - m,
        -1,
        -1,
    ):
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
            f"Expected answer token "
            f"{expected!r}, found "
            f"{decoded!r} at position "
            f"{answer_position}."
        )

    return answer_position


def get_instruction_end_position(
    tokenizer: PreTrainedTokenizer,
    input_ids,
) -> int:
    """
    Return the final token position of the last [/INST] marker.
    """

    end_tokens = tokenizer.encode(
        E_INST,
        add_special_tokens=False,
    )

    start = find_last_subsequence(
        input_ids.tolist(),
        end_tokens,
    )

    return start + len(end_tokens) - 1


def _llama2_answer_token_ids(
    tokenizer,
) -> Dict[str, int]:
    """
    Build label -> answer-letter token mapping.

    Unsupported labels are omitted. Evaluation checks that the
    particular requested labels are available.
    """

    result = {}

    for answer in sorted(VALID_ANSWERS):
        letter = _answer_letter(answer)

        try:
            token_id = _single_token_id(
                tokenizer,
                letter,
            )

        except (
            KeyError,
            ValueError,
        ):
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
            answer_position=(answer_position),
        )

    def build_scoring_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        question: str,
    ) -> ScoringPrompt:
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
                "Could not construct at least "
                "two answer-choice tokens "
                "for Llama-2 scoring."
            )

        return ScoringPrompt(
            input_ids=input_ids,
            steering_start_position=(from_position),
            answer_token_ids=(answer_token_ids),
        )

    def build_continuation_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        context: str,
        continuation: str,
    ) -> ContinuationPrompt:
        continuation = _validate_continuation(continuation)

        prefix = tokenize_llama2_chat(
            tokenizer,
            context,
        )

        continuation_ids = _encode_1d(
            tokenizer,
            " " + continuation,
        )

        if continuation_ids.numel() == 0:
            raise ValueError("Continuation produced no tokens.")

        continuation_start_position = int(prefix.shape[1])

        input_ids = torch.cat(
            [
                prefix,
                continuation_ids.unsqueeze(0),
            ],
            dim=1,
        )

        steering_start_position = get_instruction_end_position(
            tokenizer,
            prefix[0],
        )

        return ContinuationPrompt(
            input_ids=input_ids,
            continuation_start_position=(continuation_start_position),
            steering_start_position=(steering_start_position),
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
        if not hasattr(
            tokenizer,
            "apply_chat_template",
        ):
            raise ValueError("Tokenizer does not support apply_chat_template().")

        messages = [
            {
                "role": "user",
                "content": (question.strip()),
            }
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if (
            not isinstance(
                input_ids,
                torch.Tensor,
            )
            or input_ids.ndim != 2
        ):
            raise ValueError(
                "Expected "
                "apply_chat_template(..., "
                "return_tensors='pt') "
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
                "Expected first answer token "
                f"for {answer!r} to contain "
                f"{expected_letter!r}, "
                f"found {decoded_first!r}."
            )

        return ContrastivePrompt(
            input_ids=input_ids,
            answer_position=(answer_position),
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

            except (
                KeyError,
                ValueError,
            ):
                continue

            if not ids:
                continue

            token_id = int(ids[0])

            decoded = tokenizer.decode([token_id])

            if _answer_letter(answer) not in decoded:
                continue

            if token_id in token_to_answer:
                continue

            answer_token_ids[answer] = token_id

            token_to_answer[token_id] = answer

        if len(answer_token_ids) < 2:
            raise ValueError(
                "Could not construct at least "
                "two distinct answer-bearing "
                "tokens for Llama-3 scoring."
            )

        from_position = prefix.shape[1] - 1

        return ScoringPrompt(
            input_ids=prefix,
            steering_start_position=(from_position),
            answer_token_ids=(answer_token_ids),
        )

    def build_continuation_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        context: str,
        continuation: str,
    ) -> ContinuationPrompt:
        continuation = _validate_continuation(continuation)

        prefix = self._generation_prefix(
            tokenizer,
            context,
        )

        continuation_ids = _encode_1d(
            tokenizer,
            continuation,
        )

        if continuation_ids.numel() == 0:
            raise ValueError("Continuation produced no tokens.")

        continuation_start_position = int(prefix.shape[1])

        input_ids = torch.cat(
            [
                prefix,
                continuation_ids.unsqueeze(0),
            ],
            dim=1,
        )

        steering_start_position = continuation_start_position - 1

        return ContinuationPrompt(
            input_ids=input_ids,
            continuation_start_position=(continuation_start_position),
            steering_start_position=(steering_start_position),
        )


# ---------------------------------------------------------------------------
# Mistral Instruct
# ---------------------------------------------------------------------------


class MistralInstructAdapter(PromptAdapter):
    family = "mistral-instruct"

    @staticmethod
    def _generation_prefix(
        tokenizer: PreTrainedTokenizer,
        question: str,
    ) -> torch.Tensor:
        if not hasattr(
            tokenizer,
            "apply_chat_template",
        ):
            raise ValueError("Tokenizer does not support apply_chat_template().")

        messages = [
            {
                "role": "user",
                "content": (question.strip()),
            }
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if (
            not isinstance(
                input_ids,
                torch.Tensor,
            )
            or input_ids.ndim != 2
        ):
            raise ValueError(
                "Expected "
                "apply_chat_template(..., "
                "return_tensors='pt') "
                "to return a rank-2 tensor."
            )

        return input_ids

    @staticmethod
    def _full_answer_chat(
        tokenizer: PreTrainedTokenizer,
        question: str,
        answer: str,
    ) -> torch.Tensor:
        messages = [
            {
                "role": "user",
                "content": (question.strip()),
            },
            {
                "role": "assistant",
                "content": (answer.strip()),
            },
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )

        if (
            not isinstance(
                input_ids,
                torch.Tensor,
            )
            or input_ids.ndim != 2
        ):
            raise ValueError(
                "Expected full Mistral chat template to return a rank-2 tensor."
            )

        return input_ids

    @staticmethod
    def _answer_components(
        tokenizer: PreTrainedTokenizer,
        answer: str,
    ):
        answer = _validate_answer(answer)

        ids = tokenizer.encode(
            answer,
            add_special_tokens=False,
        )

        if not ids:
            raise ValueError(f"Answer {answer!r} produced no tokens.")

        expected_letter = _answer_letter(answer)

        matching_offsets = []

        for offset, token_id in enumerate(ids):
            decoded = tokenizer.decode([int(token_id)]).strip()

            if decoded == expected_letter:
                matching_offsets.append(offset)

        if len(matching_offsets) != 1:
            raise ValueError(
                f"Expected exactly one "
                f"answer-letter token in "
                f"{answer!r}, got token IDs "
                f"{ids} with letter matches "
                f"{matching_offsets}."
            )

        letter_offset = matching_offsets[0]

        return (
            ids,
            letter_offset,
            int(ids[letter_offset]),
        )

    @staticmethod
    def _answer_token_ids(
        tokenizer: PreTrainedTokenizer,
    ) -> Dict[str, int]:
        result = {}

        token_to_answer = {}

        open_paren_ids = tokenizer.encode(
            "(",
            add_special_tokens=False,
        )

        if len(open_paren_ids) != 1:
            raise ValueError(
                "Expected '(' to tokenize to exactly one token for Mistral scoring."
            )

        open_paren_id = int(open_paren_ids[0])

        for answer in sorted(VALID_ANSWERS):
            try:
                (
                    ids,
                    letter_offset,
                    token_id,
                ) = MistralInstructAdapter._answer_components(
                    tokenizer,
                    answer,
                )

            except (
                KeyError,
                ValueError,
            ):
                continue

            if letter_offset != 1:
                continue

            if int(ids[0]) != open_paren_id:
                continue

            if token_id in token_to_answer:
                continue

            result[answer] = token_id

            token_to_answer[token_id] = answer

        return result

    def build_contrastive_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        question: str,
        answer: str,
    ) -> ContrastivePrompt:
        answer = _validate_answer(answer)

        input_ids = self._full_answer_chat(
            tokenizer,
            question,
            answer,
        )

        (
            answer_ids,
            letter_offset,
            _,
        ) = self._answer_components(
            tokenizer,
            answer,
        )

        answer_start = find_last_subsequence(
            input_ids[0].tolist(),
            answer_ids,
        )

        answer_position = answer_start + letter_offset

        decoded = tokenizer.decode(
            [
                int(
                    input_ids[
                        0,
                        answer_position,
                    ]
                )
            ]
        ).strip()

        expected = _answer_letter(answer)

        if decoded != expected:
            raise ValueError(
                f"Expected Mistral answer "
                f"token {expected!r}, found "
                f"{decoded!r} at position "
                f"{answer_position}."
            )

        return ContrastivePrompt(
            input_ids=input_ids,
            answer_position=(answer_position),
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

        open_paren_ids = tokenizer.encode(
            "(",
            add_special_tokens=False,
        )

        if len(open_paren_ids) != 1:
            raise ValueError(
                "Expected '(' to tokenize to exactly one token for Mistral scoring."
            )

        open_paren = torch.tensor(
            [[int(open_paren_ids[0])]],
            dtype=prefix.dtype,
        )

        input_ids = torch.cat(
            [
                prefix,
                open_paren,
            ],
            dim=1,
        )

        answer_token_ids = self._answer_token_ids(tokenizer)

        if len(answer_token_ids) < 2:
            raise ValueError(
                "Could not construct at least "
                "two distinct context-conditioned "
                "answer-letter tokens for "
                "Mistral scoring."
            )

        steering_start_position = prefix.shape[1] - 1

        return ScoringPrompt(
            input_ids=input_ids,
            steering_start_position=(steering_start_position),
            answer_token_ids=(answer_token_ids),
        )

    def build_continuation_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        context: str,
        continuation: str,
    ) -> ContinuationPrompt:
        continuation = _validate_continuation(continuation)

        prefix = self._generation_prefix(
            tokenizer,
            context,
        )

        continuation_ids = _encode_1d(
            tokenizer,
            continuation,
        )

        if continuation_ids.numel() == 0:
            raise ValueError("Continuation produced no tokens.")

        continuation_start_position = int(prefix.shape[1])

        input_ids = torch.cat(
            [
                prefix,
                continuation_ids.unsqueeze(0),
            ],
            dim=1,
        )

        steering_start_position = continuation_start_position - 1

        return ContinuationPrompt(
            input_ids=input_ids,
            continuation_start_position=(continuation_start_position),
            steering_start_position=(steering_start_position),
        )


# ---------------------------------------------------------------------------
# Qwen 2.5 Instruct
# ---------------------------------------------------------------------------


class Qwen25InstructAdapter(PromptAdapter):
    family = "qwen2.5-instruct"

    @staticmethod
    def _generation_prefix(
        tokenizer: PreTrainedTokenizer,
        question: str,
    ) -> torch.Tensor:
        if not hasattr(
            tokenizer,
            "apply_chat_template",
        ):
            raise ValueError("Tokenizer does not support apply_chat_template().")

        messages = [
            {
                "role": "user",
                "content": (question.strip()),
            }
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if (
            not isinstance(
                input_ids,
                torch.Tensor,
            )
            or input_ids.ndim != 2
        ):
            raise ValueError("Expected Qwen chat template to return a rank-2 tensor.")

        return input_ids

    @staticmethod
    def _answer_token_ids(
        tokenizer: PreTrainedTokenizer,
    ) -> Dict[str, int]:
        """
        Qwen2.5 tokenizes labels such as '(A)' as:

            '(A' ')'

        The first token is answer-bearing and differs across choices.
        """

        result = {}
        token_to_answer = {}

        for answer in sorted(VALID_ANSWERS):
            try:
                ids = tokenizer.encode(
                    answer,
                    add_special_tokens=False,
                )

            except (
                KeyError,
                ValueError,
            ):
                continue

            if not ids:
                continue

            token_id = int(ids[0])

            decoded = tokenizer.decode([token_id])

            expected_letter = _answer_letter(answer)

            if expected_letter not in decoded:
                continue

            if token_id in token_to_answer:
                continue

            result[answer] = token_id

            token_to_answer[token_id] = answer

        return result

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

        answer_position = int(prefix.shape[1])

        first_answer_token = int(answer_ids[0])

        decoded_first = tokenizer.decode([first_answer_token])

        expected_letter = _answer_letter(answer)

        if expected_letter not in decoded_first:
            raise ValueError(
                "Expected first Qwen answer token "
                f"for {answer!r} to contain "
                f"{expected_letter!r}, "
                f"found {decoded_first!r}."
            )

        input_ids = torch.cat(
            [
                prefix,
                answer_ids.unsqueeze(0),
            ],
            dim=1,
        )

        return ContrastivePrompt(
            input_ids=input_ids,
            answer_position=(answer_position),
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

        answer_token_ids = self._answer_token_ids(tokenizer)

        if len(answer_token_ids) < 2:
            raise ValueError(
                "Could not construct at least "
                "two distinct answer-bearing "
                "tokens for Qwen scoring."
            )

        steering_start_position = prefix.shape[1] - 1

        return ScoringPrompt(
            input_ids=prefix,
            steering_start_position=(steering_start_position),
            answer_token_ids=(answer_token_ids),
        )

    def build_continuation_prompt(
        self,
        tokenizer: PreTrainedTokenizer,
        context: str,
        continuation: str,
    ) -> ContinuationPrompt:
        continuation = _validate_continuation(continuation)

        prefix = self._generation_prefix(
            tokenizer,
            context,
        )

        continuation_ids = _encode_1d(
            tokenizer,
            continuation,
        )

        if continuation_ids.numel() == 0:
            raise ValueError("Continuation produced no tokens.")

        continuation_start_position = int(prefix.shape[1])

        input_ids = torch.cat(
            [
                prefix,
                continuation_ids.unsqueeze(0),
            ],
            dim=1,
        )

        steering_start_position = continuation_start_position - 1

        return ContinuationPrompt(
            input_ids=input_ids,
            continuation_start_position=(continuation_start_position),
            steering_start_position=(steering_start_position),
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
    "mistral": MistralInstructAdapter,
    "mistral-instruct": MistralInstructAdapter,
    "qwen": Qwen25InstructAdapter,
    "qwen2": Qwen25InstructAdapter,
    "qwen2.5": Qwen25InstructAdapter,
    "qwen2.5-instruct": Qwen25InstructAdapter,
}


def infer_prompt_family(
    model=None,
    tokenizer=None,
    model_name: Optional[str] = None,
) -> str:
    """
    Infer supported prompt family from model/tokenizer metadata.
    """

    name = (
        model_name
        or model_name_from_objects(
            model=model,
            tokenizer=tokenizer,
        )
        or ""
    )

    normalized = name.lower().replace(
        "_",
        "-",
    )

    if "qwen2.5" in normalized or "qwen2" in normalized or "qwen" in normalized:
        return "qwen2.5-instruct"

    if "mistral" in normalized:
        return "mistral-instruct"

    if "llama-3" in normalized or "llama3" in normalized:
        return "llama3-instruct"

    if "llama-2" in normalized or "llama2" in normalized:
        return "llama2-chat"

    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()

        except Exception:
            vocab = {}

        if "<|im_start|>" in vocab and "<|im_end|>" in vocab:
            return "qwen2.5-instruct"

        if "<|start_header_id|>" in vocab and "<|eot_id|>" in vocab:
            return "llama3-instruct"

        chat_template = (
            getattr(
                tokenizer,
                "chat_template",
                None,
            )
            or ""
        )

        if "[INST]" in chat_template or "[/INST]" in chat_template:
            raise ValueError(
                "Found an [INST]-style chat "
                "template but could not determine "
                "whether it belongs to Llama-2 "
                "or Mistral from model metadata. "
                "Pass prompt_family explicitly."
            )

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
    """
    Return the prompt adapter for the requested model family.
    """

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
            "Unsupported prompt family "
            f"{family!r}. Supported aliases: "
            f"{sorted(_ADAPTERS)}"
        )

    return adapter_cls()
