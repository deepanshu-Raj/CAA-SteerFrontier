from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from steerfrontier.activations import LayerActivationCapture
from steerfrontier.config import ExperimentConfig
from steerfrontier.evaluation import evaluate_behavior, evaluate_mmlu_detailed
from steerfrontier.experiment import ExperimentLayout
from steerfrontier.models import get_decoder_layers, model_summary
from steerfrontier.normalization import normalize_behavior_vectors
from steerfrontier.prompts import (
    Llama2ChatAdapter,
    Llama3InstructAdapter,
    PromptAdapter,
    ScoringPrompt,
    get_prompt_adapter,
)
from steerfrontier.steering import SteeringIntervention
from steerfrontier.workflow import (
    compute_layer_selection_metrics,
    infer_candidate_layers,
    parse_layer_specs,
    parse_selected_layers,
)


class IdentityDecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states,
        position_ids=None,
        cache_position=None,
        **kwargs,
    ):
        return (hidden_states,)


class TinyModel(nn.Module):
    """Minimal CPU-only decoder model exposing model.layers like Llama."""

    def __init__(self, num_layers=2, hidden_size=3, vocab_size=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [IdentityDecoderLayer() for _ in range(num_layers)]
        )
        self.config = SimpleNamespace(
            _name_or_path="dummy/llama",
            model_type="llama",
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=1,
            num_key_value_heads=1,
            intermediate_size=hidden_size * 2,
            max_position_embeddings=128,
        )
        self.vocab_size = vocab_size

    def forward(self, input_ids, use_cache=False):
        batch, seq_len = input_ids.shape
        hidden = torch.zeros(
            batch,
            seq_len,
            self.config.hidden_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        position_ids = torch.arange(
            seq_len,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)

        for layer in self.model.layers:
            hidden = layer(hidden, position_ids=position_ids)[0]

        logits = torch.zeros(
            batch,
            seq_len,
            self.vocab_size,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        logits[..., 0] = 1.0 + hidden[..., 0]  # token 0 = A
        logits[..., 1] = 0.0  # token 1 = B
        return SimpleNamespace(logits=logits)


class StaticScoringAdapter(PromptAdapter):
    family = "dummy"

    def build_contrastive_prompt(self, tokenizer, question, answer):
        raise NotImplementedError

    def build_scoring_prompt(self, tokenizer, question):
        return ScoringPrompt(
            input_ids=torch.tensor([[10, 11, 12]], dtype=torch.long),
            steering_start_position=2,
            answer_token_ids={"(A)": 0, "(B)": 1},
        )


class FakeLlama2Tokenizer:
    name_or_path = "meta-llama/Llama-2-7b-chat-hf"

    def encode(self, text, add_special_tokens=True, return_tensors=None):
        if text == "[/INST]":
            ids = [12, 13]
        elif text == "A":
            ids = [319]
        elif text == "B":
            ids = [350]
        elif text.endswith(" (A)"):
            ids = [10, 11, 12, 13, 319, 8]
        elif text.endswith(" (B)"):
            ids = [10, 11, 12, 13, 350, 8]
        elif text.endswith(" ("):
            ids = [10, 11, 12, 13, 7]
        else:
            ids = [10, 11, 12, 13]

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids):
        mapping = {319: "A", 350: "B", 8: ")", 7: "("}
        return "".join(mapping.get(int(i), "?") for i in ids)


class FakeLlama3Tokenizer:
    name_or_path = "meta-llama/Llama-3.1-8B-Instruct"

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        return_tensors,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        return torch.tensor([[50, 51, 52, 53]], dtype=torch.long)

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        mapping = {"(A)": [4444, 8], "(B)": [5462, 8]}
        ids = mapping[text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids):
        mapping = {4444: "(A", 5462: "(B", 8: ")"}
        return "".join(mapping[int(i)] for i in ids)


def test_experiment_layout_is_model_root_agnostic():
    layout = ExperimentLayout(
        data_root=Path("data"),
        vector_root=Path("vectors/example-model"),
        results_root=Path("results/example-model"),
        plots_root=Path("plots/example-model"),
    )
    assert str(layout.vector_data_path("refusal")) == "data/refusal/vector.json"
    assert str(layout.validation_data_path("refusal")) == "data/refusal/validation.json"
    assert str(layout.test_data_path("refusal")) == "data/refusal/test.json"
    assert (
        str(layout.normalized_vectors_path("refusal"))
        == "vectors/example-model/refusal/normalized_vectors.pt"
    )
    assert (
        str(layout.reference_norms_path())
        == "vectors/example-model/normalization/reference_layer_norms.pt"
    )
    assert str(layout.results_dir("refusal")) == "results/example-model/refusal"


def test_experiment_config_dtype_and_defaults():
    config = ExperimentConfig(
        model_name="dummy/model",
        model_label="Dummy",
        model_slug="dummy",
        behaviors=["sycophancy", "refusal", "hallucination"],
        data_root=Path("data"),
        vector_root=Path("vectors/dummy"),
        results_root=Path("results/dummy"),
        plots_root=Path("plots/dummy"),
        dtype="float32",
    )
    assert config.torch_dtype == torch.float32
    assert 0.0 in config.alphas
    assert config.layer_sweep_alphas == [-1.0, 1.0]


def test_layer_spec_parsers():
    assert parse_layer_specs(["sycophancy=10-12", "refusal=13"]) == {
        "sycophancy": [10, 11, 12],
        "refusal": [13],
    }
    assert parse_selected_layers(
        ["sycophancy=16", "refusal=13", "hallucination=13"]
    ) == {
        "sycophancy": 16,
        "refusal": 13,
        "hallucination": 13,
    }


def test_candidate_window_centers_on_largest_coarse_span():
    df = pd.DataFrame(
        [
            {"layer": 0, "behavior_score": 0.50},
            {"layer": 0, "behavior_score": 0.52},
            {"layer": 1, "behavior_score": 0.48},
            {"layer": 1, "behavior_score": 0.55},
            {"layer": 2, "behavior_score": 0.30},
            {"layer": 2, "behavior_score": 0.80},
            {"layer": 3, "behavior_score": 0.49},
            {"layer": 3, "behavior_score": 0.56},
            {"layer": 4, "behavior_score": 0.50},
            {"layer": 4, "behavior_score": 0.53},
        ]
    )
    assert infer_candidate_layers(df, num_layers=5, width=3) == [1, 2, 3]


def test_shared_normalization_equalizes_each_layer_norm():
    vectors = {
        "sycophancy": torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
        "refusal": torch.tensor([[0.0, 4.0], [5.0, 0.0]], dtype=torch.float32),
        "hallucination": torch.tensor([[6.0, 0.0], [0.0, 8.0]], dtype=torch.float32),
    }

    normalized, reference = normalize_behavior_vectors(vectors)
    expected_reference = torch.tensor(
        [(3.0 + 4.0 + 6.0) / 3.0, (4.0 + 5.0 + 8.0) / 3.0]
    )

    assert torch.allclose(reference, expected_reference)
    for behavior_vectors in normalized.values():
        assert torch.allclose(
            behavior_vectors.norm(dim=1),
            reference,
            atol=1e-6,
        )


def test_llama2_adapter_preserves_verified_scoring_contract():
    tokenizer = FakeLlama2Tokenizer()
    adapter = Llama2ChatAdapter()

    positive = adapter.build_contrastive_prompt(tokenizer, "Question?", "(A)")
    scoring = adapter.build_scoring_prompt(tokenizer, "Question?")

    assert positive.answer_position == positive.input_ids.shape[1] - 2
    assert (
        tokenizer.decode([int(positive.input_ids[0, positive.answer_position])]) == "A"
    )
    assert scoring.steering_start_position == 3
    assert scoring.answer_token_ids == {"(A)": 319, "(B)": 350}


def test_llama3_adapter_uses_native_answer_bearing_tokens():
    tokenizer = FakeLlama3Tokenizer()
    adapter = Llama3InstructAdapter()

    positive = adapter.build_contrastive_prompt(tokenizer, "Question?", "(A)")
    negative = adapter.build_contrastive_prompt(tokenizer, "Question?", "(B)")
    scoring = adapter.build_scoring_prompt(tokenizer, "Question?")

    assert positive.answer_position == 4
    assert negative.answer_position == 4
    assert int(positive.input_ids[0, 4]) == 4444
    assert int(negative.input_ids[0, 4]) == 5462
    assert scoring.steering_start_position == 3
    assert scoring.answer_token_ids == {"(A)": 4444, "(B)": 5462}


def test_prompt_adapter_auto_detection():
    assert isinstance(
        get_prompt_adapter(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            prompt_family="auto",
        ),
        Llama2ChatAdapter,
    )
    assert isinstance(
        get_prompt_adapter(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            prompt_family="auto",
        ),
        Llama3InstructAdapter,
    )


def test_decoder_layer_discovery_and_summary():
    model = TinyModel(num_layers=3, hidden_size=5)
    assert len(get_decoder_layers(model)) == 3

    summary = model_summary(model)
    assert summary["num_hidden_layers"] == 3
    assert summary["hidden_size"] == 5
    assert summary["model_type"] == "llama"


def test_activation_capture_returns_requested_layer_position():
    model = TinyModel(num_layers=2, hidden_size=3)
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    positions = torch.arange(4).unsqueeze(0)

    with LayerActivationCapture(model, layers=[1]) as capture:
        output = hidden
        for layer in model.model.layers:
            output = layer(output, position_ids=positions)[0]
        activation = capture.at_position(2)

    assert activation.shape == (1, 3)
    assert torch.equal(activation[0], hidden[0, 2])


def test_steering_only_modifies_positions_at_or_after_boundary():
    model = TinyModel(num_layers=1, hidden_size=3)
    hidden = torch.zeros(1, 4, 3)
    positions = torch.tensor([[0, 1, 2, 3]])
    vector = torch.tensor([1.0, 2.0, 3.0])

    with SteeringIntervention(
        model=model,
        layer_idx=0,
        vector=vector,
        coefficient=2.0,
        from_position=2,
    ):
        steered = model.model.layers[0](
            hidden,
            position_ids=positions,
        )[0]

    assert torch.equal(steered[0, 0], torch.zeros(3))
    assert torch.equal(steered[0, 1], torch.zeros(3))
    assert torch.equal(steered[0, 2], 2.0 * vector)
    assert torch.equal(steered[0, 3], 2.0 * vector)


def test_behavior_evaluation_runs_end_to_end_without_real_model():
    model = TinyModel(num_layers=1, hidden_size=3, vocab_size=4)
    adapter = StaticScoringAdapter()
    vector = torch.tensor([1.0, 0.0, 0.0])
    dataset = [
        {
            "question": "dummy",
            "answer_matching_behavior": "(A)",
            "answer_not_matching_behavior": "(B)",
        }
    ]

    baseline = evaluate_behavior(
        model=model,
        tokenizer=None,
        dataset=dataset,
        layer_idx=0,
        vector=vector,
        coefficient=0.0,
        device="cpu",
        prompt_adapter=adapter,
    )
    positive = evaluate_behavior(
        model=model,
        tokenizer=None,
        dataset=dataset,
        layer_idx=0,
        vector=vector,
        coefficient=1.0,
        device="cpu",
        prompt_adapter=adapter,
    )

    assert baseline["num_examples"] == 1
    assert 0.5 < baseline["behavior_score"] < 1.0
    assert positive["behavior_score"] > baseline["behavior_score"]


def test_mmlu_evaluation_runs_end_to_end_without_real_model():
    model = TinyModel(num_layers=1, hidden_size=3, vocab_size=4)
    adapter = StaticScoringAdapter()
    vector = torch.zeros(3)
    dataset = [
        {
            "prompt": "dummy",
            "correct": "(A)",
            "incorrect": "(B)",
            "category": "dummy_category",
        }
    ]

    result = evaluate_mmlu_detailed(
        model=model,
        tokenizer=None,
        dataset=dataset,
        layer_idx=0,
        vector=vector,
        coefficient=0.0,
        device="cpu",
        prompt_adapter=adapter,
    )

    assert result["num_examples"] == 1
    assert result["accuracy"] == 1.0
    assert result["capability_score"] > 0.5
    assert result["items"][0]["prediction"] == "(A)"


def test_layer_selection_metrics_without_writing(monkeypatch):
    monkeypatch.setattr(
        "pathlib.Path.mkdir",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda *args, **kwargs: None,
    )

    coefficient_df = pd.DataFrame(
        [
            {
                "behavior": "refusal",
                "layer": 13,
                "alpha": -1.0,
                "behavior_score": 0.40,
                "baseline": 0.60,
                "delta_from_baseline": -0.20,
            },
            {
                "behavior": "refusal",
                "layer": 13,
                "alpha": 0.0,
                "behavior_score": 0.60,
                "baseline": 0.60,
                "delta_from_baseline": 0.00,
            },
            {
                "behavior": "refusal",
                "layer": 13,
                "alpha": 1.0,
                "behavior_score": 0.75,
                "baseline": 0.60,
                "delta_from_baseline": 0.15,
            },
        ]
    )

    metrics = compute_layer_selection_metrics(
        behavior="refusal",
        coefficient_df=coefficient_df,
        output_path=Path("DO_NOT_WRITE.csv"),
    )

    row = metrics.iloc[0]
    assert row["layer"] == 13
    assert row["spearman_rho"] == pytest.approx(1.0)
    assert row["monotonic_violations"] == 0
    assert row["negative_gain"] == pytest.approx(0.20)
    assert row["positive_gain"] == pytest.approx(0.15)
    assert row["bidirectional_gain"] == pytest.approx(0.35)
