# CAA-SteerFrontier

**How Far Can You Steer? — Mapping Behavior–Capability Tradeoffs in Contrastive Activation Addition**

This repository implements a model-aware evaluation harness for **Contrastive Activation Addition (CAA)**. It extracts steering vectors directly from contrastive behavioral examples, sweeps intervention layers and coefficients, measures behavioral control and MMLU capability, and compares steering geometry across behaviors and models.

The experiments use:

- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-3.1-8B-Instruct`
- Behaviors: **sycophancy**, **refusal**, and **hallucination**

The implementation is inference-only. No upstream steering vectors are required.

## Method

For behavior $\(b\)$ and layer $\(\ell\)$, the raw CAA vector is

$$
\[
v_{b,\ell} = \frac{1}{N}\sum_i\left(h^{+}_{i,\ell}-h^{-}_{i,\ell}\right)
\]
$$

where the activations are taken at the answer-bearing token of each positive/negative contrast pair.

The canonical experiments use a fully self-contained **3-behavior per-layer normalization**:

$$
\[
m_\ell=\frac{1}{3}\sum_{b \in \{S,R,H\}}\lVert v_{b,\ell}\rVert_2,\qquad\tilde v_{b,\ell}=v_{b,\ell}\frac{m_\ell}{\lVert v_{b,\ell}\rVert_2}.
\]
$$

At inference time,

$$
\[
x'_{\ell,t}=x_{\ell,t}+\alpha \tilde v_{b,\ell}
\]
$$

for answer/generation positions after the model-specific instruction boundary.

The harness uses model-specific prompt adapters so that Llama 2 retains the verified `[INST] ... [/INST]` CAA convention while Llama 3.1 uses its native chat template and native A/B answer-tokenization.

## Repository Layout

```text
configs/
  llama2_7b_chat_3beh.yaml
  llama31_8b_instruct.yaml

data/
  sycophancy/
  refusal/
  hallucination/
  mmlu/

vectors/
  llama2-7b-chat/
  llama3.1-8b-instruct/

results/
  llama2-7b-chat/
  llama3.1-8b-instruct/

plots/
  llama2-7b-chat/
  llama3.1-8b-instruct/
  cross_model/

scripts/
  check_model_adapter.py
  run_experiment.py
  run_cross_behavior.py
  plot_behavior_artifacts.py
  plot_comparison_artifacts.py
  plot_cross_model_artifacts.py

src/steerfrontier/
tests/
```

Historical Llama-2 normalization/reproduction artifacts are retained under `archive/`, but they are **not required** for the canonical reproduction below.

## Setup

The tested environment uses **Python 3.9**, **CUDA 11.8**, and FP32 model loading.

```bash
python3.9 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

Both Meta models are gated on Hugging Face. Make sure your account has access and authenticate before running:

```bash
huggingface-cli login
```

On a cluster, optionally place the Hugging Face cache on persistent storage:

```bash
export HF_HOME=/path/to/persistent/huggingface
```

**Compute environment:** Experiments were run on the University of Chicago CS research cluster using a single NVIDIA L40S GPU (48 GB VRAM) with 128 GB host memory. Models were evaluated in FP32 using CUDA 11.8 and Python 3.9.21; all experiments were inference-only and required no model training or fine-tuning.

### 1. Adapter smoke test

**Llama2-7B Chat**:

```bash
python scripts/check_model_adapter.py \
       --config configs/llama2_7b_chat_3beh.yaml \
       --behavior refusal
```

**Llama3.1-8B Instruct**:

```bash
python scripts/check_model_adapter.py \
       --config configs/llama31_8b_instruct.yaml \
       --behavior refusal
```

Expected sanity checks:

- 32 decoder layers
- hidden size 4096
- residual activation shape `(4096,)`
- non-zero steered-vs-baseline logit difference
- `SMOKE TEST PASSED`

### 2. Analysis phase

The analysis phase runs:

$$
\text{
Extraction\
  → shared 3-behavior normalization\
  → full layer sweep\
  → coefficient sweep\
  → layer-selection metrics\
}
$$

**Llama2-7B Chat**:

```bash
python scripts/run_experiment.py \
       --config configs/llama2_7b_chat_3beh.yaml \
       --phase analyze \
       --extract-mode always
```

**Llama3.1-8B Instruct**:

```bash
python scripts/run_experiment.py \
       --config configs/llama31_8b_instruct.yaml \
       --phase analyze \
       --extract-mode always
```

For each behavior, review:

```text
results/<model>/<behavior>/coefficient_layer_sweep.csv
results/<model>/<behavior>/layer_selection_metrics.csv
```

**Layer selection is deliberately manual. We jointly consider steering magnitude, directional consistency, and monotonicity rather than maximizing behavioral change alone.**

For the reported runs, the selected layers are:

| Behavior | Llama 2 7B Chat | Llama 3.1 8B Instruct |
|---|---:|---:|
| Sycophancy | 16 | 16 |
| Refusal | 13 | 13 |
| Hallucination | 13 | 13 |

### 3. Finalize: MMLU + summaries

**Llama2-7B Chat**:

```bash
python scripts/run_experiment.py \
       --config configs/llama2_7b_chat_3beh.yaml \
       --phase finalize \
       --selected-layer-spec \
           sycophancy=16 \
           refusal=13 \
           hallucination=13
```

**Llama3.1-8B Instruct**:

```bash
python scripts/run_experiment.py \
       --config configs/llama31_8b_instruct.yaml \
       --phase finalize \
       --selected-layer-spec \
           sycophancy=16 \
           refusal=13 \
           hallucination=13
```

This evaluates the fixed 570-example binary MMLU subset across:

$$
\[
\alpha \in \{-2,-1.5,-1,-0.5,0,0.5,1,1.5,2\}
\]
$$

and writes one `summary.json` per behavior.

As a coarse reproduction check, the unsteered MMLU probability score is approximately:

| Model | $C(0)$ |
|---|---:|
| Llama 2 7B Chat | 0.65 |
| Llama 3.1 8B Instruct | 0.71 |

Small numerical differences can occur across library/CUDA environments.

### 4. Cross-behavior geometry and causal transfer

Run the same common-layer analysis for each model:

**Llama2-7B Chat**:

```bash
python scripts/run_cross_behavior.py \
       --config configs/llama2_7b_chat_3beh.yaml \
       --layers 13 16 \
       --alphas -1 0 1
```

**Llama3.1-8B Instruct**:

```bash
python scripts/run_cross_behavior.py \
       --config configs/llama31_8b_instruct.yaml \
       --layers 13 16 \
       --alphas -1 0 1
```

Outputs include:

```text
results/<model>/comparison/
  cross_behavior_vector_cosine.csv
  cross_behavior_steering_layer13.csv
  cross_behavior_steering_layer16.csv
```

A useful geometry sanity check is the refusal–hallucination cosine:

| Model | Layer 13 | Layer 16 |
|---|---:|---:|
| Llama 2 7B Chat | ~ $-0.74$ | ~ $-0.58$ |
| Llama 3.1 8B Instruct | ~ $-0.76$ | ~ $-0.86$ |

### 5. Per-model plots

#### Individual behaviors

**Llama2-7B Chat**:

```bash
python scripts/plot_behavior_artifacts.py \
       --config configs/llama2_7b_chat_3beh.yaml \
       --behavior sycophancy \
       --selected-layer 16 \
       --include-diagnostics

python scripts/plot_behavior_artifacts.py \
       --config configs/llama2_7b_chat_3beh.yaml \
       --behavior refusal \
       --selected-layer 13 \
       --include-diagnostics

python scripts/plot_behavior_artifacts.py \
       --config configs/llama2_7b_chat_3beh.yaml \
       --behavior hallucination \
       --selected-layer 13 \
       --include-diagnostics
```

Use the same commands with `configs/llama31_8b_instruct.yaml` for Llama 3.1.

#### Intra-model cross-behavior comparison

```bash
python scripts/plot_comparison_artifacts.py \
       --config configs/llama2_7b_chat_3beh.yaml \
       --selected-layer-spec \
           sycophancy=16 \
           refusal=13 \
           hallucination=13 \
       --cross-layer 13
```

```bash
python scripts/plot_comparison_artifacts.py \
       --config configs/llama31_8b_instruct.yaml \
       --selected-layer-spec \
           sycophancy=16 \
           refusal=13 \
           hallucination=13 \
       --cross-layer 13
```

### 6. Cross-model figures

```bash
python scripts/plot_cross_model_artifacts.py \
       --configs \
           configs/llama2_7b_chat_3beh.yaml \
           configs/llama31_8b_instruct.yaml \
       --output-dir plots/cross_model \
       --cross-layer 13
```

This generates the cross-model behavioral response, behavioral shift, capability shift, vector-geometry, and causal-transfer figures, along with a selected-layer summary CSV.

## Data

`data/` contains the processed contrastive generation/test examples for sycophancy, refusal, and hallucination, plus the fixed binary MMLU subset used by the experiments.

Steering vectors are always extracted from these local examples. Reproducing the canonical experiment therefore does **not** require cloning the original CAA repository or downloading its precomputed vectors.

## Reference

This project builds on:

**Panickssery et al., *Steering Llama 2 via Contrastive Activation Addition*, ACL 2024.** 
