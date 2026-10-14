# CAA-SteerFrontier

**How Far Can You Steer? Behavior–Capability Frontiers in Contrastive Activation Addition**

CAA-SteerFrontier studies **how far activation steering can move a language model's behavior before general capability degrades beyond an explicit tolerance**. The repository implements the full experimental pipeline for the accompanying preprint across four instruction-tuned LLMs and seven behavioral dimensions, with strict separation between intervention selection and frozen held-out evaluation.

The project is inference-only: no model parameters are trained or fine-tuned. Steering directions are extracted independently for each model from local contrastive examples.

## Contents

- [Overview](#overview)
- [Mathematical Formulation](#mathematical-formulation)
  - [Contrastive behavior directions](#contrastive-behavior-directions)
  - [Normalization and relative intervention strength](#normalization-and-relative-intervention-strength)
  - [Behavioral response and layer selection](#behavioral-response-and-layer-selection)
  - [Capability-constrained frontier](#capability-constrained-frontier)
  - [Frozen holdout evaluation](#frozen-holdout-evaluation)
  - [Post-selection audits](#post-selection-audits)
- [Experimental Scope](#experimental-scope)
- [Repository Layout](#repository-layout)
- [Setup](#setup)
- [Reproducing the Core Paper Pipeline](#reproducing-the-core-paper-pipeline)
  - [1. Verify model adapters](#1-verify-model-adapters)
  - [2. Measure residual-stream scale](#2-measure-residual-stream-scale)
  - [3. Run behavior-side analysis](#3-run-behavior-side-analysis)
  - [4. Freeze behavior-selected layers](#4-freeze-behavior-selected-layers)
  - [5. Run capability-selection sweeps](#5-run-capability-selection-sweeps)
  - [6. Freeze capability-constrained frontiers](#6-freeze-capability-constrained-frontiers)
  - [7. Run frozen final evaluation](#7-run-frozen-final-evaluation)
  - [8. Aggregate cross-model results](#8-aggregate-cross-model-results)
- [Reproducing Post-Selection Audits](#reproducing-post-selection-audits)
  - [Specificity controls](#specificity-controls)
  - [Geometry and causal transfer](#geometry-and-causal-transfer)
  - [Secondary capability](#secondary-capability)
  - [OOD generalization](#ood-generalization)
  - [Open-ended generation and judging](#open-ended-generation-and-judging)
- [Figures](#figures)
- [Key Output Artifacts](#key-output-artifacts)
- [Compute and Reproducibility Notes](#compute-and-reproducibility-notes)
- [Tests](#tests)
- [Data and Model Access](#data-and-model-access)
- [Citation](#citation)
- [License](#license)

## Overview

For each behavior, the pipeline proceeds in a fixed order:

1. Construct layer-wise Contrastive Activation Addition (CAA) directions from a dedicated vector-construction split;
2. Select an intervention layer using **behavioral validation only**;
3. At that fixed layer, choose a signed coefficient that maximizes directional behavioral gain while satisfying simultaneous capability constraints;
4. **Freeze** the resulting operating point;
5. Evaluate the frozen intervention on untouched behavioral test data and capability holdout data;
6. Run specificity, geometry/transfer, secondary-capability, OOD, and open-ended-generation audits without reselection.

The central object is therefore not a single steering coefficient, but a **behavior–capability frontier** indexed by behavior, steering direction, and capability-loss budget.

## Mathematical Formulation

### Contrastive behavior directions

For behavior $`b`$, let

```math
\mathcal{D}^{\mathrm{vec}}_b
=
\{(x_i,y_i^+,y_i^-)\}_{i=1}^{N_b}
```
be the vector-construction split. If $`\mathbf{h}^{+}_{i,\ell}`$ and $`\mathbf{h}^{-}_{i,\ell}`$ denote residual-stream activations after decoder block $`\ell`$ at the answer-bearing position, the raw CAA direction is

```math
\mathbf{v}_{b,\ell}
=
\frac{1}{N_b}
\sum_{i=1}^{N_b}
\left(
\mathbf{h}^{+}_{i,\ell}
-
\mathbf{h}^{-}_{i,\ell}
\right)
```
Directions are constructed independently at every layer and independently for every model. Validation, capability, and final-test examples are excluded from vector construction.

### Normalization and relative intervention strength

Let $`\mathcal{B}`$ be the seven-behavior set. The within-layer reference norm is

```math
m_\ell
=
\frac{1}{|\mathcal{B}|}
\sum_{b\in\mathcal{B}}
\|\mathbf{v}_{b,\ell}\|_2,
```
and each behavior direction is normalized to that common layer norm:

```math
\widetilde{\mathbf{v}}_{b,\ell}
=
\frac{m_\ell}{\|\mathbf{v}_{b,\ell}\|_2}
\mathbf{v}_{b,\ell}
```
Steering modifies the residual stream as

```math
\mathbf{x}'_{\ell,t}
=
\mathbf{x}_{\ell,t}
+
\alpha\widetilde{\mathbf{v}}_{b,\ell},
\qquad
 t\in\mathcal{T}_{\mathrm{steer}}
```
Only the added vector is normalized; the residual stream itself is **never rescaled**.

Because raw $`\alpha`$ is not directly comparable across layers or architectures, the pipeline also measures the baseline residual scale

```math
r_\ell
=
\mathbb{E}
\left[
\|\mathbf{x}_{\ell,t}\|_2
\right]
```
and reports the dimensionless residual-relative perturbation strength

```math
\eta(\alpha,\ell)
=
\frac{\|\alpha\widetilde{\mathbf{v}}_{b,\ell}\|_2}{r_\ell}
=
|\alpha|\frac{m_\ell}{r_\ell}
```
Thus $`\alpha`$ is the signed intervention coefficient used by the model, while $`\eta`$ measures intervention magnitude relative to the typical residual-stream scale at the intervention layer.

### Behavioral response and layer selection

For validation example $`i`$, let $`z_i^+`$ and $`z_i^-`$ be the logits of the behavior-matching and behavior-opposed alternatives under intervention $`(\alpha,\ell)`$. The two-choice probability assigned to the behavior-matching alternative is

```math
\pi_i^b(\alpha,\ell)
=
\frac{e^{z_i^+}}{e^{z_i^+}+e^{z_i^-}}
```
The behavior score and change from baseline are

```math
B_b(\alpha,\ell)
=
\frac{1}{M_b}\sum_i \pi_i^b(\alpha,\ell),
```
```math
\Delta B_b(\alpha,\ell)
=
B_b(\alpha,\ell)-B_b(0,\ell)
```
Every decoder layer is first scanned at $`\alpha\in\{-1,+1\}`$. The coarse anchor is

```math
\bar{\ell}_b
=
\arg\max_\ell
\left[
B_b(+1,\ell)-B_b(-1,\ell)
\right]
```
A nine-layer candidate window $`\mathcal{L}_b`$ is centered on that anchor. Candidate layers are then evaluated on

```math
\mathcal{A}
=
\{-2,-1.5,-1,-0.5,0,0.5,1,1.5,2\}
```
For each candidate layer,

```math
G_b^+(\ell)
=
\max_{\alpha>0}\Delta B_b(\alpha,\ell),
```
```math
G_b^-(\ell)
=
\max_{\alpha<0}\left[-\Delta B_b(\alpha,\ell)\right],
```
and the behavior-selected layer is

```math
\ell_b^*
=
\arg\max_{\ell\in\mathcal{L}_b}
\left[
G_b^+(\ell)+G_b^-(\ell)
\right]
```
This selects layers supporting strong **bidirectional** behavioral control rather than a single one-sided response.

### Capability-constrained frontier

Once $`\ell_b^*`$ is fixed, coefficient selection protects

```math
\mathcal{K}
=
\{\mathrm{MMLU},\mathrm{ARC\text{-}C},\mathrm{HellaSwag}\}.
```
For capability benchmark $`k`$, let $`\Delta A_k(\alpha)`$ be the accuracy change from baseline and

```math
L_k(\alpha)
=
\mathrm{LCB}_{95}
\left[
\Delta A_k(\alpha)
\right]
```
its paired-bootstrap lower confidence bound. The conservative capability margin is

```math
C(\alpha)
=
\min_{k\in\mathcal{K}}L_k(\alpha)
```
Coefficient $`\alpha`$ is feasible under capability-loss budget $`\varepsilon`$ when

```math
C(\alpha)\ge -\varepsilon
```
For steering direction $`d\in\{-1,+1\}`$, define

```math
G_b^d(\alpha)
=
d\,\Delta B_b(\alpha,\ell_b^*)
```
and the sign-compatible coefficient set

```math
\mathcal{A}_d
=
\{0\}
\cup
\{\alpha\in\mathcal{A}:d\alpha>0\}
```
The frozen operating point is

```math
\alpha^*_{b,d,\varepsilon}
=
\arg\max_{\substack{
\alpha\in\mathcal{A}_d\\
C(\alpha)\ge -\varepsilon
}}
G_b^d(\alpha)
```
The paper evaluates

```math
\varepsilon\in\{0.01,0.03,0.05\}
```
Including $`\alpha=0`$ allows the frontier to retain the unsteered model when no nonzero coefficient is simultaneously useful and capability-feasible.

### Frozen holdout evaluation

After

```math
\theta^*_{b,d,\varepsilon}
=
(\ell_b^*,\alpha^*_{b,d,\varepsilon})
```
is selected, **no component of the intervention is changed**.

The held-out directional behavioral gain is

```math
G_b^{d,\mathrm{test}}
=
d
\left[
B_b^{\mathrm{test}}(\alpha^*_{b,d,\varepsilon},\ell_b^*)
-
B_b^{\mathrm{test}}(0,\ell_b^*)
\right]
```
A nonzero intervention has behavioral support when

```math
\mathrm{LCB}_{95}
\left[
G_b^{d,\mathrm{test}}
\right]
>0
```
It passes the capability holdout only when every protected benchmark satisfies

```math
\mathrm{LCB}_{95}
\left[
\Delta A_k^{\mathrm{hold}}
(\alpha^*_{b,d,\varepsilon})
\right]
\ge -\varepsilon
```
A nonzero frontier claim **survives** only if both conditions hold. Holdout failure never triggers reselection.

### Post-selection audits

Frozen interventions are evaluated with four families of audits.

**Specificity controls.** CAA is compared against norm-matched isotropic random directions, label-shuffled CAA directions, and random directions matched to the CAA intervention's off-target next-token KL divergence.

**Geometry and causal transfer.** Pairwise direction geometry is

```math
\Gamma_{ij}(\ell)
=
\frac{
\mathbf{v}_{i,\ell}^{\top}\mathbf{v}_{j,\ell}
}{
\|\mathbf{v}_{i,\ell}\|_2
\|\mathbf{v}_{j,\ell}\|_2
},
```
while causal transfer from steering behavior $`i`$ and measuring behavior $`j`$ is

```math
T_{i\to j}(\alpha,\ell)
=
B_j^{(i)}(\alpha,\ell)-B_j(0,\ell)
```
Transfer is evaluated at a fixed residual-relative dose $`\eta=0.20`$, using

```math
\alpha_\ell
=
\eta\frac{r_\ell}{m_\ell}
```
**Secondary capability.** Frozen $`\varepsilon=0.03`$ interventions are evaluated on GSM8K and WikiText-2, neither of which participates in frontier selection.

**External behavioral validity.** The repository evaluates cross-domain sycophancy, hallucination transfer to TruthfulQA, and open-ended generation for sycophancy, refusal, and hallucination.

## Experimental Scope

### Models

| Model | Hugging Face ID | Decoder layers | Residual width |
|---|---|---:|---:|
| Llama 2 7B Chat | `meta-llama/Llama-2-7b-chat-hf` | 32 | 4096 |
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 32 | 4096 |
| Mistral 7B Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | 32 | 4096 |
| Qwen2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct` | 28 | 3584 |

All models are evaluated in `float32` and use their native instruction/chat format through model-specific prompt adapters.

### Behaviors

The seven steering dimensions are:

- sycophancy
- refusal
- hallucination
- corrigibility
- survival instinct
- myopic reward
- AI coordination

### Core capability benchmarks

| Benchmark | Selection | Holdout |
|---|---:|---:|
| MMLU | 1500 | 1500 |
| ARC-Challenge | 1418 | 1172 |
| HellaSwag | 1500 | 1500 |

The same processed behavioral split membership and capability split membership are reused across all models.

## Repository Layout

```text
configs/                       model-specific experiment configurations

data/
  <behavior>/                  vector / validation / test behavioral splits
  capability/                  selection and holdout capability splits
  ood/                         OOD sycophancy and TruthfulQA data
  open_ended/                  open-ended evaluation prompts
  secondary/                   GSM8K and WikiText-2 audit data
  source/                      source CAA datasets

vectors/<model>/               extracted / normalized steering vectors

results/<model>/
  <behavior>/                  behavior-side layer/coefficient sweeps
  capability_selection/        protected-benchmark selection sweeps
  final_evaluation/            frozen behavior/capability holdout results
  controls/                    specificity controls
  causal_transfer/             cross-behavior transfer results
  geometry/                    direction geometry
  secondary_capability/        GSM8K / WikiText-2
  ood_generalization/          OOD behavioral audits
  open_ended_generation/       open-ended generations

results/cross_model/           aggregated paper-level summaries

plots/
  per_model/                   model-specific diagnostic figures
  appendix_summary/            layer/frontier summary figures
  controls/                    specificity-control figures
  geometry_transfer/           geometry/transfer figures
  secondary_capability/        GSM8K / WikiText-2 figures
  ood_generalization/          OOD figures
  open_ended_generation/       open-ended figures

scripts/                       experiment / analysis / plotting entry points
src/steerfrontier/             reusable implementation
archive/                       historical experiments; not required for paper reproduction
tests/                         unit and workflow smoke tests
```

## Setup

The current publication harness targets **Python 3.11** and the package versions pinned in `requirements.txt` / `requirements.lock.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For development/tests:

```bash
python -m pip install pytest
```

The recorded project environment is:

```text
Python                3.11.11
PyTorch               2.5.1+cu118
Transformers          4.46.3
Datasets              3.1.0
Accelerate            1.1.1
NumPy                  1.26.4
SciPy                  1.14.1
Pandas                 2.2.3
Scikit-learn           1.5.2
CUDA runtime in torch  11.8
Precision              float32
```

Authenticate with Hugging Face before loading gated Meta models:

```bash
huggingface-cli login
```

On a cluster, it is useful to place the Hugging Face cache on persistent storage:

```bash
export HF_HOME=/path/to/persistent/huggingface
```

The open-ended judge additionally requires the `openai` package and an API credential in the environment.

## Reproducing the Core Paper Pipeline

The commands below should be run from the repository root.

For convenience, define the four canonical configs:

```bash
CONFIGS=(
  configs/llama2_7b_chat.yaml
  configs/llama31_8b_instruct.yaml
  configs/mistral_7b_instruct_v03.yaml
  configs/qwen25_7b_instruct.yaml
)
```

### 1. Verify model adapters

Before running the full experiment, smoke-test model loading, prompt rendering, activation extraction, and steering:

```bash
for cfg in "${CONFIGS[@]}"; do
  python scripts/check_model_adapter.py \
    --config "$cfg" \
    --behavior refusal
done
```

### 2. Measure residual-stream scale

The main behavior analysis requires one unsteered residual-scale calibration per model:

```bash
for cfg in "${CONFIGS[@]}"; do
  python scripts/measure_residual_scale.py \
    --config "$cfg" \
    --examples-per-behavior 100
done
```

This writes the per-layer calibration used to compute $`\eta`$.

### 3. Run behavior-side analysis

For each model, extract directions, run the all-layer coarse sweep, infer the candidate region, run the dense layer/coefficient sweep, and compute paired-bootstrap behavioral diagnostics:

```bash
for cfg in "${CONFIGS[@]}"; do
  python scripts/run_experiment.py \
    --config "$cfg" \
    --phase analyze \
    --extract-mode auto \
    --resume \
    --bootstrap-samples 10000 \
    --bootstrap-seed 42
done
```

Important outputs for each behavior are:

```text
results/<model>/<behavior>/layer_sweep.csv
results/<model>/<behavior>/layer_sweep_items.csv
results/<model>/<behavior>/coefficient_layer_sweep.csv
results/<model>/<behavior>/coefficient_layer_sweep_items.csv
results/<model>/<behavior>/layer_selection_metrics.csv
```

### 4. Freeze behavior-selected layers

Layer selection is determined **only from behavioral validation results** using the bidirectional criterion defined above. The canonical paper-selected layers are:

| Behavior | Llama 2 | Llama 3.1 | Mistral | Qwen2.5 |
|---|---:|---:|---:|---:|
| Sycophancy | 12 | 16 | 13 | 18 |
| Refusal | 13 | 13 | 13 | 18 |
| Hallucination | 13 | 13 | 13 | 18 |
| Corrigibility | 13 | 14 | 13 | 18 |
| Survival instinct | 11 | 13 | 15 | 19 |
| Myopic reward | 13 | 12 | 13 | 18 |
| AI coordination | 13 | 13 | 13 | 17 |

Store these as `selected_layers.json` under each model result root. The expected schema is:

```json
{
  "selected_layers": {
    "sycophancy": 12,
    "refusal": 13,
    "hallucination": 13,
    "corrigibility": 13,
    "survival_instinct": 11,
    "myopic_reward": 13,
    "ai_coordination": 13
  }
}
```

The example above is for Llama 2; use the table values for the other models.

### 5. Run capability-selection sweeps

At the now-frozen behavior-selected layers, evaluate every coefficient on the **capability-selection** splits of MMLU, ARC-Challenge, and HellaSwag:

```bash
python scripts/run_capability_selection.py \
  --config configs/llama2_7b_chat.yaml \
  --selected-layers results/llama2-7b-chat/selected_layers.json \
  --batch-size 16 \
  --resume \
  --bootstrap-samples 10000 \
  --bootstrap-seed 42

python scripts/run_capability_selection.py \
  --config configs/llama31_8b_instruct.yaml \
  --selected-layers results/llama3.1-8b-instruct/selected_layers.json \
  --batch-size 16 \
  --resume \
  --bootstrap-samples 10000 \
  --bootstrap-seed 42

python scripts/run_capability_selection.py \
  --config configs/mistral_7b_instruct_v03.yaml \
  --selected-layers results/mistral-7b-instruct-v0.3/selected_layers.json \
  --batch-size 16 \
  --resume \
  --bootstrap-samples 10000 \
  --bootstrap-seed 42

python scripts/run_capability_selection.py \
  --config configs/qwen25_7b_instruct.yaml \
  --selected-layers results/qwen2.5-7b-instruct/selected_layers.json \
  --batch-size 16 \
  --resume \
  --bootstrap-samples 10000 \
  --bootstrap-seed 42
```

The primary summary is

```text
results/<model>/capability_selection/summary.csv
```

with item-level files retained under the corresponding benchmark directories.

### 6. Freeze capability-constrained frontiers

Select $`\alpha^*_{b,d,\varepsilon}`$ using behavior-validation gain and capability-selection feasibility only:

```bash
for model in \
  llama2-7b-chat \
  llama3.1-8b-instruct \
  mistral-7b-instruct-v0.3 \
  qwen2.5-7b-instruct
do
  python scripts/select_steering_frontiers.py \
    --results-root "results/$model"
done
```

This writes:

```text
results/<model>/steering_frontiers.csv
results/<model>/selected_coefficients.json
```

These files define the **frozen intervention protocol** used downstream.

### 7. Run frozen final evaluation

Evaluate the frozen operating points on untouched behavioral test data and capability holdout data:

```bash
python scripts/run_final_evaluation.py \
  --config configs/llama2_7b_chat.yaml \
  --selected-coefficients results/llama2-7b-chat/selected_coefficients.json \
  --batch-size 16 \
  --resume

python scripts/run_final_evaluation.py \
  --config configs/llama31_8b_instruct.yaml \
  --selected-coefficients results/llama3.1-8b-instruct/selected_coefficients.json \
  --batch-size 16 \
  --resume

python scripts/run_final_evaluation.py \
  --config configs/mistral_7b_instruct_v03.yaml \
  --selected-coefficients results/mistral-7b-instruct-v0.3/selected_coefficients.json \
  --batch-size 16 \
  --resume

python scripts/run_final_evaluation.py \
  --config configs/qwen25_7b_instruct.yaml \
  --selected-coefficients results/qwen2.5-7b-instruct/selected_coefficients.json \
  --batch-size 16 \
  --resume
```

The confirmatory outputs are:

```text
results/<model>/final_evaluation/behavior_test_summary.csv
results/<model>/final_evaluation/capability_holdout_summary.csv
results/<model>/final_evaluation/frontier_test_results.csv
results/<model>/final_evaluation/manifest.json
```

Do **not** modify `selected_layers.json` or `selected_coefficients.json` in response to these results.

### 8. Aggregate cross-model results

After all four frozen evaluations are complete:

```bash
python scripts/analyze_cross_model_results.py
```

This performs consistency checks and produces the paper-level cross-model summaries under

```text
results/cross_model/
```

including validation/test agreement, frontier survival, limiting capability benchmarks, fractional intervention depth, and unique nonzero intervention summaries.

## Reproducing Post-Selection Audits

All audits below are **post-selection**. None may be used to change a behavior direction, selected layer, or frozen coefficient.

### Specificity controls

#### Norm-matched random directions

```bash
python scripts/run_all_specificity_controls.py \
  --num-seeds 20 \
  --base-seed 20260917 \
  --bootstrap-samples 10000
```

#### Label-shuffled directions

```bash
python scripts/run_all_label_shuffled_controls.py \
  --num-seeds 20 \
  --base-seed 20260917 \
  --bootstrap-samples 10000
```

#### Off-target-KL-matched random directions

```bash
python scripts/run_all_offtarget_kl_controls.py \
  --num-seeds 20 \
  --base-seed 20260917 \
  --offtarget-per-behavior 8 \
  --match-rtol 0.10 \
  --bootstrap-samples 10000
```

Aggregate the KL-matched control results with:

```bash
python scripts/analyze_offtarget_kl_controls.py
```

### Geometry and causal transfer

Run all ordered source-target transfer experiments at fixed relative dose $`\eta=0.20`$:

```bash
python scripts/run_all_causal_transfer.py \
  --eta 0.20 \
  --bootstrap-samples 10000 \
  --base-seed 20260918
```

Then analyze geometry-transfer association:

```bash
python scripts/analyze_geometry_transfer.py
```

Vector-geometry summaries can be generated independently with:

```bash
python scripts/analyze_vector_geometry.py
```

### Secondary capability

The secondary-capability audits use the frozen $`\varepsilon=0.03`$ interventions.

#### WikiText-2 perplexity

```bash
python scripts/run_all_perplexity_secondary.py \
  --epsilon 0.03 \
  --batch-size 4 \
  --bootstrap-samples 10000

python scripts/analyze_perplexity_secondary.py
```

#### GSM8K

```bash
python scripts/run_all_gsm8k_secondary.py \
  --epsilon 0.03 \
  --batch-size 4 \
  --max-new-tokens 2048 \
  --bootstrap-samples 10000

python scripts/postprocess_gsm8k_results.py
```

### OOD generalization

Prepare domain-specific sycophancy directions:

```bash
python scripts/run_all_ood_sycophancy_vectors.py
```

Run the OOD evaluation across all four models:

```bash
python scripts/run_all_ood_generalization.py
```

Aggregate the results:

```bash
python scripts/analyze_ood_generalization.py
```

This covers cross-domain sycophancy and hallucination transfer to TruthfulQA.

### Open-ended generation and judging

Generate the frozen negative / baseline / positive response triplets for sycophancy, refusal, and hallucination:

```bash
python scripts/run_all_open_ended_generation.py
```

The canonical initial generation cap is **2056 tokens**. Persistent truncations are rerun with a **4096-token** confirmation cap:

```bash
python scripts/repair_open_ended_truncations.py
```

Prepare blinded judging packets:

```bash
python scripts/prepare_open_ended_judging.py
```

Run the blinded judge:

```bash
python scripts/run_open_ended_judge.py
```

The released judging script uses **GPT-5.6 Sol**. Model identity and intervention condition are hidden from the judge during scoring.

After all cases are scored:

```bash
python scripts/analyze_open_ended_judging.py
```

The canonical analysis expects 288 raw triplets, 12 persistent nonterminating triplets, and 276 retained judged triplets.

## Figures

Most manuscript figures are regenerated from already-produced CSV/result artifacts rather than rerunning model inference.

After the corresponding analyses are complete:

```bash
python scripts/plot_controls.py
python scripts/plot_geometry_transfer.py
python scripts/plot_secondary_capability.py
python scripts/plot_ood_generalization.py
python scripts/plot_open_ended_generation.py
```

Per-model behavior diagnostics can be generated with `scripts/plot_behavior_artifacts.py`. Cross-model descriptive artifacts can be generated with:

```bash
python scripts/plot_cross_model_artifacts.py \
  --configs \
    configs/llama2_7b_chat.yaml \
    configs/llama31_8b_instruct.yaml \
    configs/mistral_7b_instruct_v03.yaml \
    configs/qwen25_7b_instruct.yaml \
  --output-dir plots/cross_model
```

## Key Output Artifacts

The most useful files for checking a reproduction are:

```text
results/<model>/selected_layers.json
results/<model>/steering_frontiers.csv
results/<model>/selected_coefficients.json
results/<model>/final_evaluation/frontier_test_results.csv

results/cross_model/combined_frozen_frontiers.csv
results/cross_model/combined_frontier_test_results.csv
results/cross_model/validation_test_correlation.csv
results/cross_model/frontier_survival_overall.csv
results/cross_model/unique_nonzero_interventions.csv

results/cross_model/causal_transfer/association/association_summary.csv
results/cross_model/ood_generalization/
results/cross_model/secondary_capability/
results/cross_model/open_ended_generation/
```

The result manifests record frozen protocol details and should be retained with reproduced outputs.

## Compute and Reproducibility Notes

All reported experiments are inference-only. The current repository environment records execution on a SLURM cluster using NVIDIA **L40S** and **H100 NVL** GPUs, with typical jobs using 8 CPU cores.

The current tested software stack records:

- Python 3.11.11
- PyTorch 2.5.1+cu118
- CUDA runtime 11.8 in the PyTorch build
- Transformers 4.46.3
- Datasets 3.1.0
- Accelerate 1.1.1
- float32 model activations

The experiments use **10,000 paired percentile-bootstrap resamples** for the main behavioral and capability confidence intervals with base seed **42** unless a control/audit defines its own frozen seed.

Reproducibility depends on preserving the stage ordering:

```text
vector construction
    ↓
behavior validation / layer selection
    ↓
capability selection / coefficient selection
    ↓
FREEZE
    ↓
behavior test + capability holdout
    ↓
post-selection audits
```

Held-out or audit outcomes must never trigger reselection.

### GPU memory

The canonical runs use 7B–8B models in FP32. Batch size is configurable for capability and generation stages and may need to be reduced on lower-memory GPUs.

### Resume behavior

Long-running core scripts support `--resume` where applicable. Existing outputs are reused only when the script considers them compatible with the requested condition; missing or incomplete conditions are recomputed.

## Tests

Run the repository test suite before launching expensive experiments:

```bash
pytest -q
```

The tests cover data handling, prompt adapters, statistical helpers, script entry points, environment smoke checks, and workflow-level behavior.

## Data and Model Access

Processed behavioral and capability splits used by the experiment are stored under `data/`. Source datasets and model weights remain subject to their original licenses and access terms.

Meta Llama models are gated on Hugging Face and are **not** distributed by this repository. Mistral and Qwen weights are likewise loaded from their upstream Hugging Face repositories.

The repository does not require steering vectors from the original CAA implementation; the canonical pipeline extracts model-specific vectors directly from the included processed contrastive examples.

## Citation

If you use this repository or the behavior–capability frontier protocol, please cite the accompanying preprint:

> Deepanshu Raj. *How Far Can You Steer? Behavior–Capability Frontiers in Contrastive Activation Addition.* 2026.

An arXiv identifier will be added here once the preprint is publicly listed.

This project builds on Contrastive Activation Addition:

> Panickssery et al. *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024.

## License

The code in this repository is released under the [MIT License](LICENSE).

Third-party datasets, model weights, and derived resources remain subject to their respective licenses and terms. See `LICENSE` and the cited upstream sources for details.
