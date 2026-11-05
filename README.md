# CAA-SteerFrontier 

[![arXiv](https://img.shields.io/badge/arXiv-2611.06001-b31b1b.svg)](https://arxiv.org/abs/2611.06001)

**How Far Can You Steer? Behavior–Capability Frontiers in Contrastive Activation Addition** 

How far can Contrastive Activation Addition (CAA) change a language model's behavior before it degrades capabilities we want to preserve?

This repository contains the implementation, frozen experimental results, and plotting scripts for a study of **four instruction-tuned language models**, **seven behavioral dimensions**, and capability-loss budgets of **1, 3, and 5 percentage points**. The experiments are inference-only; no model weights are updated.

The central protocol is to construct behavior directions, select a steering layer from behavioral validation data, choose a coefficient subject to capability constraints, **freeze the intervention**, and evaluate it on independent holdouts. Post-selection audits examine specificity, cross-behavior transfer, secondary capabilities, out-of-distribution behavior, and open-ended generation. The paper provides the full methodology and statistical definitions.

## Main findings

- Validation and held-out directional gains agree strongly across 109 unique nonzero interventions (Spearman ρ = **0.945**).
- Joint behavior–capability holdout survival among nonzero operating points is **37.5%**, **72.7%**, and **76.8%** at the 1%, 3%, and 5% budgets, respectively.
- Learned CAA directions outperform norm-matched, label-shuffled, and off-target-KL-matched controls, but passing the protected suite does **not** ensure preservation of other capabilities or consistent open-ended behavior.

## Repository layout

```text
configs/                 Model-specific configurations
data/                    Frozen behavior, capability, and audit datasets
src/steerfrontier/       Shared implementation (models, steering, evaluation, statistics)
scripts/                 Experiment, analysis, and plotting entry points
vectors/<model>/         Extracted steering vectors and calibration artifacts
results/<model>/         Per-model frozen experimental records
results/cross_model/     Aggregated summaries used in the paper
plots/                   Paper figures and additional diagnostic plots
tests/                   Unit and workflow smoke tests
```

The `results/` and `plots/` directories intentionally retain model- and behavior-level diagnostics beyond the figures printed in the paper.

## Models and evaluation

| Model | Hugging Face ID |
|---|---|
| Llama 2 7B Chat | `meta-llama/Llama-2-7b-chat-hf` |
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Mistral 7B Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` |
| Qwen2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct` |

The seven behaviors are **sycophancy, refusal, hallucination, corrigibility, survival instinct, myopic reward, and AI coordination**. The capability constraints use **MMLU, ARC-Challenge, and HellaSwag**, each split into a selection portion and an independent holdout. Frozen splits are included under `data/` and reused across models. Dataset provenance and sample counts are documented in the paper.

## Setup

The original runs used **Python 3.11.11**, **PyTorch 2.5.1+cu118**, **Transformers 4.46.3**, **Datasets 3.1.0**, and `float32` model activations on NVIDIA L40S/H100 GPUs. See `requirements.lock.txt`, `requirements.txt`, and `environment.txt` for the recorded environment. A CUDA-capable GPU with sufficient memory is needed for model inference.

Run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Obtain access to any gated models (notably Meta Llama) and authenticate with Hugging Face before running inference:

```bash
huggingface-cli login
# Optional on a cluster:
export HF_HOME=/path/to/persistent/huggingface
```

Run lightweight tests before expensive experiments:

```bash
python -m pip install pytest
pytest -q
```

The open-ended judging stage additionally requires the `openai` package and an API credential; it is **not** needed for the core frontier experiments.

## Reproduce the core experiment

The commands below follow the original pipeline. Frozen data, selected layers, coefficients, results, and figures are already provided for inspection; **rerunning GPU inference is not necessary to browse the published artifacts**.

All four configurations are:

```bash
CONFIGS=(
  configs/llama2_7b_chat.yaml
  configs/llama31_8b_instruct.yaml
  configs/mistral_7b_instruct_v03.yaml
  configs/qwen25_7b_instruct.yaml
)
```

### 1. Verify adapters and calibrate residual scale

```bash
for cfg in "${CONFIGS[@]}"; do
  python scripts/check_model_adapter.py --config "$cfg" --behavior refusal
  python scripts/measure_residual_scale.py --config "$cfg" --examples-per-behavior 100
done
```

The calibration estimates each layer's baseline residual scale, used to report steering dose relative to the residual stream. Only steering directions are norm-adjusted; the residual stream is **not** rescaled.

### 2. Run behavioral layer and coefficient sweeps

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

This constructs model-specific CAA directions and runs the coarse all-layer scan followed by the dense layer/coefficient sweep. Layer selection uses **behavioral validation only**, before any capability-selection information is consulted.

Expected outputs include `results/<model>/<behavior>/layer_sweep.csv`, `coefficient_layer_sweep.csv`, and their item-level files. The paper-selected layers are recorded in each model's `results/<model>/selected_layers.json`.

### 3. Run capability-selection sweeps

At the already selected layers, evaluate steering coefficients on the capability **selection** splits, not the holdouts:

```bash
python scripts/run_capability_selection.py \
  --config configs/llama2_7b_chat.yaml \
  --selected-layers results/llama2-7b-chat/selected_layers.json \
  --batch-size 16 --resume --bootstrap-samples 10000 --bootstrap-seed 42

python scripts/run_capability_selection.py \
  --config configs/llama31_8b_instruct.yaml \
  --selected-layers results/llama3.1-8b-instruct/selected_layers.json \
  --batch-size 16 --resume --bootstrap-samples 10000 --bootstrap-seed 42

python scripts/run_capability_selection.py \
  --config configs/mistral_7b_instruct_v03.yaml \
  --selected-layers results/mistral-7b-instruct-v0.3/selected_layers.json \
  --batch-size 16 --resume --bootstrap-samples 10000 --bootstrap-seed 42

python scripts/run_capability_selection.py \
  --config configs/qwen25_7b_instruct.yaml \
  --selected-layers results/qwen2.5-7b-instruct/selected_layers.json \
  --batch-size 16 --resume --bootstrap-samples 10000 --bootstrap-seed 42
```

The resulting per-item benchmark scores and `capability_selection/summary.csv` are used to impose conservative capability-loss constraints.

### 4. Freeze capability-constrained operating points

```bash
for model in \
  llama2-7b-chat \
  llama3.1-8b-instruct \
  mistral-7b-instruct-v0.3 \
  qwen2.5-7b-instruct; do
  python scripts/select_steering_frontiers.py --results-root "results/$model"
done
```

This selects the largest directional behavioral gain among feasible coefficients for each behavior, direction, and capability budget. The frozen choices are stored in `results/<model>/steering_frontiers.csv` and `selected_coefficients.json`. A zero coefficient is allowed when no useful nonzero choice is feasible.

### 5. Evaluate the frozen interventions on holdouts

```bash
python scripts/run_final_evaluation.py \
  --config configs/llama2_7b_chat.yaml \
  --selected-coefficients results/llama2-7b-chat/selected_coefficients.json \
  --batch-size 16 --resume

python scripts/run_final_evaluation.py \
  --config configs/llama31_8b_instruct.yaml \
  --selected-coefficients results/llama3.1-8b-instruct/selected_coefficients.json \
  --batch-size 16 --resume

python scripts/run_final_evaluation.py \
  --config configs/mistral_7b_instruct_v03.yaml \
  --selected-coefficients results/mistral-7b-instruct-v0.3/selected_coefficients.json \
  --batch-size 16 --resume

python scripts/run_final_evaluation.py \
  --config configs/qwen25_7b_instruct.yaml \
  --selected-coefficients results/qwen2.5-7b-instruct/selected_coefficients.json \
  --batch-size 16 --resume
```

**Do not reselect** layers or coefficients after observing held-out results. Primary outputs are under `results/<model>/final_evaluation/`, including `frontier_test_results.csv` and the behavior/capability holdout summaries.

### 6. Aggregate the paper-level results

```bash
python scripts/analyze_cross_model_results.py
```

The aggregated results under `results/cross_model/` include frozen frontier summaries, validation–test agreement, capability survival, and limiting benchmark counts.

## Post-selection audits

These analyses use the frozen operating points. They do **not** feed back into intervention selection. Commands below reproduce the original analysis workflow when the required GPU or API resources are available.

### Specificity controls

```bash
python scripts/run_all_specificity_controls.py \
  --num-seeds 20 --base-seed 20260917 --bootstrap-samples 10000

python scripts/run_all_label_shuffled_controls.py \
  --num-seeds 20 --base-seed 20260917 --bootstrap-samples 10000

python scripts/run_all_offtarget_kl_controls.py \
  --num-seeds 20 --base-seed 20260917 \
  --offtarget-per-behavior 8 --match-rtol 0.10 --bootstrap-samples 10000

python scripts/analyze_offtarget_kl_controls.py
```

These compare CAA against norm-matched random directions, label-shuffled directions, and random directions matched by off-target next-token KL divergence.

### Geometry and causal transfer

```bash
python scripts/run_all_causal_transfer.py \
  --eta 0.20 --bootstrap-samples 10000 --base-seed 20260918
python scripts/analyze_geometry_transfer.py
python scripts/analyze_vector_geometry.py
```

Transfer is measured for all ordered source–target behavior pairs at seven depth anchors, using a constant residual-relative steering dose.

### Secondary capabilities

```bash
python scripts/run_all_perplexity_secondary.py \
  --epsilon 0.03 --batch-size 4 --bootstrap-samples 10000
python scripts/analyze_perplexity_secondary.py

python scripts/run_all_gsm8k_secondary.py \
  --epsilon 0.03 --batch-size 4 --max-new-tokens 2048 \
  --bootstrap-samples 10000
python scripts/postprocess_gsm8k_results.py
```

These evaluate WikiText-2 perplexity and GSM8K accuracy outside the protected capability suite.

### Out-of-distribution behavior

```bash
python scripts/run_all_ood_sycophancy_vectors.py
python scripts/run_all_ood_generalization.py
python scripts/analyze_ood_generalization.py
```

These evaluate cross-domain sycophancy and transfer of hallucination steering to TruthfulQA.

### Open-ended generation and judging

```bash
python scripts/run_all_open_ended_generation.py
python scripts/repair_open_ended_truncations.py
python scripts/prepare_open_ended_judging.py
python scripts/run_open_ended_judge.py
python scripts/analyze_open_ended_judging.py
```

The initial generation cap is **2056 tokens**, with a **4096-token** confirmation cap for truncated responses. The blinded judge uses **GPT-5.6 Sol**. The frozen analysis contains **288 initial prompt triplets**, **12 exclusions for persistent nontermination**, and **276 retained judged triplets**. Judge setup, rubrics, and limitations are detailed in the paper and saved artifacts.

## Figures and results

The repository includes the plots used in the preprint alongside additional model-level diagnostics. Most can be recreated from frozen CSV outputs without rerunning model inference:

```bash
python scripts/plot_controls.py
python scripts/plot_geometry_transfer.py
python scripts/plot_secondary_capability.py
python scripts/plot_ood_generalization.py
python scripts/plot_open_ended_generation.py
```

For descriptive cross-model diagnostics:

```bash
python scripts/plot_cross_model_artifacts.py \
  --configs "${CONFIGS[@]}" \
  --output-dir plots/cross_model
```

Key artifacts for inspecting the paper's main claims:

```text
results/<model>/selected_layers.json
results/<model>/selected_coefficients.json
results/<model>/final_evaluation/frontier_test_results.csv
results/cross_model/combined_frozen_frontiers.csv
results/cross_model/combined_frontier_test_results.csv
results/cross_model/frontier_survival_overall.csv
results/cross_model/validation_test_correlation.csv
results/cross_model/causal_transfer/association/
results/cross_model/secondary_capability/
results/cross_model/ood_generalization/
results/cross_model/open_ended_generation/
```

## Reproducibility and access

Experiments were run on the University of Chicago SSCS SLURM GPU cluster with NVIDIA L40S and H100 accelerators. The primary confidence intervals use **10,000 paired percentile-bootstrap resamples** (base seed **42**, unless an audit specifies its own seed). Reproducing all experiments is compute-intensive; existing results and manifests are included to support inspection without rerunning every stage.

Model weights are not redistributed. Upstream models and source datasets remain subject to their respective access conditions and licenses. Processed data and manifests are located under `data/`.

## Citation

If you use this code or protocol, please cite:

> Raj, Deepanshu. *How Far Can You Steer? Behavior–Capability Frontiers in Contrastive Activation Addition.* arXiv preprint arXiv:2611.06001, 2026. 

This project builds on **Rimsky et al. (2024), *Steering Llama 2 via Contrastive Activation Addition*.**

## License

The repository code is released under the [MIT License](LICENSE). Third-party data and model weights retain their original terms.
