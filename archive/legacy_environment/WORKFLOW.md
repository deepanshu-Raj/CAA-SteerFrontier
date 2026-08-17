# Generic experiment workflow

The YAML config owns all model-, behavior-, and artifact-path-specific settings.
The scripts themselves contain no hard-coded behavior or model names.

## 1. Adapter smoke test

```bash
python scripts/check_model_adapter.py \
  --config configs/llama31_8b_instruct.yaml \
  --behavior refusal
```

## 2. Analysis phase

Runs:

```text
Extraction
  -> shared N-behavior normalization
  -> all-layer coarse sweep
  -> candidate-layer coefficient sweep
  -> layer-selection metrics
  -> STOP
```

For the first Llama 3.1 run, let the runner infer a 9-layer candidate window for
each behavior from the coarse layer sweep:

```bash
python scripts/run_experiment.py \
  --config configs/llama31_8b_instruct.yaml \
  --phase analyze \
  --extract-mode auto
```

If you want to override candidate windows:

```bash
python scripts/run_experiment.py \
  --config configs/llama31_8b_instruct.yaml \
  --phase analyze \
  --candidate-layer-spec \
      sycophancy=10-18 \
      refusal=9-17 \
      hallucination=9-17
```

Review for each behavior:

```text
results/<model>/BEHAVIOR/coefficient_layer_sweep.csv
results/<model>/BEHAVIOR/layer_selection_metrics.csv
```

Layer selection is deliberately manual. The metrics describe steering magnitude,
directional consistency and monotonicity; they do not collapse them into a single
automatic score.

## 3. Finalize phase

After choosing one selected layer per behavior:

```bash
python scripts/run_experiment.py \
  --config configs/llama31_8b_instruct.yaml \
  --phase finalize \
  --selected-layer-spec \
      sycophancy=16 \
      refusal=13 \
      hallucination=13
```

This runs only:

```text
selected layers supplied
  -> MMLU sweep
  -> behavior summary
```

The numbers above are examples only; use the layers selected from the new model's
actual coefficient sweeps.

## 4. Cross-behavior analysis

After finalization, choose one or more common causal layers to inspect:

```bash
python scripts/run_cross_behavior.py \
  --config configs/llama31_8b_instruct.yaml \
  --layers 13 16
```

This saves raw-vector pairwise cosine geometry plus cross-behavior steering CSVs.

## 5. Individual plots

```bash
python scripts/plot_behavior_artifacts.py \
  --config configs/llama31_8b_instruct.yaml \
  --behavior refusal \
  --selected-layer 13 \
  --include-diagnostics
```

## 6. Comparison plots

```bash
python scripts/plot_comparison_artifacts.py \
  --config configs/llama31_8b_instruct.yaml \
  --selected-layer-spec \
      sycophancy=16 \
      refusal=13 \
      hallucination=13 \
  --cross-layer 13
```

Again, layer numbers in these examples are placeholders until the Llama 3.1 sweep
has been reviewed.
