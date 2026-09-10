import argparse

import torch

from steerfrontier.config import load_experiment_config
from steerfrontier.models import load_model
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.workflow import (
    build_shared_normalization,
    compute_layer_selection_metrics,
    extract_or_load_raw_vectors,
    infer_candidate_layers,
    load_behavior_validation,
    load_mmlu,
    load_normalized_vectors,
    load_saved_analysis,
    parse_layer_specs,
    parse_selected_layers,
    print_selection_metrics,
    run_coefficient_sweep,
    run_layer_sweep,
    run_mmlu_sweep,
    write_behavior_summary,
    write_run_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generic CAA experiment runner. The default 'analyze' phase executes "
            "extraction -> shared normalization -> layer sweep -> coefficient "
            "sweep -> selection metrics and then stops for manual layer review. "
            "The 'finalize' phase consumes manually supplied selected layers and "
            "runs MMLU -> summary."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--phase",
        choices=["analyze", "finalize", "full"],
        default="analyze",
    )

    parser.add_argument(
        "--extract-mode",
        choices=["auto", "always", "never"],
        default="auto",
        help=("Used during analyze/full: extract missing, all, or no raw vectors."),
    )

    parser.add_argument(
        "--candidate-layer-spec",
        nargs="*",
        default=[],
        metavar="BEHAVIOR=START-END",
        help=(
            "Optional coefficient-sweep regions. Missing behaviors use an "
            "automatic fixed-width window around the strongest coarse "
            "layer effect."
        ),
    )

    parser.add_argument(
        "--selected-layer-spec",
        nargs="*",
        default=[],
        metavar="BEHAVIOR=LAYER",
        help=("Authoritative manual layer choices for finalize/full."),
    )

    parser.add_argument(
        "--reuse-analysis",
        action="store_true",
        help=("Reuse existing layer/coefficient/metric CSVs during analyze/full."),
    )

    parser.add_argument(
        "--skip-mmlu-items",
        action="store_true",
        help=("Do not save the per-example MMLU CSVs."),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    return parser.parse_args()


def validate_specs(
    config,
    candidate_specs,
    selected_specs,
):
    known = set(config.behaviors)

    unknown_candidates = sorted(set(candidate_specs) - known)

    unknown_selected = sorted(set(selected_specs) - known)

    if unknown_candidates:
        raise ValueError(
            f"Candidate-layer specs contain unknown behaviors: {unknown_candidates}"
        )

    if unknown_selected:
        raise ValueError(
            f"Selected-layer specs contain unknown behaviors: {unknown_selected}"
        )


def load_residual_scale(config):
    """
    Load the precomputed unsteered residual-stream scale r_l.

    This must be generated once per model using:
        scripts/measure_residual_scale.py
    """

    path = config.layout.normalization_dir() / "residual_scale.pt"

    if not path.exists():
        raise FileNotFoundError(
            "Missing residual-scale calibration: "
            f"{path}\n"
            "Run scripts/measure_residual_scale.py "
            "for this model before analysis."
        )

    residual_scale = torch.load(
        path,
        map_location="cpu",
    ).float()

    if residual_scale.ndim != 1:
        raise ValueError(
            "Residual-scale calibration must be rank-1, "
            f"got shape {tuple(residual_scale.shape)}."
        )

    return residual_scale


def run_analysis(
    model,
    tokenizer,
    adapter,
    config,
    extract_mode,
    candidate_specs,
    reuse_analysis,
    device,
):
    raw_vectors = extract_or_load_raw_vectors(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        config=config,
        extract_mode=extract_mode,
        device=device,
    )

    _, normalized_vectors = build_shared_normalization(
        raw_vectors=raw_vectors,
        config=config,
    )

    residual_scale = load_residual_scale(config)

    num_layers = next(iter(normalized_vectors.values())).shape[0]

    if residual_scale.shape[0] != num_layers:
        raise ValueError(
            "Residual-scale layer count does not match steering vectors: "
            f"{residual_scale.shape[0]} != {num_layers}"
        )

    print()
    print(
        "Residual-scale calibration:",
        config.layout.normalization_dir() / "residual_scale.pt",
    )
    print(
        "Layers calibrated:",
        residual_scale.shape[0],
    )

    selected_candidate_map = {}

    for behavior in config.behaviors:
        result_dir = config.layout.results_dir(behavior)

        result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if reuse_analysis:
            (
                layer_df,
                coefficient_df,
                metrics_df,
                candidate_layers,
            ) = load_saved_analysis(
                config,
                behavior,
            )

            print("\n" + "=" * 76)

            print(f"REUSING ANALYSIS — {behavior}")

            print("=" * 76)

        else:
            dataset = load_behavior_validation(
                config,
                behavior,
            )

            vectors = normalized_vectors[behavior]

            if vectors.shape[0] != residual_scale.shape[0]:
                raise ValueError(
                    f"{behavior}: vector layer count "
                    f"{vectors.shape[0]} does not match "
                    f"residual-scale layer count "
                    f"{residual_scale.shape[0]}."
                )

            layer_df = run_layer_sweep(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                behavior=behavior,
                dataset=dataset,
                vectors=vectors,
                residual_scale=residual_scale,
                alphas=config.layer_sweep_alphas,
                output_path=(result_dir / "layer_sweep.csv"),
                device=device,
            )

            candidate_layers = candidate_specs.get(behavior)

            if candidate_layers is None:
                candidate_layers = infer_candidate_layers(
                    layer_sweep_df=layer_df,
                    num_layers=vectors.shape[0],
                    width=config.candidate_width,
                )

                print(f"Auto candidate window for {behavior}: {candidate_layers}")

            else:
                print(f"Configured candidate window for {behavior}: {candidate_layers}")

            invalid = [
                layer
                for layer in candidate_layers
                if (layer < 0 or layer >= vectors.shape[0])
            ]

            if invalid:
                raise ValueError(f"Invalid candidate layers for {behavior}: {invalid}")

            coefficient_df = run_coefficient_sweep(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                behavior=behavior,
                dataset=dataset,
                vectors=vectors,
                residual_scale=residual_scale,
                candidate_layers=candidate_layers,
                alphas=config.alphas,
                output_path=(result_dir / "coefficient_layer_sweep.csv"),
                device=device,
            )

            metrics_df = compute_layer_selection_metrics(
                behavior=behavior,
                coefficient_df=coefficient_df,
                output_path=(result_dir / "layer_selection_metrics.csv"),
            )

        selected_candidate_map[behavior] = candidate_layers

        print()
        print("Layer-selection metrics:")
        print_selection_metrics(metrics_df)

    manifest = write_run_manifest(
        config=config,
        model=model,
        prompt_family=adapter.family,
        phase="analysis",
        extra={
            "candidate_layers": (selected_candidate_map),
            "residual_scale_path": str(
                config.layout.normalization_dir() / "residual_scale.pt"
            ),
            "eta_definition": ("|alpha| * vector_norm / residual_scale"),
        },
    )

    print()
    print("=" * 76)
    print("ANALYSIS COMPLETE — MANUAL LAYER SELECTION")
    print("=" * 76)

    print("Review, for each behavior:")

    for behavior in config.behaviors:
        result_dir = config.layout.results_dir(behavior)

        print(f"  {result_dir / 'coefficient_layer_sweep.csv'}")

        print(f"  {result_dir / 'layer_selection_metrics.csv'}")

    print(
        "Manifest:",
        manifest,
    )

    print()
    print("Then run:")

    print(
        "python scripts/run_experiment.py "
        f"--config <same-config> "
        "--phase finalize "
        "--selected-layer-spec "
        + " ".join(f"{behavior}=LAYER" for behavior in config.behaviors)
    )


def run_finalize(
    model,
    tokenizer,
    adapter,
    config,
    selected_specs,
    skip_mmlu_items,
    device,
):
    missing = [
        behavior for behavior in config.behaviors if behavior not in selected_specs
    ]

    if missing:
        raise ValueError(
            "Finalize requires a manually selected layer "
            "for every behavior. "
            f"Missing: {missing}"
        )

    normalized_vectors = load_normalized_vectors(config)

    mmlu_dataset = load_mmlu(config)

    selected_layers = {}

    for behavior in config.behaviors:
        (
            _,
            coefficient_df,
            metrics_df,
            candidate_layers,
        ) = load_saved_analysis(
            config,
            behavior,
        )

        selected_layer = int(selected_specs[behavior])

        if selected_layer not in set(metrics_df["layer"].astype(int)):
            raise ValueError(
                f"Selected layer L{selected_layer} "
                f"for {behavior} was not part "
                "of the coefficient sweep."
            )

        selected_layers[behavior] = selected_layer

        result_dir = config.layout.results_dir(behavior)

        mmlu_path = result_dir / f"mmlu_layer{selected_layer}.csv"

        items_path = None

        if not skip_mmlu_items:
            items_path = result_dir / (f"mmlu_layer{selected_layer}_items.csv")

        mmlu_df = run_mmlu_sweep(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            behavior=behavior,
            layer=selected_layer,
            vector=(normalized_vectors[behavior][selected_layer]),
            dataset=mmlu_dataset,
            alphas=config.alphas,
            summary_path=mmlu_path,
            items_path=items_path,
            device=device,
        )

        write_behavior_summary(
            config=config,
            behavior=behavior,
            selected_layer=selected_layer,
            candidate_layers=candidate_layers,
            coefficient_df=coefficient_df,
            metrics_df=metrics_df,
            mmlu_df=mmlu_df,
            output_path=(result_dir / "summary.json"),
        )

        print(f"Summary -> {result_dir / 'summary.json'}")

    manifest = write_run_manifest(
        config=config,
        model=model,
        prompt_family=adapter.family,
        phase="finalize",
        extra={"selected_layers": (selected_layers)},
    )

    print()
    print("FINALIZATION COMPLETE")

    print(
        "Manifest:",
        manifest,
    )


def main():
    args = parse_args()

    config = load_experiment_config(args.config)

    candidate_specs = parse_layer_specs(args.candidate_layer_spec)

    selected_specs = parse_selected_layers(args.selected_layer_spec)

    validate_specs(
        config,
        candidate_specs,
        selected_specs,
    )

    print("=" * 76)
    print("CAA EXPERIMENT")
    print("=" * 76)

    print(
        "Phase:",
        args.phase,
    )

    print(
        "Model:",
        config.model_label,
    )

    print(
        "HF model:",
        config.model_name,
    )

    print(
        "Behaviors:",
        config.behaviors,
    )

    print(
        "Prompt family:",
        config.prompt_family,
    )

    print("Normalization: shared mean per-layer vector norm")

    print("Calibration: residual-relative eta")

    print(
        "Vector root:",
        config.vector_root,
    )

    print(
        "Results root:",
        config.results_root,
    )

    print(
        "Device:",
        args.device,
    )

    print()

    model, tokenizer = load_model(
        model_name=config.model_name,
        device=args.device,
        dtype=config.torch_dtype,
    )

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=config.model_name,
        prompt_family=config.prompt_family,
    )

    print(
        "Resolved prompt family:",
        adapter.family,
    )

    if args.phase in {
        "analyze",
        "full",
    }:
        run_analysis(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            config=config,
            extract_mode=args.extract_mode,
            candidate_specs=candidate_specs,
            reuse_analysis=args.reuse_analysis,
            device=args.device,
        )

    if args.phase in {
        "finalize",
        "full",
    }:
        run_finalize(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            config=config,
            selected_specs=selected_specs,
            skip_mmlu_items=args.skip_mmlu_items,
            device=args.device,
        )


if __name__ == "__main__":
    main()
