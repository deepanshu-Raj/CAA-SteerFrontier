"""Re-run the frozen persistent open-ended nonterminations at a larger token cap.

Repair attempts are an audit of generation stability only; they do not replace
the original production outputs unless the explicit replacement checks pass.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from run_gsm8k_secondary import generate_batch

from steerfrontier.config import load_experiment_config
from steerfrontier.models import load_model
from steerfrontier.prompts import get_prompt_adapter
from steerfrontier.workflow import load_normalized_vectors

# =============================================================================
# Frozen repair protocol
# =============================================================================

MODEL_SLUG = "llama3.1-8b-instruct"
CONFIG_PATH = Path("configs/llama31_8b_instruct.yaml")

ORIGINAL_MAX_NEW_TOKENS = 2056
REPAIR_MAX_NEW_TOKENS = 4096
BATCH_SIZE = 2
EPSILON = 0.03

TARGETS = {
    (
        "hallucination",
        "positive",
    ): [
        "hallucination_00",
        "hallucination_02",
        "hallucination_07",
        "hallucination_17",
        "hallucination_18",
    ],
    (
        "refusal",
        "negative",
    ): [
        "refusal_02",
        "refusal_08",
        "refusal_10",
        "refusal_11",
        "refusal_12",
        "refusal_17",
        "refusal_23",
    ],
}

RESULT_ROOT = Path("results") / MODEL_SLUG / "open_ended_generation"

REPAIR_ROOT = RESULT_ROOT / "truncation_repair_4096"

OPEN_ENDED_DATA_ROOT = Path("data/open_ended")


# =============================================================================
# Helpers
# =============================================================================


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_json(
    path: Path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")


def write_csv(
    frame: pd.DataFrame,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    frame.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(path)


def parse_bool(value):
    if isinstance(
        value,
        bool,
    ):
        return value

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
    }:
        return True

    if text in {
        "false",
        "0",
    }:
        return False

    raise ValueError(f"Cannot parse boolean {value!r}.")


# =============================================================================
# Frozen operating point
# =============================================================================


def load_point(
    config,
    *,
    behavior: str,
    direction: str,
):
    path = config.results_root / "final_evaluation" / "frontier_test_results.csv"

    frame = pd.read_csv(path)

    rows = frame[
        (frame["behavior"] == behavior)
        & (frame["direction"] == direction)
        & np.isclose(
            frame["epsilon"].astype(float),
            EPSILON,
            atol=1e-12,
            rtol=0.0,
        )
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one frozen point for {behavior}/{direction}; found {len(rows)}."
        )

    row = rows.iloc[0]

    return {
        "layer": int(row["layer"]),
        "alpha": float(row["alpha_star"]),
        "eta": float(row["eta_star"]),
    }


# =============================================================================
# Prompt loading
# =============================================================================


def load_prompt_mapping(
    behavior: str,
):
    path = OPEN_ENDED_DATA_ROOT / f"{behavior}.json"

    rows = load_json(path)

    return {str(row["prompt_id"]): row for row in rows}


# =============================================================================
# Generation
# =============================================================================


def generate_repairs(
    *,
    model,
    tokenizer,
    adapter,
    config,
    vectors,
    device: str,
):
    repair_rows = []

    for (
        behavior,
        condition,
    ), prompt_ids in TARGETS.items():
        point = load_point(
            config,
            behavior=behavior,
            direction=condition,
        )

        layer = int(point["layer"])

        alpha = float(point["alpha"])

        eta = float(point["eta"])

        vector = vectors[behavior][layer]

        prompt_mapping = load_prompt_mapping(behavior)

        missing = [
            prompt_id for prompt_id in prompt_ids if prompt_id not in prompt_mapping
        ]

        if missing:
            raise RuntimeError(f"{behavior}: missing prompt IDs {missing}.")

        selected = [prompt_mapping[prompt_id] for prompt_id in prompt_ids]

        print()
        print("=" * 100)
        print(f"{behavior} / {condition}")
        print("=" * 100)

        print(f"Layer:   {layer}")
        print(f"alpha:   {alpha:+.6f}")
        print(f"eta:     {eta:+.6f}")
        print(f"Targets: {len(selected)}")

        for start in range(
            0,
            len(selected),
            BATCH_SIZE,
        ):
            batch = selected[start : start + BATCH_SIZE]

            prompts = [row["prompt"] for row in batch]

            outputs = generate_batch(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                prompts=prompts,
                device=device,
                max_new_tokens=(REPAIR_MAX_NEW_TOKENS),
                layer=layer,
                vector=vector,
                alpha=alpha,
            )

            if len(outputs) != len(batch):
                raise RuntimeError("Output count mismatch.")

            for source, output in zip(
                batch,
                outputs,
            ):
                repair_rows.append(
                    {
                        "model_slug": (MODEL_SLUG),
                        "behavior": (behavior),
                        "condition": (condition),
                        "direction": (condition),
                        "prompt_id": (source["prompt_id"]),
                        "domain": (source["domain"]),
                        "layer": (layer),
                        "alpha": (alpha),
                        "eta": (eta),
                        "prompt": (source["prompt"]),
                        "response": (output["response"]),
                        "prompt_token_count": int(output["prompt_token_count"]),
                        "new_token_count": int(output["new_token_count"]),
                        "ended_with_eos": bool(output["ended_with_eos"]),
                        "hit_max_tokens": bool(output["hit_max_tokens"]),
                    }
                )

            print(
                f"  generated {min(start + BATCH_SIZE, len(selected))}/{len(selected)}"
            )

    repairs = pd.DataFrame(repair_rows)

    expected = sum(len(ids) for ids in TARGETS.values())

    if len(repairs) != expected:
        raise RuntimeError(f"Expected {expected} repairs, found {len(repairs)}.")

    return repairs


# =============================================================================
# Production replacement
# =============================================================================


def perform_replacement(
    repairs: pd.DataFrame,
):
    REPAIR_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    attempt_path = REPAIR_ROOT / "repair_attempts.csv"

    write_csv(
        repairs,
        attempt_path,
    )

    # ------------------------------------------------------------
    # Do NOT modify production if the larger cap still truncates.
    # ------------------------------------------------------------

    repair_truncations = int(repairs["hit_max_tokens"].map(parse_bool).sum())

    if repair_truncations:
        raise RuntimeError(
            f"{repair_truncations} repaired "
            "generations still hit the "
            f"{REPAIR_MAX_NEW_TOKENS}-token cap. "
            "Production files were NOT modified."
        )

    if (~repairs["ended_with_eos"].map(parse_bool)).any():
        raise RuntimeError(
            "At least one repair did not end "
            "with EOS. Production files were "
            "NOT modified."
        )

    # ------------------------------------------------------------
    # Load current production combined file.
    # ------------------------------------------------------------

    generations_path = RESULT_ROOT / "generations.csv"

    summary_path = RESULT_ROOT / "generation_summary.csv"

    production = pd.read_csv(generations_path)

    # ------------------------------------------------------------
    # Verify these are exactly the currently truncated rows.
    # ------------------------------------------------------------

    original_rows = []

    for repair in repairs.to_dict(orient="records"):
        mask = (
            (production["behavior"] == repair["behavior"])
            & (production["condition"] == repair["condition"])
            & (production["prompt_id"].astype(str) == str(repair["prompt_id"]))
        )

        indices = production.index[mask].tolist()

        if len(indices) != 1:
            raise RuntimeError(
                f"Expected one production row "
                f"for {repair['behavior']}/"
                f"{repair['condition']}/"
                f"{repair['prompt_id']}; "
                f"found {len(indices)}."
            )

        index = indices[0]

        original = production.loc[index].to_dict()

        if not parse_bool(original["hit_max_tokens"]):
            raise RuntimeError(
                f"{repair['prompt_id']} was "
                "not marked truncated in the "
                "production file."
            )

        if int(original["new_token_count"]) != ORIGINAL_MAX_NEW_TOKENS:
            raise RuntimeError(
                f"{repair['prompt_id']} did "
                "not stop at the expected "
                f"{ORIGINAL_MAX_NEW_TOKENS} "
                "token cap."
            )

        original_rows.append(original)

    # Preserve the original truncated rows.
    write_csv(
        pd.DataFrame(original_rows),
        REPAIR_ROOT / "original_truncated_rows.csv",
    )

    # ------------------------------------------------------------
    # Back up affected production artifacts once.
    # ------------------------------------------------------------

    backup_root = REPAIR_ROOT / "pre_repair_backup"

    backup_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    files_to_backup = [
        generations_path,
        summary_path,
    ]

    for (
        behavior,
        condition,
    ) in TARGETS:
        files_to_backup.append(
            RESULT_ROOT / "conditions" / f"{behavior}__{condition}.csv"
        )

    for path in files_to_backup:
        if not path.is_file():
            raise FileNotFoundError(path)

        destination = backup_root / path.name

        if not destination.exists():
            shutil.copy2(
                path,
                destination,
            )

    # ------------------------------------------------------------
    # Replace target rows in each condition file.
    # ------------------------------------------------------------

    for (
        behavior,
        condition,
    ), prompt_ids in TARGETS.items():
        condition_path = RESULT_ROOT / "conditions" / f"{behavior}__{condition}.csv"

        condition_frame = pd.read_csv(condition_path)

        subset = repairs[
            (repairs["behavior"] == behavior) & (repairs["condition"] == condition)
        ]

        for repair in subset.to_dict(orient="records"):
            mask = condition_frame["prompt_id"].astype(str) == str(repair["prompt_id"])

            indices = condition_frame.index[mask].tolist()

            if len(indices) != 1:
                raise RuntimeError(f"{condition_path}: expected one target row.")

            index = indices[0]

            for column, value in repair.items():
                if column in condition_frame.columns:
                    condition_frame.loc[
                        index,
                        column,
                    ] = value

        write_csv(
            condition_frame,
            condition_path,
        )

    # ------------------------------------------------------------
    # Rebuild generations.csv from all nine condition files.
    # ------------------------------------------------------------

    condition_frames = []

    for behavior in [
        "sycophancy",
        "refusal",
        "hallucination",
    ]:
        for condition in [
            "baseline",
            "negative",
            "positive",
        ]:
            path = RESULT_ROOT / "conditions" / f"{behavior}__{condition}.csv"

            frame = pd.read_csv(path)

            if len(frame) != 24:
                raise RuntimeError(f"{path}: expected 24 rows, found {len(frame)}.")

            condition_frames.append(frame)

    combined = pd.concat(
        condition_frames,
        ignore_index=True,
    )

    if len(combined) != 216:
        raise RuntimeError(f"Expected 216 rows, found {len(combined)}.")

    if (
        combined[
            [
                "behavior",
                "condition",
                "prompt_id",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError("Duplicate production generation keys.")

    remaining_truncations = int(combined["hit_max_tokens"].map(parse_bool).sum())

    if remaining_truncations:
        raise RuntimeError(
            "Production rebuild would still "
            f"contain {remaining_truncations} "
            "truncations."
        )

    combined = combined.sort_values(
        [
            "behavior",
            "prompt_id",
            "condition",
        ]
    ).reset_index(drop=True)

    write_csv(
        combined,
        generations_path,
    )

    # ------------------------------------------------------------
    # Rebuild mechanical summary.
    # ------------------------------------------------------------

    combined["ended_with_eos"] = combined["ended_with_eos"].map(parse_bool)

    combined["hit_max_tokens"] = combined["hit_max_tokens"].map(parse_bool)

    summary = combined.groupby(
        [
            "behavior",
            "condition",
        ],
        as_index=False,
    ).agg(
        num_prompts=(
            "prompt_id",
            "count",
        ),
        mean_prompt_tokens=(
            "prompt_token_count",
            "mean",
        ),
        mean_new_tokens=(
            "new_token_count",
            "mean",
        ),
        eos_rate=(
            "ended_with_eos",
            "mean",
        ),
        truncation_rate=(
            "hit_max_tokens",
            "mean",
        ),
    )

    write_csv(
        summary,
        summary_path,
    )

    # ------------------------------------------------------------
    # Repair manifest
    # ------------------------------------------------------------

    manifest = {
        "protocol": ("open_ended_truncation_repair_v1"),
        "model_slug": (MODEL_SLUG),
        "reason": (
            "12 production generations reached "
            "the frozen 2056-token generation cap. "
            "They were regenerated without any "
            "prompt, vector, layer, coefficient, "
            "or decoding-policy change, using only "
            "a larger mechanical generation cap."
        ),
        "original_max_new_tokens": int(ORIGINAL_MAX_NEW_TOKENS),
        "repair_max_new_tokens": int(REPAIR_MAX_NEW_TOKENS),
        "num_repaired": int(len(repairs)),
        "targets": {
            f"{behavior}__{condition}": (prompt_ids)
            for (
                behavior,
                condition,
            ), prompt_ids in TARGETS.items()
        },
        "behavioral_outputs_inspected_before_repair": (False),
        "production_rows_replaced_only_after_all_repairs_passed": (True),
        "remaining_truncations": 0,
        "original_rows_path": str(REPAIR_ROOT / "original_truncated_rows.csv"),
        "repair_attempts_path": str(attempt_path),
        "backup_root": str(backup_root),
    }

    write_json(
        REPAIR_ROOT / "manifest.json",
        manifest,
    )

    print()
    print("=" * 100)
    print("TRUNCATION REPAIR COMPLETE")
    print("=" * 100)

    print(f"Repaired:               {len(repairs)}")
    print(f"Repair token cap:       {REPAIR_MAX_NEW_TOKENS}")
    print(f"Remaining truncations:  {remaining_truncations}")

    print()
    print(summary.to_string(index=False))

    print()
    print(f"Audit artifacts: {REPAIR_ROOT}")


# =============================================================================
# Main
# =============================================================================


def main():
    config = load_experiment_config(str(CONFIG_PATH))

    if config.model_slug != MODEL_SLUG:
        raise RuntimeError(
            f"Config resolves to {config.model_slug}, expected {MODEL_SLUG}."
        )

    print("=" * 100)
    print("OPEN-ENDED TRUNCATION REPAIR")
    print("=" * 100)

    print(f"Model:                  {MODEL_SLUG}")

    print(f"Original token cap:     {ORIGINAL_MAX_NEW_TOKENS}")

    print(f"Repair token cap:       {REPAIR_MAX_NEW_TOKENS}")

    print(f"Target generations:     {sum(len(v) for v in TARGETS.values())}")

    print()
    print("Loading model...")

    (
        model,
        tokenizer,
    ) = load_model(
        model_name=(config.model_name),
        device="cuda",
        dtype=(config.torch_dtype),
    )

    model.eval()

    adapter = get_prompt_adapter(
        model=model,
        tokenizer=tokenizer,
        model_name=(config.model_name),
        prompt_family=(config.prompt_family),
    )

    vectors = load_normalized_vectors(config)

    repairs = generate_repairs(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        config=config,
        vectors=vectors,
        device="cuda",
    )

    print()
    print("=" * 100)
    print("REPAIR GENERATION AUDIT")
    print("=" * 100)

    print(
        repairs[
            [
                "behavior",
                "condition",
                "prompt_id",
                "new_token_count",
                "ended_with_eos",
                "hit_max_tokens",
            ]
        ].to_string(index=False)
    )

    perform_replacement(repairs)


if __name__ == "__main__":
    main()
