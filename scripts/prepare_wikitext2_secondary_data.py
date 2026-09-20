"""Prepare the frozen WikiText-2 corpus prefix for perplexity evaluation.

Tokenizer audits verify that every model can score the same fixed number of
target tokens under the shared windowing protocol.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from steerfrontier.config import load_experiment_config

# =============================================================================
# Configuration
# =============================================================================

MODEL_ORDER = [
    "llama2-7b-chat",
    "llama3.1-8b-instruct",
    "mistral-7b-instruct-v0.3",
    "qwen2.5-7b-instruct",
]

OUTPUT_ROOT = Path("data/secondary")

TEXT_PATH = OUTPUT_ROOT / "wikitext2_test.txt"

MANIFEST_PATH = OUTPUT_ROOT / "wikitext2_test_manifest.json"

AUDIT_PATH = OUTPUT_ROOT / "wikitext2_token_audit.csv"

SCORED_TOKENS = 32_768
TARGETS_PER_WINDOW = 256
INPUT_TOKENS_PER_WINDOW = 257
NUM_WINDOWS = SCORED_TOKENS // TARGETS_PER_WINDOW

# We need exactly one initial context token plus
# 32,768 scored next-token targets.
MIN_REQUIRED_TOKENS = SCORED_TOKENS + 1


def sha256_bytes(
    payload: bytes,
):
    return hashlib.sha256(payload).hexdigest()


def discover_configs():
    configs = {}

    for path in sorted(Path("configs").glob("*.yaml")):
        try:
            config = load_experiment_config(str(path))
        except Exception:
            continue

        if config.model_slug in MODEL_ORDER:
            configs[config.model_slug] = path

    missing = [slug for slug in MODEL_ORDER if slug not in configs]

    if missing:
        raise RuntimeError(f"Missing configs: {missing}")

    return configs


def build_frozen_text():
    print("Loading WikiText-2 raw test split...")

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    texts = [str(row["text"]) for row in dataset]

    # Preserve dataset order exactly.
    #
    # WikiText raw already contains empty rows / headings.
    # Joining rows with newlines retains document structure
    # without introducing model-specific formatting.
    frozen_text = "\n".join(texts)

    if not frozen_text.strip():
        raise RuntimeError("WikiText-2 test corpus is empty.")

    return (
        frozen_text,
        len(dataset),
    )


# =============================================================================
# Main
# =============================================================================


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        frozen_text,
        num_dataset_rows,
    ) = build_frozen_text()

    TEXT_PATH.write_text(
        frozen_text,
        encoding="utf-8",
    )

    encoded = frozen_text.encode("utf-8")

    text_hash = sha256_bytes(encoded)

    configs = discover_configs()

    audit_rows = []

    print()
    print("=" * 88)
    print("WIKITEXT-2 TOKEN AUDIT")
    print("=" * 88)

    for slug in MODEL_ORDER:
        config = load_experiment_config(str(configs[slug]))

        print()
        print(f"Loading tokenizer: {config.model_label}")

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        token_ids = tokenizer.encode(
            frozen_text,
            add_special_tokens=False,
        )

        num_tokens = int(len(token_ids))

        enough = num_tokens >= MIN_REQUIRED_TOKENS

        audit_rows.append(
            {
                "model_slug": (config.model_slug),
                "model_label": (config.model_label),
                "tokenizer_name": (config.model_name),
                "total_test_tokens": (num_tokens),
                "required_tokens": (MIN_REQUIRED_TOKENS),
                "scored_tokens": (SCORED_TOKENS),
                "num_windows": (NUM_WINDOWS),
                "input_tokens_per_window": (INPUT_TOKENS_PER_WINDOW),
                "targets_per_window": (TARGETS_PER_WINDOW),
                "enough_tokens": bool(enough),
            }
        )

        print(f"  total tokens:   {num_tokens:,}")
        print(f"  required:       {MIN_REQUIRED_TOKENS:,}")
        print(f"  scored:         {SCORED_TOKENS:,}")
        print(f"  windows:        {NUM_WINDOWS}")
        print(f"  enough corpus:  {enough}")

        if not enough:
            raise RuntimeError(
                f"{config.model_slug} has only "
                f"{num_tokens} WikiText-2 tokens; "
                f"need at least "
                f"{MIN_REQUIRED_TOKENS}."
            )

    audit = pd.DataFrame(audit_rows)

    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )

    manifest = {
        "dataset": ("Salesforce/wikitext"),
        "configuration": ("wikitext-2-raw-v1"),
        "split": ("test"),
        "num_dataset_rows": int(num_dataset_rows),
        "corpus_construction": (
            "Dataset text rows joined in original order with a single newline."
        ),
        "text_path": str(TEXT_PATH),
        "text_sha256": (text_hash),
        "utf8_bytes": int(len(encoded)),
        "evaluation_protocol": {
            "scored_tokens_per_model": (SCORED_TOKENS),
            "targets_per_window": (TARGETS_PER_WINDOW),
            "input_tokens_per_window": (INPUT_TOKENS_PER_WINDOW),
            "num_windows": (NUM_WINDOWS),
            "window_stride": (TARGETS_PER_WINDOW),
            "window_overlap_tokens": 1,
            "tokenization": ("Model-specific tokenizer, add_special_tokens=False."),
            "primary_metric": (
                "Delta mean next-token NLL relative to the unsteered baseline."
            ),
            "secondary_metric": ("Perplexity ratio exp(delta_NLL)."),
        },
        "interpretation_note": (
            "Absolute perplexities should not be compared "
            "across models with different tokenizers. "
            "The experiment uses within-model changes "
            "under frozen steering interventions."
        ),
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 88)
    print("FROZEN WIKITEXT-2 DATA")
    print("=" * 88)
    print(f"Corpus:   {TEXT_PATH}")
    print(f"SHA256:   {text_hash}")
    print(f"Audit:    {AUDIT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")

    print()
    print(
        audit[
            [
                "model_slug",
                "total_test_tokens",
                "required_tokens",
                "scored_tokens",
                "num_windows",
                "enough_tokens",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
