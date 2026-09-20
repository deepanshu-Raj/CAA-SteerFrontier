"""Prepare domain-specific sycophancy vectors for every configured model.

Models with complete NLP and political vector artifacts are skipped during
resume runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
        raise RuntimeError(f"Missing configs for: {missing}")

    return configs


def domain_complete(
    model_slug: str,
    domain: str,
):
    root = Path("vectors") / model_slug / "ood" / "sycophancy" / domain

    return all(
        path.is_file()
        for path in [
            root / "raw_vectors.pt",
            root / "normalized_vectors.pt",
            root / "metadata.json",
        ]
    )


def model_complete(
    model_slug: str,
):
    return all(
        domain_complete(
            model_slug,
            domain,
        )
        for domain in [
            "nlp",
            "political",
        ]
    )


# =============================================================================
# Main
# =============================================================================


def main():
    configs = discover_configs()

    print("=" * 92)
    print("ALL-MODEL OOD SYCOPHANCY VECTOR EXTRACTION")
    print("=" * 92)

    for index, model_slug in enumerate(
        MODEL_ORDER,
        start=1,
    ):
        print()
        print("=" * 92)
        print(f"[{index}/{len(MODEL_ORDER)}] {model_slug}")
        print("=" * 92)

        if model_complete(model_slug):
            print("Both domain vectors already complete; skipping model.")
            continue

        config_path = configs[model_slug]

        command = [
            sys.executable,
            "-u",
            "scripts/prepare_ood_sycophancy_vectors.py",
            "--config",
            str(config_path),
        ]

        print("Running:")
        print(" ".join(command))
        print()

        subprocess.run(
            command,
            check=True,
        )

        if not model_complete(model_slug):
            raise RuntimeError(
                f"{model_slug}: vector extraction returned without complete outputs."
            )

        print()
        print(f"{model_slug}: COMPLETE")

    print()
    print("=" * 92)
    print("ALL OOD SYCOPHANCY DOMAIN VECTORS COMPLETE")
    print("=" * 92)

    for model_slug in MODEL_ORDER:
        status = "COMPLETE" if model_complete(model_slug) else "INCOMPLETE"

        print(f"{model_slug:28s} {status}")


if __name__ == "__main__":
    main()
