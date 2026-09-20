"""Prepare deterministic vector, validation, and test splits for behavior data.

Sampling is stratified by the designated matching answer and uses fixed seeds
so downstream vector extraction and evaluation are reproducible.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

BEHAVIORS = {
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "hallucination": "hallucination",
    "corrigibility": "corrigible-neutral-HHH",
    "survival_instinct": "survival-instinct",
    "myopic_reward": "myopic-reward",
    "ai_coordination": "coordinate-other-ais",
}

DEFAULT_SOURCE_ROOT = Path("data/source/caa")
DEFAULT_OUTPUT_ROOT = Path("data")

DEFAULT_SEED = 42
DEFAULT_MAX_POOL_SIZE = 1500

ANSWER_PATTERN = re.compile(r"^\([A-Z]\)$")

SPLIT_NAMES = ("vector", "validation", "test")
SPLIT_FRACTIONS = {
    "vector": 0.60,
    "validation": 0.20,
    "test": 0.20,
}


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def stable_seed(base_seed: int, behavior: str, purpose: str) -> int:
    payload = f"{base_seed}:{behavior}:{purpose}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def normalize_example(item, behavior: str, index: int):
    required = {
        "question",
        "answer_matching_behavior",
        "answer_not_matching_behavior",
    }

    missing = required - item.keys()

    if missing:
        raise ValueError(
            f"{behavior}: example {index} missing fields: {sorted(missing)}"
        )

    normalized = dict(item)

    matching = str(item["answer_matching_behavior"]).strip()
    nonmatching = str(item["answer_not_matching_behavior"]).strip()

    if not ANSWER_PATTERN.fullmatch(matching):
        raise ValueError(
            f"{behavior}: example {index} has invalid matching answer {matching!r}"
        )

    if not ANSWER_PATTERN.fullmatch(nonmatching):
        raise ValueError(
            f"{behavior}: example {index} has invalid non-matching answer "
            f"{nonmatching!r}"
        )

    if matching == nonmatching:
        raise ValueError(
            f"{behavior}: example {index} has identical behavior answers {matching!r}"
        )

    normalized["answer_matching_behavior"] = matching
    normalized["answer_not_matching_behavior"] = nonmatching

    return normalized


def deduplicate_by_question(data):
    seen = set()
    unique = []
    duplicates = []

    for item in data:
        question = item["question"]

        if question in seen:
            duplicates.append(question)
            continue

        seen.add(question)
        unique.append(item)

    return unique, duplicates


def proportional_allocation(group_sizes, target):
    """
    Allocate exactly `target` examples across groups approximately
    proportional to group size using the largest-remainder method.
    """
    total = sum(group_sizes.values())

    if target > total:
        raise ValueError("Allocation target exceeds available examples.")

    if target == total:
        return dict(group_sizes)

    raw = {key: target * size / total for key, size in group_sizes.items()}

    allocation = {key: int(raw[key]) for key in group_sizes}

    remaining = target - sum(allocation.values())

    order = sorted(
        group_sizes,
        key=lambda key: (
            -(raw[key] - allocation[key]),
            str(key),
        ),
    )

    for key in order:
        if remaining == 0:
            break

        if allocation[key] < group_sizes[key]:
            allocation[key] += 1
            remaining -= 1

    if remaining != 0:
        raise RuntimeError("Could not complete proportional allocation.")

    return allocation


def stratified_cap(data, max_pool_size, rng):
    """
    Cap the study pool while preserving the distribution of
    answer_matching_behavior.
    """
    if len(data) <= max_pool_size:
        result = list(data)
        rng.shuffle(result)
        return result

    groups = defaultdict(list)

    for item in data:
        groups[item["answer_matching_behavior"]].append(item)

    for items in groups.values():
        rng.shuffle(items)

    allocation = proportional_allocation(
        {key: len(items) for key, items in groups.items()},
        max_pool_size,
    )

    result = []

    for key in sorted(groups):
        result.extend(groups[key][: allocation[key]])

    rng.shuffle(result)

    return result


def global_split_sizes(n):
    raw = {split: SPLIT_FRACTIONS[split] * n for split in SPLIT_NAMES}

    sizes = {split: int(raw[split]) for split in SPLIT_NAMES}

    remaining = n - sum(sizes.values())

    order = sorted(
        SPLIT_NAMES,
        key=lambda split: (
            -(raw[split] - sizes[split]),
            SPLIT_NAMES.index(split),
        ),
    )

    for split in order[:remaining]:
        sizes[split] += 1

    assert sum(sizes.values()) == n

    return sizes


def stratified_split(data, rng):
    """
    Split data into exact global 60/20/20 sizes while preserving the
    answer_matching_behavior distribution as closely as possible.
    """
    groups = defaultdict(list)

    for item in data:
        groups[item["answer_matching_behavior"]].append(item)

    for items in groups.values():
        rng.shuffle(items)

    total = len(data)
    target_sizes = global_split_sizes(total)

    # Ideal allocation of every matching-answer group across each split.
    raw = {}
    allocation = {}

    for key, items in groups.items():
        raw[key] = {
            split: len(items) * target_sizes[split] / total for split in SPLIT_NAMES
        }

        allocation[key] = {split: int(raw[key][split]) for split in SPLIT_NAMES}

    row_remaining = {
        key: len(groups[key]) - sum(allocation[key].values()) for key in groups
    }

    col_remaining = {
        split: target_sizes[split] - sum(allocation[key][split] for key in groups)
        for split in SPLIT_NAMES
    }

    # Distribute the remaining examples according to largest fractional
    # remainders while respecting both row and column totals.
    while sum(row_remaining.values()) > 0:
        candidates = []

        for key in groups:
            if row_remaining[key] <= 0:
                continue

            for split in SPLIT_NAMES:
                if col_remaining[split] <= 0:
                    continue

                fractional = raw[key][split] - allocation[key][split]

                candidates.append(
                    (
                        fractional,
                        str(key),
                        split,
                        key,
                    )
                )

        if not candidates:
            raise RuntimeError("Could not complete stratified split allocation.")

        _, _, split, key = max(
            candidates,
            key=lambda x: (
                x[0],
                -SPLIT_NAMES.index(x[2]),
                x[1],
            ),
        )

        allocation[key][split] += 1
        row_remaining[key] -= 1
        col_remaining[split] -= 1

    if any(value != 0 for value in col_remaining.values()):
        raise RuntimeError(f"Split allocation did not meet targets: {col_remaining}")

    splits = {split: [] for split in SPLIT_NAMES}

    for key in sorted(groups):
        items = groups[key]
        start = 0

        for split in SPLIT_NAMES:
            count = allocation[key][split]
            end = start + count

            splits[split].extend(items[start:end])
            start = end

        assert start == len(items)

    # Avoid retaining answer-label grouping inside each written file.
    for split in SPLIT_NAMES:
        rng.shuffle(splits[split])

    assert {split: len(items) for split, items in splits.items()} == target_sizes

    return splits


def answer_distribution(data):
    return dict(
        sorted(Counter(item["answer_matching_behavior"] for item in data).items())
    )


def prepare_behavior(
    behavior,
    source_dir_name,
    source_root,
    output_root,
    base_seed,
    max_pool_size,
):
    source_path = source_root / source_dir_name / "dataset.json"

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source dataset for {behavior}: {source_path}")

    raw_data = load_json(source_path)

    if not isinstance(raw_data, list):
        raise ValueError(f"{source_path} must contain a JSON list.")

    normalized = [
        normalize_example(item, behavior, i) for i, item in enumerate(raw_data)
    ]

    unique, duplicates = deduplicate_by_question(normalized)

    cap_rng = random.Random(stable_seed(base_seed, behavior, "cap"))

    split_rng = random.Random(stable_seed(base_seed, behavior, "split"))

    study_pool = stratified_cap(
        unique,
        max_pool_size=max_pool_size,
        rng=cap_rng,
    )

    splits = stratified_split(
        study_pool,
        rng=split_rng,
    )

    questions = {
        split: {item["question"] for item in items} for split, items in splits.items()
    }

    assert questions["vector"].isdisjoint(questions["validation"])
    assert questions["vector"].isdisjoint(questions["test"])
    assert questions["validation"].isdisjoint(questions["test"])

    output_dir = output_root / behavior
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {}

    for split_name, items in splits.items():
        path = output_dir / f"{split_name}.json"
        write_json(path, items)
        split_paths[split_name] = path

    manifest = {
        "behavior": behavior,
        "source_directory": source_dir_name,
        "source_file": str(source_path),
        "source_sha256": file_sha256(source_path),
        "seed": base_seed,
        "max_pool_size": max_pool_size,
        "stratification": "answer_matching_behavior",
        "original_count": len(raw_data),
        "unique_count": len(unique),
        "duplicates_removed": len(duplicates),
        "study_pool_count": len(study_pool),
        "split_ratio": SPLIT_FRACTIONS,
        "study_pool_matching_distribution": answer_distribution(study_pool),
        "split_counts": {split: len(items) for split, items in splits.items()},
        "split_matching_distributions": {
            split: answer_distribution(items) for split, items in splits.items()
        },
        "output_sha256": {
            split: file_sha256(path) for split, path in split_paths.items()
        },
    }

    write_json(
        output_dir / "manifest.json",
        manifest,
    )

    return manifest


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Prepare deterministic behavior datasets."
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--max-pool-size",
        type=int,
        default=DEFAULT_MAX_POOL_SIZE,
    )

    args = parser.parse_args()

    print("=" * 76)
    print("BEHAVIOR DATA PREPARATION")
    print("=" * 76)
    print(f"Source root:   {args.source_root}")
    print(f"Output root:   {args.output_root}")
    print(f"Seed:          {args.seed}")
    print(f"Max pool size: {args.max_pool_size}")
    print("Stratify by:   answer_matching_behavior")
    print()

    manifests = []

    for behavior, source_dir_name in BEHAVIORS.items():
        manifest = prepare_behavior(
            behavior=behavior,
            source_dir_name=source_dir_name,
            source_root=args.source_root,
            output_root=args.output_root,
            base_seed=args.seed,
            max_pool_size=args.max_pool_size,
        )

        manifests.append(manifest)

        counts = manifest["split_counts"]

        print(
            f"{behavior:18s} "
            f"raw={manifest['original_count']:5d}  "
            f"unique={manifest['unique_count']:5d}  "
            f"pool={manifest['study_pool_count']:4d}  "
            f"vector={counts['vector']:4d}  "
            f"val={counts['validation']:4d}  "
            f"test={counts['test']:4d}"
        )

    global_manifest = {
        "seed": args.seed,
        "max_pool_size": args.max_pool_size,
        "stratification": "answer_matching_behavior",
        "behaviors": [manifest["behavior"] for manifest in manifests],
        "datasets": manifests,
    }

    write_json(
        args.output_root / "behavior_data_manifest.json",
        global_manifest,
    )

    print()
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
