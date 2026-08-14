#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a fixed random Pfam family-pair list for fragment-length sensitivity analysis.

This script only generates sampled_pairs.csv. It does not run the C++ KL program.

Output CSV columns:
    sample_id,i,j,start_i,end_i,start_j,end_j

The start/end columns are kept for compatibility with downstream C++ pair-list
interfaces. Each row represents one Pfam family pair:
    start_i = i
    end_i   = i + 1
    start_j = j
    end_j   = j + 1

Example:
    python3 03_generate_fixed_pairlist.py \
        --mapping-file /path/to/pfam_mapping_seed_new.txt \
        --out-dir /path/to/fixed_pairlists \
        --sample-n 50000 \
        --seed 20260511 \
        --overwrite
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


DEFAULT_SAMPLE_N = 50_000
DEFAULT_RANDOM_SEED = 20260511


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate sampled_pairs.csv for fixed random Pfam family-pair sampling. "
            "This script does not run C++ KL calculations."
        )
    )

    parser.add_argument(
        "--mapping-file",
        type=Path,
        required=True,
        help=(
            "Pfam mapping file. Expected format: PF00001:0, one record per line."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for sampled_pairs.csv.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=DEFAULT_SAMPLE_N,
        help=f"Number of unique family pairs to sample. Default: {DEFAULT_SAMPLE_N}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed. Default: {DEFAULT_RANDOM_SEED}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite sampled_pairs.csv if it already exists.",
    )

    return parser.parse_args()


def load_family_ids(mapping_file: Path) -> List[int]:
    """
    Load numeric Pfam family IDs from a mapping file.

    Expected line format:
        PF00001:0
        PF00002:1

    Empty lines and malformed lines without ':' are skipped.
    """
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    ids: List[int] = []

    with mapping_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line or ":" not in line:
                continue

            _, idx_text = line.split(":", 1)
            idx_text = idx_text.strip()

            try:
                ids.append(int(idx_text))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric family ID at line {line_no}: {line!r}"
                ) from exc

    ids = sorted(set(ids))

    if len(ids) < 2:
        raise RuntimeError(
            f"At least two family IDs are required for pair sampling. "
            f"Loaded IDs: {len(ids)}"
        )

    return ids


def max_unique_unordered_pairs(n_ids: int) -> int:
    return n_ids * (n_ids - 1) // 2


def sample_unique_unordered_pairs(
    ids: Sequence[int],
    sample_n: int,
    seed: int,
) -> List[Tuple[int, int]]:
    """
    Sample unique unordered family pairs.

    The returned pairs always satisfy i < j, so (i, j) and (j, i) are not duplicated.
    """
    if sample_n <= 0:
        raise ValueError(f"--sample-n must be positive. Got: {sample_n}")

    max_pairs = max_unique_unordered_pairs(len(ids))
    if sample_n > max_pairs:
        raise ValueError(
            f"Requested sample_n={sample_n:,}, but only {max_pairs:,} unique "
            f"unordered pairs are possible from {len(ids):,} family IDs."
        )

    rng = random.Random(seed)
    pairs = set()

    while len(pairs) < sample_n:
        i = rng.choice(ids)
        j = rng.choice(ids)

        if i == j:
            continue

        if i > j:
            i, j = j, i

        pairs.add((i, j))

        if len(pairs) % 5_000 == 0:
            print(f"[INFO] sampled {len(pairs):,} / {sample_n:,}")

    return sorted(pairs)


def write_sampled_pairs_csv(
    pairs: Iterable[Tuple[int, int]],
    out_csv: Path,
    overwrite: bool = False,
) -> None:
    """
    Write sampled family pairs in the downstream-compatible pair-list format.
    """
    if out_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {out_csv}\n"
            f"Use --overwrite to replace it."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Use plain UTF-8 without BOM to avoid parser issues in downstream C++/shell tools.
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "i", "j", "start_i", "end_i", "start_j", "end_j"])

        for sample_id, (i, j) in enumerate(pairs, start=1):
            writer.writerow([
                sample_id,
                i,
                j,
                i,
                i + 1,
                j,
                j + 1,
            ])


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("Generate fixed random Pfam family-pair list")
    print(f"mapping_file: {args.mapping_file}")
    print(f"out_dir     : {args.out_dir}")
    print(f"sample_n    : {args.sample_n:,}")
    print(f"seed        : {args.seed}")
    print(f"overwrite   : {args.overwrite}")
    print("=" * 80)

    ids = load_family_ids(args.mapping_file)

    print(f"[INFO] loaded family IDs: {len(ids):,}")
    print(f"[INFO] ID range: {min(ids)} - {max(ids)}")
    print(f"[INFO] max unique unordered pairs: {max_unique_unordered_pairs(len(ids)):,}")

    pairs = sample_unique_unordered_pairs(
        ids=ids,
        sample_n=args.sample_n,
        seed=args.seed,
    )

    out_csv = args.out_dir / "sampled_pairs.csv"
    write_sampled_pairs_csv(
        pairs=pairs,
        out_csv=out_csv,
        overwrite=args.overwrite,
    )

    print("-" * 80)
    print(f"[DONE] sampled pair list written to: {out_csv}")
    print(f"[DONE] total sampled pairs: {len(pairs):,}")
    print("-" * 80)
    print("Next step:")
    print("  Use this sampled_pairs.csv as PAIR_LIST in run_pairlist_all_fragment_lengths.sh")


if __name__ == "__main__":
    main()
