#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw non-overlapping random samples from a large filtered path table.

This script is used for large-parameter-set structural-context validation. It
extracts a fixed number of rows from a large CSV by streaming chunks, so it does
not load the full table into memory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def count_data_rows(csv_path: Path) -> int:
    """Count data rows in a CSV file, excluding the header."""
    n = 0
    with csv_path.open("rb") as handle:
        for _ in handle:
            n += 1
    return max(0, n - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random row samples from a large filtered path CSV."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--total-data-rows", type=int, default=None,
                        help="Number of data rows excluding header. If omitted, rows are counted first.")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=300000)
    parser.add_argument("--allow-overlap", action="store_true",
                        help="Draw each sample independently. By default samples are non-overlapping.")
    parser.add_argument("--add-source-row-index", action="store_true")
    parser.add_argument("--no-shuffle-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = args.input_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = args.output_prefix or input_csv.stem
    total_rows = args.total_data_rows
    if total_rows is None:
        print("[INFO] Counting input rows because --total-data-rows was not provided.")
        total_rows = count_data_rows(input_csv)

    non_overlapping = not args.allow_overlap
    total_needed = args.num_samples * args.sample_size
    if non_overlapping and total_needed > total_rows:
        raise ValueError(
            f"Need {total_needed:,} non-overlapping rows, but only {total_rows:,} rows are available."
        )

    rng = np.random.default_rng(args.random_seed)

    print("=" * 100)
    print("Random sampling from filtered path table")
    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Total data rows: {total_rows:,}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Rows per sample: {args.sample_size:,}")
    print(f"Non-overlapping samples: {non_overlapping}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 100)

    selected_indices_by_sample = []
    if non_overlapping:
        selected_all = rng.choice(total_rows, size=total_needed, replace=False)
        rng.shuffle(selected_all)
        for sample_idx in range(args.num_samples):
            start = sample_idx * args.sample_size
            end = (sample_idx + 1) * args.sample_size
            selected_indices_by_sample.append(selected_all[start:end])
    else:
        for _ in range(args.num_samples):
            selected_indices_by_sample.append(
                rng.choice(total_rows, size=args.sample_size, replace=False)
            )

    all_indices = np.concatenate(selected_indices_by_sample)
    all_sample_ids = np.concatenate([
        np.full(len(indices), sample_idx, dtype=np.int32)
        for sample_idx, indices in enumerate(selected_indices_by_sample)
    ])

    order = np.argsort(all_indices)
    sorted_indices = all_indices[order]
    sorted_sample_ids = all_sample_ids[order]
    sampled_chunks_by_sample = [[] for _ in range(args.num_samples)]

    current_row_start = 0
    selected_count = 0
    reader = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        chunksize=args.chunk_size,
        low_memory=False,
    )

    for chunk in tqdm(reader, desc="Reading chunks", unit="chunk"):
        chunk_len = len(chunk)
        if chunk_len == 0:
            continue
        current_row_end = current_row_start + chunk_len
        left = np.searchsorted(sorted_indices, current_row_start, side="left")
        right = np.searchsorted(sorted_indices, current_row_end, side="left")
        if right > left:
            local_indices = sorted_indices[left:right] - current_row_start
            sample_ids = sorted_sample_ids[left:right]
            source_indices = sorted_indices[left:right]
            for sample_idx in range(args.num_samples):
                mask = sample_ids == sample_idx
                if not np.any(mask):
                    continue
                sampled = chunk.iloc[local_indices[mask]].copy()
                if args.add_source_row_index:
                    sampled.insert(0, "Source_Row_Index_0based", source_indices[mask])
                sampled_chunks_by_sample[sample_idx].append(sampled)
                selected_count += len(sampled)
        current_row_start = current_row_end
        if selected_count >= total_needed:
            break

    for sample_idx in range(args.num_samples):
        chunks = sampled_chunks_by_sample[sample_idx]
        if not chunks:
            raise RuntimeError(f"Sample {sample_idx + 1} has no rows.")
        out_df = pd.concat(chunks, ignore_index=True)
        if len(out_df) != args.sample_size:
            raise RuntimeError(
                f"Sample {sample_idx + 1}: expected {args.sample_size:,} rows, got {len(out_df):,}."
            )
        if not args.no_shuffle_output:
            out_df = out_df.sample(frac=1.0, random_state=args.random_seed + sample_idx).reset_index(drop=True)
        out_csv = output_dir / f"{output_prefix}_sample_{sample_idx + 1:02d}_{args.sample_size}.csv"
        out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[DONE] Sample {sample_idx + 1}: {out_csv} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
