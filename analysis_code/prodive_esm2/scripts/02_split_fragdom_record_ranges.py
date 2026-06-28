#!/usr/bin/env python3
"""Compute length stats and percentile record indices for frag_dom inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from fragdom_utils import iter_fragdom_records


def _percentile_indices(lengths: List[int], percentiles: List[float]) -> List[Tuple[float, int]]:
    total = sum(lengths)
    if total <= 0:
        raise ValueError("Total sequence length is zero")
    thresholds = [total * p for p in percentiles]
    out: List[Tuple[float, int]] = []
    running = 0
    idx = 0
    for t in thresholds:
        while idx < len(lengths) and running < t:
            running += lengths[idx]
            idx += 1
        out.append((t, max(1, idx)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize frag_dom sequence lengths")
    p.add_argument("--input", type=str, required=True, help="Path to frag_dom records (txt or .gz)")
    p.add_argument("--output", type=str, default=None, help="Optional JSONL output for per-record lengths")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input file does not exist: {input_path}")

    lengths: List[int] = []
    records: List[Tuple[str, int]] = []
    for header, _, seq in iter_fragdom_records(input_path):
        length = len(seq)
        lengths.append(length)
        records.append((header, length))

    if not lengths:
        raise ValueError("No records found")

    total = sum(lengths)
    percentiles = [0.25, 0.50, 0.75, 1.0]
    indices = _percentile_indices(lengths, percentiles)

    print(f"records: {len(lengths)}")
    print(f"total_aa: {total}")
    for p, idx in zip(percentiles, indices):
        pct = int(p * 100)
        print(f"percentile_{pct}_index: {idx[1]}")
    print("record_ranges:")
    bounds = [1] + [i for _, i in indices]
    for lo, hi, pct in zip(bounds[:-1], bounds[1:], [25, 50, 75, 100]):
        print(f"  1-{pct}%: {lo}:{hi}")

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w") as f:
            for idx, (header, length) in enumerate(records, start=1):
                f.write(json.dumps({"record_index": idx, "header": header, "length": length}) + "\n")


if __name__ == "__main__":
    main()
