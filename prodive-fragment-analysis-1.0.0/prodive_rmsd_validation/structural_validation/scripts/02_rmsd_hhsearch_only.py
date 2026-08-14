#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute RMSD for HHsearch-only tasks.

Input:
  hhsuite_only_tasks.csv

Rows processed:
  all rows

Segment boundaries used:
  a1/a2 and b1/b2 columns derived from HHsearch alignments

Output:
  RMSD results for HHsearch-only family pairs / segments.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from prodive_rmsd.structural_utils import (  # noqa: E402
    RMSDConfig,
    StructureContext,
    compute_rmsd_for_hmm_segments,
    count_csv_rows,
    flush_rows,
    init_pymol,
    print_failure_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute RMSD for HHsearch-only tasks.")
    p.add_argument("--task-csv", required=True, help="HHsearch-only task CSV.")
    p.add_argument("--pfam-dir", required=True, help="Merged Pfam runtime directory containing HHM/MSA/PDB/AF files.")
    p.add_argument("--output-csv", required=True, help="Output RMSD CSV.")
    p.add_argument("--chunksize", type=int, default=2000)
    p.add_argument("--flush-every", type=int, default=500)
    p.add_argument("--debug-first-n", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--max-search-depth", type=int, default=200)
    p.add_argument("--min-len-diff", type=int, default=5)
    p.add_argument("--rel-len-diff-ratio", type=float, default=0.2)
    p.add_argument("--min-mean-plddt", type=float, default=70.0)
    p.add_argument("--max-head-tail-pae", type=float, default=10.0)
    p.add_argument("--min-aligned-coverage", type=float, default=0.0,
                   help="Optional post-superposition coverage filter. 0 disables filtering.")
    return p.parse_args()


def required_columns_exist(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}. Current columns: {list(df.columns)}")


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.task_csv):
        raise FileNotFoundError(f"Task CSV does not exist: {args.task_csv}")
    if os.path.exists(args.output_csv) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_csv}. Use --overwrite to replace it.")
    if os.path.exists(args.output_csv) and args.overwrite:
        os.remove(args.output_csv)

    cfg = RMSDConfig(
        pfam_dir=args.pfam_dir,
        max_search_depth=args.max_search_depth,
        len_diff_mode="relative",
        min_len_diff=args.min_len_diff,
        rel_len_diff_ratio=args.rel_len_diff_ratio,
        min_mean_plddt=args.min_mean_plddt,
        max_head_tail_pae=args.max_head_tail_pae,
        min_aligned_coverage=args.min_aligned_coverage,
    )
    ctx = StructureContext(cfg)
    failure_log: Dict[str, int] = {}

    print("Step 0: Initialize PyMOL...")
    init_pymol(quiet=True)

    total_tasks = count_csv_rows(args.task_csv)
    print("Step 1: Compute RMSD for HHsearch-only tasks")
    print(f"[INFO] TASK_CSV           = {args.task_csv}")
    print(f"[INFO] PFAM_DIR           = {args.pfam_dir}")
    print(f"[INFO] OUTPUT             = {args.output_csv}")
    print(f"[INFO] MIN_LEN_DIFF       = {args.min_len_diff}")
    print(f"[INFO] REL_LEN_DIFF_RATIO = {args.rel_len_diff_ratio}")
    print(f"[INFO] Total tasks        = {total_tasks}")

    wrote_header = False
    buffer_rows: List[Dict[str, Any]] = []
    total_success = 0
    seen = 0

    pbar = tqdm(total=total_tasks, desc="HHsearch-only RMSD")
    try:
        for chunk in pd.read_csv(args.task_csv, chunksize=args.chunksize):
            required_columns_exist(chunk, ["Main_HMM", "Sub_HMM", "a1", "a2", "b1", "b2"])

            for _, row in chunk.iterrows():
                seen += 1
                pbar.update(1)

                debug = seen <= args.debug_first_n
                if debug:
                    print(f"\n--- Debug HHsearch-only task {seen - 1} ---")

                try:
                    fam_a = str(row["Main_HMM"])
                    fam_b = str(row["Sub_HMM"])
                    a1, a2 = int(row["a1"]), int(row["a2"])
                    b1, b2 = int(row["b1"]), int(row["b2"])

                    rmsd_data = compute_rmsd_for_hmm_segments(
                        fam_a, fam_b, a1, a2, b1, b2, ctx, failure_log, "HHSUITE_ONLY", debug
                    )
                    if rmsd_data is None:
                        continue

                    out_row: Dict[str, Any] = {
                        "Input_Row_Index": row.get("Input_Row_Index", np.nan),
                        "Main_HMM": fam_a,
                        "Sub_HMM": fam_b,
                        "Segment_Pair_Index": row.get("Segment_Pair_Index", np.nan),
                        "Segment_Pair_Raw": row.get("Segment_Pair_Raw", f"{a1}-{a2} -> {b1}-{b2}"),
                        "Source_Label": row.get("Source_Label", "HHSUITE_ONLY"),
                        "Prob": row.get("Prob", np.nan),
                    }
                    out_row.update(rmsd_data)
                    buffer_rows.append(out_row)
                    total_success += 1

                    if len(buffer_rows) >= args.flush_every:
                        wrote_header = flush_rows(buffer_rows, args.output_csv, wrote_header)

                except Exception:
                    if debug:
                        traceback.print_exc()
                    key = "HHSUITE_ONLY:UNKNOWN_ERR"
                    failure_log[key] = failure_log.get(key, 0) + 1

        if buffer_rows:
            wrote_header = flush_rows(buffer_rows, args.output_csv, wrote_header)
    finally:
        pbar.close()

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Successful RMSD rows: {total_success}")
    print(f"Output: {args.output_csv}")
    print_failure_summary(failure_log)
    print("=" * 70)


if __name__ == "__main__":
    main()
