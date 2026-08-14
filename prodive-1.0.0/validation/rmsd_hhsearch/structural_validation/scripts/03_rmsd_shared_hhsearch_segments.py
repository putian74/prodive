#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute RMSD for Shared/Both rows using HHsearch-defined segment boundaries.

Input:
  subset_data_Top_100%_overlap_check_dual_20.csv

Rows processed:
  Is_Novel == False and HH_Main_Segment / HH_Sub_Segment exist

Segment boundaries used:
  HH_Main_Segment / HH_Sub_Segment

Output:
  RMSD results for HHsearch-defined boundaries in the shared set.
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
    flush_rows,
    init_pymol,
    is_false_like,
    parse_segment,
    print_failure_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute RMSD for shared ProDive-HHsearch rows using HHsearch segment boundaries."
    )
    p.add_argument("--ref-csv", required=True)
    p.add_argument("--pfam-dir", required=True, help="Merged Pfam runtime directory containing HHM/MSA/PDB/AF files.")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--chunksize", type=int, default=2000)
    p.add_argument("--flush-every", type=int, default=500)
    p.add_argument("--debug-first-n", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--max-search-depth", type=int, default=200)
    p.add_argument("--max-len-diff", type=int, default=5)
    p.add_argument("--min-mean-plddt", type=float, default=70.0)
    p.add_argument("--max-head-tail-pae", type=float, default=10.0)
    p.add_argument("--min-aligned-coverage", type=float, default=0.0,
                   help="Optional post-superposition coverage filter. 0 disables filtering.")
    return p.parse_args()


def required_columns_exist(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}. Current columns: {list(df.columns)}")


def count_eligible_rows(ref_csv: str, chunksize: int) -> int:
    total = 0
    for chunk in pd.read_csv(ref_csv, chunksize=chunksize, usecols=["Is_Novel", "HH_Main_Segment", "HH_Sub_Segment"]):
        mask = (
            chunk["Is_Novel"].apply(is_false_like)
            & chunk["HH_Main_Segment"].notna()
            & chunk["HH_Sub_Segment"].notna()
        )
        total += int(mask.sum())
    return total


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.ref_csv):
        raise FileNotFoundError(f"Input CSV does not exist: {args.ref_csv}")
    if os.path.exists(args.output_csv) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_csv}. Use --overwrite to replace it.")
    if os.path.exists(args.output_csv) and args.overwrite:
        os.remove(args.output_csv)

    cfg = RMSDConfig(
        pfam_dir=args.pfam_dir,
        max_search_depth=args.max_search_depth,
        len_diff_mode="fixed",
        max_len_diff=args.max_len_diff,
        min_mean_plddt=args.min_mean_plddt,
        max_head_tail_pae=args.max_head_tail_pae,
        min_aligned_coverage=args.min_aligned_coverage,
    )
    ctx = StructureContext(cfg)
    failure_log: Dict[str, int] = {}

    print("Step 0: Initialize PyMOL...")
    init_pymol(quiet=True)

    total_tasks = count_eligible_rows(args.ref_csv, args.chunksize)
    print("Step 1: Compute RMSD for shared rows using HHsearch segments")
    print(f"[INFO] REF_CSV      = {args.ref_csv}")
    print(f"[INFO] PFAM_DIR     = {args.pfam_dir}")
    print(f"[INFO] OUTPUT       = {args.output_csv}")
    print(f"[INFO] MAX_LEN_DIFF = {args.max_len_diff}")
    print(f"[INFO] Eligible rows = {total_tasks}")

    wrote_header = False
    buffer_rows: List[Dict[str, Any]] = []
    total_success = 0
    seen = 0

    pbar = tqdm(total=total_tasks, desc="Shared HHsearch-seg RMSD")
    try:
        for chunk in pd.read_csv(args.ref_csv, chunksize=args.chunksize):
            required_columns_exist(
                chunk,
                ["Main_HMM", "Sub_HMM", "Main_Segment", "Sub_Segment", "Is_Novel", "HH_Main_Segment", "HH_Sub_Segment"],
            )
            mask = (
                chunk["Is_Novel"].apply(is_false_like)
                & chunk["HH_Main_Segment"].notna()
                & chunk["HH_Sub_Segment"].notna()
            )
            sub = chunk.loc[mask]

            for row_idx, row in sub.iterrows():
                seen += 1
                pbar.update(1)

                debug = seen <= args.debug_first_n
                if debug:
                    print(f"\n--- Debug shared HHsearch task {seen - 1} | ref row={row_idx} ---")

                try:
                    fam_a = str(row["Main_HMM"])
                    fam_b = str(row["Sub_HMM"])
                    a1, a2 = parse_segment(row["HH_Main_Segment"])
                    b1, b2 = parse_segment(row["HH_Sub_Segment"])

                    rmsd_data = compute_rmsd_for_hmm_segments(
                        fam_a, fam_b, a1, a2, b1, b2, ctx, failure_log, "SHARED_HHSEARCH_SEG", debug
                    )
                    if rmsd_data is None:
                        continue

                    out_row: Dict[str, Any] = {
                        "Ref_Row_Index": row_idx,
                        "Main_HMM": fam_a,
                        "Sub_HMM": fam_b,
                        "ProDive_Main_Segment": row.get("Main_Segment", np.nan),
                        "ProDive_Sub_Segment": row.get("Sub_Segment", np.nan),
                        "HH_Main_Segment": row.get("HH_Main_Segment", np.nan),
                        "HH_Sub_Segment": row.get("HH_Sub_Segment", np.nan),
                        "Segment_Source": "HH_SEG_FROM_FALSE_ROW",
                        "Segment_Pair_Raw": f"{a1}-{a2} -> {b1}-{b2}",
                        "Dual_Min_Coverage": row.get("Dual_Min_Coverage", np.nan),
                        "Is_Novel": row.get("Is_Novel", np.nan),
                    }
                    out_row.update(rmsd_data)
                    buffer_rows.append(out_row)
                    total_success += 1

                    if len(buffer_rows) >= args.flush_every:
                        wrote_header = flush_rows(buffer_rows, args.output_csv, wrote_header)

                except Exception:
                    if debug:
                        traceback.print_exc()
                    key = "SHARED_HHSEARCH_SEG:UNKNOWN_ERR"
                    failure_log[key] = failure_log.get(key, 0) + 1

        if buffer_rows:
            wrote_header = flush_rows(buffer_rows, args.output_csv, wrote_header)
    finally:
        pbar.close()

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Eligible shared HHsearch rows: {total_tasks}")
    print(f"Successful RMSD rows: {total_success}")
    print(f"Output: {args.output_csv}")
    print_failure_summary(failure_log)
    print("=" * 70)


if __name__ == "__main__":
    main()
