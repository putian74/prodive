#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract ProDive global-score rows involving phi-related Pfam families."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_BEST_HIT_FILE = Path("CHANGE_ME")
DEFAULT_GLOBAL_SCORE_FILE = Path("<PRODIVE_DATA_ROOT>/shared/global_high_score_summary_fin.csv")
DEFAULT_OUT_DIR = Path("CHANGE_ME")


def get_match_side(row, pfam_set: set[str]) -> str:
    m = row["Main_HMM"] in pfam_set
    s = row["Sub_HMM"] in pfam_set
    if m and s:
        return "both"
    if m:
        return "main"
    if s:
        return "sub"
    return ""


def get_matched_pfams(row, pfam_set: set[str]) -> str:
    hits = []
    if row["Main_HMM"] in pfam_set:
        hits.append(row["Main_HMM"])
    if row["Sub_HMM"] in pfam_set and row["Sub_HMM"] not in hits:
        hits.append(row["Sub_HMM"])
    return ";".join(hits)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_df = pd.read_csv(args.best_hit_file)
    if "pfam_id" not in best_df.columns:
        raise ValueError(f"Best-hit file has no pfam_id column. Available columns: {list(best_df.columns)}")

    pfam_ids = best_df["pfam_id"].astype(str).str.strip().str.upper()
    pfam_ids = sorted(set(x for x in pfam_ids if x and x != "NAN"))
    pfam_set = set(pfam_ids)

    print(f"[INFO] unique pfam_id from best_hit: {len(pfam_ids)}")

    score_df = pd.read_csv(args.global_score_file)
    required_cols = ["Main_HMM", "Sub_HMM"]
    missing = [c for c in required_cols if c not in score_df.columns]
    if missing:
        raise ValueError(f"Global score file is missing columns: {missing}")

    score_df["Main_HMM"] = score_df["Main_HMM"].astype(str).str.strip().str.upper()
    score_df["Sub_HMM"] = score_df["Sub_HMM"].astype(str).str.strip().str.upper()

    main_hit = score_df["Main_HMM"].isin(pfam_set)
    sub_hit = score_df["Sub_HMM"].isin(pfam_set)
    filtered_df = score_df[main_hit | sub_hit].copy()

    filtered_df["match_side"] = filtered_df.apply(lambda row: get_match_side(row, pfam_set), axis=1)
    filtered_df["matched_pfam_ids"] = filtered_df.apply(lambda row: get_matched_pfams(row, pfam_set), axis=1)

    all_out = args.out_dir / "global_high_score_rows_matching_phi_pfams.csv"
    filtered_df.to_csv(all_out, index=False, encoding="utf-8-sig")

    per_pfam_dir = args.out_dir / "per_pfam"
    per_pfam_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for pf in pfam_ids:
        sub_df = filtered_df[(filtered_df["Main_HMM"] == pf) | (filtered_df["Sub_HMM"] == pf)].copy()
        if len(sub_df) == 0:
            continue
        out_file = per_pfam_dir / f"{pf}_matches.csv"
        sub_df.to_csv(out_file, index=False, encoding="utf-8-sig")
        summary_rows.append({
            "pfam_id": pf,
            "n_rows": len(sub_df),
            "n_main_hit": int((sub_df["Main_HMM"] == pf).sum()),
            "n_sub_hit": int((sub_df["Sub_HMM"] == pf).sum()),
            "output_file": str(out_file),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_out = args.out_dir / "per_pfam_match_summary.csv"
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    special_pf_df = filtered_df[(filtered_df["Main_HMM"] == args.special_pfam) | (filtered_df["Sub_HMM"] == args.special_pfam)].copy()
    special_pf_out = args.out_dir / f"{args.special_pfam}_matches.csv"
    special_pf_df.to_csv(special_pf_out, index=False, encoding="utf-8-sig")

    summary_lines = [
        f"unique pfam_id from best_hit: {len(pfam_ids)}",
        f"total rows in global score file: {len(score_df)}",
        f"rows matching at least one phi-related pfam: {len(filtered_df)}",
        f"unique matched pfam in filtered results: {filtered_df['matched_pfam_ids'].nunique()}",
        f"all matched rows file: {all_out}",
        f"per-pfam summary file: {summary_out}",
        f"{args.special_pfam} only file: {special_pf_out}",
    ]
    (args.out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(f"[INFO] {x}" for x in summary_lines))
    print("[INFO] Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract global-score rows involving phi-related Pfam families.")
    parser.add_argument("--best-hit-file", type=Path, default=DEFAULT_BEST_HIT_FILE, help="Best-hit CSV from step 01.")
    parser.add_argument("--global-score-file", type=Path, default=DEFAULT_GLOBAL_SCORE_FILE, help="ProDive global high-score summary CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--special-pfam", default="PF00249", help="Optional single Pfam family to export separately.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
