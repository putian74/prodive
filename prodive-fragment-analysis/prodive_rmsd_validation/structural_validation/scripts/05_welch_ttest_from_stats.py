#!/usr/bin/env python3
"""Run Welch's t-test from grouped RMSD summary statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from scipy import stats


def find_type(df: pd.DataFrame, pattern: str) -> str:
    matches: List[str] = []
    for value in df["Type"].dropna().astype(str).unique():
        if pattern.lower() in value.lower():
            matches.append(value)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Type matching '{pattern}', found {matches}")
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Welch's t-test by aligned length from summary statistics.")
    parser.add_argument("--stats-csv", type=Path, required=True, help="Summary CSV produced by 04_compare_real_random_core.py.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV for Welch's t-test results.")
    parser.add_argument("--real-pattern", default="Real", help="Substring identifying the real group in the Type column.")
    parser.add_argument("--random-pattern", default="Random", help="Substring identifying the random group in the Type column.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.stats_csv.exists():
        raise FileNotFoundError(f"Summary CSV does not exist: {args.stats_csv}")

    df = pd.read_csv(args.stats_csv)
    required = ["Type", "Aligned_Atoms", "count", "mean", "std"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.stats_csv}: {missing}")

    real_type = find_type(df, args.real_pattern)
    random_type = find_type(df, args.random_pattern)
    rows = []

    for aligned_atoms in sorted(df["Aligned_Atoms"].dropna().unique()):
        real = df[(df["Aligned_Atoms"] == aligned_atoms) & (df["Type"] == real_type)]
        random_group = df[(df["Aligned_Atoms"] == aligned_atoms) & (df["Type"] == random_type)]
        if real.empty or random_group.empty:
            continue

        r = real.iloc[0]
        q = random_group.iloc[0]
        t_stat, p_value = stats.ttest_ind_from_stats(
            mean1=float(r["mean"]),
            std1=float(r["std"]),
            nobs1=float(r["count"]),
            mean2=float(q["mean"]),
            std2=float(q["std"]),
            nobs2=float(q["count"]),
            equal_var=False,
        )
        rows.append({
            "Aligned_Atoms": int(aligned_atoms),
            "Real_Type": real_type,
            "Random_Type": random_type,
            "Real_N": float(r["count"]),
            "Random_N": float(q["count"]),
            "Real_Mean": float(r["mean"]),
            "Random_Mean": float(q["mean"]),
            "Real_Std": float(r["std"]),
            "Random_Std": float(q["std"]),
            "T_Statistic": float(t_stat),
            "P_Value": float(p_value),
        })

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Welch t-test results written to: {args.output_csv}")
    if not out.empty:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
