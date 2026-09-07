#!/usr/bin/env python3
"""Compare real RMSD distributions against length-stratified random controls.

The script filters real and random RMSD tables to a compact core region, writes a
summary table by aligned length and group, and optionally produces a boxplot.
Use the de novo-Pfam RMSD table and its corresponding random-control CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_lengths(text: str) -> List[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one core length is required.")
    return sorted(set(values))


def to_numeric_if_present(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def infer_coverage(df: pd.DataFrame, is_real: bool) -> pd.DataFrame:
    df = df.copy()
    to_numeric_if_present(df, ["Aligned_Atoms", "RMSD", "Coverage", "Target_Len", "HMM_Len", "HMM_Len_A", "HMM_Len_B"])

    if "Coverage" in df.columns and df["Coverage"].notna().any():
        return df

    if "Target_Len" in df.columns and df["Target_Len"].notna().any():
        df["Coverage"] = df["Aligned_Atoms"] / df["Target_Len"]
        return df

    if "HMM_Len" in df.columns and df["HMM_Len"].notna().any():
        df["Target_Len"] = df["HMM_Len"]
        df["Coverage"] = df["Aligned_Atoms"] / df["Target_Len"]
        return df

    if "HMM_Len_A" in df.columns and "HMM_Len_B" in df.columns:
        df["Target_Len"] = df[["HMM_Len_A", "HMM_Len_B"]].min(axis=1)
        df["Coverage"] = df["Aligned_Atoms"] / df["Target_Len"]
        return df

    if not is_real:
        df["Coverage"] = 1.0
        return df

    raise ValueError("Cannot infer Coverage from the real RMSD table. Provide Coverage, Target_Len, HMM_Len, or HMM_Len_A/HMM_Len_B.")


def load_core_table(
    csv_path: Path,
    label: str,
    core_lengths: Sequence[int],
    coverage_threshold: float,
    is_real: bool,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["RMSD", "Aligned_Atoms"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = infer_coverage(df, is_real=is_real)
    df = df[
        (df["Coverage"] >= coverage_threshold)
        & (df["Aligned_Atoms"].isin(core_lengths))
        & df["RMSD"].notna()
    ].copy()

    out = df[["RMSD", "Aligned_Atoms", "Coverage"]].copy()
    out["Type"] = label
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(["Type", "Aligned_Atoms"])["RMSD"].describe().reset_index()
    pass_rate = (
        df.groupby(["Type", "Aligned_Atoms"])["RMSD"]
        .apply(lambda x: float((x < 1.0).mean() * 100.0))
        .reset_index(name="Pass_Rate_lt_1A_percent")
    )
    merged = summary.merge(pass_rate, on=["Type", "Aligned_Atoms"], how="left")
    return merged.round(6)


def plot_boxplot(df: pd.DataFrame, output_path: Path, title: str, ymax: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid", context="talk", font_scale=0.9)
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(
        x="Aligned_Atoms",
        y="RMSD",
        hue="Type",
        data=df,
        showfliers=False,
        linewidth=1.5,
        width=0.7,
    )
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="1.0 A")
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Aligned core length (C-alpha atoms)", fontsize=14)
    ax.set_ylabel("C-alpha RMSD (A)", fontsize=14)
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare real and random RMSD distributions in the compact core region.")
    parser.add_argument("--real-csv", type=Path, required=True, help="Real RMSD CSV file.")
    parser.add_argument("--random-csv", type=Path, required=True, help="Random-control RMSD CSV file.")
    parser.add_argument("--output-stats-csv", type=Path, required=True, help="Output summary statistics CSV.")
    parser.add_argument("--output-plot", type=Path, default=None, help="Optional output boxplot path.")
    parser.add_argument("--real-label", default="Real signal", help="Label for the real group.")
    parser.add_argument("--random-label", default="Random noise", help="Label for the random-control group.")
    parser.add_argument("--core-lengths", type=parse_lengths, default=parse_lengths("8,9,10,11,12,13"), help="Comma-separated aligned core lengths.")
    parser.add_argument("--coverage-threshold", type=float, default=0.8, help="Minimum structural coverage.")
    parser.add_argument("--plot-title", default="Core region comparison: real vs. random", help="Boxplot title.")
    parser.add_argument("--ymax", type=float, default=4.0, help="Y-axis maximum for the boxplot.")
    parser.add_argument("--write-filtered-csv", type=Path, default=None, help="Optional CSV containing the filtered real and random rows used for the comparison.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    real_df = load_core_table(args.real_csv, args.real_label, args.core_lengths, args.coverage_threshold, is_real=True)
    random_df = load_core_table(args.random_csv, args.random_label, args.core_lengths, args.coverage_threshold, is_real=False)
    if real_df.empty:
        raise RuntimeError("No real rows remain after filtering.")
    if random_df.empty:
        raise RuntimeError("No random-control rows remain after filtering.")

    combined = pd.concat([real_df, random_df], ignore_index=True)
    args.output_stats_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(combined)
    summary.to_csv(args.output_stats_csv, index=False)

    if args.write_filtered_csv is not None:
        args.write_filtered_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.write_filtered_csv, index=False)

    if args.output_plot is not None:
        plot_boxplot(combined, args.output_plot, args.plot_title, args.ymax)

    print(f"Real rows after filtering: {len(real_df)}")
    print(f"Random rows after filtering: {len(random_df)}")
    print(f"Summary written to: {args.output_stats_csv}")
    if args.output_plot is not None:
        print(f"Plot written to: {args.output_plot}")


if __name__ == "__main__":
    main()
