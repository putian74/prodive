#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_SELECTED_SUBSET_CSV = "CHANGE_ME"
DEFAULT_SELECTED_SUBSET_ZIP = "CHANGE_ME"
DEFAULT_ORIGINAL_GLOBAL_CSV = "<PRODIVE_DATA_ROOT>/shared/global_high_score_summary_fin.csv"
DEFAULT_OUT_DIR = "CHANGE_ME"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge a selected high-rFCO subset with the original ProDive global score table."
    )
    p.add_argument("--selected-subset-csv", default=DEFAULT_SELECTED_SUBSET_CSV, help="Selected high-rFCO CSV. Ignored when --use-zip is set.")
    p.add_argument("--use-zip", action="store_true", help="Read selected_subset.csv from a ZIP archive.")
    p.add_argument("--selected-subset-zip", default=DEFAULT_SELECTED_SUBSET_ZIP, help="ZIP file containing selected_subset.csv.")
    p.add_argument("--zip-member", default="selected_subset.csv", help="CSV member name inside --selected-subset-zip.")
    p.add_argument("--original-global-csv", default=DEFAULT_ORIGINAL_GLOBAL_CSV, help="Original ProDive global high-score CSV.")
    p.add_argument("--outdir", default=DEFAULT_OUT_DIR, help="Output directory.")
    return p.parse_args()


def load_selected_subset(args: argparse.Namespace) -> pd.DataFrame:
    if args.use_zip:
        with zipfile.ZipFile(args.selected_subset_zip, "r") as zf:
            with zf.open(args.zip_member) as f:
                return pd.read_csv(io.BytesIO(f.read()), low_memory=False)
    return pd.read_csv(args.selected_subset_csv, low_memory=False)


def prepare_merged_df(args: argparse.Namespace) -> pd.DataFrame:
    selected = load_selected_subset(args)
    original = pd.read_csv(args.original_global_csv, low_memory=False)

    original = original.copy()
    original["global_row_index"] = range(1, len(original) + 1)

    selected["global_row_index"] = pd.to_numeric(selected["global_row_index"], errors="coerce")
    selected = selected.dropna(subset=["global_row_index"]).copy()
    selected["global_row_index"] = selected["global_row_index"].astype(int)

    merged = selected.merge(
        original,
        on="global_row_index",
        how="left",
        suffixes=("_highrfco", "_orig"),
    )

    if "Score" not in merged.columns:
        raise ValueError("The original global table does not contain a Score column. Check --original-global-csv.")

    merged["Score"] = pd.to_numeric(merged["Score"], errors="coerce")
    merged["rfco"] = pd.to_numeric(merged["rfco"], errors="coerce")
    if "rfco_delta_vs_random_mean" in merged.columns:
        merged["rfco_delta_vs_random_mean"] = pd.to_numeric(
            merged["rfco_delta_vs_random_mean"], errors="coerce"
        )

    return merged


def save_hist_score(df: pd.DataFrame, outpath: Path) -> None:
    s = df["Score"].dropna()
    if s.empty:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(s, bins=30, edgecolor="black")
    plt.xlabel("Original Score")
    plt.ylabel("Count")
    plt.title("Original Score distribution of selected high-rFCO fragments")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_ecdf_score(df: pd.DataFrame, outpath: Path) -> None:
    s = np.sort(df["Score"].dropna().to_numpy(dtype=float))
    if s.size == 0:
        return

    y = np.arange(1, len(s) + 1) / len(s)

    plt.figure(figsize=(7, 5))
    plt.plot(s, y)
    plt.xlabel("Original Score")
    plt.ylabel("ECDF")
    plt.title("ECDF of original Score for selected high-rFCO fragments")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_scatter_rfco_vs_score(df: pd.DataFrame, outpath: Path) -> None:
    sub = df[["rfco", "Score"]].dropna().copy()
    if sub.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(sub["Score"], sub["rfco"], s=18, alpha=0.7)
    plt.xlabel("Original Score")
    plt.ylabel("rFCO")
    plt.title("rFCO vs original Score")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_scatter_delta_vs_score(df: pd.DataFrame, outpath: Path) -> None:
    if "rfco_delta_vs_random_mean" not in df.columns:
        return
    sub = df[["rfco_delta_vs_random_mean", "Score"]].dropna().copy()
    if sub.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(sub["Score"], sub["rfco_delta_vs_random_mean"], s=18, alpha=0.7)
    plt.xlabel("Original Score")
    plt.ylabel("rFCO delta vs random mean")
    plt.title("rFCO delta vs original Score")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_top_table(df: pd.DataFrame, outpath: Path) -> None:
    keep_cols = [
        c for c in [
            "global_row_index",
            "pfam_id",
            "pfam_side",
            "segment_rank",
            "target_hmm_start",
            "target_hmm_end",
            "rfco",
            "rfco_delta_vs_random_mean",
            "rfco_empirical_p_le",
            "Score",
            "File",
            "Main_HMM",
            "Sub_HMM",
            "Main_Segment",
            "Sub_Segments_Details",
        ] if c in df.columns
    ]

    out = df[keep_cols].sort_values("Score", ascending=False).copy()
    out.to_csv(outpath, index=False)


def save_summary_txt(df: pd.DataFrame, outpath: Path) -> None:
    lines = []
    lines.append(f"selected rows: {len(df)}")

    s = df["Score"].dropna()
    if not s.empty:
        lines.append("")
        lines.append("Original Score summary:")
        lines.append(s.describe().to_string())

    if "rfco" in df.columns:
        r = pd.to_numeric(df["rfco"], errors="coerce").dropna()
        if not r.empty:
            corr = np.corrcoef(df.loc[r.index, "Score"], r)[0, 1] if len(r) >= 2 else np.nan
            lines.append("")
            lines.append(f"Pearson corr(Score, rFCO): {corr:.6f}")

    if "rfco_delta_vs_random_mean" in df.columns:
        sub = df[["Score", "rfco_delta_vs_random_mean"]].dropna()
        if len(sub) >= 2:
            corr = np.corrcoef(sub["Score"], sub["rfco_delta_vs_random_mean"])[0, 1]
            lines.append(f"Pearson corr(Score, delta): {corr:.6f}")

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    merged = prepare_merged_df(args)

    merged.to_csv(outdir / "selected_high_rfco_with_original_score.csv", index=False)

    save_hist_score(merged, outdir / "score_hist.png")
    save_ecdf_score(merged, outdir / "score_ecdf.png")
    save_scatter_rfco_vs_score(merged, outdir / "rfco_vs_score_scatter.png")
    save_scatter_delta_vs_score(merged, outdir / "delta_vs_score_scatter.png")
    save_top_table(merged, outdir / "selected_high_rfco_sorted_by_score.csv")
    save_summary_txt(merged, outdir / "summary.txt")

    print(f"Done. Output dir: {outdir}")


if __name__ == "__main__":
    main()
