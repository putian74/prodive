#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_SUMMARY_CSV = "CHANGE_ME"
DEFAULT_OUT_DIR = "CHANGE_ME"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check whether the contact-order signal remains after removing high-rFCO right-tail rows."
    )
    p.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV, help="Path to actual_vs_random_summary.csv.")
    p.add_argument("--outdir", default=DEFAULT_OUT_DIR, help="Output directory.")
    p.add_argument("--cutoffs", type=float, nargs="+", default=[0.35, 0.40, 0.50], help="rFCO cutoffs to remove at the high tail.")
    p.add_argument("--bins", type=int, default=40, help="Number of histogram bins.")
    return p.parse_args()


def plot_overlay_density(actual_vals, random_vals, title, xlabel, outpath, bins=40):
    a = pd.to_numeric(pd.Series(actual_vals), errors="coerce").dropna()
    r = pd.to_numeric(pd.Series(random_vals), errors="coerce").dropna()
    if a.empty or r.empty:
        return

    edges = np.histogram_bin_edges(np.concatenate([a.values, r.values]), bins=bins)

    plt.figure(figsize=(7, 5))
    plt.hist(a, bins=edges, density=True, alpha=0.55, label="Actual")
    plt.hist(r, bins=edges, density=True, alpha=0.55, label="Random mean")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_delta_hist(delta_vals, title, xlabel, outpath, bins=40):
    d = pd.to_numeric(pd.Series(delta_vals), errors="coerce").dropna()
    if d.empty:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(d, bins=bins, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_ecdf_compare(actual_vals, random_vals, title, outpath):
    a = np.sort(pd.to_numeric(pd.Series(actual_vals), errors="coerce").dropna().to_numpy(dtype=float))
    r = np.sort(pd.to_numeric(pd.Series(random_vals), errors="coerce").dropna().to_numpy(dtype=float))
    if a.size == 0 or r.size == 0:
        return

    ya = np.arange(1, len(a) + 1) / len(a)
    yr = np.arange(1, len(r) + 1) / len(r)

    plt.figure(figsize=(7, 5))
    plt.plot(a, ya, label="Actual")
    plt.plot(r, yr, label="Random mean")
    plt.title(title)
    plt.xlabel("rFCO")
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def summarize_subset(name: str, df: pd.DataFrame) -> dict:
    out = {"subset": name, "n_rows": len(df)}

    a = pd.to_numeric(df["rfco"], errors="coerce").dropna()
    r = pd.to_numeric(df["random_rfco_mean"], errors="coerce").dropna()
    d = pd.to_numeric(df["rfco_delta_vs_random_mean"], errors="coerce").dropna()

    out["actual_mean"] = float(a.mean()) if len(a) else np.nan
    out["actual_median"] = float(a.median()) if len(a) else np.nan
    out["actual_std"] = float(a.std()) if len(a) else np.nan

    out["random_mean"] = float(r.mean()) if len(r) else np.nan
    out["random_median"] = float(r.median()) if len(r) else np.nan
    out["random_std"] = float(r.std()) if len(r) else np.nan

    out["delta_mean"] = float(d.mean()) if len(d) else np.nan
    out["delta_median"] = float(d.median()) if len(d) else np.nan
    out["delta_std"] = float(d.std()) if len(d) else np.nan

    out["actual_p90"] = float(a.quantile(0.90)) if len(a) else np.nan
    out["actual_p95"] = float(a.quantile(0.95)) if len(a) else np.nan
    out["actual_p99"] = float(a.quantile(0.99)) if len(a) else np.nan

    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv, low_memory=False)

    required = ["rfco", "random_rfco_mean", "rfco_delta_vs_random_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"summary CSV missing columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).copy()

    summary_rows = []
    summary_rows.append(summarize_subset("original", df))

    plot_overlay_density(
        df["rfco"],
        df["random_rfco_mean"],
        title="Original: Actual vs Random mean rFCO",
        xlabel="rFCO",
        outpath=outdir / "original_overlay_density.png",
        bins=args.bins,
    )

    plot_delta_hist(
        df["rfco_delta_vs_random_mean"],
        title="Original: Actual rFCO - Random mean rFCO",
        xlabel="Delta rFCO",
        outpath=outdir / "original_delta_hist.png",
        bins=args.bins,
    )

    plot_ecdf_compare(
        df["rfco"],
        df["random_rfco_mean"],
        title="Original: ECDF of Actual vs Random mean rFCO",
        outpath=outdir / "original_ecdf.png",
    )

    for cutoff in args.cutoffs:
        sub = df[df["rfco"] < cutoff].copy()
        tag = f"rfco_lt_{cutoff:.2f}".replace(".", "p")

        summary_rows.append(summarize_subset(f"rfco < {cutoff:.2f}", sub))

        plot_overlay_density(
            sub["rfco"],
            sub["random_rfco_mean"],
            title=f"After removing rFCO >= {cutoff:.2f}: Actual vs Random mean",
            xlabel="rFCO",
            outpath=outdir / f"{tag}_overlay_density.png",
            bins=args.bins,
        )

        plot_delta_hist(
            sub["rfco_delta_vs_random_mean"],
            title=f"After removing rFCO >= {cutoff:.2f}: Delta rFCO",
            xlabel="Delta rFCO",
            outpath=outdir / f"{tag}_delta_hist.png",
            bins=args.bins,
        )

        plot_ecdf_compare(
            sub["rfco"],
            sub["random_rfco_mean"],
            title=f"After removing rFCO >= {cutoff:.2f}: ECDF",
            outpath=outdir / f"{tag}_ecdf.png",
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "summary_remove_high_rfco.csv", index=False)

    lines = []
    lines.append("Effect of removing high-rFCO right tail")
    lines.append("=" * 60)
    lines.append("")
    for _, row in summary_df.iterrows():
        lines.append(f"[{row['subset']}]")
        lines.append(f"n_rows        = {int(row['n_rows'])}")
        lines.append(f"actual_mean   = {row['actual_mean']:.6f}")
        lines.append(f"actual_median = {row['actual_median']:.6f}")
        lines.append(f"random_mean   = {row['random_mean']:.6f}")
        lines.append(f"random_median = {row['random_median']:.6f}")
        lines.append(f"delta_mean    = {row['delta_mean']:.6f}")
        lines.append(f"delta_median  = {row['delta_median']:.6f}")
        lines.append(f"actual_p90    = {row['actual_p90']:.6f}")
        lines.append(f"actual_p95    = {row['actual_p95']:.6f}")
        lines.append(f"actual_p99    = {row['actual_p99']:.6f}")
        lines.append("")

    (outdir / "summary_remove_high_rfco.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Done. Output dir: {outdir}")


if __name__ == "__main__":
    main()
