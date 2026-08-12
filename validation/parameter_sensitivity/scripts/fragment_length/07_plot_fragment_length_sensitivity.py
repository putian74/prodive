#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot fragment-length sensitivity results from already-saved CSV files.

This script does not read any .npy matrix and does not recompute thresholds or paths.
It only reads:
    sensitivity_summary.csv
    pass1_fragment_distribution_summary.csv

Edit CONFIG below and rerun freely.
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIG
# ==============================================================================

RESULT_DIR = Path("CHANGE_ME")

SUMMARY_CSV = RESULT_DIR / "sensitivity_summary.csv"
DIST_CSV = RESULT_DIR / "pass1_fragment_distribution_summary.csv"
PLOT_DIR = RESULT_DIR / "plots_from_saved_csv"

PLOT_DPI = 300

# Keep plots readable by default. Change these lists if needed.
PATH_MODES_TO_PLOT = ["fixed_span10", "fixed_r5"]
FIRST_LABELS_TO_PLOT = ["P0.01", "P0.05", "P0.1", "P0.5", "P1"]
SECOND_LABELS_TO_PLOT = ["NoS", "S8", "S10", "S12", "S15"]

# Set to None to keep all. Otherwise use lists above.
FILTER_TO_CONFIG = True

# Put one curve per threshold_label. If too crowded, set ONE_FILE_PER_PATH_MODE=True.
ONE_FILE_PER_PATH_MODE = True

METRICS = [
    ("retained_points", "Retained points"),
    ("candidate_point_density", "Candidate point density"),
    ("path_count", "Diagonal path count"),
    ("path_conversion_rate", "Path conversion rate"),
    ("isolated_or_noise_fraction", "Isolated/noise fraction"),
    ("frac_paths_span_8_13", "Fraction of paths with span 8–13 aa"),
    ("covered_pfams_from_filename", "Covered Pfam families from filenames"),
]




def _parse_list(text):
    if text is None or str(text).strip().lower() in {"", "none", "all"}:
        return None
    return [x.strip() for x in str(text).split(',') if x.strip()]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fragment-length sensitivity results from saved CSV files.")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR, help="Directory containing saved sensitivity CSV files.")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Path to sensitivity_summary.csv. Overrides --result-dir default.")
    parser.add_argument("--dist-csv", type=Path, default=None, help="Path to pass1_fragment_distribution_summary.csv. Overrides --result-dir default.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Output plot directory. Overrides --result-dir default.")
    parser.add_argument("--path-modes", default=','.join(PATH_MODES_TO_PLOT), help="Comma-separated path modes, or all.")
    parser.add_argument("--first-labels", default=','.join(FIRST_LABELS_TO_PLOT), help="Comma-separated first-layer labels, or all.")
    parser.add_argument("--second-labels", default=','.join(SECOND_LABELS_TO_PLOT), help="Comma-separated second-layer labels, or all.")
    parser.add_argument("--no-filter", action="store_true", help="Plot all rows without configured filtering.")
    parser.add_argument("--dpi", type=int, default=PLOT_DPI, help="Plot DPI.")
    return parser.parse_args()

def apply_runtime_args(args: argparse.Namespace) -> None:
    global RESULT_DIR, SUMMARY_CSV, DIST_CSV, PLOT_DIR, PATH_MODES_TO_PLOT, FIRST_LABELS_TO_PLOT, SECOND_LABELS_TO_PLOT, FILTER_TO_CONFIG, PLOT_DPI
    RESULT_DIR = args.result_dir
    SUMMARY_CSV = args.summary_csv if args.summary_csv is not None else RESULT_DIR / "sensitivity_summary.csv"
    DIST_CSV = args.dist_csv if args.dist_csv is not None else RESULT_DIR / "pass1_fragment_distribution_summary.csv"
    PLOT_DIR = args.plot_dir if args.plot_dir is not None else RESULT_DIR / "plots_from_saved_csv"
    PATH_MODES_TO_PLOT = _parse_list(args.path_modes) or []
    FIRST_LABELS_TO_PLOT = _parse_list(args.first_labels) or []
    SECOND_LABELS_TO_PLOT = _parse_list(args.second_labels) or []
    FILTER_TO_CONFIG = not bool(args.no_filter)
    PLOT_DPI = int(args.dpi)

# ==============================================================================
# Helpers
# ==============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def filter_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if FILTER_TO_CONFIG:
        if "path_mode" in out.columns and PATH_MODES_TO_PLOT:
            out = out[out["path_mode"].astype(str).isin(PATH_MODES_TO_PLOT)]
        if "first_threshold_label" in out.columns and FIRST_LABELS_TO_PLOT:
            out = out[out["first_threshold_label"].astype(str).isin(FIRST_LABELS_TO_PLOT)]
        if "second_layer_label" in out.columns and SECOND_LABELS_TO_PLOT:
            out = out[out["second_layer_label"].astype(str).isin(SECOND_LABELS_TO_PLOT)]

    return out


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace("|", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "p")
    )


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path, title_suffix: str = "") -> None:
    if df.empty or metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9.2, 6.4))

    grouped = df.groupby(["threshold_label", "path_mode"], sort=False)
    for (thr_label, mode), sub in grouped:
        sub = sub.sort_values("fragment_length")
        y = pd.to_numeric(sub[metric], errors="coerce")
        if y.notna().sum() == 0:
            continue
        ax.plot(
            sub["fragment_length"],
            y,
            marker="o",
            linewidth=1.5,
            label=f"{thr_label} | {mode}",
        )

    ax.axvline(6, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Fragment length k")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across fragment lengths{title_suffix}")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=7, ncol=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_distribution(dist_df: pd.DataFrame, out_dir: Path) -> None:
    if dist_df.empty:
        return

    d = dist_df.sort_values("fragment_length")

    if "raw_mean" in d.columns:
        fig, ax = plt.subplots(figsize=(8.0, 5.8))
        ax.plot(d["fragment_length"], d["raw_mean"], marker="o")
        ax.axvline(6, linestyle="--", linewidth=1.2)
        ax.set_xlabel("Fragment length k")
        ax.set_ylabel("Raw score mean")
        ax.set_title("Raw score mean across fragment lengths")
        ax.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(out_dir / "distribution_raw_mean.png", dpi=PLOT_DPI)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    plotted = False
    if "emp_norm_mean" in d.columns:
        ax.plot(d["fragment_length"], d["emp_norm_mean"], marker="o", label="empirical normalization")
        plotted = True
    if "theory_norm_mean" in d.columns:
        ax.plot(d["fragment_length"], d["theory_norm_mean"], marker="o", label="theory normalization")
        plotted = True
    if plotted:
        ax.axvline(6, linestyle="--", linewidth=1.2)
        ax.set_xlabel("Fragment length k")
        ax.set_ylabel("Normalized score mean")
        ax.set_title("Normalized score mean across fragment lengths")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "distribution_normalized_mean.png", dpi=PLOT_DPI)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_runtime_args(args)
    ensure_dir(PLOT_DIR)

    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV}")

    summary_df = pd.read_csv(SUMMARY_CSV)
    summary_df = filter_summary(summary_df)

    print("=" * 100)
    print("Plotting from saved CSV only")
    print(f"SUMMARY_CSV: {SUMMARY_CSV}")
    print(f"DIST_CSV:    {DIST_CSV}")
    print(f"PLOT_DIR:    {PLOT_DIR}")
    print(f"Rows used:   {len(summary_df)}")
    print("=" * 100)

    if ONE_FILE_PER_PATH_MODE and "path_mode" in summary_df.columns:
        for mode, sub in summary_df.groupby("path_mode", sort=False):
            mode_dir = PLOT_DIR / safe_name(mode)
            ensure_dir(mode_dir)
            for metric, ylabel in METRICS:
                plot_metric(
                    sub,
                    metric,
                    ylabel,
                    mode_dir / f"metric_{safe_name(metric)}_{safe_name(mode)}.png",
                    title_suffix=f" ({mode})",
                )
    else:
        for metric, ylabel in METRICS:
            plot_metric(summary_df, metric, ylabel, PLOT_DIR / f"metric_{safe_name(metric)}.png")

    if DIST_CSV.exists():
        dist_df = pd.read_csv(DIST_CSV)
        plot_distribution(dist_df, PLOT_DIR)
    else:
        print(f"[WARN] Missing distribution CSV, skip distribution plots: {DIST_CSV}")

    print("=" * 100)
    print("[DONE] Plots written from saved CSV")
    print(f"Output: {PLOT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
