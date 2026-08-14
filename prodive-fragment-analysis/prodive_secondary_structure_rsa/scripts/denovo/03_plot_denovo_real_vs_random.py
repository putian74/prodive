#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot real denovo fragments vs random/null controls:
    left  column: secondary-structure class composition, stacked by RSA location
    right column: average RSA probability histogram + smoothed density curve

This script is the plotting step after:
  1) denovo_features_calculated_true.csv
  2) denovo_features_random_controls.csv

"""

import os
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # scipy is optional; a small fallback is provided below
    gaussian_filter1d = None


# =========================
# Configuration
# =========================
REAL_CSV = ""
NULL_CSV = ""
OUTPUT_PNG = ""

CHUNKSIZE = 500_000

# Keep only successfully mapped fragments with interpretable RSA/location values.
# This matches the upstream output convention: only Status == OK enters the final plot.
VALID_STATUS = {"OK"}

SS_CLASSES = [
    "Helix_Dominant",
    "Sheet_Dominant",
    "Coil/Loop_Dominant",
    "Mixed",
]

LOCATIONS = [
    "Buried (Internal)",
    "Intermediate",
    "Exposed (Surface)",
]

LOCATION_COLORS = {
    "Buried (Internal)": "#4B7DB8",
    "Intermediate": "#F9E09A",
    "Exposed (Surface)": "#D62728",
}

# RSA thresholds used by the upstream classification scripts.
BURIED_THRESHOLD = 0.20
EXPOSED_THRESHOLD = 0.50

# The histogram uses probability rather than density.
RSA_BINS = np.linspace(0.0, 1.0, 51)
SMOOTH_SIGMA_BINS = 1.15


# =========================
# Data loading and counting
# =========================
def _read_header(path: str) -> List[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def load_plot_data(path: str) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Returns
    -------
    class_location_counts:
        index=SS class, columns=location, values=count
    rsa_values:
        Avg_RSA values used in the histogram
    total_n:
        total valid fragments included in this plot
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")

    header = _read_header(path)
    required = ["SS_Class", "Location", "Avg_RSA", "Status"]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}; available columns: {header}")

    counts = pd.DataFrame(0, index=SS_CLASSES, columns=LOCATIONS, dtype=np.int64)
    rsa_blocks = []
    total_n = 0

    for chunk in pd.read_csv(
        path,
        usecols=required,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):
        chunk["Status"] = chunk["Status"].astype(str)
        chunk["SS_Class"] = chunk["SS_Class"].astype(str)
        chunk["Location"] = chunk["Location"].astype(str)
        chunk["Avg_RSA"] = pd.to_numeric(chunk["Avg_RSA"], errors="coerce")

        valid = (
            chunk["Status"].isin(VALID_STATUS)
            & chunk["SS_Class"].isin(SS_CLASSES)
            & chunk["Location"].isin(LOCATIONS)
            & chunk["Avg_RSA"].notna()
            & (chunk["Avg_RSA"] >= 0.0)
            & (chunk["Avg_RSA"] <= 1.0)
        )
        df = chunk.loc[valid, ["SS_Class", "Location", "Avg_RSA"]]
        if df.empty:
            continue

        tab = pd.crosstab(df["SS_Class"], df["Location"])
        for ss in SS_CLASSES:
            if ss not in tab.index:
                continue
            for loc in LOCATIONS:
                if loc in tab.columns:
                    counts.loc[ss, loc] += int(tab.loc[ss, loc])

        vals = df["Avg_RSA"].to_numpy(dtype=np.float32, copy=True)
        rsa_blocks.append(vals)
        total_n += int(vals.size)

    rsa_values = np.concatenate(rsa_blocks) if rsa_blocks else np.array([], dtype=np.float32)
    return counts, rsa_values, total_n


# =========================
# Plotting functions
# =========================
def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.1, length=4)


def plot_ss_location_bar(ax, counts: pd.DataFrame, show_legend: bool = False) -> None:
    total = int(counts.values.sum())
    x = np.arange(len(SS_CLASSES))
    width = 0.68
    bottom = np.zeros(len(SS_CLASSES), dtype=float)

    if total == 0:
        proportions = counts.astype(float)
    else:
        proportions = counts.astype(float) / float(total)

    for loc in LOCATIONS:
        y = proportions.loc[SS_CLASSES, loc].to_numpy(dtype=float)
        ax.bar(
            x,
            y,
            width=width,
            bottom=bottom,
            color=LOCATION_COLORS[loc],
            edgecolor="black",
            linewidth=0.8,
            label=loc,
        )
        bottom += y

    # Label each secondary-structure class with its count.
    class_n = counts.loc[SS_CLASSES, LOCATIONS].sum(axis=1).astype(int)
    for xi, yi, n in zip(x, bottom, class_n):
        if n <= 0:
            continue
        ax.text(
            xi,
            min(yi + 0.018, 0.975),
            f"N={n}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xlim(-0.6, len(SS_CLASSES) - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Proportion of Total Fragments")
    ax.set_xlabel("Secondary Structure Class")
    ax.set_xticks(x)
    ax.set_xticklabels(SS_CLASSES, rotation=10, ha="right")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    despine(ax)

    if show_legend:
        ax.legend(
            title="Location",
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            borderaxespad=0.0,
        )


def _smooth_probability_curve(prob: np.ndarray) -> np.ndarray:
    if gaussian_filter1d is not None:
        return gaussian_filter1d(prob, sigma=SMOOTH_SIGMA_BINS, mode="nearest")

    # Fallback moving average when scipy is unavailable.
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    return np.convolve(prob, kernel, mode="same")


def plot_rsa_hist(ax, rsa_values: np.ndarray) -> None:
    rsa_values = np.asarray(rsa_values, dtype=float)
    rsa_values = rsa_values[np.isfinite(rsa_values)]
    rsa_values = rsa_values[(rsa_values >= 0.0) & (rsa_values <= 1.0)]

    if rsa_values.size == 0:
        prob = np.zeros(len(RSA_BINS) - 1, dtype=float)
    else:
        weights = np.ones_like(rsa_values, dtype=float) / float(rsa_values.size)
        prob, _ = np.histogram(rsa_values, bins=RSA_BINS, weights=weights)
        ax.hist(
            rsa_values,
            bins=RSA_BINS,
            weights=weights,
            color="gray",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.6,
        )

    centers = (RSA_BINS[:-1] + RSA_BINS[1:]) / 2.0
    smooth_prob = _smooth_probability_curve(prob)
    ax.plot(centers, smooth_prob, color="dimgray", linewidth=1.25)

    ax.axvline(
        BURIED_THRESHOLD,
        color="#4C78A8",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Buried threshold (0.2)",
    )
    ax.axvline(
        EXPOSED_THRESHOLD,
        color="#E45756",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Exposed threshold (0.5)",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.085)
    ax.set_xlabel("Average RSA")
    ax.set_ylabel("Probability")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(frameon=False, loc="upper right")
    despine(ax)


def make_figure(real_csv: str, null_csv: str, output_png: str) -> None:
    print(f"[READ] real: {real_csv}")
    real_counts, real_rsa, real_n = load_plot_data(real_csv)
    print(f"       valid real fragments: {real_n:,}")
    print(real_counts)

    print(f"[READ] null/random: {null_csv}")
    null_counts, null_rsa, null_n = load_plot_data(null_csv)
    print(f"       valid null fragments: {null_n:,}")
    print(null_counts)

    setup_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.15), dpi=128)

    plot_ss_location_bar(axes[0, 0], real_counts, show_legend=True)
    plot_rsa_hist(axes[0, 1], real_rsa)
    plot_ss_location_bar(axes[1, 0], null_counts, show_legend=False)
    plot_rsa_hist(axes[1, 1], null_rsa)

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.075,
        top=0.985,
        wspace=0.12,
        hspace=0.23,
    )

    out_dir = os.path.dirname(output_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_png, dpi=128)
    plt.close(fig)
    print(f"[OUT] {output_png}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot denovo real fragments vs random/null controls: secondary structure + RSA."
    )
    p.add_argument("--real-csv", required=True, help="denovo_features_calculated_true.csv")
    p.add_argument("--null-csv", required=True, help="denovo_features_random_controls.csv")
    p.add_argument("--out", required=True, help="Output PNG path.")
    return p.parse_args()


def main():
    args = parse_args()
    make_figure(args.real_csv, args.null_csv, args.out)


if __name__ == "__main__":
    main()
