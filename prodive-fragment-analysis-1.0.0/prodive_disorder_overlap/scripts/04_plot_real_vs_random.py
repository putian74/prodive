#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot observed and randomized DisProt overlap distributions.

Statistics are recomputed from the completed CSV outputs before plotting.

Required input files in --in-dir:
  1. global_residue_level_observed_summary.csv
  2. randomized_segment_control_distribution.csv

Optional input file in --in-dir:
  3. observed_vs_expected_randomized_summary.csv
     This is only used for validation. The plotted values are recomputed.

Outputs:
  1. real_vs_random_disorder_distribution.png/pdf
  2. real_vs_random_all_term_distribution.png/pdf
  3. real_vs_random_observed_expected_bar.png/pdf
  4. recomputed_real_vs_random_summary.csv
  5. validation_against_existing_summary.csv, if the optional summary file exists

"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LABEL_MAP: Dict[str, str] = {
    "disorder": "Strict disorder",
    "all_term": "All DisProt terms",
}

OBS_COL: Dict[str, str] = {
    "disorder": "global_Observed_Disorder_fraction",
    "all_term": "global_Observed_AllTerm_fraction",
}

RAND_COL: Dict[str, str] = {
    "disorder": "Random_Disorder_fraction",
    "all_term": "Random_AllTerm_fraction",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def setup_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update({
        "font.family": args.font_family,
        "font.size": args.font_size,
        "axes.labelsize": args.axis_label_size,
        "axes.titlesize": args.title_size,
        "xtick.labelsize": args.tick_label_size,
        "ytick.labelsize": args.tick_label_size,
        "legend.fontsize": args.legend_size,
        "figure.dpi": args.preview_dpi,
        "savefig.dpi": args.save_dpi,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def recompute_summary(global_df: pd.DataFrame, random_df: pd.DataFrame) -> pd.DataFrame:
    if len(global_df) != 1:
        raise ValueError("global_residue_level_observed_summary.csv should contain exactly one row.")

    rows = []

    for label in ["disorder", "all_term"]:
        observed = float(global_df.loc[0, OBS_COL[label]])
        random_values = pd.to_numeric(random_df[RAND_COL[label]], errors="coerce").dropna().to_numpy()

        if len(random_values) == 0:
            raise ValueError(f"No valid randomized values found in column: {RAND_COL[label]}")

        expected_mean = float(np.mean(random_values))
        expected_sd = float(np.std(random_values, ddof=1)) if len(random_values) > 1 else math.nan
        observed_over_expected = observed / expected_mean if expected_mean != 0 else math.nan

        # One-sided empirical p-values with plus-one correction:
        # P_enrichment = fraction of random values >= observed
        # P_depletion  = fraction of random values <= observed
        # Plus-one correction is applied.
        p_enrichment = (float(np.sum(random_values >= observed)) + 1.0) / (len(random_values) + 1.0)
        p_depletion = (float(np.sum(random_values <= observed)) + 1.0) / (len(random_values) + 1.0)

        z_score = (observed - expected_mean) / expected_sd if expected_sd and expected_sd > 0 else math.nan

        rows.append({
            "label": label,
            "display_label": LABEL_MAP[label],
            "Observed_fraction": observed,
            "Expected_fraction_mean": expected_mean,
            "Expected_fraction_sd": expected_sd,
            "Observed_over_Expected": observed_over_expected,
            "Z_observed_minus_expected": z_score,
            "P_enrichment": p_enrichment,
            "P_depletion": p_depletion,
            "n_permutations": int(len(random_values)),
            "Observed_percent": observed * 100.0,
            "Expected_percent_mean": expected_mean * 100.0,
            "Expected_percent_sd": expected_sd * 100.0 if np.isfinite(expected_sd) else math.nan,
        })

    return pd.DataFrame(rows)


def validate_existing_summary(recomputed: pd.DataFrame, existing_path: Path, out_dir: Path, tolerance: float) -> None:
    if not existing_path.exists():
        return

    existing = read_csv(existing_path)
    fields = [
        "Observed_fraction",
        "Expected_fraction_mean",
        "Expected_fraction_sd",
        "Observed_over_Expected",
        "P_enrichment",
        "P_depletion",
        "n_permutations",
    ]

    rows = []
    for label in ["disorder", "all_term"]:
        r = recomputed[recomputed["label"] == label].iloc[0]
        e = existing[existing["label"].astype(str) == label].iloc[0]

        for field in fields:
            recomputed_value = float(r[field])
            existing_value = float(e[field])
            diff = abs(recomputed_value - existing_value)
            rows.append({
                "label": label,
                "field": field,
                "recomputed": recomputed_value,
                "existing_summary": existing_value,
                "abs_difference": diff,
                "match_within_tolerance": diff <= tolerance,
            })

    validation = pd.DataFrame(rows)
    validation.to_csv(out_dir / "validation_against_existing_summary.csv", index=False, encoding="utf-8-sig")

    failed = validation[~validation["match_within_tolerance"]]
    if not failed.empty:
        raise ValueError(
            "Recomputed values do not match observed_vs_expected_randomized_summary.csv. "
            f"See: {out_dir / 'validation_against_existing_summary.csv'}"
        )


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_distribution(
    summary: pd.DataFrame,
    random_df: pd.DataFrame,
    label: str,
    out_dir: Path,
    bins: int,
) -> None:
    row = summary[summary["label"] == label].iloc[0]

    random_values_percent = pd.to_numeric(random_df[RAND_COL[label]], errors="coerce").dropna() * 100.0
    observed = float(row["Observed_percent"])
    expected = float(row["Expected_percent_mean"])
    sd = float(row["Expected_percent_sd"])
    z = float(row["Z_observed_minus_expected"])

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    ax.hist(random_values_percent, bins=bins, density=True, alpha=0.75)
    ax.axvline(observed, linestyle="-", linewidth=2.2, label=f"Observed = {observed:.2f}%")
    ax.axvline(expected, linestyle="--", linewidth=2.0, label=f"Random mean = {expected:.2f}%")
    if np.isfinite(sd):
        ax.axvspan(expected - sd, expected + sd, alpha=0.18, label="Random mean ± 1 SD")

    ax.set_xlabel("Residue-level overlap fraction (%)")
    ax.set_ylabel("Density")
    ax.set_title(f"Real vs randomized overlap: {LABEL_MAP[label]}")
    ax.legend(frameon=False, loc="upper right")

    ax.text(
        0.02,
        0.95,
        f"O/E = {float(row['Observed_over_Expected']):.3f}\n"
        f"z = {z:.2f}\n"
        f"P(depletion) = {float(row['P_depletion']):.3g}\n"
        f"P(enrichment) = {float(row['P_enrichment']):.3g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )

    stem = f"real_vs_random_{label}_distribution"
    save_fig(fig, out_dir, stem)


def plot_bar(summary: pd.DataFrame, out_dir: Path) -> None:
    summary = summary.set_index("label").loc[["disorder", "all_term"]].reset_index()

    x = np.arange(len(summary))
    width = 0.34

    observed = summary["Observed_percent"].to_numpy()
    expected = summary["Expected_percent_mean"].to_numpy()
    sd = summary["Expected_percent_sd"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    bars_obs = ax.bar(x - width / 2, observed, width, label="Observed")
    bars_exp = ax.bar(x + width / 2, expected, width, yerr=sd, capsize=4, label="Random expected")

    ax.set_ylabel("Residue-level overlap fraction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["display_label"])
    ax.set_title("Real vs randomized expected disorder overlap")
    ax.legend(frameon=False)

    y_max = max(np.max(observed), np.max(expected + sd))
    ax.set_ylim(0, y_max * 1.25)

    for bars in (bars_obs, bars_exp):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + y_max * 0.015,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for i, row in summary.iterrows():
        ax.text(
            x[i],
            y_max * 1.09,
            f"O/E={float(row['Observed_over_Expected']):.3f}\n"
            f"Pdep={float(row['P_depletion']):.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_fig(fig, out_dir, "real_vs_random_observed_expected_bar")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot real-vs-random disorder overlap comparison from completed result CSVs."
    )

    p.add_argument("--in-dir", required=True, help="Directory containing previous result CSVs.")
    p.add_argument("--out-dir", required=True, help="Directory for output figures.")
    p.add_argument("--bins", type=int, default=30, help="Number of histogram bins.")
    p.add_argument("--validation-tolerance", type=float, default=1e-12)

    p.add_argument("--font-family", default="DejaVu Sans")
    p.add_argument("--font-size", type=int, default=11)
    p.add_argument("--axis-label-size", type=int, default=12)
    p.add_argument("--title-size", type=int, default=13)
    p.add_argument("--tick-label-size", type=int, default=11)
    p.add_argument("--legend-size", type=int, default=10)
    p.add_argument("--preview-dpi", type=int, default=150)
    p.add_argument("--save-dpi", type=int, default=300)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_path = in_dir / "global_residue_level_observed_summary.csv"
    random_path = in_dir / "randomized_segment_control_distribution.csv"
    existing_summary_path = in_dir / "observed_vs_expected_randomized_summary.csv"

    global_df = read_csv(global_path)
    random_df = read_csv(random_path)

    summary = recompute_summary(global_df, random_df)
    summary.to_csv(out_dir / "recomputed_real_vs_random_summary.csv", index=False, encoding="utf-8-sig")

    validate_existing_summary(
        recomputed=summary,
        existing_path=existing_summary_path,
        out_dir=out_dir,
        tolerance=args.validation_tolerance,
    )

    setup_matplotlib(args)

    plot_distribution(summary, random_df, "disorder", out_dir, args.bins)
    plot_distribution(summary, random_df, "all_term", out_dir, args.bins)
    plot_bar(summary, out_dir)

    print("[INFO] Finished.")
    print(f"[INFO] Input directory : {in_dir}")
    print(f"[INFO] Output directory: {out_dir}")
    print("[INFO] Generated:")
    print("  real_vs_random_disorder_distribution.png/pdf")
    print("  real_vs_random_all_term_distribution.png/pdf")
    print("  real_vs_random_observed_expected_bar.png/pdf")
    print("  recomputed_real_vs_random_summary.csv")
    if existing_summary_path.exists():
        print("  validation_against_existing_summary.csv")


if __name__ == "__main__":
    main()
