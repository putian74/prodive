#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 
# =========================================================
RESULT_DIR = Path(
    "CHANGE_ME"
)

PLOT_DIR = RESULT_DIR / "plots_only"

# 
FOCUS_LABELS = ["Any", "Fold_EARLY", "Fold_LATE", "Stab_STRONG", "Stab_MEDIUM"]

# ，，
MAX_RANDOM_POINTS_FOR_PLOT = 50000

RANDOM_SEED = 20260409


# =========================================================
# 
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_col(label: str, metric: str) -> str:
    """
    metric:
      count
      frac
      runfrac
    """
    if label == "Any":
        return {
            "count": "Any_Start2Fold_Overlap_Count",
            "frac": "Any_Start2Fold_Overlap_Frac",
            "runfrac": "Any_Start2Fold_Longest_Run_Frac",
        }[metric]

    return {
        "count": f"{label}_segment_overlap_count",
        "frac": f"{label}_segment_overlap_frac",
        "runfrac": f"{label}_segment_longest_run_frac",
    }[metric]


def load_required_files(result_dir: Path) -> Dict[str, pd.DataFrame]:
    paths = {
        "actual": result_dir / "all_mapped_segments_with_start2fold_metrics.csv",
        "random": result_dir / "random_sampled_windows_all.csv",
        "overall": result_dir / "overall_class_enrichment_summary.csv",
        "segment": result_dir / "segment_random_test_summary.csv",
    }

    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f": {p}")

    return {
        "actual": pd.read_csv(paths["actual"]),
        "random": pd.read_csv(paths["random"]),
        "overall": pd.read_csv(paths["overall"]),
        "segment": pd.read_csv(paths["segment"]),
    }


def get_plot_arrays(
    actual_df: pd.DataFrame,
    random_df: pd.DataFrame,
    label: str,
    metric: str,
    max_random_points: int = MAX_RANDOM_POINTS_FOR_PLOT,
) -> Tuple[np.ndarray, np.ndarray]:
    col = pick_col(label, metric)

    a = pd.to_numeric(actual_df[col], errors="coerce").dropna().to_numpy()
    r = pd.to_numeric(random_df[col], errors="coerce").dropna().to_numpy()

    if len(r) > max_random_points:
        rng = np.random.default_rng(RANDOM_SEED)
        r = rng.choice(r, size=max_random_points, replace=False)

    return a, r


def save_overlay_density_hist(
    actual_df: pd.DataFrame,
    random_df: pd.DataFrame,
    label: str,
    metric: str,
    out_path: Path,
) -> None:
    a, r = get_plot_arrays(actual_df, random_df, label, metric)

    if len(a) == 0 or len(r) == 0:
        return

    title_map = {
        "count": "Overlap count",
        "frac": "Overlap fraction",
        "runfrac": "Longest-run fraction",
    }

    plt.figure(figsize=(8, 5))
    plt.hist(r, bins=40, density=True, alpha=0.5, label="Random")
    plt.hist(a, bins=40, density=True, alpha=0.5, label="Real")
    plt.xlabel(title_map[metric])
    plt.ylabel("Density")
    plt.title(f"{label}: Real vs Random {title_map[metric]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def ecdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.sort(arr)
    y = np.arange(1, len(arr) + 1) / len(arr)
    return arr, y


def save_ecdf_plot(
    actual_df: pd.DataFrame,
    random_df: pd.DataFrame,
    label: str,
    metric: str,
    out_path: Path,
) -> None:
    a, r = get_plot_arrays(actual_df, random_df, label, metric)

    if len(a) == 0 or len(r) == 0:
        return

    xa, ya = ecdf(a)
    xr, yr = ecdf(r)

    title_map = {
        "count": "Overlap count",
        "frac": "Overlap fraction",
        "runfrac": "Longest-run fraction",
    }

    plt.figure(figsize=(8, 5))
    plt.plot(xr, yr, label="Random")
    plt.plot(xa, ya, label="Real")
    plt.xlabel(title_map[metric])
    plt.ylabel("ECDF")
    plt.title(f"{label}: ECDF of {title_map[metric]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_enrichment_bar(
    overall_df: pd.DataFrame,
    metric_col: str,
    title: str,
    out_path: Path,
) -> None:
    if metric_col not in overall_df.columns:
        return

    x = overall_df["label"].astype(str)
    y = pd.to_numeric(overall_df[metric_col], errors="coerce")

    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Enrichment")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_empirical_p_hist(
    segment_df: pd.DataFrame,
    p_col: str,
    title: str,
    out_path: Path,
) -> None:
    if p_col not in segment_df.columns:
        return

    vals = pd.to_numeric(segment_df[p_col], errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=40, density=True, alpha=0.8)
    plt.xlabel("Empirical p-value")
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_segment_scatter(
    segment_df: pd.DataFrame,
    label: str,
    metric_kind: str,
    out_path: Path,
) -> None:
    """
    metric_kind:
      frac
      count
      runfrac
    """
    if label == "Any":
        prefix = "Any"
    else:
        prefix = label

    col_map = {
        "count": (f"{prefix}_real_count", f"{prefix}_random_mean_count"),
        "frac": (f"{prefix}_real_frac", f"{prefix}_random_mean_frac"),
        "runfrac": (f"{prefix}_real_runfrac", f"{prefix}_random_mean_runfrac"),
    }

    real_col, rand_col = col_map[metric_kind]
    if real_col not in segment_df.columns or rand_col not in segment_df.columns:
        return

    x = pd.to_numeric(segment_df[rand_col], errors="coerce")
    y = pd.to_numeric(segment_df[real_col], errors="coerce")
    keep = x.notna() & y.notna()
    x = x[keep].to_numpy()
    y = y[keep].to_numpy()

    if len(x) == 0:
        return

    vmax = max(float(np.max(x)), float(np.max(y))) if len(x) else 1.0

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.plot([0, vmax], [0, vmax])
    plt.xlabel("Random mean")
    plt.ylabel("Real")
    plt.title(f"{label}: Real vs random mean ({metric_kind})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_summary_text(
    overall_df: pd.DataFrame,
    out_path: Path,
    focus_labels: List[str],
) -> None:
    lines = []
    lines.append("Plot summary")
    lines.append("=" * 60)

    for label in focus_labels:
        sub = overall_df[overall_df["label"].astype(str) == label]
        if sub.empty:
            continue
        row = sub.iloc[0]

        lines.append(f"[{label}]")
        for key in [
            "real_mean_overlap_count",
            "random_mean_overlap_count",
            "count_mean_enrichment",
            "real_mean_overlap_frac",
            "random_mean_overlap_frac",
            "frac_mean_enrichment",
            "real_mean_longest_run_frac",
            "random_mean_longest_run_frac",
            "runfrac_mean_enrichment",
            "real_hit_rate_ge1",
            "random_hit_rate_ge1",
            "hit_enrichment_ge1",
            "real_hit_rate_ge2",
            "random_hit_rate_ge2",
            "hit_enrichment_ge2",
        ]:
            if key in row.index:
                lines.append(f"{key}: {row[key]}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# 
# =========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Start2Fold/HDX real-vs-random overlap results.")
    p.add_argument("--result-dir", type=Path, default=RESULT_DIR, help="Directory containing step 04 output tables.")
    p.add_argument("--plot-dir", type=Path, default=None, help="Output plot directory. Default: <result-dir>/plots_only")
    p.add_argument("--focus-labels", nargs="+", default=FOCUS_LABELS, help="Labels to plot.")
    p.add_argument("--max-random-points-for-plot", type=int, default=MAX_RANDOM_POINTS_FOR_PLOT, help="Random-point downsampling limit for plotting.")
    p.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed for plot downsampling.")
    return p.parse_args()

def main() -> None:
    global RESULT_DIR, PLOT_DIR, FOCUS_LABELS, MAX_RANDOM_POINTS_FOR_PLOT, RANDOM_SEED
    args = parse_args()
    RESULT_DIR = args.result_dir
    PLOT_DIR = args.plot_dir if args.plot_dir is not None else RESULT_DIR / "plots_only"
    FOCUS_LABELS = args.focus_labels
    MAX_RANDOM_POINTS_FOR_PLOT = args.max_random_points_for_plot
    RANDOM_SEED = args.random_seed
    ensure_dir(PLOT_DIR)

    data = load_required_files(RESULT_DIR)
    actual_df = data["actual"]
    random_df = data["random"]
    overall_df = data["overall"]
    segment_df = data["segment"]

    # 
    if "Map_Status" in actual_df.columns:
        actual_df = actual_df[actual_df["Map_Status"].astype(str) == "OK"].copy()

    generated_files = []

    # 1)  vs  
    for label in FOCUS_LABELS:
        for metric in ["frac", "runfrac"]:
            out1 = PLOT_DIR / f"{label}_{metric}_density.png"
            save_overlay_density_hist(actual_df, random_df, label, metric, out1)
            if out1.exists():
                generated_files.append(out1)

            out2 = PLOT_DIR / f"{label}_{metric}_ecdf.png"
            save_ecdf_plot(actual_df, random_df, label, metric, out2)
            if out2.exists():
                generated_files.append(out2)

    # 2) enrichment 
    enrich_specs = [
        ("frac_mean_enrichment", "Mean overlap-fraction enrichment", "enrichment_overlap_fraction.png"),
        ("runfrac_mean_enrichment", "Mean longest-run-fraction enrichment", "enrichment_longest_run_fraction.png"),
        ("hit_enrichment_ge1", "Hit enrichment (>=1 residue)", "enrichment_hit_ge1.png"),
        ("hit_enrichment_ge2", "Hit enrichment (>=2 residues)", "enrichment_hit_ge2.png"),
        ("count_mean_enrichment", "Mean overlap-count enrichment", "enrichment_overlap_count.png"),
    ]

    for metric_col, title, fname in enrich_specs:
        out = PLOT_DIR / fname
        save_enrichment_bar(overall_df, metric_col, title, out)
        if out.exists():
            generated_files.append(out)

    # 3) p-value 
    p_specs = [
        ("Any_frac_empirical_p_ge", "Any empirical p-values (fraction metric)", "Any_empirical_pvalues_frac.png"),
        ("Fold_EARLY_frac_empirical_p_ge", "Fold_EARLY empirical p-values (fraction metric)", "Fold_EARLY_empirical_pvalues_frac.png"),
        ("Fold_LATE_frac_empirical_p_ge", "Fold_LATE empirical p-values (fraction metric)", "Fold_LATE_empirical_pvalues_frac.png"),
        ("Stab_STRONG_frac_empirical_p_ge", "Stab_STRONG empirical p-values (fraction metric)", "Stab_STRONG_empirical_pvalues_frac.png"),
        ("Stab_MEDIUM_frac_empirical_p_ge", "Stab_MEDIUM empirical p-values (fraction metric)", "Stab_MEDIUM_empirical_pvalues_frac.png"),
    ]

    for p_col, title, fname in p_specs:
        out = PLOT_DIR / fname
        save_empirical_p_hist(segment_df, p_col, title, out)
        if out.exists():
            generated_files.append(out)

    # 4)  real vs random mean 
    scatter_specs = [
        ("Any", "frac"),
        ("Fold_EARLY", "frac"),
        ("Fold_LATE", "frac"),
        ("Stab_STRONG", "frac"),
        ("Stab_MEDIUM", "frac"),
        ("Any", "runfrac"),
        ("Fold_EARLY", "runfrac"),
        ("Fold_LATE", "runfrac"),
        ("Stab_STRONG", "runfrac"),
        ("Stab_MEDIUM", "runfrac"),
    ]

    for label, metric_kind in scatter_specs:
        out = PLOT_DIR / f"{label}_{metric_kind}_real_vs_random_mean_scatter.png"
        save_segment_scatter(segment_df, label, metric_kind, out)
        if out.exists():
            generated_files.append(out)

    # 5)  summary 
    overall_copy = PLOT_DIR / "overall_class_enrichment_summary_copy.csv"
    overall_df.to_csv(overall_copy, index=False)
    generated_files.append(overall_copy)

    segment_copy = PLOT_DIR / "segment_random_test_summary_copy.csv"
    segment_df.to_csv(segment_copy, index=False)
    generated_files.append(segment_copy)

    # 6)  summary
    summary_txt = PLOT_DIR / "plot_summary.txt"
    write_summary_text(overall_df, summary_txt, FOCUS_LABELS)
    generated_files.append(summary_txt)

    print("=" * 60)
    print("[INFO] Plot generation done.")
    print(f"[INFO] Result dir: {RESULT_DIR}")
    print(f"[INFO] Plot dir: {PLOT_DIR}")
    print(f"[INFO] Number of generated files: {len(generated_files)}")
    print("=" * 60)

    for p in generated_files:
        print(p)


if __name__ == "__main__":
    main()