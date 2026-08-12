#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Raw-count logistic derivative-extrema analysis with multiprocessing.

Purpose
-------
This script reads per-combo score-threshold response curves:

    per_combo_score_cutoff_curves/*/*_score_cutoff_xaxis_curve.csv

and fits each curve with a bounded decreasing Richards/logistic function.
It then calculates the first and second numerical derivatives on the fitted
curve and records the most important second-derivative extrema.

Key change from the previous version
------------------------------------
The previous script used NORMALIZE_Y=True, so fitted_y and derivative values
were in 0-1 normalized units. This version uses raw counts by default:

    retained_rows       -> actual row counts
    covered_families    -> actual family counts

Therefore the output y-axis, fitted_y, first_derivative, and second_derivative
are all in the original count scale.

Main outputs
------------
    logistic_second_derivative_extrema_long.csv
        All local second-derivative extrema.

    logistic_second_derivative_global_summary.csv
        For each combo and metric, the global second-derivative min/max points.

    dense_logistic_curves/*.csv
        Dense fitted curves with raw-count fitted_y, first_derivative, and
        second_derivative.

    plots_logistic_second_derivative/*.png
        Per-combo diagnostic plots.
"""

import os
import argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import glob
import traceback
from multiprocessing import get_context

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from tqdm import tqdm


# ==============================================================================
# 1. Runtime configuration. Values are set from command-line arguments in main().
# ==============================================================================
INPUT_CURVE_ROOT = None
OUTPUT_ROOT = None
DENSE_CURVE_DIR = None
PLOT_DIR = None
EXTREMA_LONG_CSV = None
GLOBAL_SUMMARY_CSV = None
FAILED_TXT = None
Y_COLUMNS = ["retained_rows", "covered_families"]
NORMALIZE_Y = False
DENSE_POINTS = 5000
PROMINENCE_MIN = 0.0
MIN_PEAK_DISTANCE_POINTS = 5
MAKE_PLOTS = True
PLOT_DPI = 300
FIGSIZE = (12, 7.5)
WORKERS = 20
CHUNKSIZE = 1
MAXTASKSPERCHILD = 20
MIN_Y_RANGE_TO_FIT = 1e-9

# ==============================================================================
# 2. Model
# ==============================================================================

def richards_decreasing(x, bottom, top, x0, scale, nu):
    """
    Decreasing 5-parameter logistic/Richards curve.

    y = bottom + (top - bottom) / (1 + exp((x - x0) / scale))^nu
    """
    z = np.clip((x - x0) / scale, -80, 80)
    return bottom + (top - bottom) / ((1.0 + np.exp(z)) ** nu)


# ==============================================================================
# 3. Helper functions
# ==============================================================================

def parse_combo_key_from_path(path):
    base = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    text = base + " " + parent

    m = re.search(r"(F[0-9]+p[0-9]+_S[0-9]+p[0-9]+)", text)
    if not m:
        return None, np.nan, np.nan

    combo_key = m.group(1)

    m2 = re.match(r"^F([0-9]+p[0-9]+)_S([0-9]+p[0-9]+)$", combo_key)
    if not m2:
        return combo_key, np.nan, np.nan

    first_threshold = float(m2.group(1).replace("p", "."))
    second_threshold = float(m2.group(2).replace("p", "."))

    return combo_key, first_threshold, second_threshold


def clean_xy(df, x_col, y_col):
    sub = df[[x_col, y_col]].copy()

    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")

    sub = sub.dropna(subset=[x_col, y_col])
    sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col])]

    if sub.empty:
        raise ValueError(f"No valid data for {x_col}, {y_col}")

    # If there are duplicated score thresholds, average them.
    sub = sub.groupby(x_col, as_index=False)[y_col].mean()
    sub = sub.sort_values(x_col).reset_index(drop=True)

    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)

    if len(x) < 10:
        raise ValueError(f"Too few points for fitting: {len(x)}")

    y_range = float(np.nanmax(y) - np.nanmin(y))
    if y_range <= MIN_Y_RANGE_TO_FIT:
        raise ValueError(f"Nearly flat curve for {y_col}; y_range={y_range}")

    return x, y


def normalize_y_values(y):
    y = np.asarray(y, dtype=float)
    baseline = float(y[0])

    if baseline <= 0 or not np.isfinite(baseline):
        raise ValueError(f"Invalid baseline for normalization: {baseline}")

    return y / baseline, baseline


def estimate_initial_params(x, y):
    head_n = max(5, len(y) // 10)
    tail_n = max(5, len(y) // 10)

    bottom0 = float(np.nanmin(y[-tail_n:]))
    top0 = float(np.nanmax(y[:head_n]))

    if not np.isfinite(bottom0) or not np.isfinite(top0):
        raise ValueError("Invalid initial bottom/top")

    if top0 <= bottom0:
        top0 = float(np.nanmax(y))
        bottom0 = float(np.nanmin(y))

    half = bottom0 + 0.5 * (top0 - bottom0)
    idx = int(np.argmin(np.abs(y - half)))
    x0 = float(x[idx])

    # Conservative starting width; the optimizer can adjust.
    scale0 = 0.08
    nu0 = 1.0

    return [bottom0, top0, x0, scale0, nu0]


def fit_logistic_curve(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    p0 = estimate_initial_params(x, y)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_span = max(y_max - y_min, 1.0)

    if NORMALIZE_Y:
        lower_bounds = [-0.05, 0.80, x_min, 0.005, 0.10]
        upper_bounds = [0.30, 1.20, x_max, 1.50, 10.0]
    else:
        # Raw-count fitting. These bounds keep the fit monotonic and positive-ish,
        # but avoid forcing the bottom to exactly 0.
        lower_bounds = [
            max(-0.05 * y_span, -0.05 * y_max),  # bottom
            max(0.50 * y_max, 1.0),              # top
            x_min,                                # x0
            0.005,                                # scale
            0.10,                                 # nu
        ]
        upper_bounds = [
            max(1.10 * y_max, y_max + 1.0),       # bottom
            max(1.50 * y_max, y_max + 1.0),       # top
            x_max,                                # x0
            1.50,                                 # scale
            10.0,                                 # nu
        ]

        # Ensure p0 lies inside the bounds.
        p0 = [
            float(np.clip(p0[0], lower_bounds[0] + 1e-9, upper_bounds[0] - 1e-9)),
            float(np.clip(p0[1], lower_bounds[1] + 1e-9, upper_bounds[1] - 1e-9)),
            float(np.clip(p0[2], lower_bounds[2] + 1e-9, upper_bounds[2] - 1e-9)),
            float(np.clip(p0[3], lower_bounds[3] + 1e-9, upper_bounds[3] - 1e-9)),
            float(np.clip(p0[4], lower_bounds[4] + 1e-9, upper_bounds[4] - 1e-9)),
        ]

    popt, pcov = curve_fit(
        richards_decreasing,
        x,
        y,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=50000,
    )

    return popt, pcov


def numerical_derivatives(dense_x, fitted_y):
    d1 = np.gradient(fitted_y, dense_x)
    d2 = np.gradient(d1, dense_x)
    return d1, d2


def find_all_second_derivative_extrema(dense_x, fitted_y, d1, d2):
    valid = (
        np.isfinite(dense_x)
        & np.isfinite(fitted_y)
        & np.isfinite(d1)
        & np.isfinite(d2)
    )

    valid_indices = np.where(valid)[0]
    if valid_indices.size == 0:
        return []

    d2_valid = d2[valid]

    max_idx_valid, max_props = find_peaks(
        d2_valid,
        prominence=PROMINENCE_MIN,
        distance=MIN_PEAK_DISTANCE_POINTS,
    )

    min_idx_valid, min_props = find_peaks(
        -d2_valid,
        prominence=PROMINENCE_MIN,
        distance=MIN_PEAK_DISTANCE_POINTS,
    )

    records = []

    for rank, idx_valid in enumerate(min_idx_valid, start=1):
        original_idx = int(valid_indices[idx_valid])
        prominence = np.nan
        if "prominences" in min_props:
            prominence = float(min_props["prominences"][rank - 1])

        records.append({
            "extremum_type": "second_derivative_local_min",
            "extremum_rank_within_type": rank,
            "score_threshold": float(dense_x[original_idx]),
            "fitted_y": float(fitted_y[original_idx]),
            "first_derivative": float(d1[original_idx]),
            "second_derivative": float(d2[original_idx]),
            "abs_second_derivative": float(abs(d2[original_idx])),
            "prominence": prominence,
            "dense_index": original_idx,
        })

    for rank, idx_valid in enumerate(max_idx_valid, start=1):
        original_idx = int(valid_indices[idx_valid])
        prominence = np.nan
        if "prominences" in max_props:
            prominence = float(max_props["prominences"][rank - 1])

        records.append({
            "extremum_type": "second_derivative_local_max",
            "extremum_rank_within_type": rank,
            "score_threshold": float(dense_x[original_idx]),
            "fitted_y": float(fitted_y[original_idx]),
            "first_derivative": float(d1[original_idx]),
            "second_derivative": float(d2[original_idx]),
            "abs_second_derivative": float(abs(d2[original_idx])),
            "prominence": prominence,
            "dense_index": original_idx,
        })

    records = sorted(
        records,
        key=lambda r: (
            r["score_threshold"],
            r["extremum_type"],
        ),
    )

    return records


def summarize_global_second_derivative(dense_x, fitted_y, d1, d2):
    valid = (
        np.isfinite(dense_x)
        & np.isfinite(fitted_y)
        & np.isfinite(d1)
        & np.isfinite(d2)
    )

    idx_candidates = np.where(valid)[0]
    if idx_candidates.size == 0:
        raise ValueError("No valid dense derivative points")

    idx_min = idx_candidates[np.argmin(d2[valid])]
    idx_max = idx_candidates[np.argmax(d2[valid])]

    return {
        "global_second_derivative_min_score_threshold": float(dense_x[idx_min]),
        "global_second_derivative_min_value": float(d2[idx_min]),
        "fitted_y_at_global_second_derivative_min": float(fitted_y[idx_min]),
        "first_derivative_at_global_second_derivative_min": float(d1[idx_min]),

        "global_second_derivative_max_score_threshold": float(dense_x[idx_max]),
        "global_second_derivative_max_value": float(d2[idx_max]),
        "fitted_y_at_global_second_derivative_max": float(fitted_y[idx_max]),
        "first_derivative_at_global_second_derivative_max": float(d1[idx_max]),
    }


def format_count_axis(x, _pos=None):
    if not np.isfinite(x):
        return ""
    abs_x = abs(x)
    if abs_x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs_x >= 1_000:
        return f"{x/1_000:.0f}k"
    if abs_x >= 10:
        return f"{x:.0f}"
    return f"{x:.2g}"


def plot_result(
    combo_key,
    y_col,
    x_raw,
    y_raw,
    dense_x,
    fitted_y,
    d2,
    extrema_records,
    out_png,
):
    fig, ax1 = plt.subplots(figsize=FIGSIZE)

    ax1.plot(
        x_raw,
        y_raw,
        marker="o",
        linestyle="none",
        markersize=3,
        alpha=0.35,
        label="Original curve points",
    )

    ax1.plot(
        dense_x,
        fitted_y,
        linewidth=2.0,
        label="Bounded logistic fit",
    )

    ax1.set_xlabel("Score threshold")
    ax1.set_ylabel(y_col if not NORMALIZE_Y else y_col + " normalized")
    ax1.grid(True, linestyle=":", alpha=0.5)

    if not NORMALIZE_Y:
        ax1.yaxis.set_major_formatter(FuncFormatter(format_count_axis))

    ax2 = ax1.twinx()

    ax2.plot(
        dense_x,
        d2,
        linewidth=1.5,
        linestyle="--",
        label="Second derivative",
    )

    ax2.set_ylabel("Second derivative d²y/d(threshold)²")
    if not NORMALIZE_Y:
        ax2.yaxis.set_major_formatter(FuncFormatter(format_count_axis))

    min_points = [
        r for r in extrema_records
        if r["extremum_type"] == "second_derivative_local_min"
    ]

    max_points = [
        r for r in extrema_records
        if r["extremum_type"] == "second_derivative_local_max"
    ]

    if min_points:
        ax2.scatter(
            [r["score_threshold"] for r in min_points],
            [r["second_derivative"] for r in min_points],
            marker="v",
            s=40,
            label="Local minima of second derivative",
            zorder=5,
        )

    if max_points:
        ax2.scatter(
            [r["score_threshold"] for r in max_points],
            [r["second_derivative"] for r in max_points],
            marker="^",
            s=40,
            label="Local maxima of second derivative",
            zorder=5,
        )

    y_scale_label = "raw counts" if not NORMALIZE_Y else "normalized y"
    ax1.set_title(
        f"{combo_key}: logistic fit and second-derivative extrema for {y_col} ({y_scale_label})"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        loc="best",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 4. Per-file analysis
# ==============================================================================

def analyze_one_curve_csv(csv_path):
    combo_key, first_threshold, second_threshold = parse_combo_key_from_path(csv_path)

    if combo_key is None:
        raise ValueError(f"Cannot parse combo key from path: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    all_extrema_records = []
    global_records = []

    for y_col in Y_COLUMNS:
        if y_col not in df.columns:
            raise ValueError(f"Missing column {y_col} in {csv_path}")

        # Some older outputs may still call this score_cutoff. Keep the input
        # column name for compatibility, but write new outputs as score_threshold.
        x_raw, y_raw_original = clean_xy(df, "score_cutoff", y_col)

        if NORMALIZE_Y:
            y_raw, baseline_y = normalize_y_values(y_raw_original)
        else:
            y_raw = y_raw_original.copy()
            baseline_y = float(y_raw_original[0])

        popt, pcov = fit_logistic_curve(x_raw, y_raw)

        dense_x = np.linspace(float(np.min(x_raw)), float(np.max(x_raw)), DENSE_POINTS)
        fitted_y = richards_decreasing(dense_x, *popt)
        d1, d2 = numerical_derivatives(dense_x, fitted_y)

        dense_csv = os.path.join(
            DENSE_CURVE_DIR,
            f"{combo_key}_{y_col}_dense_logistic_second_derivative.csv",
        )

        dense_df = pd.DataFrame({
            "combo_key": combo_key,
            "first_threshold": first_threshold,
            "second_threshold": second_threshold,
            "metric": y_col,
            "score_threshold_dense": dense_x,
            "fitted_y": fitted_y,
            "first_derivative": d1,
            "second_derivative": d2,
            "normalized_y": NORMALIZE_Y,
            "baseline_y": baseline_y,
            "fit_bottom": popt[0],
            "fit_top": popt[1],
            "fit_x0": popt[2],
            "fit_scale": popt[3],
            "fit_nu": popt[4],
            "y_units": "raw_counts" if not NORMALIZE_Y else "baseline_fraction",
        })

        dense_df.to_csv(dense_csv, index=False, encoding="utf-8-sig")

        extrema_records = find_all_second_derivative_extrema(
            dense_x=dense_x,
            fitted_y=fitted_y,
            d1=d1,
            d2=d2,
        )

        for r in extrema_records:
            rec = {
                "combo_key": combo_key,
                "first_threshold": first_threshold,
                "second_threshold": second_threshold,
                "metric": y_col,
                "normalized_y": NORMALIZE_Y,
                "y_units": "raw_counts" if not NORMALIZE_Y else "baseline_fraction",
                "baseline_y": baseline_y,
                "input_csv": csv_path,
                "dense_derivative_csv": dense_csv,
                "fit_bottom": popt[0],
                "fit_top": popt[1],
                "fit_x0": popt[2],
                "fit_scale": popt[3],
                "fit_nu": popt[4],
            }
            rec.update(r)
            all_extrema_records.append(rec)

        global_summary = summarize_global_second_derivative(
            dense_x=dense_x,
            fitted_y=fitted_y,
            d1=d1,
            d2=d2,
        )

        global_rec = {
            "combo_key": combo_key,
            "first_threshold": first_threshold,
            "second_threshold": second_threshold,
            "metric": y_col,
            "normalized_y": NORMALIZE_Y,
            "y_units": "raw_counts" if not NORMALIZE_Y else "baseline_fraction",
            "baseline_y": baseline_y,
            "input_csv": csv_path,
            "dense_derivative_csv": dense_csv,
            "fit_bottom": popt[0],
            "fit_top": popt[1],
            "fit_x0": popt[2],
            "fit_scale": popt[3],
            "fit_nu": popt[4],
        }
        global_rec.update(global_summary)
        global_records.append(global_rec)

        if MAKE_PLOTS:
            out_png = os.path.join(
                PLOT_DIR,
                f"{combo_key}_{y_col}_logistic_second_derivative_extrema_raw_counts.png",
            )

            plot_result(
                combo_key=combo_key,
                y_col=y_col,
                x_raw=x_raw,
                y_raw=y_raw,
                dense_x=dense_x,
                fitted_y=fitted_y,
                d2=d2,
                extrema_records=extrema_records,
                out_png=out_png,
            )

    return all_extrema_records, global_records


def worker_analyze_one_curve_csv(csv_path):
    try:
        extrema_records, global_records = analyze_one_curve_csv(csv_path)
        return {
            "ok": True,
            "csv_path": csv_path,
            "extrema_records": extrema_records,
            "global_records": global_records,
            "error": "",
            "traceback": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "csv_path": csv_path,
            "extrema_records": [],
            "global_records": [],
            "error": str(e),
            "traceback": traceback.format_exc(limit=8),
        }


# ==============================================================================
# 5. Main analysis
# ==============================================================================



def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit score-cutoff response curves and report logistic second-derivative extrema."
    )
    parser.add_argument("--input-curve-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--y-columns", nargs="+", default=["retained_rows", "covered_families"])
    parser.add_argument("--normalize-y", action="store_true")
    parser.add_argument("--dense-points", type=int, default=5000)
    parser.add_argument("--prominence-min", type=float, default=0.0)
    parser.add_argument("--min-peak-distance-points", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--max-tasks-per-child", type=int, default=20)
    parser.add_argument("--min-y-range-to-fit", type=float, default=1e-9)
    return parser.parse_args()


def configure_from_args(args):
    global INPUT_CURVE_ROOT, OUTPUT_ROOT, DENSE_CURVE_DIR, PLOT_DIR
    global EXTREMA_LONG_CSV, GLOBAL_SUMMARY_CSV, FAILED_TXT, Y_COLUMNS
    global NORMALIZE_Y, DENSE_POINTS, PROMINENCE_MIN, MIN_PEAK_DISTANCE_POINTS
    global MAKE_PLOTS, PLOT_DPI, WORKERS, CHUNKSIZE, MAXTASKSPERCHILD, MIN_Y_RANGE_TO_FIT
    INPUT_CURVE_ROOT = args.input_curve_root
    OUTPUT_ROOT = args.output_root
    DENSE_CURVE_DIR = os.path.join(OUTPUT_ROOT, "dense_logistic_curves")
    PLOT_DIR = os.path.join(OUTPUT_ROOT, "plots_logistic_second_derivative")
    EXTREMA_LONG_CSV = os.path.join(OUTPUT_ROOT, "logistic_second_derivative_extrema_long.csv")
    GLOBAL_SUMMARY_CSV = os.path.join(OUTPUT_ROOT, "logistic_second_derivative_global_summary.csv")
    FAILED_TXT = os.path.join(OUTPUT_ROOT, "failed_logistic_second_derivative_files.txt")
    Y_COLUMNS = list(args.y_columns)
    NORMALIZE_Y = args.normalize_y
    DENSE_POINTS = args.dense_points
    PROMINENCE_MIN = args.prominence_min
    MIN_PEAK_DISTANCE_POINTS = args.min_peak_distance_points
    MAKE_PLOTS = not args.no_plots
    PLOT_DPI = args.dpi
    WORKERS = args.workers
    CHUNKSIZE = args.chunksize
    MAXTASKSPERCHILD = args.max_tasks_per_child
    MIN_Y_RANGE_TO_FIT = args.min_y_range_to_fit

def main():
    args = parse_args()
    configure_from_args(args)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(DENSE_CURVE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 100)
    print("Bounded logistic second-derivative extrema analysis, raw-count y-axis, multiprocessing")
    print(f"INPUT_CURVE_ROOT: {INPUT_CURVE_ROOT}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"NORMALIZE_Y: {NORMALIZE_Y}")
    print(f"Y units: {'raw_counts' if not NORMALIZE_Y else 'baseline_fraction'}")
    print(f"WORKERS: {WORKERS}")
    print(f"CHUNKSIZE: {CHUNKSIZE}")
    print(f"PROMINENCE_MIN: {PROMINENCE_MIN}")
    print("=" * 100)

    pattern = os.path.join(
        INPUT_CURVE_ROOT,
        "*",
        "*_score_cutoff_xaxis_curve.csv",
    )

    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        raise RuntimeError(f"No curve CSV files found with pattern: {pattern}")

    print(f"[INFO] Found curve CSV files: {len(csv_files)}")

    all_extrema_records = []
    all_global_records = []
    failed = []

    ctx = get_context("fork")
    with ctx.Pool(
        processes=WORKERS,
        maxtasksperchild=MAXTASKSPERCHILD,
    ) as pool:
        iterator = pool.imap_unordered(
            worker_analyze_one_curve_csv,
            csv_files,
            chunksize=CHUNKSIZE,
        )

        for res in tqdm(iterator, total=len(csv_files), desc="Fitting curves", unit="csv"):
            if res["ok"]:
                all_extrema_records.extend(res["extrema_records"])
                all_global_records.extend(res["global_records"])
            else:
                failed.append((res["csv_path"], res["error"], res["traceback"]))

    if all_extrema_records:
        extrema_df = pd.DataFrame(all_extrema_records)
        extrema_df = extrema_df.sort_values(
            by=[
                "metric",
                "combo_key",
                "score_threshold",
                "extremum_type",
            ],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    else:
        extrema_df = pd.DataFrame()

    extrema_df.to_csv(
        EXTREMA_LONG_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    if all_global_records:
        global_df = pd.DataFrame(all_global_records)
        global_df = global_df.sort_values(
            by=[
                "metric",
                "combo_key",
            ],
            ascending=[True, True],
        ).reset_index(drop=True)
    else:
        global_df = pd.DataFrame()

    global_df.to_csv(
        GLOBAL_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    if failed:
        with open(FAILED_TXT, "w", encoding="utf-8") as f:
            for path, msg, tb in failed:
                f.write(path + "\n")
                f.write(msg + "\n")
                f.write(tb + "\n")
                f.write("-" * 100 + "\n")

    print("=" * 100)
    print("[DONE] Logistic second-derivative extrema analysis finished.")
    print(f"All local extrema CSV: {EXTREMA_LONG_CSV}")
    print(f"Global extrema summary CSV: {GLOBAL_SUMMARY_CSV}")
    print(f"Dense logistic curves: {DENSE_CURVE_DIR}")
    print(f"Plots: {PLOT_DIR}")
    if failed:
        print(f"Failed files: {FAILED_TXT}")
        print(f"Failed count: {len(failed)}")
    print("=" * 100)

    if not extrema_df.empty:
        display_cols = [
            "combo_key",
            "metric",
            "extremum_type",
            "score_threshold",
            "second_derivative",
            "first_derivative",
            "fitted_y",
            "prominence",
            "fit_x0",
            "fit_scale",
            "fit_nu",
        ]
        existing_cols = [c for c in display_cols if c in extrema_df.columns]
        print(extrema_df[existing_cols].head(100).to_string(index=False))


if __name__ == "__main__":
    main()
