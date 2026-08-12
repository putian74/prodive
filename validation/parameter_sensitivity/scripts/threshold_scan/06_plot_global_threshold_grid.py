#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global distributed threshold plotting with additional 3D terrain/surface plots.

What this script does
---------------------
1. Reads per-server local threshold summaries and family-set JSON files.
2. Builds a global summary:
   - kept_candidate_points_global: summed across server-local summaries
   - retained_family_pairs_with_points_global: summed across server-local summaries
   - covered_families_global: union of family sets across server-local JSON files
3. Generates the original plot set:
   - 2D line plots for candidate points, linear and log10
   - 2D line plots for global family coverage
   - 3D line plots for candidate points, linear and log10
   - 3D line plots for global family coverage
   - heatmaps for candidate points and global family coverage
4. Additionally generates terrain/surface-style 3D plots:
   - raw candidate-point terrain surface
   - log10(candidate-point) terrain surface

No hidden subset or sampling is used.
"""

import os
import argparse
import re
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ==============================================================================
# Runtime configuration. Values are set from command-line arguments in main().
# ==============================================================================
ROOT_DIR = None
FAMILY_SET_DIR_PATTERN = "*_family_set_extract_local_mp"
OUT_DIR = None
LOCAL_ID_JSON_NAME = "candidate_family_id_sets_local.json"
LOCAL_NAME_JSON_NAME = "candidate_family_sets_local.json"
LOCAL_SUMMARY_NAME = "candidate_family_summary_local.csv"
WRITE_GLOBAL_LONG_CSV = False
FIGSIZE_2D = (11, 7)
FIGSIZE_3D = (11, 8)
FIGSIZE_HEATMAP = (10, 7)
PLOT_DPI = 600
ELEV = 25
AZIM = -135
TERRAIN_ELEV = 32
TERRAIN_AZIM = -135
TARGET_TICK_COUNT = 7
FORCE_AXIS_START_AT_ZERO = True
NICE_MULTIPLIERS = (1, 2, 5, 10)

# Output paths are initialized in configure_from_args().

# ==============================================================================
# Helper functions
# ==============================================================================

def comma_format(x, pos=None):
    if not np.isfinite(x):
        return ""
    return f"{x:,.0f}"


def human_format(x, pos=None):
    if not np.isfinite(x):
        return ""

    x = float(x)

    if abs(x) >= 1e9:
        v = x / 1e9
        return f"{v:.0f}B" if abs(v - round(v)) < 1e-9 else f"{v:.1f}B"

    if abs(x) >= 1e6:
        v = x / 1e6
        return f"{v:.0f}M" if abs(v - round(v)) < 1e-9 else f"{v:.1f}M"

    if abs(x) >= 1e3:
        v = x / 1e3
        return f"{v:.0f}K" if abs(v - round(v)) < 1e-9 else f"{v:.1f}K"

    return f"{x:.0f}"


def plain_float_format(x, pos=None):
    if not np.isfinite(x):
        return ""
    return f"{x:.2f}"


def integer_format(x, pos=None):
    if not np.isfinite(x):
        return ""
    return f"{int(round(x))}"


def nice_step(raw_step):
    raw_step = float(raw_step)

    if not np.isfinite(raw_step) or raw_step <= 0:
        return 1.0

    exponent = np.floor(np.log10(raw_step))
    base = 10 ** exponent

    for m in NICE_MULTIPLIERS:
        step = m * base
        if step >= raw_step:
            return float(step)

    return float(10 * base)


def build_nice_ticks(values, target_tick_count=TARGET_TICK_COUNT, force_zero=FORCE_AXIS_START_AT_ZERO):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.array([0, 1], dtype=float), 0.0, 1.0

    data_min = float(np.min(arr))
    data_max = float(np.max(arr))

    if force_zero:
        lower = 0.0
    else:
        span = data_max - data_min
        pad = max(span * 0.05, 1.0)
        lower = max(0.0, data_min - pad) if data_min >= 0 else data_min - pad

    if data_max <= lower:
        upper = lower + 1.0
        return np.array([lower, upper], dtype=float), lower, upper

    raw_step = (data_max - lower) / max(target_tick_count - 1, 1)
    step = nice_step(raw_step)

    upper = np.ceil(data_max / step) * step

    if upper <= lower:
        upper = lower + step

    ticks = np.arange(lower, upper + step * 0.5, step, dtype=float)

    while ticks.size > 12:
        step = nice_step(step * 1.5)
        upper = np.ceil(data_max / step) * step
        ticks = np.arange(lower, upper + step * 0.5, step, dtype=float)

    return ticks, lower, upper


def apply_nice_linear_yaxis(ax, values, target_tick_count=TARGET_TICK_COUNT, force_zero=FORCE_AXIS_START_AT_ZERO):
    ticks, lower, upper = build_nice_ticks(
        values,
        target_tick_count=target_tick_count,
        force_zero=force_zero,
    )

    ax.set_ylim(lower, upper)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(comma_format))


def apply_scientific_linear_yaxis(ax, values, target_tick_count=TARGET_TICK_COUNT, force_zero=FORCE_AXIS_START_AT_ZERO):
    ticks, lower, upper = build_nice_ticks(
        values,
        target_tick_count=target_tick_count,
        force_zero=force_zero,
    )

    ax.set_ylim(lower, upper)
    ax.set_yticks(ticks)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def apply_scientific_linear_zaxis(ax, values, target_tick_count=TARGET_TICK_COUNT, force_zero=FORCE_AXIS_START_AT_ZERO):
    ticks, lower, upper = build_nice_ticks(
        values,
        target_tick_count=target_tick_count,
        force_zero=force_zero,
    )

    ax.set_zlim(lower, upper)
    ax.set_zticks(ticks)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    ax.zaxis.set_major_formatter(formatter)


def apply_nice_linear_zaxis(ax, values, target_tick_count=TARGET_TICK_COUNT, force_zero=FORCE_AXIS_START_AT_ZERO):
    ticks, lower, upper = build_nice_ticks(
        values,
        target_tick_count=target_tick_count,
        force_zero=force_zero,
    )

    ax.set_zlim(lower, upper)
    ax.set_zticks(ticks)
    ax.zaxis.set_major_formatter(FuncFormatter(comma_format))


def apply_integer_log10_zaxis(ax, values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        ax.set_zlim(0, 1)
        ax.set_zticks([0, 1])
        ax.zaxis.set_major_formatter(FuncFormatter(integer_format))
        return

    zmin = int(np.floor(np.min(arr)))
    zmax = int(np.ceil(np.max(arr)))

    if zmax <= zmin:
        zmax = zmin + 1

    ticks = np.arange(zmin, zmax + 1, 1, dtype=int)

    ax.set_zlim(zmin, zmax)
    ax.set_zticks(ticks)
    ax.zaxis.set_major_formatter(FuncFormatter(integer_format))


def apply_integer_log10_yaxis(ax, values):
    """
    For log10-transformed y-values, force integer y ticks:
      4, 5, 6, 7, 8, ...
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.yaxis.set_major_formatter(FuncFormatter(integer_format))
        return

    ymin = int(np.floor(np.min(arr)))
    ymax = int(np.ceil(np.max(arr)))

    if ymax <= ymin:
        ymax = ymin + 1

    ticks = np.arange(ymin, ymax + 1, 1, dtype=int)

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(integer_format))


def style_3d_tick_labels(ax):
    """
    Make 3D tick labels less crowded.
    This is especially useful for second-threshold labels such as 8, 8.5, ..., 15.
    """
    ax.tick_params(axis="x", labelsize=8, pad=2)
    ax.tick_params(axis="y", labelsize=8, pad=2)
    ax.tick_params(axis="z", labelsize=8, pad=4)

    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_horizontalalignment("right")

    for label in ax.get_yticklabels():
        label.set_rotation(35)
        label.set_horizontalalignment("right")


def tag_float(x):
    return f"{float(x):.1f}".replace(".", "p")


def parse_combo_key(key):
    m = re.match(r"^F([0-9]+p[0-9]+)_S([0-9]+p[0-9]+)$", str(key))

    if not m:
        return None

    first = float(m.group(1).replace("p", "."))
    second = float(m.group(2).replace("p", "."))

    return first, second


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def save_fig(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved: {out_path}")


def find_family_set_dirs(root_dir):
    dirs = sorted(
        d for d in glob.glob(os.path.join(root_dir, FAMILY_SET_DIR_PATTERN))
        if os.path.isdir(d)
    )

    if not dirs:
        raise RuntimeError(
            f"No family-set directories found under: {root_dir}\n"
            f"Expected pattern: {FAMILY_SET_DIR_PATTERN}"
        )

    return dirs


def load_local_family_sets(local_dir):
    id_json = os.path.join(local_dir, LOCAL_ID_JSON_NAME)
    name_json = os.path.join(local_dir, LOCAL_NAME_JSON_NAME)

    if os.path.exists(id_json):
        with open(id_json, "r", encoding="utf-8") as f:
            obj = json.load(f)

        result = {
            str(k): set(str(x) for x in v)
            for k, v in obj.items()
        }

        return result, "id"

    if os.path.exists(name_json):
        with open(name_json, "r", encoding="utf-8") as f:
            obj = json.load(f)

        result = {
            str(k): set(str(x) for x in v)
            for k, v in obj.items()
        }

        return result, "name"

    raise FileNotFoundError(
        f"Neither {LOCAL_ID_JSON_NAME} nor {LOCAL_NAME_JSON_NAME} found in {local_dir}"
    )


def load_local_summary(local_dir):
    summary_path = os.path.join(local_dir, LOCAL_SUMMARY_NAME)

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Local summary not found: {summary_path}")

    df = read_csv_safely(summary_path)
    df = normalize_columns(df)

    required = {
        "combo_key",
        "first_threshold",
        "second_threshold",
        "kept_candidate_points_local",
        "retained_family_pairs_with_points_local",
        "covered_families_local",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns in {summary_path}: {missing}"
        )

    df["combo_key"] = df["combo_key"].astype(str)

    numeric_cols = [
        "first_threshold",
        "second_threshold",
        "kept_candidate_points_local",
        "retained_family_pairs_with_points_local",
        "covered_families_local",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["combo_key", "first_threshold", "second_threshold"]).copy()

    return df


# ==============================================================================
# Merge distributed local results
# ==============================================================================

def build_global_summary():
    family_dirs = find_family_set_dirs(ROOT_DIR)

    print("=" * 100)
    print("[INFO] Found family-set directories:")

    for d in family_dirs:
        print(f"  - {d}")

    print("=" * 100)

    global_family_sets = defaultdict(set)

    points_by_combo = defaultdict(int)
    pairs_by_combo = defaultdict(int)
    local_family_count_sum_by_combo = defaultdict(int)

    first_second_by_combo = {}
    source_dirs_by_combo = defaultdict(list)

    json_source_types = []

    for local_dir in family_dirs:
        print(f"[INFO] Reading local directory: {local_dir}")

        local_summary = load_local_summary(local_dir)
        local_family_sets, source_type = load_local_family_sets(local_dir)
        json_source_types.append(source_type)

        for row in local_summary.itertuples(index=False):
            key = str(getattr(row, "combo_key"))
            first = float(getattr(row, "first_threshold"))
            second = float(getattr(row, "second_threshold"))

            points = int(getattr(row, "kept_candidate_points_local"))
            pairs = int(getattr(row, "retained_family_pairs_with_points_local"))
            local_families = int(getattr(row, "covered_families_local"))

            points_by_combo[key] += points
            pairs_by_combo[key] += pairs
            local_family_count_sum_by_combo[key] += local_families
            first_second_by_combo[key] = (first, second)
            source_dirs_by_combo[key].append(local_dir)

        for key, fams in local_family_sets.items():
            global_family_sets[key].update(fams)

            if key not in first_second_by_combo:
                parsed = parse_combo_key(key)

                if parsed is not None:
                    first_second_by_combo[key] = parsed

    all_combo_keys = sorted(
        set(first_second_by_combo.keys())
        | set(global_family_sets.keys())
        | set(points_by_combo.keys())
        | set(pairs_by_combo.keys()),
        key=lambda k: (
            first_second_by_combo.get(k, parse_combo_key(k) or (999, 999))[0],
            first_second_by_combo.get(k, parse_combo_key(k) or (999, 999))[1],
        )
    )

    rows = []

    for key in all_combo_keys:
        fs = first_second_by_combo.get(key)

        if fs is None:
            parsed = parse_combo_key(key)

            if parsed is None:
                print(f"[WARN] Cannot parse combo key, skipped: {key}")
                continue

            fs = parsed

        first, second = fs

        global_family_n = len(global_family_sets.get(key, set()))
        points = int(points_by_combo.get(key, 0))
        pairs = int(pairs_by_combo.get(key, 0))
        local_family_sum = int(local_family_count_sum_by_combo.get(key, 0))

        rows.append({
            "combo_key": key,
            "first_threshold": float(first),
            "second_threshold": float(second),
            "kept_candidate_points_global": points,
            "retained_family_pairs_with_points_global": pairs,
            "covered_families_global": global_family_n,
            "covered_families_local_sum_diagnostic": local_family_sum,
            "family_count_overcount_if_summed": local_family_sum - global_family_n,
            "log10_kept_candidate_points_global": (
                float(np.log10(points)) if points > 0 else np.nan
            ),
            "num_source_dirs": len(set(source_dirs_by_combo.get(key, []))),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("Global summary is empty.")

    df = df.sort_values(
        by=["first_threshold", "second_threshold"],
        ascending=[True, True],
    ).reset_index(drop=True)

    max_family = int(df["covered_families_global"].max())
    df["family_loss_from_max"] = max_family - df["covered_families_global"]

    df.to_csv(
        OUT_GLOBAL_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[DONE] Global summary saved: {OUT_GLOBAL_SUMMARY_CSV}")
    print(f"[INFO] Maximum global covered families: {max_family}")

    global_family_sets_for_json = {
        key: sorted(list(global_family_sets.get(key, set())))
        for key in all_combo_keys
        if key in global_family_sets
    }

    with open(OUT_GLOBAL_FAMILY_SET_JSON, "w", encoding="utf-8") as f:
        json.dump(global_family_sets_for_json, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Global family-set JSON saved: {OUT_GLOBAL_FAMILY_SET_JSON}")

    if WRITE_GLOBAL_LONG_CSV:
        with open(OUT_GLOBAL_LONG_CSV, "w", encoding="utf-8-sig") as f:
            f.write("combo_key,first_threshold,second_threshold,family\n")

            for key in all_combo_keys:
                fs = first_second_by_combo.get(key) or parse_combo_key(key)

                if fs is None:
                    continue

                first, second = fs

                for fam in sorted(global_family_sets.get(key, set())):
                    f.write(f"{key},{first},{second},{fam}\n")

        print(f"[DONE] Global long family CSV saved: {OUT_GLOBAL_LONG_CSV}")

    return df, max_family, family_dirs, sorted(set(json_source_types))


# ==============================================================================
# 2D plots
# ==============================================================================

def subset_float(df, col, value):
    return df[np.isclose(df[col].astype(float), float(value))].copy()


def plot_2d_points_x_first_by_second_linear(df, first_values_asc, second_values):
    fig, ax = plt.subplots(figsize=FIGSIZE_2D)

    all_y = []

    for second in second_values:
        sub = subset_float(df, "second_threshold", second).sort_values("first_threshold")

        y = sub["kept_candidate_points_global"].to_numpy(dtype=float)
        all_y.extend(y[np.isfinite(y)].tolist())

        ax.plot(
            sub["first_threshold"],
            sub["kept_candidate_points_global"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=f"Second={second:g}",
        )

    ax.set_title("Global candidate points vs first threshold, linear scale")
    ax.set_xlabel("First threshold")
    ax.set_ylabel("Global kept candidate points")

    apply_scientific_linear_yaxis(ax, all_y)

    ax.set_xticks(first_values_asc)
    ax.set_xticklabels([f"{x:g}" for x in first_values_asc], rotation=45, ha="right")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="Second threshold", fontsize=8, ncol=2)

    save_fig(fig, OUT_POINTS_X_FIRST_BY_SECOND_LINEAR)


def plot_2d_points_x_second_by_first_linear(df, first_values_desc, second_values):
    fig, ax = plt.subplots(figsize=FIGSIZE_2D)

    all_y = []

    for first in first_values_desc:
        sub = subset_float(df, "first_threshold", first).sort_values("second_threshold")

        y = sub["kept_candidate_points_global"].to_numpy(dtype=float)
        all_y.extend(y[np.isfinite(y)].tolist())

        ax.plot(
            sub["second_threshold"],
            sub["kept_candidate_points_global"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=f"First={first:g}",
        )

    ax.set_title("Global candidate points vs second threshold, linear scale")
    ax.set_xlabel("Second threshold")
    ax.set_ylabel("Global kept candidate points")

    apply_scientific_linear_yaxis(ax, all_y)

    ax.set_xticks(second_values)
    ax.set_xticklabels([f"{x:g}" for x in second_values], rotation=45, ha="right")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="First threshold", fontsize=8, ncol=2)

    save_fig(fig, OUT_POINTS_X_SECOND_BY_FIRST_LINEAR)


def plot_2d_points_x_first_by_second_log(df, first_values_asc, second_values):
    """
    Plot log10(candidate points) directly.

    This is different from plotting raw points on a log-scale axis.
    Here the y values themselves are log10-transformed numbers.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_2D)

    all_y = []

    for second in second_values:
        sub = subset_float(df, "second_threshold", second).sort_values("first_threshold").copy()

        y = pd.to_numeric(
            sub["log10_kept_candidate_points_global"],
            errors="coerce"
        ).to_numpy(dtype=float)

        all_y.extend(y[np.isfinite(y)].tolist())

        ax.plot(
            sub["first_threshold"],
            y,
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=f"Second={second:g}",
        )

    ax.set_title("Global candidate points vs first threshold, log10 scale")
    ax.set_xlabel("First threshold")
    ax.set_ylabel("log10(global kept candidate points)")

    apply_integer_log10_yaxis(ax, all_y)

    ax.set_xticks(first_values_asc)
    ax.set_xticklabels([f"{x:g}" for x in first_values_asc], rotation=45, ha="right")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="Second threshold", fontsize=8, ncol=2)

    save_fig(fig, OUT_POINTS_X_FIRST_BY_SECOND_LOG)


def plot_2d_points_x_second_by_first_log(df, first_values_desc, second_values):
    """
    Plot log10(candidate points) directly.

    This is different from plotting raw points on a log-scale axis.
    Here the y values themselves are log10-transformed numbers.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_2D)

    all_y = []

    for first in first_values_desc:
        sub = subset_float(df, "first_threshold", first).sort_values("second_threshold").copy()

        y = pd.to_numeric(
            sub["log10_kept_candidate_points_global"],
            errors="coerce"
        ).to_numpy(dtype=float)

        all_y.extend(y[np.isfinite(y)].tolist())

        ax.plot(
            sub["second_threshold"],
            y,
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=f"First={first:g}",
        )

    ax.set_title("Global candidate points vs second threshold, log10 scale")
    ax.set_xlabel("Second threshold")
    ax.set_ylabel("log10(global kept candidate points)")

    apply_integer_log10_yaxis(ax, all_y)

    ax.set_xticks(second_values)
    ax.set_xticklabels([f"{x:g}" for x in second_values], rotation=45, ha="right")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="First threshold", fontsize=8, ncol=2)

    save_fig(fig, OUT_POINTS_X_SECOND_BY_FIRST_LOG)


def plot_2d_families_x_first_by_second(df, first_values_asc, second_values):
    fig, ax = plt.subplots(figsize=FIGSIZE_2D)

    all_y = []

    for second in second_values:
        sub = subset_float(df, "second_threshold", second).sort_values("first_threshold")

        y = sub["covered_families_global"].to_numpy(dtype=float)
        all_y.extend(y[np.isfinite(y)].tolist())

        ax.plot(
            sub["first_threshold"],
            sub["covered_families_global"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=f"Second={second:g}",
        )

    ax.set_title("Global involved families vs first threshold")
    ax.set_xlabel("First threshold")
    ax.set_ylabel("Global involved families, union count")

    apply_nice_linear_yaxis(ax, all_y)

    ax.set_xticks(first_values_asc)
    ax.set_xticklabels([f"{x:g}" for x in first_values_asc], rotation=45, ha="right")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="Second threshold", fontsize=8, ncol=2)

    save_fig(fig, OUT_FAMILIES_X_FIRST_BY_SECOND)


def plot_2d_families_x_second_by_first(df, first_values_desc, second_values):
    fig, ax = plt.subplots(figsize=FIGSIZE_2D)

    all_y = []

    for first in first_values_desc:
        sub = subset_float(df, "first_threshold", first).sort_values("second_threshold")

        y = sub["covered_families_global"].to_numpy(dtype=float)
        all_y.extend(y[np.isfinite(y)].tolist())

        ax.plot(
            sub["second_threshold"],
            sub["covered_families_global"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=f"First={first:g}",
        )

    ax.set_title("Global involved families vs second threshold")
    ax.set_xlabel("Second threshold")
    ax.set_ylabel("Global involved families, union count")

    apply_nice_linear_yaxis(ax, all_y)

    ax.set_xticks(second_values)
    ax.set_xticklabels([f"{x:g}" for x in second_values], rotation=45, ha="right")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="First threshold", fontsize=8, ncol=2)

    save_fig(fig, OUT_FAMILIES_X_SECOND_BY_FIRST)


# ==============================================================================
# 3D line plots
# ==============================================================================

def plot_3d_lines_fixed_second(
    df,
    value_col,
    zlabel,
    title,
    out_path,
    z_mode="linear",
):
    """
    Fixed-second lines.

    Coordinates are always:
      x = first_threshold
      y = second_threshold
      z = value_col

    Important:
      We draw larger second thresholds first and smaller second thresholds later.
      This makes low-second-threshold curves such as S=8 less likely to be visually
      covered by other lines in Matplotlib's 3D rendering.
    """
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    second_values = sorted(df["second_threshold"].unique())
    first_values = sorted(df["first_threshold"].unique())

    plot_order = sorted(second_values, reverse=True)

    all_z = []
    legend_items = []

    min_second = min(second_values)

    for second in plot_order:
        sub = subset_float(df, "second_threshold", second).sort_values("first_threshold")

        x = sub["first_threshold"].to_numpy(dtype=float)
        y = np.full_like(x, fill_value=second, dtype=float)
        z = sub[value_col].to_numpy(dtype=float)

        all_z.extend(z[np.isfinite(z)].tolist())

        is_outer = np.isclose(second, min_second)

        line_objs = ax.plot(
            x,
            y,
            z,
            marker="o",
            linewidth=2.4 if is_outer else 1.6,
            markersize=5 if is_outer else 4,
            alpha=1.0 if is_outer else 0.88,
            label=f"S={second:g}",
        )

        legend_items.append((second, line_objs[0]))

    ax.set_title(title, pad=18)
    ax.set_xlabel("First threshold", labelpad=10)
    ax.set_ylabel("Second threshold", labelpad=12)
    ax.set_zlabel(zlabel, labelpad=10)

    ax.set_xticks(first_values)
    ax.set_xticklabels([f"{x:g}" for x in first_values], rotation=25, ha="right")

    ax.set_yticks(second_values)
    ax.set_yticklabels([f"{y:g}" for y in second_values], rotation=35, ha="right")

    if z_mode == "log10_integer":
        apply_integer_log10_zaxis(ax, all_z)
    elif z_mode == "scientific":
        apply_scientific_linear_zaxis(ax, all_z)
    else:
        apply_nice_linear_zaxis(ax, all_z)

    style_3d_tick_labels(ax)

    ax.view_init(elev=ELEV, azim=AZIM)

    legend_items = sorted(legend_items, key=lambda x: x[0])
    handles = [x[1] for x in legend_items]
    labels = [f"S={x[0]:g}" for x in legend_items]

    ax.legend(
        handles,
        labels,
        title="Fixed second",
        fontsize=7,
        ncol=2,
        loc="upper left",
    )

    save_fig(fig, out_path)


def plot_3d_lines_fixed_first(
    df,
    value_col,
    zlabel,
    title,
    out_path,
    z_mode="linear",
):
    """
    Fixed-first lines.

    Coordinates are always:
      x = first_threshold
      y = second_threshold
      z = value_col

    This avoids swapping x/y between different 3D plots.
    """
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    first_values = sorted(df["first_threshold"].unique())
    second_values = sorted(df["second_threshold"].unique())

    plot_order = sorted(first_values)

    all_z = []
    legend_items = []

    for first in plot_order:
        sub = subset_float(df, "first_threshold", first).sort_values("second_threshold")

        x = np.full(len(sub), fill_value=first, dtype=float)
        y = sub["second_threshold"].to_numpy(dtype=float)
        z = sub[value_col].to_numpy(dtype=float)

        all_z.extend(z[np.isfinite(z)].tolist())

        line_objs = ax.plot(
            x,
            y,
            z,
            marker="o",
            linewidth=1.8,
            markersize=4,
            alpha=0.9,
            label=f"F={first:g}",
        )

        legend_items.append((first, line_objs[0]))

    ax.set_title(title, pad=18)
    ax.set_xlabel("First threshold", labelpad=10)
    ax.set_ylabel("Second threshold", labelpad=12)
    ax.set_zlabel(zlabel, labelpad=10)

    ax.set_xticks(first_values)
    ax.set_xticklabels([f"{x:g}" for x in first_values], rotation=25, ha="right")

    ax.set_yticks(second_values)
    ax.set_yticklabels([f"{y:g}" for y in second_values], rotation=35, ha="right")

    if z_mode == "log10_integer":
        apply_integer_log10_zaxis(ax, all_z)
    elif z_mode == "scientific":
        apply_scientific_linear_zaxis(ax, all_z)
    else:
        apply_nice_linear_zaxis(ax, all_z)

    style_3d_tick_labels(ax)

    ax.view_init(elev=ELEV, azim=AZIM)

    legend_items = sorted(legend_items, key=lambda x: x[0])
    handles = [x[1] for x in legend_items]
    labels = [f"F={x[0]:g}" for x in legend_items]

    ax.legend(
        handles,
        labels,
        title="Fixed first",
        fontsize=7,
        ncol=2,
        loc="upper left",
    )

    save_fig(fig, out_path)


# ==============================================================================
# Heatmaps and grids
# ==============================================================================

def make_grid(df, value_col, first_values_asc, second_values):
    grid = np.full(
        (len(first_values_asc), len(second_values)),
        np.nan,
        dtype=float,
    )

    for i, first in enumerate(first_values_asc):
        for j, second in enumerate(second_values):
            sub = df[
                np.isclose(df["first_threshold"], first)
                & np.isclose(df["second_threshold"], second)
            ]

            if not sub.empty:
                grid[i, j] = float(sub.iloc[0][value_col])

    return grid


def make_mesh(first_values_asc, second_values):
    x_grid = np.repeat(
        np.asarray(first_values_asc, dtype=float)[:, None],
        len(second_values),
        axis=1,
    )

    y_grid = np.repeat(
        np.asarray(second_values, dtype=float)[None, :],
        len(first_values_asc),
        axis=0,
    )

    return x_grid, y_grid


def set_integer_colorbar_ticks(cbar, grid):
    arr = np.asarray(grid, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return

    vmin = int(np.floor(np.min(arr)))
    vmax = int(np.ceil(np.max(arr)))

    if vmax <= vmin:
        vmax = vmin + 1

    ticks = np.arange(vmin, vmax + 1, 1, dtype=int)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(integer_format))


def plot_3d_terrain_surface(
    df,
    value_col,
    zlabel,
    title,
    out_path,
    z_mode="linear",
    cmap="viridis",
    draw_base_contours=False,
):
    """
    Cleaner 3D surface plot.

    Coordinates are fixed as:
      x = first_threshold
      y = second_threshold
      z = value_col

    Notes:
      - The old version drew projected contour lines on the floor plane. Those
        lines can look messy in 3D figures, so they are disabled by default.
      - For candidate points, the log10 surface is usually more interpretable
        than the linear surface because the raw counts span several orders of
        magnitude.
    """
    first_values_asc = sorted(df["first_threshold"].unique())
    second_values = sorted(df["second_threshold"].unique())

    grid = make_grid(df, value_col, first_values_asc, second_values)
    x_grid, y_grid = make_mesh(first_values_asc, second_values)

    finite_z = grid[np.isfinite(grid)]

    if finite_z.size == 0:
        print(f"[WARN] No finite values for terrain plot: {out_path}")
        return

    zmin = float(np.nanmin(finite_z))
    zmax = float(np.nanmax(finite_z))

    fig = plt.figure(figsize=(12.2, 8.6))
    ax = fig.add_subplot(111, projection="3d")

    # Orthographic projection reduces perspective distortion and usually makes
    # threshold landscapes easier to read.
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    surface = ax.plot_surface(
        x_grid,
        y_grid,
        grid,
        cmap=cmap,
        rstride=1,
        cstride=1,
        linewidth=0.16,
        antialiased=True,
        edgecolor=(1.0, 1.0, 1.0, 0.28),
        alpha=0.98,
        shade=True,
    )

    # Optional floor contour projection. This was the set of lines visible at
    # the bottom of the previous figure. It is useful for topographic reading,
    # but off by default because it can make the plot look cluttered.
    if draw_base_contours:
        try:
            if z_mode == "log10_integer":
                levels = np.arange(
                    int(np.floor(zmin)),
                    int(np.ceil(zmax)) + 1,
                    1,
                )
            else:
                levels = 10

            ax.contour(
                x_grid,
                y_grid,
                grid,
                zdir="z",
                offset=zmin,
                levels=levels,
                linewidths=0.55,
                colors="0.35",
                alpha=0.35,
            )
        except Exception as e:
            print(f"[WARN] Could not draw contour projection for {out_path}: {e}")

    ax.set_title(title, pad=18, fontsize=15)
    ax.set_xlabel("First threshold F", labelpad=11)
    ax.set_ylabel("Second threshold S", labelpad=13)
    ax.set_zlabel(zlabel, labelpad=11)

    ax.set_xticks(first_values_asc)
    ax.set_xticklabels([f"{x:g}" for x in first_values_asc], rotation=20, ha="right")

    ax.set_yticks(second_values)
    ax.set_yticklabels([f"{y:g}" for y in second_values], rotation=28, ha="right")

    if z_mode == "log10_integer":
        apply_integer_log10_zaxis(ax, finite_z)
    elif z_mode == "scientific":
        apply_scientific_linear_zaxis(ax, finite_z)
    else:
        apply_nice_linear_zaxis(ax, finite_z)

    style_3d_tick_labels(ax)

    # Make the surface less like a vertical wall. This changes only the visual
    # aspect ratio, not the underlying values.
    try:
        ax.set_box_aspect((1.35, 1.05, 0.62))
    except Exception:
        pass

    # Cleaner panes. Keep light gridlines, remove the heavy grey background.
    try:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))
    except Exception:
        pass

    ax.view_init(elev=TERRAIN_ELEV, azim=TERRAIN_AZIM)

    cbar = fig.colorbar(surface, ax=ax, shrink=0.58, pad=0.08, aspect=18)
    cbar.set_label(zlabel)

    if z_mode == "log10_integer":
        cmin = int(np.floor(zmin))
        cmax = int(np.ceil(zmax))

        if cmax <= cmin:
            cmax = cmin + 1

        cbar.set_ticks(np.arange(cmin, cmax + 1, 1, dtype=int))
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(integer_format))

    elif z_mode == "scientific":
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_major_formatter(formatter)

    else:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(human_format))

    save_fig(fig, out_path)

def plot_heatmap(
    df,
    value_col,
    title,
    colorbar_label,
    out_path,
    fmt="{:.0f}",
    use_human_annotation=False,
    use_human_colorbar=True,
    integer_colorbar=False,
):
    first_values_asc = sorted(df["first_threshold"].unique())
    second_values = sorted(df["second_threshold"].unique())

    grid = make_grid(df, value_col, first_values_asc, second_values)

    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
    )

    ax.set_title(title)
    ax.set_xlabel("Second threshold")
    ax.set_ylabel("First threshold")

    ax.set_xticks(np.arange(len(second_values)))
    ax.set_xticklabels([f"{x:g}" for x in second_values], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(first_values_asc)))
    ax.set_yticklabels([f"{x:g}" for x in first_values_asc])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    if integer_colorbar:
        set_integer_colorbar_ticks(cbar, grid)
    elif use_human_colorbar:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(human_format))
    else:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(plain_float_format))

    if grid.size <= 250:
        finite_vals = grid[np.isfinite(grid)]

        if finite_vals.size > 0:
            vmin = np.nanmin(finite_vals)
            vmax = np.nanmax(finite_vals)
            threshold = vmin + 0.55 * (vmax - vmin)

            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    val = grid[i, j]

                    if not np.isfinite(val):
                        continue

                    text_color = "white" if val > threshold else "black"

                    if use_human_annotation:
                        label = human_format(val)
                    else:
                        label = fmt.format(val)

                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=text_color,
                    )

    save_fig(fig, out_path)


# ==============================================================================
# Summary
# ==============================================================================

def write_plot_summary(df, max_family, family_dirs, json_source_types):
    cols = [
        "combo_key",
        "first_threshold",
        "second_threshold",
        "covered_families_global",
        "family_loss_from_max",
        "kept_candidate_points_global",
        "log10_kept_candidate_points_global",
        "retained_family_pairs_with_points_global",
        "covered_families_local_sum_diagnostic",
        "family_count_overcount_if_summed",
        "num_source_dirs",
    ]

    with open(OUT_PLOT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("=== Global distributed threshold plot summary ===\n\n")
        f.write(f"ROOT_DIR: {ROOT_DIR}\n")
        f.write(f"OUT_DIR: {OUT_DIR}\n\n")

        f.write("Input family-set directories:\n")

        for d in family_dirs:
            f.write(f"  - {d}\n")

        f.write("\n")
        f.write(f"Family-set JSON source types used: {json_source_types}\n")
        f.write(f"Valid parameter combinations: {len(df)}\n")
        f.write(f"Maximum global involved families: {max_family}\n\n")

        f.write(
            "Important note:\n"
            "  kept_candidate_points_global and retained_family_pairs_with_points_global are summed across servers.\n"
            "  covered_families_global is computed by union of family sets across servers.\n"
            "  covered_families_local_sum_diagnostic is only diagnostic and should not be used as the real family count.\n"
            "  Candidate-point plots are exported in linear, log10, 3D-line, heatmap, and terrain-surface versions.\n"
            "  All 3D plots use fixed axes: x = first_threshold, y = second_threshold, z = metric.\n"
            "  No loss figures are generated in this version.\n\n"
        )

        f.write("Combinations reaching maximum global family count, sorted by fewer candidate points:\n\n")

        top = (
            df[df["covered_families_global"] == max_family]
            .sort_values("kept_candidate_points_global", ascending=True)
            .copy()
        )

        if top.empty:
            f.write("No combination reaches the maximum family count.\n\n")
        else:
            f.write(top[cols].head(60).to_string(index=False))
            f.write("\n\n")

        f.write("Smallest candidate-point combinations overall:\n\n")
        smallest = df.sort_values("kept_candidate_points_global", ascending=True).copy()
        f.write(smallest[cols].head(60).to_string(index=False))
        f.write("\n\n")

        f.write("Largest overcount if local family counts were naively summed:\n\n")
        over = df.sort_values("family_count_overcount_if_summed", ascending=False).copy()
        f.write(over[cols].head(60).to_string(index=False))
        f.write("\n")

    print(f"[DONE] Plot summary saved: {OUT_PLOT_SUMMARY_TXT}")


# ==============================================================================
# Main
# ==============================================================================



def parse_args():
    parser = argparse.ArgumentParser(
        description="Build global threshold summaries and plots from distributed family-set results."
    )
    parser.add_argument("--root-dir", required=True,
                        help="Directory containing server-local family-set extraction directories.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family-set-dir-pattern", default="*_family_set_extract_local_mp")
    parser.add_argument("--local-id-json-name", default="candidate_family_id_sets_local.json")
    parser.add_argument("--local-name-json-name", default="candidate_family_sets_local.json")
    parser.add_argument("--local-summary-name", default="candidate_family_summary_local.csv")
    parser.add_argument("--write-global-long-csv", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--make-family-terrain", action="store_true")
    return parser.parse_args()


def configure_from_args(args):
    global ROOT_DIR, OUT_DIR, FAMILY_SET_DIR_PATTERN, LOCAL_ID_JSON_NAME, LOCAL_NAME_JSON_NAME
    global LOCAL_SUMMARY_NAME, WRITE_GLOBAL_LONG_CSV, PLOT_DPI, MAKE_FAMILY_TERRAIN
    global OUT_GLOBAL_SUMMARY_CSV, OUT_GLOBAL_FAMILY_SET_JSON, OUT_GLOBAL_LONG_CSV, OUT_PLOT_SUMMARY_TXT
    global OUT_POINTS_X_FIRST_BY_SECOND_LINEAR, OUT_POINTS_X_SECOND_BY_FIRST_LINEAR
    global OUT_POINTS_X_FIRST_BY_SECOND_LOG, OUT_POINTS_X_SECOND_BY_FIRST_LOG
    global OUT_FAMILIES_X_FIRST_BY_SECOND, OUT_FAMILIES_X_SECOND_BY_FIRST
    global OUT_3D_POINTS_FIXED_SECOND_LINEAR, OUT_3D_POINTS_FIXED_FIRST_LINEAR
    global OUT_3D_POINTS_FIXED_SECOND_LOG, OUT_3D_POINTS_FIXED_FIRST_LOG
    global OUT_3D_TERRAIN_POINTS_LINEAR, OUT_3D_TERRAIN_POINTS_LOG
    global OUT_3D_FAMILIES_FIXED_SECOND, OUT_3D_FAMILIES_FIXED_FIRST, OUT_3D_TERRAIN_FAMILIES
    global OUT_HEATMAP_POINTS_LINEAR, OUT_HEATMAP_POINTS_LOG, OUT_HEATMAP_FAMILIES
    ROOT_DIR = args.root_dir
    OUT_DIR = args.output_dir
    FAMILY_SET_DIR_PATTERN = args.family_set_dir_pattern
    LOCAL_ID_JSON_NAME = args.local_id_json_name
    LOCAL_NAME_JSON_NAME = args.local_name_json_name
    LOCAL_SUMMARY_NAME = args.local_summary_name
    WRITE_GLOBAL_LONG_CSV = args.write_global_long_csv
    PLOT_DPI = args.dpi
    MAKE_FAMILY_TERRAIN = args.make_family_terrain
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_GLOBAL_SUMMARY_CSV = os.path.join(OUT_DIR, "global_candidate_family_union_summary.csv")
    OUT_GLOBAL_FAMILY_SET_JSON = os.path.join(OUT_DIR, "global_candidate_family_sets_union.json")
    OUT_GLOBAL_LONG_CSV = os.path.join(OUT_DIR, "global_candidate_families_long.csv")
    OUT_PLOT_SUMMARY_TXT = os.path.join(OUT_DIR, "plot_summary.txt")
    OUT_POINTS_X_FIRST_BY_SECOND_LINEAR = os.path.join(OUT_DIR, "line2d_points_x_first_by_second_linear_sci_y.png")
    OUT_POINTS_X_SECOND_BY_FIRST_LINEAR = os.path.join(OUT_DIR, "line2d_points_x_second_by_first_linear_sci_y.png")
    OUT_POINTS_X_FIRST_BY_SECOND_LOG = os.path.join(OUT_DIR, "line2d_points_x_first_by_second_log_y.png")
    OUT_POINTS_X_SECOND_BY_FIRST_LOG = os.path.join(OUT_DIR, "line2d_points_x_second_by_first_log_y.png")
    OUT_FAMILIES_X_FIRST_BY_SECOND = os.path.join(OUT_DIR, "line2d_global_families_x_first_by_second_linear.png")
    OUT_FAMILIES_X_SECOND_BY_FIRST = os.path.join(OUT_DIR, "line2d_global_families_x_second_by_first_linear.png")
    OUT_3D_POINTS_FIXED_SECOND_LINEAR = os.path.join(OUT_DIR, "line3d_points_fixed_second_x_first_y_second_linear_sci_z.png")
    OUT_3D_POINTS_FIXED_FIRST_LINEAR = os.path.join(OUT_DIR, "line3d_points_fixed_first_x_first_y_second_linear_sci_z.png")
    OUT_3D_POINTS_FIXED_SECOND_LOG = os.path.join(OUT_DIR, "line3d_log10_points_fixed_second_x_first_y_second_integer_z.png")
    OUT_3D_POINTS_FIXED_FIRST_LOG = os.path.join(OUT_DIR, "line3d_log10_points_fixed_first_x_first_y_second_integer_z.png")
    OUT_3D_TERRAIN_POINTS_LINEAR = os.path.join(OUT_DIR, "surface3d_candidate_points_global_terrain_linear_sci_z.png")
    OUT_3D_TERRAIN_POINTS_LOG = os.path.join(OUT_DIR, "surface3d_log10_candidate_points_global_terrain_integer_z.png")
    OUT_3D_FAMILIES_FIXED_SECOND = os.path.join(OUT_DIR, "line3d_global_families_fixed_second_x_first_y_second.png")
    OUT_3D_FAMILIES_FIXED_FIRST = os.path.join(OUT_DIR, "line3d_global_families_fixed_first_x_first_y_second.png")
    OUT_3D_TERRAIN_FAMILIES = os.path.join(OUT_DIR, "surface3d_global_families_terrain.png")
    OUT_HEATMAP_POINTS_LINEAR = os.path.join(OUT_DIR, "heatmap_candidate_points_global_linear.png")
    OUT_HEATMAP_POINTS_LOG = os.path.join(OUT_DIR, "heatmap_log10_candidate_points_global_integer_colorbar.png")
    OUT_HEATMAP_FAMILIES = os.path.join(OUT_DIR, "heatmap_global_covered_families.png")

def main():
    args = parse_args()
    configure_from_args(args)

    print("=" * 100)
    print("Global distributed threshold plotting")
    print("Family counts are computed by set union across server-local family-set JSON files.")
    print("Candidate points and retained family pairs are summed across server-local summary CSV files.")
    print("Candidate-point plots will be generated in linear, log10, 3D-line, heatmap, and terrain-surface versions.")
    print("No loss figures will be generated.")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    print("=" * 100)

    df, max_family, family_dirs, json_source_types = build_global_summary()

    # Ensure log10 column exists even if input format changes later.
    if "log10_kept_candidate_points_global" not in df.columns:
        df["log10_kept_candidate_points_global"] = np.where(
            df["kept_candidate_points_global"] > 0,
            np.log10(df["kept_candidate_points_global"]),
            np.nan,
        )

    print(f"[INFO] Valid rows: {len(df)}")
    print(f"[INFO] Maximum global involved families: {max_family}")

    first_values_asc = sorted(df["first_threshold"].unique())
    first_values_desc = sorted(df["first_threshold"].unique(), reverse=True)
    second_values = sorted(df["second_threshold"].unique())

    print(f"[INFO] First thresholds asc:  {first_values_asc}")
    print(f"[INFO] First thresholds desc: {first_values_desc}")
    print(f"[INFO] Second thresholds:     {second_values}")

    # -------------------------
    # 2D candidate-point outputs: linear + log10-value plots
    # -------------------------
    plot_2d_points_x_first_by_second_linear(df, first_values_asc, second_values)
    plot_2d_points_x_second_by_first_linear(df, first_values_desc, second_values)

    plot_2d_points_x_first_by_second_log(df, first_values_asc, second_values)
    plot_2d_points_x_second_by_first_log(df, first_values_desc, second_values)

    # -------------------------
    # 3D candidate-point line outputs: linear + log10
    # -------------------------
    plot_3d_lines_fixed_second(
        df=df,
        value_col="kept_candidate_points_global",
        zlabel="Global candidate points",
        title="3D line plot: global candidate points, fixed second threshold",
        out_path=OUT_3D_POINTS_FIXED_SECOND_LINEAR,
        z_mode="scientific",
    )

    plot_3d_lines_fixed_first(
        df=df,
        value_col="kept_candidate_points_global",
        zlabel="Global candidate points",
        title="3D line plot: global candidate points, fixed first threshold",
        out_path=OUT_3D_POINTS_FIXED_FIRST_LINEAR,
        z_mode="scientific",
    )

    plot_3d_lines_fixed_second(
        df=df,
        value_col="log10_kept_candidate_points_global",
        zlabel="log10(global candidate points)",
        title="3D line plot: log10(global candidate points), fixed second threshold",
        out_path=OUT_3D_POINTS_FIXED_SECOND_LOG,
        z_mode="log10_integer",
    )

    plot_3d_lines_fixed_first(
        df=df,
        value_col="log10_kept_candidate_points_global",
        zlabel="log10(global candidate points)",
        title="3D line plot: log10(global candidate points), fixed first threshold",
        out_path=OUT_3D_POINTS_FIXED_FIRST_LOG,
        z_mode="log10_integer",
    )

    # -------------------------
    # NEW: 3D terrain/surface candidate-point outputs
    # -------------------------
    plot_3d_terrain_surface(
        df=df,
        value_col="kept_candidate_points_global",
        zlabel="Global candidate points",
        title="3D surface: global candidate points (linear scale)",
        out_path=OUT_3D_TERRAIN_POINTS_LINEAR,
        z_mode="scientific",
        cmap="viridis",
        draw_base_contours=False,
    )

    plot_3d_terrain_surface(
        df=df,
        value_col="log10_kept_candidate_points_global",
        zlabel="log10(global candidate points)",
        title="3D surface: log10(global candidate points)",
        out_path=OUT_3D_TERRAIN_POINTS_LOG,
        z_mode="log10_integer",
        cmap="viridis",
        draw_base_contours=False,
    )

    # -------------------------
    # 3D family outputs only
    # -------------------------
    plot_3d_lines_fixed_second(
        df=df,
        value_col="covered_families_global",
        zlabel="Global involved families",
        title="3D line plot: global involved families, fixed second threshold",
        out_path=OUT_3D_FAMILIES_FIXED_SECOND,
        z_mode="linear",
    )

    plot_3d_lines_fixed_first(
        df=df,
        value_col="covered_families_global",
        zlabel="Global involved families",
        title="3D line plot: global involved families, fixed first threshold",
        out_path=OUT_3D_FAMILIES_FIXED_FIRST,
        z_mode="linear",
    )

    if MAKE_FAMILY_TERRAIN:
        plot_3d_terrain_surface(
            df=df,
            value_col="covered_families_global",
            zlabel="Global involved families",
            title="3D surface: global involved families",
            out_path=OUT_3D_TERRAIN_FAMILIES,
            z_mode="linear",
            cmap="viridis",
            draw_base_contours=False,
        )

    # -------------------------
    # 2D family outputs only
    # -------------------------
    plot_2d_families_x_first_by_second(df, first_values_asc, second_values)
    plot_2d_families_x_second_by_first(df, first_values_desc, second_values)

    # -------------------------
    # Heatmaps
    # -------------------------
    plot_heatmap(
        df=df,
        value_col="kept_candidate_points_global",
        title="Heatmap: global kept candidate points",
        colorbar_label="Global kept candidate points",
        out_path=OUT_HEATMAP_POINTS_LINEAR,
        use_human_annotation=True,
        use_human_colorbar=True,
    )

    plot_heatmap(
        df=df,
        value_col="log10_kept_candidate_points_global",
        title="Heatmap: log10(global kept candidate points)",
        colorbar_label="log10(global kept candidate points)",
        out_path=OUT_HEATMAP_POINTS_LOG,
        fmt="{:.2f}",
        use_human_annotation=False,
        use_human_colorbar=False,
        integer_colorbar=True,
    )

    plot_heatmap(
        df=df,
        value_col="covered_families_global",
        title="Heatmap: global involved families",
        colorbar_label="Global involved families",
        out_path=OUT_HEATMAP_FAMILIES,
        fmt="{:.0f}",
        use_human_annotation=False,
        use_human_colorbar=True,
    )

    write_plot_summary(df, max_family, family_dirs, json_source_types)

    print("=" * 100)
    print("[DONE] All plots finished.")
    print(f"Output directory: {OUT_DIR}")
    print(f"Terrain surface plot, linear: {OUT_3D_TERRAIN_POINTS_LINEAR}")
    print(f"Terrain surface plot, log10:  {OUT_3D_TERRAIN_POINTS_LOG}")
    print("=" * 100)


if __name__ == "__main__":
    main()
