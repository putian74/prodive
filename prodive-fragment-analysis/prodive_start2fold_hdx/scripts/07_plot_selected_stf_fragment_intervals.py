#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# =========================================================
# 
# =========================================================
MULTITYPE_FILE = Path(
    "CHANGE_ME"
    "multitype_analysis_with_random/actual_segment_multitype_annotations.csv"
)

RAW_FILE = Path(
    "CHANGE_ME"
    "all_mapped_segments_with_start2fold_metrics.csv"
)

OUT_DIR = Path(
    "CHANGE_ME"
    "fragment_interval_plots"
)

TARGET_STF_IDS = ["STF0006", "STF0025"]

# ：
#  "type_level_2" / "dominant_class" / "uniform"
COLOR_MODE = "type_level_2"

#  segment_uid（）
SHOW_SEGMENT_LABEL = False

# 
X_PAD = 5


# =========================================================
# 
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def load_input_dataframe() -> pd.DataFrame:
    """
     multitype ；，。
    """
    if MULTITYPE_FILE.exists():
        df = pd.read_csv(MULTITYPE_FILE, dtype=str)
    elif RAW_FILE.exists():
        df = pd.read_csv(RAW_FILE, dtype=str)
    else:
        raise FileNotFoundError(
            f"Neither file exists:\n  {MULTITYPE_FILE}\n  {RAW_FILE}"
        )

    if "Map_Status" in df.columns:
        df = df[df["Map_Status"].astype(str) == "OK"].copy()

    required_cols = ["STF_ID", "Target_Seq_Start", "Target_Seq_End"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Target_Seq_Start"] = safe_int_series(df["Target_Seq_Start"])
    df["Target_Seq_End"] = safe_int_series(df["Target_Seq_End"])
    df = df[df["Target_Seq_Start"].notna() & df["Target_Seq_End"].notna()].copy()

    df["Target_Seq_Start"] = df["Target_Seq_Start"].astype(int)
    df["Target_Seq_End"] = df["Target_Seq_End"].astype(int)

    # ： segment_uid ，
    if "segment_uid" not in df.columns:
        df["segment_uid"] = (
            df["STF_ID"].astype(str)
            + "|"
            + df.get("pfam_id", pd.Series(["NA"] * len(df))).astype(str)
            + "|"
            + df.get("global_row_index_1based", pd.Series(["NA"] * len(df))).astype(str)
            + "|"
            + df.get("segment_rank", pd.Series(["NA"] * len(df))).astype(str)
        )

    return df


def assign_lanes(intervals: List[Tuple[int, int]]) -> List[int]:
    """
    ：
     start >  end，；
    。
    """
    lane_ends: List[int] = []
    lane_ids: List[int] = []

    for start, end in intervals:
        placed = False
        for lane_idx, last_end in enumerate(lane_ends):
            if start > last_end:
                lane_ids.append(lane_idx)
                lane_ends[lane_idx] = end
                placed = True
                break
        if not placed:
            lane_ids.append(len(lane_ends))
            lane_ends.append(end)

    return lane_ids


def build_depth_track(intervals: List[Tuple[int, int]], x_min: int, x_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    。
    """
    xs = np.arange(x_min, x_max + 1, dtype=int)
    depth = np.zeros_like(xs, dtype=int)

    for start, end in intervals:
        depth[(xs >= start) & (xs <= end)] += 1

    return xs, depth


def get_color_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    。
    """
    if COLOR_MODE == "type_level_2" and "type_level_2" in df.columns:
        return {
            "no_hit": "#B0B0B0",
            "folding_only": "#4C78A8",
            "stability_only": "#F58518",
            "folding_stability_mixed": "#54A24B",
        }

    if COLOR_MODE == "dominant_class" and "dominant_class" in df.columns:
        return {
            "Fold_EARLY": "#4C78A8",
            "Fold_INTER": "#72B7B2",
            "Fold_LATE": "#A0CBE8",
            "Stab_STRONG": "#E45756",
            "Stab_MEDIUM": "#F58518",
            "Stab_WEAK": "#FFBF79",
            "ambiguous": "#8E8E8E",
            "no_hit": "#B0B0B0",
        }

    return {"uniform": "#4C78A8"}


def get_segment_color(row: pd.Series, color_map: Dict[str, str]) -> str:
    if COLOR_MODE == "type_level_2" and "type_level_2" in row.index:
        return color_map.get(str(row["type_level_2"]), "#4C78A8")
    if COLOR_MODE == "dominant_class" and "dominant_class" in row.index:
        return color_map.get(str(row["dominant_class"]), "#4C78A8")
    return color_map.get("uniform", "#4C78A8")


def make_title(sub: pd.DataFrame, stf_id: str) -> str:
    n = len(sub)
    seed_names = []
    if "seed_seq_name" in sub.columns:
        seed_names = sorted(set(sub["seed_seq_name"].dropna().astype(str)))
    seed_part = seed_names[0] if len(seed_names) == 1 else f"{len(seed_names)} seed names"
    return f"{stf_id} | {seed_part} | n={n} raw segments"


def plot_one_stf(sub: pd.DataFrame, stf_id: str, out_png: Path, out_pdf: Path) -> None:
    sub = sub.copy()

    sub["seg_start"] = sub["Target_Seq_Start"].astype(int)
    sub["seg_end"] = sub["Target_Seq_End"].astype(int)
    sub["seg_len"] = sub["seg_end"] - sub["seg_start"] + 1

    sub = sub.sort_values(["seg_start", "seg_end", "segment_uid"]).reset_index(drop=True)

    intervals = list(zip(sub["seg_start"].tolist(), sub["seg_end"].tolist()))
    lane_ids = assign_lanes(intervals)
    sub["lane_id"] = lane_ids

    x_min = int(sub["seg_start"].min())
    x_max = int(sub["seg_end"].max())

    #  seed_start / seed_end， seed 
    if "seed_start" in sub.columns:
        seed_start_num = pd.to_numeric(sub["seed_start"], errors="coerce").dropna()
        if len(seed_start_num) > 0:
            x_min = min(x_min, int(seed_start_num.min()))
    if "seed_end" in sub.columns:
        seed_end_num = pd.to_numeric(sub["seed_end"], errors="coerce").dropna()
        if len(seed_end_num) > 0:
            x_max = max(x_max, int(seed_end_num.max()))

    xs, depth = build_depth_track(intervals, x_min, x_max)
    color_map = get_color_map(sub)

    lane_n = int(sub["lane_id"].max()) + 1
    fig_h = max(6.0, 2.2 + 0.28 * lane_n)

    fig = plt.figure(figsize=(16, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 4.2], hspace=0.08)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # -----  coverage depth -----
    ax_top.fill_between(xs, depth, step="mid", alpha=0.6)
    ax_top.plot(xs, depth, linewidth=1.0)
    ax_top.set_ylabel("Depth")
    ax_top.set_title(make_title(sub, stf_id))
    ax_top.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax_top.set_xlim(x_min - X_PAD, x_max + X_PAD)

    # -----  interval tracks -----
    for _, row in sub.iterrows():
        y = row["lane_id"]
        x0 = row["seg_start"]
        width = row["seg_end"] - row["seg_start"] + 1
        color = get_segment_color(row, color_map)

        rect = mpatches.Rectangle(
            (x0, y - 0.35),
            width,
            0.7,
            facecolor=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
        )
        ax_bot.add_patch(rect)

        if SHOW_SEGMENT_LABEL:
            ax_bot.text(
                row["seg_end"] + 0.8,
                y,
                str(row["segment_uid"]),
                fontsize=6,
                va="center",
                ha="left",
            )

    ax_bot.set_ylim(-1, lane_n)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel("Seed sequence coordinate")
    ax_bot.set_ylabel("Packed lane")
    ax_bot.grid(True, axis="x", linestyle="--", alpha=0.25)

    # 
    handles = []
    if COLOR_MODE == "type_level_2" and "type_level_2" in sub.columns:
        legend_keys = ["folding_only", "stability_only", "folding_stability_mixed", "no_hit"]
        handles = [
            mpatches.Patch(color=color_map[k], label=k)
            for k in legend_keys
            if k in set(sub["type_level_2"].astype(str))
        ]
    elif COLOR_MODE == "dominant_class" and "dominant_class" in sub.columns:
        present = list(dict.fromkeys(sub["dominant_class"].astype(str).tolist()))
        handles = [
            mpatches.Patch(color=color_map.get(k, "#4C78A8"), label=k)
            for k in present
        ]
    else:
        handles = [mpatches.Patch(color=color_map["uniform"], label="segment")]

    if handles:
        ax_bot.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_combined(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    targets = [x for x in TARGET_STF_IDS if x in set(df["STF_ID"].astype(str))]
    if not targets:
        return

    fig = plt.figure(figsize=(17, 10))
    outer = fig.add_gridspec(len(targets), 1, hspace=0.28)

    for i, stf_id in enumerate(targets):
        sub = df[df["STF_ID"].astype(str) == stf_id].copy()
        sub["seg_start"] = sub["Target_Seq_Start"].astype(int)
        sub["seg_end"] = sub["Target_Seq_End"].astype(int)
        sub = sub.sort_values(["seg_start", "seg_end", "segment_uid"]).reset_index(drop=True)

        intervals = list(zip(sub["seg_start"].tolist(), sub["seg_end"].tolist()))
        sub["lane_id"] = assign_lanes(intervals)

        x_min = int(sub["seg_start"].min())
        x_max = int(sub["seg_end"].max())
        if "seed_start" in sub.columns:
            s = pd.to_numeric(sub["seed_start"], errors="coerce").dropna()
            if len(s) > 0:
                x_min = min(x_min, int(s.min()))
        if "seed_end" in sub.columns:
            s = pd.to_numeric(sub["seed_end"], errors="coerce").dropna()
            if len(s) > 0:
                x_max = max(x_max, int(s.max()))

        xs, depth = build_depth_track(intervals, x_min, x_max)
        color_map = get_color_map(sub)

        gs = outer[i].subgridspec(2, 1, height_ratios=[1.1, 3.7], hspace=0.06)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

        ax_top.fill_between(xs, depth, step="mid", alpha=0.6)
        ax_top.plot(xs, depth, linewidth=1.0)
        ax_top.set_ylabel("Depth")
        ax_top.set_title(make_title(sub, stf_id))
        ax_top.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax_top.set_xlim(x_min - X_PAD, x_max + X_PAD)

        for _, row in sub.iterrows():
            y = row["lane_id"]
            x0 = row["seg_start"]
            width = row["seg_end"] - row["seg_start"] + 1
            color = get_segment_color(row, color_map)

            rect = mpatches.Rectangle(
                (x0, y - 0.35),
                width,
                0.7,
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
            )
            ax_bot.add_patch(rect)

        ax_bot.set_ylim(-1, int(sub["lane_id"].max()) + 1)
        ax_bot.invert_yaxis()
        ax_bot.set_xlabel("Seed sequence coordinate")
        ax_bot.set_ylabel("Packed lane")
        ax_bot.grid(True, axis="x", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot selected STF fragment intervals and coverage depth.")
    p.add_argument("--multitype-file", type=Path, default=MULTITYPE_FILE, help="actual_segment_multitype_annotations.csv")
    p.add_argument("--raw-file", type=Path, default=RAW_FILE, help="all_mapped_segments_with_start2fold_metrics.csv")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory.")
    p.add_argument("--target-stf-ids", nargs="+", default=TARGET_STF_IDS, help="STF IDs to plot.")
    p.add_argument("--color-mode", choices=["type_level_2", "dominant_class", "uniform"], default=COLOR_MODE, help="Segment coloring mode.")
    p.add_argument("--show-segment-label", action="store_true", help="Show segment_uid labels beside fragments.")
    p.add_argument("--x-pad", type=int, default=X_PAD, help="Horizontal axis padding.")
    return p.parse_args()

def main() -> None:
    global MULTITYPE_FILE, RAW_FILE, OUT_DIR, TARGET_STF_IDS, COLOR_MODE, SHOW_SEGMENT_LABEL, X_PAD
    args = parse_args()
    MULTITYPE_FILE = args.multitype_file
    RAW_FILE = args.raw_file
    OUT_DIR = args.out_dir
    TARGET_STF_IDS = args.target_stf_ids
    COLOR_MODE = args.color_mode
    SHOW_SEGMENT_LABEL = args.show_segment_label
    X_PAD = args.x_pad
    ensure_dir(OUT_DIR)
    df = load_input_dataframe()

    df = df[df["STF_ID"].astype(str).isin(TARGET_STF_IDS)].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for target STF IDs: {TARGET_STF_IDS}")

    # 
    for stf_id in TARGET_STF_IDS:
        sub = df[df["STF_ID"].astype(str) == stf_id].copy()
        if sub.empty:
            print(f"[WARN] No rows for {stf_id}, skipped.")
            continue

        out_png = OUT_DIR / f"{stf_id}_fragment_intervals.png"
        out_pdf = OUT_DIR / f"{stf_id}_fragment_intervals.pdf"
        plot_one_stf(sub, stf_id, out_png, out_pdf)
        print(f"[INFO] Saved: {out_png}")
        print(f"[INFO] Saved: {out_pdf}")

    # 
    combined_png = OUT_DIR / "STF0006_STF0025_fragment_intervals_combined.png"
    combined_pdf = OUT_DIR / "STF0006_STF0025_fragment_intervals_combined.pdf"
    plot_combined(df, combined_png, combined_pdf)
    print(f"[INFO] Saved: {combined_png}")
    print(f"[INFO] Saved: {combined_pdf}")


if __name__ == "__main__":
    main()