#!/usr/bin/env python3
"""
Classify ProDive percentile-subset paths as Novel or Exist according to overlap with parsed HHsearch hits.

This script intentionally preserves the original overlap logic and output schema.

Inputs
------
1. hhsuite{threshold}.csv from 01_parse_hhr_to_csv.py
   Required columns:
       Main_HMM, Sub_HMM, Sub_Segments_Details

2. subset_data_Top_*.csv from 02_make_percentile_subsets.py
   Required columns:
       Main_HMM, Sub_HMM, Main_Segment, Sub_Segments_Details

Outputs
-------
For each subset file:
    subset_data_Top_*.csv -> subset_data_Top_*_overlap_check_dual_{threshold}.csv

Summary:
    summary_novel_vs_exist_dual_{threshold}.csv
    plot_novel_trend_dual_check_{coverage_threshold}_{threshold}.png
"""

import argparse
import os
import re
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_FILES_TO_PROCESS = [
    "subset_data_Top_05%.csv",
    "subset_data_Top_10%.csv",
    "subset_data_Top_20%.csv",
    "subset_data_Top_50%.csv",
    "subset_data_Top_100%.csv",
]

Range = Tuple[Optional[int], Optional[int]]
HHSegmentPair = Tuple[Tuple[int, int], Tuple[int, int]]


def parse_range(s: Any) -> Range:
    try:
        if pd.isna(s):
            return None, None
        m = re.match(r"(\d+)-(\d+)", str(s).strip())
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None, None


def calculate_overlap_len(range1: Range, range2: Range) -> int:
    if not range1 or not range2 or not range1[0] or not range2[0]:
        return 0

    start1, end1 = range1
    start2, end2 = range2
    assert start1 is not None and end1 is not None and start2 is not None and end2 is not None

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_end >= overlap_start:
        return overlap_end - overlap_start + 1
    return 0


def get_segment_len(rng: Range) -> int:
    if not rng or not rng[0]:
        return 1
    assert rng[0] is not None and rng[1] is not None
    return rng[1] - rng[0] + 1


def parse_hhsuite_details(details_str: Any) -> List[HHSegmentPair]:
    segments: List[HHSegmentPair] = []

    if pd.isna(details_str):
        return segments

    parts = str(details_str).split(",")
    for part in parts:
        m = re.search(r"(\d+)-(\d+)\s*->\s*(\d+)-(\d+)", part)
        if m:
            groups = tuple(map(int, m.groups()))
            segments.append(((groups[0], groups[1]), (groups[2], groups[3])))

    return segments


def load_hhsuite_index(path: str) -> Dict[frozenset, List[Dict[str, Any]]]:
    print(f"Loading HHsearch reference: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"HHsearch CSV does not exist: {path}")

    df = pd.read_csv(path)

    required_cols = ["Main_HMM", "Sub_HMM", "Sub_Segments_Details"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"HHsearch CSV is missing required columns: {missing_cols}\n"
            f"Current columns: {list(df.columns)}"
        )

    lookup: Dict[frozenset, List[Dict[str, Any]]] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing HHsearch"):
        key = frozenset([row["Main_HMM"], row["Sub_HMM"]])
        if key not in lookup:
            lookup[key] = []

        segs = parse_hhsuite_details(row.get("Sub_Segments_Details", ""))
        lookup[key].append({"Main_ID": row["Main_HMM"], "Segments": segs})

    return lookup


def analyze_file_dual(
    filepath: str,
    hh_lookup: Dict[frozenset, List[Dict[str, Any]]],
    coverage_threshold: float,
    hhsuite_threshold: int,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(filepath):
        print(f"Warning: file skipped because it does not exist: {filepath}")
        return None

    df = pd.read_csv(filepath)

    required_cols = ["Main_HMM", "Sub_HMM", "Main_Segment", "Sub_Segments_Details"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input subset file is missing required columns: {missing_cols}\n"
            f"File: {filepath}\n"
            f"Current columns: {list(df.columns)}"
        )

    results: List[Dict[str, Any]] = []
    c_novel = 0
    c_exist = 0

    print(f"Analyzing: {os.path.basename(filepath)}")

    for _, row in df.iterrows():
        main_id = row["Main_HMM"]
        sub_id = row["Sub_HMM"]

        main_range = parse_range(row.get("Main_Segment"))

        sub_str = str(row.get("Sub_Segments_Details", ""))
        sub_range_str = sub_str.split("->")[-1].strip() if "->" in sub_str else ""
        sub_range = parse_range(sub_range_str)

        if not main_range[0] or not sub_range[0]:
            continue

        key = frozenset([main_id, sub_id])
        hits = hh_lookup.get(key, [])

        max_min_cov = 0.0
        best_hh_main_seg = None
        best_hh_sub_seg = None

        for hit in hits:
            for hh_q, hh_t in hit["Segments"]:
                # Preserve original direction check.
                if hit["Main_ID"] == main_id:
                    target_m = hh_q
                    target_s = hh_t
                else:
                    target_m = hh_t
                    target_s = hh_q

                ov_m = calculate_overlap_len(main_range, target_m)
                ov_s = calculate_overlap_len(sub_range, target_s)

                len_m = get_segment_len(main_range)
                len_s = get_segment_len(sub_range)

                cov_main = ov_m / len_m if len_m > 0 else 0.0
                cov_sub = ov_s / len_s if len_s > 0 else 0.0

                min_cov_this_hit = min(cov_main, cov_sub)

                if min_cov_this_hit > max_min_cov:
                    max_min_cov = min_cov_this_hit
                    best_hh_main_seg = target_m
                    best_hh_sub_seg = target_s

        is_novel = max_min_cov < coverage_threshold

        if is_novel:
            c_novel += 1
        else:
            c_exist += 1

        results.append({
            "Main_HMM": main_id,
            "Sub_HMM": sub_id,
            "Main_Segment": str(main_range),
            "Sub_Segment": str(sub_range),
            "Dual_Min_Coverage": max_min_cov,
            "Is_Novel": is_novel,
            "HH_Main_Segment": str(best_hh_main_seg) if best_hh_main_seg else "",
            "HH_Sub_Segment": str(best_hh_sub_seg) if best_hh_sub_seg else "",
        })

    res_df = pd.DataFrame(results)
    detail_out = filepath.replace(".csv", f"_overlap_check_dual_{hhsuite_threshold}.csv")

    cols = [
        "Main_HMM",
        "Sub_HMM",
        "Main_Segment",
        "Sub_Segment",
        "Dual_Min_Coverage",
        "Is_Novel",
        "HH_Main_Segment",
        "HH_Sub_Segment",
    ]
    cols = [c for c in cols if c in res_df.columns]
    res_df = res_df[cols]
    res_df.to_csv(detail_out, index=False)

    return {"File": os.path.basename(filepath), "Novel": c_novel, "Exist": c_exist}


def plot_with_matplotlib(
    summary_df: pd.DataFrame,
    output_png: str,
    coverage_threshold: float,
    hhsuite_threshold: int,
) -> None:
    print("Generating plot with Matplotlib...")

    labels = summary_df["File"].apply(lambda x: x.replace("subset_data_", "").replace(".csv", ""))
    exist_counts = summary_df["Exist"]
    novel_counts = summary_df["Novel"]

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars1 = ax.bar(
        x,
        exist_counts,
        width,
        label="Exist",
        color="#4c72b0",
        edgecolor="white",
        linewidth=0.7,
    )
    bars2 = ax.bar(
        x,
        novel_counts,
        width,
        bottom=exist_counts,
        label="Novel",
        color="#dd8452",
        edgecolor="white",
        linewidth=0.7,
    )

    ax.set_title(
        f"ProDive hits partitioned by overlap with HHsearch "
        f"(Prob >= {hhsuite_threshold}%, coverage threshold = {coverage_threshold})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel("Number of Pairs", fontsize=12)
    ax.set_xlabel("Dataset (Percentile)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=11)

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                label_text = f"{height / 1000:.1f}k" if height >= 1000 else f"{int(height)}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    label_text,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()
    plt.savefig(output_png, format="png", bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_png}")


def run_overlap_check(
    base_dir: str,
    hhsuite_csv: str,
    files_to_process: List[str],
    coverage_threshold: float,
    hhsuite_threshold: int,
) -> None:
    os.makedirs(base_dir, exist_ok=True)

    hh_lookup = load_hhsuite_index(hhsuite_csv)
    summary_list: List[Dict[str, Any]] = []

    print("Starting dual coverage analysis...")
    for fname in files_to_process:
        fpath = os.path.join(base_dir, fname)
        stats = analyze_file_dual(
            filepath=fpath,
            hh_lookup=hh_lookup,
            coverage_threshold=coverage_threshold,
            hhsuite_threshold=hhsuite_threshold,
        )
        if stats:
            summary_list.append(stats)

    if not summary_list:
        print("Warning: no files were analyzed successfully.")
        return

    summary_df = pd.DataFrame(summary_list)

    summary_csv_out = os.path.join(base_dir, f"summary_novel_vs_exist_dual_{hhsuite_threshold}.csv")
    summary_df.to_csv(summary_csv_out, index=False)
    print(f"Saved summary CSV: {summary_csv_out}")

    plot_img_out = os.path.join(
        base_dir,
        f"plot_novel_trend_dual_check_{coverage_threshold}_{hhsuite_threshold}.png",
    )
    plot_with_matplotlib(
        summary_df=summary_df,
        output_png=plot_img_out,
        coverage_threshold=coverage_threshold,
        hhsuite_threshold=hhsuite_threshold,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify ProDive percentile subsets as Novel/Exist by HHsearch overlap."
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Directory containing subset_data_Top_*.csv files and where outputs will be written.",
    )
    parser.add_argument(
        "--hhsuite-csv",
        required=True,
        help="Parsed HHsearch CSV, e.g. hhsuite20.csv from 01_parse_hhr_to_csv.py.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.2,
        help="Dual coverage threshold. Original default: 0.2.",
    )
    parser.add_argument(
        "--hhsuite-threshold",
        type=int,
        default=20,
        help="HHsearch threshold label used in output filenames. Default: 20.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES_TO_PROCESS,
        help="Subset CSV filenames under --base-dir. Default: Top_05, Top_10, Top_20, Top_50, Top_100.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_overlap_check(
        base_dir=args.base_dir,
        hhsuite_csv=args.hhsuite_csv,
        files_to_process=args.files,
        coverage_threshold=args.coverage_threshold,
        hhsuite_threshold=args.hhsuite_threshold,
    )


if __name__ == "__main__":
    main()
