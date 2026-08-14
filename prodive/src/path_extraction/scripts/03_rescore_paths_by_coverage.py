#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply MSA-completeness rescoring to ProDive path summaries.

Input
-----
The input CSV is normally produced by `02_build_paths_from_filtered_pk.py`:

    global_high_score_summary.csv

It must contain the following columns:

    Main_HMM
    Sub_HMM
    Avg_Similarity
    Sub_Segments_Details

Coverage tables are normally produced by `04_generate_hhm_coverage.py`. Each
coverage CSV contains at least:

    Match_State
    Completeness_Percent

Scoring
-------
For each path, the mean coverage over the matched segment is calculated on both
sides. The adjusted score is:

    weight = (coverage_main + coverage_sub) / 2
    adjusted_score = log10(avg_similarity * weight^2)

Rows are retained when:

    coverage_main >= coverage_threshold
    coverage_sub  >= coverage_threshold
    adjusted_score > score_threshold
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


REQUIRED_INPUT_COLUMNS = ["Main_HMM", "Sub_HMM", "Avg_Similarity", "Sub_Segments_Details"]
REQUIRED_COVERAGE_COLUMNS = ["Match_State", "Completeness_Percent"]


# ==============================================================================
# Utilities
# ==============================================================================


def safe_output_stem(record_id: str) -> str:
    """Return the filesystem-safe stem used by the coverage-generation script."""
    s = str(record_id).strip().replace("\\", "/")
    s = re.sub(r"[/:]+", "__", s)
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", s)
    return s.strip("_") or "record"


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Read a CSV file with UTF-8-SIG and GBK fallback for old local outputs."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def parse_segment_details(details_str: object) -> List[Tuple[int, int, int, int]]:
    """
    Parse segment-pair strings such as `10-15 -> 22-27; 30-35 -> 40-45`.
    """
    segments: List[Tuple[int, int, int, int]] = []
    if pd.isna(details_str) or str(details_str).strip() == "":
        return segments

    for part in str(details_str).split(";"):
        match = re.search(r"(\d+)\s*-\s*(\d+)\s*->\s*(\d+)\s*-\s*(\d+)", part)
        if match:
            segments.append((
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
            ))
    return segments


def candidate_coverage_paths(coverage_dir: Path, record_id: str) -> List[Path]:
    """Return possible coverage CSV paths for a record ID."""
    safe = safe_output_stem(record_id)
    candidates = [
        coverage_dir / f"{safe}_coverage.csv",                 # current batch output
        coverage_dir / f"{record_id}_coverage.csv",            # old flat output
        coverage_dir / record_id / f"{record_id}_coverage.csv", # old Pfam-style nested output
        coverage_dir / safe / f"{safe}_coverage.csv",
    ]

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def load_coverage_file(record_id: str, coverage_dir: Path, cache: Dict[str, Optional[Dict[int, float]]]) -> Optional[Dict[int, float]]:
    """Load one coverage table as `{match_state: completeness_fraction}`."""
    if record_id in cache:
        return cache[record_id]

    chosen: Optional[Path] = None
    for path in candidate_coverage_paths(coverage_dir, record_id):
        if path.exists():
            chosen = path
            break

    if chosen is None:
        cache[record_id] = None
        return None

    try:
        df = pd.read_csv(chosen, usecols=REQUIRED_COVERAGE_COLUMNS, encoding="utf-8-sig")
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        data = {
            int(row.Match_State): float(row.Completeness_Percent) / 100.0
            for row in df.itertuples(index=False)
        }
        cache[record_id] = data
        return data
    except Exception as exc:
        print(f"[WARN] Failed to read coverage table for {record_id}: {chosen} ({exc!r})", file=sys.stderr)
        cache[record_id] = None
        return None


def mean_coverage(cov_data: Optional[Dict[int, float]], start: int, end: int, missing_default: float) -> float:
    """Calculate mean completeness over a closed match-state interval."""
    if cov_data is None:
        return float(missing_default)

    start_i = int(start)
    end_i = int(end)
    if start_i > end_i:
        start_i, end_i = end_i, start_i

    values = [cov_data[i] for i in range(start_i, end_i + 1) if i in cov_data]
    if not values:
        return float(missing_default)
    return float(sum(values) / len(values))


# ==============================================================================
# Rescoring
# ==============================================================================


def rescore_dataframe(
    df: pd.DataFrame,
    coverage_dir: Path,
    missing_coverage_default: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Add coverage and adjusted-score columns to a path summary table."""
    missing_columns = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {missing_columns}")

    coverage_cache: Dict[str, Optional[Dict[int, float]]] = {}

    new_scores: List[float] = []
    main_covs: List[float] = []
    sub_covs: List[float] = []
    parse_failures = 0
    missing_coverage_records = 0
    zero_or_invalid_similarity = 0

    iterator = df.itertuples(index=False)
    if HAS_TQDM:
        iterator = tqdm(list(iterator), desc="Coverage rescoring", unit="path")

    for row in iterator:
        try:
            main_hmm = str(getattr(row, "Main_HMM"))
            sub_hmm = str(getattr(row, "Sub_HMM"))
            raw_similarity = float(getattr(row, "Avg_Similarity"))
            details = getattr(row, "Sub_Segments_Details")
        except Exception:
            parse_failures += 1
            new_scores.append(float("nan"))
            main_covs.append(float(missing_coverage_default))
            sub_covs.append(float(missing_coverage_default))
            continue

        segments = parse_segment_details(details)
        if not segments:
            parse_failures += 1
            new_scores.append(float("nan"))
            main_covs.append(float(missing_coverage_default))
            sub_covs.append(float(missing_coverage_default))
            continue

        main_data = load_coverage_file(main_hmm, coverage_dir, coverage_cache)
        sub_data = load_coverage_file(sub_hmm, coverage_dir, coverage_cache)
        if main_data is None:
            missing_coverage_records += 1
        if sub_data is None:
            missing_coverage_records += 1

        seg_main_means: List[float] = []
        seg_sub_means: List[float] = []

        for m_s, m_e, s_s, s_e in segments:
            seg_main_means.append(mean_coverage(main_data, m_s, m_e, missing_coverage_default))
            seg_sub_means.append(mean_coverage(sub_data, s_s, s_e, missing_coverage_default))

        final_main_cov = float(sum(seg_main_means) / len(seg_main_means)) if seg_main_means else float(missing_coverage_default)
        final_sub_cov = float(sum(seg_sub_means) / len(seg_sub_means)) if seg_sub_means else float(missing_coverage_default)

        weight_factor = (final_main_cov + final_sub_cov) / 2.0
        adjusted_raw = raw_similarity * (weight_factor ** 2)

        if not np.isfinite(adjusted_raw) or adjusted_raw <= 0:
            zero_or_invalid_similarity += 1
            adjusted_score = float("nan")
        else:
            adjusted_score = float(np.log10(adjusted_raw))

        new_scores.append(adjusted_score)
        main_covs.append(final_main_cov)
        sub_covs.append(final_sub_cov)

    result_df = df.copy()
    if "Score" in result_df.columns:
        result_df["Original_Score"] = pd.to_numeric(result_df["Score"], errors="coerce")
    else:
        result_df["Original_Score"] = np.nan

    result_df["Rescored_Score"] = new_scores
    result_df["Coverage_Main"] = main_covs
    result_df["Coverage_Sub"] = sub_covs

    stats = {
        "input_rows": int(len(df)),
        "parse_failures": int(parse_failures),
        "missing_coverage_record_events": int(missing_coverage_records),
        "zero_or_invalid_adjusted_similarity": int(zero_or_invalid_similarity),
        "coverage_records_loaded": int(sum(1 for v in coverage_cache.values() if v is not None)),
        "coverage_records_missing_or_failed": int(sum(1 for v in coverage_cache.values() if v is None)),
    }
    return result_df, stats


# ==============================================================================
# Plotting
# ==============================================================================


def clean_scores_for_plot(scores: object, xmin: float, xmax: float) -> np.ndarray:
    arr = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= xmin) & (arr <= xmax)]
    return arr


def draw_score_distribution(
    before_scores: object,
    after_scores: object,
    after_cov_scores: object,
    output_png: Path,
    coverage_threshold: float,
    score_threshold: float,
    bins: int,
    xmin: float,
    xmax: float,
    dpi: int,
) -> None:
    """Draw the before/after score distribution plot."""
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting but is not installed")

    before = clean_scores_for_plot(before_scores, xmin, xmax)
    after = clean_scores_for_plot(after_scores, xmin, xmax)
    after_cov = clean_scores_for_plot(after_cov_scores, xmin, xmax)

    if len(before) == 0 and len(after) == 0 and len(after_cov) == 0:
        print("[WARN] Score distribution plot skipped because all score arrays are empty.", file=sys.stderr)
        return

    bin_edges = np.linspace(xmin, xmax, bins + 1)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))

    def _hist(data: np.ndarray, label: str, alpha: float, linewidth: float) -> None:
        if len(data) == 0:
            return
        weights = np.ones(len(data), dtype=float) / len(data)
        ax.hist(data, bins=bin_edges, weights=weights, alpha=alpha, edgecolor="none", label=label)
        ax.hist(data, bins=bin_edges, weights=weights, histtype="step", linewidth=linewidth)

    _hist(before, "Before rescoring", 0.28, 1.8)
    _hist(after, "After rescoring (all)", 0.28, 1.8)
    _hist(after_cov, f"After rescoring (coverage >= {coverage_threshold})", 0.22, 2.0)

    ax.axvline(score_threshold, linestyle="--", linewidth=1.8, label=f"Score threshold = {score_threshold}")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Proportion")
    ax.set_title("Score distributions before and after rescoring")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=dpi)
    plt.close(fig)


# ==============================================================================
# CLI
# ==============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply coverage-based completeness rescoring to ProDive path summaries."
    )
    parser.add_argument("--input-csv", required=True, help="Input global_high_score_summary.csv from path building.")
    parser.add_argument("--coverage-dir", required=True, help="Directory containing *_coverage.csv files.")
    parser.add_argument("--output-csv", required=True, help="Final filtered output CSV.")
    parser.add_argument("--rescored-all-csv", default="", help="Optional output CSV containing all rows after rescoring before final filtering.")
    parser.add_argument("--summary-json", default="", help="Optional JSON summary path. Default: output_csv with .summary.json suffix.")
    parser.add_argument("--coverage-threshold", type=float, default=0.85, help="Minimum coverage required on both sides. Default: 0.85.")
    parser.add_argument("--score-threshold", type=float, default=1.2, help="Adjusted score threshold. Rows must satisfy score > threshold. Default: 1.2.")
    parser.add_argument("--missing-coverage-default", type=float, default=0.0, help="Coverage value used when a coverage table or state is missing. Default: 0.0.")
    parser.add_argument("--score-distribution-png", default="", help="Optional score-distribution plot path.")
    parser.add_argument("--plot-bins", type=int, default=120)
    parser.add_argument("--plot-xmin", type=float, default=0.5)
    parser.add_argument("--plot-xmax", type=float, default=3.5)
    parser.add_argument("--plot-dpi", type=int, default=300)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    t0 = time.time()

    input_csv = Path(args.input_csv)
    coverage_dir = Path(args.coverage_dir)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json) if args.summary_json else output_csv.with_suffix(output_csv.suffix + ".summary.json")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not coverage_dir.exists():
        raise FileNotFoundError(f"Coverage directory not found: {coverage_dir}")

    print("=" * 80)
    print("Coverage-based ProDive path rescoring")
    print(f"Input CSV:          {input_csv}")
    print(f"Coverage directory: {coverage_dir}")
    print(f"Output CSV:         {output_csv}")
    print(f"Coverage threshold: {args.coverage_threshold}")
    print(f"Score threshold:    {args.score_threshold}")
    print("=" * 80)

    df = read_csv_with_fallback(input_csv)
    rescored_df, stats = rescore_dataframe(
        df=df,
        coverage_dir=coverage_dir,
        missing_coverage_default=args.missing_coverage_default,
    )

    coverage_filtered_df = rescored_df[
        (rescored_df["Coverage_Main"] >= args.coverage_threshold)
        & (rescored_df["Coverage_Sub"] >= args.coverage_threshold)
    ].copy()

    final_df = coverage_filtered_df[
        pd.to_numeric(coverage_filtered_df["Rescored_Score"], errors="coerce") > args.score_threshold
    ].copy()
    final_df["Score"] = final_df["Rescored_Score"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if args.rescored_all_csv:
        rescored_all_csv = Path(args.rescored_all_csv)
        rescored_all_csv.parent.mkdir(parents=True, exist_ok=True)
        rescored_df.to_csv(rescored_all_csv, index=False, encoding="utf-8-sig")
    else:
        rescored_all_csv = None

    if args.score_distribution_png:
        draw_score_distribution(
            before_scores=rescored_df["Original_Score"],
            after_scores=rescored_df["Rescored_Score"],
            after_cov_scores=coverage_filtered_df["Rescored_Score"],
            output_png=Path(args.score_distribution_png),
            coverage_threshold=args.coverage_threshold,
            score_threshold=args.score_threshold,
            bins=args.plot_bins,
            xmin=args.plot_xmin,
            xmax=args.plot_xmax,
            dpi=args.plot_dpi,
        )

    stats.update({
        "input_csv": str(input_csv),
        "coverage_dir": str(coverage_dir),
        "output_csv": str(output_csv),
        "rescored_all_csv": str(rescored_all_csv) if rescored_all_csv else "",
        "score_distribution_png": str(args.score_distribution_png) if args.score_distribution_png else "",
        "coverage_threshold": float(args.coverage_threshold),
        "score_threshold": float(args.score_threshold),
        "rows_after_coverage_filter": int(len(coverage_filtered_df)),
        "rows_after_final_score_filter": int(len(final_df)),
        "duration_seconds": round(time.time() - t0, 3),
        "score_definition": "Rescored_Score = log10(Avg_Similarity * (((Coverage_Main + Coverage_Sub) / 2)^2))",
        "filter_definition": "Coverage_Main >= coverage_threshold and Coverage_Sub >= coverage_threshold and Rescored_Score > score_threshold",
    })

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Coverage rescoring completed")
    print(f"Input rows:                 {len(df):,}")
    print(f"After coverage filtering:   {len(coverage_filtered_df):,}")
    print(f"After final score filtering:{len(final_df):,}")
    print(f"Output CSV:                 {output_csv}")
    print(f"Summary JSON:               {summary_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
