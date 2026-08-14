#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from tqdm import tqdm


# ==============================================================================
# 1. Runtime configuration. Values are set from command-line arguments in main().
# ==============================================================================
ROOT_DIR = None
FULL_GRID_DIR_PATTERN = "*_full_grid"
PFAM_IDS_FILE = None
OUTPUT_ROOT = None
GLOBAL_GRID_SUMMARY_CSV = None
GLOBAL_FAILED_COMBOS_TXT = None
AUTO_DISCOVER_COMBOS = True
FIRST_THRESHOLDS = [6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]
SECOND_THRESHOLDS = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0]
NUM_WORKERS_COMBOS = 3
CHUNKSIZE = 1
MAX_TASKS_PER_CHILD = 1
RESUME = True
SAVE_MERGED_CSV = False
SAVE_FILTERED_CSV = False
SAVE_CURVE_CSV = True
SAVE_MISSING_PFAMS = True
REQUIRE_FULL_COVERAGE = False
NORMALIZE_PFAM_ID = True
READ_ONLY_NEEDED_COLUMNS = True

# ==============================================================================
# 2. Column compatibility
# ==============================================================================

COLUMN_ALIASES = {
    "Score": [
        "Score",
        "Rescored_Score",
        "Sadj",
        "Adjusted_Score",
        "Final_Score",
    ],
    "Main_HMM": [
        "Main_HMM",
        "Query_HMM",
        "HMM_A",
        "Family_A",
        "Pfam_A",
        "Main_Family",
        "Main",
    ],
    "Sub_HMM": [
        "Sub_HMM",
        "Target_HMM",
        "HMM_B",
        "Family_B",
        "Pfam_B",
        "Sub_Family",
        "Sub",
    ],
}


# ==============================================================================
# 3. Worker global variables
# ==============================================================================

G_FULL_GRID_DIRS = None
G_ALL_PFAMS_SET = None
G_TARGET_PFAMS_SET = None


# ==============================================================================
# 4. Helper functions
# ==============================================================================

def tag_float(x: float) -> str:
    return f"{float(x):.1f}".replace(".", "p")


def combo_key(first_threshold: float, second_threshold: float) -> str:
    return f"F{tag_float(first_threshold)}_S{tag_float(second_threshold)}"


def parse_combo_key(key: str):
    """
    F1p0_S8p0 -> (1.0, 8.0)
    """
    m = re.match(r"^F([0-9]+p[0-9]+)_S([0-9]+p[0-9]+)$", str(key))
    if not m:
        return None

    first = float(m.group(1).replace("p", "."))
    second = float(m.group(2).replace("p", "."))

    return first, second


def normalize_colname(c: str) -> str:
    return str(c).strip().replace("\ufeff", "")


def normalize_pfam_id(x) -> str:
    s = str(x).strip()

    if not NORMALIZE_PFAM_ID:
        return s

    m = re.search(r"PF\d{5}", s)
    if m:
        return m.group(0)

    return s


def find_column_from_names(columns, logical_name: str) -> str:
    candidates = COLUMN_ALIASES[logical_name]
    columns = [normalize_colname(c) for c in columns]

    existing = set(columns)

    for c in candidates:
        if c in existing:
            return c

    lower_map = {c.lower(): c for c in columns}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    raise RuntimeError(
        f"Cannot find required column for {logical_name}. "
        f"Tried aliases: {candidates}. "
        f"Existing columns: {columns}"
    )


def find_column(df: pd.DataFrame, logical_name: str) -> str:
    return find_column_from_names(df.columns, logical_name)


def read_csv_header(path: str):
    try:
        df0 = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    except UnicodeDecodeError:
        df0 = pd.read_csv(path, encoding="gbk", nrows=0)

    df0.columns = [normalize_colname(c) for c in df0.columns]
    return list(df0.columns)


def read_csv_robust(path: str, needed_logical_cols=None) -> pd.DataFrame:
    """
    Robust CSV reader.

    If needed_logical_cols is provided, only read columns needed for:
      Score, Main_HMM, Sub_HMM
    """
    if needed_logical_cols is None:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="gbk", low_memory=False)

        df.columns = [normalize_colname(c) for c in df.columns]
        return df

    header_cols = read_csv_header(path)

    logical_to_real = {}
    for logical in needed_logical_cols:
        logical_to_real[logical] = find_column_from_names(header_cols, logical)

    usecols = list(logical_to_real.values())

    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=usecols,
            low_memory=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding="gbk",
            usecols=usecols,
            low_memory=False,
        )

    df.columns = [normalize_colname(c) for c in df.columns]

    rename_map = {
        logical_to_real["Score"]: "Score",
        logical_to_real["Main_HMM"]: "Main_HMM",
        logical_to_real["Sub_HMM"]: "Sub_HMM",
    }

    df = df.rename(columns=rename_map)

    return df


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def discover_full_grid_dirs():
    dirs = sorted(
        d for d in glob.glob(os.path.join(ROOT_DIR, FULL_GRID_DIR_PATTERN))
        if os.path.isdir(d)
    )

    if not dirs:
        raise RuntimeError(
            f"No full_grid directories found under {ROOT_DIR} "
            f"with pattern {FULL_GRID_DIR_PATTERN}"
        )

    return dirs


def discover_combos(full_grid_dirs):
    combo_set = set()

    for full_grid_dir in full_grid_dirs:
        for d in glob.glob(os.path.join(full_grid_dir, "F*_S*")):
            if not os.path.isdir(d):
                continue

            name = os.path.basename(d)
            parsed = parse_combo_key(name)

            if parsed is not None:
                combo_set.add(name)

    combos = sorted(
        combo_set,
        key=lambda k: parse_combo_key(k),
    )

    if not combos:
        raise RuntimeError("No F*_S* combo directories found.")

    return combos


def get_combos(full_grid_dirs):
    if AUTO_DISCOVER_COMBOS:
        return discover_combos(full_grid_dirs)

    return [
        combo_key(f, s)
        for f in FIRST_THRESHOLDS
        for s in SECOND_THRESHOLDS
    ]


def find_combo_csvs(full_grid_dirs, ckey):
    """
    For one combo, find:
      /ROOT/52_full_grid/F1p0_S8p0/F1p0_S8p0_global_clean_paths_Sadj.csv
      /ROOT/54_full_grid/F1p0_S8p0/F1p0_S8p0_global_clean_paths_Sadj.csv
      ...
    """
    paths = []

    expected_name = f"{ckey}_global_clean_paths_Sadj.csv"

    for full_grid_dir in full_grid_dirs:
        p = os.path.join(full_grid_dir, ckey, expected_name)

        if os.path.exists(p):
            paths.append(p)

    return sorted(paths)


def load_target_pfams():
    """
    Load all target Pfam families.

    Load target Pfam families from the supplied ID list.
    No manual Pfam exclusion is applied.
    Missing target families are written to the missing-target output files.
    """
    with open(PFAM_IDS_FILE, "r", encoding="utf-8") as f:
        raw_pfams = [line.strip() for line in f if line.strip()]

    all_pfams_set = set(normalize_pfam_id(x) for x in raw_pfams)

    target_pfams_set = set(all_pfams_set)

    return all_pfams_set, target_pfams_set


def count_rows_ge_score_factory(scores_desc):
    """
    scores_desc must be sorted descending.
    Return a fast function to count rows with Score >= threshold.
    """
    neg_scores_asc = -scores_desc

    def count_rows_ge_score(score_value: float) -> int:
        return int(np.searchsorted(neg_scores_asc, -score_value, side="right"))

    return count_rows_ge_score


# ==============================================================================
# 5. Load and merge one combo
# ==============================================================================

def load_and_merge_combo_csvs(csv_paths, ckey):
    dfs = []

    needed = ["Score", "Main_HMM", "Sub_HMM"] if READ_ONLY_NEEDED_COLUMNS else None

    for src_idx, path in enumerate(csv_paths):
        try:
            df = read_csv_robust(path, needed_logical_cols=needed)
        except Exception as e:
            print(f"[WARN] {ckey}: failed to read {path}: {e}", flush=True)
            continue

        if df.empty:
            continue

        if not READ_ONLY_NEEDED_COLUMNS:
            score_col = find_column(df, "Score")
            main_col = find_column(df, "Main_HMM")
            sub_col = find_column(df, "Sub_HMM")

            df = df.rename(
                columns={
                    score_col: "Score",
                    main_col: "Main_HMM",
                    sub_col: "Sub_HMM",
                }
            )

        
        if len(csv_paths) < 32767:
            df["__source_idx__"] = np.int16(src_idx)
        else:
            df["__source_idx__"] = np.int32(src_idx)

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.columns = [normalize_colname(c) for c in merged_df.columns]

    merged_df["Score"] = pd.to_numeric(merged_df["Score"], errors="coerce")
    merged_df["Main_HMM"] = merged_df["Main_HMM"].map(normalize_pfam_id)
    merged_df["Sub_HMM"] = merged_df["Sub_HMM"].map(normalize_pfam_id)

    before_drop = len(merged_df)
    merged_df = merged_df.dropna(subset=["Score", "Main_HMM", "Sub_HMM"]).copy()
    after_drop = len(merged_df)

    print(
        f"[INFO] {ckey}: valid rows = {after_drop:,}, "
        f"dropped invalid = {before_drop - after_drop:,}",
        flush=True,
    )

    if merged_df.empty:
        return merged_df

    merged_df = merged_df.sort_values(
        by="Score",
        ascending=False,
    ).reset_index(drop=True)

    return merged_df


# ==============================================================================
# 6. Analyze one combo
# ==============================================================================

def analyze_one_combo(ckey):
    global G_FULL_GRID_DIRS
    global G_ALL_PFAMS_SET
    global G_TARGET_PFAMS_SET

    parsed = parse_combo_key(ckey)
    if parsed is None:
        raise RuntimeError(f"Bad combo key: {ckey}")

    first_threshold, second_threshold = parsed

    csv_paths = find_combo_csvs(G_FULL_GRID_DIRS, ckey)

    combo_out_dir = os.path.join(OUTPUT_ROOT, ckey)
    safe_mkdir(combo_out_dir)

    combo_summary_path = os.path.join(combo_out_dir, f"{ckey}_third_score_summary.csv")
    output_merged_csv = os.path.join(combo_out_dir, f"{ckey}_merged_global_clean_paths_Sadj.csv")
    output_filtered_csv = os.path.join(combo_out_dir, f"{ckey}_selected_score_filtered_paths.csv")
    output_curve_asc_csv = os.path.join(combo_out_dir, f"{ckey}_score_coverage_curve_asc.csv")
    output_curve_desc_csv = os.path.join(combo_out_dir, f"{ckey}_score_coverage_curve_desc.csv")
    output_missing_pfams_txt = os.path.join(combo_out_dir, f"{ckey}_missing_target_pfams.txt")
    output_missing_pfams_csv = os.path.join(combo_out_dir, f"{ckey}_missing_target_pfams.csv")
    output_report_txt = os.path.join(combo_out_dir, f"{ckey}_third_score_threshold_report.txt")
    output_source_files_txt = os.path.join(combo_out_dir, f"{ckey}_source_files.txt")

    if RESUME and os.path.exists(combo_summary_path):
        try:
            old = pd.read_csv(combo_summary_path, encoding="utf-8-sig")
            if not old.empty:
                summary = old.iloc[0].to_dict()
                summary["_status"] = "skipped_existing"
                return summary
        except Exception:
            pass

    with open(output_source_files_txt, "w", encoding="utf-8") as f:
        for p in csv_paths:
            f.write(p + "\n")

    target_n = len(G_TARGET_PFAMS_SET)

    if not csv_paths:
        summary = {
            "combo_key": ckey,
            "first_threshold": first_threshold,
            "second_threshold": second_threshold,
            "source_files": 0,
            "valid_rows": 0,
            "target_families": target_n,
            "actual_covered_families": 0,
            "missing_target_families": target_n,
            "coverage_mode": "NO_INPUT_FILES",
            "selected_coverage_n": 0,
            "selected_coverage_fraction_of_target": 0.0,
            "selected_threshold_score": np.nan,
            "selected_rows_with_score_ge_threshold": 0,
            "max_score": np.nan,
            "min_score": np.nan,
            "curve_asc_csv": "",
            "curve_desc_csv": "",
            "report_txt": "",
            "merged_csv": "",
            "filtered_csv": "",
            "_status": "ok_no_input",
        }

        pd.DataFrame([summary]).drop(columns=["_status"]).to_csv(
            combo_summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        return summary

    print(
        f"[RUN] {ckey}: source files = {len(csv_paths)}",
        flush=True,
    )

    merged_df = load_and_merge_combo_csvs(csv_paths, ckey)

    if merged_df.empty:
        summary = {
            "combo_key": ckey,
            "first_threshold": first_threshold,
            "second_threshold": second_threshold,
            "source_files": len(csv_paths),
            "valid_rows": 0,
            "target_families": target_n,
            "actual_covered_families": 0,
            "missing_target_families": target_n,
            "coverage_mode": "EMPTY_AFTER_MERGE",
            "selected_coverage_n": 0,
            "selected_coverage_fraction_of_target": 0.0,
            "selected_threshold_score": np.nan,
            "selected_rows_with_score_ge_threshold": 0,
            "max_score": np.nan,
            "min_score": np.nan,
            "curve_asc_csv": "",
            "curve_desc_csv": "",
            "report_txt": "",
            "merged_csv": "",
            "filtered_csv": "",
            "_status": "ok_empty",
        }

        pd.DataFrame([summary]).drop(columns=["_status"]).to_csv(
            combo_summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        return summary

    if SAVE_MERGED_CSV:
        merged_df.to_csv(output_merged_csv, index=False, encoding="utf-8-sig")
        merged_csv_for_summary = output_merged_csv
    else:
        merged_csv_for_summary = ""

    actual_seen_anywhere = set(merged_df["Main_HMM"]) | set(merged_df["Sub_HMM"])
    actual_seen_target = actual_seen_anywhere & G_TARGET_PFAMS_SET
    missing_targets = sorted(G_TARGET_PFAMS_SET - actual_seen_anywhere)
    actual_covered_n = len(actual_seen_target)

    if SAVE_MISSING_PFAMS:
        with open(output_missing_pfams_txt, "w", encoding="utf-8") as f:
            for pf in missing_targets:
                f.write(pf + "\n")

        pd.DataFrame({"Missing_Pfam": missing_targets}).to_csv(
            output_missing_pfams_csv,
            index=False,
            encoding="utf-8-sig",
        )

    if missing_targets and REQUIRE_FULL_COVERAGE:
        raise RuntimeError(
            f"{ckey}: current result does not reach full target coverage. "
            f"Missing {len(missing_targets)} target families."
        )

    max_score = float(merged_df["Score"].max())
    min_score = float(merged_df["Score"].min())

    if actual_covered_n == 0:
        summary = {
            "combo_key": ckey,
            "first_threshold": first_threshold,
            "second_threshold": second_threshold,
            "source_files": len(csv_paths),
            "valid_rows": int(len(merged_df)),
            "target_families": target_n,
            "actual_covered_families": 0,
            "missing_target_families": len(missing_targets),
            "coverage_mode": "NO_TARGET_FAMILY_COVERED",
            "selected_coverage_n": 0,
            "selected_coverage_fraction_of_target": 0.0,
            "selected_threshold_score": np.nan,
            "selected_rows_with_score_ge_threshold": 0,
            "max_score": max_score,
            "min_score": min_score,
            "curve_asc_csv": "",
            "curve_desc_csv": "",
            "report_txt": "",
            "merged_csv": merged_csv_for_summary,
            "filtered_csv": "",
            "_status": "ok_no_target",
        }

        pd.DataFrame([summary]).drop(columns=["_status"]).to_csv(
            combo_summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        return summary

    scores_desc = merged_df["Score"].to_numpy(dtype=float)
    count_rows_ge_score = count_rows_ge_score_factory(scores_desc)

    seen = set()
    records_by_coverage = [None] * (actual_covered_n + 1)
    previous_coverage = 0

    main_arr = merged_df["Main_HMM"].to_numpy()
    sub_arr = merged_df["Sub_HMM"].to_numpy()
    score_arr = merged_df["Score"].to_numpy(dtype=float)

    if "__source_idx__" in merged_df.columns:
        source_idx_arr = merged_df["__source_idx__"].to_numpy()
    else:
        source_idx_arr = np.full(len(merged_df), -1, dtype=np.int32)

    for idx in range(len(merged_df)):
        main_hmm = main_arr[idx]
        sub_hmm = sub_arr[idx]
        score = float(score_arr[idx])
        source_idx = int(source_idx_arr[idx])

        if main_hmm in G_TARGET_PFAMS_SET:
            seen.add(main_hmm)

        if sub_hmm in G_TARGET_PFAMS_SET:
            seen.add(sub_hmm)

        after = len(seen)

        if after > previous_coverage:
            retained_rows = count_rows_ge_score(score)
            newly_added = after - previous_coverage
            upper = min(after, actual_covered_n)

            if 0 <= source_idx < len(csv_paths):
                source_file = csv_paths[source_idx]
                source_node = os.path.basename(os.path.dirname(os.path.dirname(source_file)))
            else:
                source_file = ""
                source_node = ""

            for c in range(previous_coverage + 1, upper + 1):
                records_by_coverage[c] = {
                    "combo_key": ckey,
                    "first_threshold": first_threshold,
                    "second_threshold": second_threshold,
                    "Coverage_N": c,
                    "Coverage_Fraction_of_Target": c / target_n,
                    "Coverage_Fraction_of_Actually_Coverable": c / actual_covered_n,
                    "Threshold_Score": score,
                    "First_Reached_Row_Index_0Based": idx,
                    "First_Reached_Row_Number_1Based": idx + 1,
                    "Rows_With_Score_GE_Threshold": retained_rows,
                    "New_Families_Added_At_This_Row": newly_added,
                    "Main_HMM_At_Threshold_Row": main_hmm,
                    "Sub_HMM_At_Threshold_Row": sub_hmm,
                    "Source_Node_At_Threshold_Row": source_node,
                    "Source_File_At_Threshold_Row": source_file,
                }

            previous_coverage = after

        if previous_coverage >= actual_covered_n:
            break

    missing_levels = [
        c for c in range(1, actual_covered_n + 1)
        if records_by_coverage[c] is None
    ]

    if missing_levels:
        raise RuntimeError(
            f"{ckey}: internal error, missing coverage levels: {missing_levels[:20]}"
        )

    curve_asc_df = pd.DataFrame(records_by_coverage[1:])
    curve_desc_df = curve_asc_df.sort_values(
        by="Coverage_N",
        ascending=False,
    ).reset_index(drop=True)

    if SAVE_CURVE_CSV:
        curve_asc_df.to_csv(output_curve_asc_csv, index=False, encoding="utf-8-sig")
        curve_desc_df.to_csv(output_curve_desc_csv, index=False, encoding="utf-8-sig")
        curve_asc_for_summary = output_curve_asc_csv
        curve_desc_for_summary = output_curve_desc_csv
    else:
        curve_asc_for_summary = ""
        curve_desc_for_summary = ""

    if actual_covered_n == target_n:
        coverage_mode = "FULL_TARGET_COVERAGE"
        selected_coverage_n = target_n
        selected_record = records_by_coverage[target_n]
    else:
        coverage_mode = "MAX_REACHED_COVERAGE_ONLY"
        selected_coverage_n = actual_covered_n
        selected_record = records_by_coverage[actual_covered_n]

    selected_threshold_score = float(selected_record["Threshold_Score"])
    selected_retained_rows = int(selected_record["Rows_With_Score_GE_Threshold"])
    selected_threshold_index = int(selected_record["First_Reached_Row_Index_0Based"])

    if SAVE_FILTERED_CSV:
        filtered_df = merged_df[merged_df["Score"] >= selected_threshold_score].copy()
        filtered_df.to_csv(output_filtered_csv, index=False, encoding="utf-8-sig")
        filtered_csv_for_summary = output_filtered_csv
        filtered_len = len(filtered_df)
    else:
        filtered_csv_for_summary = ""
        filtered_len = selected_retained_rows

    example_levels = [
        actual_covered_n,
        actual_covered_n - 1,
        actual_covered_n - 2,
        actual_covered_n - 5,
        actual_covered_n - 10,
        actual_covered_n - 50,
        actual_covered_n - 100,
        25000,
        24000,
        23000,
        22000,
        21000,
        20000,
        15000,
        10000,
        5000,
        1000,
        100,
        10,
        1,
    ]

    example_levels = [
        x for x in example_levels
        if isinstance(x, int) and 1 <= x <= actual_covered_n
    ]
    example_levels = sorted(set(example_levels), reverse=True)

    with open(output_report_txt, "w", encoding="utf-8") as f:
        f.write("=== Third score-threshold report ===\n\n")

        f.write("=== Combo ===\n")
        f.write(f"combo_key: {ckey}\n")
        f.write(f"first_threshold: {first_threshold}\n")
        f.write(f"second_threshold: {second_threshold}\n")
        f.write(f"source_files: {len(csv_paths)}\n\n")

        f.write("=== Target Pfam summary ===\n")
        f.write(f"Target Pfam families from pfam_ids.txt: {target_n}\n")
        f.write("Manual excluded Pfam families: 0\n")
        f.write("Manual excluded list: none\n\n")

        f.write("=== Merged path-level result ===\n")
        f.write(f"Valid merged rows: {len(merged_df)}\n")
        f.write(f"Max Score: {max_score}\n")
        f.write(f"Min Score: {min_score}\n\n")

        f.write("=== Actual coverage ===\n")
        f.write(f"Actual covered target families: {actual_covered_n} / {target_n}\n")
        f.write(f"Missing target families: {len(missing_targets)}\n")
        f.write(f"Coverage mode: {coverage_mode}\n\n")

        f.write("=== Selected threshold ===\n")
        f.write(f"Selected Coverage_N: {selected_coverage_n}\n")
        f.write(f"Selected Coverage_Fraction_of_Target: {selected_coverage_n / target_n:.8f}\n")
        f.write(f"Selected Threshold_Score: {selected_threshold_score}\n")
        f.write(f"First reached row index, 0-based: {selected_threshold_index}\n")
        f.write(f"First reached row number, 1-based: {selected_threshold_index + 1}\n")
        f.write(f"Rows with Score >= threshold: {selected_retained_rows}\n")
        f.write(f"Filtered CSV saved: {SAVE_FILTERED_CSV}\n")

        if SAVE_FILTERED_CSV:
            f.write(f"Filtered CSV row count: {filtered_len}\n")
            f.write(f"Filtered CSV path: {output_filtered_csv}\n")

        f.write("\n")

        f.write("=== Example coverage levels ===\n")
        for c in example_levels:
            rec = records_by_coverage[c]
            f.write(
                f"Coverage_N={c}, "
                f"Coverage_Fraction_of_Target={rec['Coverage_Fraction_of_Target']:.8f}, "
                f"Coverage_Fraction_of_Actually_Coverable={rec['Coverage_Fraction_of_Actually_Coverable']:.8f}, "
                f"Threshold_Score={rec['Threshold_Score']}, "
                f"Rows_GE_Threshold={rec['Rows_With_Score_GE_Threshold']}, "
                f"First_Row_0Based={rec['First_Reached_Row_Index_0Based']}\n"
            )

        f.write("\n=== Source files ===\n")
        for p in csv_paths:
            f.write(p + "\n")

        f.write("\n=== Output files ===\n")
        f.write(f"Combo summary CSV: {combo_summary_path}\n")
        f.write(f"Curve ASC CSV: {output_curve_asc_csv if SAVE_CURVE_CSV else ''}\n")
        f.write(f"Curve DESC CSV: {output_curve_desc_csv if SAVE_CURVE_CSV else ''}\n")
        f.write(f"Missing PFAM TXT: {output_missing_pfams_txt if SAVE_MISSING_PFAMS else ''}\n")
        f.write(f"Missing PFAM CSV: {output_missing_pfams_csv if SAVE_MISSING_PFAMS else ''}\n")
        f.write(f"Merged CSV: {output_merged_csv if SAVE_MERGED_CSV else ''}\n")
        f.write(f"Filtered CSV: {output_filtered_csv if SAVE_FILTERED_CSV else ''}\n")

    summary = {
        "combo_key": ckey,
        "first_threshold": first_threshold,
        "second_threshold": second_threshold,
        "source_files": len(csv_paths),
        "valid_rows": int(len(merged_df)),
        "target_families": int(target_n),
        "actual_covered_families": int(actual_covered_n),
        "missing_target_families": int(len(missing_targets)),
        "coverage_mode": coverage_mode,
        "selected_coverage_n": int(selected_coverage_n),
        "selected_coverage_fraction_of_target": float(selected_coverage_n / target_n),
        "selected_threshold_score": float(selected_threshold_score),
        "selected_rows_with_score_ge_threshold": int(selected_retained_rows),
        "max_score": max_score,
        "min_score": min_score,
        "curve_asc_csv": curve_asc_for_summary,
        "curve_desc_csv": curve_desc_for_summary,
        "report_txt": output_report_txt,
        "merged_csv": merged_csv_for_summary,
        "filtered_csv": filtered_csv_for_summary,
        "_status": "ok",
    }

    pd.DataFrame([summary]).drop(columns=["_status"]).to_csv(
        combo_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    return summary


def worker_run_combo(ckey):
    try:
        summary = analyze_one_combo(ckey)
        return {
            "ok": True,
            "combo_key": ckey,
            "summary": summary,
            "error": "",
            "traceback": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "combo_key": ckey,
            "summary": None,
            "error": str(e),
            "traceback": traceback.format_exc(limit=5),
        }


def init_pool_worker(full_grid_dirs, all_pfams, target_pfams):
    global G_FULL_GRID_DIRS
    global G_ALL_PFAMS_SET
    global G_TARGET_PFAMS_SET

    G_FULL_GRID_DIRS = list(full_grid_dirs)
    G_ALL_PFAMS_SET = set(all_pfams)
    G_TARGET_PFAMS_SET = set(target_pfams)

    np.seterr(all="ignore")


# ==============================================================================
# 7. Main
# ==============================================================================



def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect global third-layer score thresholds from distributed full-grid path results."
    )
    parser.add_argument("--root-dir", required=True,
                        help="Directory containing server-local *_full_grid directories.")
    parser.add_argument("--pfam-ids-file", required=True,
                        help="Text file with one target Pfam ID per line.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--full-grid-dir-pattern", default="*_full_grid")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--max-tasks-per-child", type=int, default=1)
    parser.add_argument("--manual-combos", action="store_true",
                        help="Use the built-in FIRST_THRESHOLDS x SECOND_THRESHOLDS grid instead of auto-discovery.")
    parser.add_argument("--save-merged-csv", action="store_true")
    parser.add_argument("--save-filtered-csv", action="store_true")
    parser.add_argument("--no-curve-csv", action="store_true")
    parser.add_argument("--no-missing-pfams", action="store_true")
    parser.add_argument("--require-full-coverage", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--read-all-columns", action="store_true")
    return parser.parse_args()


def configure_from_args(args):
    global ROOT_DIR, PFAM_IDS_FILE, OUTPUT_ROOT, GLOBAL_GRID_SUMMARY_CSV, GLOBAL_FAILED_COMBOS_TXT
    global FULL_GRID_DIR_PATTERN, NUM_WORKERS_COMBOS, CHUNKSIZE, MAX_TASKS_PER_CHILD
    global AUTO_DISCOVER_COMBOS, SAVE_MERGED_CSV, SAVE_FILTERED_CSV, SAVE_CURVE_CSV
    global SAVE_MISSING_PFAMS, REQUIRE_FULL_COVERAGE, RESUME, READ_ONLY_NEEDED_COLUMNS
    ROOT_DIR = args.root_dir
    PFAM_IDS_FILE = args.pfam_ids_file
    OUTPUT_ROOT = args.output_root
    FULL_GRID_DIR_PATTERN = args.full_grid_dir_pattern
    NUM_WORKERS_COMBOS = args.workers
    CHUNKSIZE = args.chunksize
    MAX_TASKS_PER_CHILD = args.max_tasks_per_child
    AUTO_DISCOVER_COMBOS = not args.manual_combos
    SAVE_MERGED_CSV = args.save_merged_csv
    SAVE_FILTERED_CSV = args.save_filtered_csv
    SAVE_CURVE_CSV = not args.no_curve_csv
    SAVE_MISSING_PFAMS = not args.no_missing_pfams
    REQUIRE_FULL_COVERAGE = args.require_full_coverage
    RESUME = not args.no_resume
    READ_ONLY_NEEDED_COLUMNS = not args.read_all_columns
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    GLOBAL_GRID_SUMMARY_CSV = os.path.join(OUTPUT_ROOT, "third_score_threshold_grid_summary.csv")
    GLOBAL_FAILED_COMBOS_TXT = os.path.join(OUTPUT_ROOT, "failed_or_missing_combos.txt")

def main():
    args = parse_args()
    configure_from_args(args)

    print("=" * 100)
    print("Multiprocessing global third-layer score-threshold analysis")
    print("Parallel level: one worker processes one F/S combo.")
    print("Each combo merges all node-level *_global_clean_paths_Sadj.csv files.")
    print("No manual Pfam exclusion is used.")
    print("=" * 100)
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"PFAM_IDS_FILE: {PFAM_IDS_FILE}")
    print(f"AUTO_DISCOVER_COMBOS: {AUTO_DISCOVER_COMBOS}")
    print(f"NUM_WORKERS_COMBOS: {NUM_WORKERS_COMBOS}")
    print(f"SAVE_MERGED_CSV: {SAVE_MERGED_CSV}")
    print(f"SAVE_FILTERED_CSV: {SAVE_FILTERED_CSV}")
    print("=" * 100)

    all_pfams_set, target_pfams_set = load_target_pfams()

    print("[INFO] Pfam target set")
    print(f"  Target Pfam families from pfam_ids.txt: {len(target_pfams_set)}")
    print("  Manual excluded Pfam families: 0")
    print("=" * 100)

    full_grid_dirs = discover_full_grid_dirs()

    print("[INFO] Found full_grid directories:")
    for d in full_grid_dirs:
        print(f"  - {d}")
    print("=" * 100)

    combos = get_combos(full_grid_dirs)

    print(f"[INFO] Total F/S combos to analyze: {len(combos)}")
    print("[INFO] First 20 combos:")
    for c in combos[:20]:
        print(f"  - {c}")
    print("=" * 100)

    all_summaries = []
    failed = []

    ctx = get_context("fork")

    with ctx.Pool(
        processes=NUM_WORKERS_COMBOS,
        initializer=init_pool_worker,
        initargs=(
            list(full_grid_dirs),
            list(all_pfams_set),
            list(target_pfams_set),
        ),
        maxtasksperchild=MAX_TASKS_PER_CHILD,
    ) as pool:

        iterator = pool.imap_unordered(
            worker_run_combo,
            combos,
            chunksize=CHUNKSIZE,
        )

        pbar = tqdm(iterator, total=len(combos), desc="Third-layer combos", unit="combo")

        for res in pbar:
            ckey = res["combo_key"]

            if not res["ok"]:
                msg = f"{ckey}\t{res['error']}"
                failed.append(msg)

                with open(GLOBAL_FAILED_COMBOS_TXT, "w", encoding="utf-8") as f:
                    for x in failed:
                        f.write(x + "\n")
                    f.write("\n\n=== Last traceback ===\n")
                    f.write(res["traceback"] + "\n")

                pbar.set_postfix_str(f"failed={len(failed)}")
                continue

            summary = res["summary"]
            all_summaries.append(summary)

            
            tmp_df = pd.DataFrame(all_summaries)
            if "_status" in tmp_df.columns:
                tmp_df = tmp_df.drop(columns=["_status"])

            tmp_df.to_csv(
                GLOBAL_GRID_SUMMARY_CSV,
                index=False,
                encoding="utf-8-sig",
            )

            pbar.set_postfix_str(
                f"ok={len(all_summaries)} failed={len(failed)}"
            )

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)

        if "_status" in summary_df.columns:
            summary_df = summary_df.drop(columns=["_status"])

        summary_df = summary_df.sort_values(
            by=[
                "actual_covered_families",
                "selected_rows_with_score_ge_threshold",
                "selected_threshold_score",
                "valid_rows",
            ],
            ascending=[False, True, False, True],
        ).reset_index(drop=True)

        summary_df.to_csv(
            GLOBAL_GRID_SUMMARY_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        print("=" * 100)
        print("[DONE] All combo analyses finished.")
        print(f"Global summary CSV: {GLOBAL_GRID_SUMMARY_CSV}")

        display_cols = [
            "combo_key",
            "first_threshold",
            "second_threshold",
            "valid_rows",
            "actual_covered_families",
            "missing_target_families",
            "selected_threshold_score",
            "selected_rows_with_score_ge_threshold",
            "coverage_mode",
        ]

        print("\nTop parameter combinations by coverage and compactness:")
        print(summary_df[display_cols].head(50).to_string(index=False))

    if failed:
        with open(GLOBAL_FAILED_COMBOS_TXT, "w", encoding="utf-8") as f:
            for x in failed:
                f.write(x + "\n")

        print(f"[WARN] Failed combos saved to: {GLOBAL_FAILED_COMBOS_TXT}")

    print("=" * 100)
    print("[DONE] Finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()
