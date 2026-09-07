#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from pathlib import Path
from typing import Dict, List

import pandas as pd


# =========================================================
# 
# =========================================================
REAL_INPUT_FILE = Path(
    "CHANGE_ME"
)

RANDOM_INPUT_FILE = Path(
    "CHANGE_ME"
)

OUT_DIR = Path(
    "CHANGE_ME"
)

# 
# : "count"  "frac"
THRESH_MODE = "count"

#  THRESH_MODE == "count"，
COUNT_THRESHOLD = 2

#  THRESH_MODE == "frac"，
FRAC_THRESHOLD = 0.20


# =========================================================
# 
# =========================================================
FOLDING_CLASSES = ["Fold_EARLY", "Fold_INTER", "Fold_LATE"]
STABILITY_CLASSES = ["Stab_STRONG", "Stab_MEDIUM", "Stab_WEAK"]
ALL_CLASSES = FOLDING_CLASSES + STABILITY_CLASSES


# =========================================================
# 
# =========================================================
def dedupe_real_by_exact_position(
    df: pd.DataFrame,
    id_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    """
    “”。
     id_col ，start/end 。
    """
    out = df.copy()

    out[start_col] = pd.to_numeric(out[start_col], errors="coerce")
    out[end_col] = pd.to_numeric(out[end_col], errors="coerce")

    out = out.dropna(subset=[id_col, start_col, end_col]).copy()
    out[start_col] = out[start_col].astype(int)
    out[end_col] = out[end_col].astype(int)

    out = out.drop_duplicates(subset=[id_col, start_col, end_col], keep="first").reset_index(drop=True)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def class_hit(row: pd.Series, cls: str) -> bool:
    if THRESH_MODE == "count":
        val = pd.to_numeric(
            pd.Series([row[f"{cls}_segment_overlap_count"]]), errors="coerce"
        ).fillna(0).iloc[0]
        return val >= COUNT_THRESHOLD

    if THRESH_MODE == "frac":
        val = pd.to_numeric(
            pd.Series([row[f"{cls}_segment_overlap_frac"]]), errors="coerce"
        ).fillna(0).iloc[0]
        return val >= FRAC_THRESHOLD

    raise ValueError(f"Unsupported THRESH_MODE: {THRESH_MODE}")


def pick_dominant_class(row: pd.Series, hit_classes: List[str]) -> str:
    """
    ：
    1. overlap_frac 
    2. longest_run_frac 
    3. overlap_count 
    4.  -> ambiguous
    """
    if len(hit_classes) == 0:
        return "no_hit"
    if len(hit_classes) == 1:
        return hit_classes[0]

    candidates = []
    for cls in hit_classes:
        frac = pd.to_numeric(
            pd.Series([row[f"{cls}_segment_overlap_frac"]]), errors="coerce"
        ).fillna(0).iloc[0]
        runfrac = pd.to_numeric(
            pd.Series([row[f"{cls}_segment_longest_run_frac"]]), errors="coerce"
        ).fillna(0).iloc[0]
        count = pd.to_numeric(
            pd.Series([row[f"{cls}_segment_overlap_count"]]), errors="coerce"
        ).fillna(0).iloc[0]
        candidates.append((cls, float(frac), float(runfrac), float(count)))

    max_frac = max(x[1] for x in candidates)
    c1 = [x for x in candidates if x[1] == max_frac]
    if len(c1) == 1:
        return c1[0][0]

    max_runfrac = max(x[2] for x in c1)
    c2 = [x for x in c1 if x[2] == max_runfrac]
    if len(c2) == 1:
        return c2[0][0]

    max_count = max(x[3] for x in c2)
    c3 = [x for x in c2 if x[3] == max_count]
    if len(c3) == 1:
        return c3[0][0]

    return "ambiguous"


def pick_dominant_family(row: pd.Series, hit_classes: List[str]) -> str:
    """
    ：
    1. folding  overlap_frac  vs stability  overlap_frac 
    2. longest_run_frac 
    3. overlap_count 
    4.  -> ambiguous
    """
    if len(hit_classes) == 0:
        return "no_hit"

    folding_hit = any(c in FOLDING_CLASSES for c in hit_classes)
    stability_hit = any(c in STABILITY_CLASSES for c in hit_classes)

    if folding_hit and not stability_hit:
        return "folding"
    if stability_hit and not folding_hit:
        return "stability"

    fold_frac = sum(
        pd.to_numeric(
            pd.Series([row[f"{c}_segment_overlap_frac"]]), errors="coerce"
        ).fillna(0).iloc[0]
        for c in FOLDING_CLASSES
    )
    stab_frac = sum(
        pd.to_numeric(
            pd.Series([row[f"{c}_segment_overlap_frac"]]), errors="coerce"
        ).fillna(0).iloc[0]
        for c in STABILITY_CLASSES
    )

    if fold_frac > stab_frac:
        return "folding"
    if stab_frac > fold_frac:
        return "stability"

    fold_run = sum(
        pd.to_numeric(
            pd.Series([row[f"{c}_segment_longest_run_frac"]]), errors="coerce"
        ).fillna(0).iloc[0]
        for c in FOLDING_CLASSES
    )
    stab_run = sum(
        pd.to_numeric(
            pd.Series([row[f"{c}_segment_longest_run_frac"]]), errors="coerce"
        ).fillna(0).iloc[0]
        for c in STABILITY_CLASSES
    )

    if fold_run > stab_run:
        return "folding"
    if stab_run > fold_run:
        return "stability"

    fold_count = sum(
        pd.to_numeric(
            pd.Series([row[f"{c}_segment_overlap_count"]]), errors="coerce"
        ).fillna(0).iloc[0]
        for c in FOLDING_CLASSES
    )
    stab_count = sum(
        pd.to_numeric(
            pd.Series([row[f"{c}_segment_overlap_count"]]), errors="coerce"
        ).fillna(0).iloc[0]
        for c in STABILITY_CLASSES
    )

    if fold_count > stab_count:
        return "folding"
    if stab_count > fold_count:
        return "stability"

    return "ambiguous"


def summarize_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = (
        df[col]
        .value_counts(dropna=False)
        .rename_axis(col)
        .reset_index(name="n")
    )
    out["fraction"] = out["n"] / len(df) if len(df) > 0 else 0.0
    return out


def compare_summary(
    actual_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    key_col: str,
) -> pd.DataFrame:
    a = actual_summary.rename(columns={"n": "actual_n", "fraction": "actual_fraction"})
    r = random_summary.rename(columns={"n": "random_n", "fraction": "random_fraction"})

    out = pd.merge(a, r, on=key_col, how="outer").fillna(0)
    out["fraction_enrichment"] = out.apply(
        lambda x: (x["actual_fraction"] / x["random_fraction"])
        if float(x["random_fraction"]) > 0 else pd.NA,
        axis=1,
    )
    out = out.sort_values("actual_n", ascending=False).reset_index(drop=True)
    return out


def annotate_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        hit_flags: Dict[str, bool] = {cls: class_hit(row, cls) for cls in ALL_CLASSES}
        hit_classes = [cls for cls in ALL_CLASSES if hit_flags[cls]]

        folding_hits = [c for c in hit_classes if c in FOLDING_CLASSES]
        stability_hits = [c for c in hit_classes if c in STABILITY_CLASSES]

        n_hit = len(hit_classes)

        if n_hit == 0:
            type_level_1 = "no_hit"
        elif n_hit == 1:
            type_level_1 = "single_type"
        else:
            type_level_1 = "multi_type"

        if n_hit == 0:
            type_level_2 = "no_hit"
        elif len(folding_hits) > 0 and len(stability_hits) == 0:
            type_level_2 = "folding_only"
        elif len(stability_hits) > 0 and len(folding_hits) == 0:
            type_level_2 = "stability_only"
        else:
            type_level_2 = "folding_stability_mixed"

        dominant_class = pick_dominant_class(row, hit_classes)
        dominant_family = pick_dominant_family(row, hit_classes)

        out_row = row.to_dict()
        for cls in ALL_CLASSES:
            out_row[f"{cls}_hit"] = int(hit_flags[cls])

        out_row["n_hit_classes"] = n_hit
        out_row["hit_class_list"] = ";".join(hit_classes)
        out_row["folding_hit_list"] = ";".join(folding_hits)
        out_row["stability_hit_list"] = ";".join(stability_hits)
        out_row["type_level_1"] = type_level_1
        out_row["type_level_2"] = type_level_2
        out_row["dominant_class"] = dominant_class
        out_row["dominant_family"] = dominant_family
        out_row["exact_class_combo"] = ";".join(sorted(hit_classes)) if hit_classes else "no_hit"

        rows.append(out_row)

    return pd.DataFrame(rows)


def per_stf_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stf_id, sub in df.groupby("STF_ID", sort=False):
        rows.append({
            "STF_ID": stf_id,
            "n_rows": len(sub),
            "n_single_type": int((sub["type_level_1"] == "single_type").sum()),
            "n_multi_type": int((sub["type_level_1"] == "multi_type").sum()),
            "n_no_hit": int((sub["type_level_1"] == "no_hit").sum()),
            "n_folding_only": int((sub["type_level_2"] == "folding_only").sum()),
            "n_stability_only": int((sub["type_level_2"] == "stability_only").sum()),
            "n_folding_stability_mixed": int((sub["type_level_2"] == "folding_stability_mixed").sum()),
            "dominant_classes_seen": ";".join(sorted(sub["dominant_class"].astype(str).unique())),
            "exact_combos_seen": ";".join(sorted(sub["exact_class_combo"].astype(str).unique())),
        })
    return pd.DataFrame(rows)


# =========================================================
# 
# =========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate Start2Fold real/random fragments by exact class combinations and compare distributions.")
    p.add_argument("--real-input-file", type=Path, default=REAL_INPUT_FILE, help="all_mapped_segments_with_start2fold_metrics.csv")
    p.add_argument("--random-input-file", type=Path, default=RANDOM_INPUT_FILE, help="random_sampled_windows_all.csv")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory.")
    p.add_argument("--thresh-mode", choices=["count", "frac"], default=THRESH_MODE, help="Class-hit threshold mode.")
    p.add_argument("--count-threshold", type=float, default=COUNT_THRESHOLD, help="Hit-count threshold when --thresh-mode=count.")
    p.add_argument("--frac-threshold", type=float, default=FRAC_THRESHOLD, help="Hit-fraction threshold when --thresh-mode=frac.")
    return p.parse_args()

def main() -> None:
    global REAL_INPUT_FILE, RANDOM_INPUT_FILE, OUT_DIR, THRESH_MODE, COUNT_THRESHOLD, FRAC_THRESHOLD
    args = parse_args()
    REAL_INPUT_FILE = args.real_input_file
    RANDOM_INPUT_FILE = args.random_input_file
    OUT_DIR = args.out_dir
    THRESH_MODE = args.thresh_mode
    COUNT_THRESHOLD = args.count_threshold
    FRAC_THRESHOLD = args.frac_threshold
    ensure_dir(OUT_DIR)

    if not REAL_INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing real input file: {REAL_INPUT_FILE}")
    if not RANDOM_INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing random input file: {RANDOM_INPUT_FILE}")

    real_df = pd.read_csv(REAL_INPUT_FILE, dtype=str)
    random_df = pd.read_csv(RANDOM_INPUT_FILE, dtype=str)

    # 
    if "Map_Status" in real_df.columns:
        real_df = real_df[real_df["Map_Status"].astype(str) == "OK"].copy()

    if real_df.empty:
        raise RuntimeError("No mapped OK segments found in real input file.")
    if random_df.empty:
        raise RuntimeError("No rows found in random input file.")

    if "segment_uid" not in real_df.columns:
        raise ValueError("real_df is missing required column: segment_uid")
    if "segment_uid" not in random_df.columns:
        raise ValueError("random_df is missing required column: segment_uid")

    # =====================================================
    # “”
    #  STF_ID ，Target_Seq_Start / Target_Seq_End ，
    # ，“”
    # =====================================================
    real_before = len(real_df)
    random_before = len(random_df)

    real_df = dedupe_real_by_exact_position(
        real_df,
        id_col="STF_ID",
        start_col="Target_Seq_Start",
        end_col="Target_Seq_End",
    )

    kept_segment_uids = set(real_df["segment_uid"].astype(str))

    random_df = random_df[
        random_df["segment_uid"].astype(str).isin(kept_segment_uids)
    ].copy().reset_index(drop=True)

    print(f"[INFO] Real rows before exact-position dedupe: {real_before}")
    print(f"[INFO] Real rows after exact-position dedupe:  {len(real_df)}")
    print(f"[INFO] Random rows before segment_uid filtering: {random_before}")
    print(f"[INFO] Random rows after segment_uid filtering:  {len(random_df)}")

    # =====================================================
    #  / 
    # =====================================================
    real_ann = annotate_df(real_df)
    random_ann = annotate_df(random_df)

    real_ann_out = OUT_DIR / "actual_segment_multitype_annotations.csv"
    random_ann_out = OUT_DIR / "random_window_multitype_annotations.csv"
    real_ann.to_csv(real_ann_out, index=False, encoding="utf-8-sig")
    random_ann.to_csv(random_ann_out, index=False, encoding="utf-8-sig")

    # =====================================================
    #  summary： /  / 
    # =====================================================
    summary_targets = [
        "type_level_1",
        "type_level_2",
        "dominant_class",
        "dominant_family",
        "exact_class_combo",
    ]

    comparison_paths = []

    for col in summary_targets:
        real_sum = summarize_counts(real_ann, col)
        random_sum = summarize_counts(random_ann, col)
        comp_sum = compare_summary(real_sum, random_sum, col)

        real_path = OUT_DIR / f"actual_{col}_summary.csv"
        random_path = OUT_DIR / f"random_{col}_summary.csv"
        comp_path = OUT_DIR / f"actual_vs_random_{col}_summary.csv"

        real_sum.to_csv(real_path, index=False, encoding="utf-8-sig")
        random_sum.to_csv(random_path, index=False, encoding="utf-8-sig")
        comp_sum.to_csv(comp_path, index=False, encoding="utf-8-sig")

        comparison_paths.append(comp_path)

    # =====================================================
    # per STF
    # =====================================================
    real_per_stf = per_stf_summary(real_ann)
    random_per_stf = per_stf_summary(random_ann)

    real_per_stf_out = OUT_DIR / "actual_per_STF_multitype_summary.csv"
    random_per_stf_out = OUT_DIR / "random_per_STF_multitype_summary.csv"

    real_per_stf.to_csv(real_per_stf_out, index=False, encoding="utf-8-sig")
    random_per_stf.to_csv(random_per_stf_out, index=False, encoding="utf-8-sig")

    # =====================================================
    # summary.txt
    # =====================================================
    summary_lines = [
        f"Real input file: {REAL_INPUT_FILE}",
        f"Random input file: {RANDOM_INPUT_FILE}",
        f"Threshold mode: {THRESH_MODE}",
        f"Count threshold: {COUNT_THRESHOLD}",
        f"Frac threshold: {FRAC_THRESHOLD}",
        f"Real rows before exact-position dedupe: {real_before}",
        f"Real rows after exact-position dedupe: {len(real_df)}",
        f"Random rows before segment_uid filtering: {random_before}",
        f"Random rows after segment_uid filtering: {len(random_df)}",
        f"Actual mapped OK segments analyzed: {len(real_ann)}",
        f"Random windows analyzed: {len(random_ann)}",
        f"Actual no_hit: {(real_ann['type_level_1'] == 'no_hit').sum()}",
        f"Actual single_type: {(real_ann['type_level_1'] == 'single_type').sum()}",
        f"Actual multi_type: {(real_ann['type_level_1'] == 'multi_type').sum()}",
        f"Actual folding_only: {(real_ann['type_level_2'] == 'folding_only').sum()}",
        f"Actual stability_only: {(real_ann['type_level_2'] == 'stability_only').sum()}",
        f"Actual folding_stability_mixed: {(real_ann['type_level_2'] == 'folding_stability_mixed').sum()}",
        f"Random no_hit: {(random_ann['type_level_1'] == 'no_hit').sum()}",
        f"Random single_type: {(random_ann['type_level_1'] == 'single_type').sum()}",
        f"Random multi_type: {(random_ann['type_level_1'] == 'multi_type').sum()}",
        f"Random folding_only: {(random_ann['type_level_2'] == 'folding_only').sum()}",
        f"Random stability_only: {(random_ann['type_level_2'] == 'stability_only').sum()}",
        f"Random folding_stability_mixed: {(random_ann['type_level_2'] == 'folding_stability_mixed').sum()}",
        f"Actual annotation file: {real_ann_out}",
        f"Random annotation file: {random_ann_out}",
        f"Actual per-STF summary: {real_per_stf_out}",
        f"Random per-STF summary: {random_per_stf_out}",
    ] + [f"Comparison file: {p}" for p in comparison_paths]

    summary_out = OUT_DIR / "summary.txt"
    summary_out.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    for line in summary_lines:
        print("[INFO]", line)


if __name__ == "__main__":
    main()