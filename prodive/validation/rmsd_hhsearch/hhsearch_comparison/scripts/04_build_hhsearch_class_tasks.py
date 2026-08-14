#!/usr/bin/env python3
"""
Build ProDive/HHsearch comparison task tables.

This script generates two files:

1. false_tasks.csv
   ProDive results that overlap HHsearch hits. These are the shared / both class.
   They are taken from a ProDive overlap-check CSV where Is_Novel == False.

2. hhsuite_only_tasks.csv
   HHsearch hits whose unordered family pair does not appear in false_tasks.csv.
   This is a family-pair-level HHsearch-only definition used for downstream
   structural validation.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

Range = Tuple[int, int]
ParsedPair = Tuple[Range, Range, str]


def parse_bool_value(value) -> bool:
    """Parse pandas/CSV boolean-like values robustly."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        raise ValueError("Cannot parse NaN as boolean")

    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_range_any(value) -> Range:
    """
    Parse a segment range from formats such as:
    - '1-212'
    - '(1, 212)'
    - '[1, 212]'
    - '1, 212'
    """
    if pd.isna(value):
        raise ValueError("Cannot parse NaN as segment range")

    nums = re.findall(r"\d+", str(value))
    if len(nums) < 2:
        raise ValueError(f"Cannot parse segment range: {value!r}")

    start, end = int(nums[0]), int(nums[1])
    if start > end:
        raise ValueError(f"Invalid segment range with start > end: {value!r}")

    return start, end


def parse_hhsuite_details(details_str) -> List[ParsedPair]:
    """
    Parse HHsearch segment details.

    Expected examples:
    - '10-66 -> 2-43'
    - '10-66 -> 2-43, 80-95 -> 120-135'
    """
    if pd.isna(details_str):
        return []

    text = str(details_str).strip()
    if not text:
        return []

    parsed: List[ParsedPair] = []
    parts = [p.strip() for p in text.split(",") if p.strip()]

    for part in parts:
        m = re.search(r"(\d+)\s*-\s*(\d+)\s*->\s*(\d+)\s*-\s*(\d+)", part)
        if not m:
            continue

        a1, a2, b1, b2 = map(int, m.groups())
        if a1 > a2 or b1 > b2:
            raise ValueError(f"Invalid HHsearch segment pair: {part!r}")

        raw = f"{a1}-{a2} -> {b1}-{b2}"
        parsed.append(((a1, a2), (b1, b2), raw))

    return parsed


def unordered_pair(main_hmm: str, sub_hmm: str) -> Tuple[str, str]:
    """Return an unordered Pfam-family pair key."""
    a, b = str(main_hmm), str(sub_hmm)
    return tuple(sorted((a, b)))


def require_columns(df: pd.DataFrame, required_cols: List[str], file_label: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {missing}\n"
            f"Current columns: {list(df.columns)}"
        )


def build_false_tasks(prodive_overlap_csv: str, output_path: str, encoding: str) -> Tuple[pd.DataFrame, set]:
    """
    Build false_tasks.csv from an overlap-check CSV.

    Is_Novel == False is interpreted as shared/both with HHsearch.
    """
    print("=" * 80)
    print("Building false_tasks.csv from ProDive overlap-check results")
    print(f"Input:  {prodive_overlap_csv}")
    print(f"Output: {output_path}")
    print("=" * 80)

    df = pd.read_csv(prodive_overlap_csv, encoding=encoding)
    require_columns(
        df,
        ["Main_HMM", "Sub_HMM", "Main_Segment", "Sub_Segment", "Dual_Min_Coverage", "Is_Novel"],
        "ProDive overlap CSV",
    )

    records = []
    shared_pairs = set()

    for row in tqdm(df.itertuples(index=True), total=len(df), desc="Scanning ProDive overlap rows"):
        input_idx = row.Index
        is_novel = parse_bool_value(row.Is_Novel)
        if is_novel:
            continue

        main_hmm = row.Main_HMM
        sub_hmm = row.Sub_HMM
        a1, a2 = parse_range_any(row.Main_Segment)
        b1, b2 = parse_range_any(row.Sub_Segment)

        shared_pairs.add(unordered_pair(main_hmm, sub_hmm))

        records.append({
            "Input_Row_Index": int(input_idx),
            "Main_HMM": main_hmm,
            "Sub_HMM": sub_hmm,
            "a1": a1,
            "a2": a2,
            "b1": b1,
            "b2": b2,
            "Segment_Pair_Index": 1,
            "Segment_Pair_Raw": f"{a1}-{a2} -> {b1}-{b2}",
            "Source_Label": "FALSE_FROM_REF",
            "Dual_Min_Coverage": row.Dual_Min_Coverage,
            "Is_Novel": False,
        })

    out_df = pd.DataFrame(records, columns=[
        "Input_Row_Index",
        "Main_HMM",
        "Sub_HMM",
        "a1",
        "a2",
        "b1",
        "b2",
        "Segment_Pair_Index",
        "Segment_Pair_Raw",
        "Source_Label",
        "Dual_Min_Coverage",
        "Is_Novel",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved false_tasks rows: {len(out_df):,}")
    print(f"Shared unordered family pairs: {len(shared_pairs):,}")

    return out_df, shared_pairs


def build_hhsuite_only_tasks(
    hhsuite_csv: str,
    shared_pairs: set,
    output_path: str,
    encoding: str,
) -> pd.DataFrame:
    """
    Build hhsuite_only_tasks.csv.

    A HHsearch row is HHsearch-only if its unordered family pair does not appear
    in the shared-pair set derived from false_tasks.csv. If a HHsearch row contains
    multiple segment pairs in Sub_Segments_Details, each parsed segment pair is
    emitted as one task row.
    """
    print("\n" + "=" * 80)
    print("Building hhsuite_only_tasks.csv from HHsearch hits")
    print(f"Input:  {hhsuite_csv}")
    print(f"Output: {output_path}")
    print("=" * 80)

    df = pd.read_csv(hhsuite_csv, encoding=encoding)
    require_columns(
        df,
        ["Main_HMM", "Sub_HMM", "Sub_Segments_Details", "Prob"],
        "HHsearch CSV",
    )

    records = []
    skipped_shared_rows = 0
    rows_with_no_parsed_segment = 0

    for row in tqdm(df.itertuples(index=True), total=len(df), desc="Scanning HHsearch rows"):
        input_idx = row.Index
        main_hmm = row.Main_HMM
        sub_hmm = row.Sub_HMM

        if unordered_pair(main_hmm, sub_hmm) in shared_pairs:
            skipped_shared_rows += 1
            continue

        parsed_pairs = parse_hhsuite_details(row.Sub_Segments_Details)
        if not parsed_pairs:
            rows_with_no_parsed_segment += 1
            continue

        for pair_idx, ((a1, a2), (b1, b2), raw) in enumerate(parsed_pairs, start=1):
            records.append({
                "Input_Row_Index": int(input_idx),
                "Main_HMM": main_hmm,
                "Sub_HMM": sub_hmm,
                "a1": a1,
                "a2": a2,
                "b1": b1,
                "b2": b2,
                "Segment_Pair_Index": pair_idx,
                "Segment_Pair_Raw": raw,
                "Source_Label": "HHSUITE_ONLY",
                "Prob": row.Prob,
            })

    out_df = pd.DataFrame(records, columns=[
        "Input_Row_Index",
        "Main_HMM",
        "Sub_HMM",
        "a1",
        "a2",
        "b1",
        "b2",
        "Segment_Pair_Index",
        "Segment_Pair_Raw",
        "Source_Label",
        "Prob",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"HHsearch rows excluded as shared family pairs: {skipped_shared_rows:,}")
    print(f"HHsearch rows skipped because no segment pair could be parsed: {rows_with_no_parsed_segment:,}")
    print(f"Saved hhsuite_only task rows: {len(out_df):,}")

    return out_df


def write_summary(
    output_path: str,
    prodive_overlap_csv: str,
    hhsuite_csv: str,
    false_df: pd.DataFrame,
    hhsuite_only_df: pd.DataFrame,
    shared_pairs: set,
) -> None:
    summary = pd.DataFrame([
        {"metric": "prodive_overlap_csv", "value": prodive_overlap_csv},
        {"metric": "hhsuite_csv", "value": hhsuite_csv},
        {"metric": "false_tasks_rows", "value": len(false_df)},
        {"metric": "shared_unordered_family_pairs", "value": len(shared_pairs)},
        {"metric": "hhsuite_only_tasks_rows", "value": len(hhsuite_only_df)},
    ])
    summary.to_csv(output_path, index=False)
    print(f"\nSaved summary: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate false_tasks.csv and hhsuite_only_tasks.csv. "
            "This script classifies ProDive/HHsearch comparison results into shared/both "
            "and HHsearch-only task tables."
        )
    )

    parser.add_argument(
        "--prodive-overlap-csv",
        required=True,
        help="Path to subset_data_Top_100%%_overlap_check_dual_20.csv or equivalent overlap-check CSV.",
    )
    parser.add_argument(
        "--hhsuite-csv",
        required=True,
        help="Path to parsed HHsearch CSV, e.g. hhsuite20.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for false_tasks.csv and hhsuite_only_tasks.csv.",
    )
    parser.add_argument(
        "--false-output-name",
        default="false_tasks.csv",
        help="Output filename for shared/both ProDive tasks. Default: false_tasks.csv",
    )
    parser.add_argument(
        "--hhsuite-only-output-name",
        default="hhsuite_only_tasks.csv",
        help="Output filename for HHsearch-only tasks. Default: hhsuite_only_tasks.csv",
    )
    parser.add_argument(
        "--summary-output-name",
        default="classification_summary.csv",
        help="Output filename for summary statistics. Default: classification_summary.csv",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding used for reading input files. Default: utf-8-sig",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    false_output_path = os.path.join(args.output_dir, args.false_output_name)
    hhsuite_only_output_path = os.path.join(args.output_dir, args.hhsuite_only_output_name)
    summary_output_path = os.path.join(args.output_dir, args.summary_output_name)

    false_df, shared_pairs = build_false_tasks(
        prodive_overlap_csv=args.prodive_overlap_csv,
        output_path=false_output_path,
        encoding=args.encoding,
    )

    hhsuite_only_df = build_hhsuite_only_tasks(
        hhsuite_csv=args.hhsuite_csv,
        shared_pairs=shared_pairs,
        output_path=hhsuite_only_output_path,
        encoding=args.encoding,
    )

    write_summary(
        output_path=summary_output_path,
        prodive_overlap_csv=args.prodive_overlap_csv,
        hhsuite_csv=args.hhsuite_csv,
        false_df=false_df,
        hhsuite_only_df=hhsuite_only_df,
        shared_pairs=shared_pairs,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
