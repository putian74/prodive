#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter RMSD result CSV files by a stricter HHsearch probability threshold.

This script reproduces the post-RMSD HHsearch-probability filtering used for
Fig. 2E-G style comparisons. It is intentionally separated from the RMSD
calculation scripts because it operates only on already-computed CSV tables.

Outputs:
  1. high-prob HHsearch key table from hhsuite20.csv;
  2. high-prob HHsearch-only RMSD rows;
  3. high-prob shared rows evaluated with HHsearch-defined boundaries;
  4. high-prob shared rows evaluated with ProDive-defined boundaries via the
     HH segment -> ProDive segment mapping in the overlap-check CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


Key = Tuple[str, str, str]
REF_KEEP_COLS = ["Main_HMM", "Sub_HMM", "Sub_Segments_Details", "Prob"]
REF_KEY_COLS = ["Main_HMM", "Sub_HMM", "Sub_Segments_Details"]
TARGET_KEY_COLS = ["Main_HMM", "Sub_HMM", "Segment_Pair_Raw"]
OVERLAP_REQUIRED_COLS = [
    "Main_HMM", "Sub_HMM", "Main_Segment", "Sub_Segment", "Is_Novel",
    "HH_Main_Segment", "HH_Sub_Segment",
]


class CSVError(RuntimeError):
    pass


def normalize_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def normalize_segment(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def make_key(main_hmm: object, sub_hmm: object, segment: object) -> Key:
    return normalize_text(main_hmm), normalize_text(sub_hmm), normalize_segment(segment)


def parse_prob(value: object) -> float:
    text = normalize_text(value)
    if text == "":
        raise ValueError("empty Prob")
    return float(text)


def is_false_like(value: object) -> bool:
    text = normalize_text(value).lower()
    return text in {"false", "0", "no", "n"}


def ensure_columns(fieldnames: Sequence[str] | None, required: Sequence[str], csv_path: Path) -> None:
    if fieldnames is None:
        raise CSVError(f"CSV has no header: {csv_path}")
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise CSVError(f"Missing required columns in {csv_path}: {missing}. Available: {list(fieldnames)}")


def tuple_segment_to_range(value: object) -> str:
    nums = re.findall(r"-?\d+", normalize_text(value))
    if len(nums) >= 2:
        a, b = int(nums[0]), int(nums[1])
        if a > b:
            a, b = b, a
        return f"{a}-{b}"
    return normalize_segment(value)


def build_pair(main_segment: object, sub_segment: object) -> str:
    return f"{tuple_segment_to_range(main_segment)} -> {tuple_segment_to_range(sub_segment)}"


def read_high_prob_hhsearch(ref_csv: Path, prob_threshold: float) -> Tuple[List[Dict[str, str]], Set[Key], int]:
    selected: List[Dict[str, str]] = []
    keys: Set[Key] = set()
    total = 0
    with ref_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        ensure_columns(reader.fieldnames, REF_KEEP_COLS, ref_csv)
        for row in reader:
            total += 1
            try:
                prob = parse_prob(row.get("Prob"))
            except Exception:
                continue
            if prob > prob_threshold:
                kept = {c: row.get(c, "") for c in REF_KEEP_COLS}
                selected.append(kept)
                keys.add(make_key(row.get("Main_HMM", ""), row.get("Sub_HMM", ""), row.get("Sub_Segments_Details", "")))
    return selected, keys, total


def write_rows(output_csv: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, str]]) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return n


def filter_target_csv(target_csv: Path, output_csv: Path, allowed_keys: Set[Key]) -> Tuple[int, int]:
    matched: List[Dict[str, str]] = []
    total = 0
    with target_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        ensure_columns(reader.fieldnames, TARGET_KEY_COLS, target_csv)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            total += 1
            key = make_key(row.get("Main_HMM", ""), row.get("Sub_HMM", ""), row.get("Segment_Pair_Raw", ""))
            if key in allowed_keys:
                matched.append(row)
    kept = write_rows(output_csv, fieldnames, matched)
    return total, kept


def read_shared_overlap_rows(overlap_csv: Path) -> Tuple[List[Dict[str, str]], List[str], Set[Key], int]:
    rows: List[Dict[str, str]] = []
    hh_keys: Set[Key] = set()
    total = 0
    with overlap_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        ensure_columns(reader.fieldnames, OVERLAP_REQUIRED_COLS, overlap_csv)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            total += 1
            if not is_false_like(row.get("Is_Novel", "")):
                continue
            if not row.get("HH_Main_Segment") or not row.get("HH_Sub_Segment"):
                continue
            rows.append(row)
            hh_keys.add(make_key(row.get("Main_HMM", ""), row.get("Sub_HMM", ""), build_pair(row.get("HH_Main_Segment", ""), row.get("HH_Sub_Segment", ""))))
    return rows, fieldnames, hh_keys, total


def match_reference_keys(rows: Sequence[Dict[str, str]], allowed_hh_keys: Set[Key]) -> Tuple[List[Dict[str, str]], Set[Key]]:
    matched_rows: List[Dict[str, str]] = []
    matched_keys: Set[Key] = set()
    for row in rows:
        key = make_key(row.get("Main_HMM", ""), row.get("Sub_HMM", ""), row.get("Sub_Segments_Details", ""))
        if key in allowed_hh_keys:
            matched_rows.append(row)
            matched_keys.add(key)
    return matched_rows, matched_keys


def derive_prodive_keys(shared_rows: Sequence[Dict[str, str]], matched_hh_keys: Set[Key]) -> Tuple[List[Dict[str, str]], Set[Key]]:
    matched_shared: List[Dict[str, str]] = []
    prodive_keys: Set[Key] = set()
    for row in shared_rows:
        hh_key = make_key(row.get("Main_HMM", ""), row.get("Sub_HMM", ""), build_pair(row.get("HH_Main_Segment", ""), row.get("HH_Sub_Segment", "")))
        if hh_key not in matched_hh_keys:
            continue
        matched_shared.append(row)
        prodive_keys.add(make_key(row.get("Main_HMM", ""), row.get("Sub_HMM", ""), build_pair(row.get("Main_Segment", ""), row.get("Sub_Segment", ""))))
    return matched_shared, prodive_keys


def default_out(input_path: Path, output_dir: Path, suffix: str) -> Path:
    return output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter RMSD CSVs by HHsearch probability.")
    p.add_argument("--hhsuite-csv", required=True, type=Path, help="Parsed HHsearch table, e.g. hhsuite20.csv")
    p.add_argument("--overlap-csv", required=True, type=Path, help="subset_data_Top_100%_overlap_check_dual_20.csv")
    p.add_argument("--hhsuite-only-rmsd", required=True, type=Path)
    p.add_argument("--shared-prodive-rmsd", required=True, type=Path, help="shared rows evaluated by ProDive boundaries")
    p.add_argument("--shared-hh-rmsd", required=True, type=Path, help="shared rows evaluated by HHsearch boundaries")
    p.add_argument("--prob-threshold", type=float, default=70.0)
    p.add_argument("--output-dir", required=True, type=Path)
    return p


def main() -> int:
    args = build_parser().parse_args()
    for p in [args.hhsuite_csv, args.overlap_csv, args.hhsuite_only_rmsd, args.shared_prodive_rmsd, args.shared_hh_rmsd]:
        if not p.exists():
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
            return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = str(int(args.prob_threshold)) if float(args.prob_threshold).is_integer() else str(args.prob_threshold)

    ref_out = default_out(args.hhsuite_csv, args.output_dir, f"_prob_gt{tag}_keys")
    hh_only_out = default_out(args.hhsuite_only_rmsd, args.output_dir, f"_matched_prob_gt{tag}")
    shared_hh_out = default_out(args.shared_hh_rmsd, args.output_dir, f"_matched_prob_gt{tag}")
    matched_ref_out = default_out(ref_out, args.output_dir, "_matched_shared_hh_segments")
    matched_overlap_out = default_out(args.overlap_csv, args.output_dir, f"_shared_rows_hh_prob_gt{tag}_matched")
    shared_prodive_out = default_out(args.shared_prodive_rmsd, args.output_dir, f"_special_matched_prob_gt{tag}")

    high_prob_rows, high_prob_keys, ref_total = read_high_prob_hhsearch(args.hhsuite_csv, args.prob_threshold)
    write_rows(ref_out, REF_KEEP_COLS, high_prob_rows)

    hh_total, hh_kept = filter_target_csv(args.hhsuite_only_rmsd, hh_only_out, high_prob_keys)
    shh_total, shh_kept = filter_target_csv(args.shared_hh_rmsd, shared_hh_out, high_prob_keys)

    shared_rows, overlap_fieldnames, shared_hh_keys, overlap_total = read_shared_overlap_rows(args.overlap_csv)
    matched_ref_rows, matched_hh_keys = match_reference_keys(high_prob_rows, shared_hh_keys)
    write_rows(matched_ref_out, REF_KEEP_COLS, matched_ref_rows)

    matched_overlap_rows, prodive_keys = derive_prodive_keys(shared_rows, matched_hh_keys)
    write_rows(matched_overlap_out, overlap_fieldnames, matched_overlap_rows)
    sp_total, sp_kept = filter_target_csv(args.shared_prodive_rmsd, shared_prodive_out, prodive_keys)

    print("=== HHsearch probability filtering completed ===")
    print(f"HHsearch reference rows scanned: {ref_total}")
    print(f"HHsearch rows with Prob > {args.prob_threshold}: {len(high_prob_rows)}")
    print(f"High-prob key count: {len(high_prob_keys)}")
    print(f"HHsearch-only RMSD kept: {hh_kept} / {hh_total} -> {hh_only_out}")
    print(f"Shared HH-boundary RMSD kept: {shh_kept} / {shh_total} -> {shared_hh_out}")
    print(f"Overlap shared rows scanned: {overlap_total}; false/shared with HH segments: {len(shared_rows)}")
    print(f"Matched high-prob HH keys in shared rows: {len(matched_hh_keys)}")
    print(f"Shared ProDive-boundary RMSD kept: {sp_kept} / {sp_total} -> {shared_prodive_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CSVError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
