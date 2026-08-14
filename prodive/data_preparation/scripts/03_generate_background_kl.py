#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate per-window HHM background-information arrays.

For each HHM record, this script reads the HHsuite NULL background row and
match-state emission rows. It computes a per-column information value:

    I_t = KL(p_t || q)

where p_t is the match-state emission distribution at column t and q is the
NULL/background amino-acid distribution. The per-column values are then mapped
onto fixed-length sliding windows by taking the window mean:

    K(i) = mean(I_i, ..., I_{i+k-1})

The resulting JSON is used by the filtering stage to suppress low-information
or background-like window pairs.

The implementation accepts arbitrary HHM filenames. Pfam-style names are not
required. Record names are matched against the mapping file produced by
01_generate_pfam_hhm_mapping.py.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


DEFAULT_FRAGMENT = "6"
DEFAULT_ROUND_DIGITS = 4


# ==============================================================================
# Mapping and HHM discovery
# ==============================================================================


def load_mapping(mapping_file: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load a mapping file with lines such as `record_id: 0`."""
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}

    with mapping_file.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or ":" not in line:
                continue
            left, right = line.split(":", 1)
            record_id = left.strip()
            try:
                numeric_id = int(right.strip())
            except ValueError:
                print(f"[WARN] Bad mapping value at line {line_no}: {line}", file=sys.stderr)
                continue

            if record_id in name_to_id:
                raise ValueError(f"Duplicate record ID in mapping file: {record_id!r}")
            if numeric_id in id_to_name:
                raise ValueError(
                    f"Duplicate numeric ID in mapping file: {numeric_id}; "
                    f"records={id_to_name[numeric_id]!r}, {record_id!r}"
                )

            name_to_id[record_id] = numeric_id
            id_to_name[numeric_id] = record_id

    if not name_to_id:
        raise RuntimeError(f"No valid mapping entries were loaded from: {mapping_file}")

    return name_to_id, id_to_name


def mapping_candidates_for_path(hhm_file_path: Path, input_dir: Path) -> List[str]:
    """Return possible mapping keys for one HHM file."""
    path_obj = hhm_file_path.resolve()
    root_obj = input_dir.resolve()

    candidates: List[str] = []
    try:
        rel = path_obj.relative_to(root_obj).as_posix()
        rel_no_suffix = str(Path(rel).with_suffix(""))
        candidates.extend([rel, rel_no_suffix])
    except ValueError:
        pass

    candidates.extend([path_obj.name, path_obj.stem])

    unique: List[str] = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def scan_hhm_files(
    input_dir: Path,
    include_pattern: str,
    exclude_patterns: Sequence[str],
) -> List[Path]:
    """Recursively scan HHM files without assuming Pfam-style names."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    raw_files = sorted((p for p in input_dir.rglob(include_pattern) if p.is_file()), key=lambda p: p.as_posix())
    selected: List[Path] = []

    for path in raw_files:
        rel = path.relative_to(input_dir).as_posix()
        filename = path.name
        excluded = any(
            fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(rel, pattern)
            for pattern in exclude_patterns
        )
        if not excluded:
            selected.append(path)

    return selected


def resolve_hhm_records(
    input_dir: Path,
    mapping: Dict[str, int],
    include_pattern: str,
    exclude_patterns: Sequence[str],
) -> Tuple[Dict[str, Path], List[dict]]:
    """Match scanned HHM files to mapping records."""
    record_to_path: Dict[str, Path] = {}
    unmatched_files: List[dict] = []

    for path in scan_hhm_files(input_dir, include_pattern, exclude_patterns):
        matched_record: Optional[str] = None
        for candidate in mapping_candidates_for_path(path, input_dir):
            if candidate in mapping:
                matched_record = candidate
                break

        if matched_record is None:
            unmatched_files.append({
                "file": str(path),
                "reason": "no_mapping",
                "tried_mapping_keys": mapping_candidates_for_path(path, input_dir),
            })
            continue

        if matched_record in record_to_path:
            raise RuntimeError(
                f"Multiple HHM files matched the same record ID {matched_record!r}: "
                f"{record_to_path[matched_record]} and {path}"
            )

        record_to_path[matched_record] = path

    return record_to_path, unmatched_files


# ==============================================================================
# HHM parsing and background KL calculation
# ==============================================================================


def parse_hhm_column_kl_values(hhm_file_path: Path) -> Optional[List[float]]:
    """
    Parse one HHsuite .hhm file and return per-match-state KL values.

    The returned list length should equal the number of match-state columns that
    can be parsed from the file.
    """
    if not hhm_file_path.exists():
        return None

    background_probs = None
    column_kls: List[float] = []

    try:
        with hhm_file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue

                # HHsuite NULL background row. The first 20 values correspond to
                # the standard amino-acid background scores.
                if parts[0] == "NULL":
                    if len(parts) < 21:
                        continue
                    raw_scores = np.asarray([float(x) for x in parts[1:21]], dtype=np.float64)
                    background_probs = np.power(2.0, -raw_scores / 1000.0)
                    bg_sum = float(background_probs.sum())
                    if np.isfinite(bg_sum) and bg_sum > 0:
                        background_probs = background_probs / bg_sum
                    else:
                        background_probs = None
                    continue

                # Match-state emission row. This follows the original parser rule:
                # residue letter, match-state index, then 20 emission scores.
                if (
                    len(parts) > 21
                    and parts[0].isalpha()
                    and parts[0].isupper()
                    and len(parts) > 1
                    and parts[1].isdigit()
                ):
                    if background_probs is None:
                        continue

                    raw_emission_scores = np.asarray([float(x) for x in parts[2:22]], dtype=np.float64)
                    with np.errstate(over="ignore", invalid="ignore"):
                        emission_probs = np.power(2.0, -raw_emission_scores / 1000.0)

                    emission_sum = float(emission_probs.sum())
                    if not np.isfinite(emission_sum) or emission_sum <= 0:
                        continue
                    emission_probs = emission_probs / emission_sum

                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.divide(
                            emission_probs,
                            background_probs,
                            out=np.ones_like(emission_probs),
                            where=background_probs > 0,
                        )
                        terms = emission_probs * np.log(ratio)

                    kl_value = float(np.nansum(terms))
                    if np.isfinite(kl_value):
                        column_kls.append(kl_value)

        return column_kls

    except Exception as exc:
        print(f"[WARN] Failed to parse {hhm_file_path}: {exc}", file=sys.stderr)
        return None


def sliding_window_mean(values: Sequence[float], fragment_length: int, round_digits: int) -> List[float]:
    """Convert per-column KL values into per-window background values."""
    if fragment_length <= 0:
        raise ValueError("fragment_length must be positive")

    n = len(values)
    if n < fragment_length:
        return []

    arr = np.asarray(values, dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    win_sum = csum[fragment_length:] - csum[:-fragment_length]
    win_mean = win_sum / float(fragment_length)
    return [round(float(x), round_digits) for x in win_mean]


def atomic_write_json(obj: object, path: Path, indent: Optional[int] = None) -> None:
    """Write a JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


# ==============================================================================
# Main run
# ==============================================================================


def parse_fragment_list(text: str) -> List[int]:
    items: List[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError(f"Fragment length must be positive: {value}")
        items.append(value)
    if not items:
        raise ValueError("At least one fragment length is required")
    return items


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate background/window-information JSON files from HHsuite HHM files."
    )
    parser.add_argument("--input_dir", "-i", type=Path, required=True,
                        help="Root directory containing .hhm files.")
    parser.add_argument("--mapping", "-m", type=Path, required=True,
                        help="HHM mapping file generated by 01_generate_pfam_hhm_mapping.py.")
    parser.add_argument("--output_dir", "-o", type=Path, required=True,
                        help="Output directory for background JSON files and summary.")
    parser.add_argument("--fragments", default=DEFAULT_FRAGMENT,
                        help="Comma-separated fragment lengths. Default: 6")
    parser.add_argument("--output_json", type=Path, default=None,
                        help="Optional explicit output JSON path. Valid only when exactly one fragment is requested.")
    parser.add_argument("--round_digits", type=int, default=DEFAULT_ROUND_DIGITS,
                        help="Decimal digits used for stored window-background values. Default: 4")
    parser.add_argument("--include_pattern", default="*.hhm",
                        help="Recursive filename pattern passed to Path.rglob. Default: *.hhm")
    parser.add_argument("--exclude_pattern", action="append", default=[],
                        help="Optional pattern for excluding HHM files. Can be supplied multiple times.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with an error if any mapped HHM is missing or fails to parse.")
    parser.add_argument("--pretty_json", action="store_true",
                        help="Write indented JSON. This is larger but easier to inspect.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    fragments = parse_fragment_list(args.fragments)

    if args.output_json is not None and len(fragments) != 1:
        raise ValueError("--output_json can only be used when exactly one fragment length is requested")

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping, id_to_name = load_mapping(args.mapping)
    record_to_path, unmatched_files = resolve_hhm_records(
        input_dir=args.input_dir,
        mapping=mapping,
        include_pattern=args.include_pattern,
        exclude_patterns=args.exclude_pattern,
    )

    ordered_records = [id_to_name[i] for i in sorted(id_to_name)]
    missing_records = [record for record in ordered_records if record not in record_to_path]

    print("=" * 100)
    print("Generate HHM background KL JSON")
    print(f"Input directory : {args.input_dir}")
    print(f"Mapping file    : {args.mapping}")
    print(f"Output directory: {args.output_dir}")
    print(f"Fragments       : {fragments}")
    print(f"Round digits    : {args.round_digits}")
    print(f"Mapped records  : {len(mapping)}")
    print(f"Matched HHM files: {len(record_to_path)}")
    print(f"Missing records : {len(missing_records)}")
    print("=" * 100)

    per_record_column_kls: Dict[str, List[float]] = {}
    parse_failed: List[dict] = []

    iterator: Iterable[str] = ordered_records
    if HAS_TQDM:
        iterator = tqdm(ordered_records, desc="Parsing HHM files", unit="record")

    for record_id in iterator:
        hhm_path = record_to_path.get(record_id)
        if hhm_path is None:
            continue
        kls = parse_hhm_column_kl_values(hhm_path)
        if kls is None:
            parse_failed.append({"record_id": record_id, "file": str(hhm_path), "reason": "parse_failed"})
            continue
        per_record_column_kls[record_id] = kls

    if args.strict and (missing_records or parse_failed):
        raise RuntimeError(
            f"Strict mode failed: missing_records={len(missing_records)}, parse_failed={len(parse_failed)}"
        )

    json_indent = 2 if args.pretty_json else None
    outputs: List[dict] = []

    for fragment in fragments:
        result: Dict[str, List[float]] = {}
        empty_records = 0

        iterator2 = per_record_column_kls.items()
        if HAS_TQDM:
            iterator2 = tqdm(list(iterator2), desc=f"Generating k={fragment}", unit="record")

        for record_id, column_kls in iterator2:
            values = sliding_window_mean(column_kls, fragment_length=fragment, round_digits=args.round_digits)
            if not values:
                empty_records += 1
            result[record_id] = values

        if args.output_json is not None:
            out_json = args.output_json
        else:
            out_json = args.output_dir / f"background_kl_fragment_{fragment}.json"

        atomic_write_json(result, out_json, indent=json_indent)
        total_windows = sum(len(v) for v in result.values())

        outputs.append({
            "fragment": int(fragment),
            "records": int(len(result)),
            "empty_records": int(empty_records),
            "total_windows": int(total_windows),
            "output_json": str(out_json),
        })

        print(
            f"[DONE] k={fragment}: records={len(result)}, empty={empty_records}, "
            f"total_windows={total_windows}, output={out_json}"
        )

    summary = {
        "input_dir": str(args.input_dir),
        "mapping_file": str(args.mapping),
        "output_dir": str(args.output_dir),
        "fragments": fragments,
        "round_digits": int(args.round_digits),
        "mapping_records": int(len(mapping)),
        "matched_hhm_files": int(len(record_to_path)),
        "parsed_records": int(len(per_record_column_kls)),
        "missing_records": missing_records,
        "parse_failed": parse_failed,
        "unmatched_hhm_files": unmatched_files,
        "outputs": outputs,
        "definition": "window_background_K = mean per-column KL(match-state emission || NULL background)",
        "coordinate_system": "Output arrays are indexed by 0-based window start in JSON; downstream PK output converts matrix coordinates to 1-based starts.",
        "duration_seconds": float(time.time() - t0),
    }

    summary_path = args.output_dir / "background_kl_generation_summary.json"
    atomic_write_json(summary, summary_path, indent=2)

    print("=" * 100)
    print("Background KL generation finished.")
    print(f"Summary: {summary_path}")
    print(f"Time: {summary['duration_seconds']:.1f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
