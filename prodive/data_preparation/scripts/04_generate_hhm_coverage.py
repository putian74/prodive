#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate per-match-state MSA coverage tables from HHM files and their source alignments.

This script uses the explicit Match_State -> MSA_Column mapping recorded in the
HHsuite HHM file. It does not infer match columns from a residue-fraction cutoff
such as `hhmake -M 50`.

For each HHM match state, the script locates the corresponding MSA column in the
paired alignment file, counts non-gap residues, and writes:

    Match_State,MSA_Column,Residue_Count,Total_Seqs,Completeness_Percent

Supported alignment formats:
    .sto / .stockholm   -> Biopython AlignIO format "stockholm"
    .fas / .fasta / .fa -> Biopython AlignIO format "fasta"
    .a3m / .afa         -> Biopython AlignIO format "fasta"

The batch mode assumes the common layout:

    root/PF00001/PF00001.hhm
    root/PF00001/PF00001.sto   # preferred if present
    root/PF00001/PF00001.fas   # fallback

but it also supports arbitrary HHM names as long as a same-stem alignment file is
present in the same directory or under a separate --alignment-root.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from Bio import AlignIO

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:  # pragma: no cover
    HAS_TQDM = False


GAP_CHARS = {"-", ".", "~"}
ALIGNMENT_EXTENSIONS_PRIORITY = (
    ".sto",
    ".stockholm",
    ".fas",
    ".fasta",
    ".fa",
    ".a3m",
    ".afa",
)


# ==============================================================================
# Name and path utilities
# ==============================================================================


def safe_output_stem(record_id: str) -> str:
    """Convert a record ID into a filesystem-safe filename stem."""
    text = str(record_id).strip().replace("\\", "/")
    text = re.sub(r"[/:]+", "__", text)
    text = re.sub(r"[^A-Za-z0-9._+-]+", "_", text)
    return text.strip("_") or "record"


def get_record_id(path: Path, root: Path, id_source: str) -> str:
    """Return the record ID used for naming coverage output files."""
    if id_source == "stem":
        return path.stem
    if id_source == "filename":
        return path.name
    if id_source == "relative_path":
        return path.relative_to(root).as_posix()
    raise ValueError(f"Unknown id_source: {id_source}")


def scan_hhm_files(root: Path) -> List[Path]:
    """Recursively scan HHM files under a root directory."""
    return sorted(p for p in root.rglob("*.hhm") if p.is_file())


def alignment_format(path: Path) -> str:
    """Return the Biopython AlignIO format for an alignment file."""
    suffix = path.suffix.lower()
    if suffix in {".sto", ".stockholm"}:
        return "stockholm"
    if suffix in {".fas", ".fasta", ".fa", ".a3m", ".afa"}:
        return "fasta"
    raise ValueError(f"Unsupported alignment extension: {path}")


def build_alignment_index(root: Path) -> Dict[str, List[Path]]:
    """Index alignment files under a root directory by filename stem."""
    index: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALIGNMENT_EXTENSIONS_PRIORITY:
            continue
        index.setdefault(p.stem, []).append(p)

    extension_rank = {ext: i for i, ext in enumerate(ALIGNMENT_EXTENSIONS_PRIORITY)}
    for paths in index.values():
        paths.sort(key=lambda x: (extension_rank.get(x.suffix.lower(), 999), str(x)))
    return index


def find_alignment_for_hhm(
    hhm_path: Path,
    hhm_root: Path,
    alignment_root: Optional[Path] = None,
    alignment_index: Optional[Dict[str, List[Path]]] = None,
) -> Optional[Path]:
    """
    Find the alignment paired with one HHM file.

    Search order:
        1. Same directory and same stem as the HHM, .sto preferred over .fas.
        2. Same relative path under --alignment-root, if supplied.
        3. Unique same-stem alignment discovered under --alignment-root.
    """
    for ext in ALIGNMENT_EXTENSIONS_PRIORITY:
        candidate = hhm_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    if alignment_root is not None:
        try:
            rel = hhm_path.relative_to(hhm_root)
            for ext in ALIGNMENT_EXTENSIONS_PRIORITY:
                candidate = (alignment_root / rel).with_suffix(ext)
                if candidate.exists():
                    return candidate
        except ValueError:
            pass

    if alignment_index is not None:
        matches = alignment_index.get(hhm_path.stem, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple alignment candidates found for {hhm_path.name}: "
                + "; ".join(str(x) for x in matches[:10])
            )

    return None


# ==============================================================================
# HHM parsing and coverage calculation
# ==============================================================================


def parse_hhm_match_to_msa_column(hhm_path: Path) -> Dict[int, int]:
    """
    Parse explicit HHM Match_State -> MSA_Column mapping.

    HHsuite HHM match-state rows contain the match-state index and, in the HHM
    files used by this pipeline, the corresponding MSA column as the last field.
    The parser enters the HMM block after the header line beginning with `HMM`
    and stops at `//`.
    """
    mapping: Dict[int, int] = {}

    with hhm_path.open("r", encoding="utf-8", errors="ignore") as handle:
        in_hmm_block = False
        for raw_line in handle:
            line = raw_line.strip()

            if line.startswith("HMM") and "A" in line and "C" in line:
                in_hmm_block = True
                continue
            if line.startswith("//"):
                break
            if not in_hmm_block:
                continue

            parts = line.split()
            if not (
                len(parts) > 20
                and len(parts[0]) == 1
                and parts[0].isalpha()
                and len(parts) > 1
                and parts[1].isdigit()
            ):
                continue

            try:
                state_idx = int(parts[1])
                msa_col_idx = int(parts[-1])
            except ValueError:
                continue

            mapping[state_idx] = msa_col_idx

    if not mapping:
        raise RuntimeError(f"No match-state to MSA-column mapping parsed from HHM: {hhm_path}")

    return mapping


def read_alignment(path: Path):
    """Read one MSA file with Biopython AlignIO."""
    fmt = alignment_format(path)
    return AlignIO.read(str(path), fmt)


def build_coverage_rows(
    hhm_path: Path,
    alignment_path: Path,
    completeness_digits: int = 4,
    skip_out_of_bounds: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Generate coverage rows for one HHM/alignment pair."""
    hhm_map = parse_hhm_match_to_msa_column(hhm_path)
    alignment = read_alignment(alignment_path)

    total_seqs = len(alignment)
    align_len = int(alignment.get_alignment_length())
    if total_seqs == 0 or align_len == 0:
        raise RuntimeError(f"Empty alignment: {alignment_path}")

    rows: List[Dict[str, object]] = []
    out_of_bounds = 0

    for state_idx in sorted(hhm_map.keys()):
        msa_col_1based = int(hhm_map[state_idx])
        msa_col_0based = msa_col_1based - 1

        if msa_col_0based < 0 or msa_col_0based >= align_len:
            out_of_bounds += 1
            if skip_out_of_bounds:
                continue
            raise RuntimeError(
                f"MSA column out of bounds for {hhm_path.name}: "
                f"state={state_idx}, msa_column={msa_col_1based}, alignment_length={align_len}"
            )

        column_chars = alignment[:, msa_col_0based]
        residue_count = sum(1 for char in column_chars if char not in GAP_CHARS)
        completeness = round(residue_count / float(total_seqs) * 100.0, completeness_digits)

        rows.append({
            "Match_State": int(state_idx),
            "MSA_Column": int(msa_col_1based),
            "Residue_Count": int(residue_count),
            "Total_Seqs": int(total_seqs),
            "Completeness_Percent": f"{completeness:.{completeness_digits}f}",
        })

    stats = {
        "total_seqs": int(total_seqs),
        "alignment_length": int(align_len),
        "hhm_match_states": int(len(hhm_map)),
        "coverage_rows": int(len(rows)),
        "out_of_bounds_states": int(out_of_bounds),
    }
    return rows, stats


def write_coverage_csv(rows: Sequence[Dict[str, object]], output_csv: Path) -> None:
    """Write coverage rows as CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Match_State",
        "MSA_Column",
        "Residue_Count",
        "Total_Seqs",
        "Completeness_Percent",
    ]

    tmp = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, output_csv)


# ==============================================================================
# Execution modes
# ==============================================================================


def run_single(args: argparse.Namespace) -> None:
    hhm_path = Path(args.hhm)
    alignment_path = Path(args.alignment)
    output_csv = Path(args.output_csv)

    rows, stats = build_coverage_rows(
        hhm_path=hhm_path,
        alignment_path=alignment_path,
        completeness_digits=args.completeness_digits,
        skip_out_of_bounds=not args.fail_on_out_of_bounds,
    )
    write_coverage_csv(rows, output_csv)

    print("=" * 80)
    print("HHM coverage table generated")
    print(f"HHM:        {hhm_path}")
    print(f"Alignment:  {alignment_path}")
    print(f"Rows:       {len(rows)}")
    print(f"Output CSV: {output_csv}")
    print(f"Stats:      {json.dumps(stats, ensure_ascii=False)}")
    print("=" * 80)


def _process_one_batch_task(task: Dict[str, object]) -> Dict[str, object]:
    hhm_path = Path(str(task["hhm_path"]))
    hhm_root = Path(str(task["hhm_root"]))
    output_dir = Path(str(task["output_dir"]))
    id_source = str(task["id_source"])
    completeness_digits = int(task["completeness_digits"])
    fail_on_out_of_bounds = bool(task["fail_on_out_of_bounds"])
    alignment_path = Path(str(task["alignment_path"])) if task.get("alignment_path") else None

    record_id = get_record_id(hhm_path, hhm_root, id_source)
    output_csv = output_dir / f"{safe_output_stem(record_id)}_coverage.csv"

    summary: Dict[str, object] = {
        "record_id": record_id,
        "id_source": id_source,
        "hhm_file": str(hhm_path),
        "alignment_file": str(alignment_path) if alignment_path else "",
        "output_csv": str(output_csv),
        "status": "",
        "reason": "",
        "rows": 0,
        "total_seqs": 0,
        "alignment_length": 0,
        "hhm_match_states": 0,
        "out_of_bounds_states": 0,
    }

    try:
        if alignment_path is None:
            raise FileNotFoundError(f"No matching alignment file found for {hhm_path}")

        rows, stats = build_coverage_rows(
            hhm_path=hhm_path,
            alignment_path=alignment_path,
            completeness_digits=completeness_digits,
            skip_out_of_bounds=not fail_on_out_of_bounds,
        )
        write_coverage_csv(rows, output_csv)

        summary.update({
            "status": "ok",
            "rows": int(len(rows)),
            "total_seqs": int(stats["total_seqs"]),
            "alignment_length": int(stats["alignment_length"]),
            "hhm_match_states": int(stats["hhm_match_states"]),
            "out_of_bounds_states": int(stats["out_of_bounds_states"]),
        })

    except Exception as exc:
        summary.update({
            "status": "error",
            "reason": repr(exc),
            "output_csv": "",
            "rows": 0,
        })

    return summary


def run_batch(args: argparse.Namespace) -> None:
    t0 = time.time()
    hhm_root = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    alignment_root = Path(args.alignment_root) if args.alignment_root else hhm_root

    if not hhm_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {hhm_root}")
    if not alignment_root.is_dir():
        raise FileNotFoundError(f"Alignment root not found: {alignment_root}")

    hhm_files = scan_hhm_files(hhm_root)
    if args.max_files is not None and args.max_files > 0:
        hhm_files = hhm_files[:args.max_files]
        print(f"[TEST MODE] Processing first {len(hhm_files)} HHM files only.")

    if not hhm_files:
        raise RuntimeError(f"No .hhm files found under: {hhm_root}")

    alignment_index = build_alignment_index(alignment_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, object]] = []
    output_stems = set()
    duplicate_stems: Dict[str, List[str]] = {}

    for hhm_path in hhm_files:
        record_id = get_record_id(hhm_path, hhm_root, args.id_source)
        stem = safe_output_stem(record_id)
        if stem in output_stems:
            duplicate_stems.setdefault(stem, []).append(str(hhm_path))
        output_stems.add(stem)

        alignment_path = find_alignment_for_hhm(
            hhm_path=hhm_path,
            hhm_root=hhm_root,
            alignment_root=alignment_root,
            alignment_index=alignment_index,
        )

        tasks.append({
            "hhm_path": str(hhm_path),
            "hhm_root": str(hhm_root),
            "output_dir": str(output_dir),
            "id_source": args.id_source,
            "alignment_path": str(alignment_path) if alignment_path else "",
            "completeness_digits": args.completeness_digits,
            "fail_on_out_of_bounds": args.fail_on_out_of_bounds,
        })

    if duplicate_stems:
        examples = "; ".join(f"{k}: {v[:3]}" for k, v in list(duplicate_stems.items())[:5])
        raise RuntimeError(
            "Duplicate coverage output stems detected. Use --id-source relative_path. "
            f"Examples: {examples}"
        )

    iterator: Iterable[Dict[str, object]]
    if args.jobs <= 1:
        iterator = map(_process_one_batch_task, tasks)
    else:
        pool = mp.Pool(processes=args.jobs)
        iterator = pool.imap_unordered(_process_one_batch_task, tasks)

    if HAS_TQDM:
        iterator = tqdm(iterator, total=len(tasks), desc="Generating coverage", unit="file")

    rows_out: List[Dict[str, object]] = []
    try:
        for summary in iterator:
            rows_out.append(summary)
    finally:
        if args.jobs > 1:
            pool.close()
            pool.join()

    ok_count = sum(1 for row in rows_out if row.get("status") == "ok")
    error_count = sum(1 for row in rows_out if row.get("status") == "error")

    summary_csv = output_dir / "coverage_generation_summary.csv"
    summary_json = output_dir / "coverage_generation_summary.json"

    fieldnames = [
        "status",
        "reason",
        "record_id",
        "id_source",
        "hhm_file",
        "alignment_file",
        "output_csv",
        "rows",
        "total_seqs",
        "alignment_length",
        "hhm_match_states",
        "out_of_bounds_states",
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump({
            "input_dir": str(hhm_root),
            "alignment_root": str(alignment_root),
            "output_dir": str(output_dir),
            "id_source": args.id_source,
            "jobs": int(args.jobs),
            "ok_files": int(ok_count),
            "error_files": int(error_count),
            "summary_csv": str(summary_csv),
            "method": "Coverage is computed from HHM explicit Match_State -> MSA_Column mapping; no -M 50 re-inference is performed.",
            "duration_seconds": time.time() - t0,
        }, handle, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("HHM coverage batch generation finished")
    print(f"OK files:     {ok_count}")
    print(f"Error files:  {error_count}")
    print(f"Output dir:   {output_dir}")
    print(f"Summary CSV:  {summary_csv}")
    print("=" * 80)

    if args.strict and error_count > 0:
        raise RuntimeError(f"Coverage generation finished with {error_count} error files.")


# ==============================================================================
# CLI
# ==============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate HHM match-state coverage CSV files from HHM explicit MSA-column mappings."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("single", help="Generate one coverage CSV from one HHM/alignment pair.")
    ps.add_argument("--hhm", required=True, help="Input HHM file.")
    ps.add_argument("--alignment", required=True,
                    help="Corresponding aligned MSA file (.sto, .fas, .fasta, .fa, .a3m, .afa).")
    ps.add_argument("--output-csv", required=True, help="Output coverage CSV file.")
    ps.add_argument("--completeness-digits", type=int, default=4,
                    help="Decimal places for Completeness_Percent. Default: 4.")
    ps.add_argument("--fail-on-out-of-bounds", action="store_true",
                    help="Fail if an HHM-reported MSA column exceeds the alignment length. Default: skip such states.")

    pb = sub.add_parser("batch", help="Generate coverage CSV files for all HHM files under a root directory.")
    pb.add_argument("--input-dir", required=True, help="HHM root directory.")
    pb.add_argument("--output-dir", required=True, help="Directory for coverage CSV files.")
    pb.add_argument("--alignment-root", default=None,
                    help="Optional root directory for alignment files. Defaults to --input-dir.")
    pb.add_argument("--id-source", choices=["stem", "filename", "relative_path"], default="stem",
                    help="Record naming strategy for output coverage filenames. Default: stem.")
    pb.add_argument("--jobs", type=int, default=1,
                    help="Number of worker processes. Default: 1.")
    pb.add_argument("--completeness-digits", type=int, default=4,
                    help="Decimal places for Completeness_Percent. Default: 4.")
    pb.add_argument("--fail-on-out-of-bounds", action="store_true",
                    help="Fail a family if an HHM-reported MSA column exceeds the alignment length. Default: skip such states.")
    pb.add_argument("--strict", action="store_true",
                    help="Exit with an error if any family fails in batch mode.")
    pb.add_argument("--max-files", type=int, default=None,
                    help="Optional test mode: process only the first N scanned HHM files.")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.mode == "single":
        run_single(args)
    elif args.mode == "batch":
        run_batch(args)
    else:  # pragma: no cover
        raise RuntimeError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
