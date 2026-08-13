#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a stable HHM-record-to-integer mapping file.

The mapper accepts arbitrary HHM filenames, including names such as
protein-1a.hhm and graslim2.hhm. Pfam-style names are not required.

Default ID rule:
    record_id = HHM filename stem

Example:
    PF00001/PF00001.hhm  -> PF00001: 0
    protein-1a.hhm       -> protein-1a: 0
    graslim2.hhm         -> graslim2: 1

Duplicate record IDs are treated as errors by default. For datasets that
contain repeated filenames in different directories, use:
    --id_source relative_path
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_OUTPUT_FILE = Path("hhm_mapping.txt")
PF_PATTERN = re.compile(r"^PF(\d+)$")


@dataclass(frozen=True)
class HHMRecord:
    record_id: str
    path: Path
    relative_path: str
    stem: str


def is_pf_id(name: str) -> bool:
    return PF_PATTERN.match(name) is not None


def pf_number(name: str) -> int:
    match = PF_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Not a PFxxxxx ID: {name}")
    return int(match.group(1))


def make_record_id(hhm_path: Path, root_dir: Path, id_source: str) -> str:
    relative_path = hhm_path.relative_to(root_dir).as_posix()

    if id_source == "stem":
        return hhm_path.stem
    if id_source == "filename":
        return hhm_path.name
    if id_source == "relative_path":
        return relative_path

    raise ValueError(f"Unsupported id_source: {id_source}")


def validate_record_id(record_id: str, hhm_path: Path) -> None:
    if not record_id:
        raise ValueError(f"Empty record ID generated for {hhm_path}")
    if ":" in record_id:
        raise ValueError(
            f"Record ID contains ':' and cannot be written in 'ID: integer' format: {record_id!r}. "
            f"Source file: {hhm_path}"
        )


def collect_hhm_records(root_dir: Path, id_source: str, include_pattern: str) -> List[HHMRecord]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Input root directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {root_dir}")

    hhm_files = sorted((p for p in root_dir.rglob(include_pattern) if p.is_file()), key=lambda p: p.as_posix())
    if not hhm_files:
        raise RuntimeError(f"No HHM files matched {include_pattern!r} under: {root_dir}")

    records: List[HHMRecord] = []
    for hhm_path in hhm_files:
        record_id = make_record_id(hhm_path, root_dir, id_source)
        validate_record_id(record_id, hhm_path)
        records.append(
            HHMRecord(
                record_id=record_id,
                path=hhm_path,
                relative_path=hhm_path.relative_to(root_dir).as_posix(),
                stem=hhm_path.stem,
            )
        )

    return records


def check_duplicate_ids(records: Sequence[HHMRecord]) -> None:
    locations: Dict[str, List[str]] = {}
    for record in records:
        locations.setdefault(record.record_id, []).append(record.relative_path)

    duplicates = {record_id: paths for record_id, paths in locations.items() if len(paths) > 1}
    if not duplicates:
        return

    lines = ["Duplicate HHM record IDs were detected."]
    for record_id, paths in sorted(duplicates.items())[:20]:
        lines.append(f"  {record_id!r}: " + "; ".join(paths[:5]))
        if len(paths) > 5:
            lines.append(f"    ... {len(paths) - 5} additional files omitted")
    lines.append("Use --id_source relative_path when duplicate filenames or stems are expected.")
    raise RuntimeError("\n".join(lines))


def sort_records(records: Sequence[HHMRecord], sort_mode: str) -> List[HHMRecord]:
    if sort_mode == "pf_numeric_then_path":
        def key_func(record: HHMRecord):
            if is_pf_id(record.stem):
                return (0, pf_number(record.stem), record.relative_path)
            return (1, record.relative_path)
        return sorted(records, key=key_func)

    if sort_mode == "path":
        return sorted(records, key=lambda record: record.relative_path)

    if sort_mode == "id":
        return sorted(records, key=lambda record: record.record_id)

    raise ValueError(f"Unsupported sort_mode: {sort_mode}")


def write_mapping(records: Sequence[HHMRecord], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            f.write(f"{record.record_id}: {idx}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a stable integer mapping for HHM files. Pfam-style filenames are not required."
    )
    parser.add_argument("--root_dir", "-i", type=Path, required=True,
                        help="Root directory containing .hhm files.")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_FILE,
                        help=f"Output mapping file. Default: {DEFAULT_OUTPUT_FILE}")
    parser.add_argument("--id_source", choices=["stem", "filename", "relative_path"], default="stem",
                        help="Record ID source. Default: stem.")
    parser.add_argument("--sort_mode", choices=["pf_numeric_then_path", "path", "id"],
                        default="pf_numeric_then_path",
                        help="Mapping order. Default: Pfam numeric order first, then path order.")
    parser.add_argument("--include_pattern", default="*.hhm",
                        help="Recursive filename pattern passed to Path.rglob. Default: *.hhm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()

    try:
        records = collect_hhm_records(root_dir, args.id_source, args.include_pattern)
        check_duplicate_ids(records)
        ordered_records = sort_records(records, args.sort_mode)
        write_mapping(ordered_records, args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("HHM mapping finished.")
    print(f"Root dir          : {root_dir}")
    print(f"Matched HHM files : {len(records)}")
    print(f"ID source         : {args.id_source}")
    print(f"Sort mode         : {args.sort_mode}")
    print(f"Output file       : {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
