#!/usr/bin/env python3
"""Parsing helpers for frag_dom formatted records."""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def parse_len_from_header(header: str) -> Optional[int]:
    match = re.search(r"len=(\d+)", header)
    if match:
        return int(match.group(1))
    return None


def parse_frag_dom(line: str) -> List[Tuple[int, int]]:
    if not line.startswith("frag_dom="):
        raise ValueError(f"Expected frag_dom line, got: {line!r}")
    raw = line.split("=", 1)[1].strip()
    ranges: List[Tuple[int, int]] = []
    truncated_parts: List[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "more" in part or part.startswith("...") or part.startswith("…"):
            truncated_parts.append(part)
            continue
        if "-" not in part:
            raise ValueError(f"Invalid frag_dom range: {part!r}")
        lo, hi = part.split("-", 1)
        lo_i = int(lo)
        hi_i = int(hi)
        if lo_i <= 0 or hi_i <= 0:
            raise ValueError(f"Invalid frag_dom range (non-positive): {part!r}")
        if lo_i > hi_i:
            raise ValueError(f"Invalid frag_dom range (start>end): {part!r}")
        ranges.append((lo_i, hi_i))
    if truncated_parts:
        print(
            "[fragdom_utils] Warning: frag_dom line appears truncated; ignored tokens: "
            + ", ".join(truncated_parts),
            file=sys.stderr,
        )
    if not ranges:
        raise ValueError("frag_dom line contained no ranges")
    return ranges


def iter_fragdom_records(path: Path) -> Iterable[Tuple[str, List[Tuple[int, int]], str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    header: Optional[str] = None
    frag_line: Optional[str] = None
    seq_parts: List[str] = []
    expected_len: Optional[int] = None

    def _flush():
        if header is None:
            return None
        if frag_line is None:
            raise ValueError(f"Missing frag_dom for record: {header}")
        if not seq_parts:
            raise ValueError(f"Missing sequence for record: {header}")
        seq = "".join(seq_parts)
        if expected_len is not None and len(seq) != expected_len:
            raise ValueError(
                f"Sequence length mismatch for {header}: expected {expected_len}, got {len(seq)}"
            )
        ranges = parse_frag_dom(frag_line)
        return header, ranges, seq

    with opener(path, "rt") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                record = _flush()
                if record is not None:
                    yield record
                header = line[1:]
                frag_line = None
                seq_parts = []
                expected_len = parse_len_from_header(header)
                continue
            if header is None:
                raise ValueError(f"Found data before header: {line!r}")
            if frag_line is None:
                frag_line = line
                continue
            seq_parts.append(line)
            if expected_len is not None and sum(len(s) for s in seq_parts) > expected_len:
                raise ValueError(
                    f"Sequence length exceeds expected len={expected_len} for {header}"
                )

    record = _flush()
    if record is not None:
        yield record


def iter_fragdom_sequences(path: Path) -> Iterable[Tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    header: Optional[str] = None
    frag_line: Optional[str] = None
    seq_parts: List[str] = []
    expected_len: Optional[int] = None

    def _flush():
        if header is None:
            return None
        if frag_line is None:
            raise ValueError(f"Missing frag_dom for record: {header}")
        if not seq_parts:
            raise ValueError(f"Missing sequence for record: {header}")
        seq = "".join(seq_parts)
        if expected_len is not None and len(seq) != expected_len:
            raise ValueError(
                f"Sequence length mismatch for {header}: expected {expected_len}, got {len(seq)}"
            )
        return header, seq

    with opener(path, "rt") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                record = _flush()
                if record is not None:
                    yield record
                header = line[1:]
                frag_line = None
                seq_parts = []
                expected_len = parse_len_from_header(header)
                continue
            if header is None:
                raise ValueError(f"Found data before header: {line!r}")
            if frag_line is None:
                frag_line = line
                continue
            seq_parts.append(line)
            if expected_len is not None and sum(len(s) for s in seq_parts) > expected_len:
                raise ValueError(
                    f"Sequence length exceeds expected len={expected_len} for {header}"
                )

    record = _flush()
    if record is not None:
        yield record
