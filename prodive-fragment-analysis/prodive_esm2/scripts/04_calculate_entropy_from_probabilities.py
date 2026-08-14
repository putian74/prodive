#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze ESM masked-token outputs for frag_dom records.

Features:
1) Compare entropy distribution of conserved/frag_dom positions vs non-frag_dom positions
2) Compare high-sharing vs low-sharing fragment positions
   - preferred: use a real cluster/share mapping file
   - fallback: use frag_dom overlap coverage count as a proxy
3) Multiprocessing for per-record entropy computation
4) Progress bars for:
   - loading frag_dom records
   - loading ESM probability files
   - entropy computation
5) Duplicate ESM headers handling:
   - error
   - skip-identical (recommended; good for boundary-overlap duplicates)

Input A: frag_dom formatted records
>HEADER
frag_dom=10-20,31-40,...
SEQUENCE

Input B: ESM per-position probability output (one or more files)
>HEADER
A:... C:... D:... ... Y:...
A:... C:... D:... ... Y:...
...

"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import re
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


# =========================
# Parsing: frag_dom input
# =========================

def parse_len_from_header(header: str) -> Optional[int]:
    m = re.search(r"len=(\d+)", header)
    return int(m.group(1)) if m else None


def parse_frag_dom_line(line: str) -> List[Tuple[int, int]]:
    """
    Parse:
        frag_dom=30-40,31-40,31-41,...
    into:
        [(30,40), (31,40), (31,41), ...]
    """
    if not line.startswith("frag_dom="):
        raise ValueError(f"Expected frag_dom line, got: {line!r}")

    raw = line.split("=", 1)[1].strip()
    ranges: List[Tuple[int, int]] = []

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "more" in part or part.startswith("...") or part.startswith("…"):
            continue
        if "-" not in part:
            raise ValueError(f"Invalid frag_dom range: {part!r}")
        lo_s, hi_s = part.split("-", 1)
        lo = int(lo_s)
        hi = int(hi_s)
        if lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError(f"Invalid frag_dom range: {part!r}")
        ranges.append((lo, hi))

    if not ranges:
        raise ValueError("No valid frag_dom ranges found")

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
            raise ValueError(f"Missing frag_dom line for record: {header}")
        if not seq_parts:
            raise ValueError(f"Missing sequence for record: {header}")

        seq = "".join(seq_parts).strip().upper()
        if expected_len is not None and len(seq) != expected_len:
            raise ValueError(
                f"Sequence length mismatch for {header}: expected {expected_len}, got {len(seq)}"
            )

        ranges = parse_frag_dom_line(frag_line)
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
                raise ValueError(f"Found content before first header: {line!r}")

            if frag_line is None:
                frag_line = line
            else:
                seq_parts.append(line)

    record = _flush()
    if record is not None:
        yield record


def load_fragdom_records(path: Path) -> Dict[str, Tuple[List[Tuple[int, int]], str]]:
    """
    Load all frag_dom records into:
      header -> (ranges, seq)
    """
    records: Dict[str, Tuple[List[Tuple[int, int]], str]] = {}

    for header, ranges, seq in tqdm(
        iter_fragdom_records(path),
        desc="Loading frag_dom",
        unit="record",
    ):
        if header in records:
            raise ValueError(f"Duplicate header in fragdom file: {header}")
        records[header] = (ranges, seq)

    return records


# =========================
# Parsing: ESM output files
# =========================

def parse_prob_line(line: str) -> np.ndarray:
    """
    Parse one line like:
      A:0.12345 C:0.00012 D:0.01234 ... Y:0.00001
    Return np.array shape (20,)
    """
    vals = np.zeros(20, dtype=np.float64)

    parts = line.strip().split()
    seen = set()

    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid probability token: {part!r}")
        aa, prob_s = part.split(":", 1)
        aa = aa.strip()
        if aa not in AA_SET:
            raise ValueError(f"Unexpected amino acid token: {aa!r}")
        prob = float(prob_s)
        vals[AA_TO_IDX[aa]] = prob
        seen.add(aa)

    missing = [aa for aa in AA_ORDER if aa not in seen]
    if missing:
        raise ValueError(f"Probability line missing AAs: {missing}")

    return vals


def load_esm_prob_records(
    paths: List[Path],
    duplicate_policy: str = "skip-identical",
) -> Dict[str, np.ndarray]:
    """
    Load one or more ESM probability output files.

    Format:
      >HEADER
      A:... C:... ... Y:...
      ...
      >NEXT_HEADER
      ...

    duplicate_policy:
      - error
      - skip-identical
    """
    if duplicate_policy not in {"error", "skip-identical"}:
        raise ValueError("duplicate_policy must be one of: error, skip-identical")

    data: Dict[str, np.ndarray] = {}
    duplicate_identical = 0

    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        header: Optional[str] = None
        rows: List[np.ndarray] = []

        with opener(path, "rt") as f, tqdm(
            desc=f"Loading {path.name}",
            unit="record",
            leave=False,
        ) as pbar:

            def _flush():
                nonlocal header, rows, duplicate_identical
                if header is None:
                    return

                arr = np.vstack(rows) if rows else np.zeros((0, 20), dtype=np.float64)

                if header in data:
                    if duplicate_policy == "error":
                        raise ValueError(f"Duplicate header found across ESM output files: {header}")

                    # skip-identical
                    old = data[header]
                    same = (old.shape == arr.shape) and np.allclose(old, arr, rtol=0.0, atol=0.0)
                    if same:
                        duplicate_identical += 1
                        pbar.write(f"[WARN] Skip duplicate identical header: {header}")
                    else:
                        raise ValueError(
                            f"Duplicate header with DIFFERENT content found: {header}"
                        )
                else:
                    data[header] = arr

                pbar.update(1)

            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    _flush()
                    header = line[1:]
                    rows = []
                else:
                    if header is None:
                        raise ValueError(f"Probability line before header in {path}: {line!r}")
                    rows.append(parse_prob_line(line))

            _flush()

    if duplicate_identical > 0:
        print(f"[INFO] duplicate identical headers skipped: {duplicate_identical}")

    return data


# =========================
# Optional cluster/share map
# =========================

def sniff_delimiter(path: Path) -> str:
    with path.open("r", newline="") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        return dialect.delimiter
    except Exception:
        return "\t" if "\t" in sample else ","


def load_cluster_segments_map(path: Path) -> Tuple[Dict[str, List[Tuple[int, int, float]]], int]:
    """
    Expected columns:
      header, start, end, share_count

    Returns:
      header -> [(start, end, share_count), ...]
      total_row_count
    """
    delim = sniff_delimiter(path)
    df = pd.read_csv(path, sep=delim)

    required = {"header", "start", "end", "share_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"cluster-map must contain columns {sorted(required)}; missing {sorted(missing)}"
        )

    df = df.copy()
    df["header"] = df["header"].astype(str)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["share_count"] = pd.to_numeric(df["share_count"])

    bad = df[(df["start"] <= 0) | (df["end"] <= 0) | (df["start"] > df["end"])]
    if not bad.empty:
        raise ValueError("cluster-map contains invalid start/end ranges")

    mapping: Dict[str, List[Tuple[int, int, float]]] = {}
    for _, row in df.iterrows():
        mapping.setdefault(row["header"], []).append(
            (int(row["start"]), int(row["end"]), float(row["share_count"]))
        )

    return mapping, len(df)


# =========================
# Entropy + masks
# =========================

def shannon_entropy_from_probs(p20: np.ndarray, log_base: str = "e") -> Tuple[float, float]:
    """
    Compute entropy after renormalizing the 20-AA probabilities.

    Returns:
      entropy, aa_mass

    aa_mass = original sum over the 20 standard AAs before renormalization.
    """
    aa_mass = float(np.sum(p20))
    if aa_mass <= 0.0:
        return float("nan"), aa_mass

    p = p20 / aa_mass
    p = p[p > 0]

    if p.size == 0:
        return float("nan"), aa_mass

    h = -float(np.sum(p * np.log(p)))
    if log_base == "2":
        h /= math.log(2.0)

    return h, aa_mass


def build_frag_masks(seq_len: int, ranges: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      in_fragment: bool array shape (seq_len+1,), 1-indexed
      coverage_count: int array shape (seq_len+1,), number of overlapping frag_dom ranges covering each position
    """
    in_fragment = np.zeros(seq_len + 1, dtype=bool)
    coverage_count = np.zeros(seq_len + 1, dtype=np.int32)

    for start, end in ranges:
        if start < 1 or end > seq_len:
            raise ValueError(
                f"frag_dom range {start}-{end} exceeds sequence length {seq_len}"
            )
        in_fragment[start:end + 1] = True
        coverage_count[start:end + 1] += 1

    return in_fragment, coverage_count


def build_share_mask_from_segments(
    seq_len: int,
    segments: List[Tuple[int, int, float]],
) -> np.ndarray:
    """
    Per-position share_count from cluster/share segments.
    If overlapping ranges exist, use the maximum share_count at that position.
    """
    share_count = np.zeros(seq_len + 1, dtype=np.float64)

    for start, end, val in segments:
        if start < 1 or end > seq_len:
            raise ValueError(
                f"cluster-map range {start}-{end} exceeds sequence length {seq_len}"
            )
        share_count[start:end + 1] = np.maximum(share_count[start:end + 1], val)

    return share_count


def summarize_series(values: pd.Series, label: str) -> Dict[str, float]:
    s = values.dropna()
    if s.empty:
        return {
            "group": label,
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "group": label,
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
    }


def save_histogram(
    values_a: pd.Series,
    label_a: str,
    values_b: pd.Series,
    label_b: str,
    out_png: Path,
    title: str,
    bins: int = 80,
    density: bool = True,
) -> None:
    a = values_a.dropna().to_numpy()
    b = values_b.dropna().to_numpy()

    if a.size == 0 or b.size == 0:
        return

    plt.figure(figsize=(9, 6))
    plt.hist(a, bins=bins, alpha=0.55, density=density, label=label_a)
    plt.hist(b, bins=bins, alpha=0.55, density=density, label=label_b)
    plt.xlabel("Entropy")
    plt.ylabel("Density" if density else "Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================
# Worker
# =========================

def process_one_record(job: Tuple) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Worker-side per-record entropy computation.
    """
    (
        header,
        ranges,
        seq,
        probs,
        cluster_segments,
        log_base,
    ) = job

    seq_len = len(seq)
    if probs.shape[0] != seq_len:
        raise ValueError(
            f"Length mismatch for {header}: sequence length={seq_len}, ESM rows={probs.shape[0]}"
        )

    in_fragment, coverage_count = build_frag_masks(seq_len, ranges)

    if cluster_segments is not None:
        share_count_pos = build_share_mask_from_segments(seq_len, cluster_segments)
        share_source = "cluster_map"
    else:
        share_count_pos = coverage_count.astype(np.float64)
        share_source = "frag_dom_overlap_proxy"

    per_pos_rows: List[Dict[str, object]] = []
    record_entropy_fragment: List[float] = []
    record_entropy_background: List[float] = []

    for pos in range(1, seq_len + 1):
        p20 = probs[pos - 1]
        entropy, aa_mass = shannon_entropy_from_probs(p20, log_base=log_base)

        wt_aa = seq[pos - 1]
        wt_prob = np.nan
        if wt_aa in AA_TO_IDX:
            aa_mass_safe = float(np.sum(p20))
            if aa_mass_safe > 0:
                wt_prob = float(p20[AA_TO_IDX[wt_aa]] / aa_mass_safe)

        is_frag = bool(in_fragment[pos])
        cov = int(coverage_count[pos])
        share_val = float(share_count_pos[pos])

        if is_frag and not np.isnan(entropy):
            record_entropy_fragment.append(entropy)
        elif (not is_frag) and not np.isnan(entropy):
            record_entropy_background.append(entropy)

        per_pos_rows.append(
            {
                "header": header,
                "position": pos,
                "wt_aa": wt_aa,
                "is_fragment": is_frag,
                "fragment_coverage_count": cov,
                "share_count": share_val,
                "share_count_source": share_source,
                "entropy": entropy,
                "aa_mass_20aa": aa_mass,
                "wt_prob_20aa_renorm": wt_prob,
            }
        )

    frag_arr = np.array(record_entropy_fragment, dtype=np.float64)
    bg_arr = np.array(record_entropy_background, dtype=np.float64)

    per_record_row = {
        "header": header,
        "seq_len": seq_len,
        "n_fragment_positions": int(frag_arr.size),
        "n_background_positions": int(bg_arr.size),
        "mean_entropy_fragment": float(np.nanmean(frag_arr)) if frag_arr.size else np.nan,
        "median_entropy_fragment": float(np.nanmedian(frag_arr)) if frag_arr.size else np.nan,
        "mean_entropy_background": float(np.nanmean(bg_arr)) if bg_arr.size else np.nan,
        "median_entropy_background": float(np.nanmedian(bg_arr)) if bg_arr.size else np.nan,
        "delta_fragment_minus_background": (
            float(np.nanmean(frag_arr) - np.nanmean(bg_arr))
            if frag_arr.size and bg_arr.size
            else np.nan
        ),
    }

    return per_pos_rows, per_record_row


# =========================
# Main
# =========================

def main() -> None:
    p = argparse.ArgumentParser(description="Analyze ESM entropy for frag_dom conserved segments")
    p.add_argument("--fragdom", required=True, help="frag_dom formatted input file")
    p.add_argument(
        "--esm-prob-files",
        nargs="+",
        required=True,
        help="One or more ESM probability output files (.dat/.txt/.gz)",
    )
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument(
        "--cluster-map",
        default=None,
        help="Optional TSV/CSV with columns: header,start,end,share_count",
    )
    p.add_argument(
        "--log-base",
        choices=["e", "2"],
        default="e",
        help="Entropy log base (natural log by default, or bits with 2)",
    )
    p.add_argument(
        "--low-quantile",
        type=float,
        default=0.33,
        help="Quantile cutoff for low-share group among fragment positions",
    )
    p.add_argument(
        "--high-quantile",
        type=float,
        default=0.67,
        help="Quantile cutoff for high-share group among fragment positions",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of worker processes for per-record entropy calculation",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=16,
        help="Multiprocessing chunksize (bigger = less overhead, too big = less responsive)",
    )
    p.add_argument(
        "--duplicate-policy",
        choices=["error", "skip-identical"],
        default="skip-identical",
        help="How to handle duplicate headers across ESM output files",
    )
    args = p.parse_args()

    fragdom_path = Path(args.fragdom)
    prob_paths = [Path(x) for x in args.esm_prob_files]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not fragdom_path.exists():
        raise FileNotFoundError(f"fragdom file not found: {fragdom_path}")
    for path in prob_paths:
        if not path.exists():
            raise FileNotFoundError(f"ESM probability file not found: {path}")

    # ------------------------------
    # Load optional cluster/share map
    # ------------------------------
    cluster_segments_map: Optional[Dict[str, List[Tuple[int, int, float]]]] = None
    if args.cluster_map:
        cluster_map_path = Path(args.cluster_map)
        if not cluster_map_path.exists():
            raise FileNotFoundError(f"cluster-map not found: {cluster_map_path}")
        cluster_segments_map, n_rows = load_cluster_segments_map(cluster_map_path)
        print(f"[INFO] Loaded cluster/share map: {cluster_map_path} ({n_rows} rows)")
    else:
        print("[INFO] No cluster/share map provided. Using frag_dom overlap coverage_count as a temporary share proxy.")

    # ------------------------------
    # Load inputs
    # ------------------------------
    print("[INFO] Loading frag_dom records...")
    fragdom_records = load_fragdom_records(fragdom_path)
    print(f"[INFO] Loaded {len(fragdom_records)} frag_dom records")

    print("[INFO] Loading ESM probability files...")
    esm_probs = load_esm_prob_records(
        prob_paths,
        duplicate_policy=args.duplicate_policy,
    )
    print(f"[INFO] Loaded {len(esm_probs)} unique ESM records from {len(prob_paths)} file(s)")

    # ------------------------------
    # Build jobs
    # ------------------------------
    jobs: List[Tuple] = []
    total_records = len(fragdom_records)
    matched_records = 0

    for header, (ranges, seq) in fragdom_records.items():
        probs = esm_probs.get(header)
        if probs is None:
            continue
        matched_records += 1
        cluster_segments = None
        if cluster_segments_map is not None:
            cluster_segments = cluster_segments_map.get(header)
        jobs.append(
            (
                header,
                ranges,
                seq,
                probs,
                cluster_segments,
                args.log_base,
            )
        )

    print(f"[INFO] frag_dom total records: {total_records}")
    print(f"[INFO] matched records with ESM outputs: {matched_records}")

    if matched_records == 0:
        raise ValueError("No overlapping headers between frag_dom records and ESM probability outputs")

    # ------------------------------
    # Compute entropy (parallel/sequential)
    # ------------------------------
    per_pos_rows: List[Dict[str, object]] = []
    per_record_rows: List[Dict[str, object]] = []

    print(f"[INFO] Computing entropy with workers={args.workers}, chunksize={args.chunksize} ...")

    if args.workers <= 1:
        for result in tqdm(jobs, desc="Entropy", unit="record"):
            pos_rows, rec_row = process_one_record(result)
            per_pos_rows.extend(pos_rows)
            per_record_rows.append(rec_row)
    else:
        # On Linux, fork is efficient here. On Windows, spawn will work but be slower.
        start_method = "fork" if os.name != "nt" else "spawn"
        ctx = get_context(start_method)

        with ctx.Pool(processes=args.workers) as pool:
            iterator = pool.imap_unordered(process_one_record, jobs, chunksize=args.chunksize)
            for pos_rows, rec_row in tqdm(iterator, total=len(jobs), desc="Entropy", unit="record"):
                per_pos_rows.extend(pos_rows)
                per_record_rows.append(rec_row)

    # ------------------------------
    # Build DataFrames
    # ------------------------------
    per_pos_df = pd.DataFrame(per_pos_rows)
    per_record_df = pd.DataFrame(per_record_rows)

    # ---------------------------------
    # Core comparison 1:
    # fragment vs background
    # ---------------------------------
    frag_entropy = per_pos_df.loc[per_pos_df["is_fragment"], "entropy"]
    bg_entropy = per_pos_df.loc[~per_pos_df["is_fragment"], "entropy"]

    summary_rows: List[Dict[str, object]] = []
    summary_rows.append(summarize_series(frag_entropy, "fragment_positions"))
    summary_rows.append(summarize_series(bg_entropy, "background_positions"))

    delta_series = per_record_df["delta_fragment_minus_background"].dropna()
    summary_rows.append(summarize_series(delta_series, "per_record_delta_fragment_minus_background"))

    # ---------------------------------
    # Core comparison 2:
    # high-share vs low-share among fragment positions
    # ---------------------------------
    frag_only = per_pos_df[per_pos_df["is_fragment"]].copy()

    valid_share = frag_only["share_count"].replace([np.inf, -np.inf], np.nan).dropna()
    if not valid_share.empty:
        low_q = float(args.low_quantile)
        high_q = float(args.high_quantile)
        if not (0.0 <= low_q <= 1.0 and 0.0 <= high_q <= 1.0 and low_q <= high_q):
            raise ValueError("Require 0 <= low_quantile <= high_quantile <= 1")

        low_cut = float(valid_share.quantile(low_q))
        high_cut = float(valid_share.quantile(high_q))

        def classify_share(x: float) -> str:
            if pd.isna(x):
                return "NA"
            if x <= low_cut:
                return "LOW"
            if x >= high_cut:
                return "HIGH"
            return "MID"

        frag_only["share_group"] = frag_only["share_count"].apply(classify_share)

        low_entropy = frag_only.loc[frag_only["share_group"] == "LOW", "entropy"]
        high_entropy = frag_only.loc[frag_only["share_group"] == "HIGH", "entropy"]

        summary_rows.append(
            {
                "group": "share_group_cutoffs",
                "n": int(valid_share.shape[0]),
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": low_cut,
                "max": high_cut,
            }
        )
        summary_rows.append(summarize_series(low_entropy, "low_share_fragment_positions"))
        summary_rows.append(summarize_series(high_entropy, "high_share_fragment_positions"))
    else:
        frag_only["share_group"] = "NA"
        low_entropy = pd.Series(dtype=float)
        high_entropy = pd.Series(dtype=float)

    # Merge share_group back for output
    per_pos_df = per_pos_df.merge(
        frag_only[["header", "position", "share_group"]],
        on=["header", "position"],
        how="left",
    )
    per_pos_df["share_group"] = per_pos_df["share_group"].fillna("NON_FRAGMENT")

    global_summary_df = pd.DataFrame(summary_rows)

    # ---------------------------------
    # Save tables
    # ---------------------------------
    per_pos_path = outdir / "per_position_entropy.tsv"
    per_record_path = outdir / "per_record_summary.tsv"
    global_summary_path = outdir / "global_summary.tsv"

    per_pos_df.to_csv(per_pos_path, sep="\t", index=False)
    per_record_df.to_csv(per_record_path, sep="\t", index=False)
    global_summary_df.to_csv(global_summary_path, sep="\t", index=False)

    # ---------------------------------
    # Save plots
    # ---------------------------------
    # Density histogram
    save_histogram(
        frag_entropy,
        "Fragment",
        bg_entropy,
        "Background",
        outdir / "fragment_vs_background_entropy_hist.png",
        "Entropy: frag_dom conserved positions vs background positions (density)",
        density=True,
    )

    # Count histogram
    save_histogram(
        frag_entropy,
        "Fragment",
        bg_entropy,
        "Background",
        outdir / "fragment_vs_background_entropy_hist_counts.png",
        "Entropy: frag_dom conserved positions vs background positions (count)",
        density=False,
    )

    if not low_entropy.empty and not high_entropy.empty:
        save_histogram(
            low_entropy,
            "Low share",
            high_entropy,
            "High share",
            outdir / "high_vs_low_share_entropy_hist.png",
            "Entropy: low-share vs high-share fragment positions",
            density=True,
        )

    # ---------------------------------
    # Print concise summary
    # ---------------------------------
    print("[DONE] Wrote:")
    print(f"  - {per_pos_path}")
    print(f"  - {per_record_path}")
    print(f"  - {global_summary_path}")
    print(f"  - {outdir / 'fragment_vs_background_entropy_hist.png'}")
    print(f"  - {outdir / 'fragment_vs_background_entropy_hist_counts.png'}")
    if not low_entropy.empty and not high_entropy.empty:
        print(f"  - {outdir / 'high_vs_low_share_entropy_hist.png'}")

    print("\n[SUMMARY]")
    for _, row in global_summary_df.iterrows():
        print(
            f"{row['group']}: "
            f"n={row['n']}, mean={row['mean']}, median={row['median']}, std={row['std']}, "
            f"min={row['min']}, max={row['max']}"
        )


if __name__ == "__main__":
    main()
