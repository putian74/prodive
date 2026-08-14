#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESM2 entropy vs Pfam seed FASTA-MSA conservation only.

This script intentionally removes all fragment annotation, fragment/background
comparison, and complete-fragment FASTA processing. It only does:

1. Read the full representative sequences.
2. Map each representative sequence back to the corresponding Pfam seed FASTA MSA.
3. Compute per-position MSA conservation metrics for the representative sequence.
4. Merge those MSA metrics with ESM2 per-position entropy.
5. Report global and per-record ESM2-vs-MSA correlations, plus optional low-entropy overlap.

Main interpretation direction:
- Lower ESM2 entropy means the masked-token model is more certain at that residue.
- Lower MSA entropy means the MSA column is more conserved.
- Therefore, ESM2 entropy should be positively correlated with MSA entropy,
  but negatively correlated with MSA information and majority frequency.
"""
from __future__ import annotations

import argparse
import gzip
import math
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# Config
# ==============================================================================
REPRESENTATIVE_FILE = Path()
ESM_ENTROPY_TSV = Path()
PFAM_ROOT = Path()
OUT_DIR = Path()
# If True, existing outputs under OUT_DIR are removed before running.
OVERWRITE_OUTPUT = True

# If False and PER_POSITION_MSA_TSV already exists, the expensive MSA step is skipped
# and the existing per-position MSA table is reused.
RECOMPUTE_MSA = True

NUM_WORKERS = 32
MAX_INFLIGHT = NUM_WORKERS * 4
POSITION_WRITE_CHUNK_SIZE = 200_000
STATUS_WRITE_CHUNK_SIZE = 2_000

# MSA quality filters used for correlation analysis.
MIN_COLUMN_AA_COUNT = 5
MAX_GAP_FRACTION = 0.80
MIN_RECORD_POSITIONS_FOR_CORR = 30

# For the optional comparison between ESM2-low-entropy positions and MSA-conserved positions.
LOW_FRACTION = 0.20

MAKE_PLOTS = True

# ==============================================================================
# Output files
# ==============================================================================
PER_POSITION_MSA_TSV = OUT_DIR / "representative_position_msa_conservation.tsv"
RECORD_STATUS_TSV = OUT_DIR / "representative_fasta_msa_mapping_status.tsv"
MERGED_TSV = OUT_DIR / "esm_entropy_with_fasta_msa_conservation.tsv"
MERGE_DIAGNOSTICS_TXT = OUT_DIR / "merge_header_overlap_diagnostics.txt"

GLOBAL_CORR_TSV = OUT_DIR / "global_esm_msa_correlation_summary.tsv"
PER_RECORD_CORR_TSV = OUT_DIR / "per_record_esm_msa_correlation.tsv"
PER_RECORD_CORR_SUMMARY_TSV = OUT_DIR / "per_record_esm_msa_correlation_summary.tsv"
LOW_ENTROPY_OVERLAP_TSV = OUT_DIR / "low_esm_entropy_vs_msa_conserved_overlap_per_record.tsv"
LOW_ENTROPY_OVERLAP_SUMMARY_TSV = OUT_DIR / "low_esm_entropy_vs_msa_conserved_overlap_summary.tsv"
MSA_DECILE_TREND_TSV = OUT_DIR / "msa_entropy_decile_vs_esm_entropy_trend.tsv"

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
RESIDUE_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
GAP_SET = {"-", ".", "_", "~"}

POSITION_COLUMNS = [
    "header", "pfam_id", "uid", "src_header", "selected_fasta_row_id", "alignment_path", "alignment_type",
    "domain_len", "position", "msa_column", "rep_aa", "msa_total_rows", "msa_aa_count", "msa_gap_count",
    "msa_nonstandard_count", "msa_invalid_count", "msa_missing_count", "gap_fraction", "nonstandard_fraction",
    "msa_entropy", "msa_entropy_norm", "msa_information_uniform", "majority_aa", "majority_freq",
    "effective_aa_number", "mapping_status", "row_match_status",
]

STATUS_COLUMNS = [
    "header", "pfam_id", "uid", "src_header", "alignment_path", "alignment_type", "selected_fasta_row_id",
    "row_match_status", "domain_len", "sequence_len", "msa_start", "msa_end", "mapped_positions",
    "mismatch_count", "mapping_status", "status", "error_message",
]

_FASTA_CACHE: Dict[str, Tuple[Optional[Dict[str, str]], Optional[Path], str]] = {}


def str_to_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def configure_runtime(args: argparse.Namespace) -> None:
    global REPRESENTATIVE_FILE, ESM_ENTROPY_TSV, PFAM_ROOT, OUT_DIR
    global OVERWRITE_OUTPUT, RECOMPUTE_MSA, NUM_WORKERS, MAX_INFLIGHT
    global POSITION_WRITE_CHUNK_SIZE, STATUS_WRITE_CHUNK_SIZE
    global MIN_COLUMN_AA_COUNT, MAX_GAP_FRACTION, MIN_RECORD_POSITIONS_FOR_CORR
    global LOW_FRACTION, MAKE_PLOTS
    global PER_POSITION_MSA_TSV, RECORD_STATUS_TSV, MERGED_TSV, MERGE_DIAGNOSTICS_TXT
    global GLOBAL_CORR_TSV, PER_RECORD_CORR_TSV, PER_RECORD_CORR_SUMMARY_TSV
    global LOW_ENTROPY_OVERLAP_TSV, LOW_ENTROPY_OVERLAP_SUMMARY_TSV, MSA_DECILE_TREND_TSV

    REPRESENTATIVE_FILE = Path(args.representative_file)
    ESM_ENTROPY_TSV = Path(args.esm_entropy_tsv)
    PFAM_ROOT = Path(args.pfam_root)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OVERWRITE_OUTPUT = bool(args.overwrite_output)
    RECOMPUTE_MSA = bool(args.recompute_msa)
    NUM_WORKERS = int(args.workers)
    MAX_INFLIGHT = int(args.max_inflight) if args.max_inflight is not None else NUM_WORKERS * 4
    POSITION_WRITE_CHUNK_SIZE = int(args.position_write_chunk_size)
    STATUS_WRITE_CHUNK_SIZE = int(args.status_write_chunk_size)
    MIN_COLUMN_AA_COUNT = int(args.min_column_aa_count)
    MAX_GAP_FRACTION = float(args.max_gap_fraction)
    MIN_RECORD_POSITIONS_FOR_CORR = int(args.min_record_positions_for_corr)
    LOW_FRACTION = float(args.low_fraction)
    MAKE_PLOTS = bool(args.make_plots)

    PER_POSITION_MSA_TSV = OUT_DIR / "representative_position_msa_conservation.tsv"
    RECORD_STATUS_TSV = OUT_DIR / "representative_fasta_msa_mapping_status.tsv"
    MERGED_TSV = OUT_DIR / "esm_entropy_with_fasta_msa_conservation.tsv"
    MERGE_DIAGNOSTICS_TXT = OUT_DIR / "merge_header_overlap_diagnostics.txt"

    GLOBAL_CORR_TSV = OUT_DIR / "global_esm_msa_correlation_summary.tsv"
    PER_RECORD_CORR_TSV = OUT_DIR / "per_record_esm_msa_correlation.tsv"
    PER_RECORD_CORR_SUMMARY_TSV = OUT_DIR / "per_record_esm_msa_correlation_summary.tsv"
    LOW_ENTROPY_OVERLAP_TSV = OUT_DIR / "low_esm_entropy_vs_msa_conserved_overlap_per_record.tsv"
    LOW_ENTROPY_OVERLAP_SUMMARY_TSV = OUT_DIR / "low_esm_entropy_vs_msa_conserved_overlap_summary.tsv"
    MSA_DECILE_TREND_TSV = OUT_DIR / "msa_entropy_decile_vs_esm_entropy_trend.tsv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ESM2 per-position entropy with Pfam seed MSA conservation."
    )
    parser.add_argument("--representative-file", required=True, help="Full representative sequence file with frag_dom headers.")
    parser.add_argument("--esm-entropy-tsv", required=True, help="ESM2 per-position entropy TSV.")
    parser.add_argument("--pfam-root", required=True, help="PfamA_seed root directory.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--overwrite-output", type=str_to_bool, default=OVERWRITE_OUTPUT, help="Whether to remove existing outputs in out-dir before running.")
    parser.add_argument("--recompute-msa", type=str_to_bool, default=RECOMPUTE_MSA, help="Whether to recompute MSA conservation instead of reusing an existing table.")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes.")
    parser.add_argument("--max-inflight", type=int, default=None, help="Maximum submitted jobs; default is workers * 4.")
    parser.add_argument("--position-write-chunk-size", type=int, default=POSITION_WRITE_CHUNK_SIZE)
    parser.add_argument("--status-write-chunk-size", type=int, default=STATUS_WRITE_CHUNK_SIZE)
    parser.add_argument("--min-column-aa-count", type=int, default=MIN_COLUMN_AA_COUNT)
    parser.add_argument("--max-gap-fraction", type=float, default=MAX_GAP_FRACTION)
    parser.add_argument("--min-record-positions-for-corr", type=int, default=MIN_RECORD_POSITIONS_FOR_CORR)
    parser.add_argument("--low-fraction", type=float, default=LOW_FRACTION)
    parser.add_argument("--make-plots", type=str_to_bool, default=MAKE_PLOTS)
    return parser


# ==============================================================================
# General utilities
# ==============================================================================
def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return open(path, "r", errors="ignore")


def parse_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def parse_range(x: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"(\d+)-(\d+)", str(x))
    if not m:
        return None, None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if b >= a else (None, None)


def parse_pipe_header(h: str) -> Dict[str, str]:
    parts = str(h).strip().split("|")
    out = {"pfam_id": parts[0].strip() if parts else ""}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def clean_seq(seq: str) -> str:
    out: List[str] = []
    for c in str(seq).upper():
        if c in GAP_SET or c.isspace():
            continue
        if c in RESIDUE_SET:
            out.append(c)
    return "".join(out)


def acc_base(x: str) -> str:
    return str(x).strip().split(".")[0] if x else ""


def range_suffix(x: str) -> str:
    m = re.search(r"/\d+-\d+\s*$", str(x).strip())
    return m.group(0) if m else ""


def corr_safe(x: pd.Series, y: pd.Series, method: str) -> float:
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 2 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return np.nan
    return float(d["x"].corr(d["y"], method=method))


def append_rows(rows: List[Dict[str, object]], path: Path, cols: List[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    mode = "a" if path.exists() else "w"
    df[cols].to_csv(path, sep="\t", index=False, mode=mode, header=(mode == "w"))

# ==============================================================================
# Representative file parsing
# ==============================================================================
def iter_representatives(path: Path) -> Iterable[Dict[str, object]]:
    header: Optional[str] = None
    first_seq_line: Optional[str] = None
    seq_parts: List[str] = []

    def flush() -> Optional[Dict[str, object]]:
        if header is None:
            return None
        info = parse_pipe_header(header)
        msa_start, msa_end = parse_range(info.get("msa_span", ""))
        seq = clean_seq("".join(seq_parts))
        try:
            n = int(info.get("len", len(seq)))
        except Exception:
            n = len(seq)
        return {
            "header": header,
            "pfam_id": info.get("pfam_id", ""),
            "uid": info.get("uid", ""),
            "src_header": info.get("src_header", ""),
            "msa_start": msa_start,
            "msa_end": msa_end,
            "len": n,
            "sequence": seq,
        }

    with open_text(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                rec = flush()
                if rec is not None:
                    yield rec
                header = line[1:].strip()
                first_seq_line = None
                seq_parts = []
            else:
                if header is None:
                    continue
                if first_seq_line is None:
                    first_seq_line = line
                    # The first non-header line may contain old fragment-domain metadata.
                    # This script does not use it.
                    if line.startswith("frag_dom="):
                        continue
                seq_parts.append(line)
    rec = flush()
    if rec is not None:
        yield rec


def load_representatives() -> List[Dict[str, object]]:
    if not REPRESENTATIVE_FILE.exists():
        raise FileNotFoundError(REPRESENTATIVE_FILE)
    records = list(iter_representatives(REPRESENTATIVE_FILE))
    print(f"[INFO] representative records: {len(records):,}")
    return records

# ==============================================================================
# FASTA MSA loading and representative-to-MSA mapping
# ==============================================================================
def load_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    cur: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        if cur is not None:
            seqs[cur] = "".join(buf).replace(" ", "").replace("\t", "").upper()

    with open_text(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                cur = line[1:].strip()
                buf.clear()
            else:
                if cur is not None:
                    buf.append(line)
    flush()
    return seqs


def find_fasta_for_pfam(pfam_id: str) -> Tuple[Optional[Path], str]:
    fam_dir = PFAM_ROOT / pfam_id
    for ext in ["fas", "fasta", "fa", "fas.gz", "fasta.gz", "fa.gz"]:
        p = fam_dir / f"{pfam_id}.{ext}"
        if p.exists():
            return p, "FASTA"
    return None, "NOT_FOUND"


def split_src(src: str) -> Tuple[str, str]:
    m = re.match(r"^([A-Za-z0-9]+(?:\.\d+)?)_(.+/\d+-\d+)$", str(src).strip())
    return (m.group(1), m.group(2)) if m else ("", str(src).strip())


def fasta_variants(h: str) -> List[str]:
    h = str(h).strip()
    hs = re.sub(r"\s+", " ", h)
    hu = re.sub(r"\s+", "_", hs)
    out = {h, hs, hu}
    t = hs.split()
    if t:
        out.add(t[0])
    if len(t) >= 2:
        out.update([t[1], t[0] + "_" + t[1], t[0] + " " + t[1]])
    return [x for x in out if x]


def src_variants(src: str) -> List[str]:
    src = str(src).strip()
    out = {src}
    acc, rest = split_src(src)
    if acc and rest:
        out.update([acc, rest, acc + "_" + rest, acc + " " + rest])
    return [x for x in out if x]


def build_variant_index(seqs: Dict[str, str]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = defaultdict(list)
    for h in seqs:
        for v in fasta_variants(h):
            idx[v].append(h)
    return idx


def fasta_acc(header: str) -> str:
    return str(header).strip().split()[0] if str(header).strip() else ""


def find_selected_row(
    seqs: Dict[str, str],
    src_header: str,
    uid: str,
    rep_seq: str,
) -> Tuple[Optional[str], Optional[str], str]:
    rep_seq = clean_seq(rep_seq)
    idx = build_variant_index(seqs)

    for v in src_variants(src_header):
        hits = list(dict.fromkeys(idx.get(v, [])))
        if len(hits) == 1:
            h = hits[0]
            return h, seqs[h], "HEADER_VARIANT_UNIQUE_MATCH"
        if len(hits) > 1:
            content = [h for h in hits if clean_seq(seqs[h]) == rep_seq]
            if len(content) == 1:
                h = content[0]
                return h, seqs[h], "HEADER_VARIANT_AMBIGUOUS_RESOLVED_BY_SEQUENCE"
            if len(content) > 1:
                return None, None, "AMBIGUOUS_HEADER_VARIANT_AND_SEQUENCE_MATCH"
            return None, None, "AMBIGUOUS_HEADER_VARIANT_MATCH"

    acc, _ = split_src(src_header)
    accb = acc_base(acc)
    uidb = acc_base(uid)
    suf = range_suffix(src_header)

    cand = []
    for h, s in seqs.items():
        ha = fasta_acc(h)
        hab = acc_base(ha)
        ok = (acc and (acc == ha or accb == hab)) or (uid and (uid == ha or uidb == hab))
        if ok and suf and suf in h:
            cand.append((h, s))
    if len(cand) == 1:
        return cand[0][0], cand[0][1], "ACCESSION_AND_RANGE_MATCH"
    if len(cand) > 1:
        content = [(h, s) for h, s in cand if clean_seq(s) == rep_seq]
        if len(content) == 1:
            return content[0][0], content[0][1], "ACCESSION_AND_RANGE_AMBIGUOUS_RESOLVED_BY_SEQUENCE"
        if len(content) > 1:
            return None, None, "AMBIGUOUS_ACCESSION_AND_RANGE_AND_SEQUENCE_MATCH"
        return None, None, "AMBIGUOUS_ACCESSION_AND_RANGE_MATCH"

    if suf:
        cand = [(h, s) for h, s in seqs.items() if suf in h]
        if len(cand) == 1:
            return cand[0][0], cand[0][1], "RANGE_ONLY_UNIQUE_MATCH"
        if len(cand) > 1:
            content = [(h, s) for h, s in cand if clean_seq(s) == rep_seq]
            if len(content) == 1:
                return content[0][0], content[0][1], "RANGE_ONLY_AMBIGUOUS_RESOLVED_BY_SEQUENCE"
            if len(content) > 1:
                return None, None, "AMBIGUOUS_RANGE_ONLY_AND_SEQUENCE_MATCH"
            return None, None, "AMBIGUOUS_RANGE_ONLY_MATCH"

    cand = [(h, s) for h, s in seqs.items() if clean_seq(s) == rep_seq]
    if len(cand) == 1:
        return cand[0][0], cand[0][1], "SEQUENCE_CONTENT_UNIQUE_MATCH"
    if len(cand) > 1:
        return None, None, "AMBIGUOUS_SEQUENCE_CONTENT_MATCH"
    return None, None, "SELECTED_ROW_NOT_FOUND"


def col_metrics(seqs: Dict[str, str], col: int) -> Dict[str, object]:
    idx = col - 1
    counts = {aa: 0 for aa in AA_ORDER}
    gap = 0
    nonstd = 0
    invalid = 0
    missing = 0
    n = len(seqs)

    for s in seqs.values():
        if idx >= len(s):
            missing += 1
            gap += 1
            continue
        c = s[idx].upper()
        if c in GAP_SET:
            gap += 1
        elif c in AA_SET:
            counts[c] += 1
        elif c in RESIDUE_SET:
            nonstd += 1
        else:
            invalid += 1

    aa_count = sum(counts.values())
    if aa_count <= 0:
        return {
            "msa_total_rows": n,
            "msa_aa_count": 0,
            "msa_gap_count": gap,
            "msa_nonstandard_count": nonstd,
            "msa_invalid_count": invalid,
            "msa_missing_count": missing,
            "gap_fraction": gap / n if n else np.nan,
            "nonstandard_fraction": nonstd / n if n else np.nan,
            "msa_entropy": np.nan,
            "msa_entropy_norm": np.nan,
            "msa_information_uniform": np.nan,
            "majority_aa": "",
            "majority_freq": np.nan,
            "effective_aa_number": np.nan,
        }

    p = np.array([counts[aa] / aa_count for aa in AA_ORDER], dtype=float)
    nz = p[p > 0]
    ent = -float(np.sum(nz * np.log(nz)))
    maj = max(counts, key=lambda aa: counts[aa])

    return {
        "msa_total_rows": n,
        "msa_aa_count": aa_count,
        "msa_gap_count": gap,
        "msa_nonstandard_count": nonstd,
        "msa_invalid_count": invalid,
        "msa_missing_count": missing,
        "gap_fraction": gap / n if n else np.nan,
        "nonstandard_fraction": nonstd / n if n else np.nan,
        "msa_entropy": ent,
        "msa_entropy_norm": ent / math.log(20.0),
        "msa_information_uniform": math.log(20.0) - ent,
        "majority_aa": maj,
        "majority_freq": counts[maj] / aa_count,
        "effective_aa_number": math.exp(ent),
    }


def map_rep_to_msa(
    aln_seq: str,
    rep_seq: str,
    msa_start: Optional[int],
    msa_end: Optional[int],
) -> Tuple[Dict[int, int], int, int, str]:
    if msa_start is None or msa_end is None or msa_start < 1 or msa_end < msa_start:
        return {}, 0, 0, "BAD_MSA_SPAN"
    if len(aln_seq) < msa_end:
        return {}, 0, 0, f"ALIGNMENT_ROW_TOO_SHORT({len(aln_seq)}<{msa_end})"

    span = aln_seq[msa_start - 1:msa_end]
    pos_to_col: Dict[int, int] = {}
    obs: List[str] = []
    pos = 0
    for off, c in enumerate(span):
        c = c.upper()
        if c in GAP_SET or c not in RESIDUE_SET:
            continue
        pos += 1
        pos_to_col[pos] = msa_start + off
        obs.append(c)

    obs_seq = "".join(obs)
    rep = clean_seq(rep_seq)
    mismatch_count = sum(a != b for a, b in zip(obs_seq, rep)) + abs(len(obs_seq) - len(rep))
    if len(obs_seq) != len(rep):
        status = f"SEQ_LENGTH_MISMATCH(msa_nongap={len(obs_seq)},rep_len={len(rep)})"
    elif mismatch_count > 0:
        status = f"SEQ_CONTENT_MISMATCH(n={mismatch_count})"
    else:
        status = "OK"
    return pos_to_col, len(obs_seq), mismatch_count, status


def load_fasta_cached(pfam_id: str) -> Tuple[Optional[Dict[str, str]], Optional[Path], str]:
    if pfam_id in _FASTA_CACHE:
        return _FASTA_CACHE[pfam_id]

    p, typ = find_fasta_for_pfam(pfam_id)
    if p is None:
        res = (None, None, typ)
    else:
        try:
            res = (load_fasta(p), p, typ)
        except Exception:
            res = (None, p, "FASTA_LOAD_FAIL")
    _FASTA_CACHE[pfam_id] = res
    return res


def fail_status(
    rec: Dict[str, object],
    aln_path: object,
    aln_type: str,
    selected: str,
    row_status: str,
    mapped: int,
    mism: object,
    map_status: str,
    err: str,
) -> Dict[str, object]:
    return {
        "header": rec.get("header", ""),
        "pfam_id": rec.get("pfam_id", ""),
        "uid": rec.get("uid", ""),
        "src_header": rec.get("src_header", ""),
        "alignment_path": str(aln_path) if aln_path else "",
        "alignment_type": aln_type,
        "selected_fasta_row_id": selected,
        "row_match_status": row_status,
        "domain_len": rec.get("len", np.nan),
        "sequence_len": len(clean_seq(rec.get("sequence", ""))),
        "msa_start": rec.get("msa_start", np.nan),
        "msa_end": rec.get("msa_end", np.nan),
        "mapped_positions": mapped,
        "mismatch_count": mism,
        "mapping_status": map_status,
        "status": "FAIL",
        "error_message": err,
    }


def process_one_record(rec: Dict[str, object]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rep = clean_seq(str(rec.get("sequence", "")))
    domain_len = int(rec.get("len", len(rep)))

    if domain_len != len(rep):
        return [], fail_status(
            rec, "", "", "", "", 0, np.nan,
            "REPRESENTATIVE_LEN_MISMATCH",
            f"header len={domain_len}, sequence_len={len(rep)}",
        )

    seqs, fasta_path, typ = load_fasta_cached(str(rec["pfam_id"]))
    if seqs is None or not seqs:
        return [], fail_status(
            rec, fasta_path, typ, "", "", 0, np.nan,
            "FASTA_NOT_FOUND_OR_LOAD_FAIL",
            f"No valid FASTA for {rec['pfam_id']}",
        )

    sid, aln_seq, row_status = find_selected_row(seqs, str(rec["src_header"]), str(rec["uid"]), rep)
    if sid is None or aln_seq is None:
        return [], fail_status(
            rec, fasta_path, typ, "", row_status, 0, np.nan,
            "SELECTED_ROW_NOT_FOUND",
            row_status,
        )

    pos_to_col, mapped_n, mism, map_status = map_rep_to_msa(
        aln_seq, rep, rec.get("msa_start"), rec.get("msa_end")
    )
    if mapped_n != domain_len or map_status != "OK":
        return [], fail_status(
            rec, fasta_path, typ, sid, row_status, mapped_n, mism,
            map_status,
            f"mapped_n={mapped_n}, domain_len={domain_len}",
        )

    rows: List[Dict[str, object]] = []
    col_cache: Dict[int, Dict[str, object]] = {}
    for pos in range(1, domain_len + 1):
        col = pos_to_col.get(pos)
        if col is None:
            continue
        if col not in col_cache:
            col_cache[col] = col_metrics(seqs, col)
        rows.append({
            "header": rec["header"],
            "pfam_id": rec["pfam_id"],
            "uid": rec["uid"],
            "src_header": rec["src_header"],
            "selected_fasta_row_id": sid,
            "alignment_path": str(fasta_path),
            "alignment_type": typ,
            "domain_len": domain_len,
            "position": pos,
            "msa_column": col,
            "rep_aa": rep[pos - 1],
            **col_cache[col],
            "mapping_status": "OK",
            "row_match_status": row_status,
        })

    status = {
        "header": rec["header"],
        "pfam_id": rec["pfam_id"],
        "uid": rec["uid"],
        "src_header": rec["src_header"],
        "alignment_path": str(fasta_path),
        "alignment_type": typ,
        "selected_fasta_row_id": sid,
        "row_match_status": row_status,
        "domain_len": domain_len,
        "sequence_len": len(rep),
        "msa_start": rec["msa_start"],
        "msa_end": rec["msa_end"],
        "mapped_positions": len(rows),
        "mismatch_count": mism,
        "mapping_status": "OK",
        "status": "OK",
        "error_message": "",
    }
    return rows, status


def run_msa(records: List[Dict[str, object]]) -> None:
    print(f"[INFO] MSA records: {len(records):,}, workers={NUM_WORKERS}")
    pos_buf: List[Dict[str, object]] = []
    status_buf: List[Dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        inflight = {}
        it = iter(records)

        def submit(r: Dict[str, object]) -> None:
            inflight[ex.submit(process_one_record, r)] = 1

        for _ in range(min(MAX_INFLIGHT, len(records))):
            try:
                submit(next(it))
            except StopIteration:
                break

        with tqdm(total=len(records), desc="FASTA MSA conservation", unit="record") as bar:
            while inflight:
                for fut in as_completed(list(inflight.keys())):
                    inflight.pop(fut, None)
                    try:
                        rows, st = fut.result()
                    except Exception as e:
                        rows = []
                        st = {c: "" for c in STATUS_COLUMNS}
                        st.update({
                            "mapping_status": "WORKER_EXCEPTION",
                            "status": "FAIL",
                            "error_message": str(e),
                            "mapped_positions": 0,
                        })

                    pos_buf.extend(rows)
                    status_buf.append(st)
                    bar.update(1)

                    try:
                        submit(next(it))
                    except StopIteration:
                        pass

                    if len(pos_buf) >= POSITION_WRITE_CHUNK_SIZE:
                        append_rows(pos_buf, PER_POSITION_MSA_TSV, POSITION_COLUMNS)
                        pos_buf = []
                    if len(status_buf) >= STATUS_WRITE_CHUNK_SIZE:
                        append_rows(status_buf, RECORD_STATUS_TSV, STATUS_COLUMNS)
                        status_buf = []
                    break

    append_rows(pos_buf, PER_POSITION_MSA_TSV, POSITION_COLUMNS)
    append_rows(status_buf, RECORD_STATUS_TSV, STATUS_COLUMNS)

# ==============================================================================
# ESM2 entropy merge
# ==============================================================================
def load_entropy_for_merge() -> pd.DataFrame:
    keep = {
        "header", "position", "wt_aa", "entropy",
        "aa_mass_20aa", "wt_prob_20aa_renorm",
    }
    df = pd.read_csv(ESM_ENTROPY_TSV, sep="\t", usecols=lambda c: c in keep, low_memory=False)
    required = {"header", "position", "wt_aa", "entropy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ESM entropy file missing columns: {sorted(missing)}")

    df["header"] = df["header"].astype(str)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    df["wt_aa"] = df["wt_aa"].astype(str).str.upper()
    df["esm_entropy"] = pd.to_numeric(df["entropy"], errors="coerce")

    keep2 = ["header", "position", "wt_aa", "esm_entropy"]
    for c in ["aa_mass_20aa", "wt_prob_20aa_renorm"]:
        if c in df.columns:
            keep2.append(c)
    return df[keep2]


def merge_diag(msa: pd.DataFrame, ent: pd.DataFrame) -> None:
    mh = set(msa["header"].astype(str).unique())
    eh = set(ent["header"].astype(str).unique())
    mp = set(zip(msa["header"].astype(str), msa["position"].astype(int)))
    ep = set(zip(ent["header"].astype(str), ent["position"].astype(int)))

    with open(MERGE_DIAGNOSTICS_TXT, "w") as f:
        f.write("[HEADER OVERLAP]\n")
        f.write(f"MSA mapped headers: {len(mh)}\n")
        f.write(f"ESM entropy headers: {len(eh)}\n")
        f.write(f"Header intersection: {len(mh & eh)}\n")
        f.write(f"Only in MSA: {len(mh - eh)}\n")
        f.write(f"Only in ESM: {len(eh - mh)}\n\n")
        f.write("[HEADER-POSITION PAIR OVERLAP]\n")
        f.write(f"MSA header-position pairs: {len(mp)}\n")
        f.write(f"ESM header-position pairs: {len(ep)}\n")
        f.write(f"Pair intersection: {len(mp & ep)}\n")
        f.write("\n[EXAMPLE ONLY IN MSA]\n")
        for x in list(mh - eh)[:20]:
            f.write(str(x) + "\n")
        f.write("\n[EXAMPLE ONLY IN ESM]\n")
        for x in list(eh - mh)[:20]:
            f.write(str(x) + "\n")

    print(
        f"[INFO] merge headers MSA={len(mh):,}, ESM={len(eh):,}, "
        f"intersection={len(mh & eh):,}, pair_intersection={len(mp & ep):,}"
    )


def merge_esm_and_msa() -> pd.DataFrame:
    ent = load_entropy_for_merge()
    msa = pd.read_csv(PER_POSITION_MSA_TSV, sep="\t", low_memory=False)

    msa["header"] = msa["header"].astype(str)
    msa["position"] = pd.to_numeric(msa["position"], errors="coerce").astype("Int64")
    for c in [
        "msa_entropy", "msa_entropy_norm", "msa_information_uniform", "majority_freq",
        "gap_fraction", "nonstandard_fraction", "msa_aa_count", "msa_nonstandard_count",
    ]:
        if c in msa.columns:
            msa[c] = pd.to_numeric(msa[c], errors="coerce")

    merge_diag(msa, ent)

    merged = ent.merge(msa, on=["header", "position"], how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged["analysis_ok"] = (
        merged["esm_entropy"].notna()
        & merged["msa_entropy"].notna()
        & (merged["mapping_status"] == "OK")
        & (merged["msa_aa_count"] >= MIN_COLUMN_AA_COUNT)
        & (merged["gap_fraction"] <= MAX_GAP_FRACTION)
    )
    merged.to_csv(MERGED_TSV, sep="\t", index=False)
    print(
        f"[INFO] merged rows={len(merged):,}, "
        f"analysis rows={int(merged['analysis_ok'].sum()):,}, "
        f"headers={merged.loc[merged['analysis_ok'], 'header'].nunique():,}"
    )
    return merged

# ==============================================================================
# ESM2-vs-MSA analyses
# ==============================================================================
def global_corr(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("esm_entropy", "msa_entropy", "ESM_entropy_vs_MSA_entropy"),
        ("esm_entropy", "msa_entropy_norm", "ESM_entropy_vs_MSA_entropy_norm"),
        ("esm_entropy", "msa_information_uniform", "ESM_entropy_vs_MSA_information_uniform"),
        ("esm_entropy", "majority_freq", "ESM_entropy_vs_MSA_majority_freq"),
        ("esm_entropy", "gap_fraction", "ESM_entropy_vs_gap_fraction"),
        ("esm_entropy", "nonstandard_fraction", "ESM_entropy_vs_nonstandard_fraction"),
    ]
    rows: List[Dict[str, object]] = []
    for x, y, label in pairs:
        if x not in df.columns or y not in df.columns:
            continue
        rows.append({
            "comparison": label,
            "x": x,
            "y": y,
            "n_positions": int(df[[x, y]].dropna().shape[0]),
            "pearson": corr_safe(df[x], df[y], "pearson"),
            "spearman": corr_safe(df[x], df[y], "spearman"),
        })
    return pd.DataFrame(rows)


def per_record_corr(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    required = ["esm_entropy", "msa_entropy", "msa_information_uniform", "majority_freq"]

    for h, sub in tqdm(df.groupby("header", sort=False), desc="Per-record correlation", unit="record"):
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        if len(sub) < MIN_RECORD_POSITIONS_FOR_CORR:
            continue
        rows.append({
            "header": h,
            "pfam_id": sub["pfam_id"].iloc[0],
            "uid": sub["uid"].iloc[0],
            "n_positions": int(len(sub)),
            "pearson_esm_vs_msa_entropy": corr_safe(sub["esm_entropy"], sub["msa_entropy"], "pearson"),
            "spearman_esm_vs_msa_entropy": corr_safe(sub["esm_entropy"], sub["msa_entropy"], "spearman"),
            "pearson_esm_vs_msa_information": corr_safe(sub["esm_entropy"], sub["msa_information_uniform"], "pearson"),
            "spearman_esm_vs_msa_information": corr_safe(sub["esm_entropy"], sub["msa_information_uniform"], "spearman"),
            "pearson_esm_vs_majority_freq": corr_safe(sub["esm_entropy"], sub["majority_freq"], "pearson"),
            "spearman_esm_vs_majority_freq": corr_safe(sub["esm_entropy"], sub["majority_freq"], "spearman"),
        })

    corr = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    corr_cols = [
        "pearson_esm_vs_msa_entropy",
        "spearman_esm_vs_msa_entropy",
        "pearson_esm_vs_msa_information",
        "spearman_esm_vs_msa_information",
        "pearson_esm_vs_majority_freq",
        "spearman_esm_vs_majority_freq",
    ]
    for c in corr_cols:
        s = corr[c].replace([np.inf, -np.inf], np.nan).dropna() if c in corr.columns else pd.Series(dtype=float)
        summary_rows.append({
            "correlation": c,
            "n_records": int(s.size),
            "mean": float(s.mean()) if not s.empty else np.nan,
            "median": float(s.median()) if not s.empty else np.nan,
            "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
            "positive_fraction": float((s > 0).mean()) if not s.empty else np.nan,
            "negative_fraction": float((s < 0).mean()) if not s.empty else np.nan,
            "abs_ge_0p3_fraction": float((s.abs() >= 0.3).mean()) if not s.empty else np.nan,
            "abs_ge_0p5_fraction": float((s.abs() >= 0.5).mean()) if not s.empty else np.nan,
        })
    return corr, pd.DataFrame(summary_rows)


def low_entropy_overlap(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare ESM2-low-entropy positions with MSA-low-entropy conserved positions per record."""
    rows: List[Dict[str, object]] = []
    for h, sub in tqdm(df.groupby("header", sort=False), desc="Low-entropy overlap", unit="record"):
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["esm_entropy", "msa_entropy"])
        n = len(sub)
        if n < MIN_RECORD_POSITIONS_FOR_CORR:
            continue
        esm_cutoff = float(sub["esm_entropy"].quantile(LOW_FRACTION))
        msa_cutoff = float(sub["msa_entropy"].quantile(LOW_FRACTION))
        esm_low = sub["esm_entropy"] <= esm_cutoff
        msa_low = sub["msa_entropy"] <= msa_cutoff

        n_esm = int(esm_low.sum())
        n_msa = int(msa_low.sum())
        n_overlap = int((esm_low & msa_low).sum())
        expected = n_esm * n_msa / n if n else np.nan

        rows.append({
            "header": h,
            "pfam_id": sub["pfam_id"].iloc[0],
            "uid": sub["uid"].iloc[0],
            "n_positions": int(n),
            "low_fraction": LOW_FRACTION,
            "esm_entropy_cutoff": esm_cutoff,
            "msa_entropy_cutoff": msa_cutoff,
            "n_esm_low": n_esm,
            "n_msa_conserved": n_msa,
            "n_overlap": n_overlap,
            "expected_overlap": float(expected),
            "precision_overlap_over_esm_low": n_overlap / n_esm if n_esm else np.nan,
            "recall_overlap_over_msa_conserved": n_overlap / n_msa if n_msa else np.nan,
            "jaccard": n_overlap / (n_esm + n_msa - n_overlap) if (n_esm + n_msa - n_overlap) else np.nan,
            "fold_enrichment_over_independence": n_overlap / expected if expected and expected > 0 else np.nan,
        })

    overlap = pd.DataFrame(rows)
    if overlap.empty:
        return overlap, pd.DataFrame()

    summary_rows: List[Dict[str, object]] = []
    for c in [
        "precision_overlap_over_esm_low",
        "recall_overlap_over_msa_conserved",
        "jaccard",
        "fold_enrichment_over_independence",
    ]:
        s = overlap[c].replace([np.inf, -np.inf], np.nan).dropna()
        summary_rows.append({
            "metric": c,
            "n_records": int(s.size),
            "mean": float(s.mean()) if not s.empty else np.nan,
            "median": float(s.median()) if not s.empty else np.nan,
            "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
            "q25": float(s.quantile(0.25)) if not s.empty else np.nan,
            "q75": float(s.quantile(0.75)) if not s.empty else np.nan,
            "min": float(s.min()) if not s.empty else np.nan,
            "max": float(s.max()) if not s.empty else np.nan,
        })
    return overlap, pd.DataFrame(summary_rows)


def decile_trend(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["msa_entropy", "esm_entropy"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if d.empty:
        return pd.DataFrame()
    d["msa_entropy_decile"] = pd.qcut(d["msa_entropy"], 10, duplicates="drop")
    rows: List[Dict[str, object]] = []
    for dec, sub in d.groupby("msa_entropy_decile", observed=False):
        rows.append({
            "msa_entropy_decile": str(dec),
            "n_positions": int(len(sub)),
            "mean_msa_entropy": float(sub["msa_entropy"].mean()),
            "median_msa_entropy": float(sub["msa_entropy"].median()),
            "mean_esm_entropy": float(sub["esm_entropy"].mean()),
            "median_esm_entropy": float(sub["esm_entropy"].median()),
        })
    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, pcorr: pd.DataFrame, dec: pd.DataFrame) -> None:
    if not MAKE_PLOTS:
        return

    d = df[["msa_entropy", "esm_entropy"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not d.empty:
        plt.figure(figsize=(7, 6))
        plt.hexbin(
            d["msa_entropy"].to_numpy(),
            d["esm_entropy"].to_numpy(),
            gridsize=120,
            bins="log",
            mincnt=1,
        )
        plt.xlabel("MSA column entropy")
        plt.ylabel("ESM2 masked-token entropy")
        plt.title("ESM2 entropy vs MSA column entropy")
        cb = plt.colorbar()
        cb.set_label("log10(count)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "hexbin_esm_entropy_vs_msa_entropy.png", dpi=200)
        plt.close()

    if pcorr is not None and not pcorr.empty and "spearman_esm_vs_msa_entropy" in pcorr.columns:
        vals = pcorr["spearman_esm_vs_msa_entropy"].replace([np.inf, -np.inf], np.nan).dropna()
        if not vals.empty:
            plt.figure(figsize=(8, 6))
            plt.hist(vals.to_numpy(), bins=60)
            plt.xlabel("Per-record Spearman correlation")
            plt.ylabel("Count")
            plt.title("Per-record Spearman: ESM2 entropy vs MSA entropy")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "hist_per_record_spearman_esm_vs_msa_entropy.png", dpi=200)
            plt.close()

    if dec is not None and not dec.empty:
        x = np.arange(len(dec))
        plt.figure(figsize=(10, 6))
        plt.plot(x, dec["mean_esm_entropy"], marker="o", label="Mean ESM2 entropy")
        plt.plot(x, dec["median_esm_entropy"], marker="o", label="Median ESM2 entropy")
        plt.xticks(x, [str(i + 1) for i in range(len(dec))])
        plt.xlabel("MSA entropy decile, low to high")
        plt.ylabel("ESM2 entropy")
        plt.title("ESM2 entropy across MSA entropy deciles")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "trend_esm_entropy_by_msa_entropy_decile.png", dpi=200)
        plt.close()


def run_analysis(merged: pd.DataFrame) -> None:
    df = merged[merged["analysis_ok"]].copy()
    print(
        f"[INFO] analysis positions={len(df):,}, "
        f"headers={df['header'].nunique():,}"
    )

    gc = global_corr(df)
    gc.to_csv(GLOBAL_CORR_TSV, sep="\t", index=False)

    pc, pcs = per_record_corr(df)
    pc.to_csv(PER_RECORD_CORR_TSV, sep="\t", index=False)
    pcs.to_csv(PER_RECORD_CORR_SUMMARY_TSV, sep="\t", index=False)

    ov, ovs = low_entropy_overlap(df)
    ov.to_csv(LOW_ENTROPY_OVERLAP_TSV, sep="\t", index=False)
    ovs.to_csv(LOW_ENTROPY_OVERLAP_SUMMARY_TSV, sep="\t", index=False)

    dec = decile_trend(df)
    dec.to_csv(MSA_DECILE_TREND_TSV, sep="\t", index=False)

    save_plots(df, pc, dec)

    print("\n[GLOBAL ESM2-MSA CORRELATION SUMMARY]")
    print(gc.to_string(index=False))
    print("\n[PER-RECORD ESM2-MSA CORRELATION SUMMARY]")
    print(pcs.to_string(index=False))
    print("\n[LOW ESM2 ENTROPY VS MSA CONSERVED OVERLAP SUMMARY]")
    print(ovs.to_string(index=False))
    print("\n[DONE] outputs written under:", OUT_DIR)

# ==============================================================================
# Run control
# ==============================================================================
def cleanup() -> None:
    if not OVERWRITE_OUTPUT:
        return
    for p in [
        PER_POSITION_MSA_TSV,
        RECORD_STATUS_TSV,
        MERGED_TSV,
        MERGE_DIAGNOSTICS_TXT,
        GLOBAL_CORR_TSV,
        PER_RECORD_CORR_TSV,
        PER_RECORD_CORR_SUMMARY_TSV,
        LOW_ENTROPY_OVERLAP_TSV,
        LOW_ENTROPY_OVERLAP_SUMMARY_TSV,
        MSA_DECILE_TREND_TSV,
    ]:
        if p.exists():
            p.unlink()
    for p in OUT_DIR.glob("*.png"):
        p.unlink()


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_runtime(args)
    cleanup()

    if RECOMPUTE_MSA or not PER_POSITION_MSA_TSV.exists():
        records = load_representatives()
        run_msa(records)
    else:
        print(f"[INFO] reuse existing MSA table: {PER_POSITION_MSA_TSV}")

    merged = merge_esm_and_msa()
    run_analysis(merged)


if __name__ == "__main__":
    main()
