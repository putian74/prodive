#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Purpose
-------
For de novo–Pfam result CSV (Main_HMM = de novo ID, Sub_HMM = Pfam family),
this script does NOT compute NW stats.

It only:
  1) Loads de novo full sequences from a FASTA file
  2) Selects ONE representative sequence for each Pfam family
  3) Parses conserved/local regions from Sub_Segments_Details ("a-b -> c-d")
  4) Maps HMM ranges to MSA ranges on selected sequences
  5) Extracts fragment sequences (ungapped)
  6) Exports:
       - full_sequences.mixed.fasta
       - conserved_fragments.pairs.fasta
       - conserved_fragments.with_fullseq.tsv
  7) Human txt output (NOT FASTA standard):
       - full_sequences.mixed.with_frag_ranges.txt

Notes
-----
- A side = de novo (Main_HMM)
- B side = Pfam representative sequence
- Fragment FASTA header intentionally does NOT include pair=...

Representative selection on Pfam side (modified)
-----------------------------------------------
For each Pfam family:
  - collect all target HMM ranges actually used by this family in CSV
  - for each aligned candidate sequence:
      * for each target segment h1-h2:
          - map to continuous MSA window msa_s~msa_e
          - extract that window from the candidate sequence
          - ungap it
          - compute abs(len(fragment) - (h2-h1+1))
      * sum these absolute differences over all target segments
  - choose the sequence with the smallest total difference
  - tie-break by:
      1) fewer EMPTY target windows
      2) smaller family-level len_diff to HMM total length
      3) fewer gaps inside family span
      4) fewer total gaps in the whole alignment

IMPORTANT:
- This keeps your original fragment extraction definition:
  continuous MSA window + ungap
- It does NOT try to exclude insertion residues separately
"""

import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, Any, Tuple, List, Optional, Set

import pandas as pd
import numpy as np
from tqdm import tqdm


# ==============================================================================
# 0) Config
# ==============================================================================
DEFAULT_INPUT_CSV = "<PRODIVE_DATA_ROOT>/shared/denovo_global_high_score_summary_fin.csv"
DEFAULT_DENOVO_FASTA = "<PRODIVE_DATA_ROOT>/shared/structures/denovo_structures/all_1927_sequences.fasta"
DEFAULT_PFAM_DIR = "<PRODIVE_DATA_ROOT>/shared/PfamA_seed/"
DEFAULT_OUT_DIR = "CHANGE_ME"

MIN_PURE_LEN_FOR_REP = 3
MIN_LOCAL_LEN = 1
REQ_COLS = ["Main_HMM", "Sub_HMM", "Sub_Segments_Details"]


# ==============================================================================
# 1) Helpers: alignment / FASTA / parsing
# ==============================================================================
def read_alignment(aln_path: str) -> Tuple[Dict[str, str], str]:
    if not os.path.exists(aln_path):
        return {}, "ALIGN_FILE_MISSING"

    seqs: Dict[str, str] = {}
    current_header = None
    try:
        with open(aln_path, "r", errors="ignore") as f:
            first_line = f.readline()
            f.seek(0)
            is_fasta = first_line.startswith(">")

            if is_fasta:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    if line.startswith(">"):
                        current_header = line[1:].strip()
                        seqs[current_header] = ""
                    elif current_header is not None:
                        seqs[current_header] += line.strip()
            else:
                # relaxed Stockholm-like parsing
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("//"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    sid, sfrag = parts[0], parts[1]
                    seqs[sid] = seqs.get(sid, "") + sfrag

        if not seqs:
            return {}, "ALIGN_EMPTY"
        return seqs, "OK"
    except Exception as e:
        return {}, f"READ_ERR:{e}"


def load_denovo_sequences(fasta_path: str) -> Dict[str, str]:
    """
    Load de novo sequences.
    Header parsing rule: use first token before '|' as sequence ID
    Example:
      >7U4P_1|xxx...
    -> ID = 7U4P_1
    """
    seqs: Dict[str, str] = {}
    if not os.path.exists(fasta_path):
        return seqs

    current_id = None
    current_seq_parts: List[str] = []
    try:
        with open(fasta_path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id is not None:
                        seqs[current_id] = "".join(current_seq_parts)
                    hdr = line[1:].strip()
                    current_id = hdr.split("|")[0].split()[0]
                    current_seq_parts = []
                else:
                    current_seq_parts.append(line)
            if current_id is not None:
                seqs[current_id] = "".join(current_seq_parts)
        return seqs
    except Exception:
        return {}


def ungap(s: str) -> str:
    return (s or "").replace(".", "").replace("-", "")


def count_residues(aln_s: str) -> int:
    """Count residues excluding gaps '.' and '-'."""
    if not aln_s:
        return 0
    return sum(1 for c in aln_s if c not in ".-")


def extract_uid_from_header(header: str) -> str:
    uid_match = re.search(
        r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})",
        header
    )
    if uid_match:
        return uid_match.group(1)
    return header.split("/")[0].split("|")[0].split()[0]


def wrap_fasta(seq: str, width: int = 100) -> str:
    seq = seq or ""
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def safe_token(s: str) -> str:
    s = str(s)
    s = s.replace(" ", "_").replace("\t", "_")
    s = re.sub(r"[^A-Za-z0-9._|:+\-=/,]", "_", s)
    return s


# ==============================================================================
# 2) HHM -> MSA mapping
# ==============================================================================
def extract_match_states_from_hhm(hhm_path: str) -> Optional[int]:
    try:
        with open(hhm_path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("LENG"):
                    m = re.search(r"(\d+)\s+match states", line)
                    if m:
                        return int(m.group(1))
                    return None
    except Exception:
        return None
    return None


def extract_hmm_to_msa_map(hhm_path: str) -> Tuple[Dict[int, int], str]:
    if not os.path.exists(hhm_path):
        return {}, "HM_FILE_MISSING"

    mapping: Dict[int, int] = {}
    try:
        match_states = extract_match_states_from_hhm(hhm_path)
        in_hmm_block = False

        with open(hhm_path, "r", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")

                if line.startswith("HMM"):
                    in_hmm_block = True
                    continue
                if line.startswith("//"):
                    break
                if not in_hmm_block:
                    continue

                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                aa = parts[0]
                idx = parts[1]
                msa_col = parts[-1]

                if len(aa) == 1 and aa.isalpha() and idx.isdigit() and msa_col.isdigit():
                    idx_i = int(idx)
                    msa_i = int(msa_col)

                    if match_states is not None and not (1 <= idx_i <= match_states):
                        continue
                    if msa_i <= 0:
                        continue

                    mapping[idx_i] = msa_i

        if not mapping:
            return {}, "HHM_CONTENT_EMPTY"
        return mapping, "OK"
    except Exception as e:
        return {}, f"HHM_PARSE_ERROR:{e}"


def hmm_range_to_msa_range(hmm_map: Dict[int, int], h1: int, h2: int) -> Tuple[Optional[int], Optional[int]]:
    if h1 is None or h2 is None:
        return None, None
    if h1 > h2:
        h1, h2 = h2, h1

    cols = []
    for i in range(h1, h2 + 1):
        c = hmm_map.get(i)
        if c is not None:
            cols.append(c)

    if not cols:
        return None, None
    return int(min(cols)), int(max(cols))


# ==============================================================================
# 2.5) MSA range -> domain-seq range (Pfam only)
# ==============================================================================
def msa_range_to_dom_range(rep_pfam: Dict[str, Any], msa_s: int, msa_e: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Convert an MSA col range (absolute MSA coords) to residue coords (1-based)
    on rep_dom_seq, where rep_dom_seq is ungapped rep_dom_aln_span.
    """
    if msa_s is None or msa_e is None:
        return None, None
    if msa_s > msa_e:
        msa_s, msa_e = msa_e, msa_s

    span_s, span_e = rep_pfam.get("msa_span", (None, None))
    dom_aln = rep_pfam.get("rep_dom_aln_span", "")

    if span_s is None or span_e is None or not dom_aln:
        return None, None

    if msa_e < span_s or msa_s > span_e:
        return None, None

    msa_s2 = max(msa_s, span_s)
    msa_e2 = min(msa_e, span_e)

    rel_s = msa_s2 - span_s
    rel_e = msa_e2 - span_s
    if rel_s < 0 or rel_e >= len(dom_aln) or rel_s > rel_e:
        return None, None

    frag_aln = dom_aln[rel_s:rel_e + 1]
    if count_residues(frag_aln) == 0:
        return None, None

    dom_start = count_residues(dom_aln[:rel_s]) + 1
    dom_end = count_residues(dom_aln[:rel_e + 1])

    if dom_start > dom_end:
        return None, None
    return dom_start, dom_end


# ==============================================================================
# 3) Parse Sub_Segments_Details + collect Pfam target ranges
# ==============================================================================
SEG_RE = re.compile(r"(\d+)\s*-\s*(\d+)\s*->\s*(\d+)\s*-\s*(\d+)")


def parse_sub_segments_details(s) -> List[Tuple[int, int, int, int]]:
    pairs: List[Tuple[int, int, int, int]] = []
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return pairs
    for a1, a2, b1, b2 in SEG_RE.findall(str(s)):
        pairs.append((int(a1), int(a2), int(b1), int(b2)))
    return pairs


def collect_pfam_target_ranges(df: pd.DataFrame) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build Pfam family -> target HMM ranges from CSV.
    Since this is de novo-vs-Pfam:
      Main_HMM = de novo
      Sub_HMM  = Pfam
    We only collect B-side ranges for Pfam representatives.
    """
    fam_to_ranges: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for r in df.itertuples(index=False):
        pfam_id = str(getattr(r, "Sub_HMM"))
        seg_details = getattr(r, "Sub_Segments_Details")

        seg_pairs = parse_sub_segments_details(seg_details)
        for _a1, _a2, b1, b2 in seg_pairs:
            if b1 > b2:
                b1, b2 = b2, b1
            fam_to_ranges[pfam_id].append((b1, b2))

    out: Dict[str, List[Tuple[int, int]]] = {}
    for fam, ranges in fam_to_ranges.items():
        out[fam] = sorted(set(ranges), key=lambda x: (x[0], x[1]))
    return out


# ==============================================================================
# 4) Representative selection
# ==============================================================================
def get_denovo_representative(seq_id: str, seq_dict: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    De novo side: use the sequence itself as representative.
    Use identity hmm_map so HMM coords == sequence coords == MSA coords.
    """
    seq = seq_dict.get(seq_id)
    if not seq:
        return None, "SEQ_NOT_FOUND_IN_FASTA"

    seq = ungap(seq).upper()
    L = len(seq)
    if L <= 0:
        return None, "SEQ_EMPTY"

    return {
        "fam": seq_id,
        "uid": seq_id,
        "header": f"DeNovo_{seq_id}",
        "rep_aln_seq": seq,
        "rep_dom_seq": seq,
        "hmm_map": {i: i for i in range(1, L + 1)},
        "msa_span": (1, L),
        "aln_len": L,
        "type": "DENOVO",
    }, "OK"


def select_pfam_representative(
    fam_id: str,
    pfam_dir: str,
    target_ranges: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Select exactly one representative sequence for a Pfam family.

    New rule:
      For each candidate sequence, compute:
        total_seg_abs_diff = sum(
            abs(len(ungap(candidate[msa_s:msa_e])) - (h2-h1+1))
        ) over all target ranges of this family

      Choose the sequence with the smallest total_seg_abs_diff.

    Tie-break:
      1) fewer EMPTY target windows
      2) smaller family-level abs(len(ungapped_family_span) - HMM_total_len)
      3) fewer gaps inside family span
      4) fewer total gaps in whole alignment

    Notes:
      - This intentionally keeps the original continuous-window extraction logic.
      - It does NOT try to exclude insertion residues separately.
    """
    fam_dir = os.path.join(pfam_dir, fam_id)
    hhm_path = os.path.join(fam_dir, f"{fam_id}.hhm")

    hmm_map, st = extract_hmm_to_msa_map(hhm_path)
    if not hmm_map:
        return None, f"HMM_MAP_FAIL:{st}"

    aln_path = os.path.join(fam_dir, f"{fam_id}.fas")
    if not os.path.exists(aln_path):
        aln_path = os.path.join(fam_dir, f"{fam_id}.sto")

    seqs, st2 = read_alignment(aln_path)
    if not seqs:
        return None, f"ALIGN_READ_FAIL:{st2}"

    aln_len = max(len(s) for s in seqs.values())
    hmm_map = {k: v for k, v in hmm_map.items() if 1 <= v <= aln_len}
    if not hmm_map:
        return None, "HMM_MAP_OUT_OF_RANGE"

    msa_s = min(hmm_map.values())
    msa_e = max(hmm_map.values())
    if msa_s is None or msa_e is None or msa_s > msa_e:
        return None, "MSA_SPAN_INVALID"

    idx_s = msa_s - 1
    idx_e = msa_e

    expected_len = len(hmm_map)
    if expected_len <= 0:
        return None, "HMM_LEN_INVALID"

    target_ranges = target_ranges or []

    best_header = None
    best_aln_seq = None
    best_key = None
    best_metrics = None

    for header, aln_seq in seqs.items():
        if len(aln_seq) < idx_e:
            continue

        span = aln_seq[idx_s:idx_e]
        pure = ungap(span)
        if len(pure) < MIN_PURE_LEN_FOR_REP:
            continue

        gaps_in_span = span.count(".") + span.count("-")
        total_gaps = aln_seq.count(".") + aln_seq.count("-")
        family_len_diff = abs(len(pure) - expected_len)

        total_seg_abs_diff = 0
        n_empty_targets = 0
        n_mapped_targets = 0
        n_unmapped_targets = 0

        if target_ranges:
            for h1, h2 in target_ranges:
                target_len = int(h2) - int(h1) + 1

                msa_t_s, msa_t_e = hmm_range_to_msa_range(hmm_map, h1, h2)
                if msa_t_s is None or msa_t_e is None:
                    total_seg_abs_diff += target_len
                    n_unmapped_targets += 1
                    continue

                n_mapped_targets += 1

                frag_aln = aln_seq[msa_t_s - 1: msa_t_e]
                frag = ungap(frag_aln).upper()
                frag_len = len(frag)

                if frag_len == 0:
                    n_empty_targets += 1

                total_seg_abs_diff += abs(frag_len - target_len)

            key = (
                total_seg_abs_diff,
                n_empty_targets,
                family_len_diff,
                gaps_in_span,
                total_gaps,
                header,
            )
        else:
            key = (
                family_len_diff,
                gaps_in_span,
                total_gaps,
                header,
            )

        if best_key is None or key < best_key:
            best_key = key
            best_header = header
            best_aln_seq = aln_seq
            best_metrics = {
                "total_seg_abs_diff": total_seg_abs_diff,
                "n_empty_targets": n_empty_targets,
                "n_mapped_targets": n_mapped_targets,
                "n_unmapped_targets": n_unmapped_targets,
                "family_len_diff": family_len_diff,
                "target_count": len(target_ranges),
            }

    if best_aln_seq is None:
        return None, "NO_REP_FOUND"

    rep_uid = extract_uid_from_header(best_header)
    rep_dom_aln_span = best_aln_seq[idx_s:idx_e]
    rep_dom_seq = ungap(rep_dom_aln_span).upper()

    out = {
        "fam": fam_id,
        "uid": rep_uid,
        "header": best_header,
        "rep_aln_seq": best_aln_seq,
        "rep_dom_aln_span": rep_dom_aln_span,
        "rep_dom_seq": rep_dom_seq,
        "hmm_map": hmm_map,
        "msa_span": (msa_s, msa_e),
        "aln_len": aln_len,
        "expected_len": expected_len,
        "selected_len": len(rep_dom_seq),
        "selected_len_diff": abs(len(rep_dom_seq) - expected_len),
        "type": "PFAM",
    }

    if best_metrics is not None:
        out.update(best_metrics)

    return out, "OK"


# ==============================================================================
# 5) Aggregate range formatting for frag_dom line
# ==============================================================================
def format_ranges(ranges: List[Tuple[int, int]], max_items: int = 200) -> Tuple[str, int]:
    if not ranges:
        return "NA", 0
    uniq = sorted(set((int(a), int(b)) for a, b in ranges), key=lambda x: (x[0], x[1]))
    total = len(uniq)
    if max_items is not None and max_items > 0 and total > max_items:
        shown = uniq[:max_items]
        s = ",".join(f"{a}-{b}" for a, b in shown) + f",...+{total - max_items}more"
        return s, total
    else:
        s = ",".join(f"{a}-{b}" for a, b in uniq)
        return s, total


# ==============================================================================
# 6) Main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    ap.add_argument("--denovo_fasta", default=DEFAULT_DENOVO_FASTA)
    ap.add_argument("--pfam_dir", default=DEFAULT_PFAM_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)

    ap.add_argument("--debug_max_rows", type=int, default=None,
                    help="If set, process only first N rows.")
    ap.add_argument("--only_ok_fragments", action="store_true",
                    help="If set, only write fragment FASTA for rows whose fragment extraction status == OK.")
    ap.add_argument("--min_local_len", type=int, default=MIN_LOCAL_LEN,
                    help="Skip extracted fragment if ungapped length < this value (applies to both A and B sides).")

    ap.add_argument("--skip_human_txt", action="store_true",
                    help="Skip writing full_sequences.mixed.with_frag_ranges.txt")
    ap.add_argument("--max_frags_in_line", type=int, default=200,
                    help="Max fragment ranges stored in frag_dom line per entity. 0 means unlimited.")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    full_fasta_path = os.path.join(args.out_dir, "full_sequences.mixed.fasta")
    frag_fasta_path = os.path.join(args.out_dir, "conserved_fragments.pairs.fasta")
    tsv_path = os.path.join(args.out_dir, "conserved_fragments.with_fullseq.tsv")
    human_txt_path = os.path.join(args.out_dir, "full_sequences.mixed.with_frag_ranges.txt")

    # --------------------------------------------------------------------------
    # Load input CSV
    # --------------------------------------------------------------------------
    df = pd.read_csv(args.input_csv)
    for c in REQ_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if args.debug_max_rows is not None:
        df = df.head(args.debug_max_rows).copy()

    df = df.reset_index(drop=False).rename(columns={"index": "Source_Row_Index"})

    # --------------------------------------------------------------------------
    # Collect Pfam target ranges from CSV
    # --------------------------------------------------------------------------
    pfam_to_target_ranges = collect_pfam_target_ranges(df)

    # --------------------------------------------------------------------------
    # Load de novo FASTA
    # --------------------------------------------------------------------------
    print(f"[INFO] Loading de novo FASTA: {args.denovo_fasta}")
    denovo_seqs = load_denovo_sequences(args.denovo_fasta)
    if not denovo_seqs:
        raise FileNotFoundError(f"De novo FASTA not found or empty: {args.denovo_fasta}")

    # --------------------------------------------------------------------------
    # Prepare representative cache
    # --------------------------------------------------------------------------
    main_ids = sorted(df["Main_HMM"].astype(str).unique())
    pfam_ids = sorted(df["Sub_HMM"].astype(str).unique())

    denovo_cache: Dict[str, Dict[str, Any]] = {}
    pfam_cache: Dict[str, Dict[str, Any]] = {}

    print(f"[INFO] Preparing de novo representatives for {len(main_ids)} IDs ...")
    for did in tqdm(main_ids, desc="Load de novo"):
        rep, st = get_denovo_representative(did, denovo_seqs)
        if rep is None:
            denovo_cache[did] = {"status": f"ERR:{st}", "fam": did, "type": "DENOVO"}
        else:
            rep["status"] = "OK"
            denovo_cache[did] = rep

    print(f"[INFO] Preparing Pfam representatives for {len(pfam_ids)} families ...")
    for pf in tqdm(pfam_ids, desc="Load Pfam"):
        target_ranges = pfam_to_target_ranges.get(pf, [])
        rep, st = select_pfam_representative(pf, args.pfam_dir, target_ranges=target_ranges)
        if rep is None:
            pfam_cache[pf] = {"status": f"ERR:{st}", "fam": pf, "type": "PFAM"}
        else:
            rep["status"] = "OK"
            pfam_cache[pf] = rep

    # --------------------------------------------------------------------------
    # Write full FASTA
    # --------------------------------------------------------------------------
    print(f"[INFO] Writing full FASTA: {full_fasta_path}")

    n_full_denovo_ok = 0
    n_full_pfam_ok = 0
    n_full_denovo_fail = 0
    n_full_pfam_fail = 0
    written_full_keys: Set[str] = set()

    with open(full_fasta_path, "w") as fw:
        for did in main_ids:
            rep = denovo_cache.get(did, {"status": "NOT_FOUND"})
            if rep.get("status") != "OK":
                n_full_denovo_fail += 1
                continue

            uniq_key = f"DENOVO::{did}"
            if uniq_key in written_full_keys:
                continue
            written_full_keys.add(uniq_key)

            uid = safe_token(rep.get("uid", did))
            seq = rep.get("rep_dom_seq", "") or ""
            msa_s, msa_e = rep.get("msa_span", (None, None))
            src_hdr = safe_token(rep.get("header", f"DeNovo_{did}"))

            fasta_header = (
                f">{did}|uid={uid}|type=denovo_full|len={len(seq)}"
                f"|msa_span={msa_s}-{msa_e}|src_header={src_hdr}"
            )
            fw.write(fasta_header + "\n")
            fw.write(wrap_fasta(seq) + "\n")
            n_full_denovo_ok += 1

        for pf in pfam_ids:
            rep = pfam_cache.get(pf, {"status": "NOT_FOUND"})
            if rep.get("status") != "OK":
                n_full_pfam_fail += 1
                continue

            uniq_key = f"PFAM::{pf}"
            if uniq_key in written_full_keys:
                continue
            written_full_keys.add(uniq_key)

            uid = safe_token(rep.get("uid", "NA"))
            seq = rep.get("rep_dom_seq", "") or ""
            msa_s, msa_e = rep.get("msa_span", (None, None))
            src_hdr = safe_token(rep.get("header", "NA"))
            expected_len = rep.get("expected_len", "NA")
            selected_len_diff = rep.get("selected_len_diff", "NA")
            total_seg_abs_diff = rep.get("total_seg_abs_diff", "NA")

            fasta_header = (
                f">{pf}|uid={uid}|type=pfam_full_rep_domain|len={len(seq)}"
                f"|expected_hmm_len={expected_len}|len_diff={selected_len_diff}"
                f"|seg_abs_diff_sum={total_seg_abs_diff}"
                f"|msa_span={msa_s}-{msa_e}|src_header={src_hdr}"
            )
            fw.write(fasta_header + "\n")
            fw.write(wrap_fasta(seq) + "\n")
            n_full_pfam_ok += 1

    # --------------------------------------------------------------------------
    # Extract fragments and write outputs
    # --------------------------------------------------------------------------
    print(f"[INFO] Writing fragment FASTA: {frag_fasta_path}")
    print(f"[INFO] Writing mapping TSV  : {tsv_path}")

    denovo_to_dom_ranges: Dict[str, List[Tuple[int, int]]] = {did: [] for did in main_ids}
    pfam_to_dom_ranges: Dict[str, List[Tuple[int, int]]] = {pf: [] for pf in pfam_ids}

    tsv_rows: List[Dict[str, Any]] = []
    n_seg_pairs_processed = 0
    n_frag_fasta_entries = 0

    with open(frag_fasta_path, "w") as ffrag:
        for r in tqdm(df.itertuples(index=False), total=len(df), desc="Extract fragments"):
            src_idx = int(getattr(r, "Source_Row_Index"))
            denovo_id = str(getattr(r, "Main_HMM"))
            pfam_id = str(getattr(r, "Sub_HMM"))
            seg_details = getattr(r, "Sub_Segments_Details")

            rep_a = denovo_cache.get(denovo_id, {"status": "MAIN_NOT_LOADED", "fam": denovo_id, "type": "DENOVO"})
            rep_b = pfam_cache.get(pfam_id, {"status": "SUB_NOT_LOADED", "fam": pfam_id, "type": "PFAM"})

            seg_pairs = parse_sub_segments_details(seg_details)
            if not seg_pairs:
                continue

            for seg_i, (a1, a2, b1, b2) in enumerate(seg_pairs, start=1):
                row_base = {
                    "Source_Row_Index": src_idx,
                    "Main_HMM": denovo_id,
                    "Sub_HMM": pfam_id,
                    "Segment_Pair_Index": seg_i,
                    "Sub_Segments_Details_Raw": str(seg_details),
                    "A_HMM_Start": a1,
                    "A_HMM_End": a2,
                    "B_HMM_Start": b1,
                    "B_HMM_End": b2,
                    "RepStatus_A": rep_a.get("status", "NA"),
                    "RepStatus_B": rep_b.get("status", "NA"),
                    "SideA_Type": "DENOVO",
                    "SideB_Type": "PFAM",
                }

                if rep_a.get("status") != "OK" or rep_b.get("status") != "OK":
                    tsv_rows.append({
                        **row_base,
                        "Status": "REP_SELECTION_FAIL",
                        "Side": "",
                        "Source_Type": "",
                        "Entity_ID": "",
                        "Rep_UID": "",
                        "Rep_Header": "",
                        "Full_Rep_Fasta_ID": "",
                        "Fragment_Fasta_ID": "",
                        "Expected_HMM_Len": np.nan,
                        "Selected_Rep_Len": np.nan,
                        "Selected_Rep_Len_Diff": np.nan,
                        "MSA_Start": np.nan,
                        "MSA_End": np.nan,
                        "DOM_Start": np.nan,
                        "DOM_End": np.nan,
                        "Frag_Aligned": "",
                        "Frag_Ungapped": "",
                        "Frag_Ungapped_Len": 0,
                    })
                    n_seg_pairs_processed += 1
                    continue

                msa_a_s, msa_a_e = hmm_range_to_msa_range(rep_a["hmm_map"], a1, a2)
                msa_b_s, msa_b_e = hmm_range_to_msa_range(rep_b["hmm_map"], b1, b2)

                status = "OK"
                if msa_a_s is None or msa_a_e is None:
                    status = "MAP_FAIL_A"
                if msa_b_s is None or msa_b_e is None:
                    status = "MAP_FAIL_B" if status == "OK" else f"{status}|MAP_FAIL_B"

                frag_a_aln = ""
                frag_b_aln = ""
                frag_a = ""
                frag_b = ""
                dom_b_s = None
                dom_b_e = None

                if status == "OK":
                    try:
                        frag_a_aln = rep_a["rep_aln_seq"][msa_a_s - 1: msa_a_e]
                        frag_b_aln = rep_b["rep_aln_seq"][msa_b_s - 1: msa_b_e]

                        frag_a = ungap(frag_a_aln).upper()
                        frag_b = ungap(frag_b_aln).upper()

                        if len(frag_a) < args.min_local_len:
                            status = f"SHORT_FRAGMENT_A(<{args.min_local_len})"
                        if len(frag_b) < args.min_local_len:
                            status = (
                                f"SHORT_FRAGMENT_B(<{args.min_local_len})"
                                if status == "OK"
                                else f"{status}|SHORT_FRAGMENT_B(<{args.min_local_len})"
                            )

                        if status == "OK":
                            denovo_to_dom_ranges.setdefault(denovo_id, []).append((msa_a_s, msa_a_e))

                            dom_b_s, dom_b_e = msa_range_to_dom_range(rep_b, msa_b_s, msa_b_e)
                            if dom_b_s is not None and dom_b_e is not None:
                                pfam_to_dom_ranges.setdefault(pfam_id, []).append((dom_b_s, dom_b_e))

                    except Exception as e:
                        status = f"EXTRACT_ERR:{e}"

                full_id_a = f"{denovo_id}|uid={safe_token(rep_a.get('uid', denovo_id))}|type=denovo_full"
                full_id_b = f"{pfam_id}|uid={safe_token(rep_b.get('uid', 'NA'))}|type=pfam_full_rep_domain"

                frag_id_a = (
                    f"{denovo_id}|uid={safe_token(rep_a.get('uid', denovo_id))}|side=A|src=denovo"
                    f"|srcrow={src_idx}"
                    f"|hmm={a1}-{a2}|msa={msa_a_s}-{msa_a_e if msa_a_e is not None else 'NA'}"
                    f"|mate={pfam_id}:{b1}-{b2}"
                )
                frag_id_b = (
                    f"{pfam_id}|uid={safe_token(rep_b.get('uid', 'NA'))}|side=B|src=pfam"
                    f"|srcrow={src_idx}"
                    f"|hmm={b1}-{b2}|msa={msa_b_s}-{msa_b_e if msa_b_e is not None else 'NA'}"
                    f"|mate={denovo_id}:{a1}-{a2}"
                )

                if (status == "OK") or (not args.only_ok_fragments):
                    if frag_a:
                        ffrag.write(f">{frag_id_a}\n")
                        ffrag.write(wrap_fasta(frag_a) + "\n")
                        n_frag_fasta_entries += 1

                    if frag_b:
                        ffrag.write(f">{frag_id_b}\n")
                        ffrag.write(wrap_fasta(frag_b) + "\n")
                        n_frag_fasta_entries += 1

                tsv_rows.append({
                    **row_base,
                    "Status": status,
                    "Side": "A",
                    "Source_Type": "DENOVO",
                    "Entity_ID": denovo_id,
                    "Rep_UID": rep_a.get("uid", ""),
                    "Rep_Header": rep_a.get("header", ""),
                    "Full_Rep_Fasta_ID": full_id_a,
                    "Fragment_Fasta_ID": frag_id_a,
                    "Expected_HMM_Len": np.nan,
                    "Selected_Rep_Len": len(rep_a.get("rep_dom_seq", "") or ""),
                    "Selected_Rep_Len_Diff": np.nan,
                    "MSA_Start": int(msa_a_s) if msa_a_s is not None else np.nan,
                    "MSA_End": int(msa_a_e) if msa_a_e is not None else np.nan,
                    "DOM_Start": int(msa_a_s) if msa_a_s is not None else np.nan,
                    "DOM_End": int(msa_a_e) if msa_a_e is not None else np.nan,
                    "Frag_Aligned": frag_a_aln,
                    "Frag_Ungapped": frag_a,
                    "Frag_Ungapped_Len": len(frag_a) if frag_a else 0,
                })

                tsv_rows.append({
                    **row_base,
                    "Status": status,
                    "Side": "B",
                    "Source_Type": "PFAM",
                    "Entity_ID": pfam_id,
                    "Rep_UID": rep_b.get("uid", ""),
                    "Rep_Header": rep_b.get("header", ""),
                    "Full_Rep_Fasta_ID": full_id_b,
                    "Fragment_Fasta_ID": frag_id_b,
                    "Expected_HMM_Len": rep_b.get("expected_len", np.nan),
                    "Selected_Rep_Len": rep_b.get("selected_len", np.nan),
                    "Selected_Rep_Len_Diff": rep_b.get("selected_len_diff", np.nan),
                    "MSA_Start": int(msa_b_s) if msa_b_s is not None else np.nan,
                    "MSA_End": int(msa_b_e) if msa_b_e is not None else np.nan,
                    "DOM_Start": int(dom_b_s) if dom_b_s is not None else np.nan,
                    "DOM_End": int(dom_b_e) if dom_b_e is not None else np.nan,
                    "Frag_Aligned": frag_b_aln,
                    "Frag_Ungapped": frag_b,
                    "Frag_Ungapped_Len": len(frag_b) if frag_b else 0,
                })

                n_seg_pairs_processed += 1

    tsv_df = pd.DataFrame(tsv_rows)
    tsv_df.to_csv(tsv_path, sep="\t", index=False)

    # --------------------------------------------------------------------------
    # Human TXT-like output
    # Format aligned with your requested style
    # --------------------------------------------------------------------------
    if not args.skip_human_txt:
        print(f"[INFO] Writing human TXT (full + frag_dom line): {human_txt_path}")
        max_items = args.max_frags_in_line
        if max_items == 0:
            max_items = 10**18

        with open(human_txt_path, "w") as fw:
            for did in main_ids:
                rep = denovo_cache.get(did, {"status": "NOT_FOUND"})
                if rep.get("status") != "OK":
                    continue

                dom_ranges = denovo_to_dom_ranges.get(did, [])
                if not dom_ranges:
                    continue

                uid = safe_token(rep.get("uid", did))
                seq = rep.get("rep_dom_seq", "") or ""
                msa_s, msa_e = rep.get("msa_span", (None, None))
                src_hdr = safe_token(rep.get("header", f"DeNovo_{did}"))

                header_line = (
                    f">{did}|uid={uid}|type=denovo_full|len={len(seq)}"
                    f"|msa_span={msa_s}-{msa_e}|src_header={src_hdr}"
                )
                frag_dom_str, _ = format_ranges(dom_ranges, max_items=max_items)

                fw.write(header_line + "\n")
                fw.write(f"frag_dom={frag_dom_str}\n")
                fw.write(wrap_fasta(seq) + "\n")

            for pf in pfam_ids:
                rep = pfam_cache.get(pf, {"status": "NOT_FOUND"})
                if rep.get("status") != "OK":
                    continue

                dom_ranges = pfam_to_dom_ranges.get(pf, [])
                if not dom_ranges:
                    continue

                uid = safe_token(rep.get("uid", "NA"))
                seq = rep.get("rep_dom_seq", "") or ""
                msa_s, msa_e = rep.get("msa_span", (None, None))
                src_hdr = safe_token(rep.get("header", "NA"))

                header_line = (
                    f">{pf}|uid={uid}|type=full_rep_domain|len={len(seq)}"
                    f"|msa_span={msa_s}-{msa_e}|src_header={src_hdr}"
                )
                frag_dom_str, _ = format_ranges(dom_ranges, max_items=max_items)

                fw.write(header_line + "\n")
                fw.write(f"frag_dom={frag_dom_str}\n")
                fw.write(wrap_fasta(seq) + "\n")

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    print("\n[DONE]")
    print(f"  Full FASTA (de novo + Pfam rep): {full_fasta_path}")
    print(f"  Fragment FASTA                : {frag_fasta_path}")
    print(f"  Mapping TSV                   : {tsv_path}")
    if not args.skip_human_txt:
        print(f"  Human TXT (full+fragdom)      : {human_txt_path}")
    print("")
    print(f"[INFO] De novo full written      : {n_full_denovo_ok}")
    print(f"[INFO] De novo full failed       : {n_full_denovo_fail}")
    print(f"[INFO] Pfam full written         : {n_full_pfam_ok}")
    print(f"[INFO] Pfam full failed          : {n_full_pfam_fail}")
    print(f"[INFO] Segment pairs processed   : {n_seg_pairs_processed}")
    print(f"[INFO] Fragment FASTA entries    : {n_frag_fasta_entries} (A+B combined)")

    denovo_failed = [(k, v.get("status")) for k, v in denovo_cache.items() if v.get("status") != "OK"]
    pfam_failed = [(k, v.get("status")) for k, v in pfam_cache.items() if v.get("status") != "OK"]

    if denovo_failed:
        print("\n[WARN] De novo representative load failures (showing up to 30):")
        for k, st in denovo_failed[:30]:
            print(f"  {k}: {st}")

    if pfam_failed:
        print("\n[WARN] Pfam representative selection failures (showing up to 30):")
        for k, st in pfam_failed[:30]:
            print(f"  {k}: {st}")

    if len(tsv_df) > 0 and "Status" in tsv_df.columns:
        print("\n[INFO] Fragment extraction Status summary (TSV rows, A/B both included):")
        print(tsv_df["Status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()