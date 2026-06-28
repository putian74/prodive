#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Purpose
-------
Given a result CSV with:
  - Main_HMM
  - Sub_HMM
  - Sub_Segments_Details (e.g., "a-b -> c-d; ...")

This script:
  1) Selects ONE representative sequence per Pfam family from
     PFAM_DIR/<PFxxxxx>/<PFxxxxx>.fas or .sto

     Representative selection rule (your requested version):
       - collect all target HMM segments actually used by this family in the CSV
       - for each candidate aligned sequence:
           * for each target segment h1-h2:
               - map it to continuous MSA window msa_s~msa_e
               - extract that continuous window from the candidate sequence
               - ungap it
               - compute abs(len(fragment) - (h2-h1+1))
           * sum these absolute differences across all target segments
       - choose the sequence with the smallest total difference
       - tie-break by:
           1) fewer EMPTY target windows
           2) smaller family-level len_diff to HMM total length
           3) fewer gaps inside family span
           4) fewer total gaps in whole alignment

     IMPORTANT:
       - This version intentionally keeps your original fragment extraction logic:
         continuous MSA window + ungap
       - It does NOT try to exclude insertion residues separately

  2) Parses conserved/local regions from Sub_Segments_Details ("a-b -> c-d")
  3) Maps HMM ranges to MSA ranges on the selected representatives
  4) Extracts representative aligned fragments and ungapped fragment sequences
  5) Exports:
       - representatives.full.fasta
       - conserved_fragments.pairs.fasta
       - conserved_fragments.with_fullseq.tsv
  6) Exports a human-readable combined text file (NOT valid FASTA):
       - representatives.full.with_frag_ranges.txt
         format:
           >PFxxxxx|uid=...|type=full_rep_domain|len=...|msa_span=...|src_header=...
           frag_dom=1-32,10-30,5-51
           SEQUENCE...

No NW alignment/scoring is performed.
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
DEFAULT_INPUT_CSV = "<PRODIVE_DATA_ROOT>/shared/global_high_score_summary_fin.csv"
DEFAULT_PFAM_DIR = "<PRODIVE_DATA_ROOT>/shared/PfamA_seed/"
DEFAULT_OUT_DIR = "CHANGE_ME"

MIN_PURE_LEN_FOR_REP = 3
REQ_COLS = ["Main_HMM", "Sub_HMM", "Sub_Segments_Details"]


# ==============================================================================
# 1) Helpers: alignment reading (FASTA / Stockholm)
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
    return header.split("/")[0].split()[0]


def wrap_fasta(seq: str, width: int = 100) -> str:
    seq = seq or ""
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def safe_token(s: str) -> str:
    """Make string safer for header tokens (NOT for strict FASTA parsing)."""
    s = str(s)
    s = s.replace(" ", "_").replace("\t", "_")
    s = re.sub(r"[^A-Za-z0-9.,_|:+\-=/]", "_", s)
    return s


# ==============================================================================
# 2) Robust HHM -> MSA mapping
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


def msa_range_to_dom_range(rep: Dict[str, Any], msa_s: int, msa_e: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Convert an MSA column range to domain-seq residue coords (1-based) on rep_dom_seq.

    Domain is defined as rep_aln_seq[msa_span_start-1 : msa_span_end] ungapped.
    """
    if msa_s is None or msa_e is None:
        return None, None
    if msa_s > msa_e:
        msa_s, msa_e = msa_e, msa_s

    span_s, span_e = rep.get("msa_span", (None, None))
    if span_s is None or span_e is None:
        return None, None

    if msa_e < span_s or msa_s > span_e:
        return None, None

    msa_s2 = max(msa_s, span_s)
    msa_e2 = min(msa_e, span_e)

    dom_aln = rep.get("rep_dom_aln_span", "")
    if not dom_aln:
        try:
            dom_aln = rep["rep_aln_seq"][span_s - 1: span_e]
        except Exception:
            return None, None

    rel_s = msa_s2 - span_s
    rel_e = msa_e2 - span_s
    if rel_s < 0 or rel_e >= len(dom_aln) or rel_s > rel_e:
        return None, None

    frag_aln = dom_aln[rel_s: rel_e + 1]
    if count_residues(frag_aln) == 0:
        return None, None

    dom_start = count_residues(dom_aln[:rel_s]) + 1
    dom_end = count_residues(dom_aln[:rel_e + 1])

    if dom_start > dom_end:
        return None, None
    return dom_start, dom_end


# ==============================================================================
# 3) Parse Sub_Segments_Details and collect target ranges
# ==============================================================================
SEG_RE = re.compile(r"(\d+)\s*-\s*(\d+)\s*->\s*(\d+)\s*-\s*(\d+)")


def parse_sub_segments_details(s) -> List[Tuple[int, int, int, int]]:
    pairs: List[Tuple[int, int, int, int]] = []
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return pairs
    for a1, a2, b1, b2 in SEG_RE.findall(str(s)):
        pairs.append((int(a1), int(a2), int(b1), int(b2)))
    return pairs


def collect_family_target_ranges(df: pd.DataFrame) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build family -> target HMM ranges from the actual CSV rows.

    For each row:
      Main_HMM gets (a1, a2)
      Sub_HMM  gets (b1, b2)
    """
    fam_to_ranges: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for r in df.itertuples(index=False):
        fam_a = str(getattr(r, "Main_HMM"))
        fam_b = str(getattr(r, "Sub_HMM"))
        seg_details = getattr(r, "Sub_Segments_Details")

        seg_pairs = parse_sub_segments_details(seg_details)
        for a1, a2, b1, b2 in seg_pairs:
            if a1 > a2:
                a1, a2 = a2, a1
            if b1 > b2:
                b1, b2 = b2, b1

            fam_to_ranges[fam_a].append((a1, a2))
            fam_to_ranges[fam_b].append((b1, b2))

    out: Dict[str, List[Tuple[int, int]]] = {}
    for fam, ranges in fam_to_ranges.items():
        out[fam] = sorted(set(ranges), key=lambda x: (x[0], x[1]))
    return out


# ==============================================================================
# 4) Representative selection per family
# ==============================================================================
def select_family_representative(
    fam_id: str,
    pfam_dir: str,
    target_ranges: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Select exactly one representative per family.

    Rule:
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
    }

    if best_metrics is not None:
        out.update(best_metrics)

    return out, "OK"


# ==============================================================================
# 5) Formatting aggregated fragment ranges
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
    ap.add_argument("--pfam_dir", default=DEFAULT_PFAM_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--debug_max_rows", type=int, default=None,
                    help="If set, process only first N rows.")
    ap.add_argument("--only_ok_fragments", action="store_true",
                    help="If set, only export fragments with Status == OK.")

    ap.add_argument("--skip_full_fasta", action="store_true",
                    help="Skip writing representatives.full.fasta")
    ap.add_argument("--skip_frag_fasta", action="store_true",
                    help="Skip writing conserved_fragments.pairs.fasta")
    ap.add_argument("--skip_human_txt", action="store_true",
                    help="Skip writing representatives.full.with_frag_ranges.txt")
    ap.add_argument("--max_frags_in_line", type=int, default=200,
                    help="Max fragment ranges stored in frag_dom line per family. 0 means unlimited.")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    full_fasta_path = os.path.join(args.out_dir, "representatives.full.fasta")
    frag_fasta_path = os.path.join(args.out_dir, "conserved_fragments.pairs.fasta")
    tsv_path = os.path.join(args.out_dir, "conserved_fragments.with_fullseq.tsv")
    human_txt_path = os.path.join(args.out_dir, "representatives.full.with_frag_ranges.txt")

    # -------------------------
    # Load input
    # -------------------------
    df = pd.read_csv(args.input_csv)
    for c in REQ_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if args.debug_max_rows is not None:
        df = df.head(args.debug_max_rows).copy()

    df = df.reset_index(drop=False).rename(columns={"index": "Source_Row_Index"})

    # -------------------------
    # Collect family target ranges from CSV
    # -------------------------
    fam_to_target_ranges = collect_family_target_ranges(df)

    # -------------------------
    # Preload representatives
    # -------------------------
    fams = sorted(set(df["Main_HMM"].astype(str)).union(set(df["Sub_HMM"].astype(str))))
    fam_cache: Dict[str, Dict[str, Any]] = {}

    print(f"[INFO] Loading representatives for {len(fams)} families ...")
    for fam in tqdm(fams, desc="Load family reps"):
        target_ranges = fam_to_target_ranges.get(fam, [])
        rep, status = select_family_representative(
            fam,
            args.pfam_dir,
            target_ranges=target_ranges
        )
        if rep is None:
            fam_cache[fam] = {"status": status, "fam": fam}
        else:
            rep["status"] = "OK"
            fam_cache[fam] = rep

    # -------------------------
    # Write full representative FASTA (one per family)
    # -------------------------
    n_full_ok = 0
    n_full_fail = 0
    written_full: Set[str] = set()

    if not args.skip_full_fasta:
        print(f"[INFO] Writing full representative FASTA: {full_fasta_path}")
        with open(full_fasta_path, "w") as fw:
            for fam in fams:
                rep = fam_cache.get(fam, {"status": "FAM_NOT_LOADED"})
                if rep.get("status") != "OK":
                    n_full_fail += 1
                    continue

                if fam in written_full:
                    continue
                written_full.add(fam)

                uid = safe_token(rep.get("uid", "NA"))
                hdr = safe_token(rep.get("header", "NA"))
                msa_s, msa_e = rep.get("msa_span", (None, None))
                seq = rep.get("rep_dom_seq", "") or ""
                expected_len = rep.get("expected_len", "NA")
                selected_len_diff = rep.get("selected_len_diff", "NA")
                total_seg_abs_diff = rep.get("total_seg_abs_diff", "NA")

                fasta_header = (
                    f">{fam}|uid={uid}|type=full_rep_domain|len={len(seq)}"
                    f"|expected_hmm_len={expected_len}|len_diff={selected_len_diff}"
                    f"|seg_abs_diff_sum={total_seg_abs_diff}"
                    f"|msa_span={msa_s}-{msa_e}|src_header={hdr}"
                )
                fw.write(fasta_header + "\n")
                fw.write(wrap_fasta(seq) + "\n")
                n_full_ok += 1
    else:
        for fam in fams:
            rep = fam_cache.get(fam, {"status": "FAM_NOT_LOADED"})
            if rep.get("status") == "OK":
                n_full_ok += 1
            else:
                n_full_fail += 1

    # -------------------------
    # Extract fragments and write fragment FASTA + mapping TSV
    # Also aggregate fragment ranges per family (domain coords on rep_dom_seq)
    # -------------------------
    print(f"[INFO] Writing mapping TSV  : {tsv_path}")

    fam_to_dom_ranges: Dict[str, List[Tuple[int, int]]] = {fam: [] for fam in fams}

    tsv_rows: List[Dict[str, Any]] = []
    n_seg_pairs = 0
    n_frag_fasta_entries = 0
    n_frag_ok = 0

    ffrag = None
    if not args.skip_frag_fasta:
        print(f"[INFO] Writing fragment FASTA: {frag_fasta_path}")
        ffrag = open(frag_fasta_path, "w")

    try:
        for r in tqdm(df.itertuples(index=False), total=len(df), desc="Extract & write fragments"):
            src_idx = int(getattr(r, "Source_Row_Index"))
            fam_a = str(getattr(r, "Main_HMM"))
            fam_b = str(getattr(r, "Sub_HMM"))
            seg_details = getattr(r, "Sub_Segments_Details")

            rep_a = fam_cache.get(fam_a, {"status": "FAM_NOT_LOADED", "fam": fam_a})
            rep_b = fam_cache.get(fam_b, {"status": "FAM_NOT_LOADED", "fam": fam_b})

            seg_pairs = parse_sub_segments_details(seg_details)
            if not seg_pairs:
                continue

            for seg_i, (a1, a2, b1, b2) in enumerate(seg_pairs, start=1):
                n_seg_pairs += 1

                row_base = {
                    "Source_Row_Index": src_idx,
                    "Main_HMM": fam_a,
                    "Sub_HMM": fam_b,
                    "Segment_Pair_Index": seg_i,
                    "Sub_Segments_Details_Raw": str(seg_details),
                    "A_HMM_Start": a1, "A_HMM_End": a2,
                    "B_HMM_Start": b1, "B_HMM_End": b2,
                    "RepStatus_A": rep_a.get("status", "NA"),
                    "RepStatus_B": rep_b.get("status", "NA"),
                }

                if rep_a.get("status") != "OK" or rep_b.get("status") != "OK":
                    tsv_rows.append({
                        **row_base,
                        "Status": "REP_SELECTION_FAIL",
                        "Side": "",
                        "Pfam": "",
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
                    continue

                msa_a_s, msa_a_e = hmm_range_to_msa_range(rep_a["hmm_map"], a1, a2)
                msa_b_s, msa_b_e = hmm_range_to_msa_range(rep_b["hmm_map"], b1, b2)

                status = "OK"
                if msa_a_s is None or msa_a_e is None:
                    status = "MAP_FAIL_A"
                if msa_b_s is None or msa_b_e is None:
                    status = "MAP_FAIL_B" if status == "OK" else f"{status}|MAP_FAIL_B"

                frag_a_aln = frag_b_aln = ""
                frag_a = frag_b = ""
                dom_a_s = dom_a_e = None
                dom_b_s = dom_b_e = None

                if status == "OK":
                    try:
                        frag_a_aln = rep_a["rep_aln_seq"][msa_a_s - 1: msa_a_e]
                        frag_b_aln = rep_b["rep_aln_seq"][msa_b_s - 1: msa_b_e]
                        frag_a = ungap(frag_a_aln).upper()
                        frag_b = ungap(frag_b_aln).upper()

                        if len(frag_a) == 0:
                            status = "EMPTY_FRAGMENT_A"
                        if len(frag_b) == 0:
                            status = "EMPTY_FRAGMENT_B" if status == "OK" else f"{status}|EMPTY_FRAGMENT_B"

                        if status == "OK":
                            dom_a_s, dom_a_e = msa_range_to_dom_range(rep_a, msa_a_s, msa_a_e)
                            dom_b_s, dom_b_e = msa_range_to_dom_range(rep_b, msa_b_s, msa_b_e)

                            if dom_a_s is not None and dom_a_e is not None:
                                fam_to_dom_ranges.setdefault(fam_a, []).append((dom_a_s, dom_a_e))
                            if dom_b_s is not None and dom_b_e is not None:
                                fam_to_dom_ranges.setdefault(fam_b, []).append((dom_b_s, dom_b_e))

                    except Exception as e:
                        status = f"EXTRACT_ERR:{e}"

                full_id_a = f"{fam_a}|uid={safe_token(rep_a.get('uid', 'NA'))}|type=full_rep_domain"
                full_id_b = f"{fam_b}|uid={safe_token(rep_b.get('uid', 'NA'))}|type=full_rep_domain"

                frag_id_a = (
                    f"{fam_a}|uid={safe_token(rep_a.get('uid', 'NA'))}|side=A"
                    f"|srcrow={src_idx}"
                    f"|hmm={a1}-{a2}|msa={msa_a_s}-{msa_a_e if msa_a_e is not None else 'NA'}"
                    f"|mate={fam_b}:{b1}-{b2}"
                )
                frag_id_b = (
                    f"{fam_b}|uid={safe_token(rep_b.get('uid', 'NA'))}|side=B"
                    f"|srcrow={src_idx}"
                    f"|hmm={b1}-{b2}|msa={msa_b_s}-{msa_b_e if msa_b_e is not None else 'NA'}"
                    f"|mate={fam_a}:{a1}-{a2}"
                )

                if ffrag is not None and ((status == "OK") or (not args.only_ok_fragments)):
                    if frag_a:
                        ffrag.write(f">{frag_id_a}\n")
                        ffrag.write(wrap_fasta(frag_a) + "\n")
                        n_frag_fasta_entries += 1
                        if status == "OK":
                            n_frag_ok += 1

                    if frag_b:
                        ffrag.write(f">{frag_id_b}\n")
                        ffrag.write(wrap_fasta(frag_b) + "\n")
                        n_frag_fasta_entries += 1
                        if status == "OK":
                            n_frag_ok += 1

                tsv_rows.append({
                    **row_base,
                    "Status": status,
                    "Side": "A",
                    "Pfam": fam_a,
                    "Rep_UID": rep_a.get("uid", ""),
                    "Rep_Header": rep_a.get("header", ""),
                    "Full_Rep_Fasta_ID": full_id_a,
                    "Fragment_Fasta_ID": frag_id_a,
                    "Expected_HMM_Len": rep_a.get("expected_len", np.nan),
                    "Selected_Rep_Len": rep_a.get("selected_len", np.nan),
                    "Selected_Rep_Len_Diff": rep_a.get("selected_len_diff", np.nan),
                    "MSA_Start": int(msa_a_s) if msa_a_s is not None else np.nan,
                    "MSA_End": int(msa_a_e) if msa_a_e is not None else np.nan,
                    "DOM_Start": int(dom_a_s) if dom_a_s is not None else np.nan,
                    "DOM_End": int(dom_a_e) if dom_a_e is not None else np.nan,
                    "Frag_Aligned": frag_a_aln,
                    "Frag_Ungapped": frag_a,
                    "Frag_Ungapped_Len": len(frag_a) if frag_a else 0,
                })

                tsv_rows.append({
                    **row_base,
                    "Status": status,
                    "Side": "B",
                    "Pfam": fam_b,
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

    finally:
        if ffrag is not None:
            ffrag.close()

    tsv_df = pd.DataFrame(tsv_rows)
    tsv_df.to_csv(tsv_path, sep="\t", index=False)

    # -------------------------
    # Human TXT-like output (NOT FASTA)
    # Format exactly like your example
    # -------------------------
    if not args.skip_human_txt:
        print(f"[INFO] Writing human TXT (full + frag_dom line): {human_txt_path}")
        max_items = args.max_frags_in_line
        if max_items == 0:
            max_items = 10 ** 18

        with open(human_txt_path, "w") as fw:
            for fam in fams:
                rep = fam_cache.get(fam, {"status": "FAM_NOT_LOADED"})
                if rep.get("status") != "OK":
                    continue

                dom_ranges = fam_to_dom_ranges.get(fam, [])
                if not dom_ranges:
                    continue

                uid = safe_token(rep.get("uid", "NA"))
                hdr = safe_token(rep.get("header", "NA"))
                msa_s, msa_e = rep.get("msa_span", (None, None))
                seq = rep.get("rep_dom_seq", "") or ""

                frag_dom_str, frag_dom_n = format_ranges(dom_ranges, max_items=max_items)

                header_line = (
                    f">{fam}|uid={uid}|type=full_rep_domain|len={len(seq)}"
                    f"|msa_span={msa_s}-{msa_e}|src_header={hdr}"
                )
                fw.write(header_line + "\n")
                fw.write(f"frag_dom={frag_dom_str}\n")
                fw.write(wrap_fasta(seq, width=100) + "\n")

    # -------------------------
    # Summaries
    # -------------------------
    print("\n[DONE]")
    if not args.skip_full_fasta:
        print(f"  Full representative FASTA : {full_fasta_path}")
    if not args.skip_frag_fasta:
        print(f"  Fragment FASTA           : {frag_fasta_path}")
    if not args.skip_human_txt:
        print(f"  Human TXT (full+fragdom) : {human_txt_path}")
    print(f"  Mapping TSV              : {tsv_path}")
    print("")
    print(f"[INFO] Families rep-ok           : {n_full_ok}")
    print(f"[INFO] Families rep-failed       : {n_full_fail}")
    print(f"[INFO] Segment pairs processed   : {n_seg_pairs}")
    if not args.skip_frag_fasta:
        print(f"[INFO] Fragment FASTA entries    : {n_frag_fasta_entries} (A+B combined)")
        print(f"[INFO] OK fragments written      : {n_frag_ok} (A+B combined)")

    failed = [(fam, v.get("status")) for fam, v in fam_cache.items() if v.get("status") != "OK"]
    if failed:
        print("\n[WARN] Representative selection failures (showing up to 30):")
        for fam, st in failed[:30]:
            print(f"  {fam}: {st}")

    if len(tsv_df) > 0 and "Status" in tsv_df.columns:
        print("\n[INFO] Fragment extraction Status summary (TSV rows, A/B both included):")
        print(tsv_df["Status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()