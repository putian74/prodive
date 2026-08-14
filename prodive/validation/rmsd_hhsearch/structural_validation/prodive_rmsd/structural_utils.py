#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared structural-validation utilities for ProDive analysis scripts.

This module contains the common logic used by the RMSD scripts:
  - HHM match-state to MSA-column mapping
  - Pfam seed alignment parsing
  - seed-sequence fragment to residue-coordinate mapping
  - hybrid structure lookup: experimental PDB first, AlphaFold fallback
  - AlphaFold pLDDT / optional PAE quality filtering
  - PyMOL C-alpha superposition and RMSD extraction

The module is intentionally analysis-oriented rather than a general-purpose API.
It keeps the original behavior of the working scripts while removing duplicated
code and making paths/thresholds configurable.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import PDB

import pymol
from pymol import cmd


@dataclass
class RMSDConfig:
    pfam_dir: str
    max_search_depth: int = 200
    min_candidate_pure_len: int = 3
    min_mean_plddt: float = 70.0
    max_head_tail_pae: float = 10.0
    len_diff_mode: str = "fixed"  # fixed or relative
    max_len_diff: int = 5
    min_len_diff: int = 5
    rel_len_diff_ratio: float = 0.2
    min_aligned_coverage: float = 0.0  # 0 disables coverage filtering
    pymol_quiet: bool = True

    def allowed_len_diff(self, expected_len: int) -> int:
        if self.len_diff_mode == "relative":
            if expected_len is None or expected_len <= 0:
                return self.min_len_diff
            return max(self.min_len_diff, int(math.ceil(self.rel_len_diff_ratio * expected_len)))
        return self.max_len_diff


def init_pymol(quiet: bool = True) -> None:
    args = ["pymol", "-c"]
    if quiet:
        args.append("-q")
    pymol.finish_launching(args)


def is_false_like(v: Any) -> bool:
    """Robust test for Is_Novel == False across bool/string/numeric CSV variants."""
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v) is False
    s = str(v).strip().lower()
    return s in {"false", "0", "no"}


def is_true_like(v: Any) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v) is True
    s = str(v).strip().lower()
    return s in {"true", "1", "yes"}


def ungap(s: str) -> str:
    return (s or "").replace(".", "").replace("-", "")


def count_residues(s: str) -> int:
    return len(ungap(s))


def parse_segment(text: Any) -> Tuple[int, int]:
    """
    Parse common segment formats:
      (1, 30), [1, 30], 1,30, 1-30
    """
    if pd.isna(text):
        raise ValueError("Segment is NaN")

    s = str(text).strip()

    m = re.match(r"^\(?\s*(\d+)\s*-\s*(\d+)\s*\)?$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a, b) if a <= b else (b, a)

    nums = re.findall(r"-?\d+", s)
    if len(nums) < 2:
        raise ValueError(f"Cannot parse segment: {text}")

    a, b = int(nums[0]), int(nums[1])
    return (a, b) if a <= b else (b, a)


def extract_uid_from_header(header: str) -> str:
    uid_match = re.search(
        r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})",
        header,
    )
    if uid_match:
        return uid_match.group(1)
    return header.split("/")[0].split()[0]


def count_csv_rows(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", errors="ignore") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def extract_match_states_from_hhm(hhm_path: str) -> Optional[int]:
    if not os.path.exists(hhm_path):
        return None
    try:
        with open(hhm_path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("LENG"):
                    m = re.search(r"(\d+)", line)
                    if m:
                        return int(m.group(1))
    except Exception:
        return None
    return None


def extract_hmm_to_msa_map(hhm_path: str) -> Tuple[Dict[int, int], str]:
    """Parse HHsuite HHM HMM block: match-state index -> MSA column."""
    if not os.path.exists(hhm_path):
        return {}, "HHM_FILE_MISSING"

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
                    hmm_idx = int(idx)
                    msa_idx = int(msa_col)

                    if match_states is not None and not (1 <= hmm_idx <= match_states):
                        continue
                    if msa_idx <= 0:
                        continue

                    mapping[hmm_idx] = msa_idx

        if not mapping:
            return {}, "HHM_CONTENT_EMPTY"
        return mapping, "OK"
    except Exception:
        return {}, "HHM_PARSE_ERROR"


def hmm_range_to_msa_range(hmm_map: Dict[int, int], h1: int, h2: int) -> Tuple[Optional[int], Optional[int]]:
    """Convert an HMM interval [h1, h2] into the covered MSA-column interval."""
    if h1 is None or h2 is None:
        return None, None
    if h1 > h2:
        h1, h2 = h2, h1

    cols = [hmm_map[i] for i in range(h1, h2 + 1) if i in hmm_map]
    if not cols:
        return None, None
    return int(min(cols)), int(max(cols))


def read_alignment(aln_path: str) -> Tuple[Dict[str, str], str]:
    """Read FASTA or relaxed Stockholm alignment into header -> aligned sequence."""
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
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        current_header = line[1:].strip()
                        seqs[current_header] = ""
                    elif current_header is not None:
                        seqs[current_header] += line
            else:
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


def get_candidates_from_alignment(
    aln_path: str,
    msa_start_col: int,
    msa_end_col: int,
    expected_len: int,
    cfg: RMSDConfig,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Build candidate structure residue ranges from a Pfam seed alignment.

    expected_len should be the HMM interval length, not the MSA-column span.
    """
    seqs, status = read_alignment(aln_path)
    if not seqs:
        return [], status

    candidates = []
    idx_start = msa_start_col - 1
    idx_end = msa_end_col

    for header, seq in seqs.items():
        if len(seq) < idx_end:
            continue

        match = re.search(r"/(\d+)-(\d+)", header)
        if not match:
            continue

        global_offset = int(match.group(1))
        uid = extract_uid_from_header(header)

        frag_raw = seq[idx_start:idx_end]
        gaps = frag_raw.count(".") + frag_raw.count("-")
        frag_pure = ungap(frag_raw)

        if len(frag_pure) < cfg.min_candidate_pure_len:
            continue

        prefix = seq[:idx_start]
        local_res_before = count_residues(prefix)
        pdb_start = global_offset + local_res_before
        pdb_end = pdb_start + len(frag_pure) - 1

        candidates.append({
            "uid": uid,
            "pdb_start": pdb_start,
            "pdb_end": pdb_end,
            "gaps": gaps,
            "seq_len": len(frag_pure),
            "len_diff": abs(len(frag_pure) - expected_len),
        })

    candidates.sort(key=lambda x: (x["len_diff"], x["gaps"]))
    return candidates[: cfg.max_search_depth], "OK"


def parse_exp_chain(filename: str) -> Optional[str]:
    # e.g. G0S2I7_exp_8i9r_A.pdb -> A
    m = re.search(r"_([A-Za-z0-9]+)\.pdb$", filename)
    return m.group(1) if m else None


def check_af_quality(pdb_path: str, start_res: int, end_res: int, cfg: RMSDConfig) -> Tuple[bool, float, str]:
    """AlphaFold quality check using segment mean pLDDT and optional head-tail PAE."""
    mean_plddt = 0.0

    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("X", pdb_path)

        plddt_vals = []
        for res in structure[0]["A"]:
            if start_res <= res.id[1] <= end_res:
                if "CA" in res:
                    plddt_vals.append(res["CA"].get_bfactor())

        if not plddt_vals:
            return False, 0.0, "NO_ATOMS"

        mean_plddt = float(np.mean(plddt_vals))
        if mean_plddt < cfg.min_mean_plddt:
            return False, mean_plddt, "LOW_PLDDT"
    except Exception:
        return False, 0.0, "PARSE_ERR"

    json_path = pdb_path.replace(".pdb", ".json")
    if not os.path.exists(json_path):
        base = pdb_path[:-4]
        jsons = glob.glob(f"{base}*.json")
        json_path = jsons[0] if jsons else None

    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            block = None
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                block = data[0]
            elif isinstance(data, dict):
                block = data

            if block is not None:
                pae_matrix = block.get("predicted_aligned_error", None)
                if pae_matrix is None:
                    pae_matrix = block.get("pae", None)

                if pae_matrix is not None:
                    idx_s = max(0, start_res - 1)
                    idx_e = min(len(pae_matrix) - 1, end_res - 1)
                    pae_val = float(pae_matrix[idx_s][idx_e])
                    if pae_val > cfg.max_head_tail_pae:
                        return False, mean_plddt, f"HIGH_PAE({pae_val:.1f})"
        except Exception:
            # The original analysis code treated unreadable/missing PAE JSON as non-fatal.
            pass

    return True, mean_plddt, "OK"


def find_best_structure_hybrid(
    family_dir: str,
    candidates: List[Dict[str, Any]],
    allowed_diff: int,
    cfg: RMSDConfig,
    debug: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Find the first acceptable structure, preferring experimental PDB over AlphaFold."""
    for rank, cand in enumerate(candidates):
        uid = cand["uid"]
        s, e = cand["pdb_start"], cand["pdb_end"]

        if cand["len_diff"] > allowed_diff:
            if debug:
                print(f"    [STOP] len_diff={cand['len_diff']} > {allowed_diff}, stop searching.")
            break

        exp_pattern = os.path.join(family_dir, f"{uid}_exp_*.pdb")
        exp_files = sorted(glob.glob(exp_pattern))
        if exp_files:
            best_pdb = exp_files[0]
            chain = parse_exp_chain(best_pdb) or "A"
            if debug:
                print(
                    f"    [EXP] Found {os.path.basename(best_pdb)} "
                    f"(rank={rank}, len_diff={cand['len_diff']}, seq_len={cand['seq_len']})"
                )
            return {
                "path": best_pdb,
                "type": "EXP",
                "chain": chain,
                "uid": uid,
                "start": s,
                "end": e,
                "plddt": "N/A",
                "info": "Experimental",
                "len_diff": cand["len_diff"],
                "seq_len": cand["seq_len"],
            }, None

        af_pattern = os.path.join(family_dir, f"AF-{uid}-F1-model*.pdb")
        af_files = sorted(glob.glob(af_pattern), reverse=True)
        if af_files:
            af_pdb = af_files[0]
            passed, plddt, msg = check_af_quality(af_pdb, s, e, cfg)
            if passed:
                if debug:
                    print(
                        f"    [AF ] Found {os.path.basename(af_pdb)} "
                        f"(rank={rank}, len_diff={cand['len_diff']}, seq_len={cand['seq_len']}, pLDDT={plddt:.1f})"
                    )
                return {
                    "path": af_pdb,
                    "type": "AF",
                    "chain": "A",
                    "uid": uid,
                    "start": s,
                    "end": e,
                    "plddt": round(float(plddt), 2),
                    "info": msg,
                    "len_diff": cand["len_diff"],
                    "seq_len": cand["seq_len"],
                }, None
            if debug:
                print(f"    [AF ] Skip {uid}: {msg} (len_diff={cand['len_diff']}, seq_len={cand['seq_len']})")

    return None, "NO_VALID_STRUCT_FOUND_WITHIN_ALLOWED_DIFF"


def resolve_chain(obj: str, expected_chain: str) -> str:
    chains = cmd.get_chains(obj)
    if expected_chain in chains:
        return expected_chain
    if expected_chain and expected_chain.upper() in chains:
        return expected_chain.upper()
    if len(chains) == 1:
        return chains[0]
    return expected_chain


class StructureContext:
    """Small cache wrapper for family-level HHM maps and alignment paths."""

    def __init__(self, cfg: RMSDConfig):
        self.cfg = cfg
        self.hhm_map_cache: Dict[str, Tuple[Dict[int, int], str]] = {}
        self.aln_path_cache: Dict[str, Optional[str]] = {}

    def get_hhm_map(self, fam: str) -> Tuple[Dict[int, int], str]:
        if fam in self.hhm_map_cache:
            return self.hhm_map_cache[fam]
        hhm_path = os.path.join(self.cfg.pfam_dir, fam, f"{fam}.hhm")
        res = extract_hmm_to_msa_map(hhm_path)
        self.hhm_map_cache[fam] = res
        return res

    def get_alignment_path(self, fam: str) -> Optional[str]:
        if fam in self.aln_path_cache:
            return self.aln_path_cache[fam]

        p1 = os.path.join(self.cfg.pfam_dir, fam, f"{fam}.fas")
        p2 = os.path.join(self.cfg.pfam_dir, fam, f"{fam}.sto")

        if os.path.exists(p1):
            self.aln_path_cache[fam] = p1
        elif os.path.exists(p2):
            self.aln_path_cache[fam] = p2
        else:
            self.aln_path_cache[fam] = None
        return self.aln_path_cache[fam]


def compute_rmsd_for_hmm_segments(
    fam_a: str,
    fam_b: str,
    a1: int,
    a2: int,
    b1: int,
    b2: int,
    ctx: StructureContext,
    failure_log: Dict[str, int],
    label: str,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run the full HMM-segment -> structure -> PyMOL RMSD path for one pair."""
    cfg = ctx.cfg
    hmm_len_a = int(a2) - int(a1) + 1
    hmm_len_b = int(b2) - int(b1) + 1

    allowed_diff_a = cfg.allowed_len_diff(hmm_len_a)
    allowed_diff_b = cfg.allowed_len_diff(hmm_len_b)

    def fail(reason: str) -> None:
        key = f"{label}:{reason}"
        failure_log[key] = failure_log.get(key, 0) + 1

    map_a, st_a = ctx.get_hhm_map(fam_a)
    map_b, st_b = ctx.get_hhm_map(fam_b)
    if not map_a or not map_b:
        fail(f"HMM_MAP_FAIL:{st_a}|{st_b}")
        return None

    msa_a_s, msa_a_e = hmm_range_to_msa_range(map_a, int(a1), int(a2))
    msa_b_s, msa_b_e = hmm_range_to_msa_range(map_b, int(b1), int(b2))
    if msa_a_s is None or msa_a_e is None or msa_b_s is None or msa_b_e is None:
        fail("HMM_RANGE_TO_MSA_FAIL")
        return None

    aln_a = ctx.get_alignment_path(fam_a)
    aln_b = ctx.get_alignment_path(fam_b)
    if aln_a is None or aln_b is None:
        fail("ALIGN_FILE_MISSING")
        return None

    list_a, st_list_a = get_candidates_from_alignment(aln_a, msa_a_s, msa_a_e, hmm_len_a, cfg)
    list_b, st_list_b = get_candidates_from_alignment(aln_b, msa_b_s, msa_b_e, hmm_len_b, cfg)
    if not list_a or not list_b:
        fail(f"NO_CANDIDATES:{st_list_a}|{st_list_b}")
        return None

    struct_a, fail_a = find_best_structure_hybrid(
        os.path.join(cfg.pfam_dir, fam_a), list_a, allowed_diff_a, cfg, debug
    )
    if not struct_a:
        fail(f"FAIL_A:{fail_a}")
        return None

    struct_b, fail_b = find_best_structure_hybrid(
        os.path.join(cfg.pfam_dir, fam_b), list_b, allowed_diff_b, cfg, debug
    )
    if not struct_b:
        fail(f"FAIL_B:{fail_b}")
        return None

    cmd.reinitialize()
    cmd.load(struct_a["path"], "obj_a")
    cmd.load(struct_b["path"], "obj_b")

    real_chain_a = resolve_chain("obj_a", struct_a["chain"])
    real_chain_b = resolve_chain("obj_b", struct_b["chain"])

    sel_a = f"obj_a and chain {real_chain_a} and resi {struct_a['start']}-{struct_a['end']} and name CA"
    sel_b = f"obj_b and chain {real_chain_b} and resi {struct_b['start']}-{struct_b['end']} and name CA"

    cmd.select("sel_a", sel_a)
    cmd.select("sel_b", sel_b)

    if cmd.count_atoms("sel_a") < 3 or cmd.count_atoms("sel_b") < 3:
        fail("ATOM_COUNT_LOW")
        return None

    aln = cmd.super("sel_a", "sel_b", object="aln_obj")
    rmsd = float(aln[0])
    aligned_cnt = int(aln[1])

    aligned_cov_a = aligned_cnt / hmm_len_a if hmm_len_a > 0 else 0.0
    aligned_cov_b = aligned_cnt / hmm_len_b if hmm_len_b > 0 else 0.0
    min_aligned_cov = min(aligned_cov_a, aligned_cov_b)

    if cfg.min_aligned_coverage > 0 and min_aligned_cov < cfg.min_aligned_coverage:
        fail(f"LOW_ALIGNED_COVERAGE<{cfg.min_aligned_coverage}")
        return None

    if debug:
        print(f"  [Success] RMSD={rmsd:.3f}, Atoms={aligned_cnt}")
        print(f"            Src A/B: {struct_a['type']} / {struct_b['type']}")
        print(f"            HMM Len A/B: {hmm_len_a} / {hmm_len_b}")
        print(f"            AllowedDiff A/B: {allowed_diff_a} / {allowed_diff_b}")
        print(f"            LenDiff A/B: {struct_a['len_diff']} / {struct_b['len_diff']}")

    return {
        "RMSD": rmsd,
        "Aligned_Atoms": aligned_cnt,
        "Aligned_Coverage_A": aligned_cov_a,
        "Aligned_Coverage_B": aligned_cov_b,
        "Min_Aligned_Coverage": min_aligned_cov,

        "Source_A": struct_a["type"],
        "UID_A": struct_a["uid"],
        "PDB_A": os.path.basename(struct_a["path"]),
        "Range_A": f"{struct_a['start']}-{struct_a['end']}",
        "pLDDT_A": struct_a["plddt"],
        "LenDiff_A": struct_a["len_diff"],
        "SeqLen_A": struct_a["seq_len"],

        "Source_B": struct_b["type"],
        "UID_B": struct_b["uid"],
        "PDB_B": os.path.basename(struct_b["path"]),
        "Range_B": f"{struct_b['start']}-{struct_b['end']}",
        "pLDDT_B": struct_b["plddt"],
        "LenDiff_B": struct_b["len_diff"],
        "SeqLen_B": struct_b["seq_len"],

        "HMM_Len_A": hmm_len_a,
        "HMM_Len_B": hmm_len_b,
        "AllowedDiff_A": allowed_diff_a,
        "AllowedDiff_B": allowed_diff_b,
        "Len_Diff_Mode": cfg.len_diff_mode,
        "Max_Len_Diff": cfg.max_len_diff,
        "Min_Len_Diff": cfg.min_len_diff,
        "Rel_Len_Diff_Ratio": cfg.rel_len_diff_ratio,
    }


def flush_rows(rows: List[Dict[str, Any]], out_csv: str, wrote_header: bool) -> bool:
    if not rows:
        return wrote_header
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not wrote_header, index=False)
    rows.clear()
    return True


def print_failure_summary(failure_log: Dict[str, int], top_n: int = 20) -> None:
    if not failure_log:
        return
    print("Top failures:")
    for k, v in sorted(failure_log.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"  {k}: {v}")
