#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map ProDive Pfam–Pfam fragments to residue contacts in ``3did_flat``.

Each HMM segment is projected through the local Pfam seed alignment onto an
experimental PDB chain. The mapped residues are compared with 3did contact
positions, classified as ``NonInterface``, ``InterfacePartial``, or
``InterfaceMajor``, and evaluated against same-chain, same-length random
windows. Row-level, fragment-side, and summary outputs are written.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import glob
import random
import hashlib
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1


# =============================================================================
# Global worker state
# =============================================================================
G_ARGS = None
G_COMPLEX_INDEX: Dict[Tuple[str, str, str], Set[int]] = {}
G_COMPLEX_BY_PFAM: Dict[str, Set[Tuple[str, str]]] = {}
G_COMPLEX_META: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

_HHM_MAP_CACHE: Dict[str, Dict[int, int]] = {}
_ALIGNMENT_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_PDB_CHAIN_CACHE: OrderedDict[Tuple[str, str], Optional[Dict[str, Any]]] = OrderedDict()
_MAX_PDB_CACHE_SIZE = 256


# =============================================================================
# Data classes
# =============================================================================
@dataclass(frozen=True)
class StructureMeta:
    uid: str
    pdb_id: str
    chain_id: str
    path: str


# =============================================================================
# Generic helpers
# =============================================================================
def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def open_maybe_gzip(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return open(path, "r", errors="ignore")


def parse_family_id(value: Any) -> Optional[str]:
    m = re.search(r"PF\d{5}", str(value))
    return m.group(0) if m else None


def parse_int_range(text: Any) -> Optional[Tuple[int, int]]:
    m = re.search(r"(\d+)\s*-\s*(\d+)", str(text))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if a > b:
        a, b = b, a
    return a, b


def parse_segment_pair_from_row(row: Dict[str, Any]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return ((main_start, main_end), (sub_start, sub_end))."""
    for col in ["Sub_Segments_Details", "Segments_Details", "Segment_Details", "Details"]:
        if col in row and pd.notna(row[col]):
            m = re.search(r"(\d+)\s*-\s*(\d+)\s*[-=]*>\s*(\d+)\s*-\s*(\d+)", str(row[col]))
            if m:
                a1, a2, b1, b2 = map(int, m.groups())
                if a1 > a2:
                    a1, a2 = a2, a1
                if b1 > b2:
                    b1, b2 = b2, b1
                return (a1, a2), (b1, b2)

    main_range = None
    sub_range = None
    for col in ["Main_Segment", "Main_Seg", "Query_Segment", "Query_Seg"]:
        if col in row and pd.notna(row[col]):
            main_range = parse_int_range(row[col])
            if main_range:
                break
    for col in ["Sub_Segment", "Sub_Seg", "Target_Segment", "Target_Seg"]:
        if col in row and pd.notna(row[col]):
            sub_range = parse_int_range(row[col])
            if sub_range:
                break

    if main_range is None:
        for a, b in [("Main_Start", "Main_End"), ("Query_Start", "Query_End"), ("A_Start", "A_End")]:
            if a in row and b in row and pd.notna(row[a]) and pd.notna(row[b]):
                x, y = int(row[a]), int(row[b])
                main_range = (min(x, y), max(x, y))
                break

    if sub_range is None:
        for a, b in [("Sub_Start", "Sub_End"), ("Target_Start", "Target_End"), ("B_Start", "B_End")]:
            if a in row and b in row and pd.notna(row[a]) and pd.notna(row[b]):
                x, y = int(row[a]), int(row[b])
                sub_range = (min(x, y), max(x, y))
                break

    if main_range and sub_range:
        return main_range, sub_range
    return None


def safe_three_to_one(resname: str) -> str:
    r = resname.strip().upper()
    if r == "MSE":
        return "M"
    if r == "SEC":
        return "U"
    if r == "PYL":
        return "O"
    try:
        aa = seq1(r)
        if aa and len(aa) == 1:
            return aa
    except Exception:
        pass
    return "X"


# =============================================================================
# Known complex interface parsing
# =============================================================================
def classify_complex_pair_type(pfam_a: Optional[str], pfam_b: Optional[str], chain_a: Optional[str], chain_b: Optional[str]) -> Tuple[str, str, str]:
    """Return (pair_type, pfam_relation, chain_relation)."""
    pa = parse_family_id(pfam_a) if pfam_a else None
    pb = parse_family_id(pfam_b) if pfam_b else None
    chain_relation = "unknown_chain_relation"
    if chain_a is not None and chain_b is not None:
        chain_relation = "intrachain" if str(chain_a).upper() == str(chain_b).upper() else "interchain"

    pfam_relation = "unknown_pfam_relation"
    if pa and pb:
        pfam_relation = "same_pfam" if pa.upper() == pb.upper() else "different_pfam"

    if chain_relation == "intrachain":
        pair_type = "intrachain_domain_contact"
    elif chain_relation == "interchain" and pfam_relation == "different_pfam":
        pair_type = "interchain_different_pfam"
    elif chain_relation == "interchain" and pfam_relation == "same_pfam":
        pair_type = "interchain_same_pfam"
    elif chain_relation == "interchain":
        pair_type = "interchain_unknown_pfam"
    else:
        pair_type = "unknown_complex_pair_type"
    return pair_type, pfam_relation, chain_relation


def summarize_different_protein_proxy(pair_types: Set[str]) -> str:
    """
    Conservative reporting label.

    This is not a perfect same-protein/different-protein call because 3did_flat does not
    provide polymer entity IDs or UniProt IDs. interchain_different_pfam is used as a
    high-confidence proxy for a different-domain / likely different-protein interface.
    """
    if not pair_types:
        return "UNKNOWN"
    has_diff = "interchain_different_pfam" in pair_types
    has_same = "interchain_same_pfam" in pair_types
    has_intra = "intrachain_domain_contact" in pair_types
    if has_diff and (has_same or has_intra or len(pair_types) > 1):
        return "MIXED_WITH_DIFFERENT_PFAM_INTERCHAIN"
    if has_diff:
        return "YES_DIFFERENT_PFAM_INTERCHAIN"
    if has_same and (has_intra or len(pair_types) > 1):
        return "MIXED_SAME_PFAM_OR_INTRACHAIN"
    if has_same:
        return "UNCERTAIN_SAME_PFAM_INTERCHAIN"
    if has_intra:
        return "NO_INTRACHAIN_DOMAIN_CONTACT"
    return "UNKNOWN"


def add_complex_side(
    index: Dict[Tuple[str, str, str], Set[int]],
    by_pfam: Dict[str, Set[Tuple[str, str]]],
    meta_index: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    pfam_id: Optional[str],
    pdb_id: Optional[str],
    chain_id: Optional[str],
    residues: Iterable[int],
    partner_pfam: Optional[str] = None,
    partner_chain: Optional[str] = None,
    source: str = "unknown",
    instance_id: str = "",
    pair_type: Optional[str] = None,
    pfam_relation: Optional[str] = None,
    chain_relation: Optional[str] = None,
) -> None:
    if not pfam_id or not pdb_id or not chain_id:
        return
    residues = set(int(x) for x in residues if x is not None)
    if not residues:
        return
    pfam_clean = parse_family_id(pfam_id)
    partner_clean = parse_family_id(partner_pfam) if partner_pfam else None
    if not pfam_clean:
        return
    key = (pfam_clean.upper(), pdb_id.upper(), str(chain_id))
    index[key].update(residues)
    by_pfam[pfam_clean.upper()].add((pdb_id.upper(), str(chain_id)))

    if pair_type is None or pfam_relation is None or chain_relation is None:
        pair_type, pfam_relation, chain_relation = classify_complex_pair_type(pfam_clean, partner_clean, chain_id, partner_chain)

    meta_index[key].append(
        {
            "source": source,
            "instance_id": instance_id,
            "pfam": pfam_clean.upper(),
            "partner_pfam": partner_clean.upper() if partner_clean else "",
            "pdb_id": pdb_id.upper(),
            "chain": str(chain_id),
            "partner_chain": str(partner_chain) if partner_chain is not None else "",
            "pair_type": pair_type,
            "pfam_relation": pfam_relation,
            "chain_relation": chain_relation,
            "n_interface_residues": len(residues),
        }
    )


def parse_3did_flat(
    path: str,
) -> Tuple[Dict[Tuple[str, str, str], Set[int]], Dict[str, Set[Tuple[str, str]]], Dict[Tuple[str, str, str], List[Dict[str, Any]]]]:
    """
    Parse 3did_flat.gz.

    3did_flat contact positions are original PDB residue numbers. This parser also
    preserves the partner Pfam/chain and a conservative pair type:
      - interchain_different_pfam
      - interchain_same_pfam
      - intrachain_domain_contact
    """
    index: Dict[Tuple[str, str, str], Set[int]] = defaultdict(set)
    by_pfam: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    meta_index: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    current_pfams: Optional[Tuple[str, str]] = None
    current_domains: Optional[Tuple[str, str]] = None
    current_instance: Optional[Dict[str, Any]] = None

    def flush_instance() -> None:
        nonlocal current_instance, current_pfams, current_domains
        if current_instance is None or current_pfams is None:
            current_instance = None
            return
        pfam_a, pfam_b = current_pfams
        chain_a = current_instance.get("chain_a")
        chain_b = current_instance.get("chain_b")
        pair_type, pfam_relation, chain_relation = classify_complex_pair_type(pfam_a, pfam_b, chain_a, chain_b)
        instance_id = f"3did:{current_instance.get('pdb_id')}:{chain_a}-{chain_b}:{current_instance.get('range_a','')}:{current_instance.get('range_b','')}"
        add_complex_side(
            index,
            by_pfam,
            meta_index,
            pfam_a,
            current_instance.get("pdb_id"),
            chain_a,
            current_instance.get("res_a", set()),
            partner_pfam=pfam_b,
            partner_chain=chain_b,
            source="3did_flat",
            instance_id=instance_id,
            pair_type=pair_type,
            pfam_relation=pfam_relation,
            chain_relation=chain_relation,
        )
        add_complex_side(
            index,
            by_pfam,
            meta_index,
            pfam_b,
            current_instance.get("pdb_id"),
            chain_b,
            current_instance.get("res_b", set()),
            partner_pfam=pfam_a,
            partner_chain=chain_a,
            source="3did_flat",
            instance_id=instance_id,
            pair_type=pair_type,
            pfam_relation=pfam_relation,
            chain_relation=chain_relation,
        )
        current_instance = None

    n_id = 0
    n_3d = 0
    n_contact = 0

    with open_maybe_gzip(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("//"):
                flush_instance()
                current_pfams = None
                current_domains = None
                continue
            if line.startswith("#=ID"):
                flush_instance()
                pfams = re.findall(r"PF\d{5}", line)
                parts = line.split()
                if len(pfams) >= 2:
                    current_pfams = (pfams[0].upper(), pfams[1].upper())
                    # Best-effort domain-name capture; not used for classification.
                    current_domains = (parts[1], parts[2]) if len(parts) >= 3 else ("", "")
                    n_id += 1
                else:
                    current_pfams = None
                    current_domains = None
                continue
            if line.startswith("#=3D"):
                flush_instance()
                if current_pfams is None:
                    continue
                body = re.sub(r"^#=3D\s*", "", line)
                parts = body.split()
                if len(parts) < 3:
                    continue
                pdb_id = parts[0].upper()

                m1 = re.match(r"([^:\s]+):\(?(-?\d+)\s*-\s*(-?\d+)\)?", parts[1])
                m2 = re.match(r"([^:\s]+):\(?(-?\d+)\s*-\s*(-?\d+)\)?", parts[2])
                if not (m1 and m2):
                    m = re.search(
                        r"(\w{4})\s+([^:\s]+):\(?(-?\d+)\s*-\s*(-?\d+)\)?\s+([^:\s]+):\(?(-?\d+)\s*-\s*(-?\d+)\)?",
                        body,
                    )
                    if not m:
                        continue
                    pdb_id = m.group(1).upper()
                    chain_a = m.group(2)
                    a1, a2 = m.group(3), m.group(4)
                    chain_b = m.group(5)
                    b1, b2 = m.group(6), m.group(7)
                else:
                    chain_a = m1.group(1)
                    a1, a2 = m1.group(2), m1.group(3)
                    chain_b = m2.group(1)
                    b1, b2 = m2.group(2), m2.group(3)

                current_instance = {
                    "pdb_id": pdb_id,
                    "chain_a": chain_a,
                    "chain_b": chain_b,
                    "range_a": f"{a1}-{a2}",
                    "range_b": f"{b1}-{b2}",
                    "res_a": set(),
                    "res_b": set(),
                }
                n_3d += 1
                continue

            if line.startswith("#"):
                continue

            if current_instance is not None:
                ints = re.findall(r"(?<![A-Za-z])(-?\d+)(?![A-Za-z])", line)
                if len(ints) >= 2:
                    current_instance["res_a"].add(int(ints[0]))
                    current_instance["res_b"].add(int(ints[1]))
                    n_contact += 1

    flush_instance()
    eprint(
        f"[3did] parsed IDs={n_id:,}, structural instances={n_3d:,}, contact lines={n_contact:,}, "
        f"indexed sides={len(index):,}, Pfams={len(by_pfam):,}"
    )
    return dict(index), dict(by_pfam), dict(meta_index)


# =============================================================================
# Local HHM/MSA mapping
# =============================================================================
def extract_hhm_to_msa_map(hhm_path: str) -> Dict[int, int]:
    """
    Extract local HHsuite HHM state -> MSA column mapping.

    This follows the format used in the current ProDive pipeline: HHM match-state lines
    contain the HMM index in field 2 and the MSA column index as the final field.
    """
    if hhm_path in _HHM_MAP_CACHE:
        return _HHM_MAP_CACHE[hhm_path]

    mapping: Dict[int, int] = {}
    if not os.path.exists(hhm_path):
        _HHM_MAP_CACHE[hhm_path] = mapping
        return mapping

    try:
        with open(hhm_path, "r", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 10 and parts[1].isdigit():
                    try:
                        hmm_idx = int(parts[1])
                        msa_col = parts[-1]
                        if msa_col.isdigit():
                            mapping[hmm_idx] = int(msa_col)
                    except Exception:
                        continue
    except Exception:
        mapping = {}

    _HHM_MAP_CACHE[hhm_path] = mapping
    return mapping


def find_alignment_file(family_dir: str, pfam_id: str) -> Optional[str]:
    for ext in ["fas", "sto", "afa", "fasta", "aln"]:
        p = os.path.join(family_dir, f"{pfam_id}.{ext}")
        if os.path.exists(p):
            return p
    for pattern in ["*.fas", "*.sto", "*.afa", "*.fasta", "*.aln"]:
        hits = sorted(glob.glob(os.path.join(family_dir, pattern)))
        if hits:
            return hits[0]
    return None


def read_alignment_records(aln_path: str) -> List[Dict[str, Any]]:
    if aln_path in _ALIGNMENT_CACHE:
        return _ALIGNMENT_CACHE[aln_path]

    seqs: Dict[str, str] = OrderedDict()
    current = None

    try:
        with open(aln_path, "r", errors="ignore") as f:
            first = f.readline()
            f.seek(0)
            is_fasta = first.startswith(">")
            if is_fasta:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        current = line[1:].strip()
                        seqs[current] = ""
                    elif current is not None:
                        seqs[current] += line
            else:
                # Minimal Stockholm/A2M style parser: concatenate non-comment two-column sequence lines.
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or line == "//":
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    sid, frag = parts[0], parts[1]
                    seqs[sid] = seqs.get(sid, "") + frag
    except Exception:
        _ALIGNMENT_CACHE[aln_path] = []
        return []

    records: List[Dict[str, Any]] = []
    for header, gapped in seqs.items():
        m_range = re.search(r"/(\d+)-(\d+)", header)
        if not m_range:
            # Cannot map to absolute FASTA coordinates without /start-end.
            continue
        global_start = int(m_range.group(1))
        global_end = int(m_range.group(2))
        ungapped = gapped.replace("-", "").replace(".", "")
        if len(ungapped) < 3:
            continue

        uid_match = re.search(
            r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})",
            header,
        )
        uid = uid_match.group(1) if uid_match else header.split("/")[0].split()[0]

        records.append(
            {
                "header": header,
                "uid": uid,
                "gapped_seq": gapped,
                "ungapped_seq": ungapped,
                "global_start": global_start,
                "global_end": global_end,
            }
        )

    _ALIGNMENT_CACHE[aln_path] = records
    return records


def get_seed_candidates_for_hmm_segment(
    aln_path: str,
    msa_start_col: int,
    msa_end_col: int,
    hmm_len: int,
    min_seq_ratio: float,
    max_seq_ratio: float,
    max_search_depth: int,
) -> List[Dict[str, Any]]:
    records = read_alignment_records(aln_path)
    if not records:
        return []

    idx_start = msa_start_col - 1
    idx_end = msa_end_col
    out: List[Dict[str, Any]] = []

    for rec in records:
        gapped = rec["gapped_seq"]
        if len(gapped) < idx_end:
            continue
        raw_frag = gapped[idx_start:idx_end]
        pure_frag = raw_frag.replace("-", "").replace(".", "")
        seq_len = len(pure_frag)
        if seq_len <= 0:
            continue
        ratio = seq_len / float(hmm_len) if hmm_len > 0 else 0.0
        if ratio < min_seq_ratio or ratio > max_seq_ratio:
            continue

        prefix = gapped[:idx_start]
        before = len(prefix.replace("-", "").replace(".", ""))
        seq_start = int(rec["global_start"]) + before
        seq_end = seq_start + seq_len - 1
        gaps = raw_frag.count("-") + raw_frag.count(".")

        out.append(
            {
                "uid": rec["uid"],
                "header": rec["header"],
                "full_seq": rec["ungapped_seq"],
                "global_start": rec["global_start"],
                "global_end": rec["global_end"],
                "seq_start": seq_start,
                "seq_end": seq_end,
                "seq_len": seq_len,
                "seq_ratio_vs_hmm": ratio,
                "gaps": gaps,
                "len_diff": abs(seq_len - hmm_len),
            }
        )

    out.sort(key=lambda x: (x["len_diff"], x["gaps"], abs(x["seq_ratio_vs_hmm"] - 1.0)))
    return out[:max_search_depth]


# =============================================================================
# Local structure parsing and seed-to-PDB mapping
# =============================================================================
def parse_exp_structure_filename(path: str) -> Optional[StructureMeta]:
    base = os.path.basename(path)
    m = re.match(r"(.+?)_exp_([A-Za-z0-9]{4})_([A-Za-z0-9]+)\.pdb$", base)
    if not m:
        return None
    return StructureMeta(uid=m.group(1), pdb_id=m.group(2).upper(), chain_id=m.group(3), path=path)


def find_matching_structure_files(
    family_dir: str,
    candidate_uid: str,
    allowed_pdb_chains: Set[Tuple[str, str]],
) -> List[StructureMeta]:
    metas: List[StructureMeta] = []
    if not os.path.isdir(family_dir):
        return metas
    for path in sorted(glob.glob(os.path.join(family_dir, f"{candidate_uid}_exp_*.pdb"))):
        meta = parse_exp_structure_filename(path)
        if meta is None:
            continue
        if (meta.pdb_id.upper(), meta.chain_id) in allowed_pdb_chains or (meta.pdb_id.upper(), meta.chain_id.upper()) in allowed_pdb_chains:
            metas.append(meta)
    return metas


def get_pdb_chain_sequence_and_numbers(pdb_path: str, chain_id: str) -> Optional[Dict[str, Any]]:
    key = (pdb_path, chain_id)
    if key in _PDB_CHAIN_CACHE:
        _PDB_CHAIN_CACHE.move_to_end(key)
        return _PDB_CHAIN_CACHE[key]

    result = None
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("x", pdb_path)
        model = next(structure.get_models())

        chain = None
        if chain_id in model:
            chain = model[chain_id]
        else:
            for ch in model:
                if str(ch.id).upper() == str(chain_id).upper():
                    chain = ch
                    break
        if chain is not None:
            seq_chars: List[str] = []
            pdb_nums: List[int] = []
            pdb_keys: List[Tuple[int, str]] = []
            for res in chain:
                if not is_aa(res, standard=False):
                    continue
                if "CA" not in res:
                    continue
                hetflag, resseq, icode = res.id
                seq_chars.append(safe_three_to_one(res.resname))
                pdb_nums.append(int(resseq))
                pdb_keys.append((int(resseq), str(icode).strip()))
            if len(seq_chars) >= 3:
                result = {
                    "seq": "".join(seq_chars),
                    "pdb_nums": pdb_nums,
                    "pdb_keys": pdb_keys,
                    "actual_chain_id": str(chain.id),
                }
    except Exception:
        result = None

    _PDB_CHAIN_CACHE[key] = result
    if len(_PDB_CHAIN_CACHE) > _MAX_PDB_CACHE_SIZE:
        _PDB_CHAIN_CACHE.popitem(last=False)
    return result


def build_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    return aligner


def map_seed_abs_positions_to_pdb_numbers(
    seed_full_seq: str,
    seed_abs_start: int,
    pdb_seq: str,
    pdb_nums: List[int],
) -> Dict[int, int]:
    """
    Return mapping: seed absolute residue position -> PDB residue number.

    seed_full_seq corresponds to absolute positions seed_abs_start ... seed_abs_start+len-1.
    pdb_seq corresponds to pdb_nums by sequence index.
    """
    if not seed_full_seq or not pdb_seq:
        return {}
    try:
        aligner = build_aligner()
        alns = aligner.align(seed_full_seq, pdb_seq)
        if len(alns) == 0:
            return {}
        aln = alns[0]
    except Exception:
        return {}

    mapping: Dict[int, int] = {}
    q_blocks = aln.aligned[0]
    s_blocks = aln.aligned[1]
    for (q0, q1), (s0, s1) in zip(q_blocks, s_blocks):
        L = min(q1 - q0, s1 - s0)
        for k in range(L):
            qi = q0 + k
            si = s0 + k
            if 0 <= si < len(pdb_nums):
                seed_pos = seed_abs_start + qi
                mapping[seed_pos] = int(pdb_nums[si])
    return mapping


# =============================================================================
# Fragment-side evaluation
# =============================================================================
def default_side_result(status: str, note: str = "") -> Dict[str, Any]:
    return {
        "Status": status,
        "Note": note,
        "Family": "",
        "HMM_Start": np.nan,
        "HMM_End": np.nan,
        "HMM_Len": np.nan,
        "MSA_Start": np.nan,
        "MSA_End": np.nan,
        "Seq_UID": "",
        "Seq_Header": "",
        "Seq_Start": np.nan,
        "Seq_End": np.nan,
        "Seq_Len": np.nan,
        "SeqLen_vs_HMMLen": np.nan,
        "PDB_IDs": "",
        "Chains": "",
        "Complex_Instance_Count": 0,
        "Struct_Coverage_vs_HMM": np.nan,
        "Mapped_Fragment_Residues": 0,
        "Interface_Count": 0,
        "Interface_Fraction": np.nan,
        "Interface_Class": "NA",
        "Complex_Partner_Pfams": "",
        "Complex_Partner_Chains": "",
        "Complex_Pair_Types": "",
        "Different_Protein_Proxy": "UNKNOWN",
        "Has_Interchain_Different_Pfam": False,
        "Has_Interchain_Same_Pfam": False,
        "Has_Intrachain_Domain_Contact": False,
        "Random_N": 0,
        "Random_Mean_Interface_Fraction": np.nan,
        "Random_Median_Interface_Fraction": np.nan,
        "Random_P_GE": np.nan,
        "Random_Z": np.nan,
        "Random_Enrichment": np.nan,
    }


def classify_interface(interface_fraction: float, th_major: float, th_partial: float) -> str:
    if pd.isna(interface_fraction):
        return "NA"
    if interface_fraction >= th_major:
        return "InterfaceMajor"
    if interface_fraction >= th_partial:
        return "InterfacePartial"
    return "NonInterface"


def stable_random_seed(*parts: Any) -> int:
    text = "|".join(str(x) for x in parts)
    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    base_seed = int(getattr(G_ARGS, "random_seed", 20260601))
    return (int(digest, 16) + base_seed) % (2 ** 32)


def get_complex_metadata_for_key(key_exact: Tuple[str, str, str], key_upper: Tuple[str, str, str]) -> List[Dict[str, Any]]:
    metas = G_COMPLEX_META.get(key_exact)
    if metas:
        return metas
    return G_COMPLEX_META.get(key_upper, [])


def random_control_same_chain(
    used_maps: List[Dict[str, Any]],
    hmm_len: int,
    window_len: int,
    real_interface_fraction: float,
    min_struct_coverage: float,
    seed_parts: Tuple[Any, ...],
) -> Dict[str, Any]:
    args = G_ARGS
    random_n = int(getattr(args, "random_n", 0))
    if random_n <= 0 or not used_maps or window_len <= 0:
        return {
            "random_n": 0,
            "random_mean": np.nan,
            "random_median": np.nan,
            "random_p_ge": np.nan,
            "random_z": np.nan,
            "random_enrichment": np.nan,
        }

    rng = random.Random(stable_random_seed(*seed_parts))
    max_attempts = max(random_n * int(getattr(args, "random_max_attempts_per_sample", 50)), random_n)
    vals: List[float] = []
    attempts = 0

    eligible_maps = [m for m in used_maps if int(m["full_seq_len"]) >= window_len]
    if not eligible_maps:
        return {
            "random_n": 0,
            "random_mean": np.nan,
            "random_median": np.nan,
            "random_p_ge": np.nan,
            "random_z": np.nan,
            "random_enrichment": np.nan,
        }

    while len(vals) < random_n and attempts < max_attempts:
        attempts += 1
        um = rng.choice(eligible_maps)
        full_len = int(um["full_seq_len"])
        q0 = rng.randint(0, full_len - window_len)
        start_abs = int(um["seed_abs_start"]) + q0
        positions = range(start_abs, start_abs + window_len)
        seed_to_pdb = um["seed_to_pdb"]
        interface_pdb_nums = um["interface_pdb_nums"]
        mapped = [p for p in positions if p in seed_to_pdb]
        if not mapped:
            continue
        struct_cov = len(mapped) / float(hmm_len) if hmm_len > 0 else 0.0
        if struct_cov < min_struct_coverage:
            continue
        n_interface = sum(1 for p in mapped if seed_to_pdb[p] in interface_pdb_nums)
        vals.append(n_interface / float(len(mapped)))

    if not vals:
        return {
            "random_n": 0,
            "random_mean": np.nan,
            "random_median": np.nan,
            "random_p_ge": np.nan,
            "random_z": np.nan,
            "random_enrichment": np.nan,
        }

    arr = np.asarray(vals, dtype=float)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    real = float(real_interface_fraction) if not pd.isna(real_interface_fraction) else np.nan
    p_ge = float((np.sum(arr >= real) + 1) / (len(arr) + 1)) if not pd.isna(real) else np.nan
    z = float((real - mean) / std) if std > 0 and not pd.isna(real) else np.nan
    enrichment = float(real / mean) if mean > 0 and not pd.isna(real) else np.nan
    return {
        "random_n": int(len(arr)),
        "random_mean": mean,
        "random_median": median,
        "random_p_ge": p_ge,
        "random_z": z,
        "random_enrichment": enrichment,
    }


def evaluate_candidate_against_complexes(
    pfam_id: str,
    candidate: Dict[str, Any],
    hmm_len: int,
    family_structure_dir: str,
    allowed_pdb_chains: Set[Tuple[str, str]],
    min_struct_coverage: float,
) -> Optional[Dict[str, Any]]:
    metas = find_matching_structure_files(family_structure_dir, candidate["uid"], allowed_pdb_chains)
    if not metas:
        return None

    frag_positions = set(range(int(candidate["seq_start"]), int(candidate["seq_end"]) + 1))
    union_mapped_frag_positions: Set[int] = set()
    union_interface_frag_positions: Set[int] = set()
    used_pdb_ids: Set[str] = set()
    used_chains: Set[str] = set()
    used_instances = 0
    used_maps: List[Dict[str, Any]] = []

    partner_pfams: Set[str] = set()
    partner_chains: Set[str] = set()
    pair_types: Set[str] = set()

    for meta in metas:
        key_exact = (pfam_id.upper(), meta.pdb_id.upper(), meta.chain_id)
        key_upper = (pfam_id.upper(), meta.pdb_id.upper(), meta.chain_id.upper())
        interface_pdb_nums = G_COMPLEX_INDEX.get(key_exact) or G_COMPLEX_INDEX.get(key_upper)
        if not interface_pdb_nums:
            continue

        meta_records = get_complex_metadata_for_key(key_exact, key_upper)
        for mr in meta_records:
            if mr.get("partner_pfam"):
                partner_pfams.add(str(mr.get("partner_pfam")))
            if mr.get("partner_chain"):
                partner_chains.add(str(mr.get("partner_chain")))
            if mr.get("pair_type"):
                pair_types.add(str(mr.get("pair_type")))

        chain_data = get_pdb_chain_sequence_and_numbers(meta.path, meta.chain_id)
        if chain_data is None:
            continue

        seed_to_pdb = map_seed_abs_positions_to_pdb_numbers(
            candidate["full_seq"],
            int(candidate["global_start"]),
            chain_data["seq"],
            chain_data["pdb_nums"],
        )
        if not seed_to_pdb:
            continue

        mapped_here = {p for p in frag_positions if p in seed_to_pdb}
        if not mapped_here:
            continue
        interface_here = {p for p in mapped_here if seed_to_pdb.get(p) in interface_pdb_nums}

        union_mapped_frag_positions.update(mapped_here)
        union_interface_frag_positions.update(interface_here)
        used_pdb_ids.add(meta.pdb_id)
        used_chains.add(meta.chain_id)
        used_instances += 1
        used_maps.append(
            {
                "pdb_id": meta.pdb_id,
                "chain_id": meta.chain_id,
                "seed_to_pdb": seed_to_pdb,
                "interface_pdb_nums": set(interface_pdb_nums),
                "seed_abs_start": int(candidate["global_start"]),
                "full_seq_len": len(candidate["full_seq"]),
            }
        )

    if used_instances == 0:
        return None

    struct_cov = len(union_mapped_frag_positions) / float(hmm_len) if hmm_len > 0 else 0.0
    if struct_cov < min_struct_coverage:
        status = "LOW_STRUCTURE_COVERAGE"
    else:
        status = "OK"

    denom = len(union_mapped_frag_positions)
    interface_fraction = len(union_interface_frag_positions) / float(denom) if denom > 0 else np.nan

    rand = random_control_same_chain(
        used_maps=used_maps,
        hmm_len=hmm_len,
        window_len=int(candidate["seq_len"]),
        real_interface_fraction=interface_fraction,
        min_struct_coverage=min_struct_coverage,
        seed_parts=(pfam_id, candidate.get("uid"), candidate.get("seq_start"), candidate.get("seq_end"), ";".join(sorted(used_pdb_ids)), ";".join(sorted(used_chains))),
    ) if status == "OK" else {
        "random_n": 0,
        "random_mean": np.nan,
        "random_median": np.nan,
        "random_p_ge": np.nan,
        "random_z": np.nan,
        "random_enrichment": np.nan,
    }

    return {
        "status": status,
        "pdb_ids": ";".join(sorted(used_pdb_ids)),
        "chains": ";".join(sorted(used_chains)),
        "instances": used_instances,
        "struct_cov": struct_cov,
        "mapped_fragment_residues": len(union_mapped_frag_positions),
        "interface_count": len(union_interface_frag_positions),
        "interface_fraction": interface_fraction,
        "partner_pfams": ";".join(sorted(x for x in partner_pfams if x)),
        "partner_chains": ";".join(sorted(x for x in partner_chains if x)),
        "pair_types": ";".join(sorted(pair_types)),
        "different_protein_proxy": summarize_different_protein_proxy(pair_types),
        "has_interchain_different_pfam": "interchain_different_pfam" in pair_types,
        "has_interchain_same_pfam": "interchain_same_pfam" in pair_types,
        "has_intrachain_domain_contact": "intrachain_domain_contact" in pair_types,
        "random_n": rand["random_n"],
        "random_mean": rand["random_mean"],
        "random_median": rand["random_median"],
        "random_p_ge": rand["random_p_ge"],
        "random_z": rand["random_z"],
        "random_enrichment": rand["random_enrichment"],
    }


def evaluate_side(family_value: Any, segment: Tuple[int, int]) -> Dict[str, Any]:
    args = G_ARGS
    pfam_id = parse_family_id(family_value)
    if not pfam_id:
        return default_side_result("NO_FAMILY_ID", f"Cannot parse Pfam ID from {family_value}")

    h_start, h_end = int(segment[0]), int(segment[1])
    if h_start > h_end:
        h_start, h_end = h_end, h_start
    hmm_len = h_end - h_start + 1

    base = default_side_result("INIT")
    base.update({"Family": pfam_id, "HMM_Start": h_start, "HMM_End": h_end, "HMM_Len": hmm_len})

    if pfam_id not in G_COMPLEX_BY_PFAM:
        base.update({"Status": "NO_COMPLEX_INSTANCE_FOR_PFAM", "Note": "No known complex-interface instance for this Pfam"})
        return base

    family_seed_dir = os.path.join(args.pfam_seed_dir, pfam_id)
    family_structure_dir = os.path.join(args.pfam_structure_dir, pfam_id)
    hhm_path = os.path.join(family_seed_dir, f"{pfam_id}.hhm")
    aln_path = find_alignment_file(family_seed_dir, pfam_id)
    if not os.path.exists(hhm_path):
        base.update({"Status": "NO_HHM_FILE", "Note": hhm_path})
        return base
    if aln_path is None:
        base.update({"Status": "NO_ALIGNMENT_FILE", "Note": family_seed_dir})
        return base

    hhm_map = extract_hhm_to_msa_map(hhm_path)
    if not hhm_map:
        base.update({"Status": "NO_HHM_MAP", "Note": "HHM-to-MSA map is empty"})
        return base
    if h_start not in hhm_map or h_end not in hhm_map:
        base.update({"Status": "HMM_TO_MSA_FAIL", "Note": "HMM segment endpoints not found in local HHM-to-MSA map"})
        return base

    msa_start = hhm_map[h_start]
    msa_end = hhm_map[h_end]
    if msa_start > msa_end:
        msa_start, msa_end = msa_end, msa_start
    base.update({"MSA_Start": msa_start, "MSA_End": msa_end})

    candidates = get_seed_candidates_for_hmm_segment(
        aln_path,
        msa_start,
        msa_end,
        hmm_len,
        args.min_seq_ratio,
        args.max_seq_ratio,
        args.max_search_depth,
    )
    if not candidates:
        base.update({"Status": "NO_SEED_CANDIDATE_80_120", "Note": "No seed sequence realizes the HMM segment within the requested FASTA/HMM length ratio"})
        return base

    allowed = G_COMPLEX_BY_PFAM.get(pfam_id, set())
    evaluated: List[Tuple[Tuple[Any, ...], Dict[str, Any], Dict[str, Any]]] = []
    had_matching_structure = False
    had_low_struct = False

    for cand in candidates:
        metas = find_matching_structure_files(family_structure_dir, cand["uid"], allowed)
        if not metas:
            continue
        had_matching_structure = True
        ev = evaluate_candidate_against_complexes(
            pfam_id,
            cand,
            hmm_len,
            family_structure_dir,
            allowed,
            args.min_struct_coverage,
        )
        if ev is None:
            continue
        if ev["status"] == "LOW_STRUCTURE_COVERAGE":
            had_low_struct = True

        # Deterministic ranking. Do not rank by interface count or interface fraction.
        rank = (
            0 if ev["status"] == "OK" else 1,
            abs(float(cand["seq_ratio_vs_hmm"]) - 1.0),
            int(cand["len_diff"]),
            int(cand["gaps"]),
            -float(ev["struct_cov"]),
            str(cand["uid"]),
            str(ev["pdb_ids"]),
        )
        evaluated.append((rank, cand, ev))

    if not had_matching_structure:
        base.update({"Status": "NO_MATCHING_COMPLEX_SEED_STRUCTURE", "Note": "Seed candidates exist, but none has a local experimental structure that matches a known complex PDB-chain"})
        return base

    if not evaluated:
        base.update({"Status": "COMPLEX_STRUCTURE_MAPPING_FAIL", "Note": "Matching complex structures exist but seed-to-PDB mapping failed"})
        return base

    evaluated.sort(key=lambda x: x[0])
    _, cand, ev = evaluated[0]

    if ev["status"] != "OK":
        base.update({"Status": "LOW_STRUCTURE_COVERAGE", "Note": "Best matching complex structure covers less than the requested fraction of the HMM segment"})
    else:
        base.update({"Status": "OK", "Note": ""})

    interface_class = classify_interface(ev["interface_fraction"], args.th_major, args.th_partial) if ev["status"] == "OK" else "NA"
    base.update(
        {
            "Seq_UID": cand["uid"],
            "Seq_Header": cand["header"],
            "Seq_Start": int(cand["seq_start"]),
            "Seq_End": int(cand["seq_end"]),
            "Seq_Len": int(cand["seq_len"]),
            "SeqLen_vs_HMMLen": round(float(cand["seq_ratio_vs_hmm"]), 4),
            "PDB_IDs": ev["pdb_ids"],
            "Chains": ev["chains"],
            "Complex_Instance_Count": int(ev["instances"]),
            "Struct_Coverage_vs_HMM": round(float(ev["struct_cov"]), 4),
            "Mapped_Fragment_Residues": int(ev["mapped_fragment_residues"]),
            "Interface_Count": int(ev["interface_count"]),
            "Interface_Fraction": round(float(ev["interface_fraction"]), 4) if not pd.isna(ev["interface_fraction"]) else np.nan,
            "Interface_Class": interface_class,
            "Complex_Partner_Pfams": ev.get("partner_pfams", ""),
            "Complex_Partner_Chains": ev.get("partner_chains", ""),
            "Complex_Pair_Types": ev.get("pair_types", ""),
            "Different_Protein_Proxy": ev.get("different_protein_proxy", "UNKNOWN"),
            "Has_Interchain_Different_Pfam": bool(ev.get("has_interchain_different_pfam", False)),
            "Has_Interchain_Same_Pfam": bool(ev.get("has_interchain_same_pfam", False)),
            "Has_Intrachain_Domain_Contact": bool(ev.get("has_intrachain_domain_contact", False)),
            "Random_N": int(ev.get("random_n", 0)),
            "Random_Mean_Interface_Fraction": round(float(ev.get("random_mean")), 4) if not pd.isna(ev.get("random_mean")) else np.nan,
            "Random_Median_Interface_Fraction": round(float(ev.get("random_median")), 4) if not pd.isna(ev.get("random_median")) else np.nan,
            "Random_P_GE": round(float(ev.get("random_p_ge")), 6) if not pd.isna(ev.get("random_p_ge")) else np.nan,
            "Random_Z": round(float(ev.get("random_z")), 4) if not pd.isna(ev.get("random_z")) else np.nan,
            "Random_Enrichment": round(float(ev.get("random_enrichment")), 4) if not pd.isna(ev.get("random_enrichment")) else np.nan,
        }
    )
    return base


def prefix_side_result(side: str, res: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{side}_{k}": v for k, v in res.items()}


def process_row(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    row_id, row = item
    out = dict(row)
    out["Row_ID"] = row_id
    try:
        segs = parse_segment_pair_from_row(row)
        if segs is None:
            main_res = default_side_result("NO_SEGMENT", "Cannot parse Main/Sub HMM segment coordinates")
            sub_res = default_side_result("NO_SEGMENT", "Cannot parse Main/Sub HMM segment coordinates")
        else:
            main_seg, sub_seg = segs
            main_family = row.get("Main_HMM", row.get("Main_Family", row.get("Query_HMM", "")))
            sub_family = row.get("Sub_HMM", row.get("Sub_Family", row.get("Target_HMM", "")))
            main_res = evaluate_side(main_family, main_seg)
            sub_res = evaluate_side(sub_family, sub_seg)
        out.update(prefix_side_result("Main", main_res))
        out.update(prefix_side_result("Sub", sub_res))

        n_assessable = int(main_res.get("Status") == "OK") + int(sub_res.get("Status") == "OK")
        n_interface = int(main_res.get("Interface_Class") in {"InterfaceMajor", "InterfacePartial"}) + int(
            sub_res.get("Interface_Class") in {"InterfaceMajor", "InterfacePartial"}
        )
        n_diff_proxy = int(bool(main_res.get("Has_Interchain_Different_Pfam"))) + int(bool(sub_res.get("Has_Interchain_Different_Pfam")))
        out["Pair_Assessable_Sides"] = n_assessable
        out["Pair_Interface_Related_Sides"] = n_interface
        out["Pair_Different_Pfam_Interface_Sides"] = n_diff_proxy
        return out
    except Exception as exc:
        err = default_side_result("ERROR", f"{type(exc).__name__}: {exc}")
        out.update(prefix_side_result("Main", err))
        out.update(prefix_side_result("Sub", err))
        out["Pair_Assessable_Sides"] = 0
        out["Pair_Interface_Related_Sides"] = 0
        out["Pair_Different_Pfam_Interface_Sides"] = 0
        return out


def worker_init(args_obj: argparse.Namespace, complex_index, complex_by_pfam, complex_meta) -> None:
    global G_ARGS, G_COMPLEX_INDEX, G_COMPLEX_BY_PFAM, G_COMPLEX_META
    G_ARGS = args_obj
    G_COMPLEX_INDEX = complex_index
    G_COMPLEX_BY_PFAM = complex_by_pfam
    G_COMPLEX_META = complex_meta


# =============================================================================
# Reporting
# =============================================================================
def write_side_level_file(df: pd.DataFrame, output_side_csv: str) -> None:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        for side in ["Main", "Sub"]:
            rows.append(
                {
                    "Row_ID": row.get("Row_ID"),
                    "Side": side,
                    "Family": row.get(f"{side}_Family"),
                    "Status": row.get(f"{side}_Status"),
                    "Interface_Class": row.get(f"{side}_Interface_Class"),
                    "HMM_Start": row.get(f"{side}_HMM_Start"),
                    "HMM_End": row.get(f"{side}_HMM_End"),
                    "HMM_Len": row.get(f"{side}_HMM_Len"),
                    "MSA_Start": row.get(f"{side}_MSA_Start"),
                    "MSA_End": row.get(f"{side}_MSA_End"),
                    "Seq_UID": row.get(f"{side}_Seq_UID"),
                    "Seq_Header": row.get(f"{side}_Seq_Header"),
                    "Seq_Start": row.get(f"{side}_Seq_Start"),
                    "Seq_End": row.get(f"{side}_Seq_End"),
                    "Seq_Len": row.get(f"{side}_Seq_Len"),
                    "SeqLen_vs_HMMLen": row.get(f"{side}_SeqLen_vs_HMMLen"),
                    "PDB_IDs": row.get(f"{side}_PDB_IDs"),
                    "Chains": row.get(f"{side}_Chains"),
                    "Complex_Instance_Count": row.get(f"{side}_Complex_Instance_Count"),
                    "Struct_Coverage_vs_HMM": row.get(f"{side}_Struct_Coverage_vs_HMM"),
                    "Mapped_Fragment_Residues": row.get(f"{side}_Mapped_Fragment_Residues"),
                    "Interface_Count": row.get(f"{side}_Interface_Count"),
                    "Interface_Fraction": row.get(f"{side}_Interface_Fraction"),
                    "Complex_Partner_Pfams": row.get(f"{side}_Complex_Partner_Pfams"),
                    "Complex_Partner_Chains": row.get(f"{side}_Complex_Partner_Chains"),
                    "Complex_Pair_Types": row.get(f"{side}_Complex_Pair_Types"),
                    "Different_Protein_Proxy": row.get(f"{side}_Different_Protein_Proxy"),
                    "Has_Interchain_Different_Pfam": row.get(f"{side}_Has_Interchain_Different_Pfam"),
                    "Has_Interchain_Same_Pfam": row.get(f"{side}_Has_Interchain_Same_Pfam"),
                    "Has_Intrachain_Domain_Contact": row.get(f"{side}_Has_Intrachain_Domain_Contact"),
                    "Random_N": row.get(f"{side}_Random_N"),
                    "Random_Mean_Interface_Fraction": row.get(f"{side}_Random_Mean_Interface_Fraction"),
                    "Random_Median_Interface_Fraction": row.get(f"{side}_Random_Median_Interface_Fraction"),
                    "Random_P_GE": row.get(f"{side}_Random_P_GE"),
                    "Random_Z": row.get(f"{side}_Random_Z"),
                    "Random_Enrichment": row.get(f"{side}_Random_Enrichment"),
                    "Note": row.get(f"{side}_Note"),
                }
            )
    pd.DataFrame(rows).to_csv(output_side_csv, index=False)


def write_summary(df: pd.DataFrame, output_summary: str) -> None:
    lines: List[str] = []
    lines.append("# Pfam-Pfam 3did interface annotation summary\n")
    lines.append(f"Total rows: {len(df):,}\n")
    lines.append(f"Total fragment-sides: {len(df) * 2:,}\n")

    side_records = []
    for side in ["Main", "Sub"]:
        cols = [
            f"{side}_Status", f"{side}_Interface_Class", f"{side}_Interface_Fraction",
            f"{side}_Different_Protein_Proxy", f"{side}_Complex_Pair_Types",
            f"{side}_Random_Mean_Interface_Fraction", f"{side}_Random_P_GE", f"{side}_Random_Enrichment"
        ]
        existing = [c for c in cols if c in df.columns]
        tmp = df[existing].copy()
        tmp.columns = [c.replace(f"{side}_", "") for c in tmp.columns]
        side_records.append(tmp)
    sdf = pd.concat(side_records, ignore_index=True)

    lines.append("\n## Side-level status counts\n\n")
    lines.append(sdf["Status"].value_counts(dropna=False).to_string())
    lines.append("\n\n## Side-level interface-class counts among OK sides\n\n")
    ok = sdf[sdf["Status"] == "OK"].copy()
    if len(ok):
        lines.append(ok["Interface_Class"].value_counts(dropna=False).to_string())
        vals = pd.to_numeric(ok["Interface_Fraction"], errors="coerce").dropna()
        lines.append("\n\n## Interface fraction among OK sides\n\n")
        lines.append(f"n = {len(vals):,}\n")
        lines.append(f"mean = {vals.mean():.6f}\n")
        lines.append(f"median = {vals.median():.6f}\n")
    else:
        lines.append("No OK sides.\n")

    lines.append("\n\n## Pair-level counts\n\n")
    lines.append(df["Pair_Assessable_Sides"].value_counts(dropna=False).sort_index().to_string())
    lines.append("\n\nInterface-related sides per row:\n")
    lines.append(df["Pair_Interface_Related_Sides"].value_counts(dropna=False).sort_index().to_string())
    lines.append("\n")

    with open(output_summary, "w") as f:
        f.write("".join(lines))


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate ProDive Pfam-Pfam fragments against known protein-complex interface residues using seed-sequence mapping."
    )
    p.add_argument("--input-csv", required=True, help="ProDive Pfam-Pfam result CSV.")
    p.add_argument("--pfam-seed-dir", required=True, help="Directory containing PFxxxxx/PFxxxxx.hhm and PFxxxxx.fas/.sto.")
    p.add_argument("--pfam-structure-dir", required=True, help="Directory containing PFxxxxx/<UID>_exp_<PDB>_<CHAIN>.pdb files.")
    p.add_argument("--three-did-flat", required=True, help="Path to 3did_flat.gz or uncompressed 3did_flat.")
    p.add_argument("--output-csv", required=True, help="Output row-level CSV with Main_* and Sub_* annotations.")
    p.add_argument("--output-side-csv", default=None, help="Optional side-level output CSV. Default: output basename + .side_level.csv")
    p.add_argument("--output-summary", default=None, help="Optional text summary. Default: output basename + .summary.txt")

    p.add_argument("--min-seq-ratio", type=float, default=0.80, help="Minimum realized FASTA fragment length / HMM segment length. Default: 0.80")
    p.add_argument("--max-seq-ratio", type=float, default=1.20, help="Maximum realized FASTA fragment length / HMM segment length. Default: 1.20")
    p.add_argument("--min-struct-coverage", type=float, default=0.80, help="Minimum mapped PDB residues / HMM segment length for assessability. Default: 0.80")
    p.add_argument("--max-search-depth", type=int, default=200, help="Maximum seed candidates per fragment-side after 80-120 filtering. Default: 200")
    p.add_argument("--random-n", type=int, default=200, help="Number of same-chain same-length random windows per assessable fragment-side. Use 0 to disable. Default: 200")
    p.add_argument("--random-seed", type=int, default=20260601, help="Base random seed for deterministic random controls. Default: 20260601")
    p.add_argument("--random-max-attempts-per-sample", type=int, default=50, help="Maximum random-window attempts per requested valid sample. Default: 50")
    p.add_argument("--th-major", type=float, default=0.50, help="InterfaceMajor threshold on interface_fraction. Default: 0.50")
    p.add_argument("--th-partial", type=float, default=0.30, help="InterfacePartial threshold on interface_fraction. Default: 0.30")
    p.add_argument("--workers", type=int, default=1, help="Number of worker processes. Default: 1")
    p.add_argument("--limit", type=int, default=0, help="Debug only: process first N rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_seq_ratio <= 0 or args.max_seq_ratio <= 0 or args.min_seq_ratio > args.max_seq_ratio:
        raise SystemExit("ERROR: invalid --min-seq-ratio / --max-seq-ratio.")
    if not 0 < args.min_struct_coverage <= 1:
        raise SystemExit("ERROR: --min-struct-coverage must be in (0, 1].")
    if not 0 <= args.th_partial <= args.th_major <= 1:
        raise SystemExit("ERROR: require 0 <= --th-partial <= --th-major <= 1.")
    if args.random_n < 0 or args.random_max_attempts_per_sample < 1:
        raise SystemExit("ERROR: invalid random-control settings.")
    if args.workers < 1 or args.max_search_depth < 1 or args.limit < 0:
        raise SystemExit("ERROR: workers and search depth must be positive; limit cannot be negative.")
    for label, path in [
        ("input CSV", args.input_csv),
        ("Pfam seed directory", args.pfam_seed_dir),
        ("Pfam structure directory", args.pfam_structure_dir),
        ("3did flat file", args.three_did_flat),
    ]:
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: {label} not found: {path}")

    complex_index, complex_by_pfam, complex_meta = parse_3did_flat(args.three_did_flat)

    eprint(f"[complex] final indexed PDB-chain sides: {len(complex_index):,}")
    eprint(f"[complex] final Pfam count: {len(complex_by_pfam):,}")
    eprint(f"[complex] final metadata-bearing sides: {len(complex_meta):,}")

    df = pd.read_csv(args.input_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    eprint(f"[input] rows: {len(df):,}")

    records = [(i, row.to_dict()) for i, row in df.iterrows()]

    if args.workers <= 1:
        worker_init(args, complex_index, complex_by_pfam, complex_meta)
        out_rows = [process_row(x) for x in tqdm(records, desc="Annotating")]
    else:
        out_rows = []
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=worker_init,
            initargs=(args, complex_index, complex_by_pfam, complex_meta),
        ) as ex:
            for res in tqdm(ex.map(process_row, records, chunksize=20), total=len(records), desc="Annotating"):
                out_rows.append(res)

    out_df = pd.DataFrame(out_rows)
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    eprint(f"[output] row-level CSV written: {args.output_csv}")

    side_csv = args.output_side_csv
    if side_csv is None:
        side_csv = re.sub(r"\.csv$", "", args.output_csv) + ".side_level.csv"
    side_dir = os.path.dirname(os.path.abspath(side_csv))
    if side_dir:
        os.makedirs(side_dir, exist_ok=True)
    write_side_level_file(out_df, side_csv)
    eprint(f"[output] side-level CSV written: {side_csv}")

    summary_path = args.output_summary
    if summary_path is None:
        summary_path = re.sub(r"\.csv$", "", args.output_csv) + ".summary.txt"
    summary_dir = os.path.dirname(os.path.abspath(summary_path))
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    write_summary(out_df, summary_path)
    eprint(f"[output] summary written: {summary_path}")


if __name__ == "__main__":
    main()
