#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map ProDive Pfam-Pfam fragments to 3did complex-interface residues.

Map local HHM states through Pfam seed sequences onto experimental PDB chains.
Evaluate observed fragments and matched same-seed-domain, same-length random
windows with the same union across structures, excluding the observed window.
Write row-level and side-level annotations, a text summary, threshold-sensitivity
statistics, and continuous real-versus-random comparisons. Confidence intervals
and directional tests account for the two sides sharing an input Row_ID.

See ../README.md and --help for inputs, options, and output definitions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys
import glob
import traceback
import random
import hashlib
import math
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
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


def parse_residue_list(value: Any) -> Set[int]:
    """
    Parse residue lists such as: "1,2,3", "1 2 3", "1;2;5-8".
    Insertion codes are ignored; only the integer part is used.
    """
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    out: Set[int] = set()
    for token in re.split(r"[,;\s]+", text):
        token = token.strip()
        if not token:
            continue
        m_range = re.match(r"^(-?\d+)[A-Za-z]?\s*-\s*(-?\d+)[A-Za-z]?$", token)
        if m_range:
            a, b = int(m_range.group(1)), int(m_range.group(2))
            if a <= b:
                out.update(range(a, b + 1))
            else:
                out.update(range(b, a + 1))
            continue
        m_int = re.match(r"^(-?\d+)", token)
        if m_int:
            out.add(int(m_int.group(1)))
    return out


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
    exclude_same_pfam_pairs: bool = False,
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
        if exclude_same_pfam_pairs and pfam_a == pfam_b:
            current_instance = None
            return
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


def first_existing_col(row: pd.Series, names: List[str]) -> Optional[Any]:
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
    return None


def parse_complex_tsv(
    path: str,
) -> Tuple[Dict[Tuple[str, str, str], Set[int]], Dict[str, Set[Tuple[str, str]]], Dict[Tuple[str, str, str], List[Dict[str, Any]]]]:
    """
    Parse a normalized complex-interface table.

    Accepted one-side columns:
        pfam_id, pdb_id, chain_id, interface_residues

    Accepted paired columns:
        pfam_a, pdb_id, chain_a, interface_residues_a,
        pfam_b, chain_b, interface_residues_b
    """
    df = pd.read_csv(path, sep=None, engine="python")
    index: Dict[Tuple[str, str, str], Set[int]] = defaultdict(set)
    by_pfam: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    meta_index: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for ridx, row in df.iterrows():
        pdb = first_existing_col(row, ["pdb_id", "PDB", "PDB_ID", "pdb"])
        if pdb is None:
            continue

        pfam = first_existing_col(row, ["pfam_id", "Pfam", "PFAM", "pfam"])
        chain = first_existing_col(row, ["chain_id", "chain", "Chain", "CHAIN"])
        residues = first_existing_col(row, ["interface_residues", "interface_positions", "residues"])
        if pfam is not None and chain is not None and residues is not None:
            add_complex_side(
                index, by_pfam, meta_index, parse_family_id(pfam), str(pdb), str(chain), parse_residue_list(residues),
                source="complex_tsv", instance_id=f"complex_tsv:{ridx}", pair_type="unknown_complex_pair_type",
                pfam_relation="unknown_pfam_relation", chain_relation="unknown_chain_relation"
            )
            continue

        pfam_a = first_existing_col(row, ["pfam_a", "pfam1", "domain_a", "domain1"])
        pfam_b = first_existing_col(row, ["pfam_b", "pfam2", "domain_b", "domain2"])
        chain_a = first_existing_col(row, ["chain_a", "chain1"])
        chain_b = first_existing_col(row, ["chain_b", "chain2"])
        res_a = first_existing_col(row, ["interface_residues_a", "residues_a", "interface_positions_a"])
        res_b = first_existing_col(row, ["interface_residues_b", "residues_b", "interface_positions_b"])
        pair_type, pfam_relation, chain_relation = classify_complex_pair_type(str(pfam_a), str(pfam_b), str(chain_a), str(chain_b))
        if pfam_a is not None and chain_a is not None and res_a is not None:
            add_complex_side(index, by_pfam, meta_index, parse_family_id(pfam_a), str(pdb), str(chain_a), parse_residue_list(res_a),
                             partner_pfam=parse_family_id(pfam_b), partner_chain=str(chain_b) if chain_b is not None else None,
                             source="complex_tsv", instance_id=f"complex_tsv:{ridx}", pair_type=pair_type,
                             pfam_relation=pfam_relation, chain_relation=chain_relation)
        if pfam_b is not None and chain_b is not None and res_b is not None:
            add_complex_side(index, by_pfam, meta_index, parse_family_id(pfam_b), str(pdb), str(chain_b), parse_residue_list(res_b),
                             partner_pfam=parse_family_id(pfam_a), partner_chain=str(chain_a) if chain_a is not None else None,
                             source="complex_tsv", instance_id=f"complex_tsv:{ridx}", pair_type=pair_type,
                             pfam_relation=pfam_relation, chain_relation=chain_relation)

    eprint(f"[complex-tsv] indexed sides={len(index):,}, Pfams={len(by_pfam):,}")
    return dict(index), dict(by_pfam), dict(meta_index)


def merge_complex_indices(
    items: List[Tuple[Dict[Tuple[str, str, str], Set[int]], Dict[str, Set[Tuple[str, str]]], Dict[Tuple[str, str, str], List[Dict[str, Any]]]]]
) -> Tuple[Dict[Tuple[str, str, str], Set[int]], Dict[str, Set[Tuple[str, str]]], Dict[Tuple[str, str, str], List[Dict[str, Any]]]]:
    index: Dict[Tuple[str, str, str], Set[int]] = defaultdict(set)
    by_pfam: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    meta_index: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for idx, by, meta in items:
        for k, vals in idx.items():
            index[k].update(vals)
        for pf, vals in by.items():
            by_pfam[pf].update(vals)
        for k, vals in meta.items():
            meta_index[k].extend(vals)
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
DEFAULT_SENSITIVITY_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)


def parse_sensitivity_thresholds(value: Any) -> List[float]:
    """Parse, validate, sort, and deduplicate interface-fraction thresholds."""
    if value is None:
        values = list(DEFAULT_SENSITIVITY_THRESHOLDS)
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = [float(x) for x in value]
    else:
        values = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if not values:
        raise ValueError("At least one sensitivity threshold is required.")
    if any((not np.isfinite(x)) or x < 0.0 or x > 1.0 for x in values):
        raise ValueError("Sensitivity thresholds must be finite values between 0 and 1.")
    return sorted(set(round(float(x), 10) for x in values))


def get_analysis_thresholds() -> List[float]:
    values = getattr(G_ARGS, "analysis_thresholds", DEFAULT_SENSITIVITY_THRESHOLDS)
    return parse_sensitivity_thresholds(values)


def threshold_tag(threshold: float) -> str:
    """Stable column suffix; e.g. 0.30 -> T030 and 1.00 -> T100."""
    return f"T{int(round(float(threshold) * 100)):03d}"


def empty_random_result() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "random_n": 0,
        "random_mean": np.nan,
        "random_median": np.nan,
        "random_p_ge": np.nan,
        "random_p_le": np.nan,
        "random_p_two_sided": np.nan,
        "random_z": np.nan,
        "random_enrichment": np.nan,
        "random_mean_difference": np.nan,
    }
    for threshold in get_analysis_thresholds():
        tag = threshold_tag(threshold)
        out[f"real_ge_{tag}"] = np.nan
        out[f"random_count_ge_{tag}"] = 0
        out[f"random_prop_ge_{tag}"] = np.nan
    return out


def default_side_result(status: str, note: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {
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
        "Random_P_LE": np.nan,
        "Random_P_Two_Sided": np.nan,
        "Random_Z": np.nan,
        "Random_Enrichment": np.nan,
        "Random_Mean_Difference": np.nan,
    }
    for threshold in get_analysis_thresholds():
        tag = threshold_tag(threshold)
        out[f"Real_GE_{tag}"] = np.nan
        out[f"Random_Count_GE_{tag}"] = 0
        out[f"Random_Prop_GE_{tag}"] = np.nan
    return out


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
    real_seq_start: int,
    min_struct_coverage: float,
    seed_parts: Tuple[Any, ...],
) -> Dict[str, Any]:
    args = G_ARGS
    random_n = int(getattr(args, "random_n", 0))
    if random_n <= 0 or not used_maps or window_len <= 0:
        return empty_random_result()

    rng = random.Random(stable_random_seed(*seed_parts))
    max_attempts = max(random_n * int(getattr(args, "random_max_attempts_per_sample", 50)), random_n)
    vals: List[float] = []
    attempts = 0

    eligible_maps = [m for m in used_maps if int(m["full_seq_len"]) >= window_len]
    if not eligible_maps:
        return empty_random_result()

    # All maps describe the same selected Pfam seed sequence in different
    # experimental structures. Restrict to the same coordinate frame, then
    # evaluate every random window across the union of those structures exactly
    # as the real fragment is evaluated. The previous implementation selected
    # only one structure per random draw, which did not match the real union rule.
    frame_start = int(eligible_maps[0]["seed_abs_start"])
    frame_len = int(eligible_maps[0]["full_seq_len"])
    eligible_maps = [
        m for m in eligible_maps
        if int(m["seed_abs_start"]) == frame_start and int(m["full_seq_len"]) == frame_len
    ]
    possible_offsets = [
        q0 for q0 in range(0, frame_len - window_len + 1)
        if frame_start + q0 != int(real_seq_start)
    ]
    if not possible_offsets:
        return empty_random_result()

    while len(vals) < random_n and attempts < max_attempts:
        attempts += 1
        q0 = rng.choice(possible_offsets)
        start_abs = frame_start + q0
        positions = set(range(start_abs, start_abs + window_len))
        union_mapped: Set[int] = set()
        union_interface: Set[int] = set()
        for um in eligible_maps:
            seed_to_pdb = um["seed_to_pdb"]
            interface_pdb_nums = um["interface_pdb_nums"]
            mapped_here = {p for p in positions if p in seed_to_pdb}
            interface_here = {
                p for p in mapped_here if seed_to_pdb[p] in interface_pdb_nums
            }
            union_mapped.update(mapped_here)
            union_interface.update(interface_here)
        if not union_mapped:
            continue
        struct_cov = len(union_mapped) / float(hmm_len) if hmm_len > 0 else 0.0
        if struct_cov < min_struct_coverage:
            continue
        vals.append(len(union_interface) / float(len(union_mapped)))

    if not vals:
        return empty_random_result()

    arr = np.asarray(vals, dtype=float)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    real = float(real_interface_fraction) if not pd.isna(real_interface_fraction) else np.nan
    p_ge = float((np.sum(arr >= real) + 1) / (len(arr) + 1)) if not pd.isna(real) else np.nan
    p_le = float((np.sum(arr <= real) + 1) / (len(arr) + 1)) if not pd.isna(real) else np.nan
    p_two_sided = min(1.0, 2.0 * min(p_ge, p_le)) if not pd.isna(real) else np.nan
    z = float((real - mean) / std) if std > 0 and not pd.isna(real) else np.nan
    enrichment = float(real / mean) if mean > 0 and not pd.isna(real) else np.nan
    out: Dict[str, Any] = {
        "random_n": int(len(arr)),
        "random_mean": mean,
        "random_median": median,
        "random_p_ge": p_ge,
        "random_p_le": p_le,
        "random_p_two_sided": p_two_sided,
        "random_z": z,
        "random_enrichment": enrichment,
        "random_mean_difference": float(real - mean) if not pd.isna(real) else np.nan,
    }
    for threshold in get_analysis_thresholds():
        tag = threshold_tag(threshold)
        count_ge = int(np.sum(arr >= threshold))
        out[f"real_ge_{tag}"] = int(real >= threshold) if not pd.isna(real) else np.nan
        out[f"random_count_ge_{tag}"] = count_ge
        out[f"random_prop_ge_{tag}"] = count_ge / float(len(arr))
    return out


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
        real_seq_start=int(candidate["seq_start"]),
        min_struct_coverage=min_struct_coverage,
        seed_parts=(pfam_id, candidate.get("uid"), candidate.get("seq_start"), candidate.get("seq_end"), ";".join(sorted(used_pdb_ids)), ";".join(sorted(used_chains))),
    ) if status == "OK" else empty_random_result()

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
        **rand,
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
            "Random_P_LE": round(float(ev.get("random_p_le")), 6) if not pd.isna(ev.get("random_p_le")) else np.nan,
            "Random_P_Two_Sided": round(float(ev.get("random_p_two_sided")), 6) if not pd.isna(ev.get("random_p_two_sided")) else np.nan,
            "Random_Z": round(float(ev.get("random_z")), 4) if not pd.isna(ev.get("random_z")) else np.nan,
            "Random_Enrichment": round(float(ev.get("random_enrichment")), 4) if not pd.isna(ev.get("random_enrichment")) else np.nan,
            "Random_Mean_Difference": round(float(ev.get("random_mean_difference")), 6) if not pd.isna(ev.get("random_mean_difference")) else np.nan,
        }
    )
    for threshold in get_analysis_thresholds():
        tag = threshold_tag(threshold)
        real_value = ev.get(f"real_ge_{tag}")
        random_count = ev.get(f"random_count_ge_{tag}", 0)
        random_prop = ev.get(f"random_prop_ge_{tag}")
        base[f"Real_GE_{tag}"] = int(real_value) if not pd.isna(real_value) else np.nan
        base[f"Random_Count_GE_{tag}"] = int(random_count)
        base[f"Random_Prop_GE_{tag}"] = round(float(random_prop), 6) if not pd.isna(random_prop) else np.nan
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
def build_side_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    base_keys = [
        "Family", "Status", "Interface_Class", "HMM_Start", "HMM_End", "HMM_Len",
        "MSA_Start", "MSA_End", "Seq_UID", "Seq_Header", "Seq_Start", "Seq_End",
        "Seq_Len", "SeqLen_vs_HMMLen", "PDB_IDs", "Chains", "Complex_Instance_Count",
        "Struct_Coverage_vs_HMM", "Mapped_Fragment_Residues", "Interface_Count",
        "Interface_Fraction", "Complex_Partner_Pfams", "Complex_Partner_Chains",
        "Complex_Pair_Types", "Different_Protein_Proxy",
        "Has_Interchain_Different_Pfam", "Has_Interchain_Same_Pfam",
        "Has_Intrachain_Domain_Contact", "Random_N",
        "Random_Mean_Interface_Fraction", "Random_Median_Interface_Fraction",
        "Random_P_GE", "Random_P_LE", "Random_P_Two_Sided", "Random_Z",
        "Random_Enrichment", "Random_Mean_Difference", "Note",
    ]
    threshold_keys: List[str] = []
    for threshold in get_analysis_thresholds():
        tag = threshold_tag(threshold)
        threshold_keys.extend([
            f"Real_GE_{tag}",
            f"Random_Count_GE_{tag}",
            f"Random_Prop_GE_{tag}",
        ])

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        for side in ["Main", "Sub"]:
            side_row: Dict[str, Any] = {"Row_ID": row.get("Row_ID"), "Side": side}
            for key in base_keys + threshold_keys:
                side_row[key] = row.get(f"{side}_{key}")
            rows.append(side_row)
    return pd.DataFrame(rows)


def write_side_level_file(df: pd.DataFrame, output_side_csv: str) -> pd.DataFrame:
    sdf = build_side_level_dataframe(df)
    sdf.to_csv(output_side_csv, index=False)
    return sdf


def boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def analysis_subsets(sdf: pd.DataFrame) -> OrderedDict[str, pd.Series]:
    ok = sdf["Status"].eq("OK")
    proxy = sdf["Different_Protein_Proxy"].fillna("").astype(str)
    has_diff = boolean_series(sdf["Has_Interchain_Different_Pfam"])
    return OrderedDict(
        [
            ("All_OK", ok),
            ("Any_Interchain_Different_Pfam", ok & has_diff),
            ("Strict_Interchain_Different_Pfam_Only", ok & proxy.eq("YES_DIFFERENT_PFAM_INTERCHAIN")),
            ("Interchain_Same_Pfam_Only", ok & proxy.eq("UNCERTAIN_SAME_PFAM_INTERCHAIN")),
            ("Intrachain_Only", ok & proxy.eq("NO_INTRACHAIN_DOMAIN_CONTACT")),
        ]
    )


def normal_test_pvalues(z_value: float) -> Tuple[float, float, float]:
    if not np.isfinite(z_value):
        return np.nan, np.nan, np.nan
    p_enrichment = 0.5 * math.erfc(z_value / math.sqrt(2.0))
    p_depletion = 0.5 * math.erfc(-z_value / math.sqrt(2.0))
    p_two_sided = min(1.0, 2.0 * min(p_enrichment, p_depletion))
    return float(p_enrichment), float(p_depletion), float(p_two_sided)


def clustered_mean_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    z_crit: float,
    clamp: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float, float, int]:
    """Mean, cluster-robust CI, SE, and cluster count (clusters are ProDive rows)."""
    arr = np.asarray(values, dtype=float)
    clusters = np.asarray(cluster_ids)
    valid = np.isfinite(arr) & pd.notna(clusters)
    arr = arr[valid]
    clusters = clusters[valid]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = float(np.mean(arr))
    unique_clusters = pd.unique(clusters)
    n_clusters = int(len(unique_clusters))
    if n_clusters < 2:
        return mean, np.nan, np.nan, np.nan, n_clusters
    centered = arr - mean
    cluster_sums = np.asarray(
        [float(np.sum(centered[clusters == cluster])) for cluster in unique_clusters],
        dtype=float,
    )
    se = math.sqrt((n_clusters / float(n_clusters - 1)) * float(np.sum(cluster_sums ** 2))) / float(len(arr))
    low, high = mean - z_crit * se, mean + z_crit * se
    if clamp is not None:
        low, high = max(clamp[0], low), min(clamp[1], high)
    return mean, low, high, se, n_clusters


def calculate_threshold_sensitivity(sdf: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    z_crit = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    rows: List[Dict[str, Any]] = []
    random_n_all = pd.to_numeric(sdf["Random_N"], errors="coerce")

    for subset_name, subset_mask in analysis_subsets(sdf).items():
        for threshold in get_analysis_thresholds():
            tag = threshold_tag(threshold)
            real_col = pd.to_numeric(sdf[f"Real_GE_{tag}"], errors="coerce")
            random_count_col = pd.to_numeric(sdf[f"Random_Count_GE_{tag}"], errors="coerce")
            random_prop_col = pd.to_numeric(sdf[f"Random_Prop_GE_{tag}"], errors="coerce")
            valid = subset_mask & real_col.notna() & random_prop_col.notna() & (random_n_all > 0)
            n = int(valid.sum())
            if n == 0:
                continue

            real_values = real_col.loc[valid].to_numpy(dtype=float)
            random_props = random_prop_col.loc[valid].to_numpy(dtype=float)
            random_counts = random_count_col.loc[valid].to_numpy(dtype=float)
            random_ns = random_n_all.loc[valid].to_numpy(dtype=float)
            cluster_ids = sdf.loc[valid, "Row_ID"].to_numpy()
            paired_diff = real_values - random_props

            real_count = int(np.sum(real_values))
            real_prop = real_count / float(n)
            _, real_ci_low, real_ci_high, _, n_clusters = clustered_mean_ci(
                real_values, cluster_ids, z_crit, clamp=(0.0, 1.0)
            )
            random_mean_prop, random_ci_low, random_ci_high, _, _ = clustered_mean_ci(
                random_props, cluster_ids, z_crit, clamp=(0.0, 1.0)
            )
            diff_mean, diff_ci_low, diff_ci_high, diff_se, _ = clustered_mean_ci(
                paired_diff, cluster_ids, z_crit
            )
            z_value = diff_mean / diff_se if diff_se > 0 else np.nan
            p_enrich, p_deplete, p_two = normal_test_pvalues(z_value)
            pooled_random_n = int(np.sum(random_ns))
            pooled_random_count = int(np.sum(random_counts))

            role = "Sensitivity"
            if math.isclose(threshold, float(getattr(G_ARGS, "th_partial", 0.30)), abs_tol=1e-12):
                role = "Interface-related cutoff"
            if math.isclose(threshold, float(getattr(G_ARGS, "th_major", 0.50)), abs_tol=1e-12):
                role = "InterfaceMajor cutoff"

            rows.append(
                {
                    "Subset": subset_name,
                    "Threshold": threshold,
                    "Threshold_Role": role,
                    "N_Fragment_Sides": n,
                    "N_ProDive_Rows": n_clusters,
                    "Real_Count_GE": real_count,
                    "Real_Proportion_GE": real_prop,
                    "Real_Proportion_CI_Low": real_ci_low,
                    "Real_Proportion_CI_High": real_ci_high,
                    "Random_Expected_Count_GE": float(np.sum(random_props)),
                    "Random_Mean_Proportion_GE": random_mean_prop,
                    "Random_Mean_Proportion_CI_Low": random_ci_low,
                    "Random_Mean_Proportion_CI_High": random_ci_high,
                    "Real_Minus_Random": diff_mean,
                    "Difference_CI_Low": diff_ci_low,
                    "Difference_CI_High": diff_ci_high,
                    "Real_to_Random_Ratio": real_prop / random_mean_prop if random_mean_prop > 0 else np.nan,
                    "Z_Paired_Difference": z_value,
                    "P_Enrichment_One_Sided": p_enrich,
                    "P_Depletion_One_Sided": p_deplete,
                    "P_Two_Sided": p_two,
                    "Random_Total_Windows": pooled_random_n,
                    "Random_Count_GE_Pooled": pooled_random_count,
                    "Random_Pooled_Proportion_GE": pooled_random_count / float(pooled_random_n) if pooled_random_n > 0 else np.nan,
                    "Confidence_Level": confidence_level,
                }
            )
    return pd.DataFrame(rows)


def calculate_continuous_random_comparison(sdf: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    z_crit = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    real_all = pd.to_numeric(sdf["Interface_Fraction"], errors="coerce")
    random_all = pd.to_numeric(sdf["Random_Mean_Interface_Fraction"], errors="coerce")
    random_n_all = pd.to_numeric(sdf["Random_N"], errors="coerce")
    rows: List[Dict[str, Any]] = []

    for subset_name, subset_mask in analysis_subsets(sdf).items():
        valid = subset_mask & real_all.notna() & random_all.notna() & (random_n_all > 0)
        n = int(valid.sum())
        if n == 0:
            continue
        real = real_all.loc[valid].to_numpy(dtype=float)
        random_mean = random_all.loc[valid].to_numpy(dtype=float)
        cluster_ids = sdf.loc[valid, "Row_ID"].to_numpy()
        diff = real - random_mean
        diff_mean, ci_low, ci_high, se, n_clusters = clustered_mean_ci(diff, cluster_ids, z_crit)
        z_value = diff_mean / se if se > 0 else np.nan
        p_enrich, p_deplete, p_two = normal_test_pvalues(z_value)
        rows.append(
            {
                "Subset": subset_name,
                "N_Fragment_Sides": n,
                "N_ProDive_Rows": n_clusters,
                "Real_Mean_Interface_Fraction": float(np.mean(real)),
                "Real_Median_Interface_Fraction": float(np.median(real)),
                "Random_Mean_Interface_Fraction": float(np.mean(random_mean)),
                "Random_Median_of_Side_Means": float(np.median(random_mean)),
                "Mean_Paired_Difference": diff_mean,
                "Difference_CI_Low": ci_low,
                "Difference_CI_High": ci_high,
                "Real_Greater_Than_Random_Count": int(np.sum(real > random_mean)),
                "Real_Greater_Than_Random_Proportion": float(np.mean(real > random_mean)),
                "Z_Paired_Difference": z_value,
                "P_Enrichment_One_Sided": p_enrich,
                "P_Depletion_One_Sided": p_deplete,
                "P_Two_Sided": p_two,
                "Confidence_Level": confidence_level,
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    df: pd.DataFrame,
    output_summary: str,
    sdf: Optional[pd.DataFrame] = None,
    threshold_df: Optional[pd.DataFrame] = None,
    continuous_df: Optional[pd.DataFrame] = None,
) -> None:
    lines: List[str] = []
    lines.append("# Pfam-Pfam known-complex interface annotation summary\n")
    lines.append(f"Total rows: {len(df):,}\n")
    lines.append(f"Total fragment-sides: {len(df) * 2:,}\n")
    if sdf is None:
        sdf = build_side_level_dataframe(df)

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

    if threshold_df is not None and len(threshold_df):
        lines.append("\n## Threshold sensitivity: Real versus matched random background\n\n")
        lines.append(
            "Real_Minus_Random is the paired side-level difference between the observed "
            "threshold indicator and the matched random exceedance probability. Positive values "
            "indicate enrichment; negative values indicate depletion. Confidence intervals and "
            "normal-approximation p-values use ProDive Row_ID as the correlation cluster.\n\n"
        )
        show_cols = [
            "Subset", "Threshold", "N_Fragment_Sides", "N_ProDive_Rows", "Real_Proportion_GE",
            "Random_Mean_Proportion_GE", "Real_Minus_Random", "Difference_CI_Low",
            "Difference_CI_High", "P_Enrichment_One_Sided",
            "P_Depletion_One_Sided", "P_Two_Sided",
        ]
        lines.append(threshold_df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        lines.append("\n")

    if continuous_df is not None and len(continuous_df):
        lines.append("\n## Continuous interface-fraction comparison\n\n")
        show_cols = [
            "Subset", "N_Fragment_Sides", "N_ProDive_Rows", "Real_Mean_Interface_Fraction",
            "Random_Mean_Interface_Fraction", "Mean_Paired_Difference",
            "Difference_CI_Low", "Difference_CI_High", "P_Enrichment_One_Sided",
            "P_Depletion_One_Sided", "P_Two_Sided",
        ]
        lines.append(continuous_df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
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
    p.add_argument("--three-did-flat", default=None, help="Path to 3did_flat.gz or uncompressed 3did_flat.")
    p.add_argument("--complex-tsv", default=None, help="Optional normalized complex-interface TSV/CSV. Can be used instead of or together with --three-did-flat.")
    p.add_argument("--output-csv", required=True, help="Output row-level CSV with Main_* and Sub_* annotations.")
    p.add_argument("--output-side-csv", default=None, help="Optional side-level output CSV. Default: output basename + .side_level.csv")
    p.add_argument("--output-summary", default=None, help="Optional text summary. Default: output basename + .summary.txt")
    p.add_argument("--output-threshold-csv", default=None, help="Threshold-sensitivity CSV. Default: output basename + .threshold_sensitivity.csv")
    p.add_argument("--output-continuous-csv", default=None, help="Continuous Real-vs-Random comparison CSV. Default: output basename + .continuous_random_comparison.csv")

    p.add_argument("--min-seq-ratio", type=float, default=0.80, help="Minimum realized FASTA fragment length / HMM segment length. Default: 0.80")
    p.add_argument("--max-seq-ratio", type=float, default=1.20, help="Maximum realized FASTA fragment length / HMM segment length. Default: 1.20")
    p.add_argument("--min-struct-coverage", type=float, default=0.80, help="Minimum mapped PDB residues / HMM segment length for assessability. Default: 0.80")
    p.add_argument("--max-search-depth", type=int, default=200, help="Maximum seed candidates per fragment-side after 80-120 filtering. Default: 200")
    p.add_argument("--random-n", type=int, default=200, help="Number of same-chain same-length random windows per assessable fragment-side. Use 0 to disable. Default: 200")
    p.add_argument("--random-seed", type=int, default=20260601, help="Base random seed for deterministic random controls. Default: 20260601")
    p.add_argument("--random-max-attempts-per-sample", type=int, default=50, help="Maximum random-window attempts per requested valid sample. Default: 50")
    p.add_argument("--th-major", type=float, default=0.50, help="InterfaceMajor threshold on interface_fraction. Default: 0.50")
    p.add_argument("--th-partial", type=float, default=0.30, help="InterfacePartial threshold on interface_fraction. Default: 0.30")
    p.add_argument("--sensitivity-thresholds", default="0.10,0.20,0.30,0.40,0.50", help="Comma-separated interface-fraction thresholds for Real-vs-Random sensitivity analysis. The partial and major cutoffs are always included.")
    p.add_argument("--confidence-level", type=float, default=0.95, help="Confidence level for proportion and paired-difference intervals. Default: 0.95")
    p.add_argument("--exclude-same-pfam-pairs", action="store_true", help="For 3did input, skip instances where the two interacting Pfam IDs are identical. This is a conservative proxy for excluding homomeric same-domain contacts.")
    p.add_argument("--workers", type=int, default=1, help="Number of worker processes. Default: 1")
    p.add_argument("--limit", type=int, default=0, help="Debug only: process first N rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.three_did_flat and not args.complex_tsv:
        raise SystemExit("ERROR: provide --three-did-flat and/or --complex-tsv.")

    if args.min_seq_ratio <= 0 or args.max_seq_ratio <= 0 or args.min_seq_ratio > args.max_seq_ratio:
        raise SystemExit("ERROR: invalid --min-seq-ratio / --max-seq-ratio.")
    if not (0.0 <= args.th_partial <= args.th_major <= 1.0):
        raise SystemExit("ERROR: require 0 <= --th-partial <= --th-major <= 1.")
    if not (0.0 < args.confidence_level < 1.0):
        raise SystemExit("ERROR: --confidence-level must be between 0 and 1.")
    if args.random_n < 0:
        raise SystemExit("ERROR: --random-n must be >= 0.")
    try:
        args.analysis_thresholds = parse_sensitivity_thresholds(
            list(parse_sensitivity_thresholds(args.sensitivity_thresholds))
            + [args.th_partial, args.th_major]
        )
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    index_items = []
    if args.three_did_flat:
        index_items.append(parse_3did_flat(args.three_did_flat, exclude_same_pfam_pairs=args.exclude_same_pfam_pairs))
    if args.complex_tsv:
        index_items.append(parse_complex_tsv(args.complex_tsv))
    complex_index, complex_by_pfam, complex_meta = merge_complex_indices(index_items)
    # Initialize the parent as well as worker processes so reporting uses the
    # same thresholds and configuration under both single- and multi-process runs.
    worker_init(args, complex_index, complex_by_pfam, complex_meta)

    eprint(f"[complex] final indexed PDB-chain sides: {len(complex_index):,}")
    eprint(f"[complex] final Pfam count: {len(complex_by_pfam):,}")
    eprint(f"[complex] final metadata-bearing sides: {len(complex_meta):,}")

    df = pd.read_csv(args.input_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    eprint(f"[input] rows: {len(df):,}")

    records = [(i, row.to_dict()) for i, row in df.iterrows()]

    if args.workers <= 1:
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
    side_df = write_side_level_file(out_df, side_csv)
    eprint(f"[output] side-level CSV written: {side_csv}")

    threshold_df = calculate_threshold_sensitivity(side_df, args.confidence_level)
    threshold_csv = args.output_threshold_csv
    if threshold_csv is None:
        threshold_csv = re.sub(r"\.csv$", "", args.output_csv) + ".threshold_sensitivity.csv"
    threshold_df.to_csv(threshold_csv, index=False)
    eprint(f"[output] threshold-sensitivity CSV written: {threshold_csv}")

    continuous_df = calculate_continuous_random_comparison(side_df, args.confidence_level)
    continuous_csv = args.output_continuous_csv
    if continuous_csv is None:
        continuous_csv = re.sub(r"\.csv$", "", args.output_csv) + ".continuous_random_comparison.csv"
    continuous_df.to_csv(continuous_csv, index=False)
    eprint(f"[output] continuous random-comparison CSV written: {continuous_csv}")

    summary_path = args.output_summary
    if summary_path is None:
        summary_path = re.sub(r"\.csv$", "", args.output_csv) + ".summary.txt"
    write_summary(out_df, summary_path, side_df, threshold_df, continuous_df)
    eprint(f"[output] summary written: {summary_path}")


if __name__ == "__main__":
    main()
