#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example command:
python3 scripts/01_compute_contact_order_and_random.py \
  --pfam-dir <PRODIVE_DATA_ROOT>/shared/PfamA_seed \
  --global-score-csv <PRODIVE_DATA_ROOT>/shared/global_high_score_summary_fin.csv \
  --actual-out-csv CHANGE_ME \
  --random-out-csv CHANGE_ME \
  --cache-dir CHANGE_ME \
  --jobs 24 \
  --cutoff 8 \
  --preprocess-jobs 24 \
  --preprocess-mode process \
  --chunk-size 500 \
  --random-samples 1000 \
  --ok-only \
  --min-coverage 0.8 \
  --max-coverage 1.2 \
  --summary-txt CHANGE_ME
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import time
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# ========================================================
# Basic helpers
# ========================================================

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\xa0", " ").strip()


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        s = safe_str(x)
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def ungap(seq: str) -> str:
    return seq.replace(".", "").replace("-", "")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ========================================================
# Data classes
# ========================================================

@dataclass
class HMMSegment:
    pfam_id: str
    hmm_start: int
    hmm_end: int
    pfam_side: str = ""


@dataclass
class SequenceMapping:
    pfam_id: str
    seed_seq_name: str
    accession: str
    target_header: str
    target_match_mode: str
    seed_global_start: int
    seed_global_end: int
    target_hmm_start: int
    target_hmm_end: int
    target_msa_start: int
    target_msa_end: int
    target_seq_start: int
    target_seq_end: int
    target_seq_len: int
    target_frag: str
    target_aligned_frag: str
    gap_count_in_msa_window: int


@dataclass
class LocalPDBInfo:
    accession: str
    pdb_id: str
    chain_id: str
    pdb_path: Path


@dataclass
class SIFTSMappingSegment:
    accession: str
    pdb_id: str
    chain_id: str
    unp_start: int
    unp_end: int
    pdb_start_num: int
    pdb_start_icode: str
    pdb_end_num: int
    pdb_end_icode: str


@dataclass
class Residue:
    chain_id: str
    resseq: int
    icode: str
    resname: str
    atoms: Dict[str, Tuple[float, float, float]]

    @property
    def pdb_token(self) -> str:
        return f"{self.resseq}{self.icode}".strip()


# ========================================================
# HHM / alignment mapping helpers
# ========================================================

def extract_hmm_to_msa_map(hhm_path: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    with hhm_path.open("r", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 10 and parts[1].isdigit():
                try:
                    hmm_idx = int(parts[1])
                    msa_col_str = parts[-1]
                    if msa_col_str.isdigit():
                        mapping[hmm_idx] = int(msa_col_str)
                except Exception:
                    continue
    if not mapping:
        raise RuntimeError(f"No HMM->MSA mapping parsed from HHM: {hhm_path}")
    return mapping


def read_alignment(aln_path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    current_header = None
    with aln_path.open("r", errors="ignore") as f:
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
        raise RuntimeError(f"Alignment file is empty or failed to parse: {aln_path}")
    return seqs


def choose_alignment_path(family_dir: Path, pfam_id: str) -> Path:
    fas_path = family_dir / f"{pfam_id}.fas"
    sto_path = family_dir / f"{pfam_id}.sto"
    if fas_path.exists():
        return fas_path
    if sto_path.exists():
        return sto_path
    raise FileNotFoundError(f"Neither {pfam_id}.fas nor {pfam_id}.sto exists under {family_dir}")


def normalize_header_text(text: str) -> str:
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def build_target_candidates(seed_seq_name: str, accession: str = "") -> List[str]:
    candidates: List[str] = []
    for value in [seed_seq_name, accession]:
        value = safe_str(value)
        if value:
            candidates.append(value)
            if "." in value:
                candidates.append(value.split(".", 1)[0])
    uniq: List[str] = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def find_target_sequence(seqs: Dict[str, str], seed_seq_name: str, accession: str = "") -> Tuple[str, str, str]:
    candidates = build_target_candidates(seed_seq_name, accession)
    headers = list(seqs.keys())
    normalized = {h: normalize_header_text(h) for h in headers}

    for c in candidates:
        c_norm = normalize_header_text(c)
        for h in headers:
            if normalized[h] == c_norm:
                return h, seqs[h], f"exact:{c}"

    for c in candidates:
        c_norm = normalize_header_text(c)
        for h in headers:
            if normalized[h].startswith(c_norm):
                return h, seqs[h], f"startswith:{c}"

    for c in candidates:
        c_plain = re.escape(c.split(".")[0])
        pattern = re.compile(rf"(?<![A-Za-z0-9]){c_plain}(?![A-Za-z0-9])")
        for h in headers:
            if pattern.search(normalized[h]):
                return h, seqs[h], f"token:{c}"

    for c in candidates:
        c0 = c.split(".")[0]
        for h in headers:
            if c0 in normalized[h]:
                return h, seqs[h], f"contains:{c}"

    raise RuntimeError(f"Cannot find target sequence in alignment. Candidates: {candidates[:10]}")


def parse_header_global_range(header: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"/(\d+)-(\d+)", safe_str(header))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def map_msa_interval_to_real_seq(aligned_seq: str, global_start: int, msa_start_col: int, msa_end_col: int) -> Dict[str, Any]:
    idx_start = msa_start_col - 1
    idx_end = msa_end_col

    if len(aligned_seq) < idx_end:
        return {
            "status": "MSA_Out_Of_Range",
            "msa_start": msa_start_col,
            "msa_end": msa_end_col,
            "seq_start": None,
            "seq_end": None,
            "seq_len": 0,
            "frag_raw": "",
            "frag_pure": "",
            "gaps": None,
        }

    frag_raw = aligned_seq[idx_start:idx_end]
    frag_pure = ungap(frag_raw)
    gaps = frag_raw.count(".") + frag_raw.count("-")
    if len(frag_pure) == 0:
        return {
            "status": "Only_Gaps_On_Target",
            "msa_start": msa_start_col,
            "msa_end": msa_end_col,
            "seq_start": None,
            "seq_end": None,
            "seq_len": 0,
            "frag_raw": frag_raw,
            "frag_pure": frag_pure,
            "gaps": gaps,
        }

    prefix = aligned_seq[:idx_start]
    real_res_before = len(ungap(prefix))
    seq_start = global_start + real_res_before
    seq_end = seq_start + len(frag_pure) - 1
    return {
        "status": "OK",
        "msa_start": msa_start_col,
        "msa_end": msa_end_col,
        "seq_start": seq_start,
        "seq_end": seq_end,
        "seq_len": len(frag_pure),
        "frag_raw": frag_raw,
        "frag_pure": frag_pure,
        "gaps": gaps,
    }


def parse_main_segment(text: str) -> List[Tuple[int, int]]:
    pairs = re.findall(r"(\d+)-(\d+)", safe_str(text))
    return [(int(a), int(b)) for a, b in pairs]


def parse_sub_segments_details(text: str) -> List[Tuple[int, int, int, int]]:
    pairs = re.findall(r"(\d+)-(\d+)\s*->\s*(\d+)-(\d+)", safe_str(text))
    return [(int(a), int(b), int(c), int(d)) for a, b, c, d in pairs]


def extract_target_hmm_segments(match_row: Dict[str, Any], target_pfam: str) -> List[HMMSegment]:
    segments: List[HMMSegment] = []
    main_hmm = safe_str(match_row.get("Main_HMM", ""))
    sub_hmm = safe_str(match_row.get("Sub_HMM", ""))
    main_segment = safe_str(match_row.get("Main_Segment", ""))
    details = safe_str(match_row.get("Sub_Segments_Details", ""))
    detail_pairs = parse_sub_segments_details(details)

    if main_hmm == target_pfam:
        main_pairs = parse_main_segment(main_segment)
        if main_pairs:
            for s, e in main_pairs:
                segments.append(HMMSegment(pfam_id=target_pfam, hmm_start=s, hmm_end=e, pfam_side="main"))
        else:
            for a, b, _, _ in detail_pairs:
                segments.append(HMMSegment(pfam_id=target_pfam, hmm_start=a, hmm_end=b, pfam_side="main"))

    if sub_hmm == target_pfam:
        for _, _, c, d in detail_pairs:
            segments.append(HMMSegment(pfam_id=target_pfam, hmm_start=c, hmm_end=d, pfam_side="sub"))
    return segments


# ========================================================
# UniProt accession extraction and local PDB discovery
# ========================================================

UNIPROT_PATTERNS = [
    re.compile(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9])\b", re.I),
    re.compile(r"\b([A-NR-Z][0-9][A-Z0-9]{3}[0-9])\b", re.I),
    re.compile(r"\b([A-NR-Z][0-9](?:[A-Z0-9]{3}[0-9]){2})\b", re.I),
]


def infer_accessions(text: str) -> List[str]:
    text = safe_str(text)
    hits: List[str] = []
    for pat in UNIPROT_PATTERNS:
        for m in pat.finditer(text):
            hits.append(m.group(1).upper())
    uniq: List[str] = []
    seen = set()
    for h in hits:
        if h not in seen:
            uniq.append(h)
            seen.add(h)
    return uniq


def parse_local_pdb_filename(path: Path) -> Optional[LocalPDBInfo]:
    m = re.match(r"^([^_]+)_exp_([0-9A-Za-z]{4})_([^\.]+)\.pdb$", path.name)
    if not m:
        return None
    accession, pdb_id, chain_id = m.group(1).upper(), m.group(2).lower(), m.group(3)
    return LocalPDBInfo(accession=accession, pdb_id=pdb_id, chain_id=chain_id, pdb_path=path)


# ========================================================
# SIFTS / PDBe API helpers
# ========================================================

PDBe_API = "https://www.ebi.ac.uk/pdbe/api/mappings"


def http_get_json(url: str, retries: int = 2, timeout: int = 30) -> Any:
    last_exc: Optional[Exception] = None
    headers = {"User-Agent": "contact-order-pipeline/1.0"}
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"HTTP GET failed for {url}: {last_exc}")


def load_json_cache(path: Path) -> Optional[Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json_cache(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)


def sifts_best_structures(accession: str, cache_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    if cache_dir:
        path = ensure_dir(cache_dir) / f"best_{accession.upper()}.json"
        data = load_json_cache(path)
        if data is None:
            url = f"{PDBe_API}/best_structures/{urllib.parse.quote(accession)}"
            data = http_get_json(url)
            save_json_cache(path, data)
    else:
        url = f"{PDBe_API}/best_structures/{urllib.parse.quote(accession)}"
        data = http_get_json(url)
    out: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and safe_str(item.get("pdb_id")):
                        out.append(item)
    return out


def sifts_uniprot_mappings_for_pdb(pdb_id: str, cache_dir: Optional[Path] = None) -> Any:
    pdb_id = pdb_id.lower()
    if cache_dir:
        path = ensure_dir(cache_dir) / f"uniprot_{pdb_id}.json"
        data = load_json_cache(path)
        if data is None:
            url = f"{PDBe_API}/uniprot/{urllib.parse.quote(pdb_id)}"
            data = http_get_json(url)
            save_json_cache(path, data)
        return data
    url = f"{PDBe_API}/uniprot/{urllib.parse.quote(pdb_id)}"
    return http_get_json(url)


def _get_nested_residue_number(d: Dict[str, Any]) -> Tuple[Optional[int], str]:
    if not isinstance(d, dict):
        return None, ""
    num = None
    for key in ["author_residue_number", "residue_number", "seq_num", "number"]:
        if key in d:
            num = safe_int(d.get(key), None)
            if num is not None:
                break
    icode = safe_str(d.get("author_insertion_code", d.get("insertion_code", "")))
    return num, icode


def parse_sifts_uniprot_segments(data: Any, accession: str, pdb_id: str, chain_id: str) -> List[SIFTSMappingSegment]:
    accession = accession.upper()
    pdb_id = pdb_id.lower()
    requested_chain = safe_str(chain_id)
    segments: List[SIFTSMappingSegment] = []

    if isinstance(data, dict):
        top = data.get(pdb_id, data.get(pdb_id.upper(), data))
        if isinstance(top, dict):
            uni = top.get("UniProt", top.get("uniprot", {}))
            if isinstance(uni, dict):
                for acc_key, acc_val in uni.items():
                    if safe_str(acc_key).split("-")[0].upper() != accession.split("-")[0]:
                        continue
                    mappings = []
                    if isinstance(acc_val, dict):
                        mappings = acc_val.get("mappings", [])
                    elif isinstance(acc_val, list):
                        mappings = acc_val
                    for m in mappings:
                        if not isinstance(m, dict):
                            continue
                        chain = safe_str(m.get("chain_id") or m.get("struct_asym_id") or m.get("asym_id") or m.get("auth_asym_id") or m.get("chain"))
                        if requested_chain and chain and chain != requested_chain:
                            continue
                        unp_start = safe_int(m.get("unp_start"), None)
                        unp_end = safe_int(m.get("unp_end"), None)
                        start_num, start_ic = _get_nested_residue_number(m.get("start", {}))
                        end_num, end_ic = _get_nested_residue_number(m.get("end", {}))
                        if None in (unp_start, unp_end, start_num, end_num):
                            continue
                        segments.append(SIFTSMappingSegment(
                            accession=accession,
                            pdb_id=pdb_id,
                            chain_id=chain or requested_chain,
                            unp_start=int(unp_start),
                            unp_end=int(unp_end),
                            pdb_start_num=int(start_num),
                            pdb_start_icode=start_ic,
                            pdb_end_num=int(end_num),
                            pdb_end_icode=end_ic,
                        ))
    if segments:
        if requested_chain:
            chain_hits = [s for s in segments if s.chain_id == requested_chain]
            if chain_hits:
                return chain_hits
        return segments
    raise RuntimeError(f"No SIFTS UniProt mapping found for accession={accession}, pdb={pdb_id}, chain={chain_id}")


# ========================================================
# PDB parser and contact-cache computation
# ========================================================

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def parse_pdb_atom_line(line: str):
    rec = line[0:6].strip()
    if rec not in {"ATOM", "HETATM"}:
        return None
    atom_name = line[12:16].strip()
    altloc = line[16:17].strip()
    resname = line[17:20].strip().upper()
    chain_id = line[21:22].strip()
    try:
        resseq = int(line[22:26].strip())
        icode = line[26:27].strip()
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
    except ValueError:
        return None
    return rec, atom_name, altloc, resname, chain_id, resseq, icode, x, y, z


def altloc_ok(altloc: str) -> bool:
    return altloc in {"", "A", "1"}


def load_chain_residues(pdb_path: Path, chain_id: str, include_hetatm: bool = False) -> List[Residue]:
    residues: Dict[Tuple[int, str], Residue] = {}
    with pdb_path.open("r", errors="ignore") as f:
        for line in f:
            rec = line[0:6].strip()
            if rec == "ENDMDL":
                break
            parsed = parse_pdb_atom_line(line)
            if parsed is None:
                continue
            rec, atom_name, altloc, resname, atom_chain_id, resseq, icode, x, y, z = parsed
            if atom_chain_id != chain_id:
                continue
            if rec == "HETATM" and not include_hetatm:
                continue
            if resname not in AA3_TO_1:
                continue
            if not altloc_ok(altloc):
                continue
            key = (resseq, icode)
            if key not in residues:
                residues[key] = Residue(chain_id=atom_chain_id, resseq=resseq, icode=icode, resname=resname, atoms={})
            residues[key].atoms[atom_name] = (x, y, z)
    ordered = [residues[k] for k in sorted(residues.keys(), key=lambda t: (t[0], t[1]))]
    if not ordered:
        raise RuntimeError(f"No usable residues found in {pdb_path} chain {chain_id}")
    return ordered


def sqdist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def residue_rep_coord(res: Residue, atom_mode: str) -> Optional[Tuple[float, float, float]]:
    if atom_mode == "alpha":
        return res.atoms.get("CA")
    if atom_mode == "beta":
        if res.resname == "GLY":
            return res.atoms.get("CA")
        return res.atoms.get("CB")
    raise ValueError("heavy mode does not use a single representative atom")


def heavy_atom_coords(res: Residue) -> List[Tuple[float, float, float]]:
    out = []
    for atom_name, xyz in res.atoms.items():
        if atom_name.startswith("H"):
            continue
        out.append(xyz)
    return out


def residue_pair_distance(res1: Residue, res2: Residue, atom_mode: str) -> Optional[float]:
    if atom_mode == "heavy":
        c1 = heavy_atom_coords(res1)
        c2 = heavy_atom_coords(res2)
        if not c1 or not c2:
            return None
        best = math.inf
        for a in c1:
            for b in c2:
                d2 = sqdist(a, b)
                if d2 < best:
                    best = d2
        return math.sqrt(best) if math.isfinite(best) else None
    c1 = residue_rep_coord(res1, atom_mode)
    c2 = residue_rep_coord(res2, atom_mode)
    if c1 is None or c2 is None:
        return None
    return math.sqrt(sqdist(c1, c2))


def cache_key_for_chain(pdb_path: str, chain_id: str, atom_mode: str, cutoff: float, exclude_near: int, include_hetatm: bool) -> str:
    raw = f"{Path(pdb_path).resolve()}|{chain_id}|{atom_mode}|{cutoff}|{exclude_near}|{int(include_hetatm)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


_WORKER_CHAIN_CACHE: Dict[str, Dict[str, Any]] = {}


def load_or_build_chain_cache(pdb_path: str, chain_id: str, atom_mode: str, cutoff: float, exclude_near: int, include_hetatm: bool, cache_dir: Optional[str]) -> Dict[str, Any]:
    key = cache_key_for_chain(pdb_path, chain_id, atom_mode, cutoff, exclude_near, include_hetatm)
    if key in _WORKER_CHAIN_CACHE:
        return _WORKER_CHAIN_CACHE[key]

    disk_path: Optional[Path] = None
    if cache_dir:
        disk_path = ensure_dir(Path(cache_dir) / "chain_cache") / f"{key}.pkl"
        if disk_path.exists():
            with disk_path.open("rb") as f:
                obj = pickle.load(f)
            _WORKER_CHAIN_CACHE[key] = obj
            return obj

    residues = load_chain_residues(Path(pdb_path), chain_id, include_hetatm=include_hetatm)
    n = len(residues)
    contacts: List[Tuple[int, int]] = []
    for i in range(n):
        ri = residues[i]
        for j in range(i + 1, n):
            if (j - i) <= exclude_near:
                continue
            d = residue_pair_distance(ri, residues[j], atom_mode=atom_mode)
            if d is not None and d <= cutoff:
                contacts.append((i, j))
    obj = {
        "tokens": [r.pdb_token for r in residues],
        "chain_length": n,
        "contacts": contacts,
    }
    if disk_path:
        tmp = disk_path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(disk_path)
    _WORKER_CHAIN_CACHE[key] = obj
    return obj


# ========================================================
# Mapping UniProt interval to PDB residue set via SIFTS
# ========================================================

def pdb_token(num: int, icode: str = "") -> str:
    return f"{int(num)}{safe_str(icode)}".strip()


def expand_sifts_segment(seg: SIFTSMappingSegment) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    unp_len = seg.unp_end - seg.unp_start
    pdb_len = seg.pdb_end_num - seg.pdb_start_num
    if seg.pdb_start_icode or seg.pdb_end_icode:
        if seg.unp_start == seg.unp_end:
            mapping[seg.unp_start] = pdb_token(seg.pdb_start_num, seg.pdb_start_icode)
        return mapping
    if unp_len < 0 or pdb_len < 0:
        return mapping
    if unp_len != pdb_len:
        mapping[seg.unp_start] = pdb_token(seg.pdb_start_num, seg.pdb_start_icode)
        mapping[seg.unp_end] = pdb_token(seg.pdb_end_num, seg.pdb_end_icode)
        return mapping
    for offset in range(unp_len + 1):
        mapping[seg.unp_start + offset] = pdb_token(seg.pdb_start_num + offset, "")
    return mapping


def build_chain_uniprot_map(segments: Sequence[SIFTSMappingSegment]) -> Dict[str, int]:
    token_to_unp: Dict[str, int] = {}
    for seg in segments:
        expanded = expand_sifts_segment(seg)
        for unp_pos, token in expanded.items():
            token_to_unp[token] = unp_pos
    return token_to_unp


def map_uniprot_interval_to_pdb_tokens(unp_start: int, unp_end: int, segments: Sequence[SIFTSMappingSegment]) -> List[str]:
    tokens: List[str] = []
    for seg in segments:
        expanded = expand_sifts_segment(seg)
        for pos in range(unp_start, unp_end + 1):
            tok = expanded.get(pos)
            if tok is not None:
                tokens.append(tok)
    seen = set()
    uniq: List[str] = []
    for t in tokens:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


# ========================================================
# In-memory caches for preprocessing
# ========================================================

class PreprocessCache:
    def __init__(self, pfam_dir: Path, sifts_cache_dir: Optional[Path]):
        self.pfam_dir = pfam_dir
        self.sifts_cache_dir = sifts_cache_dir
        self.family: Dict[str, Dict[str, Any]] = {}
        self.best_structures: Dict[str, List[Dict[str, Any]]] = {}
        self.sifts_pdb: Dict[str, Any] = {}
        self._family_lock = threading.RLock()
        self._best_lock = threading.RLock()
        self._sifts_lock = threading.RLock()

    def get_family(self, pfam_id: str) -> Dict[str, Any]:
        with self._family_lock:
            if pfam_id in self.family:
                return self.family[pfam_id]
            family_dir = self.pfam_dir / pfam_id
            hhm_path = family_dir / f"{pfam_id}.hhm"
            aln_path = choose_alignment_path(family_dir, pfam_id)
            hhm_map = extract_hmm_to_msa_map(hhm_path)
            seqs = read_alignment(aln_path)
            local_pdb_infos: List[LocalPDBInfo] = []
            accessions_with_local = set()
            for path in sorted(family_dir.glob("*_exp_*.pdb")):
                info = parse_local_pdb_filename(path)
                if info is None:
                    continue
                local_pdb_infos.append(info)
                accessions_with_local.add(info.accession)
            obj = {
                "family_dir": family_dir,
                "hhm_map": hhm_map,
                "seqs": seqs,
                "local_pdb_infos": local_pdb_infos,
                "accessions_with_local": accessions_with_local,
            }
            self.family[pfam_id] = obj
            return obj

    def get_best_structures(self, accession: str) -> List[Dict[str, Any]]:
        acc = accession.upper()
        with self._best_lock:
            if acc not in self.best_structures:
                self.best_structures[acc] = sifts_best_structures(acc, cache_dir=self.sifts_cache_dir)
            return self.best_structures[acc]

    def get_sifts_pdb(self, pdb_id: str) -> Any:
        pdb_id = pdb_id.lower()
        with self._sifts_lock:
            if pdb_id not in self.sifts_pdb:
                self.sifts_pdb[pdb_id] = sifts_uniprot_mappings_for_pdb(pdb_id, cache_dir=self.sifts_cache_dir)
            return self.sifts_pdb[pdb_id]


# ========================================================
# Multiprocess preprocessing helpers
# ========================================================

def _args_to_worker_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "pfam_dir": str(args.pfam_dir),
        "seed_seq_name": safe_str(args.seed_seq_name),
        "accession": safe_str(args.accession),
        "seed_start": safe_int(args.seed_start, None),
        "seed_end": safe_int(args.seed_end, None),
        "auto_pick_seed": bool(args.auto_pick_seed),
        "pdb": safe_str(args.pdb),
        "chain": safe_str(args.chain),
        "atom_mode": args.atom_mode,
        "cutoff": float(args.cutoff),
        "exclude_near": int(args.exclude_near),
        "include_hetatm": bool(args.include_hetatm),
        "long_range_threshold": int(args.long_range_threshold),
        "cache_dir": safe_str(args.cache_dir) or "",
    }

def _make_args_namespace(d: Dict[str, Any]) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, v)
    return ns

def _split_evenly(items: List[Any], n_parts: int) -> List[List[Any]]:
    if n_parts <= 1 or len(items) <= 1:
        return [items]
    n_parts = max(1, min(n_parts, len(items)))
    buckets = [[] for _ in range(n_parts)]
    for idx, item in enumerate(items):
        buckets[idx % n_parts].append(item)
    return [b for b in buckets if b]

def _preprocess_job_batch_worker(job_batch: List[Tuple[int, Dict[str, Any], str, int, HMMSegment]], args_dict: Dict[str, Any]) -> List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    args = _make_args_namespace(args_dict)
    sifts_cache_dir = ensure_dir(Path(args.cache_dir) / "sifts") if safe_str(args.cache_dir) else None
    cache = PreprocessCache(Path(args.pfam_dir), sifts_cache_dir=sifts_cache_dir)
    out: List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = []
    for row_index, row, pfam_id, seg_rank, seg in job_batch:
        out.append(build_task_for_segment(row, row_index, pfam_id, seg_rank, seg, args, cache))
    return out


# ========================================================
# Workflow helpers using caches
# ========================================================

def select_target_sequence_and_interval_cached(
    cache: PreprocessCache,
    pfam_id: str,
    hmm_start: int,
    hmm_end: int,
    seed_seq_name: str = "",
    accession: str = "",
    seed_start: Optional[int] = None,
    seed_end: Optional[int] = None,
    auto_pick: bool = False,
) -> Tuple[SequenceMapping, Path]:
    fam = cache.get_family(pfam_id)
    family_dir = fam["family_dir"]
    hhm_map = fam["hhm_map"]
    seqs = fam["seqs"]
    local_accessions = fam["accessions_with_local"]

    if hmm_start not in hhm_map or hmm_end not in hhm_map:
        raise RuntimeError(f"HMM positions {hmm_start}-{hmm_end} are not present in {pfam_id}.hhm")
    msa_start = hhm_map[hmm_start]
    msa_end = hhm_map[hmm_end]

    target_header = ""
    target_aln_seq = ""
    match_mode = ""
    chosen_accession = accession.upper() if accession else ""

    if seed_seq_name:
        target_header, target_aln_seq, match_mode = find_target_sequence(seqs, seed_seq_name, accession=chosen_accession)
        if not chosen_accession:
            candidates = infer_accessions(target_header) + infer_accessions(seed_seq_name)
            if candidates:
                chosen_accession = candidates[0]
    elif auto_pick:
        nominal_len = hmm_end - hmm_start + 1
        candidates: List[Tuple[Tuple[int, int, int, str], str, str, str, Dict[str, Any]]] = []
        for header, aln_seq in seqs.items():
            gs, ge = parse_header_global_range(header)
            if gs is None:
                continue
            mapped = map_msa_interval_to_real_seq(aln_seq, gs, msa_start, msa_end)
            if mapped["status"] != "OK":
                continue
            accs = infer_accessions(header)
            acc = accs[0] if accs else ""
            has_local = 0 if (acc and acc in local_accessions) else 1
            score = (has_local, abs(mapped["seq_len"] - nominal_len), int(mapped["gaps"] or 0), header)
            candidates.append((score, header, aln_seq, acc, mapped))
        if not candidates:
            raise RuntimeError("Could not auto-select a target sequence from the alignment.")
        candidates.sort(key=lambda x: x[0])
        _, target_header, target_aln_seq, chosen_accession, pre_mapped = candidates[0]
        match_mode = "auto_best_realized_len_with_local_exp_preference"
        if seed_start is None:
            seed_start = int(pre_mapped["seq_start"])
        if seed_end is None:
            seed_end = int(pre_mapped["seq_end"])
        if not seed_seq_name:
            seed_seq_name = target_header
    else:
        raise RuntimeError("You must provide --seed-seq-name or use auto-pick.")

    header_start, header_end = parse_header_global_range(target_header)
    global_start = seed_start if seed_start is not None else header_start
    global_end = seed_end if seed_end is not None else header_end
    if global_start is None or global_end is None:
        raise RuntimeError(f"No usable sequence range found for target header: {target_header}")

    mapped = map_msa_interval_to_real_seq(target_aln_seq, global_start, msa_start, msa_end)
    if mapped["status"] != "OK":
        raise RuntimeError(f"Mapping HMM interval to real sequence failed: {mapped['status']}")

    if not chosen_accession:
        accs = infer_accessions(target_header) + infer_accessions(seed_seq_name)
        if accs:
            chosen_accession = accs[0]

    sequence_mapping = SequenceMapping(
        pfam_id=pfam_id,
        seed_seq_name=seed_seq_name or target_header,
        accession=chosen_accession,
        target_header=target_header,
        target_match_mode=match_mode,
        seed_global_start=int(global_start),
        seed_global_end=int(global_end),
        target_hmm_start=int(hmm_start),
        target_hmm_end=int(hmm_end),
        target_msa_start=int(mapped["msa_start"]),
        target_msa_end=int(mapped["msa_end"]),
        target_seq_start=int(mapped["seq_start"]),
        target_seq_end=int(mapped["seq_end"]),
        target_seq_len=int(mapped["seq_len"]),
        target_frag=safe_str(mapped["frag_pure"]),
        target_aligned_frag=safe_str(mapped["frag_raw"]),
        gap_count_in_msa_window=int(mapped["gaps"] or 0),
    )
    return sequence_mapping, family_dir


def choose_best_local_pdb(local_infos: List[LocalPDBInfo], accession: str, cache: PreprocessCache) -> LocalPDBInfo:
    if not local_infos:
        raise RuntimeError("No local experimental PDB files available to choose from.")
    if len(local_infos) == 1:
        return local_infos[0]
    try:
        best = cache.get_best_structures(accession)
        rank: Dict[Tuple[str, str], int] = {}
        for i, item in enumerate(best):
            key = (safe_str(item.get("pdb_id")).lower(), safe_str(item.get("chain_id")))
            rank[key] = i
        local_infos = sorted(local_infos, key=lambda x: (rank.get((x.pdb_id.lower(), x.chain_id), 10**9), x.pdb_id, x.chain_id))
    except Exception:
        local_infos = sorted(local_infos, key=lambda x: (x.pdb_id, x.chain_id))
    return local_infos[0]


def choose_local_pdb_for_sequence_cached(cache: PreprocessCache, mapping: SequenceMapping, family_dir: Path, explicit_pdb: str = "", explicit_chain: str = "") -> LocalPDBInfo:
    if explicit_pdb:
        path = Path(explicit_pdb)
        if not path.exists():
            raise FileNotFoundError(f"Explicit PDB file not found: {path}")
        info = parse_local_pdb_filename(path)
        if info is None:
            pdb_id = path.stem[-4:].lower()
            return LocalPDBInfo(accession=mapping.accession, pdb_id=pdb_id, chain_id=explicit_chain, pdb_path=path)
        if explicit_chain:
            info.chain_id = explicit_chain
        return info

    if not mapping.accession:
        raise RuntimeError("Cannot choose local experimental PDB automatically because no UniProt accession could be inferred.")

    fam = cache.get_family(mapping.pfam_id)
    local_infos = [x for x in fam["local_pdb_infos"] if x.accession == mapping.accession]
    if not local_infos:
        raise RuntimeError(f"No local experimental PDB file found under {family_dir} for accession {mapping.accession}")
    chosen = choose_best_local_pdb(local_infos, mapping.accession, cache)
    if explicit_chain:
        chosen.chain_id = explicit_chain
    return chosen


# ========================================================
# Task building and worker processing
# ========================================================

def extract_pfams_from_global_row(row: Dict[str, Any]) -> List[str]:
    pfams: List[str] = []
    for key in ("Main_HMM", "Sub_HMM"):
        value = safe_str(row.get(key, ""))
        if value.startswith("PF") and len(value) == 7 and value[2:].isdigit():
            pfams.append(value)
    uniq: List[str] = []
    seen = set()
    for pf in pfams:
        if pf not in seen:
            uniq.append(pf)
            seen.add(pf)
    return uniq


def fixed_fieldnames() -> List[str]:
    return [
        "status", "message", "global_row_index", "pfam_id", "pfam_side", "segment_rank",
        "target_hmm_start", "target_hmm_end", "main_hmm", "sub_hmm", "main_segment", "sub_segments_details",
        "seed_seq_name", "accession", "target_header", "target_match_mode", "seed_global_start", "seed_global_end",
        "target_msa_start", "target_msa_end", "target_seq_start", "target_seq_end", "target_seq_len",
        "target_frag", "target_aligned_frag", "gap_count_in_msa_window", "pdb_id", "chain_id", "pdb_path",
        "mapped_pdb_residue_count", "mapped_pdb_residues", "atom_mode", "cutoff", "exclude_near", "long_range_threshold",
        "chain_length", "fragment_size_mapped", "fragment_tokens", "n_total_contacts", "n_fragment_contacts",
        "sum_sequence_separation", "mean_sequence_separation", "rfco", "lr_contact_fraction",
    ]


def build_task_for_segment(row: Dict[str, Any], row_index: int, pfam_id: str, seg_rank: int, seg: HMMSegment, args: argparse.Namespace, cache: PreprocessCache) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        mapping, family_dir = select_target_sequence_and_interval_cached(
            cache=cache,
            pfam_id=pfam_id,
            hmm_start=seg.hmm_start,
            hmm_end=seg.hmm_end,
            seed_seq_name=safe_str(args.seed_seq_name),
            accession=safe_str(args.accession),
            seed_start=safe_int(args.seed_start, None),
            seed_end=safe_int(args.seed_end, None),
            auto_pick=(bool(args.auto_pick_seed) or not safe_str(args.seed_seq_name)),
        )
        local_pdb = choose_local_pdb_for_sequence_cached(
            cache=cache,
            mapping=mapping,
            family_dir=family_dir,
            explicit_pdb=safe_str(args.pdb),
            explicit_chain=safe_str(args.chain),
        )
        sifts_raw = cache.get_sifts_pdb(local_pdb.pdb_id)
        sifts_segments = parse_sifts_uniprot_segments(sifts_raw, mapping.accession, local_pdb.pdb_id, local_pdb.chain_id)
        token_to_unp = build_chain_uniprot_map(sifts_segments)
        frag_tokens = map_uniprot_interval_to_pdb_tokens(mapping.target_seq_start, mapping.target_seq_end, sifts_segments)
        if not frag_tokens:
            raise RuntimeError(
                f"SIFTS returned no mapped PDB residues for UniProt interval {mapping.target_seq_start}-{mapping.target_seq_end} "
                f"on {local_pdb.pdb_id}:{local_pdb.chain_id}"
            )
        meta = {
            "status": "OK", "message": "", "global_row_index": row_index, "pfam_id": pfam_id, "pfam_side": seg.pfam_side,
            "segment_rank": seg_rank, "target_hmm_start": seg.hmm_start, "target_hmm_end": seg.hmm_end,
            "main_hmm": safe_str(row.get("Main_HMM", "")), "sub_hmm": safe_str(row.get("Sub_HMM", "")),
            "main_segment": safe_str(row.get("Main_Segment", "")), "sub_segments_details": safe_str(row.get("Sub_Segments_Details", "")),
            "seed_seq_name": mapping.seed_seq_name, "accession": mapping.accession, "target_header": mapping.target_header,
            "target_match_mode": mapping.target_match_mode, "seed_global_start": mapping.seed_global_start, "seed_global_end": mapping.seed_global_end,
            "target_msa_start": mapping.target_msa_start, "target_msa_end": mapping.target_msa_end,
            "target_seq_start": mapping.target_seq_start, "target_seq_end": mapping.target_seq_end, "target_seq_len": mapping.target_seq_len,
            "target_frag": mapping.target_frag, "target_aligned_frag": mapping.target_aligned_frag,
            "gap_count_in_msa_window": mapping.gap_count_in_msa_window, "pdb_id": local_pdb.pdb_id, "chain_id": local_pdb.chain_id,
            "pdb_path": str(local_pdb.pdb_path), "mapped_pdb_residue_count": len(frag_tokens), "mapped_pdb_residues": ";".join(frag_tokens),
            "atom_mode": args.atom_mode, "cutoff": float(args.cutoff), "exclude_near": int(args.exclude_near),
            "long_range_threshold": int(args.long_range_threshold),
        }
        task = {
            "group_key": (str(local_pdb.pdb_path), local_pdb.chain_id, mapping.accession.upper(), local_pdb.pdb_id.lower(), args.atom_mode, float(args.cutoff), int(args.exclude_near), bool(args.include_hetatm), int(args.long_range_threshold)),
            "pdb_path": str(local_pdb.pdb_path),
            "chain_id": local_pdb.chain_id,
            "accession": mapping.accession.upper(),
            "pdb_id": local_pdb.pdb_id.lower(),
            "atom_mode": args.atom_mode,
            "cutoff": float(args.cutoff),
            "exclude_near": int(args.exclude_near),
            "include_hetatm": bool(args.include_hetatm),
            "long_range_threshold": int(args.long_range_threshold),
            "token_to_unp": token_to_unp,
            "fragment_tokens": frag_tokens,
            "meta": meta,
        }
        return task, None
    except Exception as exc:
        fail = {
            "status": "FAIL",
            "message": f"{type(exc).__name__}: {exc}",
            "global_row_index": row_index,
            "pfam_id": pfam_id,
            "pfam_side": seg.pfam_side,
            "segment_rank": seg_rank,
            "target_hmm_start": getattr(seg, "hmm_start", ""),
            "target_hmm_end": getattr(seg, "hmm_end", ""),
            "main_hmm": safe_str(row.get("Main_HMM", "")),
            "sub_hmm": safe_str(row.get("Sub_HMM", "")),
            "main_segment": safe_str(row.get("Main_Segment", "")),
            "sub_segments_details": safe_str(row.get("Sub_Segments_Details", "")),
        }
        return None, fail


def _iter_segment_jobs(rows_chunk: List[Tuple[int, Dict[str, Any]]], args: argparse.Namespace) -> Tuple[List[Tuple[int, Dict[str, Any], str, int, HMMSegment]], List[Dict[str, Any]]]:
    segment_jobs: List[Tuple[int, Dict[str, Any], str, int, HMMSegment]] = []
    immediate_results: List[Dict[str, Any]] = []
    for row_index, row in rows_chunk:
        pfams = [args.pfam_id] if safe_str(args.pfam_id) else extract_pfams_from_global_row(row)
        if not pfams:
            immediate_results.append({"status": "SKIP", "message": "No PFxxxxx found in Main_HMM/Sub_HMM", "global_row_index": row_index})
            continue
        for pfam_id in pfams:
            segs = extract_target_hmm_segments(row, pfam_id)
            if not segs:
                immediate_results.append({
                    "status": "SKIP",
                    "message": "No HMM segment extracted for this PFAM in this row",
                    "global_row_index": row_index,
                    "pfam_id": pfam_id,
                })
                continue
            for seg_rank, seg in enumerate(segs, start=1):
                segment_jobs.append((row_index, row, pfam_id, seg_rank, seg))
    return segment_jobs, immediate_results


def _preprocess_segment_job(job: Tuple[int, Dict[str, Any], str, int, HMMSegment], args: argparse.Namespace, cache: PreprocessCache) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    row_index, row, pfam_id, seg_rank, seg = job
    return build_task_for_segment(row, row_index, pfam_id, seg_rank, seg, args, cache)


def _metrics_from_indices(frag_indices: Sequence[int], adjacency: List[List[int]], contact_sep: List[int], tokens: List[str], chain_len: int, long_thr: int) -> Dict[str, Any]:
    frag_set = set(frag_indices)
    contact_ids = set()
    for idx in frag_set:
        contact_ids.update(adjacency[idx])
    frag_contacts = len(contact_ids)
    sum_sep = 0
    lr_count = 0
    for cid in contact_ids:
        sep = contact_sep[cid]
        sum_sep += sep
        if sep >= long_thr:
            lr_count += 1
    rfco = (sum_sep / (chain_len * frag_contacts)) if frag_contacts > 0 else None
    mean_sep = (sum_sep / frag_contacts) if frag_contacts > 0 else None
    lr_frac = (lr_count / frag_contacts) if frag_contacts > 0 else None
    ordered_fragment = [tokens[i] for i in sorted(frag_set)]
    return {
        "fragment_size_mapped": len(frag_set),
        "fragment_tokens": ";".join(ordered_fragment),
        "n_fragment_contacts": frag_contacts,
        "sum_sequence_separation": sum_sep,
        "mean_sequence_separation": mean_sep,
        "rfco": rfco,
        "lr_contact_fraction": lr_frac,
    }


def _precompute_random_windows(length: int, adjacency: List[List[int]], contact_sep: List[int], tokens: List[str], chain_len: int, long_thr: int) -> List[Dict[str, Any]]:
    if length <= 0 or length > chain_len:
        return []
    out: List[Dict[str, Any]] = []
    for start in range(0, chain_len - length + 1):
        inds = list(range(start, start + length))
        m = _metrics_from_indices(inds, adjacency, contact_sep, tokens, chain_len, long_thr)
        m.update({
            "random_start_index0": start,
            "random_end_index0": start + length - 1,
            "random_start_token": tokens[start],
            "random_end_token": tokens[start + length - 1],
            "random_length": length,
        })
        out.append(m)
    return out


def fixed_random_fieldnames() -> List[str]:
    return [
        "global_row_index", "pfam_id", "pfam_side", "segment_rank",
        "random_sample_index", "random_start_index0", "random_end_index0",
        "random_start_token", "random_end_token",
        "random_n_fragment_contacts", "random_mean_sequence_separation",
        "random_rfco", "random_lr_contact_fraction",
    ]


def process_group_worker(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    try:
        cache = load_or_build_chain_cache(
            pdb_path=payload["pdb_path"],
            chain_id=payload["chain_id"],
            atom_mode=payload["atom_mode"],
            cutoff=payload["cutoff"],
            exclude_near=payload["exclude_near"],
            include_hetatm=payload["include_hetatm"],
            cache_dir=payload.get("cache_dir"),
        )
    except Exception as e:
        failed = []
        for task in payload["tasks"]:
            meta = dict(task["meta"])
            meta.update({"status": "FAIL", "message": f"chain_cache_error: {type(e).__name__}: {e}"})
            failed.append(meta)
        return {"results": failed, "random_rows": []}
    tokens: List[str] = cache["tokens"]
    contacts: List[Tuple[int, int]] = cache["contacts"]
    chain_len: int = cache["chain_length"]
    token_to_index = {tok: i for i, tok in enumerate(tokens)}
    adjacency: List[List[int]] = [[] for _ in range(chain_len)]
    token_to_unp = payload["token_to_unp"]

    contact_sep: List[int] = []
    for cid, (i, j) in enumerate(contacts):
        ti = tokens[i]
        tj = tokens[j]
        ui = token_to_unp.get(ti)
        uj = token_to_unp.get(tj)
        sep = abs(uj - ui) if (ui is not None and uj is not None) else abs(j - i)
        contact_sep.append(sep)
        adjacency[i].append(cid)
        adjacency[j].append(cid)

    results: List[Dict[str, Any]] = []
    random_rows: List[Dict[str, Any]] = []
    random_samples = int(payload.get("random_samples", 0) or 0)
    window_cache: Dict[int, List[Dict[str, Any]]] = {}

    for task in payload["tasks"]:
        meta = dict(task["meta"])
        frag_indices = sorted({token_to_index[t] for t in task["fragment_tokens"] if t in token_to_index})
        if not frag_indices:
            meta.update({
                "status": "FAIL",
                "message": "No mapped PDB residues with coordinates were found for the fragment.",
            })
            results.append(meta)
            continue
        long_thr = task["long_range_threshold"]
        metrics = _metrics_from_indices(frag_indices, adjacency, contact_sep, tokens, chain_len, long_thr)
        meta.update({
            "chain_length": chain_len,
            "n_total_contacts": len(contacts),
            **metrics,
        })
        results.append(meta)

        frag_len = len(frag_indices)
        if safe_str(meta.get("status")) == "OK" and random_samples > 0 and frag_len > 0 and frag_len <= chain_len:
            if frag_len not in window_cache:
                window_cache[frag_len] = _precompute_random_windows(frag_len, adjacency, contact_sep, tokens, chain_len, long_thr)
            windows = window_cache[frag_len]
            if windows:
                seed_text = f"{meta.get('global_row_index')}|{meta.get('pfam_id')}|{meta.get('pfam_side')}|{meta.get('segment_rank')}|{payload['pdb_path']}|{payload['chain_id']}|{frag_len}"
                seed = int(hashlib.md5(seed_text.encode('utf-8')).hexdigest()[:16], 16)
                rng = random.Random(seed)
                for sample_idx in range(1, random_samples + 1):
                    picked = windows[rng.randrange(len(windows))]
                    random_rows.append({
                        "global_row_index": meta.get("global_row_index"),
                        "pfam_id": meta.get("pfam_id"),
                        "pfam_side": meta.get("pfam_side"),
                        "segment_rank": meta.get("segment_rank"),
                        "random_sample_index": sample_idx,
                        "random_start_index0": picked["random_start_index0"],
                        "random_end_index0": picked["random_end_index0"],
                        "random_start_token": picked["random_start_token"],
                        "random_end_token": picked["random_end_token"],
                        "random_n_fragment_contacts": picked["n_fragment_contacts"],
                        "random_mean_sequence_separation": picked["mean_sequence_separation"],
                        "random_rfco": picked["rfco"],
                        "random_lr_contact_fraction": picked["lr_contact_fraction"],
                    })
    return {"results": results, "random_rows": random_rows}


# ========================================================
# Batch processing
# ========================================================

def init_output_csv(path: Path, fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_results_csv(path: Path, fieldnames: Sequence[str], results: Sequence[Dict[str, Any]]) -> int:
    ok_results = [r for r in results if safe_str(r.get("status")) == "OK"]
    if not ok_results:
        return 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for r in ok_results:
            writer.writerow(r)
    return len(ok_results)


def append_random_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for r in rows:
            writer.writerow(r)
    return len(rows)


# ========================================================
# Post-filter helpers (merged from recheck_filter.py)
# ========================================================

def ensure_cols(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def load_and_filter_actual(path: str, ok_only: bool, min_cov: float, max_cov: float):
    df = pd.read_csv(path, low_memory=False)
    req = ["global_row_index", "pfam_id", "pfam_side", "segment_rank",
           "target_hmm_start", "target_hmm_end", "fragment_size_mapped"]
    ensure_cols(df, req, "actual-csv")

    if ok_only and "status" in df.columns:
        df = df[df["status"].astype(str) == "OK"].copy()

    for c in ["target_hmm_start", "target_hmm_end", "fragment_size_mapped"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["target_hmm_start", "target_hmm_end", "fragment_size_mapped"]).copy()

    df["target_hmm_start"] = df["target_hmm_start"].astype(int)
    df["target_hmm_end"] = df["target_hmm_end"].astype(int)
    df["fragment_size_mapped"] = df["fragment_size_mapped"].astype(int)

    df["hmm_len"] = df["target_hmm_end"] - df["target_hmm_start"] + 1
    df = df[df["hmm_len"] > 0].copy()
    df["coverage_ratio"] = df["fragment_size_mapped"] / df["hmm_len"]

    filtered = df[(df["coverage_ratio"] >= min_cov) & (df["coverage_ratio"] <= max_cov)].copy()

    for c in ["global_row_index", "segment_rank"]:
        filtered[c] = pd.to_numeric(filtered[c], errors="coerce").astype("Int64")
    filtered["pfam_id"] = filtered["pfam_id"].astype(str)
    filtered["pfam_side"] = filtered["pfam_side"].astype(str)

    return df, filtered


def filter_random(random_csv: str, random_out_csv: str, keep_keys_df: pd.DataFrame, chunksize: int):
    key_df = keep_keys_df[["global_row_index", "pfam_id", "pfam_side", "segment_rank"]].copy()
    key_df["global_row_index"] = pd.to_numeric(key_df["global_row_index"], errors="coerce").astype("Int64")
    key_df["segment_rank"] = pd.to_numeric(key_df["segment_rank"], errors="coerce").astype("Int64")
    key_df["pfam_id"] = key_df["pfam_id"].astype(str)
    key_df["pfam_side"] = key_df["pfam_side"].astype(str)
    key_df = key_df.drop_duplicates()

    wrote_header = False
    total_in = 0
    total_out = 0

    dtype_map = {
        "global_row_index": "Int64",
        "pfam_id": "string",
        "pfam_side": "string",
        "segment_rank": "Int64",
    }

    out_path = Path(random_out_csv)
    if out_path.exists():
        out_path.unlink()

    for chunk in pd.read_csv(random_csv, chunksize=chunksize, low_memory=False, dtype=dtype_map):
        ensure_cols(chunk, ["global_row_index", "pfam_id", "pfam_side", "segment_rank"], "random-csv chunk")
        total_in += len(chunk)
        chunk["pfam_id"] = chunk["pfam_id"].astype(str)
        chunk["pfam_side"] = chunk["pfam_side"].astype(str)

        matched = chunk.merge(key_df, on=["global_row_index", "pfam_id", "pfam_side", "segment_rank"], how="inner")
        total_out += len(matched)

        if len(matched) > 0:
            matched.to_csv(random_out_csv, index=False, mode="a", header=not wrote_header)
            wrote_header = True

    return total_in, total_out


def process_chunk(rows_chunk: List[Tuple[int, Dict[str, Any]]], args: argparse.Namespace, cache: PreprocessCache) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    immediate_results: List[Dict[str, Any]] = []
    random_rows: List[Dict[str, Any]] = []
    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    segment_jobs, precheck_results = _iter_segment_jobs(rows_chunk, args)
    immediate_results.extend(precheck_results)

    preprocess_workers = max(1, int(getattr(args, "preprocess_jobs", 0) or 0))
    if preprocess_workers <= 0:
        preprocess_workers = max(1, min(16, int(args.jobs)))

    preprocess_mode = safe_str(getattr(args, "preprocess_mode", "process")).lower()
    built_results: List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = []

    if segment_jobs:
        if preprocess_workers == 1 or len(segment_jobs) == 1:
            for job in segment_jobs:
                built_results.append(_preprocess_segment_job(job, args, cache))
        elif preprocess_mode == "thread":
            with ThreadPoolExecutor(max_workers=preprocess_workers) as ex:
                futures = [ex.submit(_preprocess_segment_job, job, args, cache) for job in segment_jobs]
                for fut in as_completed(futures):
                    built_results.append(fut.result())
        else:
            job_batches = _split_evenly(segment_jobs, preprocess_workers)
            args_dict = _args_to_worker_dict(args)
            with ProcessPoolExecutor(max_workers=min(preprocess_workers, len(job_batches))) as ex:
                futures = [ex.submit(_preprocess_job_batch_worker, batch, args_dict) for batch in job_batches]
                for fut in as_completed(futures):
                    built_results.extend(fut.result())

    for task, fail in built_results:
        if fail is not None:
            immediate_results.append(fail)
            continue
        if task is None:
            continue
        gk = task.pop("group_key")
        group = groups.get(gk)
        if group is None:
            group = {
                "pdb_path": task["pdb_path"],
                "chain_id": task["chain_id"],
                "atom_mode": task["atom_mode"],
                "cutoff": task["cutoff"],
                "exclude_near": task["exclude_near"],
                "include_hetatm": task["include_hetatm"],
                "long_range_threshold": task["long_range_threshold"],
                "token_to_unp": task["token_to_unp"],
                "cache_dir": str(args.cache_dir) if args.cache_dir else None,
                "random_samples": int(getattr(args, "random_samples", 0) or 0),
                "tasks": [],
            }
            groups[gk] = group
        group["tasks"].append({
            "fragment_tokens": task["fragment_tokens"],
            "long_range_threshold": task["long_range_threshold"],
            "meta": task["meta"],
        })

    if not groups:
        return immediate_results, random_rows

    group_payloads = list(groups.values())
    max_workers = max(1, int(args.jobs))
    if max_workers == 1 or len(group_payloads) == 1:
        for payload in group_payloads:
            out = process_group_worker(payload)
            immediate_results.extend(out.get("results", []))
            random_rows.extend(out.get("random_rows", []))
        return immediate_results, random_rows

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_group_worker, payload) for payload in group_payloads]
        for fut in as_completed(futures):
            out = fut.result()
            immediate_results.extend(out.get("results", []))
            random_rows.extend(out.get("random_rows", []))
    return immediate_results, random_rows

def iter_csv_rows(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            yield idx, row


def run_batch_from_global(args: argparse.Namespace, out_csv_override: Optional[Path] = None, random_csv_override: Optional[Path] = None) -> Path:
    global_csv = Path(args.global_score_csv)
    out_csv = Path(out_csv_override) if out_csv_override else (Path(args.out_csv) if args.out_csv else global_csv.parent / f"{global_csv.stem}.contact_order_results.csv")
    fieldnames = fixed_fieldnames()
    init_output_csv(out_csv, fieldnames)
    random_csv = None
    random_fieldnames = fixed_random_fieldnames()
    if int(getattr(args, "random_samples", 0) or 0) > 0:
        random_csv = Path(random_csv_override) if random_csv_override else (Path(args.random_out_csv) if safe_str(getattr(args, "random_out_csv", "")) else out_csv.with_name(out_csv.stem + ".random.csv"))
        init_output_csv(random_csv, random_fieldnames)
    fail_csv = Path(args.fail_csv) if safe_str(getattr(args, "fail_csv", "")) else None
    if fail_csv:
        init_output_csv(fail_csv, fieldnames)

    sifts_cache_dir = ensure_dir(Path(args.cache_dir) / "sifts") if args.cache_dir else None
    cache = PreprocessCache(Path(args.pfam_dir), sifts_cache_dir=sifts_cache_dir)

    start_idx = max(1, int(args.row_start))
    end_idx = int(args.row_end) if args.row_end else None
    chunk_size = max(1, int(args.chunk_size))

    rows_chunk: List[Tuple[int, Dict[str, Any]]] = []
    seen_rows = 0
    written = 0
    t0 = time.time()

    for row_index, row in iter_csv_rows(global_csv):
        if row_index < start_idx:
            continue
        if end_idx is not None and row_index > end_idx:
            break
        rows_chunk.append((row_index, row))
        seen_rows += 1
        if len(rows_chunk) >= chunk_size:
            results, random_rows = process_chunk(rows_chunk, args, cache)
            ok_written = append_results_csv(out_csv, fieldnames, results)
            random_written = append_random_csv(random_csv, random_fieldnames, random_rows) if random_csv else 0
            if fail_csv:
                append_results_csv(fail_csv, fieldnames, [r for r in results if safe_str(r.get("status")) != "OK"])
            fail_count = sum(1 for r in results if safe_str(r.get("status")) == "FAIL")
            skip_count = sum(1 for r in results if safe_str(r.get("status")) == "SKIP")
            written += ok_written
            print(f"[rows {rows_chunk[0][0]}-{rows_chunk[-1][0]}] outputs={len(results)} ok_written={ok_written} fail={fail_count} skip={skip_count} random_written={random_written} written_total={written} elapsed={time.time()-t0:.1f}s", file=sys.stderr)
            rows_chunk = []

    if rows_chunk:
        results, random_rows = process_chunk(rows_chunk, args, cache)
        ok_written = append_results_csv(out_csv, fieldnames, results)
        random_written = append_random_csv(random_csv, random_fieldnames, random_rows) if random_csv else 0
        if fail_csv:
            append_results_csv(fail_csv, fieldnames, [r for r in results if safe_str(r.get("status")) != "OK"])
        fail_count = sum(1 for r in results if safe_str(r.get("status")) == "FAIL")
        skip_count = sum(1 for r in results if safe_str(r.get("status")) == "SKIP")
        written += ok_written
        print(f"[rows {rows_chunk[0][0]}-{rows_chunk[-1][0]}] outputs={len(results)} ok_written={ok_written} fail={fail_count} skip={skip_count} random_written={random_written} written_total={written} elapsed={time.time()-t0:.1f}s", file=sys.stderr)

    print(f"[DONE] rows_processed={seen_rows} ok_outputs_written={written} elapsed={time.time()-t0:.1f}s", file=sys.stderr)
    return out_csv


# ========================================================
# CLI
# ========================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merged pipeline: run contact-order computation, then filter actual/random outputs by HMM coverage ratio."
    )
    p.add_argument("--pfam-dir", required=True)
    p.add_argument("--global-score-csv", required=True)

    # legacy raw-output compatibility
    p.add_argument("--out-csv", default="", help="Optional raw actual output CSV before coverage filtering.")
    p.add_argument("--random-out-csv", default="", help="Final filtered random output CSV. Also used as raw random output if --raw-random-out-csv is not provided.")

    # final filtered outputs
    p.add_argument("--actual-out-csv", required=True, help="Final filtered actual output CSV after coverage filtering.")
    p.add_argument("--raw-actual-out-csv", default="", help="Optional raw actual output CSV before coverage filtering.")
    p.add_argument("--raw-random-out-csv", default="", help="Optional raw random output CSV before coverage filtering.")
    p.add_argument("--summary-txt", default="", help="Optional summary text path.")

    p.add_argument("--pfam-id", default="", help="Optional PFxxxxx filter. If omitted, process both Main_HMM/Sub_HMM families in each row.")
    p.add_argument("--row-start", type=int, default=1)
    p.add_argument("--row-end", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=2000, help="Number of global rows to preprocess per chunk.")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Number of worker processes for chain groups.")
    p.add_argument("--preprocess-jobs", type=int, default=0, help="Number of workers for HHM/seed/SIFTS preprocessing. Default: min(16, jobs).")
    p.add_argument("--preprocess-mode", choices=["process", "thread"], default="process", help="Parallel mode for preprocessing. Default: process.")
    p.add_argument("--cache-dir", default="", help="Optional directory for persistent SIFTS and per-chain contact caches.")
    p.add_argument("--fail-csv", default="", help="Optional path to write FAIL/SKIP rows separately. By default, failed rows are not written anywhere except logs.")
    p.add_argument("--random-samples", type=int, default=1000, help="Number of same-chain same-length random windows to sample per successful result. Set 0 to disable.")

    seq = p.add_argument_group("Target seed sequence selection")
    seq.add_argument("--seed-seq-name", default="")
    seq.add_argument("--accession", default="")
    seq.add_argument("--seed-start", default=None)
    seq.add_argument("--seed-end", default=None)
    seq.add_argument("--auto-pick-seed", action="store_true")

    pdb = p.add_argument_group("Experimental structure override")
    pdb.add_argument("--pdb", default="")
    pdb.add_argument("--chain", default="")

    co = p.add_argument_group("Contact-order parameters")
    co.add_argument("--atom-mode", choices=["beta", "alpha", "heavy"], default="beta")
    co.add_argument("--cutoff", type=float, default=8.0)
    co.add_argument("--exclude-near", type=int, default=2)
    co.add_argument("--long-range-threshold", type=int, default=12)
    co.add_argument("--include-hetatm", action="store_true")

    # merged recheck_filter arguments
    p.add_argument("--min-coverage", type=float, default=0.8, help="Minimum allowed coverage ratio")
    p.add_argument("--max-coverage", type=float, default=1.2, help="Maximum allowed coverage ratio")
    p.add_argument("--ok-only", action="store_true", help="If set, only keep rows with status == OK before applying coverage filter")
    p.add_argument("--filter-chunksize", type=int, default=500000, help="Chunk size for streaming random CSV filtering")

    p.add_argument("--json-summary", action="store_true")
    return p.parse_args()




def main() -> None:
    args = parse_args()
    args.cache_dir = safe_str(args.cache_dir) or ""

    actual_out_csv = Path(args.actual_out_csv)
    actual_out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.raw_actual_out_csv:
        raw_actual_csv = Path(args.raw_actual_out_csv)
    elif args.out_csv:
        raw_actual_csv = Path(args.out_csv)
    else:
        raw_actual_csv = actual_out_csv.with_name(actual_out_csv.stem + ".raw.csv")
    raw_actual_csv.parent.mkdir(parents=True, exist_ok=True)

    filtered_random_csv: Optional[Path] = None
    raw_random_csv: Optional[Path] = None
    if int(getattr(args, "random_samples", 0) or 0) > 0:
        if not safe_str(args.random_out_csv):
            raise ValueError("--random-out-csv is required when --random-samples > 0")
        filtered_random_csv = Path(args.random_out_csv)
        filtered_random_csv.parent.mkdir(parents=True, exist_ok=True)

        if args.raw_random_out_csv:
            raw_random_csv = Path(args.raw_random_out_csv)
        else:
            raw_random_csv = filtered_random_csv.with_name(filtered_random_csv.stem + ".raw.csv")
        raw_random_csv.parent.mkdir(parents=True, exist_ok=True)

    out_csv = run_batch_from_global(args, out_csv_override=raw_actual_csv, random_csv_override=raw_random_csv)
    print(f"[OK] Raw batch output written to: {out_csv}")

    actual_all, actual_keep = load_and_filter_actual(
        str(raw_actual_csv),
        ok_only=args.ok_only,
        min_cov=args.min_coverage,
        max_cov=args.max_coverage,
    )
    actual_keep.to_csv(actual_out_csv, index=False)

    summary_lines = []
    summary_lines.append("Merged contact_order + HMM coverage filter summary")
    summary_lines.append("=" * 60)
    summary_lines.append(f"raw actual csv: {raw_actual_csv}")
    summary_lines.append(f"filtered actual csv: {actual_out_csv}")
    summary_lines.append(f"min_coverage: {args.min_coverage}")
    summary_lines.append(f"max_coverage: {args.max_coverage}")
    summary_lines.append(f"ok_only: {args.ok_only}")
    summary_lines.append(f"actual input rows: {len(actual_all)}")
    summary_lines.append(f"actual kept rows: {len(actual_keep)}")

    if len(actual_all) > 0:
        summary_lines.append(f"keep fraction: {len(actual_keep) / len(actual_all):.6f}")
        summary_lines.append(
            "coverage ratio min/median/max before filter: "
            f"{actual_all['coverage_ratio'].min():.6f} / "
            f"{actual_all['coverage_ratio'].median():.6f} / "
            f"{actual_all['coverage_ratio'].max():.6f}"
        )

    if len(actual_keep) > 0:
        summary_lines.append(
            "coverage ratio min/median/max after filter: "
            f"{actual_keep['coverage_ratio'].min():.6f} / "
            f"{actual_keep['coverage_ratio'].median():.6f} / "
            f"{actual_keep['coverage_ratio'].max():.6f}"
        )
        if "target_seq_len" in actual_keep.columns:
            summary_lines.append("kept target_seq_len counts:")
            vc = actual_keep["target_seq_len"].value_counts().sort_index()
            for k, v in vc.items():
                summary_lines.append(f"  len={k}: {v}")

    if filtered_random_csv is not None and raw_random_csv is not None:
        total_in, total_out = filter_random(
            str(raw_random_csv),
            str(filtered_random_csv),
            actual_keep,
            chunksize=args.filter_chunksize,
        )
        summary_lines.append(f"raw random csv: {raw_random_csv}")
        summary_lines.append(f"filtered random csv: {filtered_random_csv}")
        summary_lines.append(f"random input rows scanned: {total_in}")
        summary_lines.append(f"random kept rows: {total_out}")

    if args.summary_txt:
        summary_path = Path(args.summary_txt)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))

    if args.json_summary:
        summary = {
            "raw_actual_csv": str(raw_actual_csv),
            "actual_out_csv": str(actual_out_csv),
            "raw_random_csv": str(raw_random_csv) if raw_random_csv is not None else "",
            "random_out_csv": str(filtered_random_csv) if filtered_random_csv is not None else "",
            "global_score_csv": args.global_score_csv,
            "pfam_dir": args.pfam_dir,
            "pfam_id_filter": args.pfam_id,
            "row_start": args.row_start,
            "row_end": args.row_end,
            "atom_mode": args.atom_mode,
            "cutoff": args.cutoff,
            "exclude_near": args.exclude_near,
            "jobs": args.jobs,
            "chunk_size": args.chunk_size,
            "cache_dir": args.cache_dir,
            "random_samples": args.random_samples,
            "min_coverage": args.min_coverage,
            "max_coverage": args.max_coverage,
            "ok_only": args.ok_only,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
