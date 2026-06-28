#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate per-position DSSP secondary structure and RSA for the same representative
sequences used in the ESM2 entropy analysis, then merge with per_position_entropy.tsv.

Purpose:
1. Use the ESM2 representative sequence headers:
   PF00001|uid=P31388|type=full_rep_domain|len=278|msa_span=1-722|src_header=P31388.2_5HT6R_RAT/43-320

2. Use the structure-availability table generated previously:
   esm_representative_structure_availability.tsv

3. Map ESM2 domain position to structure residue index:
   structure_residue_index = src_start + position - 1

4. Run DSSP on the corresponding AF/EXP structure.

5. Output per-position SS/RSA:
   header, position, structure_residue_index, ss8, ss_class, rsa, rsa_bin, plddt_ca, ...

6. Merge with ESM2 entropy:
   per_position_entropy.tsv + per-position SS/RSA

7. Produce secondary-structure-stratified entropy comparison:
   Helix: fragment vs background
   Sheet: fragment vs background
   Coil : fragment vs background

Important:
- This script does NOT calculate RMSD. RMSD requires paired structures/fragments.
- This script calculates secondary structure and RSA for ESM2 representative sequences.
"""

from __future__ import annotations

import argparse
import os
import re
import gzip
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from Bio import PDB, BiopythonWarning
from Bio.PDB.DSSP import DSSP


# ==============================================================================
# Configuration
# ==============================================================================

FRAGDOM_FILE = Path(
    "CHANGE_ME"
)

STRUCTURE_AVAILABILITY_TSV = Path(
    "CHANGE_ME"
)

ENTROPY_TSV = Path(
    "CHANGE_ME"
)

OUT_DIR = Path(
    "CHANGE_ME"
)

PER_POSITION_SS_RSA_TSV = OUT_DIR / "esm_representative_per_position_ss_rsa.tsv"
RECORD_STATUS_TSV = OUT_DIR / "esm_representative_ss_rsa_record_status.tsv"
MERGED_ENTROPY_SS_RSA_TSV = OUT_DIR / "esm_entropy_with_ss_rsa.tsv"

SS_STRATIFIED_SUMMARY_TSV = OUT_DIR / "ss_stratified_entropy_summary.tsv"
SS_STRATIFIED_PER_RECORD_DELTA_TSV = OUT_DIR / "ss_stratified_per_record_delta.tsv"
SS_STRATIFIED_PER_RECORD_DELTA_SUMMARY_TSV = OUT_DIR / "ss_stratified_per_record_delta_summary.tsv"

RSA_STRATIFIED_SUMMARY_TSV = OUT_DIR / "rsa_stratified_entropy_summary.tsv"
SS_RSA_STRATIFIED_SUMMARY_TSV = OUT_DIR / "ss_rsa_stratified_entropy_summary.tsv"

STRUCTURE_COVERAGE_SUBSET_SUMMARY_TSV = OUT_DIR / "structure_covered_entropy_global_summary.tsv"

NUM_WORKERS = 60
MAX_INFLIGHT = NUM_WORKERS * 4

PER_POSITION_WRITE_CHUNK_SIZE = 100_000
RECORD_WRITE_CHUNK_SIZE = 2_000

OVERWRITE_OUTPUT = True

# DSSP / quality parameters
DSSP_EXE = shutil.which("mkdssp") or shutil.which("dssp")

MIN_DOMAIN_FOUND_RATIO = 0.80

# For AF structures, pLDDT is stored in B-factor.
# We do not discard low-pLDDT rows from the per-position output,
# but analysis_ok will be False if pLDDT < this threshold.
MIN_AF_CA_PLDDT_FOR_ANALYSIS = 70.0

# EXP structures do not use pLDDT.
AF_TYPES = {"AF", "AlphaFold"}



def str_to_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def configure_runtime(args: argparse.Namespace) -> None:
    global FRAGDOM_FILE, STRUCTURE_AVAILABILITY_TSV, ENTROPY_TSV, OUT_DIR
    global PER_POSITION_SS_RSA_TSV, RECORD_STATUS_TSV, MERGED_ENTROPY_SS_RSA_TSV
    global SS_STRATIFIED_SUMMARY_TSV, SS_STRATIFIED_PER_RECORD_DELTA_TSV
    global SS_STRATIFIED_PER_RECORD_DELTA_SUMMARY_TSV, RSA_STRATIFIED_SUMMARY_TSV
    global SS_RSA_STRATIFIED_SUMMARY_TSV, STRUCTURE_COVERAGE_SUBSET_SUMMARY_TSV
    global NUM_WORKERS, MAX_INFLIGHT, PER_POSITION_WRITE_CHUNK_SIZE, RECORD_WRITE_CHUNK_SIZE
    global OVERWRITE_OUTPUT, DSSP_EXE, MIN_DOMAIN_FOUND_RATIO, MIN_AF_CA_PLDDT_FOR_ANALYSIS

    FRAGDOM_FILE = Path(args.fragdom_file)
    STRUCTURE_AVAILABILITY_TSV = Path(args.structure_availability_tsv)
    ENTROPY_TSV = Path(args.entropy_tsv)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PER_POSITION_SS_RSA_TSV = OUT_DIR / "esm_representative_per_position_ss_rsa.tsv"
    RECORD_STATUS_TSV = OUT_DIR / "esm_representative_ss_rsa_record_status.tsv"
    MERGED_ENTROPY_SS_RSA_TSV = OUT_DIR / "esm_entropy_with_ss_rsa.tsv"
    SS_STRATIFIED_SUMMARY_TSV = OUT_DIR / "ss_stratified_entropy_summary.tsv"
    SS_STRATIFIED_PER_RECORD_DELTA_TSV = OUT_DIR / "ss_stratified_per_record_delta.tsv"
    SS_STRATIFIED_PER_RECORD_DELTA_SUMMARY_TSV = OUT_DIR / "ss_stratified_per_record_delta_summary.tsv"
    RSA_STRATIFIED_SUMMARY_TSV = OUT_DIR / "rsa_stratified_entropy_summary.tsv"
    SS_RSA_STRATIFIED_SUMMARY_TSV = OUT_DIR / "ss_rsa_stratified_entropy_summary.tsv"
    STRUCTURE_COVERAGE_SUBSET_SUMMARY_TSV = OUT_DIR / "structure_covered_entropy_global_summary.tsv"

    NUM_WORKERS = int(args.workers)
    MAX_INFLIGHT = int(args.max_inflight) if args.max_inflight is not None else NUM_WORKERS * 4
    PER_POSITION_WRITE_CHUNK_SIZE = int(args.per_position_write_chunk_size)
    RECORD_WRITE_CHUNK_SIZE = int(args.record_write_chunk_size)
    OVERWRITE_OUTPUT = bool(args.overwrite_output)
    DSSP_EXE = args.dssp_exe or shutil.which("mkdssp") or shutil.which("dssp")
    MIN_DOMAIN_FOUND_RATIO = float(args.min_domain_found_ratio)
    MIN_AF_CA_PLDDT_FOR_ANALYSIS = float(args.min_af_ca_plddt_for_analysis)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-position DSSP secondary structure/RSA and merge with ESM2 entropy."
    )
    parser.add_argument("--fragdom-file", default=str(FRAGDOM_FILE))
    parser.add_argument("--structure-availability-tsv", default=str(STRUCTURE_AVAILABILITY_TSV))
    parser.add_argument("--entropy-tsv", default=str(ENTROPY_TSV))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max-inflight", type=int, default=None)
    parser.add_argument("--per-position-write-chunk-size", type=int, default=PER_POSITION_WRITE_CHUNK_SIZE)
    parser.add_argument("--record-write-chunk-size", type=int, default=RECORD_WRITE_CHUNK_SIZE)
    parser.add_argument("--overwrite-output", type=str_to_bool, default=OVERWRITE_OUTPUT)
    parser.add_argument("--dssp-exe", default=None, help="Path to mkdssp/dssp. Default: search PATH.")
    parser.add_argument("--min-domain-found-ratio", type=float, default=MIN_DOMAIN_FOUND_RATIO)
    parser.add_argument("--min-af-ca-plddt-for-analysis", type=float, default=MIN_AF_CA_PLDDT_FOR_ANALYSIS)
    return parser


# ==============================================================================
# Basic parsers
# ==============================================================================

def open_text_maybe_gz(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def iter_headers_from_fragdom(path: Path) -> Iterable[str]:
    with open_text_maybe_gz(path) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith(">"):
                yield line[1:]


def parse_header(header: str) -> Dict[str, Optional[str]]:
    """
    Parse:
    PF00001|uid=P31388|type=full_rep_domain|len=278|msa_span=1-722|src_header=P31388.2_5HT6R_RAT/43-320
    """
    parts = header.split("|")
    pfam_id = parts[0].strip()

    info: Dict[str, Optional[str]] = {
        "header": header,
        "pfam_id": pfam_id,
        "uid": None,
        "type": None,
        "len": None,
        "msa_span": None,
        "src_header": None,
        "src_start": None,
        "src_end": None,
    }

    for part in parts[1:]:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        info[key.strip()] = val.strip()

    src_header = info.get("src_header")
    if src_header:
        m = re.search(r"/(\d+)-(\d+)\s*$", src_header)
        if m:
            info["src_start"] = m.group(1)
            info["src_end"] = m.group(2)

    return info


def load_fragdom_header_table(path: Path) -> pd.DataFrame:
    rows = []
    for header in tqdm(iter_headers_from_fragdom(path), desc="Loading fragdom headers", unit="record"):
        rows.append(parse_header(header))

    df = pd.DataFrame(rows)

    if "len" in df.columns:
        df["len"] = pd.to_numeric(df["len"], errors="coerce").astype("Int64")

    if "src_start" in df.columns:
        df["src_start"] = pd.to_numeric(df["src_start"], errors="coerce").astype("Int64")
    if "src_end" in df.columns:
        df["src_end"] = pd.to_numeric(df["src_end"], errors="coerce").astype("Int64")

    return df


def ensure_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


# ==============================================================================
# DSSP helpers
# ==============================================================================

def _decompress_if_needed(path: Path) -> Tuple[Path, Optional[Path]]:
    """
    Return:
      usable_path, temp_path_to_cleanup

    If input is .gz, decompress to temporary file.
    Otherwise return original path.
    """
    if not str(path).endswith(".gz"):
        return path, None

    suffix = "".join(path.suffixes)
    if suffix.endswith(".pdb.gz"):
        tmp_suffix = ".pdb"
    elif suffix.endswith(".cif.gz"):
        tmp_suffix = ".cif"
    else:
        tmp_suffix = ".tmp"

    fd, tmp_name = tempfile.mkstemp(prefix="structure_unzip_", suffix=tmp_suffix)
    os.close(fd)
    tmp_path = Path(tmp_name)

    with gzip.open(path, "rb") as fin, open(tmp_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)

    return tmp_path, tmp_path


def parse_structure_file(path: Path):
    """
    Parse PDB or mmCIF.
    """
    lower = path.name.lower()
    if lower.endswith(".cif") or lower.endswith(".mmcif"):
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(PERMISSIVE=1, QUIET=True)

    return parser.get_structure("X", str(path))


def _call_dssp(model, structure_path: Path):
    if DSSP_EXE is None:
        raise RuntimeError("mkdssp/dssp not found in PATH. Please install mkdssp.")

    try:
        return DSSP(model, str(structure_path), dssp=DSSP_EXE)
    except TypeError:
        return DSSP(model, str(structure_path))


def add_dummy_cryst1_if_missing(pdb_path: Path) -> None:
    with open(pdb_path, "r", errors="ignore") as f:
        lines = f.readlines()

    has_cryst1 = any(line.startswith("CRYST1") for line in lines)

    if not has_cryst1:
        dummy_cryst1 = (
            "CRYST1  100.000  100.000  100.000  "
            "90.00  90.00  90.00 P 1           1\n"
        )
        with open(pdb_path, "w") as f:
            f.write(dummy_cryst1)
            f.writelines(lines)


def build_dssp_safely(structure_path_raw: str):
    """
    Robust DSSP wrapper:
    1. Try original structure.
    2. If failed, save cleaned PDB with PDBIO and add CRYST1 if missing.
    """
    raw_path = Path(structure_path_raw)

    if not raw_path.exists():
        raise RuntimeError(f"Structure file not found: {raw_path}")

    usable_path, tmp_decompressed = _decompress_if_needed(raw_path)
    tmp_cleaned = None

    try:
        structure = parse_structure_file(usable_path)
        model = structure[0]

        try:
            dssp = _call_dssp(model, usable_path)
            chain_ids = {k[0] for k in dssp.keys()}
            if len(dssp) > 0:
                return structure, dssp, chain_ids, "Original"
        except Exception:
            pass

        fd, tmp_name = tempfile.mkstemp(prefix="dssp_clean_", suffix=".pdb")
        os.close(fd)
        tmp_cleaned = Path(tmp_name)

        io = PDB.PDBIO()
        io.set_structure(structure)
        io.save(str(tmp_cleaned))

        add_dummy_cryst1_if_missing(tmp_cleaned)

        dssp = _call_dssp(model, tmp_cleaned)
        chain_ids = {k[0] for k in dssp.keys()}

        if len(dssp) == 0:
            raise RuntimeError("DSSP returned zero residues")

        return structure, dssp, chain_ids, "Cleaned_PDBIO"

    except Exception as e:
        raise RuntimeError(f"DSSP failed on both Original and Cleaned: {e}")

    finally:
        for tmp in [tmp_decompressed, tmp_cleaned]:
            if tmp is not None and Path(tmp).exists():
                try:
                    Path(tmp).unlink()
                except Exception:
                    pass


def first_chain_id_from_structure(structure) -> Optional[str]:
    try:
        model = structure[0]
        chains = list(model.get_chains())
        return chains[0].id if chains else None
    except Exception:
        return None


def parse_exp_chain_from_filename(filename: str) -> Optional[str]:
    """
    Historical EXP naming often looks like:
      UID_exp_xxx_A.pdb
    If no reliable chain can be parsed, return None and let DSSP chain fallback handle it.
    """
    base = os.path.basename(filename)

    m = re.search(r"_([A-Za-z0-9]+)\.pdb(?:\.gz)?$", base)
    if m:
        return m.group(1)

    return None


def choose_chain(
    structure,
    dssp_chain_ids: set,
    structure_type: str,
    structure_path: str,
) -> Optional[str]:
    """
    AF usually uses chain A.
    EXP may use a chain encoded in filename, otherwise fallback.
    """
    desired = None

    if structure_type in AF_TYPES:
        desired = "A"
    else:
        desired = parse_exp_chain_from_filename(structure_path)

    if desired and desired in dssp_chain_ids:
        return desired

    if "A" in dssp_chain_ids:
        return "A"

    if len(dssp_chain_ids) == 1:
        return list(dssp_chain_ids)[0]

    first = first_chain_id_from_structure(structure)
    if first and first in dssp_chain_ids:
        return first

    if dssp_chain_ids:
        return sorted(dssp_chain_ids)[0]

    return None


def get_dssp_entry(dssp, chain_id: str, resseq: int):
    """
    Try exact key first, then ignore insertion code if necessary.
    DSSP key format:
      (chain_id, (' ', resseq, ' '))
    """
    exact_key = (chain_id, (" ", int(resseq), " "))
    if exact_key in dssp:
        return dssp[exact_key]

    for key in dssp.keys():
        if key[0] == chain_id and key[1][1] == int(resseq):
            return dssp[key]

    return None


def build_residue_lookup(structure, chain_id: str) -> Dict[int, object]:
    """
    Map residue number -> Bio.PDB residue object.
    Ignores insertion code; if duplicates exist, first one is kept.
    """
    lookup = {}

    try:
        model = structure[0]
        if chain_id not in model:
            return lookup

        chain = model[chain_id]

        for res in chain:
            hetflag, resseq, icode = res.id
            if hetflag != " ":
                continue
            if int(resseq) not in lookup:
                lookup[int(resseq)] = res

    except Exception:
        return lookup

    return lookup


def get_ca_bfactor(residue) -> float:
    if residue is None:
        return np.nan
    if "CA" not in residue:
        return np.nan
    try:
        return float(residue["CA"].get_bfactor())
    except Exception:
        return np.nan


def ss8_to_ss_class(ss8: str) -> str:
    if ss8 in {"H", "G", "I"}:
        return "Helix"
    if ss8 in {"E", "B"}:
        return "Sheet"
    return "Coil"


def rsa_to_bin(rsa: float) -> str:
    if pd.isna(rsa):
        return "Unknown"
    if rsa < 0.20:
        return "Buried"
    if rsa < 0.50:
        return "Intermediate"
    return "Exposed"


# ==============================================================================
# Per-record processing
# ==============================================================================

PER_POSITION_COLUMNS = [
    "header",
    "pfam_id",
    "uid",
    "domain_len",
    "src_start",
    "src_end",
    "position",
    "structure_residue_index",
    "aa_from_entropy",
    "structure_type",
    "structure_path",
    "structure_file",
    "chain_used",
    "dssp_mode",
    "ss8",
    "ss_class",
    "rsa",
    "rsa_bin",
    "plddt_ca",
    "position_status",
    "record_status",
    "domain_found_ratio",
    "analysis_ok",
]

RECORD_STATUS_COLUMNS = [
    "header",
    "pfam_id",
    "uid",
    "domain_len",
    "src_start",
    "src_end",
    "structure_type",
    "structure_path",
    "structure_file",
    "chain_used",
    "dssp_mode",
    "n_domain_positions",
    "n_dssp_found_positions",
    "domain_found_ratio",
    "n_analysis_ok_positions",
    "record_status",
    "error_message",
]


def make_empty_record_status(row: dict, status: str, error: str = "") -> Dict[str, object]:
    return {
        "header": row.get("header", ""),
        "pfam_id": row.get("pfam_id", ""),
        "uid": row.get("uid", ""),
        "domain_len": row.get("len", np.nan),
        "src_start": row.get("src_start", np.nan),
        "src_end": row.get("src_end", np.nan),
        "structure_type": row.get("best_structure_type", ""),
        "structure_path": row.get("best_structure_path", ""),
        "structure_file": os.path.basename(str(row.get("best_structure_path", ""))),
        "chain_used": "NA",
        "dssp_mode": "NA",
        "n_domain_positions": row.get("len", np.nan),
        "n_dssp_found_positions": 0,
        "domain_found_ratio": 0.0,
        "n_analysis_ok_positions": 0,
        "record_status": status,
        "error_message": error,
    }


def process_one_representative(row: dict) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Process one ESM2 representative sequence:
    - build DSSP
    - map domain positions to structure residue positions
    - return per-position SS/RSA rows + one record status row
    """
    warnings.simplefilter("ignore", BiopythonWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")

    header = str(row.get("header", ""))
    pfam_id = str(row.get("pfam_id", ""))
    uid = str(row.get("uid", ""))

    try:
        domain_len = int(row.get("len"))
    except Exception:
        return [], make_empty_record_status(row, "BAD_DOMAIN_LEN", "Cannot parse len")

    try:
        src_start = int(row.get("src_start"))
        src_end = int(row.get("src_end"))
    except Exception:
        return [], make_empty_record_status(row, "BAD_SRC_SPAN", "Cannot parse src_start/src_end")

    expected_len = src_end - src_start + 1
    if expected_len != domain_len:
        # This does not necessarily mean fatal, but coordinate mapping becomes suspicious.
        # We treat it as fatal to avoid silent coordinate mistakes.
        return [], make_empty_record_status(
            row,
            "DOMAIN_LEN_SRC_SPAN_MISMATCH",
            f"len={domain_len}, src_span={src_start}-{src_end}, src_span_len={expected_len}",
        )

    structure_path = str(row.get("best_structure_path", ""))
    structure_type = str(row.get("best_structure_type", ""))

    if not structure_path or not os.path.exists(structure_path):
        return [], make_empty_record_status(row, "STRUCTURE_FILE_NOT_FOUND", structure_path)

    try:
        structure, dssp, dssp_chain_ids, dssp_mode = build_dssp_safely(structure_path)
    except Exception as e:
        return [], make_empty_record_status(row, "DSSP_FAIL", str(e))

    chain_used = choose_chain(structure, dssp_chain_ids, structure_type, structure_path)

    if chain_used is None:
        return [], make_empty_record_status(row, "CHAIN_NOT_FOUND", "No usable DSSP chain")

    residue_lookup = build_residue_lookup(structure, chain_used)

    per_pos_rows: List[Dict[str, object]] = []

    found_positions = 0
    analysis_ok_positions = 0

    # Optional sequence from entropy/fragdom is not passed here.
    # aa_from_entropy is filled later after merging with entropy.
    for pos in range(1, domain_len + 1):
        structure_resi = src_start + pos - 1

        dssp_entry = get_dssp_entry(dssp, chain_used, structure_resi)
        residue = residue_lookup.get(structure_resi)
        plddt_ca = get_ca_bfactor(residue)

        ss8 = "-"
        ss_class = "Unknown"
        rsa = np.nan
        rsa_bin = "Unknown"
        position_status = "RES_NOT_FOUND"

        if dssp_entry is not None:
            found_positions += 1

            ss8 = dssp_entry[2]
            if ss8 == " ":
                ss8 = "-"

            ss_class = ss8_to_ss_class(ss8)

            try:
                rsa = float(dssp_entry[3])
                if rsa > 1.0:
                    rsa = 1.0
                if rsa < 0.0:
                    rsa = 0.0
            except Exception:
                rsa = np.nan

            rsa_bin = rsa_to_bin(rsa)

            if structure_type in AF_TYPES:
                if pd.isna(plddt_ca):
                    position_status = "NO_CA_PLDDT"
                elif plddt_ca < MIN_AF_CA_PLDDT_FOR_ANALYSIS:
                    position_status = "LOW_AF_PLDDT"
                else:
                    position_status = "OK"
            else:
                position_status = "OK"

        per_pos_rows.append(
            {
                "header": header,
                "pfam_id": pfam_id,
                "uid": uid,
                "domain_len": domain_len,
                "src_start": src_start,
                "src_end": src_end,
                "position": pos,
                "structure_residue_index": structure_resi,
                "aa_from_entropy": "",
                "structure_type": structure_type,
                "structure_path": structure_path,
                "structure_file": os.path.basename(structure_path),
                "chain_used": chain_used,
                "dssp_mode": dssp_mode,
                "ss8": ss8,
                "ss_class": ss_class,
                "rsa": rsa,
                "rsa_bin": rsa_bin,
                "plddt_ca": plddt_ca,
                "position_status": position_status,
                "record_status": "PENDING",
                "domain_found_ratio": np.nan,
                "analysis_ok": False,
            }
        )

    domain_found_ratio = found_positions / domain_len if domain_len > 0 else 0.0

    if domain_found_ratio < MIN_DOMAIN_FOUND_RATIO:
        record_status = f"LOW_DOMAIN_COVERAGE({found_positions}/{domain_len})"
    else:
        record_status = "OK"

    for r in per_pos_rows:
        r["record_status"] = record_status
        r["domain_found_ratio"] = round(domain_found_ratio, 6)

        # Only OK DSSP records and OK positions enter analysis.
        r["analysis_ok"] = (
            record_status == "OK"
            and r["position_status"] == "OK"
            and r["ss_class"] in {"Helix", "Sheet", "Coil"}
            and r["rsa_bin"] in {"Buried", "Intermediate", "Exposed"}
        )

        if isinstance(r["rsa"], float) and np.isfinite(r["rsa"]):
            r["rsa"] = round(float(r["rsa"]), 6)

        if isinstance(r["plddt_ca"], float) and np.isfinite(r["plddt_ca"]):
            r["plddt_ca"] = round(float(r["plddt_ca"]), 3)

    analysis_ok_positions = sum(1 for r in per_pos_rows if r["analysis_ok"])

    record_row = {
        "header": header,
        "pfam_id": pfam_id,
        "uid": uid,
        "domain_len": domain_len,
        "src_start": src_start,
        "src_end": src_end,
        "structure_type": structure_type,
        "structure_path": structure_path,
        "structure_file": os.path.basename(structure_path),
        "chain_used": chain_used,
        "dssp_mode": dssp_mode,
        "n_domain_positions": domain_len,
        "n_dssp_found_positions": found_positions,
        "domain_found_ratio": round(domain_found_ratio, 6),
        "n_analysis_ok_positions": analysis_ok_positions,
        "record_status": record_status,
        "error_message": "",
    }

    return per_pos_rows, record_row


def write_rows(rows: List[Dict[str, object]], out_path: Path, columns: List[str]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)

    for col in columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[columns]

    mode = "a" if out_path.exists() else "w"
    header = mode == "w"
    df.to_csv(out_path, sep="\t", mode=mode, header=header, index=False)


# ==============================================================================
# Entropy + structure merge and summaries
# ==============================================================================

def summarize_group(values: pd.Series) -> Dict[str, object]:
    s = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        return {
            "n_positions": 0,
            "mean_entropy": np.nan,
            "median_entropy": np.nan,
            "std_entropy": np.nan,
            "q25_entropy": np.nan,
            "q75_entropy": np.nan,
            "min_entropy": np.nan,
            "max_entropy": np.nan,
        }

    return {
        "n_positions": int(s.shape[0]),
        "mean_entropy": float(s.mean()),
        "median_entropy": float(s.median()),
        "std_entropy": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "q25_entropy": float(s.quantile(0.25)),
        "q75_entropy": float(s.quantile(0.75)),
        "min_entropy": float(s.min()),
        "max_entropy": float(s.max()),
    }


def build_stratified_summary(
    df: pd.DataFrame,
    strat_cols: List[str],
    out_path: Path,
) -> pd.DataFrame:
    rows = []

    grouped = df.groupby(strat_cols + ["is_fragment"], dropna=False)

    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_dict = dict(zip(strat_cols + ["is_fragment"], keys))
        group_name = "Fragment" if bool(key_dict["is_fragment"]) else "Background"

        row = {k: v for k, v in key_dict.items() if k != "is_fragment"}
        row["group"] = group_name
        row.update(summarize_group(sub["entropy"]))
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, sep="\t", index=False)
    return out_df


def build_per_record_delta_by_ss(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each header and ss_class:
      delta = mean_entropy_fragment - mean_entropy_background
    """
    rows = []

    for (header, ss_class), sub in df.groupby(["header", "ss_class"]):
        frag = sub.loc[sub["is_fragment"], "entropy"].replace([np.inf, -np.inf], np.nan).dropna()
        bg = sub.loc[~sub["is_fragment"], "entropy"].replace([np.inf, -np.inf], np.nan).dropna()

        if frag.empty or bg.empty:
            continue

        rows.append(
            {
                "header": header,
                "ss_class": ss_class,
                "n_fragment_positions": int(frag.shape[0]),
                "n_background_positions": int(bg.shape[0]),
                "mean_entropy_fragment": float(frag.mean()),
                "median_entropy_fragment": float(frag.median()),
                "mean_entropy_background": float(bg.mean()),
                "median_entropy_background": float(bg.median()),
                "delta_mean_fragment_minus_background": float(frag.mean() - bg.mean()),
                "delta_median_fragment_minus_background": float(frag.median() - bg.median()),
            }
        )

    delta_df = pd.DataFrame(rows)

    if delta_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "ss_class",
                "n_records",
                "mean_delta_mean",
                "median_delta_mean",
                "std_delta_mean",
                "negative_delta_fraction",
                "mean_delta_median",
                "median_delta_median",
            ]
        )
        return delta_df, summary_df

    summary_rows = []
    for ss_class, sub in delta_df.groupby("ss_class"):
        d_mean = sub["delta_mean_fragment_minus_background"].dropna()
        d_median = sub["delta_median_fragment_minus_background"].dropna()

        summary_rows.append(
            {
                "ss_class": ss_class,
                "n_records": int(sub.shape[0]),
                "mean_delta_mean": float(d_mean.mean()) if not d_mean.empty else np.nan,
                "median_delta_mean": float(d_mean.median()) if not d_mean.empty else np.nan,
                "std_delta_mean": float(d_mean.std(ddof=1)) if d_mean.shape[0] > 1 else 0.0,
                "negative_delta_fraction": float((d_mean < 0).mean()) if not d_mean.empty else np.nan,
                "mean_delta_median": float(d_median.mean()) if not d_median.empty else np.nan,
                "median_delta_median": float(d_median.median()) if not d_median.empty else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    return delta_df, summary_df


def save_ss_density_plots(df: pd.DataFrame) -> None:
    for ss_class in ["Helix", "Sheet", "Coil"]:
        sub = df[df["ss_class"] == ss_class].copy()

        frag = sub.loc[sub["is_fragment"], "entropy"].replace([np.inf, -np.inf], np.nan).dropna()
        bg = sub.loc[~sub["is_fragment"], "entropy"].replace([np.inf, -np.inf], np.nan).dropna()

        if frag.empty or bg.empty:
            continue

        plt.figure(figsize=(9, 6))
        plt.hist(frag.to_numpy(), bins=80, alpha=0.55, density=True, label="Fragment")
        plt.hist(bg.to_numpy(), bins=80, alpha=0.55, density=True, label="Background")
        plt.xlabel("ESM2 masked-token entropy")
        plt.ylabel("Density")
        plt.title(f"ESM2 entropy within {ss_class}: fragment vs background")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"entropy_density_fragment_vs_background_{ss_class}.png", dpi=200)
        plt.close()


def save_delta_boxplot(delta_df: pd.DataFrame) -> None:
    if delta_df.empty:
        return

    order = ["Helix", "Sheet", "Coil"]
    data = []
    labels = []

    for ss in order:
        vals = (
            delta_df.loc[delta_df["ss_class"] == ss, "delta_mean_fragment_minus_background"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
        )
        if vals.size > 0:
            data.append(vals)
            labels.append(ss)

    if not data:
        return

    plt.figure(figsize=(8, 6))
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel("Secondary-structure class")
    plt.ylabel("Per-record delta: fragment mean entropy - background mean entropy")
    plt.title("Per-record ESM2 entropy delta within each SS class")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ss_stratified_per_record_delta_boxplot.png", dpi=200)
    plt.close()


def run_merge_and_summaries() -> None:
    print("[INFO] Reading entropy table...")
    entropy = pd.read_csv(ENTROPY_TSV, sep="\t", low_memory=False)

    required_entropy_cols = {"header", "position", "is_fragment", "entropy"}
    missing = required_entropy_cols - set(entropy.columns)
    if missing:
        raise ValueError(f"Missing columns in entropy file: {sorted(missing)}")

    entropy["position"] = pd.to_numeric(entropy["position"], errors="coerce").astype("Int64")
    entropy["is_fragment"] = ensure_bool_series(entropy["is_fragment"])
    entropy["entropy"] = pd.to_numeric(entropy["entropy"], errors="coerce")

    if "wt_aa" in entropy.columns:
        entropy = entropy.rename(columns={"wt_aa": "aa_from_entropy"})

    keep_entropy_cols = [
        c for c in [
            "header",
            "position",
            "aa_from_entropy",
            "is_fragment",
            "fragment_coverage_count",
            "share_count",
            "entropy",
            "aa_mass_20aa",
            "wt_prob_20aa_renorm",
        ]
        if c in entropy.columns
    ]

    entropy = entropy[keep_entropy_cols].copy()

    print("[INFO] Reading per-position SS/RSA table...")
    ssrsa = pd.read_csv(PER_POSITION_SS_RSA_TSV, sep="\t", low_memory=False)

    ssrsa["position"] = pd.to_numeric(ssrsa["position"], errors="coerce").astype("Int64")
    ssrsa["analysis_ok"] = ensure_bool_series(ssrsa["analysis_ok"])

    # Avoid duplicate aa_from_entropy after merge.
    if "aa_from_entropy" in ssrsa.columns:
        ssrsa = ssrsa.drop(columns=["aa_from_entropy"])

    print("[INFO] Merging entropy with SS/RSA...")
    merged = entropy.merge(
        ssrsa,
        on=["header", "position"],
        how="inner",
    )

    merged["entropy"] = pd.to_numeric(merged["entropy"], errors="coerce")
    merged = merged.replace([np.inf, -np.inf], np.nan)

    merged.to_csv(MERGED_ENTROPY_SS_RSA_TSV, sep="\t", index=False)
    print(f"[INFO] Wrote merged table: {MERGED_ENTROPY_SS_RSA_TSV}")

    # For analysis, use only reliable mapped positions.
    analysis_df = merged[
        (merged["analysis_ok"])
        & (merged["entropy"].notna())
        & (merged["ss_class"].isin(["Helix", "Sheet", "Coil"]))
        & (merged["rsa_bin"].isin(["Buried", "Intermediate", "Exposed"]))
    ].copy()

    # Global summary on structure-covered subset.
    global_rows = []
    for is_frag, group_name in [(True, "Fragment"), (False, "Background")]:
        sub = analysis_df[analysis_df["is_fragment"] == is_frag]
        row = {"group": group_name}
        row.update(summarize_group(sub["entropy"]))
        global_rows.append(row)

    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(STRUCTURE_COVERAGE_SUBSET_SUMMARY_TSV, sep="\t", index=False)

    # SS-stratified summary.
    ss_summary = build_stratified_summary(
        analysis_df,
        ["ss_class"],
        SS_STRATIFIED_SUMMARY_TSV,
    )

    # RSA-stratified summary.
    rsa_summary = build_stratified_summary(
        analysis_df,
        ["rsa_bin"],
        RSA_STRATIFIED_SUMMARY_TSV,
    )

    # SS + RSA stratified summary.
    ss_rsa_summary = build_stratified_summary(
        analysis_df,
        ["ss_class", "rsa_bin"],
        SS_RSA_STRATIFIED_SUMMARY_TSV,
    )

    # Per-record delta within SS class.
    delta_df, delta_summary_df = build_per_record_delta_by_ss(analysis_df)

    delta_df.to_csv(SS_STRATIFIED_PER_RECORD_DELTA_TSV, sep="\t", index=False)
    delta_summary_df.to_csv(SS_STRATIFIED_PER_RECORD_DELTA_SUMMARY_TSV, sep="\t", index=False)

    save_ss_density_plots(analysis_df)
    save_delta_boxplot(delta_df)

    print("\n[STRUCTURE-COVERED GLOBAL SUMMARY]")
    print(global_df.to_string(index=False))

    print("\n[SS-STRATIFIED SUMMARY]")
    print(ss_summary.to_string(index=False))

    print("\n[SS-STRATIFIED PER-RECORD DELTA SUMMARY]")
    print(delta_summary_df.to_string(index=False))

    print("\n[DONE] Summary files:")
    print(f"  - {MERGED_ENTROPY_SS_RSA_TSV}")
    print(f"  - {STRUCTURE_COVERAGE_SUBSET_SUMMARY_TSV}")
    print(f"  - {SS_STRATIFIED_SUMMARY_TSV}")
    print(f"  - {SS_STRATIFIED_PER_RECORD_DELTA_TSV}")
    print(f"  - {SS_STRATIFIED_PER_RECORD_DELTA_SUMMARY_TSV}")
    print(f"  - {RSA_STRATIFIED_SUMMARY_TSV}")
    print(f"  - {SS_RSA_STRATIFIED_SUMMARY_TSV}")
    print(f"  - {OUT_DIR / 'entropy_density_fragment_vs_background_Helix.png'}")
    print(f"  - {OUT_DIR / 'entropy_density_fragment_vs_background_Sheet.png'}")
    print(f"  - {OUT_DIR / 'entropy_density_fragment_vs_background_Coil.png'}")
    print(f"  - {OUT_DIR / 'ss_stratified_per_record_delta_boxplot.png'}")


# ==============================================================================
# Main pipeline
# ==============================================================================

def prepare_tasks() -> List[Dict[str, object]]:
    if not FRAGDOM_FILE.exists():
        raise FileNotFoundError(f"FRAGDOM_FILE not found: {FRAGDOM_FILE}")

    if not STRUCTURE_AVAILABILITY_TSV.exists():
        raise FileNotFoundError(f"STRUCTURE_AVAILABILITY_TSV not found: {STRUCTURE_AVAILABILITY_TSV}")

    print("[INFO] Loading fragdom representative headers...")
    header_df = load_fragdom_header_table(FRAGDOM_FILE)

    print("[INFO] Loading structure availability table...")
    avail = pd.read_csv(STRUCTURE_AVAILABILITY_TSV, sep="\t", low_memory=False)

    required_avail_cols = {"header", "status", "best_structure_path"}
    missing = required_avail_cols - set(avail.columns)
    if missing:
        raise ValueError(f"Missing columns in structure availability table: {sorted(missing)}")

    # Merge header parsed coordinates back to availability table.
    # Availability table may already contain these columns, but header_df is the source of truth.
    keep_header_cols = [
        "header",
        "pfam_id",
        "uid",
        "type",
        "len",
        "msa_span",
        "src_header",
        "src_start",
        "src_end",
    ]

    avail = avail.drop(
        columns=[
            c for c in ["pfam_id", "uid", "type", "len", "msa_span", "src_header", "src_start", "src_end"]
            if c in avail.columns
        ],
        errors="ignore",
    )

    df = avail.merge(
        header_df[keep_header_cols],
        on="header",
        how="left",
    )

    # Keep only records with structure.
    df = df[df["status"].isin(["AF_FOUND", "EXP_FOUND"])].copy()

    df["best_structure_path"] = df["best_structure_path"].astype(str)

    if "best_structure_type" not in df.columns:
        df["best_structure_type"] = np.where(df["status"] == "AF_FOUND", "AF", "EXP")

    # Validate required coordinate fields.
    df["len"] = pd.to_numeric(df["len"], errors="coerce")
    df["src_start"] = pd.to_numeric(df["src_start"], errors="coerce")
    df["src_end"] = pd.to_numeric(df["src_end"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["len", "src_start", "src_end", "best_structure_path"])
    after = len(df)

    print(f"[INFO] Structure-covered records before coordinate filtering: {before}")
    print(f"[INFO] Structure-covered records after coordinate filtering:  {after}")

    tasks = df.to_dict("records")
    return tasks


def run_dssp_pipeline() -> None:
    if DSSP_EXE is None:
        raise RuntimeError("mkdssp/dssp not found in PATH. Please install DSSP first.")

    if OVERWRITE_OUTPUT:
        for p in [
            PER_POSITION_SS_RSA_TSV,
            RECORD_STATUS_TSV,
            MERGED_ENTROPY_SS_RSA_TSV,
            SS_STRATIFIED_SUMMARY_TSV,
            SS_STRATIFIED_PER_RECORD_DELTA_TSV,
            SS_STRATIFIED_PER_RECORD_DELTA_SUMMARY_TSV,
            RSA_STRATIFIED_SUMMARY_TSV,
            SS_RSA_STRATIFIED_SUMMARY_TSV,
            STRUCTURE_COVERAGE_SUBSET_SUMMARY_TSV,
        ]:
            if p.exists():
                p.unlink()

        for p in OUT_DIR.glob("entropy_density_fragment_vs_background_*.png"):
            p.unlink()
        for p in OUT_DIR.glob("ss_stratified_per_record_delta_boxplot.png"):
            p.unlink()

    tasks = prepare_tasks()
    total = len(tasks)

    print(f"[INFO] DSSP executable: {DSSP_EXE}")
    print(f"[INFO] Starting DSSP/RSA calculation on {total} ESM2 representative structures")
    print(f"[INFO] Workers={NUM_WORKERS}, MIN_DOMAIN_FOUND_RATIO={MIN_DOMAIN_FOUND_RATIO}")

    per_pos_buf: List[Dict[str, object]] = []
    record_buf: List[Dict[str, object]] = []

    submitted = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        inflight = {}

        def submit_one(task):
            nonlocal submitted
            fut = executor.submit(process_one_representative, task)
            inflight[fut] = 1
            submitted += 1

        task_iter = iter(tasks)

        for _ in range(min(MAX_INFLIGHT, total)):
            try:
                submit_one(next(task_iter))
            except StopIteration:
                break

        with tqdm(total=total, desc="ESM representative DSSP/RSA", unit="record") as pbar:
            while inflight:
                for fut in as_completed(list(inflight.keys()), timeout=None):
                    inflight.pop(fut, None)

                    try:
                        pos_rows, rec_row = fut.result()
                    except Exception as e:
                        pos_rows = []
                        rec_row = {
                            "header": "",
                            "pfam_id": "",
                            "uid": "",
                            "domain_len": np.nan,
                            "src_start": np.nan,
                            "src_end": np.nan,
                            "structure_type": "",
                            "structure_path": "",
                            "structure_file": "",
                            "chain_used": "NA",
                            "dssp_mode": "NA",
                            "n_domain_positions": np.nan,
                            "n_dssp_found_positions": 0,
                            "domain_found_ratio": 0.0,
                            "n_analysis_ok_positions": 0,
                            "record_status": "WORKER_EXCEPTION",
                            "error_message": str(e),
                        }

                    if pos_rows:
                        per_pos_buf.extend(pos_rows)
                    record_buf.append(rec_row)

                    pbar.update(1)

                    try:
                        submit_one(next(task_iter))
                    except StopIteration:
                        pass

                    if len(per_pos_buf) >= PER_POSITION_WRITE_CHUNK_SIZE:
                        write_rows(per_pos_buf, PER_POSITION_SS_RSA_TSV, PER_POSITION_COLUMNS)
                        per_pos_buf = []

                    if len(record_buf) >= RECORD_WRITE_CHUNK_SIZE:
                        write_rows(record_buf, RECORD_STATUS_TSV, RECORD_STATUS_COLUMNS)
                        record_buf = []

                    break

    if per_pos_buf:
        write_rows(per_pos_buf, PER_POSITION_SS_RSA_TSV, PER_POSITION_COLUMNS)

    if record_buf:
        write_rows(record_buf, RECORD_STATUS_TSV, RECORD_STATUS_COLUMNS)

    print(f"[INFO] Wrote per-position SS/RSA table: {PER_POSITION_SS_RSA_TSV}")
    print(f"[INFO] Wrote record status table:       {RECORD_STATUS_TSV}")


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_runtime(args)
    run_dssp_pipeline()
    run_merge_and_summaries()


if __name__ == "__main__":
    main()