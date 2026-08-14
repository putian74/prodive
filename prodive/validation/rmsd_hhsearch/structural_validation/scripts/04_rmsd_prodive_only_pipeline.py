#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProDive-only structural validation pipeline.

This script runs the local ProDive-only RMSD workflow:

1. Select representative sequence/structure candidates from an overlap-check CSV.
2. Compute C-alpha RMSD from the selected structure ranges with PyMOL.
3. Convert the RMSD output to the standardized format used by downstream plotting scripts.

Typical run:

python 04_rmsd_prodive_only_pipeline.py \
  --input-csv /path/to/subset_data_Top_100%_overlap_check_dual_20.csv \
  --pfam-dir /path/to/PfamA_seed \
  --output-dir /path/to/prodive_only_rmsd_results \
  --novel-filter true \
  --num-workers 40
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import traceback
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import PDB

# ==============================================================================
# Global worker configuration
# ==============================================================================

WORKER_CONFIG: Dict[str, Any] = {
    "pfam_dir": "",
    "max_search_depth": 200,
    "max_len_diff": 5,
    "min_mean_plddt": 70.0,
    "max_head_tail_pae": 10.0,
    "min_candidate_pure_len": 3,
}


def init_worker(config: Dict[str, Any]) -> None:
    global WORKER_CONFIG
    WORKER_CONFIG = dict(config)


# ==============================================================================
# Generic helpers
# ==============================================================================


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def normalize_is_novel(v: Any) -> str:
    """Return one of True / False / NA as strings."""
    if pd.isna(v):
        return "NA"
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return "True"
    if s in {"false", "0", "no", "n"}:
        return "False"
    return "NA"


def is_true_like(v: Any) -> bool:
    return normalize_is_novel(v) == "True"


def is_false_like(v: Any) -> bool:
    return normalize_is_novel(v) == "False"


def filter_dataframe_by_novel(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode:
      all   -> keep all rows
      true  -> keep Is_Novel == True
      false -> keep Is_Novel == False
      na    -> keep missing/unrecognized Is_Novel
    """
    mode = str(mode).strip().lower()
    valid_modes = {"all", "true", "false", "na"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid novel filter: {mode}. Valid values: {sorted(valid_modes)}")

    if "Is_Novel" not in df.columns:
        if mode == "all":
            return df.copy()
        raise ValueError("Column 'Is_Novel' is missing, but --novel-filter is not 'all'.")

    status = df["Is_Novel"].apply(normalize_is_novel)
    if mode == "all":
        return df.copy()
    if mode == "true":
        return df.loc[status == "True"].copy()
    if mode == "false":
        return df.loc[status == "False"].copy()
    return df.loc[status == "NA"].copy()


def print_is_novel_summary(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(f"Total rows: {len(df)}")
    if "Is_Novel" not in df.columns:
        print("Column 'Is_Novel' not found.")
        return
    vals = df["Is_Novel"].apply(normalize_is_novel)
    counts = vals.value_counts(dropna=False)
    print(f"Is_Novel=True : {int(counts.get('True', 0))}")
    print(f"Is_Novel=False: {int(counts.get('False', 0))}")
    print(f"Is_Novel=NA   : {int(counts.get('NA', 0))}")


def make_range(start: Any, end: Any) -> str:
    s = safe_int(start, default=None)
    e = safe_int(end, default=None)
    if s is None or e is None:
        return ""
    return f"{s}-{e}"


# ==============================================================================
# Segment parsing
# ==============================================================================


def parse_segment(seg_val: Any) -> List[int]:
    """
    Accept:
      "(1, 30)"
      "1-30"
      "[1, 30]"
      "1,30"
    Return [start, end].
    """
    s = str(seg_val).strip()

    m = re.match(r"^\(?\s*(\d+)\s*-\s*(\d+)\s*\)?$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return [a, b] if a <= b else [b, a]

    s2 = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    parts = [x.strip() for x in s2.split(",") if x.strip()]
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        a, b = int(parts[0]), int(parts[1])
        return [a, b] if a <= b else [b, a]

    raise ValueError(f"Unrecognized segment format: {seg_val}")


def parse_segments_from_row(row: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """Prefer Main_Segment/Sub_Segment. Fallback to Sub_Segments_Details."""
    if "Main_Segment" in row and "Sub_Segment" in row:
        return parse_segment(row["Main_Segment"]), parse_segment(row["Sub_Segment"])

    details = str(row.get("Sub_Segments_Details", "")).strip()
    m = re.search(r"(\d+)-(\d+)\s*->\s*(\d+)-(\d+)", details)
    if m:
        main_seg = [int(m.group(1)), int(m.group(2))]
        sub_seg = [int(m.group(3)), int(m.group(4))]
        return main_seg, sub_seg

    raise ValueError("Cannot parse segment info from row")


# ==============================================================================
# HHM -> MSA mapping
# ==============================================================================


@lru_cache(maxsize=None)
def extract_hmm_to_msa_map_cached(hhm_path: str) -> Tuple[Dict[int, int], str]:
    """
    Parse HHM mapping. The original project used the last numeric field as MSA column.
    This function keeps that behavior and returns hmm_index -> msa_column.
    """
    mapping: Dict[int, int] = {}
    if not os.path.exists(hhm_path):
        return {}, "HHM_FILE_MISSING"
    try:
        with open(hhm_path, "r", errors="ignore") as f:
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
            return {}, "HHM_CONTENT_EMPTY"
        return mapping, "OK"
    except Exception as exc:
        return {}, f"HHM_PARSE_ERROR:{exc}"


# ==============================================================================
# Alignment parsing and representative candidate generation
# ==============================================================================


@lru_cache(maxsize=None)
def load_alignment_sequences_cached(aln_path: str) -> Tuple[Dict[str, str], str]:
    if not os.path.exists(aln_path):
        return {}, "ALIGN_FILE_MISSING"

    seqs: Dict[str, str] = {}
    current_header: Optional[str] = None

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
                    elif current_header:
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
    except Exception as exc:
        return {}, f"ALIGN_READ_ERROR:{exc}"


def extract_uid_from_header(header: str) -> str:
    uid_match = re.search(
        r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})",
        header,
    )
    if uid_match:
        return uid_match.group(1)
    return header.split("/")[0].split()[0]


def ungap(seq: str) -> str:
    return (seq or "").replace(".", "").replace("-", "")


def get_candidates_from_alignment(
    aln_path: str,
    msa_start_col: int,
    msa_end_col: int,
    expected_len: int,
) -> Tuple[List[Dict[str, Any]], str]:
    if not os.path.exists(aln_path):
        return [], "ALIGN_FILE_MISSING"

    seqs, status = load_alignment_sequences_cached(aln_path)
    if not seqs:
        return [], status

    candidates: List[Dict[str, Any]] = []
    idx_start = int(msa_start_col) - 1
    idx_end = int(msa_end_col)

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

        if len(frag_pure) < int(WORKER_CONFIG["min_candidate_pure_len"]):
            continue

        prefix = seq[:idx_start]
        local_res_before = len(ungap(prefix))
        pdb_start = global_offset + local_res_before
        pdb_end = pdb_start + len(frag_pure) - 1

        candidates.append({
            "uid": uid,
            "msa_start": msa_start_col,
            "msa_end": msa_end_col,
            "pdb_start": pdb_start,
            "pdb_end": pdb_end,
            "seq_len": len(frag_pure),
            "gaps": gaps,
            "len_diff": abs(len(frag_pure) - expected_len),
        })

    candidates.sort(key=lambda x: (x["len_diff"], x["gaps"]))
    return candidates[: int(WORKER_CONFIG["max_search_depth"])], "OK"


# ==============================================================================
# Structure availability and quality checks
# ==============================================================================


def first_chain_id(structure: Any) -> Optional[str]:
    model = structure[0]
    chains = list(model.get_chains())
    return chains[0].id if chains else None


@lru_cache(maxsize=None)
def check_af_quality_cached(pdb_path: str, start_res: int, end_res: int) -> Tuple[bool, float, str]:
    mean_plddt = 0.0

    try:
        parser = PDB.PDBParser(PERMISSIVE=1, QUIET=True)
        structure = parser.get_structure("X", pdb_path)
        model = structure[0]
        chain_id = "A" if "A" in model else first_chain_id(structure)
        if chain_id is None:
            return False, 0.0, "NO_CHAIN"

        plddt_vals: List[float] = []
        for res in model[chain_id]:
            if start_res <= res.id[1] <= end_res:
                if "CA" in res:
                    plddt_vals.append(float(res["CA"].get_bfactor()))
        if not plddt_vals:
            return False, 0.0, "NO_ATOMS"

        mean_plddt = float(np.mean(plddt_vals))
        if mean_plddt < float(WORKER_CONFIG["min_mean_plddt"]):
            return False, mean_plddt, "LOW_PLDDT"
    except Exception:
        return False, 0.0, "PARSE_ERR"

    json_path = pdb_path.replace(".pdb", ".json")
    if not os.path.exists(json_path):
        base = pdb_path[:-4] if pdb_path.lower().endswith(".pdb") else pdb_path
        jsons = glob.glob(f"{base}*.json")
        json_path = jsons[0] if jsons else None

    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            block = None
            if isinstance(data, list) and data and isinstance(data[0], dict):
                block = data[0]
            elif isinstance(data, dict):
                block = data

            if block is not None:
                pae = block.get("predicted_aligned_error", None)
                if pae is None:
                    pae = block.get("pae", None)
                if pae is not None:
                    idx_s = max(0, start_res - 1)
                    idx_e = min(len(pae) - 1, end_res - 1)
                    pae_val = float(pae[idx_s][idx_e])
                    if pae_val > float(WORKER_CONFIG["max_head_tail_pae"]):
                        return False, mean_plddt, f"HIGH_PAE({pae_val:.2f})"
        except Exception:
            pass

    return True, mean_plddt, "OK"


def parse_exp_chain(filename: str) -> str:
    m = re.search(r"_([A-Za-z0-9]+)\.pdb$", filename)
    return m.group(1) if m else "A"


def find_best_structure_hybrid(family_dir: str, candidates: List[Dict[str, Any]], max_len_diff: int) -> Optional[Dict[str, Any]]:
    """
    Candidate list is already sorted by len_diff/gaps.
    EXP structures are preferred over AF structures for each candidate.
    """
    for cand in candidates:
        if cand["len_diff"] > max_len_diff:
            break

        uid = cand["uid"]
        s, e = int(cand["pdb_start"]), int(cand["pdb_end"])

        exp_files = sorted(glob.glob(os.path.join(family_dir, f"{uid}_exp_*.pdb")))
        if exp_files:
            best_pdb = exp_files[0]
            chain = parse_exp_chain(os.path.basename(best_pdb))
            return {
                "Selected_UID": uid,
                "Selected_Seq_Start": s,
                "Selected_Seq_End": e,
                "Selected_Seq_Len": cand["seq_len"],
                "Len_Diff": cand["len_diff"],
                "Source": "EXP",
                "PDB_ID": os.path.basename(best_pdb),
                "Chain": chain,
                "Selection_Status": "OK",
                "Selection_Reason": "EXP_SELECTED",
            }

        af_files = sorted(glob.glob(os.path.join(family_dir, f"AF-{uid}-F1-model*.pdb")), reverse=True)
        if af_files:
            af_pdb = af_files[0]
            passed, mean_plddt, msg = check_af_quality_cached(af_pdb, s, e)
            if passed:
                return {
                    "Selected_UID": uid,
                    "Selected_Seq_Start": s,
                    "Selected_Seq_End": e,
                    "Selected_Seq_Len": cand["seq_len"],
                    "Len_Diff": cand["len_diff"],
                    "Source": "AF",
                    "PDB_ID": os.path.basename(af_pdb),
                    "Chain": "A",
                    "Selection_Status": "OK",
                    "Selection_Reason": f"AF_SELECTED_pLDDT_{mean_plddt:.2f}",
                }
    return None


# ==============================================================================
# Representative-selection task processing
# ==============================================================================


def build_empty_selection_result(hmm: Any, seg_start: Any, seg_end: Any) -> Dict[str, Any]:
    hmm_len = np.nan
    if pd.notna(seg_start) and pd.notna(seg_end):
        try:
            hmm_len = int(seg_end) - int(seg_start) + 1
        except Exception:
            hmm_len = np.nan

    return {
        "HMM": hmm,
        "Seg_Start": seg_start,
        "Seg_End": seg_end,
        "HMM_Len": hmm_len,
        "MSA_Start": np.nan,
        "MSA_End": np.nan,
        "Selected_UID": "",
        "Selected_Seq_Start": np.nan,
        "Selected_Seq_End": np.nan,
        "Selected_Seq_Len": np.nan,
        "Len_Diff": np.nan,
        "Source": "",
        "PDB_ID": "",
        "Chain": "",
        "Selection_Status": "Not_Processed",
        "Selection_Reason": "",
    }


def process_unique_selection_task(task: Dict[str, Any]) -> Dict[str, Any]:
    hmm = task["HMM"]
    seg_start = int(task["Seg_Start"])
    seg_end = int(task["Seg_End"])
    result = build_empty_selection_result(hmm, seg_start, seg_end)

    try:
        if hmm is None or str(hmm).strip() == "" or str(hmm).lower() == "nan":
            result["Selection_Status"] = "Map_Fail"
            result["Selection_Reason"] = "HMM_MISSING"
            return result

        pfam_dir = str(WORKER_CONFIG["pfam_dir"])
        family_dir = os.path.join(pfam_dir, str(hmm))
        hhm_path = os.path.join(family_dir, f"{hmm}.hhm")

        mapping, map_status = extract_hmm_to_msa_map_cached(hhm_path)
        if not mapping or seg_start not in mapping or seg_end not in mapping:
            result["Selection_Status"] = "Map_Fail"
            result["Selection_Reason"] = map_status
            return result

        msa_start = mapping[seg_start]
        msa_end = mapping[seg_end]
        result["MSA_Start"] = msa_start
        result["MSA_End"] = msa_end

        aln_path = os.path.join(family_dir, f"{hmm}.fas")
        if not os.path.exists(aln_path):
            aln_path = os.path.join(family_dir, f"{hmm}.sto")

        cands, cand_status = get_candidates_from_alignment(aln_path, msa_start, msa_end, int(result["HMM_Len"]))
        if not cands:
            result["Selection_Status"] = "No_Candidates"
            result["Selection_Reason"] = cand_status
            return result

        struct = find_best_structure_hybrid(family_dir, cands, int(WORKER_CONFIG["max_len_diff"]))
        if struct is None:
            result["Selection_Status"] = "No_Struct"
            result["Selection_Reason"] = f"No_valid_structure_within_len_diff_{WORKER_CONFIG['max_len_diff']}"
            return result

        result.update(struct)
        return result

    except Exception as exc:
        result["Selection_Status"] = "Exception"
        result["Selection_Reason"] = str(exc)
        return result


def make_task_key(hmm: Any, seg: Sequence[int]) -> Tuple[str, int, int]:
    return (str(hmm), int(seg[0]), int(seg[1]))


def parse_rows_and_collect_unique_tasks(rows_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    unique: "OrderedDict[Tuple[str, int, int], Dict[str, Any]]" = OrderedDict()
    parsed_rows: List[Dict[str, Any]] = []

    for row in rows_data:
        row2 = dict(row)
        row2["_Main_Key"] = None
        row2["_Sub_Key"] = None
        row2["_Parse_Error"] = ""

        try:
            main_seg, sub_seg = parse_segments_from_row(row2)
            main_key = make_task_key(row2.get("Main_HMM"), main_seg)
            sub_key = make_task_key(row2.get("Sub_HMM"), sub_seg)

            row2["_Main_Key"] = main_key
            row2["_Sub_Key"] = sub_key

            if main_key not in unique:
                unique[main_key] = {"HMM": main_key[0], "Seg_Start": main_key[1], "Seg_End": main_key[2]}
            if sub_key not in unique:
                unique[sub_key] = {"HMM": sub_key[0], "Seg_Start": sub_key[1], "Seg_End": sub_key[2]}
        except Exception as exc:
            row2["_Parse_Error"] = str(exc)

        parsed_rows.append(row2)

    unique_tasks = list(unique.values())
    unique_tasks.sort(key=lambda x: (x["HMM"], x["Seg_Start"], x["Seg_End"]))
    return parsed_rows, unique_tasks


def build_task_result_map(unique_tasks: List[Dict[str, Any]], num_workers: int, config: Dict[str, Any]) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    task_result_map: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    if not unique_tasks:
        return task_result_map

    if num_workers <= 1:
        init_worker(config)
        for task in tqdm(unique_tasks, total=len(unique_tasks), desc="Unique selection tasks"):
            res = process_unique_selection_task(task)
            key = (str(res["HMM"]), int(res["Seg_Start"]), int(res["Seg_End"]))
            task_result_map[key] = res
        return task_result_map

    chunksize = max(1, min(200, len(unique_tasks) // (num_workers * 4) + 1))
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(config,)) as executor:
        results_iter = executor.map(process_unique_selection_task, unique_tasks, chunksize=chunksize)
        for res in tqdm(results_iter, total=len(unique_tasks), desc="Unique selection tasks"):
            key = (str(res["HMM"]), int(res["Seg_Start"]), int(res["Seg_End"]))
            task_result_map[key] = res
    return task_result_map


def build_output_row_base(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for prefix in ["Main", "Sub"]:
        out[f"{prefix}_HMM_Len"] = np.nan
        out[f"{prefix}_MSA_Start"] = np.nan
        out[f"{prefix}_MSA_End"] = np.nan
        out[f"{prefix}_Selected_UID"] = ""
        out[f"{prefix}_Selected_Seq_Start"] = np.nan
        out[f"{prefix}_Selected_Seq_End"] = np.nan
        out[f"{prefix}_Selected_Seq_Len"] = np.nan
        out[f"{prefix}_Len_Diff"] = np.nan
        out[f"{prefix}_Source"] = ""
        out[f"{prefix}_PDB_ID"] = ""
        out[f"{prefix}_Chain"] = ""
        out[f"{prefix}_Selection_Status"] = "Not_Processed"
        out[f"{prefix}_Selection_Reason"] = ""
    return out


def apply_side_result(out: Dict[str, Any], prefix: str, side_res: Dict[str, Any]) -> None:
    out[f"{prefix}_HMM_Len"] = side_res.get("HMM_Len", np.nan)
    out[f"{prefix}_MSA_Start"] = side_res.get("MSA_Start", np.nan)
    out[f"{prefix}_MSA_End"] = side_res.get("MSA_End", np.nan)
    out[f"{prefix}_Selected_UID"] = side_res.get("Selected_UID", "")
    out[f"{prefix}_Selected_Seq_Start"] = side_res.get("Selected_Seq_Start", np.nan)
    out[f"{prefix}_Selected_Seq_End"] = side_res.get("Selected_Seq_End", np.nan)
    out[f"{prefix}_Selected_Seq_Len"] = side_res.get("Selected_Seq_Len", np.nan)
    out[f"{prefix}_Len_Diff"] = side_res.get("Len_Diff", np.nan)
    out[f"{prefix}_Source"] = side_res.get("Source", "")
    out[f"{prefix}_PDB_ID"] = side_res.get("PDB_ID", "")
    out[f"{prefix}_Chain"] = side_res.get("Chain", "")
    out[f"{prefix}_Selection_Status"] = side_res.get("Selection_Status", "")
    out[f"{prefix}_Selection_Reason"] = side_res.get("Selection_Reason", "")


def write_dataframe_chunk(rows: List[Dict[str, Any]], output_csv: Path, output_columns: Sequence[str]) -> None:
    if not rows:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_chunk = pd.DataFrame(rows)
    for col in output_columns:
        if col not in df_chunk.columns:
            df_chunk[col] = np.nan
    df_chunk = df_chunk[list(output_columns)]
    mode = "w" if not output_csv.exists() else "a"
    df_chunk.to_csv(output_csv, mode=mode, header=(mode == "w"), index=False)


def run_selection(args: argparse.Namespace, selection_csv: Path) -> Path:
    if args.input_csv is None:
        raise ValueError("--input-csv is required for the selection step.")
    if args.pfam_dir is None:
        raise ValueError("--pfam-dir is required for the selection step.")

    input_csv = Path(args.input_csv)
    pfam_dir = Path(args.pfam_dir)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not pfam_dir.exists():
        raise FileNotFoundError(f"Pfam directory not found: {pfam_dir}")

    if selection_csv.exists() and args.overwrite:
        selection_csv.unlink()

    print(f"[Selection] Reading: {input_csv}")
    df_raw = pd.read_csv(input_csv, low_memory=False)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    print_is_novel_summary(df_raw, "[Raw input Is_Novel summary]")
    df = filter_dataframe_by_novel(df_raw, args.novel_filter)
    print_is_novel_summary(df, f"[Filtered input summary: novel_filter={args.novel_filter}]")

    if len(df) == 0:
        raise RuntimeError("No rows remain after novel filtering.")

    input_columns = list(df.columns)
    extra_columns = []
    for prefix in ["Main", "Sub"]:
        extra_columns.extend([
            f"{prefix}_HMM_Len", f"{prefix}_MSA_Start", f"{prefix}_MSA_End",
            f"{prefix}_Selected_UID", f"{prefix}_Selected_Seq_Start", f"{prefix}_Selected_Seq_End",
            f"{prefix}_Selected_Seq_Len", f"{prefix}_Len_Diff", f"{prefix}_Source", f"{prefix}_PDB_ID",
            f"{prefix}_Chain", f"{prefix}_Selection_Status", f"{prefix}_Selection_Reason",
        ])
    output_columns = input_columns + extra_columns

    rows_data = df.to_dict("records")
    print("\n[Selection] Parsing rows and collecting unique representative-selection tasks...")
    parsed_rows, unique_tasks = parse_rows_and_collect_unique_tasks(rows_data)
    print(f"Rows preserved for output: {len(parsed_rows)}")
    print(f"Unique internal (HMM, segment) tasks: {len(unique_tasks)}")

    worker_config = dict(WORKER_CONFIG)
    worker_config.update({
        "pfam_dir": str(pfam_dir),
        "max_search_depth": args.max_search_depth,
        "max_len_diff": args.max_len_diff,
        "min_mean_plddt": args.min_mean_plddt,
        "max_head_tail_pae": args.max_head_tail_pae,
        "min_candidate_pure_len": args.min_candidate_pure_len,
    })

    print(f"\n[Selection] Processing unique tasks with workers={args.num_workers}...")
    task_result_map = build_task_result_map(unique_tasks, args.num_workers, worker_config)

    print("\n[Selection] Assembling selected rows...")
    results_buf: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = defaultdict(int)

    for row in tqdm(parsed_rows, total=len(parsed_rows), desc="Assemble selection rows"):
        out = build_output_row_base(row)
        out.pop("_Main_Key", None)
        out.pop("_Sub_Key", None)
        parse_error = out.pop("_Parse_Error", "")

        if parse_error:
            out["Main_Selection_Status"] = "Exception"
            out["Main_Selection_Reason"] = parse_error
            out["Sub_Selection_Status"] = "Exception"
            out["Sub_Selection_Reason"] = parse_error
        else:
            main_res = task_result_map.get(row["_Main_Key"])
            sub_res = task_result_map.get(row["_Sub_Key"])

            if main_res is None:
                main_res = build_empty_selection_result(row.get("Main_HMM"), np.nan, np.nan)
                main_res["Selection_Status"] = "Exception"
                main_res["Selection_Reason"] = "MAIN_TASK_RESULT_MISSING"
            if sub_res is None:
                sub_res = build_empty_selection_result(row.get("Sub_HMM"), np.nan, np.nan)
                sub_res["Selection_Status"] = "Exception"
                sub_res["Selection_Reason"] = "SUB_TASK_RESULT_MISSING"

            apply_side_result(out, "Main", main_res)
            apply_side_result(out, "Sub", sub_res)

        pair_status = f"{out.get('Main_Selection_Status')}|{out.get('Sub_Selection_Status')}"
        status_counts[pair_status] += 1
        results_buf.append(out)

        if len(results_buf) >= args.write_chunk_size:
            write_dataframe_chunk(results_buf, selection_csv, output_columns)
            results_buf.clear()

    if results_buf:
        write_dataframe_chunk(results_buf, selection_csv, output_columns)

    print(f"\n[Selection] Saved: {selection_csv}")
    print("[Selection] Top selection-status pairs:")
    for k, v in sorted(status_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {k}: {v}")
    return selection_csv

# ==============================================================================
# RMSD calculation from selected representatives
# ==============================================================================


def find_structure_path_for_rmsd(structure_root: Optional[Path], pfam_dir: Optional[Path], family: Any, pdb_id: Any) -> str:
    family_s = safe_str(family)
    pdb_s = safe_str(pdb_id)
    if not family_s or not pdb_s:
        return ""

    candidates: List[Path] = []
    if structure_root is not None:
        candidates.extend([structure_root / family_s / pdb_s, structure_root / pdb_s])
    if pfam_dir is not None:
        candidates.append(pfam_dir / family_s / pdb_s)

    for p in candidates:
        if p.exists():
            return str(p)
    return ""


def resolve_chain(cmd: Any, obj: str, expected_chain: str) -> str:
    chains = cmd.get_chains(obj)
    if expected_chain and expected_chain in chains:
        return expected_chain
    if expected_chain and expected_chain.upper() in chains:
        return expected_chain.upper()
    if "A" in chains:
        return "A"
    if len(chains) == 1:
        return chains[0]
    return expected_chain


def process_rmsd_one_row(row: Dict[str, Any], cmd: Any, structure_root: Optional[Path], pfam_dir: Optional[Path], min_atoms: int, debug: bool = False) -> Dict[str, Any]:
    result = dict(row)
    result["RMSD_Status"] = "Not_Processed"
    result["RMSD"] = np.nan
    result["Aligned_Atoms"] = 0
    result["Used_PDB_A_Path"] = ""
    result["Used_PDB_B_Path"] = ""
    result["Used_Chain_A"] = ""
    result["Used_Chain_B"] = ""

    try:
        main_hmm = safe_str(row.get("Main_HMM"))
        sub_hmm = safe_str(row.get("Sub_HMM"))
        main_status = safe_str(row.get("Main_Selection_Status"))
        sub_status = safe_str(row.get("Sub_Selection_Status"))

        if main_status != "OK" or sub_status != "OK":
            result["RMSD_Status"] = f"Selection_Not_OK({main_status},{sub_status})"
            return result

        main_pdb_id = safe_str(row.get("Main_PDB_ID"))
        sub_pdb_id = safe_str(row.get("Sub_PDB_ID"))
        main_chain = safe_str(row.get("Main_Chain"))
        sub_chain = safe_str(row.get("Sub_Chain"))

        main_start = safe_int(row.get("Main_Selected_Seq_Start"))
        main_end = safe_int(row.get("Main_Selected_Seq_End"))
        sub_start = safe_int(row.get("Sub_Selected_Seq_Start"))
        sub_end = safe_int(row.get("Sub_Selected_Seq_End"))

        if None in [main_start, main_end, sub_start, sub_end]:
            result["RMSD_Status"] = "Missing_Selected_Range"
            return result
        if main_start > main_end or sub_start > sub_end:
            result["RMSD_Status"] = "Invalid_Selected_Range"
            return result

        pdb_a = find_structure_path_for_rmsd(structure_root, pfam_dir, main_hmm, main_pdb_id)
        pdb_b = find_structure_path_for_rmsd(structure_root, pfam_dir, sub_hmm, sub_pdb_id)
        result["Used_PDB_A_Path"] = pdb_a
        result["Used_PDB_B_Path"] = pdb_b

        if not pdb_a:
            result["RMSD_Status"] = "Missing_PDB_A"
            return result
        if not pdb_b:
            result["RMSD_Status"] = "Missing_PDB_B"
            return result

        cmd.reinitialize()
        cmd.load(pdb_a, "obj_a")
        cmd.load(pdb_b, "obj_b")

        real_chain_a = resolve_chain(cmd, "obj_a", main_chain)
        real_chain_b = resolve_chain(cmd, "obj_b", sub_chain)
        result["Used_Chain_A"] = real_chain_a
        result["Used_Chain_B"] = real_chain_b

        sel_a = f"obj_a and chain {real_chain_a} and resi {main_start}-{main_end} and name CA"
        sel_b = f"obj_b and chain {real_chain_b} and resi {sub_start}-{sub_end} and name CA"
        cmd.select("sel_a", sel_a)
        cmd.select("sel_b", sel_b)
        n_a = cmd.count_atoms("sel_a")
        n_b = cmd.count_atoms("sel_b")

        if n_a < min_atoms or n_b < min_atoms:
            result["RMSD_Status"] = f"ATOM_COUNT_LOW(A={n_a},B={n_b})"
            return result

        aln = cmd.super("sel_a", "sel_b", object="aln_obj")
        rmsd = float(aln[0])
        aligned_cnt = int(aln[1])
        result["RMSD"] = rmsd
        result["Aligned_Atoms"] = aligned_cnt
        result["RMSD_Status"] = "OK"

        hmm_len_a = pd.to_numeric(row.get("Main_HMM_Len"), errors="coerce")
        hmm_len_b = pd.to_numeric(row.get("Sub_HMM_Len"), errors="coerce")
        if pd.notna(hmm_len_a) and float(hmm_len_a) > 0:
            result["Aligned_Coverage_A"] = aligned_cnt / float(hmm_len_a)
        else:
            result["Aligned_Coverage_A"] = np.nan
        if pd.notna(hmm_len_b) and float(hmm_len_b) > 0:
            result["Aligned_Coverage_B"] = aligned_cnt / float(hmm_len_b)
        else:
            result["Aligned_Coverage_B"] = np.nan
        covs = [result["Aligned_Coverage_A"], result["Aligned_Coverage_B"]]
        covs = [x for x in covs if pd.notna(x)]
        result["Min_Aligned_Coverage"] = min(covs) if covs else np.nan

        if debug:
            print(f"\n[DEBUG RMSD] {main_hmm} vs {sub_hmm}")
            print(f"  A: {main_pdb_id} {main_start}-{main_end} chain={real_chain_a}")
            print(f"  B: {sub_pdb_id} {sub_start}-{sub_end} chain={real_chain_b}")
            print(f"  RMSD={rmsd:.4f}, aligned={aligned_cnt}")

        return result

    except Exception as exc:
        result["RMSD_Status"] = f"Exception:{exc}"
        if debug:
            traceback.print_exc()
        return result


def run_rmsd(args: argparse.Namespace, selection_csv: Path, rmsd_csv: Path) -> Path:
    if not selection_csv.exists():
        raise FileNotFoundError(f"Selection CSV not found: {selection_csv}")

    structure_root = None
    pfam_dir = Path(args.pfam_dir)

    if not pfam_dir.exists():
        raise FileNotFoundError(f"Pfam directory not found: {pfam_dir}")

    if rmsd_csv.exists() and args.overwrite:
        rmsd_csv.unlink()

    print("[RMSD] Initializing PyMOL...")
    import pymol
    from pymol import cmd
    pymol.finish_launching(["pymol", "-c", "-q"])

    print(f"[RMSD] Reading selection CSV: {selection_csv}")
    df = pd.read_csv(selection_csv, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "Main_HMM", "Sub_HMM",
        "Main_PDB_ID", "Sub_PDB_ID",
        "Main_Selected_Seq_Start", "Main_Selected_Seq_End",
        "Sub_Selected_Seq_Start", "Sub_Selected_Seq_End",
        "Main_Selection_Status", "Sub_Selection_Status",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Selection CSV missing required columns: {missing}")

    df = filter_dataframe_by_novel(df, args.novel_filter_for_rmsd)
    print(f"[RMSD] Rows to process after novel_filter_for_rmsd={args.novel_filter_for_rmsd}: {len(df)}")

    rows_out: List[Dict[str, Any]] = []
    failures: Dict[str, int] = defaultdict(int)

    for i, row in enumerate(tqdm(df.to_dict("records"), total=len(df), desc="RMSD from selected representatives")):
        res = process_rmsd_one_row(
            row=row,
            cmd=cmd,
            structure_root=structure_root,
            pfam_dir=pfam_dir,
            min_atoms=args.min_atoms,
            debug=(i < args.debug_first_n),
        )
        rows_out.append(res)
        if res.get("RMSD_Status") != "OK":
            failures[str(res.get("RMSD_Status"))] += 1

    pd.DataFrame(rows_out).to_csv(rmsd_csv, index=False)
    ok_n = sum(1 for r in rows_out if r.get("RMSD_Status") == "OK")
    print(f"[RMSD] Saved: {rmsd_csv}")
    print(f"[RMSD] Total rows: {len(rows_out)} | OK rows: {ok_n}")
    if failures:
        print("[RMSD] Top failures:")
        for k, v in sorted(failures.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {k}: {v}")
    return rmsd_csv


# ==============================================================================
# Standardization
# ==============================================================================


def extract_plddt(source: Any, reason: Any) -> Any:
    src = safe_str(source).upper()
    rsn = safe_str(reason)
    if src == "EXP":
        return "N/A"
    m = re.search(r"pLDDT[_:=\s-]*([0-9]+(?:\.[0-9]+)?)", rsn, flags=re.I)
    if m:
        return round(float(m.group(1)), 2)
    return np.nan


def run_standardize(args: argparse.Namespace, rmsd_csv: Path, standard_csv: Path) -> Path:
    if not rmsd_csv.exists():
        raise FileNotFoundError(f"RMSD CSV not found: {rmsd_csv}")

    print(f"[Standardize] Reading: {rmsd_csv}")
    df = pd.read_csv(rmsd_csv, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if args.standard_only_novel_true and "Is_Novel" in df.columns:
        df = df[df["Is_Novel"].apply(is_true_like)].copy()
        print(f"[Standardize] Filtered by Is_Novel == True: {len(df)} rows")
    if args.standard_only_ok and "RMSD_Status" in df.columns:
        df = df[df["RMSD_Status"].astype(str).str.strip() == "OK"].copy()
        print(f"[Standardize] Filtered by RMSD_Status == OK: {len(df)} rows")

    required = [
        "Main_HMM", "Sub_HMM", "RMSD", "Aligned_Atoms",
        "Main_Source", "Main_Selected_UID", "Main_PDB_ID",
        "Main_Selected_Seq_Start", "Main_Selected_Seq_End", "Main_Selection_Reason",
        "Sub_Source", "Sub_Selected_UID", "Sub_PDB_ID",
        "Sub_Selected_Seq_Start", "Sub_Selected_Seq_End", "Sub_Selection_Reason",
        "Main_HMM_Len", "Sub_HMM_Len",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"RMSD CSV missing required columns for standardization: {missing}")

    out_df = pd.DataFrame({
        "Main_HMM": df["Main_HMM"],
        "Sub_HMM": df["Sub_HMM"],
        "RMSD": pd.to_numeric(df["RMSD"], errors="coerce"),
        "Aligned_Atoms": pd.to_numeric(df["Aligned_Atoms"], errors="coerce"),
        "Source_A": df["Main_Source"],
        "UID_A": df["Main_Selected_UID"],
        "PDB_A": df["Main_PDB_ID"],
        "Range_A": [make_range(s, e) for s, e in zip(df["Main_Selected_Seq_Start"], df["Main_Selected_Seq_End"])],
        "pLDDT_A": [extract_plddt(src, rsn) for src, rsn in zip(df["Main_Source"], df["Main_Selection_Reason"])],
        "Source_B": df["Sub_Source"],
        "UID_B": df["Sub_Selected_UID"],
        "PDB_B": df["Sub_PDB_ID"],
        "Range_B": [make_range(s, e) for s, e in zip(df["Sub_Selected_Seq_Start"], df["Sub_Selected_Seq_End"])],
        "pLDDT_B": [extract_plddt(src, rsn) for src, rsn in zip(df["Sub_Source"], df["Sub_Selection_Reason"])],
        "HMM_Len_A": pd.to_numeric(df["Main_HMM_Len"], errors="coerce"),
        "HMM_Len_B": pd.to_numeric(df["Sub_HMM_Len"], errors="coerce"),
    })

    # Keep additional useful fields when available.
    for col in ["RMSD_Status", "Aligned_Coverage_A", "Aligned_Coverage_B", "Min_Aligned_Coverage", "Is_Novel", "Dual_Min_Coverage"]:
        if col in df.columns:
            out_df[col] = df[col].values

    standard_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(standard_csv, index=False)
    print(f"[Standardize] Saved: {standard_csv}")
    print(f"[Standardize] Rows written: {len(out_df)}")
    return standard_csv


# ==============================================================================
# CLI
# ==============================================================================



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ProDive-only representative selection, local RMSD calculation, and output standardization."
    )
    p.add_argument("--input-csv", required=True, help="Input overlap-check CSV, e.g. subset_data_Top_100%%_overlap_check_dual_20.csv")
    p.add_argument("--pfam-dir", required=True, help="PfamA_seed directory containing PFxxxxx/PFxxxxx.hhm, alignment, and structures")
    p.add_argument("--output-dir", required=True, help="Output directory")

    p.add_argument("--selection-csv", default=None, help="Representative-selection CSV. Default: output-dir/prodive_only_representative_selection.csv")
    p.add_argument("--rmsd-csv", default=None, help="RMSD output CSV. Default: output-dir/prodive_only_rmsd_from_selected.csv")
    p.add_argument("--standard-csv", default=None, help="Standardized RMSD output CSV. Default: output-dir/prodive_only_rmsd_standard.csv")

    p.add_argument("--novel-filter", default="true", choices=["all", "true", "false", "na"], help="Rows used for representative selection. Default: true for ProDive-only.")
    p.add_argument("--novel-filter-for-rmsd", default="true", choices=["all", "true", "false", "na"], help="Rows used for RMSD from the selection CSV. Default: true.")

    p.add_argument("--num-workers", type=int, default=40, help="Parallel workers for representative selection")
    p.add_argument("--max-search-depth", type=int, default=200)
    p.add_argument("--max-len-diff", type=int, default=5)
    p.add_argument("--min-mean-plddt", type=float, default=70.0)
    p.add_argument("--max-head-tail-pae", type=float, default=10.0)
    p.add_argument("--min-candidate-pure-len", type=int, default=3)
    p.add_argument("--min-atoms", type=int, default=3, help="Minimum CA atoms on each side before PyMOL superposition")
    p.add_argument("--write-chunk-size", type=int, default=1000)
    p.add_argument("--debug-first-n", type=int, default=5)

    p.add_argument("--standard-only-ok", action="store_true", default=True, help="Only keep RMSD_Status == OK in standardized output. Enabled by default.")
    p.add_argument("--no-standard-only-ok", action="store_false", dest="standard_only_ok")
    p.add_argument("--standard-only-novel-true", action="store_true", default=True, help="Only keep Is_Novel == True in standardized output. Enabled by default.")
    p.add_argument("--no-standard-only-novel-true", action="store_false", dest="standard_only_novel_true")

    p.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing output files. Enabled by default.")
    p.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_csv = Path(args.selection_csv) if args.selection_csv else output_dir / "prodive_only_representative_selection.csv"
    rmsd_csv = Path(args.rmsd_csv) if args.rmsd_csv else output_dir / "prodive_only_rmsd_from_selected.csv"
    standard_csv = Path(args.standard_csv) if args.standard_csv else output_dir / "prodive_only_rmsd_standard.csv"

    print("=" * 80)
    print("ProDive-only structural validation pipeline")
    print(f"Output dir: {output_dir}")
    print(f"Selection CSV: {selection_csv}")
    print(f"RMSD CSV: {rmsd_csv}")
    print(f"Standard CSV: {standard_csv}")
    print("=" * 80)

    run_selection(args, selection_csv)
    run_rmsd(args, selection_csv, rmsd_csv)
    run_standardize(args, rmsd_csv, standard_csv)

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
