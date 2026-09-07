#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute structural RMSD for de novo-Pfam ProDive fragment correspondences.

This script is the GitHub-ready version of the original de novo RMSD script. It
keeps the original analysis logic but makes all paths configurable and writes the
coverage columns required by the Fig. 2 plotting script directly, so a separate
post-hoc coverage-adding script is no longer required for new runs.

Input CSV requirements:
  - Main_HMM: de novo chain / query identifier
  - Sub_HMM: Pfam family identifier
  - Sub_Segments_Details: segment pair string, e.g. "1-11 -> 75-85"

Optional input columns used when present:
  - Main_Segment_Len: target length used for coverage
  - Score: retained as metadata

Output columns include:
  Query_DeNovo, Target_Pfam, Query_Range, Target_Range, RMSD, Aligned_Atoms,
  Target_Len, HMM_Len, Coverage, structure source metadata.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import PDB, SeqIO
from tqdm import tqdm

cmd = None  # Initialized when RMSD calculation starts.


AA_UID_RE = re.compile(r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})")


def parse_range(text: Any) -> Tuple[int, int]:
    nums = re.findall(r"\d+", str(text))
    if len(nums) < 2:
        raise ValueError(f"Cannot parse range: {text}")
    a, b = int(nums[0]), int(nums[1])
    return (a, b) if a <= b else (b, a)


def split_segment_pairs(details: Any, mode: str = "first") -> List[Tuple[str, str]]:
    s = str(details or "").strip()
    if not s or "->" not in s:
        return []

    raw_parts = re.split(r"\s*;\s*", s)
    pairs: List[Tuple[str, str]] = []
    for part in raw_parts:
        if "->" not in part:
            continue
        left, right = part.split("->", 1)
        pairs.append((left.strip(), right.strip()))
        if mode == "first":
            break
    return pairs


def read_tasks(input_csv: str, segment_mode: str = "first") -> List[Dict[str, Any]]:
    df = pd.read_csv(input_csv, encoding="utf-8-sig", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    required = ["Main_HMM", "Sub_HMM", "Sub_Segments_Details"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}. Available columns: {list(df.columns)}")

    tasks: List[Dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        q_id = str(row["Main_HMM"]).strip()
        t_id = str(row["Sub_HMM"]).strip()
        if not q_id or not t_id or q_id.lower() == "nan" or t_id.lower() == "nan":
            continue

        for pair_idx, (q_range, t_range) in enumerate(split_segment_pairs(row["Sub_Segments_Details"], segment_mode)):
            try:
                q1, q2 = parse_range(q_range)
                t1, t2 = parse_range(t_range)
            except Exception:
                continue

            q_len = q2 - q1 + 1
            if "Main_Segment_Len" in df.columns and pd.notna(row.get("Main_Segment_Len")):
                try:
                    target_len = int(float(row.get("Main_Segment_Len")))
                except Exception:
                    target_len = q_len
            else:
                target_len = q_len

            tasks.append({
                "Input_Row_Index": row_idx,
                "Segment_Pair_Index": pair_idx,
                "Query_DeNovo": q_id,
                "Target_Pfam": t_id,
                "Query_Range": f"{q1}-{q2}",
                "Target_Range": f"{t1}-{t2}",
                "q1": q1,
                "q2": q2,
                "t1": t1,
                "t2": t2,
                "Query_Len": q_len,
                "Target_Len": target_len,
                "Score": row.get("Score", np.nan),
            })
    return tasks


def build_id_to_chain_mapping(fasta_path: Optional[str]) -> Dict[str, str]:
    if not fasta_path or not os.path.exists(fasta_path):
        return {}
    mapping: Dict[str, str] = {}
    chain_pattern = re.compile(r"Chains?\s+([A-Za-z0-9]+)")
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.description.split("|")[0].strip()
        m = chain_pattern.search(record.description)
        mapping[seq_id] = m.group(1) if m else "A"
    return mapping


def resolve_chain_auto(obj_name: str, preferred_chain: str = "A") -> Optional[str]:
    chains = cmd.get_chains(obj_name)
    if not chains:
        return None
    if preferred_chain in chains:
        return preferred_chain
    for c in chains:
        if str(c).upper() == str(preferred_chain).upper():
            return c
    if "A" in chains:
        return "A"
    return chains[0]


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
    if not os.path.exists(hhm_path):
        return {}, "HHM_FILE_MISSING"
    mapping: Dict[int, int] = {}
    try:
        match_states = extract_match_states_from_hhm(hhm_path)
        in_hmm = False
        with open(hhm_path, "r", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("HMM"):
                    in_hmm = True
                    continue
                if line.startswith("//"):
                    break
                if not in_hmm:
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                aa, idx, msa_col = parts[0], parts[1], parts[-1]
                if len(aa) == 1 and aa.isalpha() and idx.isdigit() and msa_col.isdigit():
                    h = int(idx)
                    m = int(msa_col)
                    if match_states is not None and not (1 <= h <= match_states):
                        continue
                    if m > 0:
                        mapping[h] = m
        if not mapping:
            return {}, "HHM_CONTENT_EMPTY"
        return mapping, "OK"
    except Exception:
        return {}, "HHM_PARSE_ERROR"


def read_alignment(aln_path: str) -> Tuple[Dict[str, str], str]:
    if not os.path.exists(aln_path):
        return {}, "ALIGN_FILE_MISSING"
    seqs: Dict[str, str] = {}
    current = None
    try:
        with open(aln_path, "r", errors="ignore") as f:
            first = f.readline()
            f.seek(0)
            if first.startswith(">"):
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        current = line[1:].strip()
                        seqs[current] = ""
                    elif current is not None:
                        seqs[current] += line
            else:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("//"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        seqs[parts[0]] = seqs.get(parts[0], "") + parts[1]
    except Exception as e:
        return {}, f"READ_ERR:{e}"
    return (seqs, "OK") if seqs else ({}, "ALIGN_EMPTY")


def ungap(s: str) -> str:
    return (s or "").replace(".", "").replace("-", "")


def get_candidates_from_alignment(aln_path: str, msa_start: int, msa_end: int, expected_len: int, max_search_depth: int) -> Tuple[List[Dict[str, Any]], str]:
    seqs, status = read_alignment(aln_path)
    if not seqs:
        return [], status
    candidates: List[Dict[str, Any]] = []
    i0, i1 = msa_start - 1, msa_end
    for header, seq in seqs.items():
        if len(seq) < i1:
            continue
        m = re.search(r"/(\d+)-(\d+)", header)
        if not m:
            continue
        global_offset = int(m.group(1))
        uid_m = AA_UID_RE.search(header)
        uid = uid_m.group(1) if uid_m else header.split("/")[0].split()[0]
        frag_raw = seq[i0:i1]
        frag_pure = ungap(frag_raw)
        if len(frag_pure) < 3:
            continue
        prefix = seq[:i0]
        pdb_start = global_offset + len(ungap(prefix))
        pdb_end = pdb_start + len(frag_pure) - 1
        candidates.append({
            "uid": uid,
            "pdb_start": pdb_start,
            "pdb_end": pdb_end,
            "seq_len": len(frag_pure),
            "gaps": frag_raw.count(".") + frag_raw.count("-"),
            "len_diff": abs(len(frag_pure) - expected_len),
        })
    candidates.sort(key=lambda x: (x["len_diff"], x["gaps"]))
    return candidates[:max_search_depth], "OK"


def parse_exp_chain(filename: str) -> str:
    m = re.search(r"_([A-Za-z0-9]+)\.pdb$", filename)
    return m.group(1) if m else "A"


def check_af_quality(pdb_path: str, start_res: int, end_res: int, min_mean_plddt: float, max_head_tail_pae: Optional[float]) -> Tuple[bool, float, str]:
    mean_plddt = 0.0
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("X", pdb_path)
        model = structure[0]
        chain = "A" if "A" in model else next(model.get_chains()).id
        vals = []
        for res in model[chain]:
            if start_res <= res.id[1] <= end_res and "CA" in res:
                vals.append(float(res["CA"].get_bfactor()))
        if not vals:
            return False, 0.0, "NO_ATOMS"
        mean_plddt = float(np.mean(vals))
        if mean_plddt < min_mean_plddt:
            return False, mean_plddt, "LOW_PLDDT"
    except Exception:
        return False, 0.0, "PARSE_ERR"

    if max_head_tail_pae is not None:
        json_path = pdb_path.replace(".pdb", ".json")
        if not os.path.exists(json_path):
            base = pdb_path[:-4]
            hits = glob.glob(base + "*.json")
            json_path = hits[0] if hits else ""
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                block = data[0] if isinstance(data, list) and data else data
                pae = block.get("predicted_aligned_error", block.get("pae")) if isinstance(block, dict) else None
                if pae is not None:
                    is_ = max(0, start_res - 1)
                    ie = min(len(pae) - 1, end_res - 1)
                    if float(pae[is_][ie]) > max_head_tail_pae:
                        return False, mean_plddt, "HIGH_PAE"
            except Exception:
                pass
    return True, mean_plddt, "OK"


def find_best_pfam_structure(family_dir: str, candidates: List[Dict[str, Any]], max_len_diff: int, min_mean_plddt: float, max_head_tail_pae: Optional[float]) -> Tuple[Optional[Dict[str, Any]], str]:
    for cand in candidates:
        if cand["len_diff"] > max_len_diff:
            break
        uid = cand["uid"]
        s, e = cand["pdb_start"], cand["pdb_end"]
        exp_files = sorted(glob.glob(os.path.join(family_dir, f"{uid}_exp_*.pdb")))
        if exp_files:
            p = exp_files[0]
            return {
                "path": p, "type": "EXP", "chain": parse_exp_chain(p), "uid": uid,
                "start": s, "end": e, "plddt": "N/A", "seq_len": cand["seq_len"], "len_diff": cand["len_diff"],
            }, "OK"
        af_files = sorted(glob.glob(os.path.join(family_dir, f"AF-{uid}-F1-model*.pdb")), reverse=True)
        if af_files:
            p = af_files[0]
            ok, plddt, msg = check_af_quality(p, s, e, min_mean_plddt, max_head_tail_pae)
            if ok:
                return {
                    "path": p, "type": "AF", "chain": "A", "uid": uid,
                    "start": s, "end": e, "plddt": round(float(plddt), 2), "seq_len": cand["seq_len"], "len_diff": cand["len_diff"],
                }, "OK"
    return None, "NO_VALID_STRUCT"


def find_denovo_pdb(denovo_pdb_dir: str, q_id: str) -> str:
    candidates = [
        os.path.join(denovo_pdb_dir, f"{q_id}.pdb"),
        os.path.join(denovo_pdb_dir, f"{q_id.split('_')[0]}.pdb"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def process_tasks(args: argparse.Namespace) -> int:
    global cmd
    try:
        import pymol
        from pymol import cmd as pymol_cmd
    except ImportError as exc:
        raise RuntimeError("PyMOL is required for RMSD calculation; install pymol-open-source in the active environment.") from exc
    cmd = pymol_cmd
    chain_map = build_id_to_chain_mapping(args.fasta_mapping_file)
    tasks = read_tasks(args.input_csv, args.segment_mode)
    if not tasks:
        print("No valid tasks found.")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    pymol.finish_launching(["pymol", "-c", "-q"])

    failures: Counter[str] = Counter()
    results: List[Dict[str, Any]] = []

    for i, task in enumerate(tqdm(tasks, desc="de novo-Pfam RMSD")):
        debug = i < args.debug_first_n
        q_id = task["Query_DeNovo"]
        t_id = task["Target_Pfam"]
        try:
            denovo_pdb = find_denovo_pdb(args.denovo_pdb_dir, q_id)
            if not denovo_pdb:
                failures["DENOVO_PDB_MISSING"] += 1
                continue
            denovo_chain_hint = chain_map.get(q_id, "A")

            hhm_path = os.path.join(args.pfam_dir, t_id, f"{t_id}.hhm")
            hmap, st = extract_hmm_to_msa_map(hhm_path)
            if not hmap or task["t1"] not in hmap or task["t2"] not in hmap:
                failures[f"PFAM_HMM_MAP_FAIL:{st}"] += 1
                continue
            msa_s, msa_e = hmap[task["t1"]], hmap[task["t2"]]
            aln_path = os.path.join(args.pfam_dir, t_id, f"{t_id}.fas")
            if not os.path.exists(aln_path):
                aln_path = os.path.join(args.pfam_dir, t_id, f"{t_id}.sto")
            cands, cand_status = get_candidates_from_alignment(aln_path, msa_s, msa_e, task["t2"] - task["t1"] + 1, args.max_search_depth)
            if not cands:
                failures[f"PFAM_NO_CANDIDATES:{cand_status}"] += 1
                continue
            pfam_struct, msg = find_best_pfam_structure(
                os.path.join(args.pfam_dir, t_id), cands, args.max_len_diff,
                args.min_mean_plddt, args.max_head_tail_pae,
            )
            if not pfam_struct:
                failures[f"PFAM_STRUCT_FAIL:{msg}"] += 1
                continue

            cmd.reinitialize()
            cmd.load(denovo_pdb, "obj_denovo")
            chain_d = resolve_chain_auto("obj_denovo", denovo_chain_hint)
            if chain_d is None:
                failures["DENOVO_EMPTY_STRUCTURE"] += 1
                continue
            cmd.load(pfam_struct["path"], "obj_pfam")
            chain_p = resolve_chain_auto("obj_pfam", pfam_struct["chain"])
            if chain_p is None:
                failures["PFAM_EMPTY_STRUCTURE"] += 1
                continue

            sel_d = f"obj_denovo and chain {chain_d} and resi {task['q1']}-{task['q2']} and name CA"
            sel_p = f"obj_pfam and chain {chain_p} and resi {pfam_struct['start']}-{pfam_struct['end']} and name CA"
            if cmd.count_atoms(sel_d) < args.min_atoms or cmd.count_atoms(sel_p) < args.min_atoms:
                failures["ATOM_COUNT_LOW"] += 1
                continue
            aln = cmd.super(sel_d, sel_p, object="aln_obj")
            rmsd = float(aln[0])
            atoms = int(aln[1])
            target_len = int(task["Target_Len"]) if pd.notna(task["Target_Len"]) else task["Query_Len"]
            coverage = atoms / target_len if target_len > 0 else np.nan

            results.append({
                **task,
                "RMSD": rmsd,
                "Aligned_Atoms": atoms,
                "HMM_Len": target_len,
                "Coverage": coverage,
                "DeNovo_PDB": os.path.basename(denovo_pdb),
                "DeNovo_Chain": chain_d,
                "DeNovo_Range": task["Query_Range"],
                "Pfam_UID": pfam_struct["uid"],
                "Pfam_PDB": os.path.basename(pfam_struct["path"]),
                "Pfam_Source": pfam_struct["type"],
                "Pfam_Chain": chain_p,
                "Pfam_Range": f"{pfam_struct['start']}-{pfam_struct['end']}",
                "Pfam_pLDDT": pfam_struct["plddt"],
                "Pfam_Len_Diff": pfam_struct["len_diff"],
                "Pfam_Seq_Len": pfam_struct["seq_len"],
            })
            if debug:
                print(f"[OK] {q_id} {task['Query_Range']} vs {t_id} {task['Target_Range']} RMSD={rmsd:.3f} atoms={atoms}")
        except Exception:
            if debug:
                traceback.print_exc()
            failures["UNKNOWN_ERR"] += 1

    out = pd.DataFrame(results)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")
    print(f"Successful RMSD rows: {len(out)} / tasks={len(tasks)}")
    if failures:
        print("Top failures:")
        for k, v in failures.most_common(20):
            print(f"  {k}: {v}")
    return len(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute de novo-Pfam RMSD and coverage for ProDive fragments.")
    p.add_argument("--input-csv", required=True, help="de novo-Pfam global_high_score_summary_fin.csv")
    p.add_argument("--denovo-pdb-dir", required=True, help="Directory containing de novo PDB files")
    p.add_argument("--pfam-dir", required=True, help="PfamA_seed directory containing family HHM/MSA/PDB/AF files")
    p.add_argument("--fasta-mapping-file", default=None, help="Optional FASTA file containing de novo chain annotations")
    p.add_argument("--output-csv", required=True, help="Output RMSD CSV with Coverage")
    p.add_argument("--segment-mode", choices=["first", "all"], default="first", help="How to handle multiple segment pairs per input row")
    p.add_argument("--max-search-depth", type=int, default=200)
    p.add_argument("--max-len-diff", type=int, default=5)
    p.add_argument("--min-mean-plddt", type=float, default=70.0)
    p.add_argument("--max-head-tail-pae", type=float, default=None, help="Optional AF endpoint PAE threshold")
    p.add_argument("--min-atoms", type=int, default=3)
    p.add_argument("--debug-first-n", type=int, default=5)
    return p


def main() -> None:
    args = build_parser().parse_args()
    process_tasks(args)


if __name__ == "__main__":
    main()
