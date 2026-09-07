#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import ast
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ========================================================
# Paths to edit if needed
# ========================================================
GLOBAL_SCORE_FILE = Path(
    "<PRODIVE_DATA_ROOT>/shared/global_high_score_summary_fin.csv"
)

#  Start2Fold seed 
COVERED_PROTEIN_FILE = Path(
    "CHANGE_ME"
)

#  Start2Fold seed detail
DETAIL_FILE = Path(
    "CHANGE_ME"
)

PFAM_DIR = Path("<PRODIVE_DATA_ROOT>/shared/PfamA_seed")
OUTPUT_DIR = Path(
    "CHANGE_ME"
)

# ========================================================
# Analysis settings
# ========================================================
TASK_LIMIT = None                  # None = run all
ONLY_RESIDUE_OVERLAP_COVERED = True
RANDOM_SAMPLES_PER_SEGMENT = 1000
RANDOM_SEED = 20260409

CLASS_KEYS = [
    "Fold_EARLY",
    "Fold_INTER",
    "Fold_LATE",
    "Stab_STRONG",
    "Stab_MEDIUM",
    "Stab_WEAK",
]

# “>=1 ”“>=2 ”， summary 
HIT_THRESHOLDS = [1, 2]

# ========================================================


# -------------------------
# General helpers
# -------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("\xa0", " ").strip()


def safe_int(x, default=None):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def strip_version(ac: str) -> str:
    s = safe_str(ac)
    return s.split(".", 1)[0] if s else ""


def ungap(seq: str) -> str:
    return seq.replace(".", "").replace("-", "")


def parse_int_list(x) -> List[int]:
    s = safe_str(x)
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
    except Exception:
        return []
    if obj is None or not isinstance(obj, (list, tuple, set)):
        return []
    out = []
    for v in obj:
        try:
            out.append(int(v))
        except Exception:
            continue
    return sorted(set(out))


def safe_div(a, b, default=np.nan):
    try:
        if b is None or float(b) == 0:
            return default
        return float(a) / float(b)
    except Exception:
        return default


def empirical_p_ge(real_value: float, random_values: np.ndarray) -> float:
    """Empirical one-sided p-value: P(random >= real)."""
    if random_values is None or len(random_values) == 0:
        return np.nan
    rv = np.asarray(random_values, dtype=float)
    return (1.0 + float(np.sum(rv >= float(real_value)))) / (1.0 + float(len(rv)))


# -------------------------
# Run / overlap helpers
# -------------------------
def run_stats(xs: List[int]) -> Tuple[int, int]:
    """
    Return:
      longest_run_len
      num_runs
    """
    if not xs:
        return 0, 0

    xs = sorted(set(int(x) for x in xs))
    longest = 1
    num_runs = 1
    cur = 1

    for i in range(1, len(xs)):
        if xs[i] == xs[i - 1] + 1:
            cur += 1
            longest = max(longest, cur)
        else:
            num_runs += 1
            cur = 1

    return longest, num_runs


def residues_in_interval(residues: List[int], start: int, end: int) -> List[int]:
    lo, hi = int(start), int(end)
    return [r for r in residues if lo <= int(r) <= hi]


def compute_class_metrics(
    seg_start: int,
    seg_end: int,
    seg_len: int,
    class_seed_residues: Dict[str, List[int]],
) -> dict:
    """
    Compute overlap metrics between a mapped segment [seg_start, seg_end]
    and seed-projected Start2Fold residue sets.
    """
    rec = {}
    any_union = set()

    for key in CLASS_KEYS:
        seed_res = class_seed_residues.get(key, [])
        ov = residues_in_interval(seed_res, seg_start, seg_end)
        ov_count = len(ov)
        longest_run, num_runs = run_stats(ov)

        rec[f"{key}_seed_residue_count"] = len(seed_res)
        rec[f"{key}_segment_overlap_count"] = ov_count
        rec[f"{key}_segment_overlap_list"] = str(sorted(ov))
        rec[f"{key}_segment_overlap_frac"] = safe_div(ov_count, seg_len)
        rec[f"{key}_segment_longest_run"] = longest_run
        rec[f"{key}_segment_longest_run_frac"] = safe_div(longest_run, seg_len)
        rec[f"{key}_segment_num_runs"] = num_runs

        if ov_count > 0:
            any_union.update(ov)

    any_list = sorted(any_union)
    any_count = len(any_list)
    any_longest_run, any_num_runs = run_stats(any_list)

    rec["Any_Start2Fold_Overlap_Count"] = any_count
    rec["Any_Start2Fold_Overlap_List"] = str(any_list)
    rec["Any_Start2Fold_Overlap_Frac"] = safe_div(any_count, seg_len)
    rec["Any_Start2Fold_Longest_Run"] = any_longest_run
    rec["Any_Start2Fold_Longest_Run_Frac"] = safe_div(any_longest_run, seg_len)
    rec["Any_Start2Fold_Num_Runs"] = any_num_runs
    rec["Any_Start2Fold_Overlap_Status"] = "YES" if any_count > 0 else "NO"

    return rec


def sample_random_windows(
    seed_start: int,
    seed_end: int,
    seg_len: int,
    n_samples: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Sample same-length windows within the same seed range.
    Sampling is with replacement, so even if possible windows < 1000, it still works.
    """
    total_len = int(seed_end) - int(seed_start) + 1
    if seg_len <= 0 or total_len <= 0 or seg_len > total_len:
        return []

    min_start = int(seed_start)
    max_start = int(seed_end) - int(seg_len) + 1
    if max_start < min_start:
        return []

    starts = rng.integers(min_start, max_start + 1, size=n_samples)
    return [(int(s), int(s) + int(seg_len) - 1) for s in starts]


# ---------------- HHM / alignment mapping helpers ----------------
def extract_hmm_to_msa_map(hhm_path: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if not hhm_path.exists():
        raise FileNotFoundError(f"Missing HHM file: {hhm_path}")

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
    if not aln_path.exists():
        raise FileNotFoundError(f"Missing alignment file: {aln_path}")

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


def build_target_candidates(row: pd.Series) -> List[str]:
    candidates: List[str] = []
    for key in [
        "seed_seq_name",
        "seed_seq_id",
        "seed_ac_raw",
        "seed_ac_base",
        "uniprot_ac_base",
        "uniprot_ac_raw",
    ]:
        value = safe_str(row.get(key, ""))
        if value:
            candidates.append(value)
            if "." in value:
                candidates.append(value.split(".", 1)[0])

    uniq = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def find_target_sequence(seqs: Dict[str, str], row: pd.Series) -> Tuple[str, str, str]:
    candidates = build_target_candidates(row)
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


def map_msa_interval_to_real_seq(
    aligned_seq: str,
    global_start: int,
    msa_start_col: int,
    msa_end_col: int,
) -> dict:
    idx_start = msa_start_col - 1
    idx_end = msa_end_col

    if len(aligned_seq) < idx_end:
        return {
            "msa_start": msa_start_col,
            "msa_end": msa_end_col,
            "seq_start": np.nan,
            "seq_end": np.nan,
            "seq_len": 0,
            "frag_raw": "",
            "frag_pure": "",
            "gaps": np.nan,
            "status": "MSA_Out_Of_Range",
        }

    frag_raw = aligned_seq[idx_start:idx_end]
    frag_pure = ungap(frag_raw)
    gaps = frag_raw.count(".") + frag_raw.count("-")

    if len(frag_pure) == 0:
        return {
            "msa_start": msa_start_col,
            "msa_end": msa_end_col,
            "seq_start": np.nan,
            "seq_end": np.nan,
            "seq_len": 0,
            "frag_raw": frag_raw,
            "frag_pure": frag_pure,
            "gaps": gaps,
            "status": "Only_Gaps_On_Target",
        }

    prefix = aligned_seq[:idx_start]
    real_res_before = len(ungap(prefix))

    seq_start = global_start + real_res_before
    seq_end = seq_start + len(frag_pure) - 1

    return {
        "msa_start": msa_start_col,
        "msa_end": msa_end_col,
        "seq_start": seq_start,
        "seq_end": seq_end,
        "seq_len": len(frag_pure),
        "frag_raw": frag_raw,
        "frag_pure": frag_pure,
        "gaps": gaps,
        "status": "OK",
    }


# ---------------- Global result parsing helpers ----------------
def parse_main_segment(text: str) -> List[Tuple[int, int]]:
    pairs = re.findall(r"(\d+)-(\d+)", safe_str(text))
    return [(int(a), int(b)) for a, b in pairs]


def parse_sub_segments_details(text: str) -> List[Tuple[int, int, int, int]]:
    pairs = re.findall(r"(\d+)-(\d+)\s*->\s*(\d+)-(\d+)", safe_str(text))
    return [(int(a), int(b), int(c), int(d)) for a, b, c, d in pairs]


def extract_target_hmm_segments(match_row: pd.Series, target_pfam: str) -> List[Tuple[str, int, int]]:
    segments: List[Tuple[str, int, int]] = []
    main_hmm = safe_str(match_row.get("Main_HMM", "")).upper()
    sub_hmm = safe_str(match_row.get("Sub_HMM", "")).upper()
    main_segment = safe_str(match_row.get("Main_Segment", ""))
    details = safe_str(match_row.get("Sub_Segments_Details", ""))

    detail_pairs = parse_sub_segments_details(details)

    if main_hmm == target_pfam:
        main_pairs = parse_main_segment(main_segment)
        if main_pairs:
            for s, e in main_pairs:
                segments.append(("main", s, e))
        else:
            for a, b, _, _ in detail_pairs:
                segments.append(("main", a, b))

    if sub_hmm == target_pfam:
        for _, _, c, d in detail_pairs:
            segments.append(("sub", c, d))

    return segments


# ---------------- Input loading ----------------
def read_covered_proteins(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing covered-protein file: {path}")

    df = pd.read_csv(path, dtype=str)
    for col in df.columns:
        df[col] = df[col].map(safe_str)

    if "STF_ID" not in df.columns:
        raise ValueError(f"covered-protein file missing STF_ID. Columns: {list(df.columns)}")

    if ONLY_RESIDUE_OVERLAP_COVERED:
        if "coverage_basis" in df.columns:
            df = df[df["coverage_basis"] == "residue_overlap"].copy()
        elif "num_seed_overlap_hits" in df.columns:
            keep = pd.to_numeric(df["num_seed_overlap_hits"], errors="coerce").fillna(0) > 0
            df = df[keep].copy()
        else:
            raise ValueError(
                "covered-protein file has neither coverage_basis nor num_seed_overlap_hits"
            )

    return df.reset_index(drop=True)


def read_detail_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing detail file: {path}")

    df = pd.read_csv(path, dtype=str)
    for col in df.columns:
        df[col] = df[col].map(safe_str)

    required = [
        "STF_ID",
        "pfam_id",
        "seed_seq_name",
        "seed_ac_base",
        "uniprot_ac_base",
        "seed_start",
        "seed_end",
        "total_overlap_count",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"detail file missing columns: {missing}")

    df["pfam_id"] = df["pfam_id"].map(lambda x: safe_str(x).upper())
    df["seed_ac_base"] = df["seed_ac_base"].map(strip_version)
    df["uniprot_ac_base"] = df["uniprot_ac_base"].map(strip_version)

    df["total_overlap_count_num"] = pd.to_numeric(
        df["total_overlap_count"], errors="coerce"
    ).fillna(0).astype(int)
    df["seed_start_num"] = pd.to_numeric(df["seed_start"], errors="coerce")
    df["seed_end_num"] = pd.to_numeric(df["seed_end"], errors="coerce")

    #  seed  residue list 
    for key in CLASS_KEYS:
        list_col = f"{key}_overlap_list"
        if list_col not in df.columns:
            df[list_col] = ""
        parsed_col = f"{key}_residue_list"
        df[parsed_col] = df[list_col].map(parse_int_list)

    if "has_any_overlap" in df.columns:
        keep = pd.to_numeric(df["has_any_overlap"], errors="coerce").fillna(0).astype(int) > 0
        df = df[keep].copy()
    else:
        df = df[df["total_overlap_count_num"] > 0].copy()

    return df.reset_index(drop=True)


def build_tasks(covered_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    covered_ids = set(covered_df["STF_ID"].astype(str))

    task_df = detail_df[detail_df["STF_ID"].isin(covered_ids)].copy()
    task_df = task_df[task_df["total_overlap_count_num"] > 0].copy()
    task_df = task_df[task_df["pfam_id"] != ""].copy()
    task_df = task_df[task_df["seed_seq_name"] != ""].copy()

    for col in ["seed_start_num", "seed_end_num"]:
        task_df = task_df[task_df[col].notna()].copy()
        task_df[col] = task_df[col].astype(int)

    dedupe_cols = [
        "STF_ID",
        "pfam_id",
        "seed_seq_name",
        "seed_ac_base",
        "uniprot_ac_base",
        "seed_start",
        "seed_end",
    ]
    task_df = task_df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)

    if TASK_LIMIT is not None:
        task_df = task_df.head(TASK_LIMIT).copy()

    return task_df


def read_global_score(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing global score file: {path}")

    df = pd.read_csv(path)
    required = ["File", "Score", "Main_HMM", "Sub_HMM", "Main_Segment", "Sub_Segments_Details"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"global score file missing columns: {missing}")

    for col in ["Main_HMM", "Sub_HMM", "Main_Segment", "Sub_Segments_Details"]:
        df[col] = df[col].map(safe_str)
    return df


def build_global_index(global_df: pd.DataFrame) -> Dict[str, List[int]]:
    index: Dict[str, set] = defaultdict(set)
    for idx, row in global_df.iterrows():
        main = safe_str(row["Main_HMM"]).upper()
        sub = safe_str(row["Sub_HMM"]).upper()
        if main:
            index[main].add(idx)
        if sub:
            index[sub].add(idx)
    return {k: sorted(v) for k, v in index.items()}


# ---------------- Core mapping cache ----------------
def get_cached_pfam_data(pfam_id: str, cache: Dict[str, dict]) -> dict:
    if pfam_id in cache:
        return cache[pfam_id]

    family_dir = PFAM_DIR / pfam_id
    hhm_path = family_dir / f"{pfam_id}.hhm"
    aln_path = choose_alignment_path(family_dir, pfam_id)

    data = {
        "family_dir": family_dir,
        "hhm_path": hhm_path,
        "aln_path": aln_path,
        "hhm_map": extract_hmm_to_msa_map(hhm_path),
        "seqs": read_alignment(aln_path),
    }
    cache[pfam_id] = data
    return data


# ---------------- Core task processing ----------------
def process_one_task(
    task_row: pd.Series,
    global_df: pd.DataFrame,
    global_index: Dict[str, List[int]],
    pfam_cache: Dict[str, dict],
    rng: np.random.Generator,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Returns:
      actual_records
      random_records
      summary
    """
    pfam_id = safe_str(task_row["pfam_id"]).upper()
    stf_id = safe_str(task_row["STF_ID"])

    summary = {
        "STF_ID": stf_id,
        "pfam_id": pfam_id,
        "seed_seq_name": safe_str(task_row.get("seed_seq_name", "")),
        "status": "INIT",
        "message": "",
        "num_candidate_global_rows": 0,
        "num_target_segments": 0,
        "num_mapped_ok": 0,
        "num_any_overlap_hits": 0,
        "num_random_windows": 0,
    }
    for key in CLASS_KEYS:
        summary[f"{key}_segment_hits_ge1"] = 0
        summary[f"{key}_segment_hits_ge2"] = 0

    actual_records: List[dict] = []
    random_records: List[dict] = []

    try:
        pfam_data = get_cached_pfam_data(pfam_id, pfam_cache)
        target_header, target_aln_seq, match_mode = find_target_sequence(pfam_data["seqs"], task_row)
        header_start, header_end = parse_header_global_range(target_header)

        seed_start = safe_int(task_row.get("seed_start_num"), None)
        seed_end = safe_int(task_row.get("seed_end_num"), None)
        if seed_start is not None and seed_end is not None:
            global_start = seed_start
            global_end_expected = seed_end
        elif header_start is not None and header_end is not None:
            global_start = header_start
            global_end_expected = header_end
        else:
            raise RuntimeError(f"No usable seed range and cannot parse range from header: {target_header}")

        summary["num_candidate_global_rows"] = len(global_index.get(pfam_id, []))

        class_seed_residues = {
            key: task_row.get(f"{key}_residue_list", []) for key in CLASS_KEYS
        }

        for global_idx in global_index.get(pfam_id, []):
            grow = global_df.loc[global_idx]
            segments = extract_target_hmm_segments(grow, pfam_id)
            if not segments:
                continue

            for seg_rank, (pfam_side, hmm_start, hmm_end) in enumerate(segments, start=1):
                summary["num_target_segments"] += 1

                segment_uid = (
                    f"{stf_id}|{pfam_id}|{safe_str(task_row.get('seed_seq_name',''))}|"
                    f"{global_idx + 1}|{seg_rank}|{pfam_side}|{hmm_start}-{hmm_end}"
                )

                rec = {
                    "segment_uid": segment_uid,
                    "STF_ID": stf_id,
                    "uniprot_ac_base": safe_str(task_row.get("uniprot_ac_base", "")),
                    "UniProt_Range": safe_str(task_row.get("UniProt_Range", "")),
                    "pfam_id": pfam_id,
                    "seed_seq_name": safe_str(task_row.get("seed_seq_name", "")),
                    "seed_ac_base": safe_str(task_row.get("seed_ac_base", "")),
                    "seed_start": global_start,
                    "seed_end": global_end_expected,
                    "target_header": target_header,
                    "target_match_mode": match_mode,
                    "global_row_index_1based": global_idx + 1,
                    "segment_rank": seg_rank,
                    "File": grow.get("File", ""),
                    "Score": grow.get("Score", np.nan),
                    "Coverage_Main": grow.get("Coverage_Main", np.nan),
                    "Coverage_Sub": grow.get("Coverage_Sub", np.nan),
                    "Main_HMM": grow.get("Main_HMM", ""),
                    "Sub_HMM": grow.get("Sub_HMM", ""),
                    "Main_Segment": grow.get("Main_Segment", ""),
                    "Sub_Segments_Details": grow.get("Sub_Segments_Details", ""),
                    "Target_Pfam_Side": pfam_side,
                    "Target_HMM_Start": hmm_start,
                    "Target_HMM_End": hmm_end,
                }

                if hmm_start not in pfam_data["hhm_map"] or hmm_end not in pfam_data["hhm_map"]:
                    rec.update({
                        "Target_MSA_Start": np.nan,
                        "Target_MSA_End": np.nan,
                        "Target_Seq_Start": np.nan,
                        "Target_Seq_End": np.nan,
                        "Target_Seq_Len": 0,
                        "Target_Frag": "",
                        "Target_Aligned_Frag": "",
                        "Gap_Count_In_MSA_Window": np.nan,
                        "Map_Status": "HMM_Position_Not_In_HHM_Map",
                    })
                    # 
                    for key in CLASS_KEYS:
                        rec[f"{key}_seed_residue_count"] = len(class_seed_residues[key])
                        rec[f"{key}_segment_overlap_count"] = 0
                        rec[f"{key}_segment_overlap_list"] = "[]"
                        rec[f"{key}_segment_overlap_frac"] = np.nan
                        rec[f"{key}_segment_longest_run"] = 0
                        rec[f"{key}_segment_longest_run_frac"] = np.nan
                        rec[f"{key}_segment_num_runs"] = 0
                    rec["Any_Start2Fold_Overlap_Count"] = 0
                    rec["Any_Start2Fold_Overlap_List"] = "[]"
                    rec["Any_Start2Fold_Overlap_Frac"] = np.nan
                    rec["Any_Start2Fold_Longest_Run"] = 0
                    rec["Any_Start2Fold_Longest_Run_Frac"] = np.nan
                    rec["Any_Start2Fold_Num_Runs"] = 0
                    rec["Any_Start2Fold_Overlap_Status"] = "NO"
                    actual_records.append(rec)
                    continue

                msa_start = pfam_data["hhm_map"][hmm_start]
                msa_end = pfam_data["hhm_map"][hmm_end]

                mapped = map_msa_interval_to_real_seq(
                    aligned_seq=target_aln_seq,
                    global_start=global_start,
                    msa_start_col=msa_start,
                    msa_end_col=msa_end,
                )

                rec.update({
                    "Target_MSA_Start": mapped["msa_start"],
                    "Target_MSA_End": mapped["msa_end"],
                    "Target_Seq_Start": mapped["seq_start"],
                    "Target_Seq_End": mapped["seq_end"],
                    "Target_Seq_Len": mapped["seq_len"],
                    "Target_Frag": mapped["frag_pure"],
                    "Target_Aligned_Frag": mapped["frag_raw"],
                    "Gap_Count_In_MSA_Window": mapped["gaps"],
                    "Map_Status": mapped["status"],
                })

                if mapped["status"] == "OK":
                    summary["num_mapped_ok"] += 1

                    seg_start = int(mapped["seq_start"])
                    seg_end = int(mapped["seq_end"])
                    seg_len = int(mapped["seq_len"])

                    overlap_metrics = compute_class_metrics(
                        seg_start=seg_start,
                        seg_end=seg_end,
                        seg_len=seg_len,
                        class_seed_residues=class_seed_residues,
                    )
                    rec.update(overlap_metrics)

                    if overlap_metrics["Any_Start2Fold_Overlap_Count"] > 0:
                        summary["num_any_overlap_hits"] += 1

                    for key in CLASS_KEYS:
                        c = overlap_metrics[f"{key}_segment_overlap_count"]
                        if c >= 1:
                            summary[f"{key}_segment_hits_ge1"] += 1
                        if c >= 2:
                            summary[f"{key}_segment_hits_ge2"] += 1

                    # ---------- random sampling ----------
                    rand_windows = sample_random_windows(
                        seed_start=global_start,
                        seed_end=global_end_expected,
                        seg_len=seg_len,
                        n_samples=RANDOM_SAMPLES_PER_SEGMENT,
                        rng=rng,
                    )

                    summary["num_random_windows"] += len(rand_windows)

                    for r_idx, (r_start, r_end) in enumerate(rand_windows, start=1):
                        rrec = {
                            "segment_uid": segment_uid,
                            "random_rank": r_idx,
                            "STF_ID": stf_id,
                            "pfam_id": pfam_id,
                            "seed_seq_name": safe_str(task_row.get("seed_seq_name", "")),
                            "seed_start": global_start,
                            "seed_end": global_end_expected,
                            "global_row_index_1based": global_idx + 1,
                            "segment_rank": seg_rank,
                            "Target_Pfam_Side": pfam_side,
                            "real_Target_Seq_Start": seg_start,
                            "real_Target_Seq_End": seg_end,
                            "real_Target_Seq_Len": seg_len,
                            "random_Seq_Start": r_start,
                            "random_Seq_End": r_end,
                            "random_Seq_Len": seg_len,
                        }
                        rrec.update(
                            compute_class_metrics(
                                seg_start=r_start,
                                seg_end=r_end,
                                seg_len=seg_len,
                                class_seed_residues=class_seed_residues,
                            )
                        )
                        random_records.append(rrec)

                else:
                    for key in CLASS_KEYS:
                        rec[f"{key}_seed_residue_count"] = len(class_seed_residues[key])
                        rec[f"{key}_segment_overlap_count"] = 0
                        rec[f"{key}_segment_overlap_list"] = "[]"
                        rec[f"{key}_segment_overlap_frac"] = np.nan
                        rec[f"{key}_segment_longest_run"] = 0
                        rec[f"{key}_segment_longest_run_frac"] = np.nan
                        rec[f"{key}_segment_num_runs"] = 0
                    rec["Any_Start2Fold_Overlap_Count"] = 0
                    rec["Any_Start2Fold_Overlap_List"] = "[]"
                    rec["Any_Start2Fold_Overlap_Frac"] = np.nan
                    rec["Any_Start2Fold_Longest_Run"] = 0
                    rec["Any_Start2Fold_Longest_Run_Frac"] = np.nan
                    rec["Any_Start2Fold_Num_Runs"] = 0
                    rec["Any_Start2Fold_Overlap_Status"] = "NO"

                actual_records.append(rec)

        summary["status"] = "OK"
        summary["message"] = "Success"
        return actual_records, random_records, summary

    except Exception as exc:
        summary["status"] = "FAIL"
        summary["message"] = f"{type(exc).__name__}: {exc}"
        return actual_records, random_records, summary


# ---------------- Summaries ----------------
def build_segment_random_test_summary(
    actual_df: pd.DataFrame,
    random_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each real mapped segment, compare observed metrics to its 1000 matched random windows.
    """
    if actual_df.empty or random_df.empty:
        return pd.DataFrame()

    actual_ok = actual_df[actual_df["Map_Status"] == "OK"].copy()
    if actual_ok.empty:
        return pd.DataFrame()

    rows = []

    for _, arow in actual_ok.iterrows():
        seg_uid = arow["segment_uid"]
        sub = random_df[random_df["segment_uid"] == seg_uid].copy()
        if sub.empty:
            continue

        row = {
            "segment_uid": seg_uid,
            "STF_ID": arow["STF_ID"],
            "pfam_id": arow["pfam_id"],
            "seed_seq_name": arow["seed_seq_name"],
            "global_row_index_1based": arow["global_row_index_1based"],
            "segment_rank": arow["segment_rank"],
            "Target_Seq_Start": arow["Target_Seq_Start"],
            "Target_Seq_End": arow["Target_Seq_End"],
            "Target_Seq_Len": arow["Target_Seq_Len"],
        }

        metric_specs = [
            ("Any", "Any_Start2Fold_Overlap_Count", "Any_Start2Fold_Overlap_Frac", "Any_Start2Fold_Longest_Run_Frac"),
        ] + [
            (
                key,
                f"{key}_segment_overlap_count",
                f"{key}_segment_overlap_frac",
                f"{key}_segment_longest_run_frac",
            )
            for key in CLASS_KEYS
        ]

        for label, count_col, frac_col, runfrac_col in metric_specs:
            real_count = pd.to_numeric(pd.Series([arow[count_col]]), errors="coerce").fillna(0).iloc[0]
            real_frac = pd.to_numeric(pd.Series([arow[frac_col]]), errors="coerce").fillna(0).iloc[0]
            real_runfrac = pd.to_numeric(pd.Series([arow[runfrac_col]]), errors="coerce").fillna(0).iloc[0]

            rand_counts = pd.to_numeric(sub[count_col], errors="coerce").fillna(0).to_numpy()
            rand_fracs = pd.to_numeric(sub[frac_col], errors="coerce").fillna(0).to_numpy()
            rand_runfracs = pd.to_numeric(sub[runfrac_col], errors="coerce").fillna(0).to_numpy()

            row[f"{label}_real_count"] = real_count
            row[f"{label}_random_mean_count"] = float(np.mean(rand_counts))
            row[f"{label}_random_median_count"] = float(np.median(rand_counts))
            row[f"{label}_count_empirical_p_ge"] = empirical_p_ge(real_count, rand_counts)

            row[f"{label}_real_frac"] = real_frac
            row[f"{label}_random_mean_frac"] = float(np.mean(rand_fracs))
            row[f"{label}_random_median_frac"] = float(np.median(rand_fracs))
            row[f"{label}_frac_empirical_p_ge"] = empirical_p_ge(real_frac, rand_fracs)

            row[f"{label}_real_runfrac"] = real_runfrac
            row[f"{label}_random_mean_runfrac"] = float(np.mean(rand_runfracs))
            row[f"{label}_random_median_runfrac"] = float(np.median(rand_runfracs))
            row[f"{label}_runfrac_empirical_p_ge"] = empirical_p_ge(real_runfrac, rand_runfracs)

        rows.append(row)

    out = pd.DataFrame(rows)
    return out


def build_overall_class_enrichment_summary(
    actual_df: pd.DataFrame,
    random_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare observed real segments vs all matched random windows.
    """
    if actual_df.empty:
        return pd.DataFrame()

    actual_ok = actual_df[actual_df["Map_Status"] == "OK"].copy()

    metric_specs = [
        ("Any", "Any_Start2Fold_Overlap_Count", "Any_Start2Fold_Overlap_Frac", "Any_Start2Fold_Longest_Run_Frac"),
    ] + [
        (
            key,
            f"{key}_segment_overlap_count",
            f"{key}_segment_overlap_frac",
            f"{key}_segment_longest_run_frac",
        )
        for key in CLASS_KEYS
    ]

    rows = []

    for label, count_col, frac_col, runfrac_col in metric_specs:
        a_counts = pd.to_numeric(actual_ok[count_col], errors="coerce").fillna(0)
        a_fracs = pd.to_numeric(actual_ok[frac_col], errors="coerce").fillna(0)
        a_runfracs = pd.to_numeric(actual_ok[runfrac_col], errors="coerce").fillna(0)

        if random_df.empty:
            r_counts = pd.Series(dtype=float)
            r_fracs = pd.Series(dtype=float)
            r_runfracs = pd.Series(dtype=float)
        else:
            r_counts = pd.to_numeric(random_df[count_col], errors="coerce").fillna(0)
            r_fracs = pd.to_numeric(random_df[frac_col], errors="coerce").fillna(0)
            r_runfracs = pd.to_numeric(random_df[runfrac_col], errors="coerce").fillna(0)

        row = {
            "label": label,
            "n_real_segments": int(len(a_counts)),
            "n_random_windows": int(len(r_counts)),
            "real_mean_overlap_count": float(a_counts.mean()) if len(a_counts) else np.nan,
            "random_mean_overlap_count": float(r_counts.mean()) if len(r_counts) else np.nan,
            "real_mean_overlap_frac": float(a_fracs.mean()) if len(a_fracs) else np.nan,
            "random_mean_overlap_frac": float(r_fracs.mean()) if len(r_fracs) else np.nan,
            "real_mean_longest_run_frac": float(a_runfracs.mean()) if len(a_runfracs) else np.nan,
            "random_mean_longest_run_frac": float(r_runfracs.mean()) if len(r_runfracs) else np.nan,
            "count_mean_enrichment": safe_div(a_counts.mean(), r_counts.mean() if len(r_counts) else np.nan),
            "frac_mean_enrichment": safe_div(a_fracs.mean(), r_fracs.mean() if len(r_fracs) else np.nan),
            "runfrac_mean_enrichment": safe_div(a_runfracs.mean(), r_runfracs.mean() if len(r_runfracs) else np.nan),
        }

        for thr in HIT_THRESHOLDS:
            a_hit_rate = float((a_counts >= thr).mean()) if len(a_counts) else np.nan
            r_hit_rate = float((r_counts >= thr).mean()) if len(r_counts) else np.nan
            row[f"real_hit_rate_ge{thr}"] = a_hit_rate
            row[f"random_hit_rate_ge{thr}"] = r_hit_rate
            row[f"hit_enrichment_ge{thr}"] = safe_div(a_hit_rate, r_hit_rate)

        rows.append(row)

    out = pd.DataFrame(rows)
    return out


def build_class_specific_actual_summary(actual_df: pd.DataFrame, label: str, count_col: str) -> pd.DataFrame:
    if actual_df.empty:
        return pd.DataFrame()

    df = actual_df[actual_df["Map_Status"] == "OK"].copy()
    df = df[pd.to_numeric(df[count_col], errors="coerce").fillna(0).astype(int) > 0].copy()
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby(["STF_ID", "uniprot_ac_base", "UniProt_Range"], as_index=False)
        .agg(
            n_segments=("segment_uid", "count"),
            n_global_rows=("global_row_index_1based", pd.Series.nunique),
            matched_pfams=("pfam_id", lambda s: ";".join(sorted(pd.Series(s).astype(str).unique()))),
            matched_seed_seq_names=("seed_seq_name", lambda s: ";".join(sorted(pd.Series(s).astype(str).unique()))),
            best_overlap_count=(count_col, lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max())),
            best_overlap_frac=(f"{label}_frac", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).max())),
            best_run_frac=(f"{label}_runfrac", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).max())),
        )
        .sort_values(["n_global_rows", "n_segments", "best_overlap_frac"], ascending=[False, False, False])
    )
    return out


def write_outputs(
    actual_df: pd.DataFrame,
    random_df: pd.DataFrame,
    task_summary_df: pd.DataFrame,
) -> None:
    actual_all_path = OUTPUT_DIR / "all_mapped_segments_with_start2fold_metrics.csv"
    actual_hits_path = OUTPUT_DIR / "global_segments_with_any_start2fold_overlap.csv"
    random_all_path = OUTPUT_DIR / "random_sampled_windows_all.csv"
    segment_test_path = OUTPUT_DIR / "segment_random_test_summary.csv"
    overall_summary_path = OUTPUT_DIR / "overall_class_enrichment_summary.csv"
    task_summary_path = OUTPUT_DIR / "task_mapping_summary.csv"
    summary_txt_path = OUTPUT_DIR / "summary.txt"

    actual_df.to_csv(actual_all_path, index=False, encoding="utf-8-sig")
    random_df.to_csv(random_all_path, index=False, encoding="utf-8-sig")
    task_summary_df.to_csv(task_summary_path, index=False, encoding="utf-8-sig")

    actual_ok = actual_df[actual_df["Map_Status"] == "OK"].copy()
    actual_hits = actual_ok[
        pd.to_numeric(actual_ok["Any_Start2Fold_Overlap_Count"], errors="coerce").fillna(0).astype(int) > 0
    ].copy()
    actual_hits.to_csv(actual_hits_path, index=False, encoding="utf-8-sig")

    segment_test_df = build_segment_random_test_summary(actual_df, random_df)
    segment_test_df.to_csv(segment_test_path, index=False, encoding="utf-8-sig")

    overall_summary_df = build_overall_class_enrichment_summary(actual_df, random_df)
    overall_summary_df.to_csv(overall_summary_path, index=False, encoding="utf-8-sig")

    # class-specific detail + summary
    class_summary_lines = []
    class_specs = [
        ("Any_Start2Fold", "Any_Start2Fold_Overlap_Count", "Any_Start2Fold_Overlap_Frac", "Any_Start2Fold_Longest_Run_Frac"),
    ] + [
        (key, f"{key}_segment_overlap_count", f"{key}_segment_overlap_frac", f"{key}_segment_longest_run_frac")
        for key in CLASS_KEYS
    ]

    for label, count_col, frac_col, runfrac_col in class_specs:
        col_label = label
        tmp = actual_ok.copy()
        tmp[f"{label}_frac"] = pd.to_numeric(tmp[frac_col], errors="coerce").fillna(0)
        tmp[f"{label}_runfrac"] = pd.to_numeric(tmp[runfrac_col], errors="coerce").fillna(0)

        class_hits = tmp[pd.to_numeric(tmp[count_col], errors="coerce").fillna(0).astype(int) > 0].copy()

        detail_path = OUTPUT_DIR / f"{label}_actual_hit_segments.csv"
        summary_path = OUTPUT_DIR / f"{label}_actual_hit_summary.csv"

        class_hits.to_csv(detail_path, index=False, encoding="utf-8-sig")
        build_class_specific_actual_summary(tmp, label, count_col).to_csv(summary_path, index=False, encoding="utf-8-sig")

        class_summary_lines.extend([
            f"{label} actual hit segments: {len(class_hits)}",
            f"{label} unique STF_ID hit: {class_hits['STF_ID'].nunique() if not class_hits.empty else 0}",
            f"{label} detail file: {detail_path}",
            f"{label} summary file: {summary_path}",
        ])

    ok_tasks = int((task_summary_df["status"] == "OK").sum()) if not task_summary_df.empty else 0
    fail_tasks = int((task_summary_df["status"] == "FAIL").sum()) if not task_summary_df.empty else 0

    summary_lines = [
        f"Tasks total: {len(task_summary_df)}",
        f"Task success: {ok_tasks}",
        f"Task fail: {fail_tasks}",
        f"Actual mapped segments (all): {len(actual_df)}",
        f"Actual mapped OK segments: {len(actual_ok)}",
        f"Actual segments with any Start2Fold overlap: {len(actual_hits)}",
        f"Unique STF_ID with any overlap: {actual_hits['STF_ID'].nunique() if not actual_hits.empty else 0}",
        f"Random windows total: {len(random_df)}",
        f"Actual all file: {actual_all_path}",
        f"Actual hit file: {actual_hits_path}",
        f"Random windows file: {random_all_path}",
        f"Per-segment random test summary: {segment_test_path}",
        f"Overall class enrichment summary: {overall_summary_path}",
        f"Task mapping summary: {task_summary_path}",
    ] + class_summary_lines

    summary_txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    for line in summary_lines:
        print("[INFO]", line)


# ---------------- Main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map ProDive HMM segments to Start2Fold residues and generate same-seed same-length random windows.")
    p.add_argument("--global-score-file", type=Path, default=GLOBAL_SCORE_FILE, help="ProDive global high-score summary CSV.")
    p.add_argument("--covered-protein-file", type=Path, default=COVERED_PROTEIN_FILE, help="start2fold_proteins_covered_in_seed.csv from step 03.")
    p.add_argument("--detail-file", type=Path, default=DETAIL_FILE, help="start2fold_seed_match_detail.csv from step 03.")
    p.add_argument("--pfam-dir", type=Path, default=PFAM_DIR, help="PfamA_seed root directory.")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    p.add_argument("--task-limit", type=int, default=TASK_LIMIT, help="Optional task limit for testing.")
    p.add_argument("--random-samples-per-segment", type=int, default=RANDOM_SAMPLES_PER_SEGMENT, help="Number of random windows per mapped real segment.")
    p.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed.")
    p.add_argument("--include-ac-only-covered", action="store_true", help="Include AC-only covered proteins instead of restricting to residue-overlap covered proteins.")
    return p.parse_args()

def main() -> None:
    global GLOBAL_SCORE_FILE, COVERED_PROTEIN_FILE, DETAIL_FILE, PFAM_DIR, OUTPUT_DIR
    global TASK_LIMIT, ONLY_RESIDUE_OVERLAP_COVERED, RANDOM_SAMPLES_PER_SEGMENT, RANDOM_SEED
    args = parse_args()
    GLOBAL_SCORE_FILE = args.global_score_file
    COVERED_PROTEIN_FILE = args.covered_protein_file
    DETAIL_FILE = args.detail_file
    PFAM_DIR = args.pfam_dir
    OUTPUT_DIR = args.output_dir
    TASK_LIMIT = args.task_limit
    RANDOM_SAMPLES_PER_SEGMENT = args.random_samples_per_segment
    RANDOM_SEED = args.random_seed
    ONLY_RESIDUE_OVERLAP_COVERED = not args.include_ac_only_covered
    ensure_dir(OUTPUT_DIR)

    rng = np.random.default_rng(RANDOM_SEED)

    print("[INFO] Reading input files...")
    covered_df = read_covered_proteins(COVERED_PROTEIN_FILE)
    detail_df = read_detail_file(DETAIL_FILE)
    task_df = build_tasks(covered_df, detail_df)
    global_df = read_global_score(GLOBAL_SCORE_FILE)
    global_index = build_global_index(global_df)

    print(f"[INFO] Covered Start2Fold proteins retained: {len(covered_df)}")
    print(f"[INFO] Mapping tasks retained: {len(task_df)}")
    print(f"[INFO] Global score rows: {len(global_df)}")
    print(f"[INFO] Unique Pfams in global index: {len(global_index)}")
    print(f"[INFO] Random samples per real segment: {RANDOM_SAMPLES_PER_SEGMENT}")

    pfam_cache: Dict[str, dict] = {}
    all_actual_records: List[dict] = []
    all_random_records: List[dict] = []
    summaries: List[dict] = []

    total = len(task_df)
    for i, (_, task_row) in enumerate(task_df.iterrows(), start=1):
        print(
            f"[{i}/{total}] Mapping STF={safe_str(task_row.get('STF_ID', ''))} "
            f"pfam={safe_str(task_row.get('pfam_id', ''))} "
            f"seed={safe_str(task_row.get('seed_seq_name', ''))}"
        )
        actual_records, random_records, summary = process_one_task(
            task_row=task_row,
            global_df=global_df,
            global_index=global_index,
            pfam_cache=pfam_cache,
            rng=rng,
        )
        all_actual_records.extend(actual_records)
        all_random_records.extend(random_records)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    actual_df = pd.DataFrame(all_actual_records)
    random_df = pd.DataFrame(all_random_records)

    if actual_df.empty:
        print("[WARN] No mapped records produced.")
        write_outputs(
            actual_df=pd.DataFrame(columns=["Map_Status", "Any_Start2Fold_Overlap_Count", "STF_ID"]),
            random_df=pd.DataFrame(),
            task_summary_df=summary_df,
        )
        return

    write_outputs(
        actual_df=actual_df,
        random_df=random_df,
        task_summary_df=summary_df,
    )

    print("[INFO] Done.")
    print(f"[INFO] Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()