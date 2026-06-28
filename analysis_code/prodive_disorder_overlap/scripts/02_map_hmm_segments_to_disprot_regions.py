#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ========================================================
# Default paths. Override them from the command line when needed.
# ========================================================
DEFAULT_GLOBAL_SCORE_FILE = Path("<PRODIVE_DATA_ROOT>/shared/global_high_score_summary_fin.csv")
DEFAULT_COVERED_REGION_FILE = Path("CHANGE_ME")
DEFAULT_DETAIL_FILE = Path("CHANGE_ME")
DEFAULT_PFAM_DIR = Path("<PRODIVE_DATA_ROOT>/shared/PfamA_seed")
DEFAULT_OUTPUT_DIR = Path("CHANGE_ME")

GLOBAL_SCORE_FILE = DEFAULT_GLOBAL_SCORE_FILE
COVERED_REGION_FILE = DEFAULT_COVERED_REGION_FILE
DETAIL_FILE = DEFAULT_DETAIL_FILE
PFAM_DIR = DEFAULT_PFAM_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
TASK_LIMIT = None
# ========================================================


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


def ungap(seq: str) -> str:
    return seq.replace(".", "").replace("-", "")


def interval_overlap(a1: int, a2: int, b1: int, b2: int) -> Optional[Tuple[int, int]]:
    lo = max(a1, b1)
    hi = min(a2, b2)
    if lo <= hi:
        return lo, hi
    return None


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

    uniq: List[str] = []
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


def map_msa_interval_to_real_seq(aligned_seq: str, global_start: int, msa_start_col: int, msa_end_col: int) -> dict:
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
    main_hmm = safe_str(match_row.get("Main_HMM", ""))
    sub_hmm = safe_str(match_row.get("Sub_HMM", ""))
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

def read_covered_regions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing covered-region file: {path}")
    df = pd.read_csv(path, dtype=str)
    for col in df.columns:
        df[col] = df[col].map(safe_str)

    if "region_id" not in df.columns:
        raise ValueError(f"covered-region file missing region_id. Columns: {list(df.columns)}")

    if "num_seed_overlap_hits" in df.columns:
        keep = pd.to_numeric(df["num_seed_overlap_hits"], errors="coerce").fillna(0) > 0
        df = df[keep].copy()
    elif "coverage_basis" in df.columns:
        df = df[df["coverage_basis"] == "residue_overlap"].copy()

    return df.reset_index(drop=True)


def read_detail_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing detail file: {path}")
    df = pd.read_csv(path, dtype=str)
    for col in df.columns:
        df[col] = df[col].map(safe_str)

    required = [
        "region_id",
        "disprot_id",
        "pfam_id",
        "seed_seq_name",
        "seed_ac_base",
        "uniprot_ac_base",
        "region_start",
        "region_end",
        "seed_start",
        "seed_end",
        "overlap_len",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"detail file missing columns: {missing}")

    df["overlap_len_num"] = pd.to_numeric(df["overlap_len"], errors="coerce").fillna(0).astype(int)
    df["region_start_num"] = pd.to_numeric(df["region_start"], errors="coerce")
    df["region_end_num"] = pd.to_numeric(df["region_end"], errors="coerce")
    df["seed_start_num"] = pd.to_numeric(df["seed_start"], errors="coerce")
    df["seed_end_num"] = pd.to_numeric(df["seed_end"], errors="coerce")
    return df


def build_tasks(covered_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    covered_ids = set(covered_df["region_id"].astype(str))
    task_df = detail_df[detail_df["region_id"].isin(covered_ids)].copy()
    task_df = task_df[task_df["overlap_len_num"] > 0].copy()
    task_df = task_df[task_df["pfam_id"] != ""].copy()
    task_df = task_df[task_df["seed_seq_name"] != ""].copy()

    # Normalize numeric columns
    for col in ["region_start_num", "region_end_num", "seed_start_num", "seed_end_num"]:
        task_df = task_df[task_df[col].notna()].copy()
        task_df[col] = task_df[col].astype(int)

    dedupe_cols = [
        "disprot_id",
        "region_id",
        "pfam_id",
        "seed_seq_name",
        "seed_ac_base",
        "uniprot_ac_base",
        "region_start",
        "region_end",
        "seed_start",
        "seed_end",
        "term_name",
    ]
    available_dedupe_cols = [c for c in dedupe_cols if c in task_df.columns]
    task_df = task_df.drop_duplicates(subset=available_dedupe_cols).reset_index(drop=True)

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
        main = safe_str(row["Main_HMM"])
        sub = safe_str(row["Sub_HMM"])
        if main:
            index[main].add(idx)
        if sub:
            index[sub].add(idx)
    return {k: sorted(v) for k, v in index.items()}


# ---------------- Core mapping ----------------

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


def process_one_task(task_row: pd.Series, global_df: pd.DataFrame, global_index: Dict[str, List[int]], pfam_cache: Dict[str, dict]) -> Tuple[List[dict], dict]:
    pfam_id = safe_str(task_row["pfam_id"])
    region_id = safe_str(task_row["region_id"])

    summary = {
        "disprot_id": safe_str(task_row.get("disprot_id", "")),
        "region_id": region_id,
        "pfam_id": pfam_id,
        "seed_seq_name": safe_str(task_row.get("seed_seq_name", "")),
        "term_name": safe_str(task_row.get("term_name", "")),
        "status": "INIT",
        "message": "",
        "num_candidate_global_rows": 0,
        "num_target_segments": 0,
        "num_mapped_ok": 0,
        "num_overlap_hits": 0,
    }

    records: List[dict] = []

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

        for global_idx in global_index.get(pfam_id, []):
            grow = global_df.loc[global_idx]
            segments = extract_target_hmm_segments(grow, pfam_id)
            if not segments:
                continue

            for seg_rank, (pfam_side, hmm_start, hmm_end) in enumerate(segments, start=1):
                summary["num_target_segments"] += 1

                rec = {
                    "disprot_id": safe_str(task_row.get("disprot_id", "")),
                    "region_id": region_id,
                    "uniprot_ac_base": safe_str(task_row.get("uniprot_ac_base", "")),
                    "term_name": safe_str(task_row.get("term_name", "")),
                    "protein_name": safe_str(task_row.get("protein_name", "")),
                    "gene_name": safe_str(task_row.get("gene_name", "")),
                    "organism": safe_str(task_row.get("organism", "")),
                    "pfam_id": pfam_id,
                    "seed_seq_name": safe_str(task_row.get("seed_seq_name", "")),
                    "seed_ac_base": safe_str(task_row.get("seed_ac_base", "")),
                    "seed_start": global_start,
                    "seed_end": global_end_expected,
                    "region_start": safe_int(task_row.get("region_start_num"), None),
                    "region_end": safe_int(task_row.get("region_end_num"), None),
                    "seed_region_overlap_len": safe_int(task_row.get("overlap_len_num"), 0),
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
                        "DisProt_Overlap_Start": np.nan,
                        "DisProt_Overlap_End": np.nan,
                        "DisProt_Overlap_Len": 0,
                        "Overlap_Status": "NO",
                    })
                    records.append(rec)
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
                    ov = interval_overlap(
                        int(task_row["region_start_num"]),
                        int(task_row["region_end_num"]),
                        int(mapped["seq_start"]),
                        int(mapped["seq_end"]),
                    )
                    if ov is not None:
                        ov_s, ov_e = ov
                        ov_len = ov_e - ov_s + 1
                        rec.update({
                            "DisProt_Overlap_Start": ov_s,
                            "DisProt_Overlap_End": ov_e,
                            "DisProt_Overlap_Len": ov_len,
                            "Overlap_Status": "YES",
                        })
                        summary["num_overlap_hits"] += 1
                    else:
                        rec.update({
                            "DisProt_Overlap_Start": np.nan,
                            "DisProt_Overlap_End": np.nan,
                            "DisProt_Overlap_Len": 0,
                            "Overlap_Status": "NO",
                        })
                else:
                    rec.update({
                        "DisProt_Overlap_Start": np.nan,
                        "DisProt_Overlap_End": np.nan,
                        "DisProt_Overlap_Len": 0,
                        "Overlap_Status": "NO",
                    })

                records.append(rec)

        summary["status"] = "OK"
        summary["message"] = "Success"
        return records, summary

    except Exception as exc:
        summary["status"] = "FAIL"
        summary["message"] = f"{type(exc).__name__}: {exc}"
        return records, summary


def summarize_overlap_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "disprot_id",
            "region_id",
            "uniprot_ac_base",
            "term_name",
            "region_start",
            "region_end",
            "n_overlap_segments",
            "n_global_rows",
            "n_pfams",
            "matched_pfams",
            "matched_seed_seq_names",
            "best_overlap_len",
        ])

    out = (
        df.groupby(["disprot_id", "region_id", "uniprot_ac_base", "term_name", "region_start", "region_end"], as_index=False)
        .agg(
            n_overlap_segments=("region_id", "count"),
            n_global_rows=("global_row_index_1based", pd.Series.nunique),
            n_pfams=("pfam_id", pd.Series.nunique),
            matched_pfams=("pfam_id", lambda s: ";".join(sorted(pd.Series(s).astype(str).unique()))),
            matched_seed_seq_names=("seed_seq_name", lambda s: ";".join(sorted(pd.Series(s).astype(str).unique()))),
            best_overlap_len=("DisProt_Overlap_Len", "max"),
        )
        .sort_values(["n_global_rows", "n_overlap_segments", "best_overlap_len", "region_id"], ascending=[False, False, False, True])
    )
    return out


def write_outputs(all_mapped_df: pd.DataFrame) -> None:
    overlap_df = all_mapped_df[
        (all_mapped_df["Map_Status"] == "OK") &
        (all_mapped_df["Overlap_Status"] == "YES")
    ].copy()

    overlap_all_path = OUTPUT_DIR / "global_segments_overlapping_covered_disprot_regions_all.csv"
    overlap_disorder_path = OUTPUT_DIR / "global_segments_overlapping_covered_disprot_regions_term_disorder_only.csv"
    summary_all_path = OUTPUT_DIR / "covered_disprot_regions_supported_by_global_segments_all_summary.csv"
    summary_disorder_path = OUTPUT_DIR / "covered_disprot_regions_supported_by_global_segments_term_disorder_only_summary.csv"
    debug_all_mapped_path = OUTPUT_DIR / "all_mapped_segments_with_overlap_status.csv"

    all_mapped_df.to_csv(debug_all_mapped_path, index=False, encoding="utf-8-sig")
    overlap_df.to_csv(overlap_all_path, index=False, encoding="utf-8-sig")

    disorder_mask = overlap_df["term_name"].astype(str).str.strip().str.lower() == "disorder"
    overlap_disorder_df = overlap_df[disorder_mask].copy()
    overlap_disorder_df.to_csv(overlap_disorder_path, index=False, encoding="utf-8-sig")

    summarize_overlap_df(overlap_df).to_csv(summary_all_path, index=False, encoding="utf-8-sig")
    summarize_overlap_df(overlap_disorder_df).to_csv(summary_disorder_path, index=False, encoding="utf-8-sig")

    summary_lines = [
        f"All mapped segments (debug): {len(all_mapped_df)}",
        f"Mapped OK segments: {(all_mapped_df['Map_Status'] == 'OK').sum() if not all_mapped_df.empty else 0}",
        f"Overlap hits (all terms): {len(overlap_df)}",
        f"Overlap hits (term_name=disorder): {len(overlap_disorder_df)}",
        f"Unique covered regions hit (all terms): {overlap_df['region_id'].nunique() if not overlap_df.empty else 0}",
        f"Unique covered regions hit (term_name=disorder): {overlap_disorder_df['region_id'].nunique() if not overlap_disorder_df.empty else 0}",
        f"Overlap all file: {overlap_all_path}",
        f"Overlap disorder-only file: {overlap_disorder_path}",
        f"Summary all file: {summary_all_path}",
        f"Summary disorder-only file: {summary_disorder_path}",
        f"Debug all-mapped file: {debug_all_mapped_path}",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    for line in summary_lines:
        print("[INFO]", line)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map ProDive HMM segments to seed-sequence coordinates and test whether "
            "they overlap DisProt regions already covered by Pfam seed sequences."
        )
    )
    parser.add_argument('--global-score-file', type=Path, default=DEFAULT_GLOBAL_SCORE_FILE,
                        help='Path to global_high_score_summary_fin.csv.')
    parser.add_argument('--covered-region-file', type=Path, default=DEFAULT_COVERED_REGION_FILE,
                        help='Path to disprot_regions_covered_in_seed.csv from step 01.')
    parser.add_argument('--detail-file', type=Path, default=DEFAULT_DETAIL_FILE,
                        help='Path to disprot_seed_match_detail.csv from step 01.')
    parser.add_argument('--pfam-dir', type=Path, default=DEFAULT_PFAM_DIR,
                        help='Root directory containing PFxxxxx/PFxxxxx.hhm and seed alignments.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory.')
    parser.add_argument('--task-limit', type=int, default=None,
                        help='Optional limit for debugging. Omit to process all tasks.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global GLOBAL_SCORE_FILE, COVERED_REGION_FILE, DETAIL_FILE, PFAM_DIR, OUTPUT_DIR, TASK_LIMIT
    GLOBAL_SCORE_FILE = args.global_score_file
    COVERED_REGION_FILE = args.covered_region_file
    DETAIL_FILE = args.detail_file
    PFAM_DIR = args.pfam_dir
    OUTPUT_DIR = args.output_dir
    TASK_LIMIT = args.task_limit

    ensure_dir(OUTPUT_DIR)

    print("[INFO] Reading input files...")
    covered_df = read_covered_regions(COVERED_REGION_FILE)
    detail_df = read_detail_file(DETAIL_FILE)
    task_df = build_tasks(covered_df, detail_df)
    global_df = read_global_score(GLOBAL_SCORE_FILE)
    global_index = build_global_index(global_df)

    print(f"[INFO] Covered regions retained: {len(covered_df)}")
    print(f"[INFO] Mapping tasks retained: {len(task_df)}")
    print(f"[INFO] Global score rows: {len(global_df)}")
    print(f"[INFO] Unique Pfams in global index: {len(global_index)}")

    pfam_cache: Dict[str, dict] = {}
    all_records: List[dict] = []
    summaries: List[dict] = []

    total = len(task_df)
    for i, (_, task_row) in enumerate(task_df.iterrows(), start=1):
        print(
            f"[{i}/{total}] Mapping region={safe_str(task_row.get('region_id', ''))} "
            f"pfam={safe_str(task_row.get('pfam_id', ''))} seed={safe_str(task_row.get('seed_seq_name', ''))}"
        )
        records, summary = process_one_task(task_row, global_df, global_index, pfam_cache)
        all_records.extend(records)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / "task_mapping_summary.csv", index=False, encoding="utf-8-sig")

    all_mapped_df = pd.DataFrame(all_records)
    if all_mapped_df.empty:
        print("[WARN] No mapped records produced.")
        write_outputs(pd.DataFrame(columns=["term_name", "Map_Status", "Overlap_Status", "region_id"]))
        return

    write_outputs(all_mapped_df)

    ok_tasks = int((summary_df["status"] == "OK").sum()) if not summary_df.empty else 0
    fail_tasks = int((summary_df["status"] == "FAIL").sum()) if not summary_df.empty else 0
    print("[INFO] Done.")
    print(f"[INFO] Task success: {ok_tasks}")
    print(f"[INFO] Task fail: {fail_tasks}")
    print(f"[INFO] Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
