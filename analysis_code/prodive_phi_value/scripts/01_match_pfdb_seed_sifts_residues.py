#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Match PFDB 2-state phi-value proteins to Pfam seed sequences through PDBe/SIFTS.

The script runs two numbering modes independently:
  1. author numbering: PFDB residue ranges are interpreted as author_residue_number.
  2. residue numbering: PFDB residue ranges are interpreted as residue_number.

The output file names and CSV schemas are intentionally kept compatible with the
original analysis script.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

DEFAULT_FILE_2S = Path("CHANGE_ME")
DEFAULT_PFAM_SEED_DIR = Path("<PRODIVE_DATA_ROOT>/shared/PfamA_seed")
DEFAULT_OUT_ROOT = Path("CHANGE_ME")
DEFAULT_PDBE_API_BASE = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("\xa0", " ").strip()


def strip_version(ac: str) -> str:
    s = clean_text(ac)
    return s.split(".", 1)[0] if s else ""


def interval_overlap(a1: int, a2: int, b1: int, b2: int) -> Optional[Tuple[int, int]]:
    lo = max(a1, b1)
    hi = min(a2, b2)
    if lo <= hi:
        return lo, hi
    return None


def read_2s_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Protein short name": "protein_short_name",
        "PDB": "pdb_raw",
    })
    df["source_file"] = path.name
    return df


def extract_pdb_id(pdb_raw: str) -> str:
    s = clean_text(pdb_raw)
    m = re.search(r"\b([0-9][A-Za-z0-9]{3})\b", s)
    return m.group(1).upper() if m else ""


def parse_pfdb_pdb_field(pdb_raw: str) -> Tuple[str, str, Optional[int], Optional[int]]:
    s = clean_text(pdb_raw)
    pdb_id = extract_pdb_id(s)
    chain = ""
    start = None
    end = None

    m = re.search(r"\((.*?)\)", s)
    if not m:
        return pdb_id, chain, start, end

    inside = m.group(1).strip().replace("–", "-").replace("—", "-")

    m_chain_range = re.search(r"[Cc]hain\s+([A-Za-z0-9])\s*:\s*(-?\d+)\s*-\s*(-?\d+)", inside)
    if m_chain_range:
        chain = m_chain_range.group(1)
        start = int(m_chain_range.group(2))
        end = int(m_chain_range.group(3))
        if start > end:
            start, end = end, start
        return pdb_id, chain, start, end

    m_chain_only = re.search(r"[Cc]hain\s+([A-Za-z0-9])\b", inside)
    if m_chain_only:
        chain = m_chain_only.group(1)

    m_range = re.search(r"(-?\d+)\s*-\s*(-?\d+)", inside)
    if m_range:
        start = int(m_range.group(1))
        end = int(m_range.group(2))
        if start > end:
            start, end = end, start

    return pdb_id, chain, start, end


def standardize_pfdb_df(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if c in ["No.", "protein_short_name", "pdb_raw", "source_file"]]
    out = df[keep_cols].copy()
    if "No." not in out.columns:
        out["No."] = ""

    out["No."] = out["No."].map(clean_text)
    out["protein_short_name"] = out["protein_short_name"].map(clean_text)
    out["pdb_raw"] = out["pdb_raw"].map(clean_text)

    parsed = out["pdb_raw"].map(parse_pfdb_pdb_field)
    out["pdb_id"] = parsed.map(lambda x: x[0])
    out["pfdb_chain"] = parsed.map(lambda x: x[1])
    out["pfdb_pdb_start"] = parsed.map(lambda x: x[2])
    out["pfdb_pdb_end"] = parsed.map(lambda x: x[3])

    out = out[out["pdb_id"] != ""].copy().reset_index(drop=True)
    out["source_row_index_1based"] = [i + 1 for i in range(len(out))]
    out["source_local_row_id"] = [f"ROW_{i + 1:04d}" for i in range(len(out))]
    out["pfdb_row_id"] = [f"ROW_{i + 1:04d}" for i in range(len(out))]
    return out


GS_AC_RE = re.compile(r"^#=GS\s+(\S+)\s+AC\s+(\S+)")
SEQ_RANGE_RE = re.compile(r"^(?P<seqid>.+?)/(\d+)-(\d+)$")


def parse_sto_ac_rows(sto_path: Path) -> List[dict]:
    pfam_id = sto_path.parent.name
    rows: List[dict] = []
    with sto_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GS_AC_RE.match(line.strip())
            if not m:
                continue
            seq_name = m.group(1).strip()
            ac = m.group(2).strip()
            ac_base = strip_version(ac)

            seq_id = seq_name
            seed_start = None
            seed_end = None
            m2 = SEQ_RANGE_RE.match(seq_name)
            if m2:
                seq_id = m2.group("seqid")
                seed_start = int(m2.group(2))
                seed_end = int(m2.group(3))

            rows.append({
                "pfam_id": pfam_id,
                "sto_file": str(sto_path),
                "seed_seq_name": seq_name,
                "seed_seq_id": seq_id,
                "seed_ac_raw": ac,
                "seed_ac_base": ac_base,
                "seed_start": seed_start,
                "seed_end": seed_end,
            })
    return rows


def scan_all_seed_ac(seed_root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    sto_files = sorted(seed_root.glob("PF*/PF*.sto"))
    for i, sto in enumerate(sto_files, start=1):
        if i % 1000 == 0:
            print(f"[INFO] scanned sto files: {i}/{len(sto_files)}")
        rows.extend(parse_sto_ac_rows(sto))
    return pd.DataFrame(rows)


def cached_json_path(cache_dir: Path, pdb_id: str) -> Path:
    return cache_dir / f"{pdb_id.lower()}.json"


def fetch_pdbe_uniprot_mapping(
    pdb_id: str,
    session: requests.Session,
    cache_dir: Path,
    pdbe_api_base: str,
    timeout: int,
    request_sleep: float,
) -> dict:
    ensure_dir(cache_dir)
    cache_path = cached_json_path(cache_dir, pdb_id)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    url = pdbe_api_base + pdb_id.lower()
    r = session.get(url, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"PDBe API failed for {pdb_id}: HTTP {r.status_code} {r.text[:200]}")
    data = r.json()
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    time.sleep(request_sleep)
    return data


def flatten_pdbe_mapping_json_dual(pdb_id: str, data: dict) -> List[dict]:
    key = pdb_id.lower()
    if key not in data:
        key = pdb_id.upper()
    if key not in data:
        return []

    root = data[key]
    uniprot_block = root.get("UniProt", {})
    out: List[dict] = []

    for uniprot_ac, info in uniprot_block.items():
        mappings = info.get("mappings", [])
        for mp in mappings:
            try:
                auth_start = mp.get("start", {}).get("author_residue_number")
                auth_end = mp.get("end", {}).get("author_residue_number")
                start_ins = clean_text(mp.get("start", {}).get("author_insertion_code", ""))
                end_ins = clean_text(mp.get("end", {}).get("author_insertion_code", ""))
                res_start = mp.get("start", {}).get("residue_number")
                res_end = mp.get("end", {}).get("residue_number")
                unp_start = mp.get("unp_start")
                unp_end = mp.get("unp_end")
                chain_id = mp.get("chain_id") or mp.get("struct_asym_id") or ""
                entity_id = mp.get("entity_id")
            except Exception:
                continue

            if any(x is None for x in [unp_start, unp_end]):
                continue

            out.append({
                "pdb_id": pdb_id.upper(),
                "uniprot_ac_raw": clean_text(uniprot_ac),
                "uniprot_ac_base": strip_version(clean_text(uniprot_ac)),
                "entity_id": entity_id,
                "chain_id": clean_text(chain_id),
                "segment_auth_pdb_start": int(auth_start) if auth_start is not None else None,
                "segment_auth_pdb_end": int(auth_end) if auth_end is not None else None,
                "segment_auth_start_ins": start_ins,
                "segment_auth_end_ins": end_ins,
                "segment_res_pdb_start": int(res_start) if res_start is not None else None,
                "segment_res_pdb_end": int(res_end) if res_end is not None else None,
                "segment_unp_start": int(unp_start),
                "segment_unp_end": int(unp_end),
            })
    return out


def project_interval(
    pfdb_start: Optional[int],
    pfdb_end: Optional[int],
    seg_start: Optional[int],
    seg_end: Optional[int],
    seg_unp_start: int,
    seg_unp_end: int,
) -> Optional[Tuple[int, int, int, int]]:
    if seg_start is None or seg_end is None:
        return None

    if pfdb_start is None or pfdb_end is None:
        return seg_start, seg_end, seg_unp_start, seg_unp_end

    ov = interval_overlap(pfdb_start, pfdb_end, seg_start, seg_end)
    if ov is None:
        return None

    used_start, used_end = ov
    offset = seg_unp_start - seg_start
    mapped_unp_start = used_start + offset
    mapped_unp_end = used_end + offset

    mapped_unp_start = max(mapped_unp_start, seg_unp_start)
    mapped_unp_end = min(mapped_unp_end, seg_unp_end)
    if mapped_unp_start > mapped_unp_end:
        return None

    return used_start, used_end, mapped_unp_start, mapped_unp_end


def run_mode(
    mode_name: str,
    pfdb_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)

    seed_by_ac: Dict[str, List[dict]] = defaultdict(list)
    for row in seed_df.to_dict("records"):
        seed_by_ac[row["seed_ac_base"]].append(row)

    seg_by_pdb: Dict[str, List[dict]] = defaultdict(list)
    for row in seg_df.to_dict("records"):
        seg_by_pdb[row["pdb_id"]].append(row)

    detail_rows: List[dict] = []
    rows_with_any_sifts = 0
    rows_with_any_same_ac_seed = 0
    rows_with_any_residue_overlap = 0

    if mode_name == "author":
        seg_start_col = "segment_auth_pdb_start"
        seg_end_col = "segment_auth_pdb_end"
        seg_used_start_col = "used_auth_pdb_start"
        seg_used_end_col = "used_auth_pdb_end"
    else:
        seg_start_col = "segment_res_pdb_start"
        seg_end_col = "segment_res_pdb_end"
        seg_used_start_col = "used_res_pdb_start"
        seg_used_end_col = "used_res_pdb_end"

    print(f"[INFO] Running mode: {mode_name}")
    for idx, pfrow in pfdb_df.iterrows():
        if (idx + 1) % 25 == 0 or (idx + 1) == len(pfdb_df):
            print(f"[INFO] {mode_name}: processed PFDB rows {idx + 1}/{len(pfdb_df)}")

        pdb_id = pfrow["pdb_id"]
        pfdb_chain = clean_text(pfrow["pfdb_chain"])
        pfdb_pdb_start = pfrow["pfdb_pdb_start"]
        pfdb_pdb_end = pfrow["pfdb_pdb_end"]

        segs = seg_by_pdb.get(pdb_id, [])
        if segs:
            rows_with_any_sifts += 1

        row_has_same_ac_seed = False
        row_has_overlap = False

        for seg in segs:
            s_chain = clean_text(seg["chain_id"])
            if pfdb_chain and s_chain and pfdb_chain != s_chain:
                continue

            s_ac_base = seg["uniprot_ac_base"]
            candidate_seed_rows = seed_by_ac.get(s_ac_base, [])
            if not candidate_seed_rows:
                continue
            row_has_same_ac_seed = True

            projected = project_interval(
                pfdb_start=pfdb_pdb_start if pd.notna(pfdb_pdb_start) else None,
                pfdb_end=pfdb_pdb_end if pd.notna(pfdb_pdb_end) else None,
                seg_start=seg.get(seg_start_col),
                seg_end=seg.get(seg_end_col),
                seg_unp_start=int(seg["segment_unp_start"]),
                seg_unp_end=int(seg["segment_unp_end"]),
            )
            if projected is None:
                continue

            used_start, used_end, mapped_unp_start, mapped_unp_end = projected

            for srow in candidate_seed_rows:
                seed_start = srow.get("seed_start")
                seed_end = srow.get("seed_end")
                if seed_start is None or seed_end is None:
                    continue

                ov = interval_overlap(mapped_unp_start, mapped_unp_end, int(seed_start), int(seed_end))
                if ov is None:
                    overlap_len = 0
                    overlap_start = None
                    overlap_end = None
                else:
                    overlap_start, overlap_end = ov
                    overlap_len = overlap_end - overlap_start + 1
                    if overlap_len > 0:
                        row_has_overlap = True

                detail_rows.append({
                    "pfdb_row_id": pfrow["pfdb_row_id"],
                    "source_local_row_id": pfrow["source_local_row_id"],
                    "source_row_index_1based": pfrow["source_row_index_1based"],
                    "source_file": pfrow["source_file"],
                    "No.": pfrow["No."],
                    "protein_short_name": pfrow["protein_short_name"],
                    "pdb_raw": pfrow["pdb_raw"],
                    "pdb_id": pdb_id,
                    "pfdb_chain": pfdb_chain,
                    "pfdb_pdb_start": pfdb_pdb_start,
                    "pfdb_pdb_end": pfdb_pdb_end,
                    "sifts_uniprot_ac_raw": seg["uniprot_ac_raw"],
                    "sifts_uniprot_ac_base": s_ac_base,
                    "sifts_chain": s_chain,
                    "entity_id": seg["entity_id"],
                    "segment_auth_pdb_start": seg.get("segment_auth_pdb_start"),
                    "segment_auth_pdb_end": seg.get("segment_auth_pdb_end"),
                    "segment_auth_start_ins": seg.get("segment_auth_start_ins", ""),
                    "segment_auth_end_ins": seg.get("segment_auth_end_ins", ""),
                    "segment_res_pdb_start": seg.get("segment_res_pdb_start"),
                    "segment_res_pdb_end": seg.get("segment_res_pdb_end"),
                    "segment_unp_start": seg["segment_unp_start"],
                    "segment_unp_end": seg["segment_unp_end"],
                    seg_used_start_col: used_start,
                    seg_used_end_col: used_end,
                    "mapped_unp_start": mapped_unp_start,
                    "mapped_unp_end": mapped_unp_end,
                    "pfam_id": srow["pfam_id"],
                    "seed_seq_name": srow["seed_seq_name"],
                    "seed_seq_id": srow["seed_seq_id"],
                    "seed_ac_raw": srow["seed_ac_raw"],
                    "seed_ac_base": srow["seed_ac_base"],
                    "seed_start": seed_start,
                    "seed_end": seed_end,
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                    "overlap_len": overlap_len,
                    "numbering_mode": mode_name,
                })

        if row_has_same_ac_seed:
            rows_with_any_same_ac_seed += 1
        if row_has_overlap:
            rows_with_any_residue_overlap += 1

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        overlap_df = detail_df[detail_df["overlap_len"].fillna(0) > 0].copy()
    else:
        overlap_df = pd.DataFrame()

    best_hit_df = pd.DataFrame()
    if not overlap_df.empty:
        overlap_df = overlap_df.sort_values(
            ["pfdb_row_id", "overlap_len", "pfam_id", "sifts_uniprot_ac_base"],
            ascending=[True, False, True, True],
        )
        best_hit_df = overlap_df.groupby("pfdb_row_id", as_index=False).head(1).copy()

    row_summary_rows = []
    for pfrow in pfdb_df.to_dict("records"):
        rid = pfrow["pfdb_row_id"]
        sub_all = detail_df[detail_df["pfdb_row_id"] == rid] if not detail_df.empty else pd.DataFrame()
        sub_hit = overlap_df[overlap_df["pfdb_row_id"] == rid] if not overlap_df.empty else pd.DataFrame()
        row_summary_rows.append({
            "pfdb_row_id": rid,
            "source_local_row_id": pfrow["source_local_row_id"],
            "source_row_index_1based": pfrow["source_row_index_1based"],
            "source_file": pfrow["source_file"],
            "No.": pfrow["No."],
            "protein_short_name": pfrow["protein_short_name"],
            "pdb_raw": pfrow["pdb_raw"],
            "pdb_id": pfrow["pdb_id"],
            "pfdb_chain": pfrow["pfdb_chain"],
            "pfdb_pdb_start": pfrow["pfdb_pdb_start"],
            "pfdb_pdb_end": pfrow["pfdb_pdb_end"],
            "num_candidate_seed_matches": int(len(sub_all)),
            "num_residue_overlap_hits": int(len(sub_hit)),
            "matched_uniprot_ac_base": ";".join(sorted(sub_hit["sifts_uniprot_ac_base"].astype(str).unique().tolist())) if not sub_hit.empty else "",
            "matched_pfams": ";".join(sorted(sub_hit["pfam_id"].astype(str).unique().tolist())) if not sub_hit.empty else "",
            "best_overlap_len": int(sub_hit["overlap_len"].max()) if not sub_hit.empty else 0,
            "numbering_mode": mode_name,
        })
    row_summary_df = pd.DataFrame(row_summary_rows)

    pfam_summary_df = pd.DataFrame()
    if not overlap_df.empty:
        pfam_summary_df = (
            overlap_df.groupby("pfam_id", as_index=False)
            .agg(
                num_hits=("pfdb_row_id", "count"),
                num_unique_pfdb_rows=("pfdb_row_id", pd.Series.nunique),
                num_unique_uniprot_ac=("sifts_uniprot_ac_base", pd.Series.nunique),
                max_overlap_len=("overlap_len", "max"),
            )
            .sort_values(["num_unique_pfdb_rows", "num_hits", "pfam_id"], ascending=[False, False, True])
        )
        pfam_summary_df["numbering_mode"] = mode_name

    detail_path = out_dir / "sifts_residue_match_detail.csv"
    best_hit_path = out_dir / "sifts_residue_match_best_hit.csv"
    row_summary_path = out_dir / "pfdb_row_residue_summary.csv"
    pfam_summary_path = out_dir / "pfam_residue_summary.csv"
    summary_path = out_dir / "summary.txt"

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    best_hit_df.to_csv(best_hit_path, index=False, encoding="utf-8-sig")
    row_summary_df.to_csv(row_summary_path, index=False, encoding="utf-8-sig")
    pfam_summary_df.to_csv(pfam_summary_path, index=False, encoding="utf-8-sig")

    summary_lines = [
        f"mode: {mode_name}",
        f"PFDB 2Sm rows with parsed PDB: {len(pfdb_df)}",
        f"Unique PFDB PDB IDs: {pfdb_df['pdb_id'].nunique()}",
        f"Seed AC rows parsed: {len(seed_df)}",
        f"Unique seed base AC: {seed_df['seed_ac_base'].nunique()}",
        f"Unique Pfam families parsed: {seed_df['pfam_id'].nunique()}",
        f"SIFTS mapping segments parsed: {len(seg_df)}",
        f"Unique PDB with SIFTS segments: {seg_df['pdb_id'].nunique()}",
        f"PFDB 2Sm rows with any SIFTS data: {rows_with_any_sifts}",
        f"PFDB 2Sm rows with any same-AC seed candidate: {rows_with_any_same_ac_seed}",
        f"PFDB 2Sm rows with any residue-level overlap hit: {rows_with_any_residue_overlap}",
        f"Unique UniProt AC with residue-level overlap: {overlap_df['sifts_uniprot_ac_base'].nunique() if not overlap_df.empty else 0}",
        f"Unique Pfam families with residue-level overlap: {overlap_df['pfam_id'].nunique() if not overlap_df.empty else 0}",
        f"Detail file: {detail_path}",
        f"Best-hit file: {best_hit_path}",
        f"PFDB row summary file: {row_summary_path}",
        f"Pfam summary file: {pfam_summary_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    for line in summary_lines:
        print(f"[INFO] {line}")

    return detail_df, best_hit_df, row_summary_df, pfam_summary_df


def run_pipeline(args: argparse.Namespace) -> None:
    out_root = args.out_root
    cache_dir = args.cache_dir if args.cache_dir is not None else out_root / "sifts_api_cache"
    ensure_dir(out_root)
    ensure_dir(cache_dir)

    if not args.pfdb_2sm_csv.exists():
        raise FileNotFoundError(f"Missing file: {args.pfdb_2sm_csv}")
    if not args.pfam_seed_dir.exists():
        raise FileNotFoundError(f"Missing directory: {args.pfam_seed_dir}")

    print("[INFO] Reading PFDB 2Sm file...")
    pfdb_df = standardize_pfdb_df(read_2s_csv(args.pfdb_2sm_csv))
    print(f"[INFO] PFDB 2Sm rows with parsed PDB: {len(pfdb_df)}")
    print(f"[INFO] Unique PFDB PDB IDs: {pfdb_df['pdb_id'].nunique()}")

    print("[INFO] Scanning Pfam seed AC and intervals...")
    seed_df = scan_all_seed_ac(args.pfam_seed_dir)
    if seed_df.empty:
        raise RuntimeError("No seed AC rows parsed from PfamA_seed.")
    print(f"[INFO] Parsed seed AC rows: {len(seed_df)}")
    print(f"[INFO] Unique seed base AC: {seed_df['seed_ac_base'].nunique()}")
    print(f"[INFO] Unique Pfam families parsed: {seed_df['pfam_id'].nunique()}")

    unique_pdb = sorted(pfdb_df["pdb_id"].dropna().astype(str).str.upper().unique().tolist())
    session = requests.Session()
    all_seg_rows: List[dict] = []
    failed_pdb: List[Tuple[str, str]] = []

    print("[INFO] Fetching SIFTS mappings from PDBe API...")
    for i, pdb_id in enumerate(unique_pdb, start=1):
        if i % 20 == 0 or i == len(unique_pdb):
            print(f"[INFO] fetched or loaded SIFTS for {i}/{len(unique_pdb)} PDB entries")
        try:
            data = fetch_pdbe_uniprot_mapping(
                pdb_id=pdb_id,
                session=session,
                cache_dir=cache_dir,
                pdbe_api_base=args.pdbe_api_base,
                timeout=args.timeout,
                request_sleep=args.request_sleep,
            )
            all_seg_rows.extend(flatten_pdbe_mapping_json_dual(pdb_id, data))
        except Exception as e:
            failed_pdb.append((pdb_id, str(e)))

    seg_df = pd.DataFrame(all_seg_rows)
    if seg_df.empty:
        raise RuntimeError("No SIFTS mapping segments were parsed.")
    print(f"[INFO] Parsed SIFTS mapping segments: {len(seg_df)}")
    print(f"[INFO] Unique PDB with SIFTS segments: {seg_df['pdb_id'].nunique()}")

    failed_df = pd.DataFrame(failed_pdb, columns=["pdb_id", "error"])
    failed_df.to_csv(out_root / "failed_pdb_requests.csv", index=False, encoding="utf-8-sig")

    _, _, author_row, author_pfam = run_mode(
        mode_name="author",
        pfdb_df=pfdb_df,
        seed_df=seed_df,
        seg_df=seg_df,
        out_dir=out_root / "author_numbering",
    )
    _, _, residue_row, residue_pfam = run_mode(
        mode_name="residue",
        pfdb_df=pfdb_df,
        seed_df=seed_df,
        seg_df=seg_df,
        out_dir=out_root / "residue_numbering",
    )

    compare = pfdb_df[[
        "pfdb_row_id", "source_local_row_id", "source_row_index_1based", "source_file",
        "No.", "protein_short_name", "pdb_raw", "pdb_id", "pfdb_chain", "pfdb_pdb_start", "pfdb_pdb_end",
    ]].copy()

    author_row_sub = author_row[[
        "pfdb_row_id", "num_candidate_seed_matches", "num_residue_overlap_hits",
        "matched_uniprot_ac_base", "matched_pfams", "best_overlap_len",
    ]].copy().rename(columns={
        "num_candidate_seed_matches": "author_num_candidate_seed_matches",
        "num_residue_overlap_hits": "author_num_residue_overlap_hits",
        "matched_uniprot_ac_base": "author_matched_uniprot_ac_base",
        "matched_pfams": "author_matched_pfams",
        "best_overlap_len": "author_best_overlap_len",
    })

    residue_row_sub = residue_row[[
        "pfdb_row_id", "num_candidate_seed_matches", "num_residue_overlap_hits",
        "matched_uniprot_ac_base", "matched_pfams", "best_overlap_len",
    ]].copy().rename(columns={
        "num_candidate_seed_matches": "residue_num_candidate_seed_matches",
        "num_residue_overlap_hits": "residue_num_residue_overlap_hits",
        "matched_uniprot_ac_base": "residue_matched_uniprot_ac_base",
        "matched_pfams": "residue_matched_pfams",
        "best_overlap_len": "residue_best_overlap_len",
    })

    compare = compare.merge(author_row_sub, on="pfdb_row_id", how="left")
    compare = compare.merge(residue_row_sub, on="pfdb_row_id", how="left")
    compare.to_csv(out_root / "compare_author_vs_residue.csv", index=False, encoding="utf-8-sig")

    author_cols = ["pfam_id", "num_hits", "num_unique_pfdb_rows", "num_unique_uniprot_ac", "max_overlap_len"]
    if not author_pfam.empty:
        author_pfam_sub = author_pfam[author_cols].copy()
    else:
        author_pfam_sub = pd.DataFrame(columns=author_cols)
    author_pfam_sub = author_pfam_sub.rename(columns={
        "num_hits": "author_num_hits",
        "num_unique_pfdb_rows": "author_num_unique_pfdb_rows",
        "num_unique_uniprot_ac": "author_num_unique_uniprot_ac",
        "max_overlap_len": "author_max_overlap_len",
    })

    if not residue_pfam.empty:
        residue_pfam_sub = residue_pfam[author_cols].copy()
    else:
        residue_pfam_sub = pd.DataFrame(columns=author_cols)
    residue_pfam_sub = residue_pfam_sub.rename(columns={
        "num_hits": "residue_num_hits",
        "num_unique_pfdb_rows": "residue_num_unique_pfdb_rows",
        "num_unique_uniprot_ac": "residue_num_unique_uniprot_ac",
        "max_overlap_len": "residue_max_overlap_len",
    })
    pfam_compare = author_pfam_sub.merge(residue_pfam_sub, on="pfam_id", how="outer").sort_values("pfam_id")
    pfam_compare.to_csv(out_root / "compare_pfam_author_vs_residue.csv", index=False, encoding="utf-8-sig")

    print(f"[INFO] Compare file: {out_root / 'compare_author_vs_residue.csv'}")
    print(f"[INFO] Pfam compare file: {out_root / 'compare_pfam_author_vs_residue.csv'}")
    print("[INFO] Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match PFDB 2-state phi-value proteins to Pfam seed sequences through PDBe/SIFTS."
    )
    parser.add_argument("--pfdb-2sm-csv", type=Path, default=DEFAULT_FILE_2S, help="Input PFDB Final_2Sm.csv file.")
    parser.add_argument("--pfam-seed-dir", type=Path, default=DEFAULT_PFAM_SEED_DIR, help="PfamA_seed root directory.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output root directory.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional SIFTS API cache directory. Default: <out-root>/sifts_api_cache.")
    parser.add_argument("--pdbe-api-base", default=DEFAULT_PDBE_API_BASE, help="PDBe UniProt mapping API base URL.")
    parser.add_argument("--request-sleep", type=float, default=0.10, help="Sleep time after uncached PDBe API requests.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
