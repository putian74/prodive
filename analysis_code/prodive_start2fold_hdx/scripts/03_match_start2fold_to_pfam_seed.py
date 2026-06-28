#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# =========================
# 
# =========================
START2FOLD_CSV = Path("CHANGE_ME")
PFAM_SEED_DIR = Path("<PRODIVE_DATA_ROOT>/shared/PfamA_seed")
OUT_DIR = Path("CHANGE_ME")

# Start2Fold ：
# "uniprot"   = XML  residue index  UniProt 
# "construct" = XML  residue index ， UniProt_Range  UniProt 
RESIDUE_INDEX_MODE = "uniprot"

CLASS_KEYS = [
    "Fold_EARLY",
    "Fold_INTER",
    "Fold_LATE",
    "Stab_STRONG",
    "Stab_MEDIUM",
    "Stab_WEAK",
]

GS_AC_RE = re.compile(r"^#=GS\s+(\S+)\s+AC\s+(\S+)")
SEQ_RANGE_RE = re.compile(r"^(?P<seqid>.+?)/(\d+)-(\d+)$")
UNIPROT_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("\xa0", " ").strip()


def strip_version(ac: str) -> str:
    s = clean_text(ac)
    return s.split(".", 1)[0] if s else ""


def parse_uniprot_range(x: str) -> Optional[Tuple[int, int]]:
    s = clean_text(x)
    if not s:
        return None
    m = UNIPROT_RANGE_RE.match(s)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if a <= b else (b, a)


def parse_int_list(x) -> List[int]:
    s = clean_text(x)
    if not s:
        return []

    try:
        obj = ast.literal_eval(s)
    except Exception:
        return []

    if obj is None:
        return []

    if not isinstance(obj, (list, tuple, set)):
        return []

    vals = []
    for v in obj:
        try:
            vals.append(int(v))
        except Exception:
            continue

    return sorted(set(vals))


def to_uniprot_positions(
    residues: List[int],
    uniprot_range: str,
    index_mode: str = "uniprot",
) -> List[int]:
    if not residues:
        return []

    residues = sorted(set(int(x) for x in residues))

    if index_mode == "uniprot":
        return residues

    if index_mode == "construct":
        rg = parse_uniprot_range(uniprot_range)
        if rg is None:
            raise ValueError(
                f"RESIDUE_INDEX_MODE='construct'  UniProt_Range : {uniprot_range!r}"
            )
        start, end = rg
        shifted = [start + r - 1 for r in residues]
        return [x for x in shifted if start <= x <= end]

    raise ValueError(f"Unknown RESIDUE_INDEX_MODE: {index_mode}")


def read_start2fold_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)

    required_cols = ["STF_ID", "UniProt_ID", "UniProt_Range"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Start2Fold CSV: {missing}")

    for col in df.columns:
        df[col] = df[col].map(clean_text)

    df["uniprot_ac_raw"] = df["UniProt_ID"]
    df["uniprot_ac_base"] = df["uniprot_ac_raw"].map(strip_version)

    for key in CLASS_KEYS:
        list_col = f"{key}_List"
        if list_col not in df.columns:
            df[list_col] = ""

        parsed_col = f"{key}_parsed"
        uni_col = f"{key}_uniprot"

        df[parsed_col] = df[list_col].map(parse_int_list)
        df[uni_col] = df.apply(
            lambda r: to_uniprot_positions(
                r[parsed_col],
                r["UniProt_Range"],
                RESIDUE_INDEX_MODE,
            ),
            axis=1,
        )

        df[f"{key}_Count_Normalized"] = df[uni_col].map(len)

    df = df[df["uniprot_ac_base"] != ""].copy()
    return df.reset_index(drop=True)


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

            rows.append(
                {
                    "pfam_id": pfam_id,
                    "sto_file": str(sto_path),
                    "seed_seq_name": seq_name,
                    "seed_seq_id": seq_id,
                    "seed_ac_raw": ac,
                    "seed_ac_base": ac_base,
                    "seed_start": seed_start,
                    "seed_end": seed_end,
                }
            )

    return rows


def scan_all_seed_ac(seed_root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    sto_files = sorted(seed_root.glob("PF*/PF*.sto"))

    if not sto_files:
        raise FileNotFoundError(f"No sto files found under: {seed_root}")

    for i, sto in enumerate(sto_files, start=1):
        if i % 1000 == 0 or i == len(sto_files):
            print(f"[INFO] scanned sto files: {i}/{len(sto_files)}")
        rows.extend(parse_sto_ac_rows(sto))

    seed_df = pd.DataFrame(rows)
    if seed_df.empty:
        raise RuntimeError("No AC rows parsed from sto files.")

    return seed_df


def residues_in_seed_range(residues: List[int], seed_start, seed_end) -> List[int]:
    if seed_start is None or seed_end is None:
        return []
    lo, hi = int(seed_start), int(seed_end)
    return [r for r in residues if lo <= int(r) <= hi]


def match_start2fold_to_seed(
    stf_df: pd.DataFrame, seed_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_by_ac: Dict[str, List[dict]] = defaultdict(list)
    for row in seed_df.to_dict("records"):
        seed_by_ac[row["seed_ac_base"]].append(row)

    detail_rows: List[dict] = []
    exact_ac_match_count = 0
    overlap_protein_count = 0

    for idx, prow in stf_df.iterrows():
        if (idx + 1) % 200 == 0 or (idx + 1) == len(stf_df):
            print(f"[INFO] processed Start2Fold rows: {idx + 1}/{len(stf_df)}")

        ac = prow["uniprot_ac_base"]
        candidates = seed_by_ac.get(ac, [])

        if candidates:
            exact_ac_match_count += 1

        row_has_any_overlap = False

        for srow in candidates:
            seed_start = srow.get("seed_start")
            seed_end = srow.get("seed_end")
            has_seed_range = seed_start is not None and seed_end is not None

            total_overlap_set = set()
            out = {
                "stf_row_index_1based": idx + 1,
                "STF_ID": prow["STF_ID"],
                "uniprot_ac_raw": prow["uniprot_ac_raw"],
                "uniprot_ac_base": prow["uniprot_ac_base"],
                "UniProt_Range": prow["UniProt_Range"],
                "pfam_id": srow["pfam_id"],
                "sto_file": srow["sto_file"],
                "seed_seq_name": srow["seed_seq_name"],
                "seed_seq_id": srow["seed_seq_id"],
                "seed_ac_raw": srow["seed_ac_raw"],
                "seed_ac_base": srow["seed_ac_base"],
                "seed_start": seed_start,
                "seed_end": seed_end,
                "seed_has_range": has_seed_range,
            }

            for key in CLASS_KEYS:
                residues = prow[f"{key}_uniprot"]
                ov = residues_in_seed_range(residues, seed_start, seed_end) if has_seed_range else []
                out[f"{key}_input_count"] = len(residues)
                out[f"{key}_overlap_count"] = len(ov)
                out[f"{key}_overlap_list"] = str(sorted(ov))
                total_overlap_set.update(ov)

            total_overlap_list = sorted(total_overlap_set)
            out["total_overlap_count"] = len(total_overlap_list)
            out["total_overlap_list"] = str(total_overlap_list)
            out["has_any_overlap"] = int(len(total_overlap_list) > 0)

            if len(total_overlap_list) > 0:
                row_has_any_overlap = True

            detail_rows.append(out)

        if row_has_any_overlap:
            overlap_protein_count += 1

    detail_df = pd.DataFrame(detail_rows)

    ac_match_df = detail_df.copy() if not detail_df.empty else pd.DataFrame()
    overlap_df = (
        detail_df[detail_df["has_any_overlap"].fillna(0) > 0].copy()
        if not detail_df.empty
        else pd.DataFrame()
    )

    protein_summary_rows = []
    for idx, prow in stf_df.iterrows():
        stf_id = prow["STF_ID"]
        sub_all = ac_match_df[ac_match_df["STF_ID"] == stf_id] if not ac_match_df.empty else pd.DataFrame()
        sub_hit = overlap_df[overlap_df["STF_ID"] == stf_id] if not overlap_df.empty else pd.DataFrame()

        row = {
            "stf_row_index_1based": idx + 1,
            "STF_ID": stf_id,
            "uniprot_ac_raw": prow["uniprot_ac_raw"],
            "uniprot_ac_base": prow["uniprot_ac_base"],
            "UniProt_Range": prow["UniProt_Range"],
            "num_seed_ac_matches": int(len(sub_all)),
            "num_seed_overlap_hits": int(len(sub_hit)),
            "has_seed_ac_match": int(len(sub_all) > 0),
            "has_seed_overlap_hit": int(len(sub_hit) > 0),
            "matched_pfams": ";".join(sorted(sub_hit["pfam_id"].astype(str).unique().tolist())) if not sub_hit.empty else "",
            "matched_seed_seq_names": ";".join(sorted(sub_hit["seed_seq_name"].astype(str).unique().tolist())) if not sub_hit.empty else "",
            "best_total_overlap_count": int(sub_hit["total_overlap_count"].max()) if not sub_hit.empty else 0,
        }

        for key in CLASS_KEYS:
            row[f"{key}_input_count"] = len(prow[f"{key}_uniprot"])
            row[f"{key}_best_overlap_count"] = int(sub_hit[f"{key}_overlap_count"].max()) if not sub_hit.empty else 0
            row[f"{key}_num_seed_hits"] = int((sub_hit[f"{key}_overlap_count"] > 0).sum()) if not sub_hit.empty else 0

        protein_summary_rows.append(row)

    protein_summary_df = pd.DataFrame(protein_summary_rows)

    pfam_summary_df = pd.DataFrame()
    if not overlap_df.empty:
        agg_dict = {
            "num_overlap_hits": ("STF_ID", "count"),
            "num_unique_stf": ("STF_ID", pd.Series.nunique),
            "num_unique_uniprot": ("uniprot_ac_base", pd.Series.nunique),
            "max_total_overlap_count": ("total_overlap_count", "max"),
        }
        for key in CLASS_KEYS:
            agg_dict[f"{key}_max_overlap_count"] = (f"{key}_overlap_count", "max")
            agg_dict[f"{key}_positive_hits"] = (f"{key}_overlap_count", lambda s: int((s > 0).sum()))

        pfam_summary_df = (
            overlap_df.groupby("pfam_id", as_index=False)
            .agg(**agg_dict)
            .sort_values(
                ["num_unique_stf", "num_overlap_hits", "pfam_id"],
                ascending=[False, False, True],
            )
        )

    print(f"[INFO] Start2Fold proteins total: {len(stf_df)}")
    print(f"[INFO] Proteins with AC match in seed: {exact_ac_match_count}")
    print(f"[INFO] Proteins with residue overlap in seed: {overlap_protein_count}")

    return detail_df, protein_summary_df, pfam_summary_df


def build_covered_protein_table(protein_summary_df: pd.DataFrame) -> pd.DataFrame:
    if protein_summary_df.empty:
        return pd.DataFrame()

    covered_df = protein_summary_df[
        (protein_summary_df["num_seed_ac_matches"] > 0)
        | (protein_summary_df["num_seed_overlap_hits"] > 0)
    ].copy()

    if covered_df.empty:
        return covered_df

    covered_df["coverage_basis"] = covered_df.apply(
        lambda r: "residue_overlap" if int(r["num_seed_overlap_hits"]) > 0 else "ac_match_only",
        axis=1,
    )

    ordered_cols = [
        "stf_row_index_1based",
        "STF_ID",
        "uniprot_ac_raw",
        "uniprot_ac_base",
        "UniProt_Range",
        "num_seed_ac_matches",
        "num_seed_overlap_hits",
        "best_total_overlap_count",
        "coverage_basis",
        "matched_pfams",
        "matched_seed_seq_names",
    ]
    for key in CLASS_KEYS:
        ordered_cols.extend(
            [
                f"{key}_input_count",
                f"{key}_best_overlap_count",
                f"{key}_num_seed_hits",
            ]
        )

    keep_cols = [c for c in ordered_cols if c in covered_df.columns]
    covered_df = covered_df[keep_cols].sort_values(
        ["num_seed_overlap_hits", "num_seed_ac_matches", "best_total_overlap_count", "STF_ID"],
        ascending=[False, False, False, True],
    )
    return covered_df.reset_index(drop=True)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Match Start2Fold residue annotations to Pfam seed intervals.")
    p.add_argument("--start2fold-csv", type=Path, default=START2FOLD_CSV, help="Parsed Start2Fold all-classes CSV.")
    p.add_argument("--pfam-seed-dir", type=Path, default=PFAM_SEED_DIR, help="PfamA_seed root directory.")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory.")
    p.add_argument("--residue-index-mode", choices=["uniprot", "construct"], default=RESIDUE_INDEX_MODE, help="Interpretation of Start2Fold residue indices. For this analysis, use uniprot.")
    return p.parse_args()

def main() -> None:
    global START2FOLD_CSV, PFAM_SEED_DIR, OUT_DIR, RESIDUE_INDEX_MODE
    args = parse_args()
    START2FOLD_CSV = args.start2fold_csv
    PFAM_SEED_DIR = args.pfam_seed_dir
    OUT_DIR = args.out_dir
    RESIDUE_INDEX_MODE = args.residue_index_mode
    ensure_dir(OUT_DIR)

    if not START2FOLD_CSV.exists():
        raise FileNotFoundError(f"Missing Start2Fold CSV: {START2FOLD_CSV}")
    if not PFAM_SEED_DIR.exists():
        raise FileNotFoundError(f"Missing Pfam seed directory: {PFAM_SEED_DIR}")

    print("[INFO] Reading Start2Fold CSV...")
    stf_df = read_start2fold_csv(START2FOLD_CSV)
    print(f"[INFO] Parsed Start2Fold rows: {len(stf_df)}")
    print(f"[INFO] Unique UniProt AC: {stf_df['uniprot_ac_base'].nunique()}")

    print("[INFO] Scanning sto files...")
    seed_df = scan_all_seed_ac(PFAM_SEED_DIR)
    print(f"[INFO] Parsed seed AC rows: {len(seed_df)}")
    print(f"[INFO] Unique seed AC: {seed_df['seed_ac_base'].nunique()}")
    print(f"[INFO] Unique Pfam families: {seed_df['pfam_id'].nunique()}")

    detail_df, protein_summary_df, pfam_summary_df = match_start2fold_to_seed(stf_df, seed_df)
    covered_protein_df = build_covered_protein_table(protein_summary_df)

    detail_path = OUT_DIR / "start2fold_seed_match_detail.csv"
    protein_summary_path = OUT_DIR / "start2fold_protein_summary.csv"
    pfam_summary_path = OUT_DIR / "start2fold_pfam_summary.csv"
    covered_protein_path = OUT_DIR / "start2fold_proteins_covered_in_seed.csv"

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    protein_summary_df.to_csv(protein_summary_path, index=False, encoding="utf-8-sig")
    pfam_summary_df.to_csv(pfam_summary_path, index=False, encoding="utf-8-sig")
    covered_protein_df.to_csv(covered_protein_path, index=False, encoding="utf-8-sig")

    summary_lines = [
        f"Start2Fold rows parsed: {len(stf_df)}",
        f"Unique Start2Fold UniProt AC: {stf_df['uniprot_ac_base'].nunique()}",
        f"Seed AC rows parsed: {len(seed_df)}",
        f"Unique seed AC: {seed_df['seed_ac_base'].nunique()}",
        f"Unique Pfam families parsed: {seed_df['pfam_id'].nunique()}",
        f"Proteins with any seed AC match: {(protein_summary_df['num_seed_ac_matches'] > 0).sum()}",
        f"Proteins with any residue overlap hit: {(protein_summary_df['num_seed_overlap_hits'] > 0).sum()}",
        f"Covered protein rows (AC match or residue overlap): {len(covered_protein_df)}",
        f"Detail file: {detail_path}",
        f"Protein summary file: {protein_summary_path}",
        f"Pfam summary file: {pfam_summary_path}",
        f"Covered proteins file: {covered_protein_path}",
    ]

    (OUT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    for line in summary_lines:
        print("[INFO]", line)


if __name__ == "__main__":
    main()