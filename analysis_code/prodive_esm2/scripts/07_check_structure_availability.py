#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import gzip
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# =========================================================
# Config
# =========================================================

FRAGDOM_FILE = Path(
    "CHANGE_ME"
)

PFAM_ROOT = Path(
    "<PRODIVE_DATA_ROOT>/shared/PfamA_seed"
)

OUT_DIR = Path(
    "CHANGE_ME"
)

OUT_TSV = OUT_DIR / "full_esm_representative_structure_availability.tsv"
OUT_SUMMARY_TSV = OUT_DIR / "full_esm_representative_structure_availability_summary.tsv"



def str_to_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def configure_runtime(args: argparse.Namespace) -> None:
    global FRAGDOM_FILE, PFAM_ROOT, OUT_DIR, OUT_TSV, OUT_SUMMARY_TSV
    FRAGDOM_FILE = Path(args.fragdom_file)
    PFAM_ROOT = Path(args.pfam_root)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TSV = OUT_DIR / "full_esm_representative_structure_availability.tsv"
    OUT_SUMMARY_TSV = OUT_DIR / "full_esm_representative_structure_availability_summary.tsv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check structure availability for ESM2 representative sequences."
    )
    parser.add_argument("--fragdom-file", default=str(FRAGDOM_FILE), help="Representative frag_dom/full-sequence file.")
    parser.add_argument("--pfam-root", default=str(PFAM_ROOT), help="PfamA_seed root directory.")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory.")
    return parser


# =========================================================
# Header parsing
# =========================================================

def open_text_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def iter_headers_from_fragdom(path: Path) -> Iterable[str]:
    """
    Yield header lines without leading '>' from frag_dom formatted file.
    """
    with open_text_maybe_gz(path) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith(">"):
                yield line[1:]


def parse_header(header: str) -> Dict[str, Optional[str]]:
    """
    Parse header like:
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
    }

    for part in parts[1:]:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        info[key] = val

    return info


# =========================================================
# Structure searching
# =========================================================

def extract_af_version(path: Path) -> int:
    """
    Extract AlphaFold model version from filename:
      AF-P31388-F1-model_v4.pdb -> 4
      AF-P31388-F1-model_v6.pdb -> 6

    If no version is found, return -1.
    """
    m = re.search(r"model_v(\d+)", path.name)
    if not m:
        return -1
    return int(m.group(1))


def find_af_structures_for_uid(pfam_dir: Path, uid: str) -> List[Path]:
    """
    Find AlphaFold structures for one UID inside one Pfam directory.
    Main expected pattern:
      AF-{uid}-F1-model_v*.pdb

    Also includes a few relaxed patterns in case filenames vary.
    """
    patterns = [
        f"AF-{uid}-F1-model_v*.pdb",
        f"AF-{uid}-F1-model*.pdb",
        f"AF-{uid}-*.pdb",
        f"AF-{uid}-F1-model_v*.pdb.gz",
        f"AF-{uid}-F1-model*.pdb.gz",
        f"AF-{uid}-*.pdb.gz",
    ]

    found: List[Path] = []
    seen = set()

    for pattern in patterns:
        for p in pfam_dir.glob(pattern):
            if p not in seen:
                found.append(p)
                seen.add(p)

    found = sorted(
        found,
        key=lambda x: (extract_af_version(x), x.name),
        reverse=True,
    )

    return found


def find_exp_structures_for_uid(pfam_dir: Path, uid: str) -> List[Path]:
    """
    Optional relaxed search for experimental structures.

    This is intentionally broad because your historical file naming may vary,
    e.g. *_exp_*.pdb, *P31388*exp*.pdb, etc.
    """
    patterns = [
        f"*{uid}*exp*.pdb",
        f"*{uid}*EXP*.pdb",
        f"*{uid}*_exp_*.pdb",
        f"*{uid}*_EXP_*.pdb",
        f"*{uid}*.pdb",
        f"*{uid}*.cif",
        f"*{uid}*.pdb.gz",
        f"*{uid}*.cif.gz",
    ]

    found: List[Path] = []
    seen = set()

    for pattern in patterns:
        for p in pfam_dir.glob(pattern):
            name = p.name

            # Avoid counting AlphaFold files as EXP candidates here.
            if name.startswith("AF-"):
                continue

            if p not in seen:
                found.append(p)
                seen.add(p)

    found = sorted(found, key=lambda x: x.name)
    return found


def choose_best_structure(exp_paths: List[Path], af_paths: List[Path]) -> Tuple[str, Optional[Path]]:
    """
    Choose best structure.

    Preference:
      1. Experimental candidate, if found
      2. AlphaFold candidate with highest version
      3. None
    """
    if exp_paths:
        return "EXP", exp_paths[0]

    if af_paths:
        return "AF", af_paths[0]

    return "NO_STRUCTURE", None


# =========================================================
# Main
# =========================================================

def main() -> None:
    args = build_arg_parser().parse_args()
    configure_runtime(args)

    if not FRAGDOM_FILE.exists():
        raise FileNotFoundError(f"FRAGDOM_FILE not found: {FRAGDOM_FILE}")

    if not PFAM_ROOT.exists():
        raise FileNotFoundError(f"PFAM_ROOT not found: {PFAM_ROOT}")

    rows = []
    n_headers = 0

    for header in tqdm(iter_headers_from_fragdom(FRAGDOM_FILE), desc="Checking structures", unit="record"):
        n_headers += 1

        info = parse_header(header)
        pfam_id = info.get("pfam_id")
        uid = info.get("uid")

        if not pfam_id or not re.match(r"^PF\d{5}$", pfam_id):
            rows.append(
                {
                    **info,
                    "pfam_dir": "",
                    "pfam_dir_exists": False,
                    "status": "BAD_PFAM_ID",
                    "best_structure_type": "",
                    "best_structure_path": "",
                    "n_af_structures": 0,
                    "n_exp_structures": 0,
                    "all_af_structures": "",
                    "all_exp_structures": "",
                }
            )
            continue

        if not uid:
            rows.append(
                {
                    **info,
                    "pfam_dir": str(PFAM_ROOT / pfam_id),
                    "pfam_dir_exists": (PFAM_ROOT / pfam_id).exists(),
                    "status": "UID_NOT_FOUND_IN_HEADER",
                    "best_structure_type": "",
                    "best_structure_path": "",
                    "n_af_structures": 0,
                    "n_exp_structures": 0,
                    "all_af_structures": "",
                    "all_exp_structures": "",
                }
            )
            continue

        pfam_dir = PFAM_ROOT / pfam_id
        pfam_dir_exists = pfam_dir.exists()

        if not pfam_dir_exists:
            rows.append(
                {
                    **info,
                    "pfam_dir": str(pfam_dir),
                    "pfam_dir_exists": False,
                    "status": "PFAM_DIR_NOT_FOUND",
                    "best_structure_type": "",
                    "best_structure_path": "",
                    "n_af_structures": 0,
                    "n_exp_structures": 0,
                    "all_af_structures": "",
                    "all_exp_structures": "",
                }
            )
            continue

        af_paths = find_af_structures_for_uid(pfam_dir, uid)
        exp_paths = find_exp_structures_for_uid(pfam_dir, uid)

        best_type, best_path = choose_best_structure(exp_paths, af_paths)

        if best_type == "EXP":
            status = "EXP_FOUND"
        elif best_type == "AF":
            status = "AF_FOUND"
        else:
            status = "NO_STRUCTURE_FOR_UID"

        rows.append(
            {
                **info,
                "pfam_dir": str(pfam_dir),
                "pfam_dir_exists": True,
                "status": status,
                "best_structure_type": best_type if best_type != "NO_STRUCTURE" else "",
                "best_structure_path": str(best_path) if best_path is not None else "",
                "n_af_structures": len(af_paths),
                "n_exp_structures": len(exp_paths),
                "all_af_structures": ";".join(str(p) for p in af_paths),
                "all_exp_structures": ";".join(str(p) for p in exp_paths),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TSV, sep="\t", index=False)

    summary_rows = []

    if not df.empty:
        status_counts = df["status"].value_counts(dropna=False).reset_index()
        status_counts.columns = ["status", "count"]
        status_counts["fraction"] = status_counts["count"] / len(df)
        summary_rows.append(status_counts)

        summary_df = status_counts
    else:
        summary_df = pd.DataFrame(columns=["status", "count", "fraction"])

    summary_df.to_csv(OUT_SUMMARY_TSV, sep="\t", index=False)

    print("[DONE]")
    print(f"Total headers checked: {n_headers}")
    print(f"Wrote detail table:")
    print(f"  {OUT_TSV}")
    print(f"Wrote summary table:")
    print(f"  {OUT_SUMMARY_TSV}")

    print("\n[SUMMARY]")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No records found.")


if __name__ == "__main__":
    main()