#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Goal
----
Use the already-selected structure result in STRUCT_CSV directly.

For each side (Main / Sub) independently:
- If side Status == OK, keep this side as one observation.
- Trust CSV fields directly:
    PDB_ID
    Chain_Used
    Selected_Seq_Start / End / Len
    Avg_RSA / SS_Class / Location
- Do NOT redo HHM->MSA remapping
- Do NOT re-select representative sequence
- Do NOT re-select structure

Important note
--------------
This version only changes where structure files are searched:
- structure files are resolved under:
    <PRODIVE_DATA_ROOT>/shared/PfamA_seed/{fam}/{pdb_id}

So:
- trust CSV-selected coordinates
- but load actual structure files from PfamA_seed family directories

Output
------
One SQLite database with:
1) hit_obs
2) null_win
"""

import os
import argparse
import re
import glob
import shutil
import sqlite3
import tempfile
import warnings
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import PDB
from Bio.PDB.DSSP import DSSP
from Bio import BiopythonWarning
from concurrent.futures import ProcessPoolExecutor, as_completed


# ==============================
# Config
# ==============================

STRUCT_CSV = "CHANGE_ME"
OUTPUT_DB = "CHANGE_ME"

# Structures are resolved only under PfamA_seed.
PFAM_DIR = "<PRODIVE_DATA_ROOT>/shared/PfamA_seed"

K_PER_HIT = 50
MAX_TRIES_PER_WIN = 50

DISALLOW_OVERLAP_WITH_HIT = True
DISALLOW_DUPLICATE_WINDOWS = True

DSSP_EXE = shutil.which("mkdssp") or shutil.which("dssp")

HIT_WORKERS = 48
INFLIGHT = HIT_WORKERS * 4

MIN_FOUND_RATIO = 0.8
RANDOM_SEED = 20260129


# ==============================
# SS helpers
# ==============================

HELIX_SET = {"H", "G", "I"}
SHEET_SET = {"E", "B"}


def ss_to_3cat(code: str) -> str:
    if code in HELIX_SET:
        return "helix"
    if code in SHEET_SET:
        return "sheet"
    return "coil"


def ss_class_from_fracs(h_frac: float, e_frac: float, c_frac: float) -> str:
    if h_frac > 0.5:
        return "Helix_Dominant"
    if e_frac > 0.5:
        return "Sheet_Dominant"
    if (h_frac + e_frac) < 0.4:
        return "Coil/Loop_Dominant"
    return "Mixed"


def make_stable_int_seed(*parts) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha1(s).hexdigest(), 16)


# ==============================
# Generic helpers
# ==============================

def is_missing(x) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "na"}


def parse_int_safe(x) -> Optional[int]:
    if is_missing(x):
        return None
    try:
        return int(float(x))
    except Exception:
        m = re.search(r"(-?\d+)", str(x))
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def parse_float_safe(x) -> Optional[float]:
    if is_missing(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def infer_chain_from_path(pdb_path: str) -> str:
    if is_missing(pdb_path):
        return "A"
    base = os.path.basename(str(pdb_path))
    if base.startswith("AF-"):
        return "A"
    m = re.search(r"_([A-Za-z0-9]+)\.pdb$", base)
    if m:
        return m.group(1)
    return "A"


# ==============================
# Resolve PDB path from CSV
# ==============================

def resolve_pdb_path(fam: str, pdb_id: str) -> Optional[str]:
    """
    Resolve actual structure path strictly under:
        <PRODIVE_DATA_ROOT>/shared/PfamA_seed/{fam}/{pdb_id}
    """
    if is_missing(fam) or is_missing(pdb_id):
        return None

    fam = str(fam).strip()
    pdb_id = str(pdb_id).strip()

    # If already absolute and exists, accept directly
    if os.path.isabs(pdb_id) and os.path.exists(pdb_id):
        return pdb_id

    fam_dir = os.path.join(PFAM_DIR, fam)

    # Exact expected path
    direct = os.path.join(fam_dir, pdb_id)
    if os.path.exists(direct):
        return direct

    # Exact basename match
    hits = glob.glob(os.path.join(fam_dir, os.path.basename(pdb_id)))
    if hits:
        return hits[0]

    # Loose fallback
    hits = glob.glob(os.path.join(fam_dir, f"*{os.path.basename(pdb_id)}*"))
    if hits:
        return hits[0]

    return None


# ==============================
# DSSP safe build
# ==============================

def _call_dssp(model, pdb_path: str):
    if DSSP_EXE is None:
        raise RuntimeError("mkdssp/dssp not found in PATH.")
    try:
        return DSSP(model, pdb_path, dssp=DSSP_EXE)
    except TypeError:
        return DSSP(model, pdb_path)


def build_dssp_safely(pdb_path: str):
    """
    Old robust DSSP logic:
      1) try original PDB
      2) if fail, rewrite via PDBIO
      3) add CRYST1 if missing
      4) try cleaned PDB
    """
    parser = PDB.PDBParser(PERMISSIVE=1, QUIET=True)

    try:
        structure = parser.get_structure("X", pdb_path)
    except Exception as e:
        raise RuntimeError(f"Biopython_Parse_Failed: {e}")

    model = structure[0]

    # Try original
    try:
        dssp = _call_dssp(model, pdb_path)
        if len(dssp) > 0:
            chain_ids = {k[0] for k in dssp.keys()}
            return dssp, chain_ids, "Original"
    except Exception:
        pass

    # Fallback: clean with PDBIO
    fd, tmp_pdb = tempfile.mkstemp(prefix="dssp_clean_", suffix=".pdb")
    os.close(fd)

    try:
        io = PDB.PDBIO()
        io.set_structure(structure)
        io.save(tmp_pdb)

        with open(tmp_pdb, "r") as f:
            lines = f.readlines()

        has_cryst1 = any(line.startswith("CRYST1") for line in lines)
        if not has_cryst1:
            dummy = "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n"
            with open(tmp_pdb, "w") as f:
                f.write(dummy)
                f.writelines(lines)

        dssp = _call_dssp(model, tmp_pdb)
        if len(dssp) == 0:
            raise RuntimeError("Empty_DSSP_Output")

        chain_ids = {k[0] for k in dssp.keys()}
        return dssp, chain_ids, "Cleaned_PDBIO"

    except Exception as e:
        raise RuntimeError(f"DSSP failed to produce an output: {e}")
    finally:
        try:
            os.remove(tmp_pdb)
        except Exception:
            pass


# ==============================
# Chain cache
# ==============================

@dataclass
class ChainCache:
    res_ids: np.ndarray
    rsa: np.ndarray
    ss: np.ndarray


def build_chain_cache(dssp_obj) -> Dict[str, ChainCache]:
    buf: Dict[str, List[Tuple[int, str, float]]] = {}
    for key in dssp_obj.keys():
        chain = key[0]
        resseq = key[1][1]
        vals = dssp_obj[key]

        ss_code = vals[2] if len(vals) > 2 else " "
        rsa_raw = vals[3] if len(vals) > 3 else np.nan

        try:
            rsa_val = float(rsa_raw)
        except Exception:
            rsa_val = np.nan

        if np.isfinite(rsa_val):
            if rsa_val < 0.0:
                rsa_val = 0.0
            if rsa_val > 1.0:
                rsa_val = 1.0

        ss_code = ss_code if ss_code and ss_code != " " else "-"
        buf.setdefault(chain, []).append((resseq, ss_code, rsa_val))

    out = {}
    for chain, items in buf.items():
        items.sort(key=lambda x: x[0])
        out[chain] = ChainCache(
            res_ids=np.array([x[0] for x in items], dtype=np.int32),
            ss=np.array([x[1] for x in items], dtype="<U1"),
            rsa=np.array([x[2] for x in items], dtype=np.float32),
        )
    return out


def get_chain_used(chain_ids: set, prefer: str) -> Optional[str]:
    if prefer in chain_ids:
        return prefer
    if len(chain_ids) == 1:
        return list(chain_ids)[0]
    if "A" in chain_ids:
        return "A"
    return list(chain_ids)[0] if chain_ids else None


# ==============================
# Window features
# ==============================

def window_features_from_cache(cache: ChainCache, start_res: int, end_res: int) -> Optional[Dict[str, float]]:
    L = end_res - start_res + 1
    if L <= 0:
        return None

    mask = (cache.res_ids >= start_res) & (cache.res_ids <= end_res)
    if not np.any(mask):
        return None

    found = int(np.sum(mask))
    if (found / L) < MIN_FOUND_RATIO:
        return None

    ss_codes = cache.ss[mask]
    rsa_vals = cache.rsa[mask]

    finite = np.isfinite(rsa_vals)
    avg_rsa = float(np.mean(rsa_vals[finite])) if np.any(finite) else float("nan")

    h = e = c = 0
    for code in ss_codes:
        cat = ss_to_3cat(code)
        if cat == "helix":
            h += 1
        elif cat == "sheet":
            e += 1
        else:
            c += 1

    denom = max(1, len(ss_codes))
    h_frac = h / denom
    e_frac = e / denom
    c_frac = c / denom

    return {
        "avg_rsa": avg_rsa,
        "helix_frac": float(h_frac),
        "sheet_frac": float(e_frac),
        "coil_frac": float(c_frac),
        "ss_class": ss_class_from_fracs(h_frac, e_frac, c_frac),
    }


# ==============================
# Random windows
# ==============================

def random_windows_for_hit(
    cache: ChainCache,
    L: int,
    hit_start: int,
    hit_end: int,
    k: int,
    disallow_overlap: bool,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    if L <= 0:
        return []

    min_res = int(cache.res_ids.min())
    max_res = int(cache.res_ids.max())
    if (max_res - min_res + 1) < L:
        return []

    wins = []
    seen = set()
    tries = 0
    max_total_tries = max(k * MAX_TRIES_PER_WIN, 100)

    while len(wins) < k and tries < max_total_tries:
        tries += 1
        s = rng.randint(min_res, max_res - L + 1)
        e = s + L - 1

        if disallow_overlap:
            if not (e < hit_start or s > hit_end):
                continue

        if DISALLOW_DUPLICATE_WINDOWS and (s, e) in seen:
            continue

        feat = window_features_from_cache(cache, s, e)
        if feat is None:
            continue

        wins.append((s, e))
        seen.add((s, e))

    return wins


# ==============================
# SQLite schema
# ==============================

HIT_OBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS hit_obs (
  obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
  row_idx INTEGER,
  side TEXT,
  file_name TEXT,
  score REAL,
  main_hmm TEXT,
  sub_hmm TEXT,
  fam TEXT,
  pdb_id TEXT,
  pdb_path TEXT,
  chain_used TEXT,
  hit_start INTEGER,
  hit_end INTEGER,
  L INTEGER,

  hit_avg_rsa REAL,
  hit_ss_class TEXT,
  hit_location TEXT,

  hit_helix_frac REAL,
  hit_sheet_frac REAL,
  hit_coil_frac REAL,
  hit_avg_rsa_calc REAL,
  hit_ss_class_calc TEXT,

  dssp_mode TEXT,
  status TEXT,
  error_detail TEXT
);
"""

NULL_WIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS null_win (
  win_id INTEGER PRIMARY KEY AUTOINCREMENT,
  obs_id INTEGER,
  k_idx INTEGER,
  win_start INTEGER,
  win_end INTEGER,
  win_avg_rsa REAL,
  win_helix_frac REAL,
  win_sheet_frac REAL,
  win_coil_frac REAL,
  win_ss_class TEXT,
  status TEXT,
  FOREIGN KEY(obs_id) REFERENCES hit_obs(obs_id)
);
"""


def sqlite_init(db_path: str) -> sqlite3.Connection:
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute(HIT_OBS_SCHEMA)
    cur.execute(NULL_WIN_SCHEMA)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_null_obs_id ON null_win(obs_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hit_status ON hit_obs(status);")
    conn.commit()
    return conn


# ==============================
# Build observations directly from CSV
# ==============================

def side_to_obs(row: pd.Series, row_idx: int, side: str) -> Optional[Dict[str, Any]]:
    """
    Trust CSV-selected result directly.
    Only locate structure path under PfamA_seed.
    """
    status = str(row.get(f"{side}_Status", "")).strip()
    if status != "OK":
        return None

    fam = str(row.get("Main_HMM" if side == "Main" else "Sub_HMM", "")).strip()
    pdb_id = row.get(f"{side}_PDB_ID", None)
    chain_used = row.get(f"{side}_Chain_Used", None)
    hit_start = parse_int_safe(row.get(f"{side}_Selected_Seq_Start", None))
    hit_end = parse_int_safe(row.get(f"{side}_Selected_Seq_End", None))
    seq_len = parse_int_safe(row.get(f"{side}_Selected_Seq_Len", None))

    if is_missing(fam) or is_missing(pdb_id) or hit_start is None or hit_end is None:
        return None

    if is_missing(chain_used):
        chain_used = infer_chain_from_path(str(pdb_id))
    else:
        chain_used = str(chain_used).strip()

    L_from_range = hit_end - hit_start + 1
    if L_from_range <= 0:
        return None

    L = L_from_range
    if seq_len is not None and seq_len > 0 and seq_len != L_from_range:
        L = L_from_range

    pdb_path = resolve_pdb_path(fam, str(pdb_id))
    if pdb_path is None:
        return None

    return {
        "row_idx": row_idx,
        "side": side,
        "file_name": row.get("File", None),
        "score": parse_float_safe(row.get("Score", None)),
        "main_hmm": row.get("Main_HMM", None),
        "sub_hmm": row.get("Sub_HMM", None),
        "fam": fam,
        "pdb_id": str(pdb_id).strip(),
        "pdb_path": pdb_path,
        "chain_used": chain_used,
        "hit_start": int(hit_start),
        "hit_end": int(hit_end),
        "L": int(L),
        "hit_avg_rsa_csv": parse_float_safe(row.get(f"{side}_Avg_RSA", None)),
        "hit_ss_class_csv": None if is_missing(row.get(f"{side}_SS_Class", None)) else str(row.get(f"{side}_SS_Class")).strip(),
        "hit_location_csv": None if is_missing(row.get(f"{side}_Location", None)) else str(row.get(f"{side}_Location")).strip(),
    }


def build_observations(df: pd.DataFrame) -> pd.DataFrame:
    obs_rows = []
    for i, row in df.iterrows():
        main_obs = side_to_obs(row, i, "Main")
        if main_obs is not None:
            obs_rows.append(main_obs)

        sub_obs = side_to_obs(row, i, "Sub")
        if sub_obs is not None:
            obs_rows.append(sub_obs)

    return pd.DataFrame(obs_rows)


# ==============================
# Worker
# ==============================

_WORKER_DSSP_CACHE: Dict[str, Tuple[Dict[str, ChainCache], set, str]] = {}


def _load_dssp_chain_cache(pdb_path: str) -> Tuple[Dict[str, ChainCache], set, str]:
    if pdb_path in _WORKER_DSSP_CACHE:
        return _WORKER_DSSP_CACHE[pdb_path]

    warnings.simplefilter("ignore", BiopythonWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")

    dssp_obj, chain_ids, mode = build_dssp_safely(pdb_path)
    chain_cache = build_chain_cache(dssp_obj)
    _WORKER_DSSP_CACHE[pdb_path] = (chain_cache, chain_ids, mode)
    return chain_cache, chain_ids, mode


def build_fail_hit_row(obs: Dict[str, Any], dssp_mode="NA", status="FAIL", error_detail=None) -> Dict[str, Any]:
    return {
        "row_idx": obs["row_idx"],
        "side": obs["side"],
        "file_name": obs["file_name"],
        "score": obs["score"],
        "main_hmm": obs["main_hmm"],
        "sub_hmm": obs["sub_hmm"],
        "fam": obs["fam"],
        "pdb_id": obs["pdb_id"],
        "pdb_path": obs["pdb_path"],
        "chain_used": obs["chain_used"],
        "hit_start": obs["hit_start"],
        "hit_end": obs["hit_end"],
        "L": obs["L"],
        "hit_avg_rsa": obs["hit_avg_rsa_csv"],
        "hit_ss_class": obs["hit_ss_class_csv"],
        "hit_location": obs["hit_location_csv"],
        "hit_helix_frac": None,
        "hit_sheet_frac": None,
        "hit_coil_frac": None,
        "hit_avg_rsa_calc": None,
        "hit_ss_class_calc": None,
        "dssp_mode": dssp_mode,
        "status": status,
        "error_detail": error_detail,
    }


def process_one_obs(obs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    seed_int = make_stable_int_seed(
        RANDOM_SEED,
        obs["row_idx"],
        obs["side"],
        obs["fam"],
        obs["pdb_id"],
        obs["hit_start"],
        obs["hit_end"],
        obs["L"]
    )
    rng = random.Random(seed_int)

    if not os.path.exists(obs["pdb_path"]):
        return build_fail_hit_row(
            obs,
            dssp_mode="NA",
            status="PDB_Not_Found",
            error_detail=obs["pdb_path"],
        ), []

    try:
        chain_cache_dict, chain_ids, mode = _load_dssp_chain_cache(obs["pdb_path"])
    except Exception as e:
        return build_fail_hit_row(
            obs,
            dssp_mode="FAIL",
            status="DSSP_Fail",
            error_detail=str(e),
        ), []

    chain_used = get_chain_used(chain_ids, obs["chain_used"])
    if chain_used is None or chain_used not in chain_cache_dict:
        return build_fail_hit_row(
            obs,
            dssp_mode=mode,
            status="Chain_Not_Found",
            error_detail=f"preferred={obs['chain_used']}; available={sorted(chain_ids)}",
        ), []

    cache = chain_cache_dict[chain_used]

    hit_feat = window_features_from_cache(cache, obs["hit_start"], obs["hit_end"])
    if hit_feat is None:
        return build_fail_hit_row(
            obs,
            dssp_mode=mode,
            status="Hit_Low_Coverage",
            error_detail=f"range={obs['hit_start']}-{obs['hit_end']}",
        ), []

    hit_avg_rsa_final = obs["hit_avg_rsa_csv"]
    if hit_avg_rsa_final is None and np.isfinite(hit_feat["avg_rsa"]):
        hit_avg_rsa_final = float(hit_feat["avg_rsa"])

    hit_ss_class_final = obs["hit_ss_class_csv"]
    if hit_ss_class_final is None:
        hit_ss_class_final = hit_feat["ss_class"]

    hit_row = {
        "row_idx": obs["row_idx"],
        "side": obs["side"],
        "file_name": obs["file_name"],
        "score": obs["score"],
        "main_hmm": obs["main_hmm"],
        "sub_hmm": obs["sub_hmm"],
        "fam": obs["fam"],
        "pdb_id": obs["pdb_id"],
        "pdb_path": obs["pdb_path"],
        "chain_used": chain_used,
        "hit_start": obs["hit_start"],
        "hit_end": obs["hit_end"],
        "L": obs["L"],
        "hit_avg_rsa": hit_avg_rsa_final,
        "hit_ss_class": hit_ss_class_final,
        "hit_location": obs["hit_location_csv"],
        "hit_helix_frac": float(hit_feat["helix_frac"]),
        "hit_sheet_frac": float(hit_feat["sheet_frac"]),
        "hit_coil_frac": float(hit_feat["coil_frac"]),
        "hit_avg_rsa_calc": float(hit_feat["avg_rsa"]) if np.isfinite(hit_feat["avg_rsa"]) else None,
        "hit_ss_class_calc": hit_feat["ss_class"],
        "dssp_mode": mode,
        "status": "OK",
        "error_detail": None,
    }

    wins = random_windows_for_hit(
        cache=cache,
        L=obs["L"],
        hit_start=obs["hit_start"],
        hit_end=obs["hit_end"],
        k=K_PER_HIT,
        disallow_overlap=DISALLOW_OVERLAP_WITH_HIT,
        rng=rng,
    )

    null_rows = []
    for k_idx, (s, e) in enumerate(wins, start=1):
        feat = window_features_from_cache(cache, s, e)
        if feat is None:
            null_rows.append({
                "k_idx": k_idx,
                "win_start": int(s),
                "win_end": int(e),
                "win_avg_rsa": None,
                "win_helix_frac": None,
                "win_sheet_frac": None,
                "win_coil_frac": None,
                "win_ss_class": None,
                "status": "Win_Low_Coverage",
            })
            continue

        null_rows.append({
            "k_idx": k_idx,
            "win_start": int(s),
            "win_end": int(e),
            "win_avg_rsa": float(feat["avg_rsa"]) if np.isfinite(feat["avg_rsa"]) else None,
            "win_helix_frac": float(feat["helix_frac"]),
            "win_sheet_frac": float(feat["sheet_frac"]),
            "win_coil_frac": float(feat["coil_frac"]),
            "win_ss_class": feat["ss_class"],
            "status": "OK",
        })

    return hit_row, null_rows


# ==============================
# Main
# ==============================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a SQLite random background for Pfam-Pfam SS/RSA analysis using CSV-selected structures."
    )
    parser.add_argument("--struct-csv", default=STRUCT_CSV, help="Input Pfam-Pfam SS/RSA CSV generated from real fragments.")
    parser.add_argument("--output-db", default=OUTPUT_DB, help="Output SQLite database path. Table schemas are unchanged.")
    parser.add_argument("--pfam-dir", default=PFAM_DIR, help="PfamA_seed root directory used to resolve structure files.")
    parser.add_argument("--k-per-hit", type=int, default=K_PER_HIT, help="Number of random windows per real side observation.")
    parser.add_argument("--max-tries-per-win", type=int, default=MAX_TRIES_PER_WIN, help="Maximum random attempts per window.")
    parser.add_argument("--workers", type=int, default=HIT_WORKERS, help="Number of worker processes.")
    parser.add_argument("--inflight", type=int, default=None, help="Maximum submitted futures. Default: workers * 4.")
    parser.add_argument("--min-found-ratio", type=float, default=MIN_FOUND_RATIO, help="Minimum DSSP residue coverage required for a window.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--dssp-exe", default=DSSP_EXE, help="Path to mkdssp/dssp. Default: first executable found in PATH.")
    parser.add_argument("--allow-overlap-with-hit", action="store_true", help="Allow random windows to overlap their source hit interval.")
    parser.add_argument("--allow-duplicate-windows", action="store_true", help="Allow duplicate random windows for the same hit.")
    return parser.parse_args()

def apply_args(args):
    global STRUCT_CSV, OUTPUT_DB, PFAM_DIR, K_PER_HIT, MAX_TRIES_PER_WIN
    global DISALLOW_OVERLAP_WITH_HIT, DISALLOW_DUPLICATE_WINDOWS, DSSP_EXE
    global HIT_WORKERS, INFLIGHT, MIN_FOUND_RATIO, RANDOM_SEED
    STRUCT_CSV = args.struct_csv
    OUTPUT_DB = args.output_db
    PFAM_DIR = args.pfam_dir
    K_PER_HIT = args.k_per_hit
    MAX_TRIES_PER_WIN = args.max_tries_per_win
    DISALLOW_OVERLAP_WITH_HIT = not args.allow_overlap_with_hit
    DISALLOW_DUPLICATE_WINDOWS = not args.allow_duplicate_windows
    DSSP_EXE = args.dssp_exe
    HIT_WORKERS = args.workers
    INFLIGHT = args.inflight if args.inflight is not None else HIT_WORKERS * 4
    MIN_FOUND_RATIO = args.min_found_ratio
    RANDOM_SEED = args.random_seed

def main():
    args = parse_args()
    apply_args(args)

    if DSSP_EXE is None:
        print("[FATAL] mkdssp/dssp not found in PATH.")
        return

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(STRUCT_CSV):
        print(f"[FATAL] STRUCT_CSV not found: {STRUCT_CSV}")
        return
    if not os.path.exists(PFAM_DIR):
        print(f"[FATAL] PFAM_DIR not found: {PFAM_DIR}")
        return

    print(f"[INFO] Read CSV: {STRUCT_CSV}")
    df = pd.read_csv(STRUCT_CSV, low_memory=False)
    print(f"[INFO] Loaded rows: {len(df)}")

    obs_df = build_observations(df)
    if len(obs_df) == 0:
        print("[FATAL] No valid observations found.")
        return

    print(f"[INFO] Valid side observations: {len(obs_df)}  (K_PER_HIT={K_PER_HIT})")
    print(f"[INFO] PFAM_DIR: {PFAM_DIR}")
    print(f"[INFO] DSSP_EXE: {DSSP_EXE}")

    if os.path.exists(OUTPUT_DB):
        os.remove(OUTPUT_DB)
    conn = sqlite_init(OUTPUT_DB)
    cur = conn.cursor()

    tasks = obs_df.to_dict("records")
    total = len(tasks)

    print(f"[INFO] Start MP: workers={HIT_WORKERS}, inflight={INFLIGHT}")

    hit_written = 0
    win_written = 0
    WIN_BATCH = 2000
    win_buf = []

    with ProcessPoolExecutor(max_workers=HIT_WORKERS) as ex:
        inflight_map = {}
        it = iter(tasks)

        def submit_one(t):
            fut = ex.submit(process_one_obs, t)
            inflight_map[fut] = 1

        for _ in range(min(INFLIGHT, total)):
            try:
                submit_one(next(it))
            except StopIteration:
                break

        with tqdm(total=total, desc="Hit+Null windows (PfamA_seed paths)") as pbar:
            while inflight_map:
                for fut in as_completed(list(inflight_map.keys()), timeout=None):
                    inflight_map.pop(fut, None)
                    hit_row, null_rows = fut.result()

                    cur.execute(
                        """
                        INSERT INTO hit_obs
                        (row_idx, side, file_name, score, main_hmm, sub_hmm, fam,
                         pdb_id, pdb_path, chain_used, hit_start, hit_end, L,
                         hit_avg_rsa, hit_ss_class, hit_location,
                         hit_helix_frac, hit_sheet_frac, hit_coil_frac,
                         hit_avg_rsa_calc, hit_ss_class_calc, dssp_mode, status, error_detail)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            hit_row["row_idx"],
                            hit_row["side"],
                            hit_row["file_name"],
                            hit_row["score"],
                            hit_row["main_hmm"],
                            hit_row["sub_hmm"],
                            hit_row["fam"],
                            hit_row["pdb_id"],
                            hit_row["pdb_path"],
                            hit_row["chain_used"],
                            hit_row["hit_start"],
                            hit_row["hit_end"],
                            hit_row["L"],
                            hit_row["hit_avg_rsa"],
                            hit_row["hit_ss_class"],
                            hit_row["hit_location"],
                            hit_row["hit_helix_frac"],
                            hit_row["hit_sheet_frac"],
                            hit_row["hit_coil_frac"],
                            hit_row["hit_avg_rsa_calc"],
                            hit_row["hit_ss_class_calc"],
                            hit_row["dssp_mode"],
                            hit_row["status"],
                            hit_row["error_detail"],
                        )
                    )
                    obs_id = cur.lastrowid
                    hit_written += 1

                    for r in null_rows:
                        win_buf.append((
                            obs_id,
                            r["k_idx"],
                            r["win_start"],
                            r["win_end"],
                            r["win_avg_rsa"],
                            r["win_helix_frac"],
                            r["win_sheet_frac"],
                            r["win_coil_frac"],
                            r["win_ss_class"],
                            r["status"],
                        ))

                    if len(win_buf) >= WIN_BATCH:
                        cur.executemany(
                            """
                            INSERT INTO null_win
                            (obs_id, k_idx, win_start, win_end, win_avg_rsa,
                             win_helix_frac, win_sheet_frac, win_coil_frac,
                             win_ss_class, status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            win_buf
                        )
                        conn.commit()
                        win_written += len(win_buf)
                        win_buf = []
                    else:
                        if hit_written % 200 == 0:
                            conn.commit()

                    pbar.update(1)

                    try:
                        submit_one(next(it))
                    except StopIteration:
                        pass

                    break

    if win_buf:
        cur.executemany(
            """
            INSERT INTO null_win
            (obs_id, k_idx, win_start, win_end, win_avg_rsa,
             win_helix_frac, win_sheet_frac, win_coil_frac,
             win_ss_class, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            win_buf
        )
        conn.commit()
        win_written += len(win_buf)

    conn.commit()
    conn.close()

    print(f"[DONE] SQLite saved: {OUTPUT_DB}")
    print(f"[DONE] hit_obs rows: {hit_written}")
    print(f"[DONE] null_win rows: {win_written}")
    print("[INFO] Join by obs_id for downstream plotting/stats.")


if __name__ == "__main__":
    main()