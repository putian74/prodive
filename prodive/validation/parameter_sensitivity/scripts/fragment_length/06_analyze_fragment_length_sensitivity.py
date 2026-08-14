#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fragment-length sensitivity analysis using empirical length normalization.

Input layout expected:
    input_root/fragment_2/kl_*.npy
    input_root/fragment_3/kl_*.npy
    ...

This script does NOT use hidden sampling by default.
It reads all .npy matrices under fragment_* directories unless MAX_FILES_PER_FRAGMENT is explicitly set.

Main goals:
  1. Estimate length-normalized score distributions for each fragment length k.
  2. Define first-layer thresholds by matched lower percentiles of D_emp.
  3. For each k and threshold, quantify whether retained points become coherent diagonal paths.
  4. Compare short-k noise and long-k information loss using path conversion, isolated/noise fraction, coverage, and span-scale diagnostics.
  The previous 8-13 aa metric is no longer used as a default/main criterion because it was derived from an earlier k=6 RMSD analysis.

Definitions:
  Raw matrix value D_raw: lower means more similar.
  Empirical normalization:
      D_emp = D_raw / (3*k + EMPIRICAL_OFFSET)
  Default EMPIRICAL_OFFSET = 0.407, based on the observed raw-mean length fit.

Path extraction:
  For retained points satisfying D_emp <= threshold, points are grouped by diagonal d = col - row.
  Along each diagonal, consecutive retained x positions are joined into a chain when the x-gap <= MAX_STEP.
  MAX_STEP defaults to k-1, matching the previous logic that allows local diagonal jumps up to fragment length - 1.

Path modes evaluated:
  fixed_r5:
      min path points r = 5 for all k. This matches the original algorithmic setting.
  fixed_span10:
      min path points r_k = max(2, TARGET_MIN_SPAN - k + 1). This fixes the minimum physical span but can over-favor large k because r may become 2.
  fixed_span10_min3:
      min path points r_k = max(3, TARGET_MIN_SPAN - k + 1). This is the recommended balanced comparison mode.
  fixed_span10_min4:
      min path points r_k = max(4, TARGET_MIN_SPAN - k + 1). This is a stricter robustness check.

Outputs:
  OUTPUT_DIR/pass1_fragment_distribution_summary.csv
  OUTPUT_DIR/empirical_first_layer_thresholds.csv
  OUTPUT_DIR/sensitivity_summary.csv
  OUTPUT_DIR/failed_files.csv
  OUTPUT_DIR/*.png summary plots
"""

from __future__ import annotations

import os
import argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import gc
import json
import math
import traceback
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Dict, List, Tuple, Optional, Iterable, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# 1. Configuration
# ==============================================================================

INPUT_ROOT = None
OUTPUT_DIR = None

# Fragment lengths to scan. Set to None to auto-detect all fragment_* directories.
FRAGMENT_LENGTHS = None
# Example explicit list:
# FRAGMENT_LENGTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

# Input file pattern inside each fragment_k directory.
NPY_PATTERN = "*.npy"
RECURSIVE = True

# No hidden sampling by default. If you want a quick test, set this explicitly.
MAX_FILES_PER_FRAGMENT = None

# Empirical normalization: D_emp = D_raw / (3*k + EMPIRICAL_OFFSET)
EMPIRICAL_OFFSET = 0.407

# Optional comparison with theoretical normalization: D_theory = D_raw / (3*k + 2)
ALSO_COMPUTE_THEORY_NORM = True
THEORY_OFFSET = 2.0

# First-layer lower percentile thresholds on D_emp.
# 0.01 means bottom 0.01% most similar points.
FIRST_LAYER_PERCENTILES = [0.01, 0.05, 0.10, 0.50, 1.00]

# Optional fixed thresholds on D_emp. Usually set to [] first.
# Because D_emp scale is around 0.4-0.5 mean, very small thresholds can be too strict.
FIXED_EMP_THRESHOLDS = []
# Example:
# FIXED_EMP_THRESHOLDS = [0.05, 0.08, 0.10, 0.12]

# ------------------------------------------------------------------------------
# Second-layer background score settings
# ------------------------------------------------------------------------------
# Path points are linked by the following maximum diagonal step:
#     linear = (q_bg[row] + t_bg[col]) / (raw_val + EPSILON) * CONST_FACTOR
# which is equivalent to dividing background information by a length-normalized
# distance. For multi-k sensitivity analysis, the default below uses the empirical
# normalized distance D_emp = raw / (3*k + 0.407):
#     S_bg_emp = (q_bg[row] + t_bg[col]) / (D_emp + EPSILON)
# Set SECOND_LAYER_SCORE_MODE = "theory_original" to reproduce the old formula
# using CONST_FACTOR = 3*k + 2.
ENABLE_SECOND_LAYER = True
INCLUDE_NO_SECOND_LAYER_BASELINE = True
SECOND_LAYER_SCORE_MODE = "empirical"  # choices: "empirical", "theory_original"
SECOND_LAYER_EPSILON = 1e-4
SECOND_LAYER_FIXED_THRESHOLDS = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

# Mapping file used to convert numeric output file IDs to PFxxxxx.
MAPPING_FILE = None

# If output file names are kl_0000_0001.npy but mapping IDs are one-based, set this to 1.
# Usually keep 0. If direct lookup fails and AUTO_ONE_BASED_FALLBACK is True, the script
# also tries ID+1 before marking the file as failed.
PAIR_ID_OFFSET = 0
AUTO_ONE_BASED_FALLBACK = True

# Background JSON. For each fragment k, the JSON must map PFxxxxx -> window background vector K(i).
# You can either fill BACKGROUND_JSON_BY_FRAGMENT explicitly, or rely on the template/pattern search.
BACKGROUND_JSON_BY_FRAGMENT = {
}

BACKGROUND_JSON_CANDIDATE_PATTERNS = []

# Histogram bins for percentile estimation. More bins gives better thresholds but costs memory/time.
HIST_BINS = 4000

# Path settings.
TARGET_MIN_SPAN = 10
FIXED_MIN_PATH_POINTS = 5
# If True, use max step = k-1. If False, only directly consecutive x positions are chained.
ALLOW_JUMP_WITHIN_FRAGMENT = True

# Path modes. The old fixed_span10 is kept for backwards compatibility, but it can
# favor longer k because k=9/10 only require two points. The recommended primary
# comparison mode is fixed_span10_min3, with fixed_span10_min4 as a stricter check.
PATH_MODES = ["fixed_r5", "fixed_span10", "fixed_span10_min3", "fixed_span10_min4"]

# Optional descriptive legacy core-span metric. Default False to avoid circular
# reasoning from the previous k=6 RMSD-derived 8-13 aa observation.
INCLUDE_CORE_SPAN_METRICS = False
CORE_SPAN_MIN = 8
CORE_SPAN_MAX = 13

# Multiprocessing.
WORKERS = 20
CHUNKSIZE = 20
MP_CONTEXT = "fork"  # Linux server. Use "spawn" only if fork is unavailable.

# Plotting. Keep False for compute-only runs; use the separate plot script or rerun with True.
MAKE_PLOTS = False
PLOT_DPI = 300




def _parse_int_list(text):
    if text is None or str(text).strip().lower() in {"", "none", "auto"}:
        return None
    return [int(x.strip()) for x in str(text).split(',') if x.strip()]

def _parse_float_list(text):
    if text is None or str(text).strip().lower() in {"", "none"}:
        return []
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze fragment-length sensitivity with empirical normalization and optional second-layer filtering.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing fragment_<k>/ directories with .npy outputs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--fragment-lengths", default=None, help="Comma-separated list, or omit/auto to auto-detect fragment_* directories.")
    parser.add_argument("--npy-pattern", default=NPY_PATTERN, help="Input .npy glob pattern inside each fragment directory.")
    parser.add_argument("--recursive", action="store_true", default=RECURSIVE, help="Search recursively inside fragment directories.")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Disable recursive input search.")
    parser.add_argument("--max-files-per-fragment", type=int, default=MAX_FILES_PER_FRAGMENT, help="Optional quick-test limit per fragment length.")
    parser.add_argument("--empirical-offset", type=float, default=EMPIRICAL_OFFSET, help="Offset in D_emp = D_raw / (3*k + offset).")
    parser.add_argument("--first-layer-percentiles", default=','.join(map(str, FIRST_LAYER_PERCENTILES)), help="Comma-separated lower percentiles.")
    parser.add_argument("--fixed-emp-thresholds", default=','.join(map(str, FIXED_EMP_THRESHOLDS)), help="Comma-separated fixed empirical thresholds.")
    parser.add_argument("--disable-second-layer", action="store_true", help="Disable second-layer background-score filtering.")
    parser.add_argument("--second-layer-thresholds", default=','.join(map(str, SECOND_LAYER_FIXED_THRESHOLDS)), help="Comma-separated second-layer thresholds.")
    parser.add_argument("--mapping-file", type=Path, default=None, help="Pfam mapping file used to decode C++ output IDs. Required unless --disable-second-layer is set.")
    parser.add_argument("--background-json-pattern", default=None, help="Template path containing {k} for the per-fragment background JSON files. Required unless --disable-second-layer is set.")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Worker process count.")
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE, help="Multiprocessing chunksize.")
    parser.add_argument("--make-plots", action="store_true", default=MAKE_PLOTS, help="Write summary plots during analysis.")
    return parser.parse_args()

def apply_runtime_args(args: argparse.Namespace) -> None:
    global INPUT_ROOT, OUTPUT_DIR, FRAGMENT_LENGTHS, NPY_PATTERN, RECURSIVE, MAX_FILES_PER_FRAGMENT
    global EMPIRICAL_OFFSET, FIRST_LAYER_PERCENTILES, FIXED_EMP_THRESHOLDS
    global ENABLE_SECOND_LAYER, SECOND_LAYER_FIXED_THRESHOLDS, MAPPING_FILE, BACKGROUND_JSON_CANDIDATE_PATTERNS
    global WORKERS, CHUNKSIZE, MAKE_PLOTS
    INPUT_ROOT = args.input_root
    OUTPUT_DIR = args.output_dir
    FRAGMENT_LENGTHS = _parse_int_list(args.fragment_lengths)
    NPY_PATTERN = args.npy_pattern
    RECURSIVE = bool(args.recursive)
    MAX_FILES_PER_FRAGMENT = args.max_files_per_fragment
    EMPIRICAL_OFFSET = float(args.empirical_offset)
    FIRST_LAYER_PERCENTILES = _parse_float_list(args.first_layer_percentiles)
    FIXED_EMP_THRESHOLDS = _parse_float_list(args.fixed_emp_thresholds)
    ENABLE_SECOND_LAYER = not bool(args.disable_second_layer)
    SECOND_LAYER_FIXED_THRESHOLDS = _parse_float_list(args.second_layer_thresholds)
    MAPPING_FILE = args.mapping_file
    if args.background_json_pattern:
        BACKGROUND_JSON_CANDIDATE_PATTERNS = [args.background_json_pattern]
    WORKERS = int(args.workers)
    CHUNKSIZE = int(args.chunksize)
    MAKE_PLOTS = bool(args.make_plots)

# ==============================================================================
# 2. Helpers
# ==============================================================================

PFAM_RE = re.compile(r"PF\d{5}")
FRAG_DIR_RE = re.compile(r"fragment[_-]?(\d+)$")


def empirical_denom(k: int) -> float:
    return 3.0 * float(k) + float(EMPIRICAL_OFFSET)


def theory_denom(k: int) -> float:
    return 3.0 * float(k) + float(THEORY_OFFSET)


def min_path_points_for_mode(k: int, mode: str) -> int:
    """Return the minimum number of diagonal points required for a path."""
    k = int(k)
    if mode == "fixed_r5":
        return int(FIXED_MIN_PATH_POINTS)
    if mode == "fixed_span10":
        return max(2, int(TARGET_MIN_SPAN) - k + 1)
    if mode == "fixed_span10_min3":
        return max(3, int(TARGET_MIN_SPAN) - k + 1)
    if mode == "fixed_span10_min4":
        return max(4, int(TARGET_MIN_SPAN) - k + 1)
    raise ValueError(f"Unknown mode: {mode}")


def max_step_for_k(k: int) -> int:
    if ALLOW_JUMP_WITHIN_FRAGMENT:
        return max(1, int(k) - 1)
    return 1


def physical_span_from_xs(xs: np.ndarray, k: int) -> int:
    if xs.size == 0:
        return 0
    return int(xs.max() - xs.min() + 1 + (int(k) - 1))


def normalize_pfam_id(x) -> str:
    s = str(x).strip()
    m = PFAM_RE.search(s)
    return m.group(0) if m else s


def load_pfam_mapping(mapping_file: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    with open(mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, idx = line.split(":", 1)
            name = normalize_pfam_id(name)
            idx_i = int(idx.strip())
            id_to_name[idx_i] = name
            name_to_id[name] = idx_i
    return id_to_name, name_to_id


def resolve_numeric_id_to_pfam(raw_idx: int, id_to_name: Dict[int, str]) -> Optional[str]:
    idx = int(raw_idx) + int(PAIR_ID_OFFSET)
    if idx in id_to_name:
        return id_to_name[idx]
    if AUTO_ONE_BASED_FALLBACK and (idx + 1) in id_to_name:
        return id_to_name[idx + 1]
    return None


def parse_pair_identity_from_filename(path: Path, id_to_name: Dict[int, str]) -> Tuple[str, str]:
    """
    Supported file-name styles:
      1) contains two PF IDs, e.g. PF00001_PF00002.npy
      2) C++ output style, e.g. kl_0000_0001.npy or kl_12_345.npy

    Returns PFxxxxx names for query and target.
    """
    stem = path.stem
    pfams = PFAM_RE.findall(stem)
    if len(pfams) >= 2:
        return normalize_pfam_id(pfams[0]), normalize_pfam_id(pfams[1])

    m = re.search(r"kl[_-](\d+)[_-](\d+)$", stem)
    if not m:
        # fallback: last two integer groups in stem
        nums = re.findall(r"\d+", stem)
        if len(nums) < 2:
            raise ValueError(f"Cannot parse query/target IDs from filename: {path.name}")
        a, b = int(nums[-2]), int(nums[-1])
    else:
        a, b = int(m.group(1)), int(m.group(2))

    pf_a = resolve_numeric_id_to_pfam(a, id_to_name)
    pf_b = resolve_numeric_id_to_pfam(b, id_to_name)
    if pf_a is None or pf_b is None:
        raise KeyError(
            f"Could not map numeric IDs from {path.name}: raw=({a},{b}), "
            f"PAIR_ID_OFFSET={PAIR_ID_OFFSET}"
        )
    return pf_a, pf_b


def find_background_json_for_k(k: int) -> Optional[Path]:
    if int(k) in BACKGROUND_JSON_BY_FRAGMENT:
        p = Path(BACKGROUND_JSON_BY_FRAGMENT[int(k)])
        return p if p.exists() else None

    for pat in BACKGROUND_JSON_CANDIDATE_PATTERNS:
        p = Path(pat.format(k=int(k)))
        if p.exists():
            return p
    return None


def load_background_json(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[str, np.ndarray] = {}
    for key, val in obj.items():
        out[normalize_pfam_id(key)] = np.asarray(val, dtype=np.float32)
    return out


def load_backgrounds_for_fragments(fragments: List[int]) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, str]]:
    bg_by_k: Dict[int, Dict[str, np.ndarray]] = {}
    path_by_k: Dict[int, str] = {}

    if not ENABLE_SECOND_LAYER:
        return bg_by_k, path_by_k

    for k in fragments:
        p = find_background_json_for_k(k)
        if p is None:
            print(f"[WARN] No background JSON found for fragment {k}. Second layer for this k will fail unless disabled.")
            continue
        print(f"[INFO] Loading background for fragment {k}: {p}")
        bg_by_k[int(k)] = load_background_json(p)
        path_by_k[int(k)] = str(p)

    return bg_by_k, path_by_k


# Worker globals inherited by fork pools.
WORKER_ID_TO_NAME: Optional[Dict[int, str]] = None
WORKER_BG_BY_K: Optional[Dict[int, Dict[str, np.ndarray]]] = None


def set_worker_globals(id_to_name: Dict[int, str], bg_by_k: Dict[int, Dict[str, np.ndarray]]):
    global WORKER_ID_TO_NAME, WORKER_BG_BY_K
    WORKER_ID_TO_NAME = id_to_name
    WORKER_BG_BY_K = bg_by_k


def extract_pfams_from_path(path: Path) -> Set[str]:
    return set(PFAM_RE.findall(str(path)))


def auto_detect_fragments(input_root: Path) -> List[int]:
    fragments = []
    for p in input_root.iterdir():
        if not p.is_dir():
            continue
        m = FRAG_DIR_RE.search(p.name)
        if m:
            fragments.append(int(m.group(1)))
    return sorted(set(fragments))


def fragment_dir(input_root: Path, k: int) -> Path:
    return input_root / f"fragment_{int(k)}"


def list_npy_files(input_root: Path, k: int) -> List[Path]:
    d = fragment_dir(input_root, k)
    if not d.is_dir():
        return []
    if RECURSIVE:
        files = sorted(d.rglob(NPY_PATTERN))
    else:
        files = sorted(d.glob(NPY_PATTERN))
    if MAX_FILES_PER_FRAGMENT is not None:
        files = files[: int(MAX_FILES_PER_FRAGMENT)]
    return files


def safe_load_npy(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim != 2:
        # We allow reshape only if it is one-dimensional, but path analysis needs 2D.
        raise ValueError(f"Expected 2D matrix, got shape={arr.shape}")
    return arr


@dataclass
class BasicStats:
    k: int
    files_ok: int = 0
    files_failed: int = 0
    values_count: int = 0
    finite_count: int = 0
    zero_count: int = 0
    raw_sum: float = 0.0
    emp_sum: float = 0.0
    theory_sum: float = 0.0
    raw_min: float = math.inf
    raw_max: float = -math.inf
    emp_min: float = math.inf
    emp_max: float = -math.inf
    theory_min: float = math.inf
    theory_max: float = -math.inf

    def update_from_array(self, arr: np.ndarray):
        x = np.asarray(arr, dtype=np.float64).reshape(-1)
        finite = np.isfinite(x)
        self.values_count += int(x.size)
        if not np.any(finite):
            return
        xf = x[finite]
        self.finite_count += int(xf.size)
        self.zero_count += int(np.count_nonzero(xf == 0))
        self.raw_sum += float(np.sum(xf, dtype=np.float64))
        self.raw_min = min(self.raw_min, float(np.min(xf)))
        self.raw_max = max(self.raw_max, float(np.max(xf)))

        emp = xf / empirical_denom(self.k)
        self.emp_sum += float(np.sum(emp, dtype=np.float64))
        self.emp_min = min(self.emp_min, float(np.min(emp)))
        self.emp_max = max(self.emp_max, float(np.max(emp)))

        if ALSO_COMPUTE_THEORY_NORM:
            th = xf / theory_denom(self.k)
            self.theory_sum += float(np.sum(th, dtype=np.float64))
            self.theory_min = min(self.theory_min, float(np.min(th)))
            self.theory_max = max(self.theory_max, float(np.max(th)))

    def to_row(self) -> Dict[str, object]:
        raw_mean = self.raw_sum / self.finite_count if self.finite_count else np.nan
        emp_mean = self.emp_sum / self.finite_count if self.finite_count else np.nan
        theory_mean = self.theory_sum / self.finite_count if self.finite_count else np.nan
        return {
            "fragment_length": self.k,
            "files_ok": self.files_ok,
            "files_failed": self.files_failed,
            "values_count": self.values_count,
            "finite_count": self.finite_count,
            "zero_count": self.zero_count,
            "zero_fraction": self.zero_count / self.finite_count if self.finite_count else np.nan,
            "raw_mean": raw_mean,
            "raw_min": self.raw_min if np.isfinite(self.raw_min) else np.nan,
            "raw_max": self.raw_max if np.isfinite(self.raw_max) else np.nan,
            "emp_norm_mean": emp_mean,
            "emp_norm_min": self.emp_min if np.isfinite(self.emp_min) else np.nan,
            "emp_norm_max": self.emp_max if np.isfinite(self.emp_max) else np.nan,
            "theory_norm_mean": theory_mean,
            "theory_norm_min": self.theory_min if np.isfinite(self.theory_min) else np.nan,
            "theory_norm_max": self.theory_max if np.isfinite(self.theory_max) else np.nan,
            "empirical_denominator": empirical_denom(self.k),
            "theory_denominator": theory_denom(self.k),
        }


# ==============================================================================
# 3. Pass 1: min/max and means
# ==============================================================================


def worker_basic_stats(task: Tuple[int, str]) -> Dict[str, object]:
    k, path_str = task
    path = Path(path_str)
    st = BasicStats(k=int(k))
    try:
        arr = safe_load_npy(path)
        st.files_ok = 1
        st.update_from_array(arr)
        return {"ok": True, "k": int(k), "stats": st.to_row(), "file": str(path)}
    except Exception as e:
        return {
            "ok": False,
            "k": int(k),
            "file": str(path),
            "error": str(e),
            "traceback": traceback.format_exc(limit=2),
        }


def merge_basic_rows(rows: List[Dict[str, object]], k: int) -> Dict[str, object]:
    acc = BasicStats(k=k)
    for r in rows:
        acc.files_ok += int(r.get("files_ok", 0))
        acc.files_failed += int(r.get("files_failed", 0))
        acc.values_count += int(r.get("values_count", 0))
        acc.finite_count += int(r.get("finite_count", 0))
        acc.zero_count += int(r.get("zero_count", 0))
        acc.raw_sum += float(r.get("raw_mean", 0.0)) * int(r.get("finite_count", 0)) if pd.notna(r.get("raw_mean", np.nan)) else 0.0
        acc.emp_sum += float(r.get("emp_norm_mean", 0.0)) * int(r.get("finite_count", 0)) if pd.notna(r.get("emp_norm_mean", np.nan)) else 0.0
        acc.theory_sum += float(r.get("theory_norm_mean", 0.0)) * int(r.get("finite_count", 0)) if pd.notna(r.get("theory_norm_mean", np.nan)) else 0.0
        if pd.notna(r.get("raw_min", np.nan)):
            acc.raw_min = min(acc.raw_min, float(r["raw_min"]))
        if pd.notna(r.get("raw_max", np.nan)):
            acc.raw_max = max(acc.raw_max, float(r["raw_max"]))
        if pd.notna(r.get("emp_norm_min", np.nan)):
            acc.emp_min = min(acc.emp_min, float(r["emp_norm_min"]))
        if pd.notna(r.get("emp_norm_max", np.nan)):
            acc.emp_max = max(acc.emp_max, float(r["emp_norm_max"]))
        if pd.notna(r.get("theory_norm_min", np.nan)):
            acc.theory_min = min(acc.theory_min, float(r["theory_norm_min"]))
        if pd.notna(r.get("theory_norm_max", np.nan)):
            acc.theory_max = max(acc.theory_max, float(r["theory_norm_max"]))
    return acc.to_row()


# ==============================================================================
# 4. Pass 2: histogram thresholds
# ==============================================================================


def worker_hist(task: Tuple[int, str, float, float, int]) -> Dict[str, object]:
    k, path_str, emp_min, emp_max, bins = task
    path = Path(path_str)
    try:
        arr = safe_load_npy(path)
        x = np.asarray(arr, dtype=np.float64).reshape(-1)
        x = x[np.isfinite(x)] / empirical_denom(k)
        if x.size == 0:
            hist = np.zeros(int(bins), dtype=np.int64)
        else:
            hist, _ = np.histogram(x, bins=int(bins), range=(float(emp_min), float(emp_max)))
            hist = hist.astype(np.int64, copy=False)
        return {"ok": True, "k": int(k), "hist": hist, "file": str(path)}
    except Exception as e:
        return {
            "ok": False,
            "k": int(k),
            "file": str(path),
            "error": str(e),
            "traceback": traceback.format_exc(limit=2),
        }


def thresholds_from_hist(hist: np.ndarray, lo: float, hi: float, percentiles: List[float]) -> List[Dict[str, object]]:
    total = int(hist.sum())
    if total <= 0 or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []

    cdf = np.cumsum(hist)
    width = (hi - lo) / len(hist)
    rows = []
    for p in percentiles:
        frac = float(p) / 100.0
        target = max(1, int(math.ceil(total * frac)))
        idx = int(np.searchsorted(cdf, target, side="left"))
        idx = min(max(idx, 0), len(hist) - 1)
        thr = lo + (idx + 1) * width
        rows.append({
            "threshold_mode": "percentile",
            "percentile": float(p),
            "emp_threshold": float(thr),
            "hist_bin_index": idx,
            "target_retained_fraction": frac,
        })
    for fixed in FIXED_EMP_THRESHOLDS:
        rows.append({
            "threshold_mode": "fixed",
            "percentile": np.nan,
            "emp_threshold": float(fixed),
            "hist_bin_index": np.nan,
            "target_retained_fraction": np.nan,
        })
    return rows


# ==============================================================================
# 5. Pass 3: threshold + path metrics
# ==============================================================================


def summarize_paths_from_points(rows: np.ndarray, cols: np.ndarray, k: int, min_points: int) -> Dict[str, object]:
    """
    Fast diagonal-chain summary.

    Retained points are grouped by diagonal d = col - row.
    Within each diagonal, sort row coordinates and split when row-gap > max_step.
    A chain is a path if its point count >= min_points.
    """
    n_points = int(rows.size)
    if n_points == 0:
        return {
            "retained_points": 0,
            "path_count": 0,
            "path_points": 0,
            "path_conversion_rate": 0.0,
            "isolated_or_noise_points": 0,
            "isolated_or_noise_fraction": np.nan,
            "mean_path_points": np.nan,
            "mean_physical_span": np.nan,
            "median_physical_span": np.nan,
            "paths_span_8_13": 0,
            "frac_paths_span_8_13": np.nan,
        }

    d = cols.astype(np.int64, copy=False) - rows.astype(np.int64, copy=False)
    order = np.lexsort((rows, d))
    d_sorted = d[order]
    x_sorted = rows[order].astype(np.int64, copy=False)

    step_limit = max_step_for_k(k)
    path_count = 0
    path_points = 0
    path_point_counts = []
    spans = []
    core_count = 0

    start = 0
    n = len(x_sorted)

    while start < n:
        diag = d_sorted[start]
        end = start + 1
        while end < n and d_sorted[end] == diag:
            end += 1

        xs = x_sorted[start:end]
        # xs are sorted because of lexsort.
        seg_start = 0
        for idx in range(1, xs.size + 1):
            should_break = False
            if idx == xs.size:
                should_break = True
            else:
                if xs[idx] - xs[idx - 1] > step_limit:
                    should_break = True

            if should_break:
                seg = xs[seg_start:idx]
                if seg.size >= min_points:
                    span = physical_span_from_xs(seg, k)
                    path_count += 1
                    path_points += int(seg.size)
                    path_point_counts.append(int(seg.size))
                    spans.append(int(span))
                    if CORE_SPAN_MIN <= span <= CORE_SPAN_MAX:
                        core_count += 1
                seg_start = idx

        start = end

    noise_points = n_points - path_points
    return {
        "retained_points": n_points,
        "path_count": int(path_count),
        "path_points": int(path_points),
        "path_conversion_rate": float(path_points / n_points) if n_points else 0.0,
        "isolated_or_noise_points": int(noise_points),
        "isolated_or_noise_fraction": float(noise_points / n_points) if n_points else np.nan,
        "mean_path_points": float(np.mean(path_point_counts)) if path_point_counts else np.nan,
        "mean_physical_span": float(np.mean(spans)) if spans else np.nan,
        "median_physical_span": float(np.median(spans)) if spans else np.nan,
        "paths_span_8_13": int(core_count),
        "frac_paths_span_8_13": float(core_count / path_count) if path_count else np.nan,
    }


def get_second_layer_specs() -> List[Tuple[str, Optional[float]]]:
    """
    Returns list of (second_label, second_threshold).
    second_threshold None means no second-layer filtering.
    """
    specs: List[Tuple[str, Optional[float]]] = []
    if INCLUDE_NO_SECOND_LAYER_BASELINE:
        specs.append(("NoS", None))
    if ENABLE_SECOND_LAYER:
        for x in SECOND_LAYER_FIXED_THRESHOLDS:
            specs.append((f"S{float(x):g}", float(x)))
    return specs


def compute_second_layer_score(raw: np.ndarray, k: int, q_bg: np.ndarray, t_bg: np.ndarray) -> np.ndarray:
    """
    Compute background-normalized score matrix.

    theory_original reproduces the old code:
        linear = (q_bg[row] + t_bg[col]) / (raw + eps) * (3*k + 2)

    empirical uses the new length-corrected distance:
        D_emp = raw / (3*k + 0.407)
        S_bg = (q_bg[row] + t_bg[col]) / (D_emp + eps)
    """
    bg_sum = q_bg[:, None].astype(np.float64, copy=False) + t_bg[None, :].astype(np.float64, copy=False)
    raw64 = raw.astype(np.float64, copy=False)

    if SECOND_LAYER_SCORE_MODE == "theory_original":
        return bg_sum / (raw64 + SECOND_LAYER_EPSILON) * theory_denom(k)

    if SECOND_LAYER_SCORE_MODE == "empirical":
        d_emp = raw64 / empirical_denom(k)
        return bg_sum / (d_emp + SECOND_LAYER_EPSILON)

    raise ValueError(f"Unknown SECOND_LAYER_SCORE_MODE: {SECOND_LAYER_SCORE_MODE}")


def worker_analyze_file(task: Tuple[int, str, List[Tuple[str, float, float]], List[Tuple[str, Optional[float]]]]) -> Dict[str, object]:
    """
    threshold_specs: list of (first_threshold_label, percentile_or_nan, emp_threshold)
    second_specs: list of (second_label, second_threshold_or_None)
    """
    k, path_str, threshold_specs, second_specs = task
    path = Path(path_str)
    try:
        if WORKER_ID_TO_NAME is None:
            raise RuntimeError("WORKER_ID_TO_NAME is not initialized")

        arr = safe_load_npy(path)
        shape = tuple(arr.shape)
        raw = np.asarray(arr, dtype=np.float64)
        emp = raw / empirical_denom(k)
        finite = np.isfinite(emp)
        total_cells = int(emp.size)

        query_pfam, target_pfam = parse_pair_identity_from_filename(path, WORKER_ID_TO_NAME)
        pfams = {query_pfam, target_pfam}

        sbg = None
        second_layer_available = False
        second_layer_error = ""

        needs_second = ENABLE_SECOND_LAYER and any(thr is not None for _, thr in second_specs)
        if needs_second:
            if WORKER_BG_BY_K is None or int(k) not in WORKER_BG_BY_K:
                second_layer_error = f"No background loaded for fragment {k}"
            else:
                bg_for_k = WORKER_BG_BY_K[int(k)]
                if query_pfam not in bg_for_k or target_pfam not in bg_for_k:
                    second_layer_error = f"Missing background vector for {query_pfam} or {target_pfam}"
                else:
                    q_bg = np.asarray(bg_for_k[query_pfam], dtype=np.float32)
                    t_bg = np.asarray(bg_for_k[target_pfam], dtype=np.float32)
                    if q_bg.size != shape[0] or t_bg.size != shape[1]:
                        second_layer_error = (
                            f"Background length mismatch for {path.name}: "
                            f"matrix={shape}, {query_pfam} bg={q_bg.size}, {target_pfam} bg={t_bg.size}"
                        )
                    else:
                        sbg = compute_second_layer_score(raw, int(k), q_bg, t_bg)
                        second_layer_available = True

        out = []
        for first_label, percentile, first_thr in threshold_specs:
            first_mask = finite & (emp <= float(first_thr))

            for second_label, second_thr in second_specs:
                if second_thr is None:
                    mask = first_mask
                    second_used = False
                    current_second_error = ""
                else:
                    second_used = True
                    if not second_layer_available or sbg is None:
                        # Do not silently fake second-layer results.
                        out.append({
                            "first_threshold_label": first_label,
                            "threshold_label": first_label,
                            "percentile": percentile,
                            "emp_threshold": float(first_thr),
                            "second_layer_label": second_label,
                            "second_layer_threshold": float(second_thr),
                            "second_layer_used": True,
                            "second_layer_available": False,
                            "second_layer_error": second_layer_error,
                            "path_mode": "failed_second_layer",
                            "min_path_points": np.nan,
                            "total_cells": total_cells,
                            "retained_points": 0,
                            "candidate_point_density": 0.0,
                            "path_count": 0,
                            "path_points": 0,
                            "path_conversion_rate": np.nan,
                            "isolated_or_noise_points": 0,
                            "isolated_or_noise_fraction": np.nan,
                            "mean_path_points": np.nan,
                            "mean_physical_span": np.nan,
                            "median_physical_span": np.nan,
                            "paths_span_8_13": 0,
                            "frac_paths_span_8_13": np.nan,
                            "has_any_retained": False,
                            "pfams": [],
                        })
                        continue
                    mask = first_mask & np.isfinite(sbg) & (sbg >= float(second_thr))
                    current_second_error = ""

                retained = int(np.count_nonzero(mask))
                if retained <= 0:
                    for mode in PATH_MODES:
                        out.append({
                            "first_threshold_label": first_label,
                            "threshold_label": f"{first_label}|{second_label}",
                            "percentile": percentile,
                            "emp_threshold": float(first_thr),
                            "second_layer_label": second_label,
                            "second_layer_threshold": np.nan if second_thr is None else float(second_thr),
                            "second_layer_used": second_used,
                            "second_layer_available": (not second_used) or second_layer_available,
                            "second_layer_error": current_second_error,
                            "path_mode": mode,
                            "min_path_points": min_path_points_for_mode(k, mode),
                            "total_cells": total_cells,
                            "retained_points": 0,
                            "candidate_point_density": 0.0,
                            "path_count": 0,
                            "path_points": 0,
                            "path_conversion_rate": 0.0,
                            "isolated_or_noise_points": 0,
                            "isolated_or_noise_fraction": np.nan,
                            "mean_path_points": np.nan,
                            "mean_physical_span": np.nan,
                            "median_physical_span": np.nan,
                            "paths_span_8_13": 0,
                            "frac_paths_span_8_13": np.nan,
                            "has_any_retained": False,
                            "pfams": [],
                        })
                    continue

                rr, cc = np.nonzero(mask)
                rr = rr.astype(np.int32, copy=False)
                cc = cc.astype(np.int32, copy=False)

                for mode in PATH_MODES:
                    rmin = min_path_points_for_mode(k, mode)
                    stats = summarize_paths_from_points(rr, cc, k=int(k), min_points=rmin)
                    stats.update({
                        "first_threshold_label": first_label,
                        "threshold_label": f"{first_label}|{second_label}",
                        "percentile": percentile,
                        "emp_threshold": float(first_thr),
                        "second_layer_label": second_label,
                        "second_layer_threshold": np.nan if second_thr is None else float(second_thr),
                        "second_layer_used": second_used,
                        "second_layer_available": (not second_used) or second_layer_available,
                        "second_layer_error": current_second_error,
                        "path_mode": mode,
                        "min_path_points": rmin,
                        "total_cells": total_cells,
                        "candidate_point_density": float(retained / total_cells) if total_cells else np.nan,
                        "has_any_retained": True,
                        "pfams": sorted(pfams),
                    })
                    out.append(stats)

        return {"ok": True, "k": int(k), "file": str(path), "shape": shape, "rows": out}

    except Exception as e:
        return {
            "ok": False,
            "k": int(k),
            "file": str(path),
            "error": str(e),
            "traceback": traceback.format_exc(limit=2),
        }


def aggregate_analysis_rows(rows: Iterable[Dict[str, object]]) -> pd.DataFrame:
    """Aggregate per-file threshold/path rows."""
    buckets: Dict[Tuple, Dict[str, object]] = {}

    for r in rows:
        key = (
            int(r["fragment_length"]),
            str(r["threshold_label"]),
            str(r.get("second_layer_label", "NoS")),
            str(r["path_mode"]),
        )
        b = buckets.setdefault(key, {
            "fragment_length": int(r["fragment_length"]),
            "threshold_label": str(r["threshold_label"]),
            "first_threshold_label": str(r.get("first_threshold_label", r["threshold_label"])),
            "threshold_mode": str(r["threshold_mode"]),
            "percentile": r.get("percentile", np.nan),
            "emp_threshold": float(r["emp_threshold"]),
            "second_layer_label": str(r.get("second_layer_label", "NoS")),
            "second_layer_threshold": r.get("second_layer_threshold", np.nan),
            "second_layer_used": bool(r.get("second_layer_used", False)),
            "second_layer_available": bool(r.get("second_layer_available", True)),
            "second_layer_error_count": 0,
            "path_mode": str(r["path_mode"]),
            "min_path_points": int(r["min_path_points"]) if pd.notna(r.get("min_path_points", np.nan)) else -1,
            "files_with_any_retained": 0,
            "files_processed": 0,
            "total_cells": 0,
            "retained_points": 0,
            "path_count": 0,
            "path_points": 0,
            "isolated_or_noise_points": 0,
            "paths_span_8_13": 0,
            "weighted_sum_path_points": 0.0,
            "weighted_sum_physical_span": 0.0,
            "path_count_for_means": 0,
            "covered_pfams": set(),
        })

        b["files_processed"] += 1
        b["total_cells"] += int(r.get("total_cells", 0))
        b["retained_points"] += int(r.get("retained_points", 0))
        b["path_count"] += int(r.get("path_count", 0))
        b["path_points"] += int(r.get("path_points", 0))
        b["isolated_or_noise_points"] += int(r.get("isolated_or_noise_points", 0))
        b["paths_span_8_13"] += int(r.get("paths_span_8_13", 0))
        if r.get("has_any_retained", False):
            b["files_with_any_retained"] += 1
            b["covered_pfams"].update(r.get("pfams", []))

        if r.get("second_layer_used", False) and not r.get("second_layer_available", True):
            b["second_layer_error_count"] += 1

        pc = int(r.get("path_count", 0))
        if pc > 0:
            mp = r.get("mean_path_points", np.nan)
            ms = r.get("mean_physical_span", np.nan)
            if pd.notna(mp):
                b["weighted_sum_path_points"] += float(mp) * pc
            if pd.notna(ms):
                b["weighted_sum_physical_span"] += float(ms) * pc
            b["path_count_for_means"] += pc

    out_rows = []
    for b in buckets.values():
        retained = int(b["retained_points"])
        total = int(b["total_cells"])
        pc = int(b["path_count"])
        ppoints = int(b["path_points"])
        noise = int(b["isolated_or_noise_points"])
        out_rows.append({
            "fragment_length": b["fragment_length"],
            "threshold_label": b["threshold_label"],
            "first_threshold_label": b["first_threshold_label"],
            "threshold_mode": b["threshold_mode"],
            "percentile": b["percentile"],
            "emp_threshold": b["emp_threshold"],
            "second_layer_label": b["second_layer_label"],
            "second_layer_threshold": b["second_layer_threshold"],
            "second_layer_used": b["second_layer_used"],
            "second_layer_available": b["second_layer_available"],
            "second_layer_error_count": b["second_layer_error_count"],
            "path_mode": b["path_mode"],
            "min_path_points": b["min_path_points"],
            "files_processed": b["files_processed"],
            "files_with_any_retained": b["files_with_any_retained"],
            "total_cells": total,
            "retained_points": retained,
            "candidate_point_density": retained / total if total else np.nan,
            "path_count": pc,
            "path_points": ppoints,
            "path_conversion_rate": ppoints / retained if retained else np.nan,
            "isolated_or_noise_points": noise,
            "isolated_or_noise_fraction": noise / retained if retained else np.nan,
            "mean_path_points": b["weighted_sum_path_points"] / b["path_count_for_means"] if b["path_count_for_means"] else np.nan,
            "mean_physical_span": b["weighted_sum_physical_span"] / b["path_count_for_means"] if b["path_count_for_means"] else np.nan,
            "paths_span_8_13": b["paths_span_8_13"],
            "frac_paths_span_8_13": b["paths_span_8_13"] / pc if pc else np.nan,
            "covered_pfams_from_filename": len(b["covered_pfams"]),
        })

    df = pd.DataFrame(out_rows)
    if not df.empty:
        df = df.sort_values(
            by=["first_threshold_label", "second_layer_label", "path_mode", "fragment_length"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    return df


# ==============================================================================
# 6. Plots
# ==============================================================================


def plot_metric(summary_df: pd.DataFrame, metric: str, out_path: Path, ylabel: str):
    if summary_df.empty or metric not in summary_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    grouped = summary_df.groupby(["threshold_label", "path_mode"], sort=False)
    for (thr_label, mode), sub in grouped:
        sub = sub.sort_values("fragment_length")
        ax.plot(
            sub["fragment_length"],
            sub[metric],
            marker="o",
            linewidth=1.8,
            label=f"{thr_label} | {mode}",
        )
    ax.axvline(6, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Fragment length k")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " across fragment lengths")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def make_plots(summary_df: pd.DataFrame, dist_df: pd.DataFrame, out_dir: Path):
    if not MAKE_PLOTS:
        return

    plot_metric(
        summary_df,
        "retained_points",
        out_dir / "metric_retained_points.png",
        "Retained candidate points",
    )
    plot_metric(
        summary_df,
        "candidate_point_density",
        out_dir / "metric_candidate_point_density.png",
        "Candidate point density",
    )
    plot_metric(
        summary_df,
        "path_conversion_rate",
        out_dir / "metric_path_conversion_rate.png",
        "Path conversion rate = path_points / retained_points",
    )
    plot_metric(
        summary_df,
        "isolated_or_noise_fraction",
        out_dir / "metric_isolated_or_noise_fraction.png",
        "Isolated/noise fraction",
    )
    plot_metric(
        summary_df,
        "path_count",
        out_dir / "metric_path_count.png",
        "Diagonal path count",
    )
    plot_metric(
        summary_df,
        "covered_pfams_from_filename",
        out_dir / "metric_covered_pfams_from_filename.png",
        "Covered Pfam families from filenames",
    )
    plot_metric(
        summary_df,
        "mean_physical_span",
        out_dir / "metric_mean_physical_span.png",
        "Mean physical span of paths",
    )
    if INCLUDE_CORE_SPAN_METRICS and "frac_paths_span_8_13" in summary_df.columns:
        plot_metric(
            summary_df,
            "frac_paths_span_8_13",
            out_dir / "metric_frac_paths_span_8_13_legacy_descriptive.png",
            "Legacy descriptive fraction of paths with span 8-13 aa",
        )

    if not dist_df.empty:
        fig, ax = plt.subplots(figsize=(8.0, 5.8))
        d = dist_df.sort_values("fragment_length")
        ax.plot(d["fragment_length"], d["raw_mean"], marker="o", label="raw mean")
        ax.set_xlabel("Fragment length k")
        ax.set_ylabel("Raw mean")
        ax.set_title("Raw mean across fragment lengths")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "distribution_raw_mean.png", dpi=PLOT_DPI)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.0, 5.8))
        ax.plot(d["fragment_length"], d["emp_norm_mean"], marker="o", label=f"raw / (3k+{EMPIRICAL_OFFSET})")
        if ALSO_COMPUTE_THEORY_NORM:
            ax.plot(d["fragment_length"], d["theory_norm_mean"], marker="o", label="raw / (3k+2)")
        ax.set_xlabel("Fragment length k")
        ax.set_ylabel("Normalized mean")
        ax.set_title("Normalized mean across fragment lengths")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "distribution_normalized_mean.png", dpi=PLOT_DPI)
        plt.close(fig)


# ==============================================================================
# 7. Main
# ==============================================================================


def run_pool(tasks, worker_func, total: int, desc: str):
    ctx = get_context(MP_CONTEXT)
    with ctx.Pool(processes=WORKERS, maxtasksperchild=100) as pool:
        iterator = pool.imap_unordered(worker_func, tasks, chunksize=CHUNKSIZE)
        for res in tqdm(iterator, total=total, desc=desc, unit="file"):
            yield res


def main():
    args = parse_args()
    apply_runtime_args(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if FRAGMENT_LENGTHS is None:
        fragments = auto_detect_fragments(INPUT_ROOT)
    else:
        fragments = [int(x) for x in FRAGMENT_LENGTHS]

    if not fragments:
        raise RuntimeError(f"No fragment_* directories found under {INPUT_ROOT}")

    files_by_k: Dict[int, List[Path]] = {}
    for k in fragments:
        files_by_k[k] = list_npy_files(INPUT_ROOT, k)

    print("=" * 100)
    print("Fragment-length sensitivity analysis with empirical normalization")
    print(f"INPUT_ROOT: {INPUT_ROOT}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"Fragments: {fragments}")
    print(f"Empirical normalization: D_emp = D_raw / (3*k + {EMPIRICAL_OFFSET})")
    print(f"First-layer percentiles: {FIRST_LAYER_PERCENTILES}")
    print(f"Fixed D_emp thresholds: {FIXED_EMP_THRESHOLDS}")
    print(f"Second layer enabled: {ENABLE_SECOND_LAYER}")
    print(f"Second-layer score mode: {SECOND_LAYER_SCORE_MODE}")
    print(f"Second-layer fixed thresholds: {SECOND_LAYER_FIXED_THRESHOLDS}")
    print(f"Mapping file: {MAPPING_FILE}")
    print(f"WORKERS: {WORKERS}")
    print(f"MAX_FILES_PER_FRAGMENT: {MAX_FILES_PER_FRAGMENT}")
    print("No hidden subset or sampling is used unless MAX_FILES_PER_FRAGMENT is explicitly set.")
    print("=" * 100)

    for k in fragments:
        print(f"fragment {k}: {len(files_by_k[k])} .npy files")

    # Mapping/background are needed only for Pass 3 second-layer filtering, but loaded here
    # so fork workers can inherit them without re-reading JSON in every task.
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    bg_by_k: Dict[int, Dict[str, np.ndarray]] = {}
    bg_path_by_k: Dict[int, str] = {}

    if ENABLE_SECOND_LAYER:
        if MAPPING_FILE is None:
            raise ValueError("--mapping-file is required when second-layer filtering is enabled.")
        if not MAPPING_FILE.exists():
            raise FileNotFoundError(f"MAPPING_FILE does not exist: {MAPPING_FILE}")
        id_to_name, name_to_id = load_pfam_mapping(MAPPING_FILE)
        print(f"[INFO] Mapping entries: {len(id_to_name)} | {MAPPING_FILE}")
        bg_by_k, bg_path_by_k = load_backgrounds_for_fragments(fragments)
        missing_backgrounds = sorted(set(fragments) - set(bg_by_k))
        if missing_backgrounds:
            raise FileNotFoundError(
                "Missing background JSON files for fragment lengths: "
                + ", ".join(map(str, missing_backgrounds))
                + ". Set --background-json-pattern or use --disable-second-layer."
            )
        print(f"[INFO] Backgrounds loaded for fragments: {sorted(bg_by_k)}")
    else:
        print("[INFO] Second layer disabled; mapping/background loading skipped.")

    set_worker_globals(id_to_name, bg_by_k)

    all_failures = []

    # --------------------------------------------------------------------------
    # Pass 1
    # --------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("Pass 1/3: basic raw and normalized distribution stats")
    print("=" * 100)

    pass1_rows_by_k: Dict[int, List[Dict[str, object]]] = {k: [] for k in fragments}
    tasks = []
    for k in fragments:
        for p in files_by_k[k]:
            tasks.append((k, str(p)))

    for res in run_pool(tasks, worker_basic_stats, total=len(tasks), desc="Pass1"):
        if res["ok"]:
            pass1_rows_by_k[int(res["k"])].append(res["stats"])
        else:
            all_failures.append({
                "pass": "pass1",
                "fragment_length": int(res.get("k", -1)),
                "file": res.get("file", ""),
                "error": res.get("error", ""),
                "traceback": res.get("traceback", ""),
            })

    dist_rows = []
    for k in fragments:
        row = merge_basic_rows(pass1_rows_by_k[k], k)
        # Count failed files for this k.
        row["files_failed"] += sum(1 for x in all_failures if x["pass"] == "pass1" and x["fragment_length"] == k)
        dist_rows.append(row)

    dist_df = pd.DataFrame(dist_rows).sort_values("fragment_length").reset_index(drop=True)
    dist_csv = OUTPUT_DIR / "pass1_fragment_distribution_summary.csv"
    dist_df.to_csv(dist_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Pass1 summary: {dist_csv}")
    print(dist_df[["fragment_length", "files_ok", "finite_count", "zero_count", "raw_mean", "emp_norm_mean", "theory_norm_mean"]].to_string(index=False))

    # --------------------------------------------------------------------------
    # Pass 2: histograms for thresholds
    # --------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("Pass 2/3: histogram-based percentile threshold estimation")
    print("=" * 100)

    hist_by_k = {k: np.zeros(HIST_BINS, dtype=np.int64) for k in fragments}
    emp_minmax = {}
    for _, row in dist_df.iterrows():
        k = int(row["fragment_length"])
        lo = float(row["emp_norm_min"])
        hi = float(row["emp_norm_max"])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # Widen invalid range slightly to avoid histogram failure.
            lo, hi = 0.0, 1.0
        emp_minmax[k] = (lo, hi)

    hist_tasks = []
    for k in fragments:
        lo, hi = emp_minmax[k]
        for p in files_by_k[k]:
            hist_tasks.append((k, str(p), lo, hi, HIST_BINS))

    for res in run_pool(hist_tasks, worker_hist, total=len(hist_tasks), desc="Pass2"):
        if res["ok"]:
            hist_by_k[int(res["k"])] += res["hist"]
        else:
            all_failures.append({
                "pass": "pass2",
                "fragment_length": int(res.get("k", -1)),
                "file": res.get("file", ""),
                "error": res.get("error", ""),
                "traceback": res.get("traceback", ""),
            })

    threshold_rows = []
    for k in fragments:
        lo, hi = emp_minmax[k]
        rows = thresholds_from_hist(hist_by_k[k], lo, hi, FIRST_LAYER_PERCENTILES)
        for r in rows:
            r["fragment_length"] = k
            r["empirical_denominator"] = empirical_denom(k)
            if r["threshold_mode"] == "percentile":
                r["threshold_label"] = f"P{float(r['percentile']):g}"
            else:
                r["threshold_label"] = f"T{float(r['emp_threshold']):g}"
            threshold_rows.append(r)

    thr_df = pd.DataFrame(threshold_rows)
    thr_df = thr_df.sort_values(["threshold_label", "fragment_length"]).reset_index(drop=True)
    thr_csv = OUTPUT_DIR / "empirical_first_layer_thresholds.csv"
    thr_df.to_csv(thr_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Empirical thresholds: {thr_csv}")
    print(thr_df[["fragment_length", "threshold_label", "threshold_mode", "percentile", "emp_threshold"]].to_string(index=False))

    # --------------------------------------------------------------------------
    # Pass 3: apply thresholds and summarize diagonal path coherence
    # --------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("Pass 3/3: first-layer filtering and diagonal-path sensitivity metrics")
    print("=" * 100)

    thr_specs_by_k: Dict[int, List[Tuple[str, float, float, str]]] = {}
    # Internally: label, percentile, threshold, mode string
    for k in fragments:
        sub = thr_df[thr_df["fragment_length"] == k]
        specs = []
        for row in sub.itertuples(index=False):
            label = str(getattr(row, "threshold_label"))
            percentile = getattr(row, "percentile")
            thr = float(getattr(row, "emp_threshold"))
            specs.append((label, percentile, thr, str(getattr(row, "threshold_mode"))))
        thr_specs_by_k[k] = specs

    second_specs = get_second_layer_specs()
    second_spec_df = pd.DataFrame([
        {"second_layer_label": label, "second_layer_threshold": np.nan if thr is None else float(thr)}
        for label, thr in second_specs
    ])
    second_spec_csv = OUTPUT_DIR / "second_layer_threshold_specs.csv"
    second_spec_df.to_csv(second_spec_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Second-layer threshold specs: {second_spec_csv}")

    bg_report_csv = OUTPUT_DIR / "background_files_by_fragment.csv"
    pd.DataFrame([
        {"fragment_length": int(k), "background_json": bg_path_by_k.get(int(k), "")}
        for k in fragments
    ]).to_csv(bg_report_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Background file report: {bg_report_csv}")

    analyze_tasks = []
    for k in fragments:
        # worker receives only label, percentile, threshold. threshold_mode restored from label map later.
        specs_for_worker = [(x[0], x[1], x[2]) for x in thr_specs_by_k[k]]
        for p in files_by_k[k]:
            analyze_tasks.append((k, str(p), specs_for_worker, second_specs))

    # map (k,label) -> threshold mode
    threshold_mode_by_key = {}
    for k, specs in thr_specs_by_k.items():
        for label, percentile, thr, mode in specs:
            threshold_mode_by_key[(k, label)] = mode

    per_file_rows_for_aggregation = []

    for res in run_pool(analyze_tasks, worker_analyze_file, total=len(analyze_tasks), desc="Pass3"):
        if not res["ok"]:
            all_failures.append({
                "pass": "pass3",
                "fragment_length": int(res.get("k", -1)),
                "file": res.get("file", ""),
                "error": res.get("error", ""),
                "traceback": res.get("traceback", ""),
            })
            continue

        k = int(res["k"])
        for r in res["rows"]:
            r["fragment_length"] = k
            r["file"] = res["file"]
            r["threshold_mode"] = threshold_mode_by_key.get((k, str(r.get("first_threshold_label", r["threshold_label"]))), "unknown")
            per_file_rows_for_aggregation.append(r)

    summary_df = aggregate_analysis_rows(per_file_rows_for_aggregation)
    summary_csv = OUTPUT_DIR / "sensitivity_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Sensitivity summary: {summary_csv}")

    display_cols = [
        "fragment_length", "first_threshold_label", "second_layer_label", "threshold_label",
        "path_mode", "min_path_points", "second_layer_error_count",
        "retained_points", "candidate_point_density", "path_count",
        "path_conversion_rate", "isolated_or_noise_fraction",
        "frac_paths_span_8_13", "covered_pfams_from_filename",
    ]
    if not summary_df.empty:
        print(summary_df[display_cols].head(80).to_string(index=False))

    # Failure report.
    fail_df = pd.DataFrame(all_failures)
    fail_csv = OUTPUT_DIR / "failed_files.csv"
    fail_df.to_csv(fail_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Failure report: {fail_csv} | failed records: {len(fail_df)}")

    make_plots(summary_df, dist_df, OUTPUT_DIR)

    print("=" * 100)
    print("[DONE] Fragment-length sensitivity analysis finished.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Key file:", summary_csv)
    print("=" * 100)


if __name__ == "__main__":
    main()
