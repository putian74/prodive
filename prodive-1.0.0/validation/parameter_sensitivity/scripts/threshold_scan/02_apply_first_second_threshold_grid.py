#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import sys
import gc
import json
import time
import traceback
from multiprocessing import get_context

import numpy as np
import pandas as pd
from tqdm import tqdm


# ==============================================================================
# 1) Configuration
# ==============================================================================

# ： le6  sparse npz
# ：kl_PF00001_le6_sparse.npz
INPUT_DIR = None

# 
OUTPUT_ROOT = None

# 
MAPPING_FILE = None
JSON_BG_FILE = None



# 
FRAGMENT = 6
CONST_FACTOR = 3 * FRAGMENT + 2
EPSILON = 1e-4

# ：raw_val <= first_threshold
FIRST_THRESHOLDS = [
    6.0, 5.5, 5.0, 4.5, 4.0, 3.5,
    3.0, 2.5, 2.0, 1.5, 1.0
]

# ：linear_val >= second_threshold
# SECOND_THRESHOLDS are used for candidate-level summaries.
# Stored NPZ files use BASE_SECOND_THRESHOLD as their common lower bound.
SECOND_THRESHOLDS = [
    8.0, 8.5, 9.0, 9.5, 10.0, 10.5,
    11.0, 11.5, 12.0, 12.5, 13.0,
    13.5, 14.0, 14.5, 15.0
]

#  NPZ 
#  path-building  NPZ  linear_vals >= 8.0, 8.5, ..., 15.0
BASE_SECOND_THRESHOLD = 8.0

# 
SAVE_FLOAT32 = True

# 
WORKERS = 5
CHUNKSIZE = 5

# 
NPZ_REGEX = re.compile(r"^kl_(PF\d{5})_le(\d+)_sparse\.npz$")




def _parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply first/second threshold preprocessing to sparse raw NPZ files and write candidate-level grid summaries.")
    parser.add_argument("--input-dir", required=True, help="Input sparse NPZ directory.")
    parser.add_argument("--output-root", required=True, help="Output root for first-threshold directories and summaries.")
    parser.add_argument("--mapping-file", required=True, help="Pfam mapping file.")
    parser.add_argument("--background-json", required=True, help="Background KL JSON file.")
    parser.add_argument("--fragment", type=int, default=FRAGMENT, help="Fragment length.")
    parser.add_argument("--first-thresholds", default=','.join(map(str, FIRST_THRESHOLDS)), help="Comma-separated first thresholds.")
    parser.add_argument("--second-thresholds", default=','.join(map(str, SECOND_THRESHOLDS)), help="Comma-separated second thresholds.")
    parser.add_argument("--base-second-threshold", type=float, default=BASE_SECOND_THRESHOLD, help="Base second threshold stored in saved NPZ files.")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Worker process count.")
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE, help="Multiprocessing chunksize.")
    return parser.parse_args()

def apply_runtime_args(args: argparse.Namespace) -> None:
    global INPUT_DIR, OUTPUT_ROOT, MAPPING_FILE, JSON_BG_FILE, FRAGMENT, CONST_FACTOR, FIRST_THRESHOLDS, SECOND_THRESHOLDS, BASE_SECOND_THRESHOLD, WORKERS, CHUNKSIZE
    INPUT_DIR = args.input_dir
    OUTPUT_ROOT = args.output_root
    MAPPING_FILE = args.mapping_file
    JSON_BG_FILE = args.background_json
    FRAGMENT = int(args.fragment)
    CONST_FACTOR = 3 * FRAGMENT + 2
    FIRST_THRESHOLDS = _parse_float_list(args.first_thresholds)
    SECOND_THRESHOLDS = _parse_float_list(args.second_thresholds)
    BASE_SECOND_THRESHOLD = float(args.base_second_threshold)
    WORKERS = int(args.workers)
    CHUNKSIZE = int(args.chunksize)

# ==============================================================================
# 2) Global variables for fork workers
# ==============================================================================

worker_pfam_map = None
worker_bg = None
worker_first_thresholds = None
worker_second_thresholds = None
worker_output_dirs_by_first = None


# ==============================================================================
# 3) Helper functions
# ==============================================================================

def normalize_pfam_id(x) -> str:
    s = str(x).strip()
    m = re.search(r"PF\d{5}", s)
    if m:
        return m.group(0)
    return s


def tag_float(x: float) -> str:
    """
    6.0 -> 6p0
    5.5 -> 5p5
    """
    return f"{float(x):.1f}".replace(".", "p")


def combo_key(first_thr: float, second_thr: float) -> str:
    return f"F{tag_float(first_thr)}_S{tag_float(second_thr)}"


def first_dir_name(first_thr: float) -> str:
    return f"F{tag_float(first_thr)}_baseS{tag_float(BASE_SECOND_THRESHOLD)}"


def atomic_write_json(obj, out_path: str):
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def atomic_write_npz(out_path: str, **arrays):
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, out_path)


def load_pfam_map(file_path: str):
    """
    Mapping format:
      PFxxxxx:int_id

    Returns:
      id_to_name[int_id] = PFxxxxx
      name_to_id[PFxxxxx] = int_id
    """
    id_to_name = {}
    name_to_id = {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":", 1)
                if len(parts) != 2:
                    continue

                name = normalize_pfam_id(parts[0])
                idx = int(parts[1].strip())

                id_to_name[idx] = name
                name_to_id[name] = idx

    except FileNotFoundError:
        print(f"❌ : {file_path}")
        sys.exit(1)

    return id_to_name, name_to_id


def load_background(json_bg_file: str):
    """
     JSON  fragment=6  K(i)
    """
    try:
        with open(json_bg_file, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError:
        print(f"❌ : {json_bg_file}")
        sys.exit(1)

    bg = {}
    for k, v in obj.items():
        bg[normalize_pfam_id(k)] = np.asarray(v, dtype=np.float32)

    return bg


def parse_query_info_from_filename(filename: str):
    m = NPZ_REGEX.match(filename)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def validate_npz_payload(z, filename: str):
    required = ["i", "threshold", "meta", "rows", "cols", "vals"]
    for key in required:
        if key not in z:
            raise KeyError(f"{filename}:  {key}")


def init_worker():
    np.seterr(all="ignore")


# ==============================================================================
# 4) Worker: process one raw sparse NPZ
# ==============================================================================

def worker_process_one_npz(npz_path: str):
    """
     le6 sparse npz：

    1.  raw sparse points:
       rows, cols, vals(raw distance/divergence)

    2.  linear score:
       linear = (q_bg[row] + t_bg[col]) / (raw_val + EPSILON) * CONST_FACTOR

    3.  first_threshold  NPZ：
       raw_val <= first_threshold
       linear_val >= BASE_SECOND_THRESHOLD

    4.  first_threshold × second_threshold  candidate-level ：
       raw_val <= first_threshold
       linear_val >= second_threshold

    ：
        Coverage is counted over Pfam families.
        covered_families is a candidate-level statistic, not a path-level one.
    """
    global worker_pfam_map, worker_bg
    global worker_first_thresholds, worker_second_thresholds
    global worker_output_dirs_by_first

    try:
        if worker_pfam_map is None or worker_bg is None:
            return {
                "ok": False,
                "file": os.path.basename(npz_path),
                "error": "worker globals not initialized",
            }

        filename = os.path.basename(npz_path)

        query_name_from_fn, input_first_thr_from_fn = parse_query_info_from_filename(filename)
        if query_name_from_fn is None:
            return {
                "ok": False,
                "file": filename,
                "error": "filename pattern mismatch",
            }

        with np.load(npz_path, allow_pickle=False) as z:
            validate_npz_payload(z, filename)

            i_arr = z["i"]
            first_threshold_arr = z["threshold"]
            meta = z["meta"]
            rows = z["rows"]
            cols = z["cols"]
            vals = z["vals"]

        if i_arr.size != 1:
            raise ValueError(f"{filename}: i  1")

        if first_threshold_arr.size != 1:
            raise ValueError(f"{filename}: threshold  1")

        if meta.ndim != 2 or meta.shape[1] != 5:
            raise ValueError(f"{filename}: meta ， [N,5]， {meta.shape}")

        query_id = int(i_arr[0])
        query_name = worker_pfam_map.get(query_id)

        if query_name is None:
            raise KeyError(f"{filename}: query_id={query_id} ")

        if query_name != query_name_from_fn:
            raise ValueError(
                f"{filename}:  query={query_name_from_fn}  i  query={query_name} "
            )

        if query_name not in worker_bg:
            raise KeyError(f"{filename}:  query {query_name}")

        q_bg = worker_bg[query_name]

        rows = rows.astype(np.int64, copy=False)
        cols = cols.astype(np.int64, copy=False)
        raw_vals_all = vals.astype(np.float64, copy=False)

        total_points = int(raw_vals_all.size)

        if rows.size != total_points or cols.size != total_points:
            raise ValueError(
                f"{filename}: rows/cols/vals  | "
                f"rows={rows.size}, cols={cols.size}, vals={total_points}"
            )

        dtype_linear = np.float32 if SAVE_FLOAT32 else np.float64
        linear_vals_all = np.full(total_points, np.nan, dtype=dtype_linear)

        input_pair_count = 0
        valid_input_sparse_points = 0

        # local combo stats for this query file
        local_combo_stats = {}

        for f_thr in worker_first_thresholds:
            for s_thr in worker_second_thresholds:
                key = combo_key(f_thr, s_thr)
                local_combo_stats[key] = {
                    "kept_candidate_points": 0,
                    "retained_family_pairs_with_points": 0,
                    "covered_ids": set(),
                }

        # --------------------------------------------------
        # Calculate linear score and candidate-level summary
        # --------------------------------------------------
        for rec_idx in range(meta.shape[0]):
            j, nrows, ncols, offset, nnz = meta[rec_idx]

            j = int(j)
            nrows = int(nrows)
            ncols = int(ncols)
            offset = int(offset)
            nnz = int(nnz)

            if nnz <= 0:
                continue

            if offset < 0 or offset + nnz > total_points:
                raise ValueError(
                    f"{filename}: meta[{rec_idx}]  | "
                    f"offset={offset}, nnz={nnz}, total={total_points}"
                )

            target_name = worker_pfam_map.get(j)
            if target_name is None:
                raise KeyError(f"{filename}: target id={j} ")

            if target_name not in worker_bg:
                raise KeyError(f"{filename}:  target {target_name}")

            t_bg = worker_bg[target_name]

            if len(q_bg) != nrows:
                raise ValueError(
                    f"{filename}: query  nrows  | "
                    f"query={query_name}, len(q_bg)={len(q_bg)}, nrows={nrows}"
                )

            if len(t_bg) != ncols:
                raise ValueError(
                    f"{filename}: target  ncols  | "
                    f"target={target_name}, len(t_bg)={len(t_bg)}, ncols={ncols}"
                )

            sl = slice(offset, offset + nnz)

            rr = rows[sl]
            cc = cols[sl]
            vv = raw_vals_all[sl]

            if rr.size == 0:
                continue

            rmax = int(rr.max())
            cmax = int(cc.max())

            if rmax >= nrows:
                raise IndexError(
                    f"{filename}: meta[{rec_idx}] rows  nrows | rmax={rmax}, nrows={nrows}"
                )

            if cmax >= ncols:
                raise IndexError(
                    f"{filename}: meta[{rec_idx}] cols  ncols | cmax={cmax}, ncols={ncols}"
                )

            qv = np.asarray(q_bg[rr], dtype=np.float64)
            tv = np.asarray(t_bg[cc], dtype=np.float64)

            linear = (qv + tv) / (vv + EPSILON) * CONST_FACTOR

            finite = np.isfinite(vv) & np.isfinite(linear)
            linear_vals_all[sl] = linear.astype(dtype_linear, copy=False)

            if not np.any(finite):
                continue

            vv_finite = vv[finite]
            linear_finite = linear[finite]

            valid_input_sparse_points += int(vv_finite.size)
            input_pair_count += 1

            for f_thr in worker_first_thresholds:
                raw_keep = vv_finite <= f_thr

                if not np.any(raw_keep):
                    continue

                linear_sub = linear_finite[raw_keep]

                for s_thr in worker_second_thresholds:
                    kept_n = int(np.count_nonzero(linear_sub >= s_thr))

                    if kept_n <= 0:
                        continue

                    key = combo_key(f_thr, s_thr)
                    local_combo_stats[key]["kept_candidate_points"] += kept_n
                    local_combo_stats[key]["retained_family_pairs_with_points"] += 1
                    local_combo_stats[key]["covered_ids"].add(query_id)
                    local_combo_stats[key]["covered_ids"].add(j)

        # --------------------------------------------------
        # Save one NPZ directory for each first_threshold
        # Base second filter is fixed at linear >= BASE_SECOND_THRESHOLD
        # --------------------------------------------------
        saved_by_first = {}

        for f_thr in worker_first_thresholds:
            out_rows_list = []
            out_cols_list = []
            out_raw_vals_list = []
            out_linear_vals_list = []
            out_meta = []

            running_offset = 0
            kept_count_this_first = 0
            kept_pair_count_this_first = 0
            covered_ids_this_first = set()

            for rec_idx in range(meta.shape[0]):
                j, nrows, ncols, offset, nnz = meta[rec_idx]

                j = int(j)
                nrows = int(nrows)
                ncols = int(ncols)
                offset = int(offset)
                nnz = int(nnz)

                if nnz <= 0:
                    out_meta.append([j, nrows, ncols, running_offset, 0])
                    continue

                sl = slice(offset, offset + nnz)

                rr = rows[sl]
                cc = cols[sl]
                vv = raw_vals_all[sl]
                linear = linear_vals_all[sl].astype(np.float64, copy=False)

                keep = (
                    np.isfinite(vv)
                    & np.isfinite(linear)
                    & (vv <= f_thr)
                    & (linear >= BASE_SECOND_THRESHOLD)
                )

                kept_nnz = int(np.count_nonzero(keep))
                out_meta.append([j, nrows, ncols, running_offset, kept_nnz])

                if kept_nnz > 0:
                    kept_pair_count_this_first += 1
                    kept_count_this_first += kept_nnz
                    covered_ids_this_first.add(query_id)
                    covered_ids_this_first.add(j)

                    kept_rows = rr[keep].astype(np.int32, copy=False)
                    kept_cols = cc[keep].astype(np.int32, copy=False)
                    kept_raw = vv[keep]
                    kept_linear = linear[keep]

                    if SAVE_FLOAT32:
                        kept_raw = kept_raw.astype(np.float32, copy=False)
                        kept_linear = kept_linear.astype(np.float32, copy=False)

                    out_rows_list.append(kept_rows)
                    out_cols_list.append(kept_cols)
                    out_raw_vals_list.append(kept_raw)
                    out_linear_vals_list.append(kept_linear)

                    running_offset += kept_nnz

            if running_offset > 0:
                out_rows = np.concatenate(out_rows_list)
                out_cols = np.concatenate(out_cols_list)
                out_raw_vals = np.concatenate(out_raw_vals_list)
                out_linear_vals = np.concatenate(out_linear_vals_list)
            else:
                dtype_float = np.float32 if SAVE_FLOAT32 else np.float64
                out_rows = np.empty((0,), dtype=np.int32)
                out_cols = np.empty((0,), dtype=np.int32)
                out_raw_vals = np.empty((0,), dtype=dtype_float)
                out_linear_vals = np.empty((0,), dtype=dtype_float)

            out_meta_arr = np.asarray(out_meta, dtype=np.int64)

            out_dir = worker_output_dirs_by_first[f_thr]

            out_npz_name = filename.replace(
                "_sparse.npz",
                f"_F{tag_float(f_thr)}_baseS{tag_float(BASE_SECOND_THRESHOLD)}_sparse.npz"
            )

            out_npz_path = os.path.join(out_dir, out_npz_name)

            atomic_write_npz(
                out_npz_path,
                i=np.array([query_id], dtype=np.int32),
                input_first_threshold=first_threshold_arr.astype(np.float32, copy=False),
                first_threshold=np.array([f_thr], dtype=np.float32),
                base_second_threshold=np.array([BASE_SECOND_THRESHOLD], dtype=np.float32),
                meta=out_meta_arr,
                rows=out_rows,
                cols=out_cols,
                raw_vals=out_raw_vals,
                linear_vals=out_linear_vals,
            )

            saved_by_first[tag_float(f_thr)] = {
                "first_threshold": float(f_thr),
                "saved_points": int(kept_count_this_first),
                "saved_family_pairs_with_points": int(kept_pair_count_this_first),
                "covered_ids": list(covered_ids_this_first),
                "output_file": out_npz_name,
            }

            del out_rows_list, out_cols_list, out_raw_vals_list, out_linear_vals_list
            del out_rows, out_cols, out_raw_vals, out_linear_vals
            gc.collect()

        # Convert local sets to lists for multiprocessing return
        combo_stats_return = {}

        for key, d in local_combo_stats.items():
            if d["kept_candidate_points"] <= 0:
                continue

            combo_stats_return[key] = {
                "kept_candidate_points": int(d["kept_candidate_points"]),
                "retained_family_pairs_with_points": int(d["retained_family_pairs_with_points"]),
                "covered_ids": list(d["covered_ids"]),
            }

        return {
            "ok": True,
            "file": filename,
            "query_id": int(query_id),
            "query_name": query_name,
            "valid_input_sparse_points": int(valid_input_sparse_points),
            "input_family_pairs_with_points": int(input_pair_count),
            "saved_by_first": saved_by_first,
            "combo_stats": combo_stats_return,
        }

    except Exception as e:
        return {
            "ok": False,
            "file": os.path.basename(npz_path),
            "error": str(e),
            "traceback": traceback.format_exc(limit=3),
        }


# ==============================================================================
# 5) Main
# ==============================================================================

def main():
    args = parse_args()
    apply_runtime_args(args)
    print("=" * 100)
    print("First-threshold NPZ generation + First/Second candidate-level grid summary")
    print("This script saves NPZ only by first_threshold.")
    print("Each saved NPZ keeps: raw_val <= first_threshold AND linear_val >= BASE_SECOND_THRESHOLD.")
    print("Second-threshold grid is used for summary only; path code should dynamically filter linear_vals.")
    print("No Pfam family is excluded in this preprocessing step.")
    print("=" * 100)

    print(f"INPUT_DIR: {INPUT_DIR}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"MAPPING_FILE: {MAPPING_FILE}")
    print(f"JSON_BG_FILE: {JSON_BG_FILE}")
    print(f"FIRST_THRESHOLDS: {FIRST_THRESHOLDS}")
    print(f"SECOND_THRESHOLDS: {SECOND_THRESHOLDS}")
    print(f"BASE_SECOND_THRESHOLD for saved NPZ: {BASE_SECOND_THRESHOLD}")
    print(f"WORKERS: {WORKERS}")
    print("=" * 100)

    t0 = time.time()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # --------------------------------------------------------------------------
    # Load mapping and background
    # --------------------------------------------------------------------------
    id_to_name, name_to_id = load_pfam_map(MAPPING_FILE)
    bg_dict = load_background(JSON_BG_FILE)

    print(f"Mapping entries: {len(id_to_name)}")
    print(f"Background families: {len(bg_dict)}")

    # --------------------------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------------------------
    output_dirs_by_first = {}

    for f_thr in FIRST_THRESHOLDS:
        d = os.path.join(OUTPUT_ROOT, first_dir_name(f_thr))
        os.makedirs(d, exist_ok=True)
        output_dirs_by_first[float(f_thr)] = d

    manifest = {
        "input_dir": INPUT_DIR,
        "output_root": OUTPUT_ROOT,
        "first_thresholds": FIRST_THRESHOLDS,
        "second_thresholds_for_later_path_scan": SECOND_THRESHOLDS,
        "base_second_threshold_for_saved_npz": BASE_SECOND_THRESHOLD,
        "formula": "linear = (q_bg[row] + t_bg[col]) / (raw_val + EPSILON) * CONST_FACTOR",
        "fragment": FRAGMENT,
        "const_factor": CONST_FACTOR,
        "epsilon": EPSILON,
        "note": (
            "Saved NPZ directories vary only by first_threshold. "
            "Each saved NPZ keeps raw_val <= first_threshold and "
            "linear_val >= base_second_threshold. "
            "Second thresholds above base should be applied dynamically during path construction. "
            "No Pfam family is excluded in this preprocessing step."
        ),
    }

    manifest_path = os.path.join(OUTPUT_ROOT, "parameter_scan_manifest.json")
    atomic_write_json(manifest, manifest_path)

    # --------------------------------------------------------------------------
    # Check input files
    # --------------------------------------------------------------------------
    files = sorted(
        os.path.join(INPUT_DIR, fn)
        for fn in os.listdir(INPUT_DIR)
        if fn.startswith("kl_") and fn.endswith("_sparse.npz")
    )

    if not files:
        raise RuntimeError(f"No sparse npz files found in {INPUT_DIR}")

    print(f"Input sparse npz files: {len(files)}")

    # Check feasibility of first thresholds
    max_input_first_threshold = None
    for fn in files[:50]:
        _, le_thr = parse_query_info_from_filename(os.path.basename(fn))
        if le_thr is not None:
            max_input_first_threshold = float(le_thr)
            break

    if max_input_first_threshold is not None:
        too_large = [x for x in FIRST_THRESHOLDS if x > max_input_first_threshold]
        if too_large:
            print(
                f"[WARN] Some first thresholds are larger than input le{max_input_first_threshold}: {too_large}"
            )
            print("[WARN] These cannot recover points absent from input NPZ.")

    # --------------------------------------------------------------------------
    # Prepare global accumulators
    # --------------------------------------------------------------------------
    combo_records = {}

    for f_thr in FIRST_THRESHOLDS:
        for s_thr in SECOND_THRESHOLDS:
            key = combo_key(f_thr, s_thr)
            combo_records[key] = {
                "first_threshold": float(f_thr),
                "second_threshold": float(s_thr),
                "kept_candidate_points": 0,
                "retained_family_pairs_with_points": 0,
                "covered_ids": set(),
            }

    first_save_records = {}

    for f_thr in FIRST_THRESHOLDS:
        first_save_records[tag_float(f_thr)] = {
            "first_threshold": float(f_thr),
            "base_second_threshold": float(BASE_SECOND_THRESHOLD),
            "saved_points": 0,
            "saved_family_pairs_with_points": 0,
            "covered_ids": set(),
        }

    # --------------------------------------------------------------------------
    # Set worker globals
    # --------------------------------------------------------------------------
    global worker_pfam_map, worker_bg
    global worker_first_thresholds, worker_second_thresholds
    global worker_output_dirs_by_first

    worker_pfam_map = id_to_name
    worker_bg = bg_dict
    worker_first_thresholds = [float(x) for x in FIRST_THRESHOLDS]
    worker_second_thresholds = [float(x) for x in SECOND_THRESHOLDS]
    worker_output_dirs_by_first = output_dirs_by_first

    # --------------------------------------------------------------------------
    # Multiprocessing
    # --------------------------------------------------------------------------
    ok_count = 0
    fail_count = 0
    fail_list = []

    total_valid_input_sparse_points = 0
    total_input_family_pairs_with_points = 0

    ctx = get_context("fork")
    pool = ctx.Pool(
        processes=WORKERS,
        initializer=init_worker,
        maxtasksperchild=100
    )

    try:
        iterator = pool.imap_unordered(
            worker_process_one_npz,
            files,
            chunksize=CHUNKSIZE
        )

        for res in tqdm(iterator, total=len(files), desc="Processing sparse npz", unit="file"):
            if not res["ok"]:
                fail_count += 1
                fail_list.append((res["file"], res["error"], res.get("traceback", "")))
                continue

            ok_count += 1

            total_valid_input_sparse_points += int(res["valid_input_sparse_points"])
            total_input_family_pairs_with_points += int(res["input_family_pairs_with_points"])

            # Merge first-threshold saved summaries
            for f_tag, d in res["saved_by_first"].items():
                first_save_records[f_tag]["saved_points"] += int(d["saved_points"])
                first_save_records[f_tag]["saved_family_pairs_with_points"] += int(
                    d["saved_family_pairs_with_points"]
                )
                first_save_records[f_tag]["covered_ids"].update(d["covered_ids"])

            # Merge combo candidate-level summaries
            for key, d in res["combo_stats"].items():
                combo_records[key]["kept_candidate_points"] += int(d["kept_candidate_points"])
                combo_records[key]["retained_family_pairs_with_points"] += int(
                    d["retained_family_pairs_with_points"]
                )
                combo_records[key]["covered_ids"].update(d["covered_ids"])

            del res
            gc.collect()

    finally:
        pool.close()
        pool.join()

    # --------------------------------------------------------------------------
    # Save first-threshold saved NPZ summary
    # --------------------------------------------------------------------------
    first_rows = []

    for f_tag, d in first_save_records.items():
        first_thr = d["first_threshold"]
        covered_families = len(d["covered_ids"])

        first_rows.append({
            "first_threshold": first_thr,
            "base_second_threshold": d["base_second_threshold"],
            "saved_points": int(d["saved_points"]),
            "saved_family_pairs_with_points": int(d["saved_family_pairs_with_points"]),
            "covered_families_candidate_level": int(covered_families),
            "output_dir": output_dirs_by_first[float(first_thr)],
            "log10_saved_points": (
                float(np.log10(d["saved_points"]))
                if d["saved_points"] > 0 else 0.0
            ),
            "log10_saved_family_pairs": (
                float(np.log10(d["saved_family_pairs_with_points"]))
                if d["saved_family_pairs_with_points"] > 0 else 0.0
            ),
        })

    first_summary_df = pd.DataFrame(first_rows)
    first_summary_df = first_summary_df.sort_values(
        by="first_threshold",
        ascending=False
    ).reset_index(drop=True)

    first_summary_csv = os.path.join(OUTPUT_ROOT, "first_threshold_saved_npz_summary.csv")
    first_summary_df.to_csv(first_summary_csv, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------------------------
    # Save candidate-level F × S grid summary
    # --------------------------------------------------------------------------
    combo_rows = []

    for key, d in combo_records.items():
        covered_ids = set(d["covered_ids"])
        coverage_n = len(covered_ids)

        combo_rows.append({
            "first_threshold": d["first_threshold"],
            "second_threshold": d["second_threshold"],
            "kept_candidate_points": int(d["kept_candidate_points"]),
            "retained_family_pairs_with_points": int(d["retained_family_pairs_with_points"]),
            "covered_families_candidate_level": int(coverage_n),
            "log10_kept_candidate_points": (
                float(np.log10(d["kept_candidate_points"]))
                if d["kept_candidate_points"] > 0 else 0.0
            ),
            "log10_retained_family_pairs": (
                float(np.log10(d["retained_family_pairs_with_points"]))
                if d["retained_family_pairs_with_points"] > 0 else 0.0
            ),
        })

    combo_summary_df = pd.DataFrame(combo_rows)

    combo_summary_df = combo_summary_df.sort_values(
        by=[
            "covered_families_candidate_level",
            "kept_candidate_points",
            "first_threshold",
            "second_threshold",
        ],
        ascending=[False, True, False, True]
    ).reset_index(drop=True)

    combo_summary_csv = os.path.join(
        OUTPUT_ROOT,
        "first_second_candidate_level_grid_summary.csv"
    )
    combo_summary_df.to_csv(combo_summary_csv, index=False, encoding="utf-8-sig")

    combo_summary_xlsx = os.path.join(
        OUTPUT_ROOT,
        "first_second_candidate_level_grid_summary.xlsx"
    )

    try:
        combo_summary_df.to_excel(combo_summary_xlsx, index=False)
    except Exception as e:
        print(f"[WARN] Could not write xlsx: {e}")

    # --------------------------------------------------------------------------
    # Save failed files
    # --------------------------------------------------------------------------
    if fail_list:
        fail_path = os.path.join(OUTPUT_ROOT, "failed_files.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            for fn, msg, tb in fail_list:
                f.write(f"{fn}\t{msg}\n")
                if tb:
                    f.write(tb + "\n")

    # --------------------------------------------------------------------------
    # Console report
    # --------------------------------------------------------------------------
    dt = time.time() - t0

    print("\n" + "=" * 100)
    print("Finished first-threshold NPZ generation and candidate-level grid summary.")
    print(f"OK files: {ok_count}")
    print(f"Failed files: {fail_count}")
    print(f"Total valid input sparse points: {total_valid_input_sparse_points}")
    print(f"Total input family pairs with points: {total_input_family_pairs_with_points}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Manifest: {manifest_path}")
    print(f"First-threshold saved NPZ summary: {first_summary_csv}")
    print(f"Candidate-level F/S grid summary: {combo_summary_csv}")
    print(f"Candidate-level F/S grid xlsx: {combo_summary_xlsx}")
    print(f"Total time: {dt:.1f}s")
    print("=" * 100)

    print("\nSaved NPZ summary by first threshold:")
    first_display_cols = [
        "first_threshold",
        "base_second_threshold",
        "saved_points",
        "saved_family_pairs_with_points",
        "covered_families_candidate_level",
        "output_dir",
    ]
    print(first_summary_df[first_display_cols].to_string(index=False))

    print("\nTop 30 candidate-level parameter combinations:")
    display_cols = [
        "first_threshold",
        "second_threshold",
        "covered_families_candidate_level",
        "kept_candidate_points",
        "retained_family_pairs_with_points",
        "log10_kept_candidate_points",
    ]
    print(combo_summary_df[display_cols].head(30).to_string(index=False))

    print("\nSmallest candidate sets among combinations with nonzero coverage:")
    nonzero_df = combo_summary_df[combo_summary_df["kept_candidate_points"] > 0].copy()
    if nonzero_df.empty:
        print("[INFO] No nonzero candidate-level combinations.")
    else:
        nonzero_df = nonzero_df.sort_values(
            by=["kept_candidate_points", "covered_families_candidate_level"],
            ascending=[True, False]
        ).reset_index(drop=True)
        print(nonzero_df[display_cols].head(30).to_string(index=False))

    if fail_list:
        print("\nFailed files, first 20:")
        for fn, msg, _ in fail_list[:20]:
            print(f"  - {fn}: {msg}")

    print("=" * 100)


if __name__ == "__main__":
    main()
