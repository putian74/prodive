#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import re
import csv
import json
import glob
import time
import traceback
from multiprocessing import get_context

import numpy as np
from tqdm import tqdm


# ==============================================================================
# 1. Configuration
# ==============================================================================

#  NPZ 
# ：
#   F6p0_baseS8p0/
#   F5p5_baseS8p0/
#   ...
#   F1p0_baseS8p0/
PREPROCESSED_ROOT = "CHANGE_ME"

# ，：
#   PFxxxxx:int_id
MAPPING_FILE = "CHANGE_ME"

# 
OUTPUT_DIR = "CHANGE_ME"

#  first-threshold 
FIRST_THRESHOLDS = [
    6.0, 5.5, 5.0, 4.5, 4.0, 3.5,
    3.0, 2.5, 2.0, 1.5, 1.0
]

#  second-threshold 
SECOND_THRESHOLDS = [
    8.0, 8.5, 9.0, 9.5, 10.0, 10.5,
    11.0, 11.5, 12.0, 12.5, 13.0,
    13.5, 14.0, 14.5, 15.0
]

BASE_SECOND_THRESHOLD = 8.0

NPZ_GLOB_PATTERN = "kl_*_sparse.npz"

# 
WORKERS = 30
CHUNKSIZE = 20

#  long-format CSV：
# combo_key, first_threshold, second_threshold, Pfam
# This output is optional because the JSON summary contains the same sets.
WRITE_LONG_CSV = True

#  ID  JSON
# Store Pfam identifiers in JSON for exact set comparisons.
WRITE_ID_JSON = True




def _parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract covered family sets for first/second threshold combinations from preprocessed NPZ files.")
    parser.add_argument("--preprocessed-root", default=PREPROCESSED_ROOT, help="Root containing F*_baseS* preprocessed NPZ directories.")
    parser.add_argument("--mapping-file", default=MAPPING_FILE, help="Pfam mapping file.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--first-thresholds", default=','.join(map(str, FIRST_THRESHOLDS)), help="Comma-separated first thresholds.")
    parser.add_argument("--second-thresholds", default=','.join(map(str, SECOND_THRESHOLDS)), help="Comma-separated second thresholds.")
    parser.add_argument("--base-second-threshold", type=float, default=BASE_SECOND_THRESHOLD, help="Base second threshold used in directory names.")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Worker process count.")
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE, help="Multiprocessing chunksize.")
    parser.add_argument("--no-long-csv", action="store_true", help="Do not write long-format family membership CSV.")
    parser.add_argument("--no-id-json", action="store_true", help="Do not write per-combination ID JSON files.")
    return parser.parse_args()

def apply_runtime_args(args: argparse.Namespace) -> None:
    global PREPROCESSED_ROOT, MAPPING_FILE, OUTPUT_DIR, FIRST_THRESHOLDS, SECOND_THRESHOLDS, BASE_SECOND_THRESHOLD, WORKERS, CHUNKSIZE, WRITE_LONG_CSV, WRITE_ID_JSON
    PREPROCESSED_ROOT = args.preprocessed_root
    MAPPING_FILE = args.mapping_file
    OUTPUT_DIR = args.output_dir
    FIRST_THRESHOLDS = _parse_float_list(args.first_thresholds)
    SECOND_THRESHOLDS = _parse_float_list(args.second_thresholds)
    BASE_SECOND_THRESHOLD = float(args.base_second_threshold)
    WORKERS = int(args.workers)
    CHUNKSIZE = int(args.chunksize)
    WRITE_LONG_CSV = not bool(args.no_long_csv)
    WRITE_ID_JSON = not bool(args.no_id_json)

# ==============================================================================
# 2. Helper functions
# ==============================================================================

def normalize_pfam_id(x) -> str:
    s = str(x).strip()
    m = re.search(r"PF\d{5}", s)
    if m:
        return m.group(0)
    return s


def tag_float(x: float) -> str:
    return f"{float(x):.1f}".replace(".", "p")


def combo_key(first_threshold: float, second_threshold: float) -> str:
    return f"F{tag_float(first_threshold)}_S{tag_float(second_threshold)}"


def first_dir_name(first_threshold: float) -> str:
    return f"F{tag_float(first_threshold)}_baseS{tag_float(BASE_SECOND_THRESHOLD)}"


def load_pfam_mapping(mapping_file: str):
    """
    Input:
        PFxxxxx:int_id

    Returns:
        id_to_name[int_id] = PFxxxxx
        name_to_id[PFxxxxx] = int_id
    """
    id_to_name = {}
    name_to_id = {}

    with open(mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            pfam_name, idx = line.split(":", 1)
            pfam_name = normalize_pfam_id(pfam_name)
            idx = int(idx.strip())

            id_to_name[idx] = pfam_name
            name_to_id[pfam_name] = idx

    return id_to_name, name_to_id


def id_to_pfam(id_to_name, idx):
    idx = int(idx)
    return id_to_name.get(idx, f"ID_{idx}")


def atomic_write_json(obj, out_path: str):
    tmp = f"{out_path}.tmp.{os.getpid()}.{time.time_ns()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def validate_npz(z, filename: str):
    required = ["i", "meta", "linear_vals"]
    for key in required:
        if key not in z:
            raise KeyError(f"{filename}: missing required field {key}")


# ==============================================================================
# 3. Worker
# ==============================================================================

def init_worker():
    np.seterr(all="ignore")


def worker_scan_one_npz(args):
    """
    Process one NPZ file for one fixed first_threshold directory.

    Input NPZ already satisfies:
        raw_val <= first_threshold
        linear_val >= BASE_SECOND_THRESHOLD

    For each pair block in meta, use max(linear_vals) to determine which
    second_threshold values have at least one surviving point.

    Return:
        family_ids_by_second_index[j] = set of family IDs covered at second_threshold[j]
        kept_points_by_second_index[j] = number of points with linear >= second_threshold[j]
        retained_pairs_by_second_index[j] = number of family-pair blocks with at least one point
    """
    npz_path, second_thresholds = args
    filename = os.path.basename(npz_path)

    n_second = len(second_thresholds)

    family_ids_by_s = [set() for _ in range(n_second)]
    kept_points_by_s = [0] * n_second
    retained_pairs_by_s = [0] * n_second

    try:
        with np.load(npz_path, allow_pickle=False) as z:
            validate_npz(z, filename)

            i_arr = z["i"]
            meta = z["meta"]
            linear_vals = z["linear_vals"]

        if i_arr.size != 1:
            return {
                "ok": False,
                "file": filename,
                "error": "field i length is not 1",
            }

        if meta.ndim != 2 or meta.shape[1] != 5:
            return {
                "ok": False,
                "file": filename,
                "error": f"bad meta shape: {meta.shape}",
            }

        query_id = int(i_arr[0])
        linear_vals = linear_vals.astype(np.float32, copy=False)

        for rec_idx in range(meta.shape[0]):
            target_id, nrows, ncols, offset, nnz = meta[rec_idx]

            target_id = int(target_id)
            offset = int(offset)
            nnz = int(nnz)

            if nnz <= 0:
                continue

            if offset < 0 or offset + nnz > linear_vals.size:
                continue

            vals = linear_vals[offset:offset + nnz]

            if vals.size == 0:
                continue

            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                continue

            # For each second threshold, count points and decide pair/family coverage.
            # This is still efficient because there are only 15 second thresholds.
            for s_idx, s_thr in enumerate(second_thresholds):
                kept_n = int(np.count_nonzero(vals >= s_thr))

                if kept_n <= 0:
                    continue

                kept_points_by_s[s_idx] += kept_n
                retained_pairs_by_s[s_idx] += 1
                family_ids_by_s[s_idx].add(query_id)
                family_ids_by_s[s_idx].add(target_id)

        return {
            "ok": True,
            "file": filename,
            "family_ids_by_s": [list(x) for x in family_ids_by_s],
            "kept_points_by_s": kept_points_by_s,
            "retained_pairs_by_s": retained_pairs_by_s,
        }

    except Exception as e:
        return {
            "ok": False,
            "file": filename,
            "error": str(e) + "\n" + traceback.format_exc(limit=3),
        }


# ==============================================================================
# 4. Scan one first-threshold directory
# ==============================================================================

def scan_one_first_threshold(first_threshold: float):
    input_dir = os.path.join(PREPROCESSED_ROOT, first_dir_name(first_threshold))
    npz_files = sorted(glob.glob(os.path.join(input_dir, NPZ_GLOB_PATTERN)))

    second_thresholds = [float(x) for x in SECOND_THRESHOLDS]
    n_second = len(second_thresholds)

    family_ids_by_s = [set() for _ in range(n_second)]
    kept_points_by_s = [0] * n_second
    retained_pairs_by_s = [0] * n_second

    failed_files = []

    if not npz_files:
        print(f"[WARN] No NPZ files found for first_threshold={first_threshold}: {input_dir}")
        return {
            "first_threshold": first_threshold,
            "family_ids_by_s": family_ids_by_s,
            "kept_points_by_s": kept_points_by_s,
            "retained_pairs_by_s": retained_pairs_by_s,
            "failed_files": [(input_dir, "no npz files found")],
            "npz_files": 0,
            "ok_files": 0,
        }

    print("=" * 100)
    print(f"[INFO] Scanning first_threshold = {first_threshold:g}")
    print(f"[INFO] Input dir: {input_dir}")
    print(f"[INFO] NPZ files: {len(npz_files)}")
    print(f"[INFO] WORKERS: {WORKERS}")
    print("=" * 100)

    tasks = [(p, second_thresholds) for p in npz_files]

    ok_files = 0

    ctx = get_context("fork")

    with ctx.Pool(
        processes=WORKERS,
        initializer=init_worker,
        maxtasksperchild=100,
    ) as pool:

        iterator = pool.imap_unordered(
            worker_scan_one_npz,
            tasks,
            chunksize=CHUNKSIZE,
        )

        for res in tqdm(iterator, total=len(tasks), desc=f"F={first_threshold:g}", unit="file"):
            if not res["ok"]:
                failed_files.append((res["file"], res["error"]))
                continue

            ok_files += 1

            for s_idx in range(n_second):
                family_ids_by_s[s_idx].update(int(x) for x in res["family_ids_by_s"][s_idx])
                kept_points_by_s[s_idx] += int(res["kept_points_by_s"][s_idx])
                retained_pairs_by_s[s_idx] += int(res["retained_pairs_by_s"][s_idx])

    if failed_files:
        failed_path = os.path.join(
            OUTPUT_DIR,
            f"failed_files_F{tag_float(first_threshold)}.txt"
        )
        with open(failed_path, "w", encoding="utf-8") as f:
            for fn, msg in failed_files:
                f.write(f"{fn}\t{msg}\n")

        print(f"[WARN] Failed files for F={first_threshold:g}: {len(failed_files)}")
        print(f"[WARN] Failed list saved to: {failed_path}")

    return {
        "first_threshold": first_threshold,
        "family_ids_by_s": family_ids_by_s,
        "kept_points_by_s": kept_points_by_s,
        "retained_pairs_by_s": retained_pairs_by_s,
        "failed_files": failed_files,
        "npz_files": len(npz_files),
        "ok_files": ok_files,
    }


# ==============================================================================
# 5. Main
# ==============================================================================

def main():
    args = parse_args()
    apply_runtime_args(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 100)
    print("Multiprocessing candidate-level family-set extraction")
    print("Full grid mode: all FIRST_THRESHOLDS × SECOND_THRESHOLDS are scanned.")
    print("No Pfam family is excluded.")
    print("No hidden subset or sampling is used.")
    print(f"PREPROCESSED_ROOT: {PREPROCESSED_ROOT}")
    print(f"MAPPING_FILE: {MAPPING_FILE}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"FIRST_THRESHOLDS: {FIRST_THRESHOLDS}")
    print(f"SECOND_THRESHOLDS: {SECOND_THRESHOLDS}")
    print(f"WORKERS: {WORKERS}")
    print("=" * 100)

    id_to_name, name_to_id = load_pfam_mapping(MAPPING_FILE)
    print(f"[INFO] Mapping entries: {len(id_to_name)}")

    # combo -> set(int family IDs)
    family_ids_by_combo = {
        combo_key(f, s): set()
        for f in FIRST_THRESHOLDS
        for s in SECOND_THRESHOLDS
    }

    # additive stats
    kept_points_by_combo = {
        combo_key(f, s): 0
        for f in FIRST_THRESHOLDS
        for s in SECOND_THRESHOLDS
    }

    retained_pairs_by_combo = {
        combo_key(f, s): 0
        for f in FIRST_THRESHOLDS
        for s in SECOND_THRESHOLDS
    }

    run_file_stats = []

    second_thresholds = [float(x) for x in SECOND_THRESHOLDS]

    for f_thr in FIRST_THRESHOLDS:
        res = scan_one_first_threshold(float(f_thr))

        run_file_stats.append({
            "first_threshold": float(f_thr),
            "npz_files": int(res["npz_files"]),
            "ok_files": int(res["ok_files"]),
            "failed_files": int(len(res["failed_files"])),
        })

        for s_idx, s_thr in enumerate(second_thresholds):
            key = combo_key(float(f_thr), float(s_thr))

            family_ids_by_combo[key].update(res["family_ids_by_s"][s_idx])
            kept_points_by_combo[key] += int(res["kept_points_by_s"][s_idx])
            retained_pairs_by_combo[key] += int(res["retained_pairs_by_s"][s_idx])

    # --------------------------------------------------------------------------
    # Save local family-set JSON with Pfam names
    # --------------------------------------------------------------------------
    family_names_by_combo = {}

    for key, ids in family_ids_by_combo.items():
        family_names_by_combo[key] = sorted(
            id_to_pfam(id_to_name, x)
            for x in ids
        )

    json_path = os.path.join(OUTPUT_DIR, "candidate_family_sets_local.json")
    atomic_write_json(family_names_by_combo, json_path)

    # --------------------------------------------------------------------------
    # Optional: save ID JSON
    # --------------------------------------------------------------------------
    if WRITE_ID_JSON:
        id_json_obj = {
            key: sorted(int(x) for x in ids)
            for key, ids in family_ids_by_combo.items()
        }

        id_json_path = os.path.join(OUTPUT_DIR, "candidate_family_id_sets_local.json")
        atomic_write_json(id_json_obj, id_json_path)
        print(f"[DONE] Local family ID JSON saved to: {id_json_path}")

    # --------------------------------------------------------------------------
    # Save local summary CSV
    # --------------------------------------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, "candidate_family_summary_local.csv")

    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "combo_key",
            "first_threshold",
            "second_threshold",
            "kept_candidate_points_local",
            "retained_family_pairs_with_points_local",
            "covered_families_local",
            "log10_kept_candidate_points_local",
        ])

        for f_thr in FIRST_THRESHOLDS:
            for s_thr in SECOND_THRESHOLDS:
                key = combo_key(float(f_thr), float(s_thr))
                kept_points = int(kept_points_by_combo[key])
                retained_pairs = int(retained_pairs_by_combo[key])
                covered_n = int(len(family_ids_by_combo[key]))

                log10_points = float(np.log10(kept_points)) if kept_points > 0 else 0.0

                writer.writerow([
                    key,
                    float(f_thr),
                    float(s_thr),
                    kept_points,
                    retained_pairs,
                    covered_n,
                    log10_points,
                ])

    # --------------------------------------------------------------------------
    # Save long-format family CSV
    # --------------------------------------------------------------------------
    if WRITE_LONG_CSV:
        long_path = os.path.join(OUTPUT_DIR, "candidate_families_long_local.csv")

        with open(long_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "combo_key",
                "first_threshold",
                "second_threshold",
                "Pfam",
            ])

            for f_thr in FIRST_THRESHOLDS:
                for s_thr in SECOND_THRESHOLDS:
                    key = combo_key(float(f_thr), float(s_thr))
                    for pfam in family_names_by_combo[key]:
                        writer.writerow([
                            key,
                            float(f_thr),
                            float(s_thr),
                            pfam,
                        ])

        print(f"[DONE] Long local family CSV saved to: {long_path}")

    # --------------------------------------------------------------------------
    # Save run file stats
    # --------------------------------------------------------------------------
    run_stats_path = os.path.join(OUTPUT_DIR, "local_file_processing_stats.csv")

    with open(run_stats_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "first_threshold",
                "npz_files",
                "ok_files",
                "failed_files",
            ],
        )
        writer.writeheader()
        writer.writerows(run_file_stats)

    # --------------------------------------------------------------------------
    # Console report
    # --------------------------------------------------------------------------
    dt = time.time() - t0

    print("=" * 100)
    print("[DONE] Multiprocessing local family-set extraction finished.")
    print(f"Family set JSON: {json_path}")
    print(f"Local summary CSV: {summary_path}")
    print(f"Run file stats CSV: {run_stats_path}")
    print(f"Total time: {dt:.1f}s")
    print("=" * 100)

    print("\nTop local family coverage:")
    rows = []

    for f_thr in FIRST_THRESHOLDS:
        for s_thr in SECOND_THRESHOLDS:
            key = combo_key(float(f_thr), float(s_thr))
            rows.append({
                "first_threshold": float(f_thr),
                "second_threshold": float(s_thr),
                "covered_families_local": len(family_ids_by_combo[key]),
                "kept_candidate_points_local": kept_points_by_combo[key],
                "retained_family_pairs_with_points_local": retained_pairs_by_combo[key],
            })

    rows = sorted(
        rows,
        key=lambda x: (
            -x["covered_families_local"],
            x["kept_candidate_points_local"],
            -x["first_threshold"],
            x["second_threshold"],
        ),
    )

    for r in rows[:30]:
        print(
            f"F={r['first_threshold']:g}, "
            f"S={r['second_threshold']:g}, "
            f"families={r['covered_families_local']}, "
            f"points={r['kept_candidate_points_local']:,}, "
            f"pairs={r['retained_family_pairs_with_points_local']:,}"
        )


if __name__ == "__main__":
    main()
