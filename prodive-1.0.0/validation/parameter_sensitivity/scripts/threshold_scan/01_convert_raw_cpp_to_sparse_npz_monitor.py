import os
import sys
import argparse
import time
import json
import gc
import numpy as np
from multiprocessing import get_context
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# 1. 
# ==========================================

FOLDER = None
TARGET_DIR = None
MAPPING_FILE = None

MAX_I = 25544
SCORE_THRESHOLD = 2
HIST_BIN_WIDTH = 0.01
ROW_CHUNK = 2048

WORKERS = 12
POLL_INTERVAL = 1.0

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor C++ raw output files and convert completed i-blocks into sparse NPZ files with zero statistics.")
    parser.add_argument("--folder", required=True, help="Directory containing raw C++ .npy matrices.")
    parser.add_argument("--target-dir", required=True, help="Output directory for sparse NPZ files and stats JSON files.")
    parser.add_argument("--mapping-file", required=True, help="Pfam mapping file, format PFxxxxx:int_id.")
    parser.add_argument("--max-i", type=int, default=MAX_I, help="Maximum family index used to identify complete i-blocks.")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD, help="Keep 0 < score <= threshold as main sparse points.")
    parser.add_argument("--hist-bin-width", type=float, default=HIST_BIN_WIDTH, help="Histogram bin width for raw score stats.")
    parser.add_argument("--row-chunk", type=int, default=ROW_CHUNK, help="Matrix row chunk size.")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Worker process count.")
    parser.add_argument("--poll-interval", type=float, default=POLL_INTERVAL, help="Polling interval in seconds.")
    return parser.parse_args()

def apply_runtime_args(args: argparse.Namespace) -> None:
    global FOLDER, TARGET_DIR, MAPPING_FILE, MAX_I, SCORE_THRESHOLD, HIST_BIN_WIDTH, ROW_CHUNK, WORKERS, POLL_INTERVAL
    FOLDER = args.folder
    TARGET_DIR = args.target_dir
    MAPPING_FILE = args.mapping_file
    MAX_I = int(args.max_i)
    SCORE_THRESHOLD = float(args.score_threshold)
    HIST_BIN_WIDTH = float(args.hist_bin_width)
    ROW_CHUNK = int(args.row_chunk)
    WORKERS = int(args.workers)
    POLL_INTERVAL = float(args.poll_interval)

# ==========================================
# 2. 
# ==========================================

worker_pfam_map = None

# ==========================================
# 3. 
# ==========================================

def format_index(index: int) -> str:
    return str(index).zfill(4) if index < 1000 else str(index)

def load_pfam_map(file_path: str) -> dict:
    print(f"📖  Map: {file_path}")
    pfam_map = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    pfam_map[int(parts[1].strip())] = parts[0].strip()
    except FileNotFoundError:
        print(f"❌ :  {file_path}")
        sys.exit(1)
    return pfam_map

def _safe_remove(path: str):
    try:
        os.remove(path)
    except OSError:
        pass

def atomic_write_json(obj, out_path: str):
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, out_path)

def atomic_write_npz(out_path: str, **arrays):
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, out_path)

def init_worker():
    np.seterr(all="ignore")

# ==========================================
# 4. Sparse-cell selection
#    A) retain finite values (including zero)
#    B) retain 0 < score <= threshold
#    C) retain off-diagonal zero values when j != i
# ==========================================

def worker_process_single_file(args):
    i_val, j, folder, score_threshold, bin_width, row_chunk = args

    global worker_pfam_map
    if worker_pfam_map is None:
        return None

    i_fmt = format_index(i_val)
    j_fmt = format_index(j)

    file_path = os.path.join(folder, f"kl_{i_fmt}_{j_fmt}.npy")
    if not os.path.exists(file_path):
        alt_path = os.path.join(folder, f"kl_{i_val}_{j}.npy")
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            return None

    query_name = worker_pfam_map.get(i_val)
    target_name = worker_pfam_map.get(j)

    if query_name is None or target_name is None:
        _safe_remove(file_path)
        return None

    try:
        mat = np.load(file_path, allow_pickle=False)
        nrows, ncols = map(int, mat.shape)
        total_entries = int(mat.size)

        # ========= A) （ 0） =========
        local_hist = defaultdict(int)
        local_min = float("inf")
        local_max = float("-inf")
        zero_count = 0

        for r0 in range(0, nrows, row_chunk):
            r1 = min(nrows, r0 + row_chunk)
            block = np.asarray(mat[r0:r1, :], dtype=np.float64)

            if block.size == 0:
                continue

            zero_count += int(np.count_nonzero(block == 0.0))

            bmin = float(block.min())
            bmax = float(block.max())
            if bmin < local_min:
                local_min = bmin
            if bmax > local_max:
                local_max = bmax

            bins = np.floor(block / bin_width).astype(np.int64, copy=False)
            uniq, cnts = np.unique(bins, return_counts=True)
            for b, c in zip(uniq, cnts):
                local_hist[int(b)] += int(c)

        if total_entries == 0:
            local_min = 0.0
            local_max = 0.0

        # ========= B) ： 0 < score <= threshold =========
        rows_nz, cols_nz = np.nonzero(mat)
        kept_rows = np.empty((0,), dtype=np.int32)
        kept_cols = np.empty((0,), dtype=np.int32)
        kept_vals = np.empty((0,), dtype=np.float32)

        if rows_nz.size > 0:
            scores_nz = mat[rows_nz, cols_nz]
            keep = scores_nz <= score_threshold
            if np.any(keep):
                kept_rows = rows_nz[keep].astype(np.int32, copy=False)
                kept_cols = cols_nz[keep].astype(np.int32, copy=False)
                kept_vals = scores_nz[keep].astype(np.float32, copy=False)

        kept_count = int(kept_vals.size)

        # ========= C1) ，/ 0  =========
        self_diag_zero_count = 0
        self_offdiag_zero_count = 0

        if j == i_val and zero_count > 0:
            diag_len = min(nrows, ncols)
            if diag_len > 0:
                idx = np.arange(diag_len)
                diag_vals = mat[idx, idx]
                self_diag_zero_count = int(np.count_nonzero(diag_vals == 0.0))
            self_offdiag_zero_count = int(zero_count - self_diag_zero_count)

        # ========= C2) ， 0  =========
        zero_other_offdiag_rows = np.empty((0,), dtype=np.int32)
        zero_other_offdiag_cols = np.empty((0,), dtype=np.int32)

        if j != i_val and zero_count > 0:
            zr, zc = np.where(mat == 0.0)
            if zr.size > 0:
                keep_zero = (zr != zc)   # 
                if np.any(keep_zero):
                    zero_other_offdiag_rows = zr[keep_zero].astype(np.int32, copy=False)
                    zero_other_offdiag_cols = zc[keep_zero].astype(np.int32, copy=False)

        zero_other_offdiag_count = int(zero_other_offdiag_rows.size)

        del mat
        _safe_remove(file_path)

        hist_keys = list(local_hist.keys())
        hist_vals = [local_hist[k] for k in hist_keys]

        return (
            j,
            target_name,
            nrows,
            ncols,
            total_entries,
            zero_count,
            local_min,
            local_max,
            hist_keys,
            hist_vals,
            kept_rows,
            kept_cols,
            kept_vals,
            zero_other_offdiag_rows,
            zero_other_offdiag_cols,
            self_diag_zero_count,
            self_offdiag_zero_count,
        )

    except Exception:
        _safe_remove(file_path)
        return None

# ==========================================
# 5.  i
# ==========================================

def process_entire_i(i_val, pool, pfam_map_ref):
    i_fmt = format_index(i_val)
    query_name = pfam_map_ref.get(i_val, f"Unknown_{i_val}")
    print(f"\n🚀 []  i={i_fmt} ({query_name})...")
    t_start = time.time()

    # ： npz
    sparse_meta = []
    sparse_rows_list = []
    sparse_cols_list = []
    sparse_vals_list = []
    sparse_running_offset = 0

    #  0： npz
    zero_meta = []
    zero_rows_list = []
    zero_cols_list = []
    zero_running_offset = 0

    # 
    valid_file_count = 0
    kept_point_count = 0
    total_count_all = 0
    zero_count_all = 0
    self_diag_zero_count_all = 0
    self_offdiag_zero_count_all = 0
    other_offdiag_zero_count_all = 0

    gmin = float("inf")
    gmax = float("-inf")
    global_hist = defaultdict(int)

    total_tasks = MAX_I + 1
    tasks = (
        (i_val, j, FOLDER, SCORE_THRESHOLD, HIST_BIN_WIDTH, ROW_CHUNK)
        for j in range(total_tasks)
    )

    try:
        iterator = pool.imap_unordered(worker_process_single_file, tasks, chunksize=100)

        for res in tqdm(iterator, total=total_tasks, unit="file", desc=f" i={i_fmt}"):
            if res is None:
                continue

            (
                j,
                target_name,
                nrows,
                ncols,
                cnt_all,
                z_all,
                local_min,
                local_max,
                hist_keys,
                hist_vals,
                kept_rows,
                kept_cols,
                kept_vals,
                zero_rows,
                zero_cols,
                self_diag_zero_count,
                self_offdiag_zero_count,
            ) = res

            valid_file_count += 1
            total_count_all += int(cnt_all)
            zero_count_all += int(z_all)

            self_diag_zero_count_all += int(self_diag_zero_count)
            self_offdiag_zero_count_all += int(self_offdiag_zero_count)
            other_offdiag_zero_count_all += int(zero_rows.size)

            if cnt_all > 0:
                if local_min < gmin:
                    gmin = float(local_min)
                if local_max > gmax:
                    gmax = float(local_max)

            for k, v in zip(hist_keys, hist_vals):
                global_hist[int(k)] += int(v)

            #  meta: [j, nrows, ncols, offset, nnz]
            nnz_keep = int(kept_vals.size)
            sparse_meta.append([int(j), int(nrows), int(ncols), int(sparse_running_offset), int(nnz_keep)])
            if nnz_keep > 0:
                sparse_rows_list.append(kept_rows)
                sparse_cols_list.append(kept_cols)
                sparse_vals_list.append(kept_vals)
                sparse_running_offset += nnz_keep
                kept_point_count += nnz_keep

            # 0  meta: [j, nrows, ncols, offset, nnz_zero]
            # 
            nnz_zero = int(zero_rows.size)
            if nnz_zero > 0:
                zero_meta.append([int(j), int(nrows), int(ncols), int(zero_running_offset), int(nnz_zero)])
                zero_rows_list.append(zero_rows)
                zero_cols_list.append(zero_cols)
                zero_running_offset += nnz_zero

        if total_count_all == 0:
            gmin = 0.0
            gmax = 0.0

        os.makedirs(TARGET_DIR, exist_ok=True)

        # =========  stats =========
        stats = {
            "pfam_id": int(i_val),
            "pfam_name": query_name,
            "metric": "raw_score_distribution_all_including_zero",
            "bin_width": float(HIST_BIN_WIDTH),
            "histogram": dict(global_hist),
            "total_count": int(total_count_all),
            "zero_count_total": int(zero_count_all),
            "zero_frac_total": float(zero_count_all / total_count_all) if total_count_all > 0 else 0.0,
            "self_diag_zero_count": int(self_diag_zero_count_all),
            "self_offdiag_zero_count": int(self_offdiag_zero_count_all),
            "other_offdiag_zero_count": int(other_offdiag_zero_count_all),
            "min_score": float(gmin),
            "max_score": float(gmax),
            "valid_file_count": int(valid_file_count),
            "sparse_filter": {
                "type": "raw_score_threshold_sparse_npz",
                "threshold": float(SCORE_THRESHOLD),
                "condition": "0 < score <= threshold"
            },
            "zero_other_offdiag_saved": True,
            "zero_other_offdiag_note": "Only saved for j != i and row != col",
        }

        stats_path = os.path.join(TARGET_DIR, f"stats_{query_name}.json")
        atomic_write_json(stats, stats_path)

        # =========  npz =========
        if sparse_running_offset > 0:
            rows_arr = np.concatenate(sparse_rows_list)
            cols_arr = np.concatenate(sparse_cols_list)
            vals_arr = np.concatenate(sparse_vals_list)
        else:
            rows_arr = np.empty((0,), dtype=np.int32)
            cols_arr = np.empty((0,), dtype=np.int32)
            vals_arr = np.empty((0,), dtype=np.float32)

        sparse_meta_arr = np.asarray(sparse_meta, dtype=np.int64)

        sparse_path = os.path.join(TARGET_DIR, f"kl_{query_name}_le{int(SCORE_THRESHOLD)}_sparse.npz")
        atomic_write_npz(
            sparse_path,
            i=np.array([i_val], dtype=np.int32),
            threshold=np.array([SCORE_THRESHOLD], dtype=np.float32),
            meta=sparse_meta_arr,   # [j, nrows, ncols, offset, nnz]
            rows=rows_arr,          # 0-based
            cols=cols_arr,          # 0-based
            vals=vals_arr,
        )

        # =========  0  npz =========
        # 0， zero 
        if zero_running_offset > 0:
            zero_rows_arr = np.concatenate(zero_rows_list)
            zero_cols_arr = np.concatenate(zero_cols_list)
            zero_meta_arr = np.asarray(zero_meta, dtype=np.int64)

            zero_path = os.path.join(TARGET_DIR, f"zero_other_offdiag_{query_name}.npz")
            atomic_write_npz(
                zero_path,
                i=np.array([i_val], dtype=np.int32),
                meta=zero_meta_arr,   #  j
                rows=zero_rows_arr,
                cols=zero_cols_arr,
            )

        duration = time.time() - t_start
        print(
            f"✅ [] i={i_fmt} | "
            f"={total_count_all} | "
            f"0={zero_count_all} | "
            f"0={other_offdiag_zero_count_all} | "
            f"={kept_point_count} | "
            f"⏱️ {duration:.1f}s"
        )
        return True

    except Exception as e:
        print(f"❌  i={i_fmt}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        try:
            del sparse_meta
            del sparse_rows_list
            del sparse_cols_list
            del sparse_vals_list
            del zero_meta
            del zero_rows_list
            del zero_cols_list
            del global_hist
        except Exception:
            pass
        gc.collect()

# ==========================================
# 6. ：
# ==========================================

def monitor_and_dispatch():
    print("=" * 60)
    print("🔥 :  +  npz  + 0")
    print(f"📂 : {FOLDER}")
    print(f"🎯 : 0 < score <= {SCORE_THRESHOLD}")
    print("📌 : j != i  row != col  score == 0")
    print(f"🚀 : {WORKERS}")
    print("=" * 60)

    os.makedirs(TARGET_DIR, exist_ok=True)

    t0 = time.time()
    pfam_map = load_pfam_map(MAPPING_FILE)

    global worker_pfam_map
    worker_pfam_map = pfam_map

    print(f"✅  ({time.time() - t0:.1f}s)")
    print(f"📌 Pfam families in map: {len(worker_pfam_map)}")

    processed_set = set()
    print("🔍 ...")
    for fn in os.listdir(TARGET_DIR):
        if fn.startswith("stats_") and fn.endswith(".json"):
            p = os.path.join(TARGET_DIR, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if "pfam_id" in obj:
                    processed_set.add(int(obj["pfam_id"]))
            except Exception:
                pass
    print(f"✅  i : {len(processed_set)}")

    ctx = get_context("fork")

    print(f"🖥  {WORKERS}  (fork)...")
    pool = ctx.Pool(
        processes=WORKERS,
        initializer=init_worker,
        maxtasksperchild=1000
    )

    print("👀 ...")
    trigger_suffix = f"_{format_index(MAX_I)}.npy"

    try:
        while True:
            found_new = False
            try:
                with os.scandir(FOLDER) as it:
                    for entry in it:
                        if not entry.is_file():
                            continue

                        name = entry.name
                        if name.startswith("kl_") and name.endswith(trigger_suffix):
                            parts = name.split("_")
                            if len(parts) < 3:
                                continue

                            try:
                                i_val = int(parts[1])
                            except ValueError:
                                continue

                            if 0 <= i_val <= MAX_I and i_val not in processed_set:
                                processed_set.add(i_val)
                                process_entire_i(i_val, pool, pfam_map)
                                found_new = True

                if not found_new:
                    time.sleep(POLL_INTERVAL)

            except Exception as e:
                print(f"⚠️ : {e}")
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n🛑 ...")
        pool.terminate()
        pool.join()
        sys.exit(0)

# ==========================================
# 7. 
# ==========================================

if __name__ == "__main__":
    args = parse_args()
    apply_runtime_args(args)
    monitor_and_dispatch()
