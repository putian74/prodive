#!/usr/bin/env python3
"""Compute raw symmetric KL matrices on CPU from the packed database.

This runner is the performance counterpart to the unchanged C++/CUDA
executable: both start from the same packed float32 pi/A/B arrays and the same
pair list.  It never applies background normalization, thresholds, path
construction, or coverage rescoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
for _name in THREAD_ENVIRONMENT:
    os.environ[_name] = "1"


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CPU_REFERENCE_DIR = REPO_ROOT / "src" / "cpu_reference"
sys.path.insert(0, str(CPU_REFERENCE_DIR))

import cpu_kl_reference as reference  # noqa: E402

np = reference.np


_WORKER_MODELS: Dict[int, Dict[str, Any]] = {}
_WORKER_ACIDS_NUMBER = 21
_WORKER_OUTPUT_DIR: Optional[Path] = None
_WORKER_SAVE_MATRICES = True


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_index(path: Path) -> Dict[int, Dict[str, int]]:
    required = {
        "family_id",
        "fragment",
        "nfrag",
        "pi_offset",
        "pi_count",
        "A_offset",
        "A_count",
        "B_offset",
        "B_count",
    }
    records: Dict[int, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            record = {key: int(row[key]) for key in required}
            family_id = record["family_id"]
            if family_id in records:
                raise ValueError(f"Duplicate family_id={family_id} in {path}")
            records[family_id] = record
    if not records:
        raise ValueError(f"No family rows in {path}")
    return records


def load_pairs(path: Path) -> List[Dict[str, int]]:
    required = {"i", "j", "window_cells"}
    rows: List[Dict[str, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for rank, row in enumerate(reader, start=1):
            i = int(row["i"])
            j = int(row["j"])
            if i == j:
                raise ValueError(f"Self pair ({i}, {j}) is not allowed")
            rows.append(
                {
                    "sample_rank": int(row.get("sample_rank") or rank),
                    "i": i,
                    "j": j,
                    "window_cells": int(row["window_cells"]),
                }
            )
    if not rows:
        raise ValueError(f"No pairs in {path}")
    return rows


def output_name(i: int, j: int) -> str:
    return f"kl_{i:04d}_{j:04d}.npy"


def _copy_block(
    array: np.memmap,
    offset: int,
    count: int,
    shape: Sequence[int],
    dtype: Any,
) -> np.ndarray:
    block = np.asarray(array[offset : offset + count], dtype=np.float32)
    if int(block.size) != int(count):
        raise RuntimeError(
            f"Packed binary short read: offset={offset}, count={count}, got={block.size}"
        )
    return np.ascontiguousarray(block.reshape(tuple(shape)), dtype=dtype)


def load_packed_models(
    family_ids: Sequence[int],
    index: Mapping[int, Mapping[str, int]],
    packed_db: Path,
    fragment: int,
    m_states: int,
    n_obs: int,
    dtype: Any,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Union[float, int]]]:
    pi_all = np.memmap(packed_db / "pi_all.float32.bin", dtype=np.float32, mode="r")
    A_all = np.memmap(packed_db / "A_all.float32.bin", dtype=np.float32, mode="r")
    B_all = np.memmap(packed_db / "B_all.float32.bin", dtype=np.float32, mode="r")

    models: Dict[int, Dict[str, Any]] = {}
    load_seconds = 0.0
    inversion_seconds = 0.0
    inverse_fallbacks = 0
    for position, family_id in enumerate(family_ids, start=1):
        if family_id not in index:
            raise KeyError(f"family_id={family_id} is absent from packed index")
        row = index[family_id]
        if int(row["fragment"]) != fragment:
            raise RuntimeError(
                f"family_id={family_id}: fragment={row['fragment']}, expected={fragment}"
            )
        nfrag = int(row["nfrag"])
        expected = {
            "pi_count": nfrag * m_states,
            "A_count": nfrag * m_states * m_states,
            "B_count": nfrag * m_states * n_obs,
        }
        for key, value in expected.items():
            if int(row[key]) != value:
                raise RuntimeError(
                    f"family_id={family_id}: {key}={row[key]}, expected={value}"
                )

        print(f"[CPU PACKED LOAD {position}/{len(family_ids)}] family_id={family_id}", flush=True)
        load_start = time.perf_counter()
        starts = _copy_block(
            pi_all,
            int(row["pi_offset"]),
            int(row["pi_count"]),
            (nfrag, 1, m_states),
            dtype,
        )
        transitions = _copy_block(
            A_all,
            int(row["A_offset"]),
            int(row["A_count"]),
            (nfrag, m_states, m_states),
            dtype,
        )
        emissions = _copy_block(
            B_all,
            int(row["B_offset"]),
            int(row["B_count"]),
            (nfrag, m_states, n_obs),
            dtype,
        )
        family_load_seconds = time.perf_counter() - load_start

        inversion_start = time.perf_counter()
        left, fallbacks = reference.precompute_left_vectors(starts, transitions)
        family_inversion_seconds = time.perf_counter() - inversion_start
        load_seconds += family_load_seconds
        inversion_seconds += family_inversion_seconds
        inverse_fallbacks += int(fallbacks)
        models[family_id] = {
            "family_id": family_id,
            "nfrag": nfrag,
            "starts": starts,
            "transitions": transitions,
            "emissions": emissions,
            "left": left,
            "load_seconds": family_load_seconds,
            "inversion_seconds": family_inversion_seconds,
            "inverse_fallbacks": int(fallbacks),
        }

    return models, {
        "packed_data_load_seconds": float(load_seconds),
        "matrix_inversion_seconds": float(inversion_seconds),
        "inverse_fallback_count": int(inverse_fallbacks),
    }


def configure_worker(
    models: Dict[int, Dict[str, Any]],
    acids_number: int,
    output_dir: Path,
    save_matrices: bool,
) -> None:
    global _WORKER_MODELS
    global _WORKER_ACIDS_NUMBER
    global _WORKER_OUTPUT_DIR
    global _WORKER_SAVE_MATRICES
    _WORKER_MODELS = models
    _WORKER_ACIDS_NUMBER = int(acids_number)
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_SAVE_MATRICES = bool(save_matrices)


def compute_pair(pair: Mapping[str, int]) -> Dict[str, Any]:
    i = int(pair["i"])
    j = int(pair["j"])
    model_1 = _WORKER_MODELS[i]
    model_2 = _WORKER_MODELS[j]
    compute_start = time.perf_counter()
    matrix = reference.compute_raw_symmetric_kl(
        _WORKER_ACIDS_NUMBER,
        model_1["starts"],
        model_1["transitions"],
        model_1["emissions"],
        model_2["starts"],
        model_2["transitions"],
        model_2["emissions"],
        model_1["left"],
        model_2["left"],
    )
    compute_seconds = time.perf_counter() - compute_start
    expected_shape = (int(model_1["nfrag"]), int(model_2["nfrag"]))
    if matrix.shape != expected_shape:
        raise RuntimeError(
            f"pair=({i},{j}): shape={matrix.shape}, expected={expected_shape}"
        )
    if int(matrix.size) != int(pair["window_cells"]):
        raise RuntimeError(
            f"pair=({i},{j}): cells={matrix.size}, expected={pair['window_cells']}"
        )
    finite_count = int(np.isfinite(matrix).sum())
    if finite_count != int(matrix.size):
        raise RuntimeError(
            f"pair=({i},{j}) produced {matrix.size - finite_count} non-finite values"
        )

    save_seconds = 0.0
    output_path = None
    output_bytes = 0
    if _WORKER_SAVE_MATRICES:
        if _WORKER_OUTPUT_DIR is None:
            raise RuntimeError("CPU worker output directory is not configured")
        output = _WORKER_OUTPUT_DIR / output_name(i, j)
        save_start = time.perf_counter()
        np.save(output, matrix)
        save_seconds = time.perf_counter() - save_start
        output_path = str(output)
        output_bytes = int(output.stat().st_size)

    return {
        "sample_rank": int(pair["sample_rank"]),
        "i": i,
        "j": j,
        "nfrag_i": int(matrix.shape[0]),
        "nfrag_j": int(matrix.shape[1]),
        "window_cells": int(matrix.size),
        "compute_seconds": float(compute_seconds),
        "save_seconds": float(save_seconds),
        "finite_value_count": finite_count,
        "minimum_raw_kl": float(np.min(matrix)),
        "maximum_raw_kl": float(np.max(matrix)),
        "output_npy": output_path,
        "output_bytes": output_bytes,
        "worker_pid": int(os.getpid()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute packed-database raw symmetric KL matrices on CPU."
    )
    parser.add_argument("--pair-list", required=True, type=Path)
    parser.add_argument("--packed-db", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fragment", required=True, type=int)
    parser.add_argument("--acids-number", type=int, default=21)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--require-numba", action="store_true")
    parser.add_argument("--save-matrices", action="store_true")
    args = parser.parse_args()

    if args.fragment <= 0 or args.acids_number <= 0 or args.processes <= 0:
        raise ValueError("fragment, acids-number, and processes must be positive")
    if args.require_numba and not reference.NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for reportable CPU timing")
    if args.processes > 1 and "fork" not in mp.get_all_start_methods():
        raise RuntimeError("Multi-process CPU timing currently requires the Linux fork start method")

    pair_list = args.pair_list.resolve()
    packed_db = args.packed_db.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("kl_*.npy")):
        raise RuntimeError(f"Output already contains KL matrices: {output_dir}")

    process_start = time.perf_counter()
    with (packed_db / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    m_states = int(metadata.get("m_states", 3 * args.fragment + 1))
    n_obs = int(metadata.get("n_obs", args.acids_number))
    if int(metadata.get("fragment", args.fragment)) != args.fragment:
        raise RuntimeError("Packed metadata fragment length does not match --fragment")
    if n_obs != args.acids_number:
        raise RuntimeError(
            f"Packed n_obs={n_obs} does not match acids-number={args.acids_number}"
        )

    pairs = load_pairs(pair_list)
    index = load_index(packed_db / "index.csv")
    family_ids = sorted({row[key] for row in pairs for key in ("i", "j")})
    dtype = np.float32 if args.dtype == "float32" else np.float64
    family_prepare_start = time.perf_counter()
    models, family_metrics = load_packed_models(
        family_ids,
        index,
        packed_db,
        args.fragment,
        m_states,
        n_obs,
        dtype,
    )
    family_prepare_wall_seconds = time.perf_counter() - family_prepare_start

    first = pairs[0]
    first_1 = models[first["i"]]
    first_2 = models[first["j"]]
    warmup_start = time.perf_counter()
    reference.warm_numba(
        args.acids_number,
        first_1["starts"],
        first_1["transitions"],
        first_1["emissions"],
        first_2["starts"],
        first_2["transitions"],
        first_2["emissions"],
    )
    warmup_seconds = time.perf_counter() - warmup_start

    configure_worker(models, args.acids_number, output_dir, args.save_matrices)
    pair_stage_start = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    if args.processes == 1:
        for position, pair in enumerate(pairs, start=1):
            rows.append(compute_pair(pair))
            print(f"[CPU KL {position}/{len(pairs)}]", flush=True)
        start_method = "serial"
    else:
        context = mp.get_context("fork")
        chunksize = max(1, len(pairs) // max(1, args.processes * 8))
        with context.Pool(
            processes=args.processes,
            initializer=configure_worker,
            initargs=(models, args.acids_number, output_dir, args.save_matrices),
        ) as pool:
            for position, row in enumerate(
                pool.imap_unordered(compute_pair, pairs, chunksize=chunksize), start=1
            ):
                rows.append(row)
                if position == len(pairs) or position % max(1, len(pairs) // 10) == 0:
                    print(f"[CPU KL {position}/{len(pairs)}]", flush=True)
        start_method = "fork"
    pair_stage_wall_seconds = time.perf_counter() - pair_stage_start
    rows.sort(key=lambda row: int(row["sample_rank"]))

    pair_csv = output_dir / "cpu_pair_metrics.csv"
    with pair_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    whole_batch_seconds = time.perf_counter() - process_start
    total_cells = int(sum(int(row["window_cells"]) for row in rows))
    matrix_compute_cpu_seconds = float(sum(float(row["compute_seconds"]) for row in rows))
    npy_save_cpu_seconds = float(sum(float(row["save_seconds"]) for row in rows))
    matrix_count = sum(1 for _ in output_dir.glob("kl_*.npy"))
    summary = {
        "status": "completed",
        "mode": "packed_cpu_raw_kl_performance",
        "input_source": "same packed float32 database used by C++/CUDA",
        "raw_value_definition": "0.5 * (KL(1||2) + KL(2||1)); no /(3*fragment+2)",
        "normalization_or_downstream_filtering_included": False,
        "pair_list": str(pair_list),
        "packed_db": str(packed_db),
        "pair_count": int(len(pairs)),
        "unique_family_count": int(len(family_ids)),
        "window_cells": total_cells,
        "fragment": int(args.fragment),
        "m_states": m_states,
        "n_obs": n_obs,
        "calculation_dtype": args.dtype,
        "cpu_process_count": int(args.processes),
        "worker_start_method": start_method,
        "numba_available": bool(reference.NUMBA_AVAILABLE),
        "family_prepare_wall_seconds": float(family_prepare_wall_seconds),
        **family_metrics,
        "numba_warmup_seconds": float(warmup_seconds),
        "parallel_pair_stage_wall_seconds": float(pair_stage_wall_seconds),
        "matrix_compute_cpu_seconds_sum": matrix_compute_cpu_seconds,
        "npy_save_cpu_seconds_sum": npy_save_cpu_seconds,
        "whole_batch_internal_seconds": float(whole_batch_seconds),
        "window_cells_per_pair_stage_wall_second": (
            float(total_cells / pair_stage_wall_seconds)
            if pair_stage_wall_seconds > 0
            else None
        ),
        "window_cells_per_compute_cpu_second": (
            float(total_cells / matrix_compute_cpu_seconds)
            if matrix_compute_cpu_seconds > 0
            else None
        ),
        "save_matrices": bool(args.save_matrices),
        "matrix_output_file_count": int(matrix_count),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "pair_metrics_csv": str(pair_csv),
    }
    summary_path = output_dir / "cpu_packed_summary.json"
    write_json(summary_path, summary)
    (output_dir / "RUN_COMPLETE").write_text("completed\n", encoding="utf-8")
    print(f"[OUT] CPU packed summary: {summary_path}")
    print(
        f"[TIME] total={whole_batch_seconds:.6f}s, pair-stage={pair_stage_wall_seconds:.6f}s, "
        f"cells={total_cells:,}, processes={args.processes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
