#!/usr/bin/env python3
"""Run the plotting-free serial CPU reference on one sampled pair list.

This program computes only the raw symmetric KL matrix.  It deliberately does
not apply ``/(3*fragment+2)``, background normalization, either threshold, path
construction, or coverage rescoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping


# Set these before NumPy/Numba/BLAS are imported.  This is a strict serial CPU
# reference, so a GPU comparison is against one CPU core rather than a hidden
# multithreaded BLAS configuration.
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_index(path: Path) -> Dict[int, Dict[str, str]]:
    required = {"family_id", "hmm_file", "nfrag"}
    records: Dict[int, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            family_id = int(row["family_id"])
            if family_id in records:
                raise ValueError(f"Duplicate family_id={family_id} in {path}")
            records[family_id] = dict(row)
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
        for row in reader:
            i = int(row["i"])
            j = int(row["j"])
            if i == j:
                raise ValueError(f"Self pair ({i}, {j}) is not allowed")
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "window_cells": int(row["window_cells"]),
                    "nfrag_i": int(row.get("nfrag_i") or 0),
                    "nfrag_j": int(row.get("nfrag_j") or 0),
                }
            )
    if not rows:
        raise ValueError(f"No pairs in {path}")
    return rows


def resolve_hmm_path(
    record: Mapping[str, str], packed_db: Path, packed_metadata: Mapping[str, Any]
) -> Path:
    raw = Path(os.path.expandvars(os.path.expanduser(str(record["hmm_file"]))))
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(packed_db / raw)
        input_dir = packed_metadata.get("input_dir")
        if input_dir:
            candidates.append(Path(str(input_dir)) / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve HHM for family_id={record.get('family_id')}: {rendered}"
    )


def load_family_model(
    family_id: int,
    record: Mapping[str, str],
    packed_db: Path,
    packed_metadata: Mapping[str, Any],
    fragment: int,
    dtype: Any,
) -> Dict[str, Any]:
    hmm_path = resolve_hmm_path(record, packed_db, packed_metadata)
    build_start = time.perf_counter()
    starts, transitions, emissions = reference.build_windows(hmm_path, fragment, dtype)
    build_seconds = time.perf_counter() - build_start

    inversion_start = time.perf_counter()
    left, inverse_fallbacks = reference.precompute_left_vectors(starts, transitions)
    inversion_seconds = time.perf_counter() - inversion_start

    expected_nfrag = int(record["nfrag"])
    if len(starts) != expected_nfrag:
        raise RuntimeError(
            f"family_id={family_id}: CPU nfrag={len(starts)}, packed index nfrag={expected_nfrag}"
        )
    return {
        "family_id": int(family_id),
        "hmm_path": hmm_path,
        "starts": starts,
        "transitions": transitions,
        "emissions": emissions,
        "left": left,
        "nfrag": int(len(starts)),
        "build_seconds": float(build_seconds),
        "inversion_seconds": float(inversion_seconds),
        "inverse_fallbacks": int(inverse_fallbacks),
    }


def output_name(i: int, j: int) -> str:
    return f"kl_{i:04d}_{j:04d}.npy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute raw symmetric ProDive KL matrices with one serial CPU core."
    )
    parser.add_argument("--pair-list", required=True, type=Path)
    parser.add_argument("--packed-db", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fragment", required=True, type=int)
    parser.add_argument("--acids-number", type=int, default=21)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--require-numba", action="store_true")
    args = parser.parse_args()

    if args.fragment <= 0 or args.acids_number <= 0:
        raise ValueError("--fragment and --acids-number must be positive")
    if args.require_numba and not reference.NUMBA_AVAILABLE:
        raise RuntimeError(
            "Numba is required for reportable CPU timing. Install benchmark/requirements.txt."
        )

    pair_list = args.pair_list.resolve()
    packed_db = args.packed_db.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("kl_*.npy")):
        raise RuntimeError(
            f"CPU output directory already contains kl_*.npy files: {output_dir}"
        )

    metadata_path = packed_db / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        packed_metadata = json.load(handle)
    index = load_index(packed_db / "index.csv")
    pairs = load_pairs(pair_list)
    family_ids = sorted({row[key] for row in pairs for key in ("i", "j")})
    missing = sorted(set(family_ids) - set(index))
    if missing:
        raise KeyError(f"Pair list contains family IDs absent from packed index: {missing[:20]}")

    dtype = np.float32 if args.dtype == "float32" else np.float64
    process_start = time.perf_counter()
    models: Dict[int, Dict[str, Any]] = {}
    for position, family_id in enumerate(family_ids, start=1):
        print(f"[CPU LOAD {position}/{len(family_ids)}] family_id={family_id}", flush=True)
        models[family_id] = load_family_model(
            family_id=family_id,
            record=index[family_id],
            packed_db=packed_db,
            packed_metadata=packed_metadata,
            fragment=args.fragment,
            dtype=dtype,
        )

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

    pair_rows: List[Dict[str, Any]] = []
    total_compute_seconds = 0.0
    total_save_seconds = 0.0
    total_cells = 0
    for position, pair in enumerate(pairs, start=1):
        i = pair["i"]
        j = pair["j"]
        model_1 = models[i]
        model_2 = models[j]
        print(
            f"[CPU KL {position}/{len(pairs)}] pair=({i},{j}), "
            f"shape=({model_1['nfrag']},{model_2['nfrag']})",
            flush=True,
        )
        compute_start = time.perf_counter()
        matrix = reference.compute_raw_symmetric_kl(
            args.acids_number,
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

        expected_shape = (model_1["nfrag"], model_2["nfrag"])
        if matrix.shape != expected_shape:
            raise RuntimeError(
                f"pair=({i},{j}): matrix shape={matrix.shape}, expected={expected_shape}"
            )
        if int(matrix.size) != int(pair["window_cells"]):
            raise RuntimeError(
                f"pair=({i},{j}): matrix cells={matrix.size}, "
                f"pair-list window_cells={pair['window_cells']}"
            )

        output = output_dir / output_name(i, j)
        save_start = time.perf_counter()
        np.save(output, matrix)
        save_seconds = time.perf_counter() - save_start
        total_compute_seconds += compute_seconds
        total_save_seconds += save_seconds
        total_cells += int(matrix.size)
        pair_rows.append(
            {
                "i": i,
                "j": j,
                "nfrag_i": int(matrix.shape[0]),
                "nfrag_j": int(matrix.shape[1]),
                "window_cells": int(matrix.size),
                "compute_seconds": float(compute_seconds),
                "save_seconds": float(save_seconds),
                "window_cells_per_compute_second": (
                    float(matrix.size / compute_seconds) if compute_seconds > 0 else None
                ),
                "output_npy": str(output),
                "output_bytes": int(output.stat().st_size),
            }
        )

    process_seconds = time.perf_counter() - process_start
    pair_csv = output_dir / "cpu_pair_metrics.csv"
    with pair_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    summary = {
        "status": "completed",
        "mode": "strict_single_core_cpu_reference_batch",
        "raw_value_definition": "0.5 * (KL(1||2) + KL(2||1)); no /(3*fragment+2)",
        "matrix_orientation": "rows=family i windows; columns=family j windows",
        "pair_list": str(pair_list),
        "packed_db": str(packed_db),
        "pair_count": int(len(pairs)),
        "unique_family_count": int(len(family_ids)),
        "window_cells": int(total_cells),
        "fragment": int(args.fragment),
        "acids_number": int(args.acids_number),
        "calculation_dtype": args.dtype,
        "numba_available": bool(reference.NUMBA_AVAILABLE),
        "family_build_seconds": float(sum(row["build_seconds"] for row in models.values())),
        "matrix_inversion_seconds": float(
            sum(row["inversion_seconds"] for row in models.values())
        ),
        "numba_warmup_seconds": float(warmup_seconds),
        "matrix_compute_seconds": float(total_compute_seconds),
        "npy_save_seconds": float(total_save_seconds),
        "whole_batch_internal_seconds": float(process_seconds),
        "window_cells_per_compute_second": (
            float(total_cells / total_compute_seconds) if total_compute_seconds > 0 else None
        ),
        "inverse_fallback_count": int(
            sum(row["inverse_fallbacks"] for row in models.values())
        ),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "family_inputs": [
            {
                "family_id": family_id,
                "hmm_path": str(models[family_id]["hmm_path"]),
                "nfrag": models[family_id]["nfrag"],
                "build_seconds": models[family_id]["build_seconds"],
                "inversion_seconds": models[family_id]["inversion_seconds"],
                "inverse_fallbacks": models[family_id]["inverse_fallbacks"],
            }
            for family_id in family_ids
        ],
        "pair_metrics_csv": str(pair_csv),
    }
    summary_path = output_dir / "cpu_batch_summary.json"
    write_json(summary_path, summary)
    (output_dir / "RUN_COMPLETE").write_text("completed\n", encoding="utf-8")

    print(f"[OUT] CPU matrices: {output_dir}")
    print(f"[OUT] CPU summary:  {summary_path}")
    print(
        f"[TIME] total={process_seconds:.6f}s, compute={total_compute_seconds:.6f}s, "
        f"cells={total_cells:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
