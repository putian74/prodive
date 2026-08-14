#!/usr/bin/env python3
"""Compare CPU and GPU raw symmetric-KL NPY outputs pair by pair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def output_name(i: int, j: int) -> str:
    return f"kl_{i:04d}_{j:04d}.npy"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_pairs(path: Path) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"i", "j"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain i,j columns")
        for row in reader:
            pairs.append((int(row["i"]), int(row["j"])))
    if not pairs:
        raise ValueError(f"No pairs in {path}")
    return pairs


def compare_matrices(
    cpu: np.ndarray, gpu: np.ndarray, rtol: float, atol: float
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    result: Dict[str, Any] = {
        "cpu_shape": list(cpu.shape),
        "gpu_shape": list(gpu.shape),
        "shape_match": bool(cpu.shape == gpu.shape),
        "rtol": float(rtol),
        "atol": float(atol),
    }
    if cpu.shape != gpu.shape:
        result.update(
            {
                "allclose": False,
                "cell_count": int(cpu.size),
                "mismatch_count": None,
                "mismatch_fraction": None,
                "max_absolute_error": None,
                "mean_absolute_error": None,
                "rmse": None,
                "pearson_r": None,
            }
        )
        return result, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    cpu64 = np.asarray(cpu, dtype=np.float64).ravel()
    gpu64 = np.asarray(gpu, dtype=np.float64).ravel()
    finite = np.isfinite(cpu64) & np.isfinite(gpu64)
    difference = np.abs(cpu64 - gpu64)
    tolerance = atol + rtol * np.abs(gpu64)
    mismatch = (~finite) | (difference > tolerance)
    finite_difference = difference[finite]
    pearson = None
    if int(finite.sum()) >= 2:
        cpu_finite = cpu64[finite]
        gpu_finite = gpu64[finite]
        if np.std(cpu_finite) > 0 and np.std(gpu_finite) > 0:
            pearson = float(np.corrcoef(cpu_finite, gpu_finite)[0, 1])
    result.update(
        {
            "allclose": bool(
                np.allclose(cpu64, gpu64, rtol=rtol, atol=atol, equal_nan=True)
            ),
            "cell_count": int(cpu64.size),
            "finite_pair_count": int(finite.sum()),
            "nonfinite_cpu_count": int((~np.isfinite(cpu64)).sum()),
            "nonfinite_gpu_count": int((~np.isfinite(gpu64)).sum()),
            "mismatch_count": int(mismatch.sum()),
            "mismatch_fraction": float(mismatch.sum() / cpu64.size),
            "max_absolute_error": (
                float(finite_difference.max()) if finite_difference.size else None
            ),
            "mean_absolute_error": (
                float(finite_difference.mean()) if finite_difference.size else None
            ),
            "rmse": (
                float(np.sqrt(np.mean(np.square(cpu64[finite] - gpu64[finite]))))
                if finite.any()
                else None
            ),
            "absolute_error_p50": (
                float(np.percentile(finite_difference, 50))
                if finite_difference.size
                else None
            ),
            "absolute_error_p95": (
                float(np.percentile(finite_difference, 95))
                if finite_difference.size
                else None
            ),
            "absolute_error_p99": (
                float(np.percentile(finite_difference, 99))
                if finite_difference.size
                else None
            ),
            "pearson_r": pearson,
        }
    )
    return result, cpu64[finite], gpu64[finite]


def run_comparison(
    pair_list: Path,
    cpu_dir: Path,
    gpu_dir: Path,
    output_dir: Path,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    pairs = read_pairs(pair_list)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    pooled_cpu: List[np.ndarray] = []
    pooled_gpu: List[np.ndarray] = []
    blocking_errors: List[str] = []

    for i, j in pairs:
        filename = output_name(i, j)
        cpu_path = cpu_dir / filename
        gpu_path = gpu_dir / filename
        if not cpu_path.is_file() or not gpu_path.is_file():
            error = (
                f"pair=({i},{j}): missing CPU={not cpu_path.is_file()} "
                f"or GPU={not gpu_path.is_file()} output"
            )
            blocking_errors.append(error)
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "cpu_npy": str(cpu_path),
                    "gpu_npy": str(gpu_path),
                    "shape_match": False,
                    "allclose": False,
                    "error": error,
                }
            )
            continue
        cpu = np.load(cpu_path, allow_pickle=False)
        gpu = np.load(gpu_path, allow_pickle=False)
        metrics, cpu_finite, gpu_finite = compare_matrices(cpu, gpu, rtol, atol)
        if not metrics["shape_match"]:
            blocking_errors.append(
                f"pair=({i},{j}): CPU shape={cpu.shape}, GPU shape={gpu.shape}"
            )
        if cpu_finite.size:
            pooled_cpu.append(cpu_finite)
            pooled_gpu.append(gpu_finite)
        rows.append(
            {
                "i": i,
                "j": j,
                "cpu_npy": str(cpu_path),
                "gpu_npy": str(gpu_path),
                **metrics,
                "error": "",
            }
        )

    pair_csv = output_dir / "raw_kl_pair_comparison.csv"
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with pair_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row.get("shape_match")]
    total_cells = int(sum(int(row.get("cell_count") or 0) for row in valid_rows))
    mismatch_count = int(sum(int(row.get("mismatch_count") or 0) for row in valid_rows))
    all_errors = [
        value
        for row in valid_rows
        for value in [row.get("max_absolute_error")]
        if value is not None
    ]
    pooled_pearson = None
    pooled_rmse = None
    pooled_mean_abs = None
    if pooled_cpu:
        cpu_all = np.concatenate(pooled_cpu)
        gpu_all = np.concatenate(pooled_gpu)
        difference = cpu_all - gpu_all
        pooled_rmse = float(np.sqrt(np.mean(np.square(difference))))
        pooled_mean_abs = float(np.mean(np.abs(difference)))
        if cpu_all.size >= 2 and np.std(cpu_all) > 0 and np.std(gpu_all) > 0:
            pooled_pearson = float(np.corrcoef(cpu_all, gpu_all)[0, 1])

    allclose_pairs = int(sum(bool(row.get("allclose")) for row in valid_rows))
    summary = {
        "status": "failed" if blocking_errors else "completed",
        "comparison_scope": "raw symmetric KL only",
        "normalization_or_downstream_filtering_included": False,
        "raw_value_definition": "0.5 * (KL(1||2) + KL(2||1)); no /(3*fragment+2)",
        "pair_list": str(pair_list),
        "cpu_dir": str(cpu_dir),
        "gpu_dir": str(gpu_dir),
        "rtol": float(rtol),
        "atol": float(atol),
        "planned_pair_count": int(len(pairs)),
        "shape_matched_pair_count": int(len(valid_rows)),
        "allclose_pair_count": allclose_pairs,
        "allclose_pair_fraction": (
            float(allclose_pairs / len(valid_rows)) if valid_rows else None
        ),
        "total_compared_cells": total_cells,
        "total_mismatch_cells": mismatch_count,
        "pooled_mismatch_fraction": (
            float(mismatch_count / total_cells) if total_cells else None
        ),
        "maximum_pair_absolute_error": max(all_errors) if all_errors else None,
        "pooled_mean_absolute_error": pooled_mean_abs,
        "pooled_rmse": pooled_rmse,
        "pooled_pearson_r": pooled_pearson,
        "blocking_errors": blocking_errors,
        "pair_comparison_csv": str(pair_csv),
        "interpretation": (
            "A tolerance miss is reported as a numerical difference, not silently corrected. "
            "Only missing files or matrix-shape differences are blocking structural errors."
        ),
    }
    summary_path = output_dir / "raw_kl_comparison_summary.json"
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU and GPU raw KL NPY outputs.")
    parser.add_argument("--pair-list", required=True, type=Path)
    parser.add_argument("--cpu-dir", required=True, type=Path)
    parser.add_argument("--gpu-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("--rtol and --atol must be non-negative")
    summary = run_comparison(
        pair_list=args.pair_list.resolve(),
        cpu_dir=args.cpu_dir.resolve(),
        gpu_dir=args.gpu_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        rtol=args.rtol,
        atol=args.atol,
    )
    print(f"[OUT] {summary['pair_comparison_csv']}")
    print(f"[OUT] {args.output_dir.resolve() / 'raw_kl_comparison_summary.json'}")
    print(
        "[COMPARE] "
        f"pairs={summary['shape_matched_pair_count']}/{summary['planned_pair_count']}, "
        f"allclose_pairs={summary['allclose_pair_count']}, "
        f"pooled_mismatch_fraction={summary['pooled_mismatch_fraction']}, "
        f"pooled_rmse={summary['pooled_rmse']}, "
        f"pooled_pearson_r={summary['pooled_pearson_r']}"
    )
    if summary["blocking_errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
