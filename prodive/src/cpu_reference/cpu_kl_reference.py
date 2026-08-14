#!/usr/bin/env python3
"""Serial CPU reference for ProDive fragment-level symmetric KL matrices.

The script intentionally contains no plotting or background-filtering code.  It
computes one dense family-pair matrix and saves the same raw quantity and matrix
orientation as the C++/CUDA program:

    output.shape == (nfrag_hmm1, nfrag_hmm2)
    output[i, j] == 0.5 * (KL(fragment1_i || fragment2_j)
                           + KL(fragment2_j || fragment1_i))

The GPU NPY files store this raw symmetric KL value.  Do not divide by
``3 * fragment + 2`` when validating CPU against GPU; that factor belongs to
the downstream background-normalized filtering formula.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Make the reference run a strict one-core baseline.  These variables must be
# defined before NumPy/Numba import their BLAS/OpenMP runtimes.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*decorator_args: Any, **decorator_kwargs: Any):
        """Small correctness-only fallback when Numba is unavailable."""
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
            return decorator_args[0]

        def decorate(function):
            return function

        return decorate

import pfam_selfhmm_read_hhm
import splits_of_hhm
import tools


@njit(fastmath=True, cache=True)
def m_vector_calculation(
    acids_number: int,
    transition_1: np.ndarray,
    emission_1: np.ndarray,
    transition_2: np.ndarray,
    emission_2: np.ndarray,
) -> np.ndarray:
    profile_hmm_1 = transition_1.dot(emission_1)
    profile_hmm_2 = transition_2.dot(emission_2)
    result = np.zeros(len(transition_1), dtype=transition_1.dtype)
    for i in range(len(transition_1)):
        intermediate = 0.0
        for j in range(acids_number):
            value_1 = profile_hmm_1[i, j]
            value_2 = profile_hmm_2[i, j]
            if value_1 != 0.0 and value_2 != 0.0:
                intermediate += value_1 * np.log(value_1 / value_2)
        result[i] = intermediate
    return result


@njit(fastmath=True, cache=True)
def initial_probability_calculation(
    acids_number: int,
    start_1: np.ndarray,
    emission_1: np.ndarray,
    start_2: np.ndarray,
    emission_2: np.ndarray,
) -> float:
    distribution_1 = start_1.dot(emission_1)
    distribution_2 = start_2.dot(emission_2)
    result = 0.0
    for j in range(acids_number):
        value_1 = distribution_1[0, j]
        value_2 = distribution_2[0, j]
        if value_1 != 0.0 and value_2 != 0.0:
            result += value_1 * np.log(value_1 / value_2)
    return result


def hmm_length(path: Path) -> int:
    value = tools.get_hmm_length(str(path))
    if value is None:
        raise RuntimeError(f"Could not read LENG from HHM file: {path}")
    return int(value)


def build_windows(
    hmm_file_path: Path,
    fragment: int,
    dtype: np.dtype,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    length = hmm_length(hmm_file_path)
    nfrag = length - fragment + 1
    if nfrag <= 0:
        raise ValueError(
            f"HHM length {length} is shorter than fragment length {fragment}: {hmm_file_path}"
        )

    transition_data, emission_match, emission_insert = pfam_selfhmm_read_hhm.hmm_read(
        str(hmm_file_path)
    )
    starts: List[np.ndarray] = []
    transitions: List[np.ndarray] = []
    emissions: List[np.ndarray] = []
    for start in range(nfrag):
        end = start + fragment + 1
        start_prob, transition_prob, emission_prob = (
            splits_of_hhm.split_transition_and_emission(
                start,
                end,
                transition_data,
                emission_match,
                emission_insert,
            )
        )
        starts.append(np.ascontiguousarray(start_prob, dtype=dtype))
        transitions.append(np.ascontiguousarray(transition_prob, dtype=dtype))
        emissions.append(np.ascontiguousarray(emission_prob, dtype=dtype))
    return starts, transitions, emissions


def precompute_left_vectors(
    starts: Sequence[np.ndarray],
    transitions: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], int]:
    left_vectors: List[np.ndarray] = []
    fallback_count = 0
    for start, transition in zip(starts, transitions):
        identity_minus_transition = np.eye(
            transition.shape[0], dtype=transition.dtype
        ) - transition
        try:
            inverse = np.linalg.inv(identity_minus_transition)
        except np.linalg.LinAlgError:
            # The C++ implementation also falls back to identity when its LU
            # decomposition reports a non-invertible matrix.
            inverse = np.eye(transition.shape[0], dtype=transition.dtype)
            fallback_count += 1
        left_vectors.append(np.ascontiguousarray(start.dot(inverse)))
    return left_vectors, fallback_count


def warm_numba(
    acids_number: int,
    starts_1: Sequence[np.ndarray],
    transitions_1: Sequence[np.ndarray],
    emissions_1: Sequence[np.ndarray],
    starts_2: Sequence[np.ndarray],
    transitions_2: Sequence[np.ndarray],
    emissions_2: Sequence[np.ndarray],
) -> None:
    """Compile both Numba kernels before the measured matrix loop."""
    m_vector_calculation(
        acids_number,
        transitions_1[0],
        emissions_1[0],
        transitions_2[0],
        emissions_2[0],
    )
    initial_probability_calculation(
        acids_number,
        starts_1[0],
        emissions_1[0],
        starts_2[0],
        emissions_2[0],
    )


def compute_raw_symmetric_kl(
    acids_number: int,
    starts_1: Sequence[np.ndarray],
    transitions_1: Sequence[np.ndarray],
    emissions_1: Sequence[np.ndarray],
    starts_2: Sequence[np.ndarray],
    transitions_2: Sequence[np.ndarray],
    emissions_2: Sequence[np.ndarray],
    left_1: Sequence[np.ndarray],
    left_2: Sequence[np.ndarray],
) -> np.ndarray:
    """Return raw KL in GPU-compatible row/column orientation."""
    nfrag_1 = len(emissions_1)
    nfrag_2 = len(emissions_2)
    output = np.empty((nfrag_1, nfrag_2), dtype=np.float32)

    for i in range(nfrag_1):
        for j in range(nfrag_2):
            vector_12 = m_vector_calculation(
                acids_number,
                transitions_1[i],
                emissions_1[i],
                transitions_2[j],
                emissions_2[j],
            )
            initial_12 = initial_probability_calculation(
                acids_number,
                starts_1[i],
                emissions_1[i],
                starts_2[j],
                emissions_2[j],
            )
            kl_12 = float(left_1[i].dot(vector_12).item() + initial_12)

            vector_21 = m_vector_calculation(
                acids_number,
                transitions_2[j],
                emissions_2[j],
                transitions_1[i],
                emissions_1[i],
            )
            initial_21 = initial_probability_calculation(
                acids_number,
                starts_2[j],
                emissions_2[j],
                starts_1[i],
                emissions_1[i],
            )
            kl_21 = float(left_2[j].dot(vector_21).item() + initial_21)

            symmetric = 0.5 * (kl_12 + kl_21)
            # Match the final validity rule in the CUDA kernel.  Small numeric
            # differences remain possible because the GPU uses mixed float and
            # double intermediates plus EPS clamping.
            if not np.isfinite(symmetric) or symmetric < 0.0 or symmetric > 1000.0:
                symmetric = 1000.0
            output[i, j] = np.float32(symmetric)
    return output


def compare_with_gpu(
    cpu_matrix: np.ndarray,
    gpu_path: Path,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    gpu_matrix = np.load(gpu_path)
    report: Dict[str, Any] = {
        "gpu_npy": str(gpu_path),
        "cpu_shape": list(cpu_matrix.shape),
        "gpu_shape": list(gpu_matrix.shape),
        "rtol": float(rtol),
        "atol": float(atol),
    }
    if gpu_matrix.shape != cpu_matrix.shape:
        report.update(
            {
                "shape_match": False,
                "allclose": False,
                "note": (
                    "Shapes differ. Ensure the GPU file is kl_<hmm1_id>_<hmm2_id>.npy "
                    "in the same family order; do not silently transpose for validation."
                ),
            }
        )
        return report

    cpu64 = np.asarray(cpu_matrix, dtype=np.float64)
    gpu64 = np.asarray(gpu_matrix, dtype=np.float64)
    finite = np.isfinite(cpu64) & np.isfinite(gpu64)
    difference = np.abs(cpu64 - gpu64)
    finite_difference = difference[finite]
    tolerance = atol + rtol * np.abs(gpu64)
    mismatch = finite & (difference > tolerance)
    report.update(
        {
            "shape_match": True,
            "allclose": bool(np.allclose(cpu64, gpu64, rtol=rtol, atol=atol, equal_nan=True)),
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
        }
    )
    return report


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute one serial-CPU ProDive raw symmetric-KL matrix without plotting, "
            "and optionally compare it with a C++/CUDA NPY file."
        )
    )
    parser.add_argument("--hmm1", required=True, type=Path)
    parser.add_argument("--hmm2", required=True, type=Path)
    parser.add_argument("--fragment", type=int, default=6)
    parser.add_argument("--acids-number", type=int, default=21)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-npy", type=Path, default=None)
    parser.add_argument("--comparison-json", type=Path, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fragment <= 0:
        raise ValueError("--fragment must be positive")
    if args.acids_number <= 0:
        raise ValueError("--acids-number must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("--rtol and --atol must be non-negative")
    for path in (args.hmm1, args.hmm2):
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if args.dtype == "float32" else np.float64
    total_start = time.perf_counter()

    build_start = time.perf_counter()
    starts_1, transitions_1, emissions_1 = build_windows(
        args.hmm1.resolve(), args.fragment, dtype
    )
    starts_2, transitions_2, emissions_2 = build_windows(
        args.hmm2.resolve(), args.fragment, dtype
    )
    build_seconds = time.perf_counter() - build_start

    inversion_start = time.perf_counter()
    left_1, inverse_fallbacks_1 = precompute_left_vectors(starts_1, transitions_1)
    left_2, inverse_fallbacks_2 = precompute_left_vectors(starts_2, transitions_2)
    inversion_seconds = time.perf_counter() - inversion_start

    warmup_start = time.perf_counter()
    warm_numba(
        args.acids_number,
        starts_1,
        transitions_1,
        emissions_1,
        starts_2,
        transitions_2,
        emissions_2,
    )
    numba_warmup_seconds = time.perf_counter() - warmup_start

    compute_start = time.perf_counter()
    matrix = compute_raw_symmetric_kl(
        args.acids_number,
        starts_1,
        transitions_1,
        emissions_1,
        starts_2,
        transitions_2,
        emissions_2,
        left_1,
        left_2,
    )
    compute_seconds = time.perf_counter() - compute_start
    np.save(output, matrix)
    total_seconds = time.perf_counter() - total_start

    cells = int(matrix.size)
    metadata: Dict[str, Any] = {
        "mode": "strict_single_core_cpu_reference",
        "hmm1": str(args.hmm1.resolve()),
        "hmm2": str(args.hmm2.resolve()),
        "fragment": int(args.fragment),
        "acids_number": int(args.acids_number),
        "calculation_dtype": str(args.dtype),
        "output_dtype": str(matrix.dtype),
        "output_npy": str(output),
        "matrix_shape": list(matrix.shape),
        "nfrag1": int(matrix.shape[0]),
        "nfrag2": int(matrix.shape[1]),
        "window_cells": cells,
        "build_windows_seconds": float(build_seconds),
        "matrix_inversion_seconds": float(inversion_seconds),
        "numba_warmup_seconds_excluded_from_compute": float(numba_warmup_seconds),
        "matrix_compute_seconds": float(compute_seconds),
        "whole_process_work_seconds": float(total_seconds),
        "window_cells_per_compute_second": (
            float(cells / compute_seconds) if compute_seconds > 0 else None
        ),
        "inverse_fallbacks_hmm1": int(inverse_fallbacks_1),
        "inverse_fallbacks_hmm2": int(inverse_fallbacks_2),
        "raw_value_definition": "0.5 * (KL(1||2) + KL(2||1)); no /(3*fragment+2)",
        "matrix_orientation": "rows=hmm1 windows; columns=hmm2 windows",
        "python": sys.version.replace("\n", " "),
        "numba_available": bool(NUMBA_AVAILABLE),
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        },
    }

    metadata_path = (
        args.metadata_json.resolve()
        if args.metadata_json is not None
        else output.with_suffix(output.suffix + ".metadata.json")
    )
    write_json(metadata_path, metadata)

    print(f"[OUT] CPU matrix: {output}")
    print(f"[OUT] Metadata:   {metadata_path}")
    print(f"[INFO] shape={matrix.shape}, cells={cells:,}")
    print(
        f"[TIME] build={build_seconds:.6f}s, inverse={inversion_seconds:.6f}s, "
        f"numba_warmup={numba_warmup_seconds:.6f}s, compute={compute_seconds:.6f}s"
    )
    if not NUMBA_AVAILABLE:
        print(
            "[WARN] Numba is not installed. Results remain valid, but the pure-Python "
            "fallback is much slower and should not be used for performance benchmarking."
        )

    if args.gpu_npy is not None:
        if not args.gpu_npy.is_file():
            raise FileNotFoundError(args.gpu_npy)
        comparison = compare_with_gpu(
            cpu_matrix=matrix,
            gpu_path=args.gpu_npy.resolve(),
            rtol=args.rtol,
            atol=args.atol,
        )
        comparison_path = (
            args.comparison_json.resolve()
            if args.comparison_json is not None
            else output.with_suffix(output.suffix + ".comparison.json")
        )
        write_json(comparison_path, comparison)
        print(f"[OUT] Comparison: {comparison_path}")
        print(
            "[COMPARE] "
            f"shape_match={comparison.get('shape_match')}, "
            f"allclose={comparison.get('allclose')}, "
            f"max_abs={comparison.get('max_absolute_error')}, "
            f"rmse={comparison.get('rmse')}, "
            f"mismatch_fraction={comparison.get('mismatch_fraction')}"
        )


if __name__ == "__main__":
    main()
