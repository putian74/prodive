#!/usr/bin/env python3
"""Run CPU and GPU raw-KL calculations on the same deterministic sample."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmark_common import (
    cleanup_globs,
    command_environment,
    directory_stats,
    load_config,
    parse_cpp_stdout,
    read_json,
    read_pair_list_stats,
    resolved,
    run_monitored_command,
    stage_metric_base,
    write_json,
)
from raw_kl_compare import run_comparison


def run_is_complete(summary_path: Path) -> bool:
    if not summary_path.is_file():
        return False
    try:
        status = read_json(summary_path).get("status")
    except Exception:
        return False
    return status in {"completed", "completed_with_numerical_differences"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate CPU and GPU raw symmetric KL on the same sampled pairs."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repeat", type=int, action="append", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    validation = config.get("cpu_gpu_validation", {})
    if not bool(validation.get("enabled", True)):
        print("CPU/GPU raw-KL validation is disabled; nothing to run.")
        return

    repo_root = resolved(config, "repo_root")
    packed_db = resolved(config, "packed_db")
    output_root = resolved(config, "output_root")
    cpp_executable = resolved(config, "cpp_executable")
    pair_list = output_root / "design" / "pair_lists" / "pairs_cpu_gpu_validation.csv"
    if not pair_list.is_file():
        raise FileNotFoundError(
            f"Missing CPU/GPU validation pair list: {pair_list}. "
            "Run 01_prepare_benchmark.py first."
        )
    pair_stats = read_pair_list_stats(pair_list)
    expected_pair_count = int(validation.get("pair_count", 5))
    if pair_stats["pair_count"] != expected_pair_count:
        raise RuntimeError(
            f"{pair_list} contains {pair_stats['pair_count']} pairs, "
            f"expected {expected_pair_count}"
        )

    core = config["core"]
    configured_devices = [int(value) for value in core["gpu_device_ids"]]
    gpu_count = int(validation.get("gpu_count", 1))
    if gpu_count <= 0 or gpu_count > len(configured_devices):
        raise ValueError(
            f"cpu_gpu_validation.gpu_count={gpu_count} is incompatible with "
            f"configured devices {configured_devices}"
        )
    devices = configured_devices[:gpu_count]
    repeats = (
        sorted(set(args.repeat))
        if args.repeat
        else list(range(1, int(validation.get("repeats", 1)) + 1))
    )
    monitor_interval = float(config.get("monitor", {}).get("interval_seconds", 0.2))
    gpu_timeout = validation.get("gpu_timeout_seconds")
    cpu_timeout = validation.get("cpu_timeout_seconds")
    gpu_timeout = float(gpu_timeout) if gpu_timeout is not None else None
    cpu_timeout = float(cpu_timeout) if cpu_timeout is not None else None
    rtol = float(validation.get("rtol", 1e-4))
    atol = float(validation.get("atol", 1e-5))
    max_mismatch_fraction = float(validation.get("max_mismatch_fraction", 0.01))
    min_pearson_r = float(validation.get("min_pearson_r", 0.999))
    fail_on_tolerance = bool(validation.get("fail_on_tolerance", False))

    for repeat in repeats:
        run_id = f"rawkl_cpu_gpu_p{pair_stats['pair_count']}_g{gpu_count}_r{repeat:02d}"
        run_dir = output_root / "runs" / "cpu_gpu_validation" / run_id
        metrics_dir = run_dir / "metrics"
        gpu_output = run_dir / "gpu_raw_kl"
        cpu_output = run_dir / "cpu_raw_kl"
        comparison_output = run_dir / "comparison"
        summary_path = run_dir / "run_summary.json"
        print(f"[CPU/GPU validation] {run_id}")

        if run_is_complete(summary_path):
            print(f"  already completed; skipping: {summary_path}")
            continue
        for output in (gpu_output, cpu_output):
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(
                    f"Incomplete/non-empty validation output exists: {output}. "
                    "Move the run directory aside or choose a new output_root."
                )
            output.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        gpu_command: List[str] = [
            str(cpp_executable),
            "--pair_list",
            str(pair_list),
            "--packed_db",
            str(packed_db),
            "--num_streams",
            str(int(validation.get("num_streams", core.get("num_streams", 8)))),
            "--max_gpus",
            str(gpu_count),
            "--outdir",
            str(gpu_output),
        ]
        if bool(validation.get("memory_efficient", core.get("memory_efficient", True))):
            gpu_command.append("--memory_efficient")
        gpu_monitored = run_monitored_command(
            command=gpu_command,
            stage_dir=run_dir / "monitor_gpu_raw_kl",
            cwd=repo_root,
            env=command_environment(devices),
            gpu_device_ids=devices,
            monitor_interval_seconds=monitor_interval,
            timeout_seconds=gpu_timeout,
        )
        gpu_manifest_path = gpu_output / "prodive_run_manifest.json"
        gpu_manifest: Dict[str, Any] = (
            read_json(gpu_manifest_path) if gpu_manifest_path.is_file() else {}
        )
        gpu_matrix_stats = directory_stats(gpu_output, patterns=["kl_*.npy"])
        gpu_errors = []
        if gpu_monitored["returncode"] != 0:
            gpu_errors.append(f"GPU process return code={gpu_monitored['returncode']}")
        if gpu_manifest.get("status") != "completed":
            gpu_errors.append(f"GPU manifest status={gpu_manifest.get('status')!r}")
        if int(gpu_manifest.get("planned_pair_count", -1)) != pair_stats["pair_count"]:
            gpu_errors.append(
                f"GPU planned_pair_count={gpu_manifest.get('planned_pair_count')}"
            )
        if int(gpu_manifest.get("missing_output_count", -1)) != 0:
            gpu_errors.append(
                f"GPU missing_output_count={gpu_manifest.get('missing_output_count')}"
            )
        if gpu_matrix_stats["file_count"] != pair_stats["pair_count"]:
            gpu_errors.append(
                f"GPU matrix count={gpu_matrix_stats['file_count']}, "
                f"expected={pair_stats['pair_count']}"
            )
        if not (gpu_output / "RUN_COMPLETE").is_file():
            gpu_errors.append("GPU RUN_COMPLETE marker is missing")

        gpu_metric = stage_metric_base(
            run_id=run_id,
            run_type="cpu_gpu_validation",
            stage="gpu_raw_kl",
            repeat=repeat,
            pair_stats=pair_stats,
            gpu_count=gpu_count,
            gpu_device_ids=devices,
        )
        gpu_metric.update(gpu_monitored)
        gpu_metric.update(parse_cpp_stdout(Path(gpu_monitored["stdout_log"])))
        gpu_metric.update(
            {
                "status": "completed" if not gpu_errors else "failed_validation",
                "validation_status": "passed" if not gpu_errors else "failed",
                "validation_errors": gpu_errors,
                "raw_kl_only": True,
                "normalization_included": False,
                "matrix_output_file_count": gpu_matrix_stats["file_count"],
                "matrix_output_bytes": gpu_matrix_stats["bytes"],
                "external_pairs_per_second": (
                    pair_stats["pair_count"] / gpu_monitored["wall_seconds"]
                    if gpu_monitored["wall_seconds"] > 0
                    else None
                ),
                "external_window_cells_per_second": (
                    pair_stats["window_cells"] / gpu_monitored["wall_seconds"]
                    if gpu_monitored["wall_seconds"] > 0
                    else None
                ),
                "cpp_reported_computation_seconds": gpu_manifest.get("computation_seconds"),
                "cpp_manifest": str(gpu_manifest_path),
            }
        )
        write_json(metrics_dir / "gpu_raw_kl.json", gpu_metric)
        if gpu_errors:
            raise RuntimeError("GPU raw-KL validation stage failed: " + "; ".join(gpu_errors))

        cpu_command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "cpu_batch_runner.py"),
            "--pair-list",
            str(pair_list),
            "--packed-db",
            str(packed_db),
            "--output-dir",
            str(cpu_output),
            "--fragment",
            str(int(config["thresholds"]["fragment"])),
            "--acids-number",
            str(int(validation.get("acids_number", 21))),
            "--dtype",
            str(validation.get("dtype", "float32")),
        ]
        if bool(validation.get("require_numba", True)):
            cpu_command.append("--require-numba")
        cpu_env = command_environment([])
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            cpu_env[name] = "1"
        cpu_monitored = run_monitored_command(
            command=cpu_command,
            stage_dir=run_dir / "monitor_cpu_raw_kl",
            cwd=repo_root,
            env=cpu_env,
            gpu_device_ids=[],
            monitor_interval_seconds=monitor_interval,
            timeout_seconds=cpu_timeout,
        )
        cpu_summary_path = cpu_output / "cpu_batch_summary.json"
        cpu_summary: Dict[str, Any] = (
            read_json(cpu_summary_path) if cpu_summary_path.is_file() else {}
        )
        cpu_matrix_stats = directory_stats(cpu_output, patterns=["kl_*.npy"])
        cpu_errors = []
        if cpu_monitored["returncode"] != 0:
            cpu_errors.append(f"CPU process return code={cpu_monitored['returncode']}")
        if cpu_summary.get("status") != "completed":
            cpu_errors.append(f"CPU summary status={cpu_summary.get('status')!r}")
        if int(cpu_summary.get("pair_count", -1)) != pair_stats["pair_count"]:
            cpu_errors.append(f"CPU pair_count={cpu_summary.get('pair_count')}")
        if cpu_matrix_stats["file_count"] != pair_stats["pair_count"]:
            cpu_errors.append(
                f"CPU matrix count={cpu_matrix_stats['file_count']}, "
                f"expected={pair_stats['pair_count']}"
            )
        if not (cpu_output / "RUN_COMPLETE").is_file():
            cpu_errors.append("CPU RUN_COMPLETE marker is missing")

        cpu_metric = stage_metric_base(
            run_id=run_id,
            run_type="cpu_gpu_validation",
            stage="cpu_raw_kl",
            repeat=repeat,
            pair_stats=pair_stats,
            gpu_count=0,
            gpu_device_ids=[],
        )
        cpu_metric.update(cpu_monitored)
        cpu_metric.update(
            {
                "status": "completed" if not cpu_errors else "failed_validation",
                "validation_status": "passed" if not cpu_errors else "failed",
                "validation_errors": cpu_errors,
                "cpu_mode": "strict_single_core",
                "raw_kl_only": True,
                "normalization_included": False,
                "matrix_output_file_count": cpu_matrix_stats["file_count"],
                "matrix_output_bytes": cpu_matrix_stats["bytes"],
                "external_pairs_per_second": (
                    pair_stats["pair_count"] / cpu_monitored["wall_seconds"]
                    if cpu_monitored["wall_seconds"] > 0
                    else None
                ),
                "external_window_cells_per_second": (
                    pair_stats["window_cells"] / cpu_monitored["wall_seconds"]
                    if cpu_monitored["wall_seconds"] > 0
                    else None
                ),
                "cpu_batch_summary": str(cpu_summary_path),
                "cpu_internal_matrix_compute_seconds": cpu_summary.get(
                    "matrix_compute_seconds"
                ),
                "cpu_internal_matrix_inversion_seconds": cpu_summary.get(
                    "matrix_inversion_seconds"
                ),
                "cpu_internal_numba_warmup_seconds": cpu_summary.get(
                    "numba_warmup_seconds"
                ),
            }
        )
        write_json(metrics_dir / "cpu_raw_kl.json", cpu_metric)
        if cpu_errors:
            raise RuntimeError("CPU raw-KL validation stage failed: " + "; ".join(cpu_errors))

        compare_start = time.perf_counter()
        comparison = run_comparison(
            pair_list=pair_list,
            cpu_dir=cpu_output,
            gpu_dir=gpu_output,
            output_dir=comparison_output,
            rtol=rtol,
            atol=atol,
        )
        comparison_seconds = time.perf_counter() - compare_start
        if comparison["blocking_errors"]:
            raise RuntimeError(
                "Raw-KL comparison has structural errors: "
                + "; ".join(comparison["blocking_errors"])
            )
        mismatch_fraction = comparison.get("pooled_mismatch_fraction")
        pearson_r = comparison.get("pooled_pearson_r")
        mismatch_ok = (
            mismatch_fraction is not None
            and float(mismatch_fraction) <= max_mismatch_fraction
        )
        pearson_ok = pearson_r is not None and float(pearson_r) >= min_pearson_r
        numerical_pass = bool(mismatch_ok and pearson_ok)
        comparison_metric = stage_metric_base(
            run_id=run_id,
            run_type="cpu_gpu_validation",
            stage="raw_kl_comparison",
            repeat=repeat,
            pair_stats=pair_stats,
            gpu_count=gpu_count,
            gpu_device_ids=devices,
        )
        comparison_metric.update(comparison)
        comparison_metric.update(
            {
                "wall_seconds": float(comparison_seconds),
                "status": "completed",
                "validation_status": "passed" if numerical_pass else "warning",
                "numerical_acceptance_passed": numerical_pass,
                "configured_max_mismatch_fraction": max_mismatch_fraction,
                "configured_min_pearson_r": min_pearson_r,
            }
        )
        write_json(metrics_dir / "raw_kl_comparison.json", comparison_metric)

        cleanup = {"cpu": {}, "gpu": {}}
        if not bool(validation.get("keep_raw_matrices_after_comparison", True)):
            cleanup = {
                "cpu": cleanup_globs(cpu_output, ["kl_*.npy"]),
                "gpu": cleanup_globs(gpu_output, ["kl_*.npy"]),
            }

        status = "completed" if numerical_pass else "completed_with_numerical_differences"
        run_summary = {
            "status": status,
            "run_id": run_id,
            "comparison_scope": "CPU versus GPU raw symmetric KL only",
            "normalization_or_downstream_filtering_included": False,
            "pair_count": pair_stats["pair_count"],
            "window_cells": pair_stats["window_cells"],
            "gpu_wall_seconds": gpu_metric["wall_seconds"],
            "cpu_wall_seconds": cpu_metric["wall_seconds"],
            "cpu_over_gpu_wall_time_ratio": (
                cpu_metric["wall_seconds"] / gpu_metric["wall_seconds"]
                if gpu_metric["wall_seconds"] > 0
                else None
            ),
            "gpu_peak_cpu_rss_mib": gpu_metric["peak_cpu_rss_mib"],
            "gpu_peak_incremental_memory_mib_sum": gpu_metric[
                "gpu_peak_incremental_memory_mib_sum"
            ],
            "cpu_peak_rss_mib": cpu_metric["peak_cpu_rss_mib"],
            "raw_kl_comparison": comparison,
            "numerical_acceptance_passed": numerical_pass,
            "cleanup": cleanup,
        }
        write_json(summary_path, run_summary)
        print(
            f"  GPU wall={gpu_metric['wall_seconds']:.3f}s; "
            f"CPU wall={cpu_metric['wall_seconds']:.3f}s; "
            f"mismatch fraction={mismatch_fraction}; Pearson r={pearson_r}"
        )
        if not numerical_pass:
            print(
                "  [WARN] Numerical acceptance settings were not met. "
                "All raw results and error metrics were retained."
            )
            if fail_on_tolerance:
                raise RuntimeError(
                    "CPU/GPU raw-KL numerical acceptance failed and "
                    "fail_on_tolerance=true"
                )

    print(
        "CPU/GPU raw-KL validation finished. Root: "
        f"{output_root / 'runs' / 'cpu_gpu_validation'}"
    )


if __name__ == "__main__":
    main()
