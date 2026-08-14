#!/usr/bin/env python3
"""Run CPU packed-database scaling measurements for CPU/GPU comparison."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from benchmark_common import (
    cleanup_globs,
    command_environment,
    directory_stats,
    load_config,
    read_json,
    read_pair_list_stats,
    resolved,
    run_monitored_command,
    stage_metric_base,
    write_json,
)


THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def run_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        metric = read_json(path)
    except Exception:
        return False
    return metric.get("status") == "completed" and metric.get("validation_status") == "passed"


def configured_plan(config: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    comparison = config.get("cpu_gpu_comparison", {})
    pair_counts = sorted({int(value) for value in comparison.get("pair_counts", [])})
    repeats = list(range(1, int(comparison.get("repeats", 3)) + 1))
    base_process_counts = sorted(
        {int(value) for value in comparison.get("cpu_process_counts", [1])}
    )
    points = {
        (pair_count, process_count, repeat)
        for pair_count in pair_counts
        for process_count in base_process_counts
        for repeat in repeats
    }
    parallel = comparison.get("parallel_scaling", {})
    if bool(parallel.get("enabled", True)):
        parallel_pair_count = int(parallel.get("pair_count", max(pair_counts)))
        for process_count in parallel.get("process_counts", [1, 2, 4, 8, 16]):
            for repeat in repeats:
                points.add((parallel_pair_count, int(process_count), repeat))
    return sorted(points)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CPU scaling measurements on the same packed pair lists as GPU."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pair-count", type=int, action="append", default=None)
    parser.add_argument("--process-count", type=int, action="append", default=None)
    parser.add_argument("--repeat", type=int, action="append", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    comparison = config.get("cpu_gpu_comparison", {})
    if not bool(comparison.get("enabled", True)):
        print("CPU/GPU performance comparison is disabled; nothing to run.")
        return

    repo_root = resolved(config, "repo_root")
    packed_db = resolved(config, "packed_db")
    output_root = resolved(config, "output_root")
    monitor_interval = float(config.get("monitor", {}).get("interval_seconds", 0.2))
    timeout = comparison.get("cpu_timeout_seconds")
    timeout = float(timeout) if timeout is not None else None
    save_matrices = bool(comparison.get("save_cpu_matrices_during_run", True))

    plan = configured_plan(config)
    if args.pair_count:
        selected = set(args.pair_count)
        plan = [point for point in plan if point[0] in selected]
    if args.process_count:
        selected = set(args.process_count)
        plan = [point for point in plan if point[1] in selected]
    if args.repeat:
        selected = set(args.repeat)
        plan = [point for point in plan if point[2] in selected]
    if not plan:
        raise ValueError("No CPU comparison runs remain after applying selections")
    for pair_count, process_count, repeat in plan:
        if pair_count <= 0 or process_count <= 0 or repeat <= 0:
            raise ValueError(f"Invalid comparison point: {(pair_count, process_count, repeat)}")

    if bool(comparison.get("randomize_run_order", True)):
        rng = random.Random(int(config["sampling"]["seed"]) + 2607)
        rng.shuffle(plan)
    plan_rows = [
        {
            "execution_order": position,
            "pair_count": pair_count,
            "cpu_process_count": process_count,
            "repeat": repeat,
            "run_id": f"cpu_p{pair_count}_c{process_count}_r{repeat:02d}",
        }
        for position, (pair_count, process_count, repeat) in enumerate(plan, start=1)
    ]
    write_json(
        output_root / "design" / "cpu_gpu_comparison_execution_plan.json",
        plan_rows,
    )

    completed_now = 0
    skipped = 0
    for execution_order, (pair_count, process_count, repeat) in enumerate(plan, start=1):
        pair_list = output_root / "design" / "pair_lists" / f"pairs_{pair_count}.csv"
        if not pair_list.is_file():
            raise FileNotFoundError(
                f"Missing comparison pair list: {pair_list}. Run preparation first."
            )
        pair_stats = read_pair_list_stats(pair_list)
        if int(pair_stats["pair_count"]) != pair_count:
            raise RuntimeError(f"{pair_list} contains {pair_stats['pair_count']} rows")

        run_id = f"cpu_p{pair_count}_c{process_count}_r{repeat:02d}"
        run_dir = output_root / "runs" / "cpu_gpu_comparison" / run_id
        cpu_output = run_dir / "cpu_raw_kl"
        metrics_path = run_dir / "stage_metrics.json"
        print(f"[{execution_order}/{len(plan)}] {run_id}")
        if run_is_complete(metrics_path):
            print(f"  already completed; skipping: {metrics_path}")
            skipped += 1
            continue
        if cpu_output.exists() and any(cpu_output.iterdir()):
            raise RuntimeError(
                f"Incomplete/non-empty CPU comparison output exists: {cpu_output}"
            )
        cpu_output.mkdir(parents=True, exist_ok=True)

        command: List[str] = [
            sys.executable,
            str(Path(__file__).resolve().parent / "cpu_packed_runner.py"),
            "--pair-list",
            str(pair_list),
            "--packed-db",
            str(packed_db),
            "--output-dir",
            str(cpu_output),
            "--fragment",
            str(int(config["thresholds"]["fragment"])),
            "--acids-number",
            str(int(comparison.get("acids_number", 21))),
            "--dtype",
            str(comparison.get("dtype", "float32")),
            "--processes",
            str(process_count),
        ]
        if bool(comparison.get("require_numba", True)):
            command.append("--require-numba")
        if save_matrices:
            command.append("--save-matrices")

        env = command_environment([])
        for name in THREAD_ENVIRONMENT:
            env[name] = "1"
        monitored = run_monitored_command(
            command=command,
            stage_dir=run_dir / "monitor_cpu_raw_kl",
            cwd=repo_root,
            env=env,
            gpu_device_ids=[],
            monitor_interval_seconds=monitor_interval,
            timeout_seconds=timeout,
        )
        summary_path = cpu_output / "cpu_packed_summary.json"
        summary: Dict[str, Any] = read_json(summary_path) if summary_path.is_file() else {}
        matrix_stats = directory_stats(cpu_output, patterns=["kl_*.npy"])
        errors = []
        if monitored["returncode"] != 0:
            errors.append(f"CPU process return code={monitored['returncode']}")
        if summary.get("status") != "completed":
            errors.append(f"CPU summary status={summary.get('status')!r}")
        if int(summary.get("pair_count", -1)) != pair_count:
            errors.append(f"CPU pair_count={summary.get('pair_count')}")
        expected_matrices = pair_count if save_matrices else 0
        if matrix_stats["file_count"] != expected_matrices:
            errors.append(
                f"CPU matrix count={matrix_stats['file_count']}, expected={expected_matrices}"
            )
        if not (cpu_output / "RUN_COMPLETE").is_file():
            errors.append("CPU RUN_COMPLETE marker is missing")

        metric = stage_metric_base(
            run_id=run_id,
            run_type="cpu_gpu_comparison",
            stage="cpu_raw_kl",
            repeat=repeat,
            pair_stats=pair_stats,
            gpu_count=0,
            gpu_device_ids=[],
        )
        metric.update(monitored)
        metric.update(
            {
                "status": "completed" if not errors else "failed_validation",
                "validation_status": "passed" if not errors else "failed",
                "validation_errors": errors,
                "cpu_process_count": process_count,
                "cpu_mode": (
                    "strict_single_process_single_thread"
                    if process_count == 1
                    else "multi_process_one_thread_per_process"
                ),
                "input_source": "packed_database",
                "raw_kl_only": True,
                "normalization_included": False,
                "pair_list": str(pair_list),
                "matrix_output_file_count": matrix_stats["file_count"],
                "matrix_output_bytes": matrix_stats["bytes"],
                "external_pairs_per_second": (
                    pair_count / monitored["wall_seconds"]
                    if monitored["wall_seconds"] > 0
                    else None
                ),
                "external_window_cells_per_second": (
                    pair_stats["window_cells"] / monitored["wall_seconds"]
                    if monitored["wall_seconds"] > 0
                    else None
                ),
                "cpu_packed_summary": str(summary_path),
                "cpu_family_prepare_wall_seconds": summary.get(
                    "family_prepare_wall_seconds"
                ),
                "cpu_packed_data_load_seconds": summary.get(
                    "packed_data_load_seconds"
                ),
                "cpu_matrix_inversion_seconds": summary.get(
                    "matrix_inversion_seconds"
                ),
                "cpu_numba_warmup_seconds": summary.get("numba_warmup_seconds"),
                "cpu_parallel_pair_stage_wall_seconds": summary.get(
                    "parallel_pair_stage_wall_seconds"
                ),
                "cpu_matrix_compute_seconds_sum": summary.get(
                    "matrix_compute_cpu_seconds_sum"
                ),
                "cpu_npy_save_seconds_sum": summary.get("npy_save_cpu_seconds_sum"),
                "completed_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            }
        )
        cleanup_result = {"removed_file_count": 0, "removed_bytes": 0}
        if not errors and bool(
            config.get("cleanup", {}).get(
                "comparison_cpu_matrices_after_success", True
            )
        ):
            cleanup_result = cleanup_globs(cpu_output, ["kl_*.npy"])
        metric["cleanup"] = cleanup_result
        write_json(metrics_path, metric)
        write_json(run_dir / "run_summary.json", metric)
        if errors:
            raise RuntimeError(f"CPU comparison failed: {run_id}: {'; '.join(errors)}")
        completed_now += 1
        print(
            f"  wall={metric['wall_seconds']:.3f}s, cells/s="
            f"{metric['external_window_cells_per_second']:.1f}, "
            f"peak RSS={metric['peak_cpu_rss_mib']:.1f} MiB"
        )

    print(
        f"CPU comparison finished: {completed_now} new run(s), "
        f"{skipped} existing run(s) skipped."
    )
    print(f"Run root: {output_root / 'runs' / 'cpu_gpu_comparison'}")


if __name__ == "__main__":
    main()
