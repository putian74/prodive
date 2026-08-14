#!/usr/bin/env python3
"""Benchmark the unchanged ProDive C++/CUDA stage on sampled pair lists."""

from __future__ import annotations

import argparse
import random
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


def run_is_complete(metrics_path: Path) -> bool:
    if not metrics_path.is_file():
        return False
    try:
        metrics = read_json(metrics_path)
    except Exception:
        return False
    return metrics.get("status") == "completed" and metrics.get("validation_status") == "passed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sampled ProDive C++/CUDA benchmarks.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=["gpu", "compare"],
        default="gpu",
        help="gpu uses core.pair_counts; compare uses cpu_gpu_comparison.pair_counts.",
    )
    parser.add_argument("--pair-count", type=int, action="append", default=None)
    parser.add_argument("--gpu-count", type=int, action="append", default=None)
    parser.add_argument("--repeat", type=int, action="append", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = resolved(config, "repo_root")
    packed_db = resolved(config, "packed_db")
    output_root = resolved(config, "output_root")
    cpp_executable = resolved(config, "cpp_executable")
    core = config["core"]
    monitor_interval = float(config.get("monitor", {}).get("interval_seconds", 0.5))

    comparison = config.get("cpu_gpu_comparison", {})
    default_pair_counts = (
        core.get("pair_counts", config["sampling"]["pair_counts"])
        if args.profile == "gpu"
        else comparison.get("pair_counts", [])
    )
    default_gpu_counts = (
        core["gpu_counts"]
        if args.profile == "gpu"
        else comparison.get("gpu_counts", core["gpu_counts"])
    )
    default_repeats = int(
        core.get("repeats", 3)
        if args.profile == "gpu"
        else comparison.get("repeats", core.get("repeats", 3))
    )
    pair_counts = (
        sorted(set(args.pair_count))
        if args.pair_count
        else sorted({int(x) for x in default_pair_counts})
    )
    gpu_counts = (
        sorted(set(args.gpu_count))
        if args.gpu_count
        else sorted({int(x) for x in default_gpu_counts})
    )
    repeats = (
        sorted(set(args.repeat))
        if args.repeat
        else list(range(1, default_repeats + 1))
    )
    configured_devices = [int(x) for x in core["gpu_device_ids"]]
    timeout_seconds = core.get("timeout_seconds")
    timeout_seconds = float(timeout_seconds) if timeout_seconds is not None else None

    if not cpp_executable.is_file():
        raise FileNotFoundError(f"C++ executable not found: {cpp_executable}")

    pair_inputs: Dict[str, Dict[str, Any]] = {}
    for pair_count in pair_counts:
        pair_list = output_root / "design" / "pair_lists" / f"pairs_{pair_count}.csv"
        if not pair_list.is_file():
            raise FileNotFoundError(
                f"Missing pair list: {pair_list}. Run 01_prepare_benchmark.py first."
            )
        pair_stats = read_pair_list_stats(pair_list)
        if pair_stats["pair_count"] != pair_count:
            raise RuntimeError(f"{pair_list} contains {pair_stats['pair_count']} rows, expected {pair_count}")
        pair_inputs[f"stratified_{pair_count}"] = {
            "sample_kind": "stratified_workload",
            "pair_list": pair_list,
            "pair_stats": pair_stats,
        }

    for gpu_count in gpu_counts:
        if gpu_count <= 0 or gpu_count > len(configured_devices):
            raise ValueError(
                f"gpu_count={gpu_count} is incompatible with configured devices {configured_devices}"
            )

    plan = [
        ("stratified_workload", f"stratified_{pair_count}", gpu_count, repeat)
        for pair_count in pair_counts
        for gpu_count in gpu_counts
        for repeat in repeats
    ]
    full_family_config = core.get("full_family_coverage", {})
    if (
        args.profile == "gpu"
        and bool(full_family_config.get("enabled", False))
        and args.pair_count is None
    ):
        coverage_path = output_root / "design" / "pair_lists" / "pairs_full_family_coverage.csv"
        if not coverage_path.is_file():
            raise FileNotFoundError(
                f"Missing full-family coverage list: {coverage_path}. Run 01_prepare_benchmark.py first."
            )
        coverage_stats = read_pair_list_stats(coverage_path)
        pair_inputs["full_family_coverage"] = {
            "sample_kind": "full_family_coverage",
            "pair_list": coverage_path,
            "pair_stats": coverage_stats,
        }
        special_gpu_counts = [
            int(x) for x in full_family_config.get("gpu_counts", [max(gpu_counts)])
        ]
        if args.gpu_count:
            special_gpu_counts = [value for value in special_gpu_counts if value in gpu_counts]
        configured_special_repeats = list(
            range(1, int(full_family_config.get("repeats", 1)) + 1)
        )
        special_repeats = (
            [value for value in repeats if value in configured_special_repeats]
            if args.repeat
            else configured_special_repeats
        )
        plan.extend(
            ("full_family_coverage", "full_family_coverage", gpu_count, repeat)
            for gpu_count in special_gpu_counts
            for repeat in special_repeats
        )

    for _, _, gpu_count, _ in plan:
        if gpu_count <= 0 or gpu_count > len(configured_devices):
            raise ValueError(
                f"planned gpu_count={gpu_count} is incompatible with configured devices {configured_devices}"
            )
    if bool(core.get("randomize_run_order", True)):
        profile_offset = 1103 if args.profile == "gpu" else 2103
        order_rng = random.Random(int(config["sampling"]["seed"]) + profile_offset)
        order_rng.shuffle(plan)
    plan_manifest = [
        {
            "execution_order": position,
            "benchmark_profile": args.profile,
            "sample_kind": sample_kind,
            "pair_count": pair_inputs[pair_key]["pair_stats"]["pair_count"],
            "gpu_count": gpu_count,
            "repeat": repeat,
            "run_id": (
                f"core_fullfamilies_p{pair_inputs[pair_key]['pair_stats']['pair_count']}_g{gpu_count}_r{repeat:02d}"
                if sample_kind == "full_family_coverage"
                else f"core_p{pair_inputs[pair_key]['pair_stats']['pair_count']}_g{gpu_count}_r{repeat:02d}"
            ),
        }
        for position, (sample_kind, pair_key, gpu_count, repeat) in enumerate(plan, start=1)
    ]
    profile_plan_path = (
        output_root / "design" / f"core_execution_plan_{args.profile}.json"
    )
    write_json(profile_plan_path, plan_manifest)
    if args.profile == "gpu":
        # Retain the original manifest name for compatibility with completed
        # benchmark archives and downstream notebooks.
        write_json(output_root / "design" / "core_execution_plan.json", plan_manifest)

    total_runs = len(plan)
    completed_now = 0
    skipped = 0
    for execution_order, (sample_kind, pair_key, gpu_count, repeat) in enumerate(plan, start=1):
        pair_list = pair_inputs[pair_key]["pair_list"]
        pair_stats = pair_inputs[pair_key]["pair_stats"]
        pair_count = int(pair_stats["pair_count"])
        devices = configured_devices[:gpu_count]
        run_id = (
            f"core_fullfamilies_p{pair_count}_g{gpu_count}_r{repeat:02d}"
            if sample_kind == "full_family_coverage"
            else f"core_p{pair_count}_g{gpu_count}_r{repeat:02d}"
        )
        run_dir = output_root / "runs" / "core" / run_id
        cpp_output = run_dir / "cpp_output"
        monitor_dir = run_dir / "monitor_cpp_kl"
        metrics_path = run_dir / "stage_metrics.json"
        print(f"[{execution_order}/{total_runs}] {run_id}")

        if run_is_complete(metrics_path):
            print(f"  already completed; skipping: {metrics_path}")
            skipped += 1
            continue
        if cpp_output.exists() and any(cpp_output.iterdir()):
            raise RuntimeError(
                f"Incomplete/non-empty run directory already exists: {cpp_output}. "
                "Move it aside or use a new output_root; benchmark runs never mix old and new outputs."
            )
        cpp_output.mkdir(parents=True, exist_ok=True)

        command: List[str] = [
            str(cpp_executable),
            "--pair_list",
            str(pair_list),
            "--packed_db",
            str(packed_db),
            "--num_streams",
            str(int(core.get("num_streams", 8))),
            "--max_gpus",
            str(gpu_count),
            "--outdir",
            str(cpp_output),
        ]
        if bool(core.get("memory_efficient", True)):
            command.append("--memory_efficient")

        env = command_environment(devices)
        monitored = run_monitored_command(
            command=command,
            stage_dir=monitor_dir,
            cwd=repo_root,
            env=env,
            gpu_device_ids=devices,
            monitor_interval_seconds=monitor_interval,
            timeout_seconds=timeout_seconds,
        )
        matrix_stats = directory_stats(cpp_output, patterns=["kl_*.npy"])
        all_output_stats = directory_stats(cpp_output)
        cpp_manifest_path = cpp_output / "prodive_run_manifest.json"
        cpp_manifest: Dict[str, Any] = {}
        if cpp_manifest_path.is_file():
            cpp_manifest = read_json(cpp_manifest_path)

        validation_errors = []
        if monitored["returncode"] != 0:
            validation_errors.append(f"process return code was {monitored['returncode']}")
        if cpp_manifest.get("status") != "completed":
            validation_errors.append(
                f"C++ manifest status was {cpp_manifest.get('status')!r}, expected 'completed'"
            )
        if int(cpp_manifest.get("planned_pair_count", -1)) != pair_count:
            validation_errors.append(
                f"planned_pair_count={cpp_manifest.get('planned_pair_count')}, expected {pair_count}"
            )
        if int(cpp_manifest.get("missing_output_count", -1)) != 0:
            validation_errors.append(
                f"missing_output_count={cpp_manifest.get('missing_output_count')}"
            )
        if matrix_stats["file_count"] != pair_count:
            validation_errors.append(
                f"matrix file count={matrix_stats['file_count']}, expected {pair_count}"
            )
        if not (cpp_output / "RUN_COMPLETE").is_file():
            validation_errors.append("RUN_COMPLETE marker is missing")

        metric = stage_metric_base(
            run_id=run_id,
            run_type="core",
            stage="cpp_kl",
            repeat=repeat,
            pair_stats=pair_stats,
            gpu_count=gpu_count,
            gpu_device_ids=devices,
        )
        metric.update(monitored)
        metric.update(parse_cpp_stdout(Path(monitored["stdout_log"])))
        metric.update(
            {
                "status": "completed" if not validation_errors else "failed_validation",
                "validation_status": "passed" if not validation_errors else "failed",
                "validation_errors": validation_errors,
                "execution_order": execution_order,
                "benchmark_profile": args.profile,
                "sample_kind": sample_kind,
                "pair_list": str(pair_list),
                "packed_db": str(packed_db),
                "cpp_executable": str(cpp_executable),
                "num_streams": int(core.get("num_streams", 8)),
                "memory_efficient": bool(core.get("memory_efficient", True)),
                "matrix_output_file_count": matrix_stats["file_count"],
                "matrix_output_bytes": matrix_stats["bytes"],
                "stage_output_file_count": all_output_stats["file_count"],
                "stage_output_bytes": all_output_stats["bytes"],
                "external_pairs_per_second": (
                    pair_count / monitored["wall_seconds"] if monitored["wall_seconds"] > 0 else None
                ),
                "external_window_cells_per_second": (
                    pair_stats["window_cells"] / monitored["wall_seconds"]
                    if monitored["wall_seconds"] > 0
                    else None
                ),
                "cpp_manifest": str(cpp_manifest_path),
                "cpp_reported_computation_seconds": cpp_manifest.get("computation_seconds"),
                "cpp_reported_pairs_completed": cpp_manifest.get("pairs_completed"),
                "completed_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            }
        )

        cleanup_result = {"removed_file_count": 0, "removed_bytes": 0}
        if not validation_errors and bool(
            config.get("cleanup", {}).get("core_cpp_matrices_after_success", True)
        ):
            cleanup_result = cleanup_globs(cpp_output, ["kl_*.npy"])
        metric["cleanup"] = cleanup_result
        write_json(metrics_path, metric)
        write_json(run_dir / "run_summary.json", metric)

        if validation_errors:
            print("  validation failed:")
            for error in validation_errors:
                print(f"    - {error}")
            raise RuntimeError(f"Benchmark run failed validation: {run_id}")
        completed_now += 1
        print(
            f"  wall={metric['wall_seconds']:.3f}s, "
            f"peak CPU RSS={metric['peak_cpu_rss_mib']:.1f} MiB, "
            f"incremental GPU memory={metric['gpu_peak_incremental_memory_mib_sum']:.1f} MiB"
        )

    print(f"Core benchmark finished: {completed_now} new run(s), {skipped} existing run(s) skipped.")
    print(f"Run root: {output_root / 'runs' / 'core'}")


if __name__ == "__main__":
    main()
