#!/usr/bin/env python3
"""Benchmark the unchanged ProDive workflow through final path rescoring."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from benchmark_common import (
    cleanup_globs,
    command_environment,
    count_csv_rows,
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


def completed_run(summary_path: Path) -> bool:
    if not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
    except Exception:
        return False
    return summary.get("status") in {"completed", "completed_no_final_paths"}


def skipped_metric(
    run_id: str,
    stage: str,
    repeat: int,
    pair_stats: Mapping[str, int],
    gpu_count: int,
    devices: Sequence[int],
    reason: str,
) -> Dict[str, Any]:
    metric = stage_metric_base(
        run_id=run_id,
        run_type="end_to_end",
        stage=stage,
        repeat=repeat,
        pair_stats=pair_stats,
        gpu_count=gpu_count,
        gpu_device_ids=devices,
    )
    metric.update(
        {
            "status": "skipped",
            "validation_status": "not_applicable",
            "skip_reason": reason,
            "returncode": None,
            "wall_seconds": 0.0,
            "peak_cpu_rss_mib": 0.0,
            "gpu_peak_memory_used_mib_sum": 0.0,
            "gpu_peak_incremental_memory_mib_sum": 0.0,
            "stage_output_file_count": 0,
            "stage_output_bytes": 0,
        }
    )
    return metric


def run_stage(
    *,
    run_id: str,
    stage: str,
    repeat: int,
    pair_stats: Mapping[str, int],
    gpu_count: int,
    devices: Sequence[int],
    command: Sequence[str],
    stage_monitor_dir: Path,
    measured_output_dir: Path,
    repo_root: Path,
    env: Mapping[str, str],
    monitor_interval: float,
    timeout_seconds: Optional[float],
) -> Dict[str, Any]:
    monitored = run_monitored_command(
        command=command,
        stage_dir=stage_monitor_dir,
        cwd=repo_root,
        env=env,
        gpu_device_ids=devices,
        monitor_interval_seconds=monitor_interval,
        timeout_seconds=timeout_seconds,
    )
    output_stats = directory_stats(measured_output_dir)
    metric = stage_metric_base(
        run_id=run_id,
        run_type="end_to_end",
        stage=stage,
        repeat=repeat,
        pair_stats=pair_stats,
        gpu_count=gpu_count,
        gpu_device_ids=devices,
    )
    metric.update(monitored)
    metric["stage_output_file_count"] = output_stats["file_count"]
    metric["stage_output_bytes"] = output_stats["bytes"]
    return metric


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sampled ProDive workflow end to end.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repeat", type=int, action="append", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    e2e = config["end_to_end"]
    if not bool(e2e.get("enabled", True)):
        print("end_to_end.enabled=false; no end-to-end runs were requested.")
        return

    repo_root = resolved(config, "repo_root")
    packed_db = resolved(config, "packed_db")
    output_root = resolved(config, "output_root")
    cpp_executable = resolved(config, "cpp_executable")
    mapping_file = resolved(config, "mapping_file")
    background_json = resolved(config, "background_json")
    coverage_dir = resolved(config, "coverage_dir")
    thresholds = config["thresholds"]
    monitor_interval = float(config.get("monitor", {}).get("interval_seconds", 0.5))
    timeout_value = e2e.get("timeout_seconds_per_stage")
    timeout_seconds = float(timeout_value) if timeout_value is not None else None

    for required in [cpp_executable, mapping_file, background_json]:
        if not required.is_file():
            raise FileNotFoundError(f"Required end-to-end input does not exist: {required}")
    if not coverage_dir.is_dir():
        raise FileNotFoundError(f"Coverage directory does not exist: {coverage_dir}")

    pair_count = int(e2e["pair_count"])
    pair_list = output_root / "design" / "pair_lists" / f"pairs_{pair_count}.csv"
    if not pair_list.is_file():
        raise FileNotFoundError(f"Missing prepared pair list: {pair_list}")
    pair_stats = read_pair_list_stats(pair_list)
    if pair_stats["pair_count"] != pair_count:
        raise RuntimeError(f"Prepared pair list contains {pair_stats['pair_count']} rows, expected {pair_count}")

    configured_devices = [int(x) for x in config["core"]["gpu_device_ids"]]
    gpu_count = int(e2e["gpu_count"])
    if gpu_count <= 0 or gpu_count > len(configured_devices):
        raise ValueError(f"end_to_end.gpu_count={gpu_count} is incompatible with {configured_devices}")
    devices = configured_devices[:gpu_count]
    repeats = (
        sorted(set(args.repeat))
        if args.repeat
        else list(range(1, int(e2e.get("repeats", 3)) + 1))
    )
    env = command_environment(devices)

    filter_script = repo_root / "src" / "path_extraction" / "scripts" / "01_filter_cpp_kl_results.py"
    path_script = repo_root / "src" / "path_extraction" / "scripts" / "02_build_paths_from_filtered_pk.py"
    rescore_script = repo_root / "src" / "path_extraction" / "scripts" / "03_rescore_paths_by_coverage.py"
    for script in [filter_script, path_script, rescore_script]:
        if not script.is_file():
            raise FileNotFoundError(f"ProDive workflow script not found: {script}")

    completed_now = 0
    skipped_existing = 0
    for position, repeat in enumerate(repeats, start=1):
        run_id = f"e2e_p{pair_count}_g{gpu_count}_r{repeat:02d}"
        run_dir = output_root / "runs" / "end_to_end" / run_id
        metrics_dir = run_dir / "metrics"
        summary_path = run_dir / "run_summary.json"
        print(f"[{position}/{len(repeats)}] {run_id}")

        if completed_run(summary_path):
            print(f"  already completed; skipping: {summary_path}")
            skipped_existing += 1
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                f"Incomplete/non-empty run directory already exists: {run_dir}. "
                "Move it aside or choose a new output_root before measuring again."
            )

        cpp_output = run_dir / "05_cpp_kl"
        filtered_output = run_dir / "06_filtered_pk"
        paths_output = run_dir / "07_paths"
        final_output = run_dir / "08_final_paths"
        for directory in [cpp_output, filtered_output, paths_output, final_output, metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        stage_metrics: List[Dict[str, Any]] = []
        validation_warnings: List[str] = []

        cpp_command: List[str] = [
            str(cpp_executable),
            "--pair_list",
            str(pair_list),
            "--packed_db",
            str(packed_db),
            "--num_streams",
            str(int(e2e.get("num_streams", config["core"].get("num_streams", 8)))),
            "--max_gpus",
            str(gpu_count),
            "--outdir",
            str(cpp_output),
            "--memory_efficient",
        ]
        cpp_metric = run_stage(
            run_id=run_id,
            stage="cpp_kl",
            repeat=repeat,
            pair_stats=pair_stats,
            gpu_count=gpu_count,
            devices=devices,
            command=cpp_command,
            stage_monitor_dir=run_dir / "monitor_cpp_kl",
            measured_output_dir=cpp_output,
            repo_root=repo_root,
            env=env,
            monitor_interval=monitor_interval,
            timeout_seconds=timeout_seconds,
        )
        cpp_metric.update(parse_cpp_stdout(Path(cpp_metric["stdout_log"])))
        matrix_stats = directory_stats(cpp_output, patterns=["kl_*.npy"])
        manifest_path = cpp_output / "prodive_run_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        cpp_errors = []
        if cpp_metric["returncode"] != 0:
            cpp_errors.append(f"return code {cpp_metric['returncode']}")
        if manifest.get("status") != "completed":
            cpp_errors.append(f"manifest status {manifest.get('status')!r}")
        if int(manifest.get("planned_pair_count", -1)) != pair_count:
            cpp_errors.append(f"planned_pair_count {manifest.get('planned_pair_count')!r}")
        if int(manifest.get("missing_output_count", -1)) != 0:
            cpp_errors.append(f"missing_output_count {manifest.get('missing_output_count')!r}")
        if matrix_stats["file_count"] != pair_count:
            cpp_errors.append(f"matrix file count {matrix_stats['file_count']} != {pair_count}")
        if not (cpp_output / "RUN_COMPLETE").is_file():
            cpp_errors.append("RUN_COMPLETE marker missing")
        cpp_metric.update(
            {
                "validation_status": "passed" if not cpp_errors else "failed",
                "validation_errors": cpp_errors,
                "matrix_output_file_count": matrix_stats["file_count"],
                "matrix_output_bytes": matrix_stats["bytes"],
                "cpp_manifest": str(manifest_path),
                "cpp_reported_computation_seconds": manifest.get("computation_seconds"),
                "cpp_reported_pairs_completed": manifest.get("pairs_completed"),
                "external_pairs_per_second": pair_count / cpp_metric["wall_seconds"],
                "external_window_cells_per_second": pair_stats["window_cells"] / cpp_metric["wall_seconds"],
            }
        )
        write_json(metrics_dir / "cpp_kl.json", cpp_metric)
        stage_metrics.append(cpp_metric)
        if cpp_errors:
            write_json(summary_path, {"status": "failed", "stage": "cpp_kl", "errors": cpp_errors})
            raise RuntimeError(f"End-to-end C++ stage failed validation for {run_id}: {cpp_errors}")

        filter_command = [
            sys.executable,
            str(filter_script),
            "--input-dir",
            str(cpp_output),
            "--output-dir",
            str(filtered_output),
            "--mapping-file",
            str(mapping_file),
            "--background-json",
            str(background_json),
            "--fragment",
            str(int(thresholds.get("fragment", 6))),
            "--raw-threshold",
            str(float(thresholds.get("raw_threshold", 2.0))),
            "--filter-threshold",
            str(float(thresholds.get("filter_threshold", 13.0))),
            "--epsilon",
            str(float(thresholds.get("epsilon", 0.0001))),
            "--row-block-size",
            str(int(thresholds.get("row_block_size", 1024))),
        ]
        filter_metric = run_stage(
            run_id=run_id,
            stage="filter_kl",
            repeat=repeat,
            pair_stats=pair_stats,
            gpu_count=gpu_count,
            devices=devices,
            command=filter_command,
            stage_monitor_dir=run_dir / "monitor_filter_kl",
            measured_output_dir=filtered_output,
            repo_root=repo_root,
            env=env,
            monitor_interval=monitor_interval,
            timeout_seconds=timeout_seconds,
        )
        filter_summary_path = filtered_output / "filter_cpp_kl_run_summary.json"
        filter_summary = read_json(filter_summary_path) if filter_summary_path.is_file() else {}
        filter_errors = []
        if filter_metric["returncode"] != 0:
            filter_errors.append(f"return code {filter_metric['returncode']}")
        if int(filter_summary.get("missing_expected_cpp_outputs_count", -1)) != 0:
            filter_errors.append(
                f"missing_expected_cpp_outputs_count={filter_summary.get('missing_expected_cpp_outputs_count')}"
            )
        if int(filter_summary.get("failed_pairs", -1)) != 0:
            filter_errors.append(f"failed_pairs={filter_summary.get('failed_pairs')}")
        if int(filter_summary.get("skipped_pairs", -1)) != 0:
            filter_errors.append(f"skipped_pairs={filter_summary.get('skipped_pairs')}")
        if int(filter_summary.get("ok_pairs", -1)) != pair_count:
            filter_errors.append(
                f"ok_pairs={filter_summary.get('ok_pairs')}, expected {pair_count}"
            )
        if int(filter_summary.get("selected_matrix_files", -1)) != pair_count:
            filter_errors.append(
                f"selected_matrix_files={filter_summary.get('selected_matrix_files')}, expected {pair_count}"
            )
        filter_metric.update(
            {
                "validation_status": "passed" if not filter_errors else "failed",
                "validation_errors": filter_errors,
                "filter_summary_json": str(filter_summary_path),
                "filter_selected_matrix_files": filter_summary.get("selected_matrix_files"),
                        "filter_ok_pairs": filter_summary.get("ok_pairs"),
                        "filter_failed_pairs": filter_summary.get("failed_pairs"),
                        "filter_skipped_pairs": filter_summary.get("skipped_pairs"),
                "filter_query_output_files": len(filter_summary.get("query_outputs_written", {})),
                "filter_total_input_points": filter_summary.get("total_input_points_in_usable_shape"),
                "filter_total_raw_pass_points": filter_summary.get("total_raw_pass_points"),
                "filter_total_second_pass_points": filter_summary.get("total_second_pass_points"),
            }
        )
        write_json(metrics_dir / "filter_kl.json", filter_metric)
        stage_metrics.append(filter_metric)
        if filter_errors:
            write_json(summary_path, {"status": "failed", "stage": "filter_kl", "errors": filter_errors})
            raise RuntimeError(f"End-to-end filtering stage failed validation for {run_id}: {filter_errors}")

        filtered_pickles = directory_stats(filtered_output, patterns=["kl_*_filtered.pk"])
        if filtered_pickles["file_count"] == 0:
            reason = "Filtering produced no kl_*_filtered.pk files for this sampled workload."
            path_metric = skipped_metric(run_id, "build_paths", repeat, pair_stats, gpu_count, devices, reason)
            rescore_metric = skipped_metric(run_id, "rescore_paths", repeat, pair_stats, gpu_count, devices, reason)
            write_json(metrics_dir / "build_paths.json", path_metric)
            write_json(metrics_dir / "rescore_paths.json", rescore_metric)
            stage_metrics.extend([path_metric, rescore_metric])
            validation_warnings.append(reason)
            final_path_count = 0
        else:
            path_command = [
                sys.executable,
                str(path_script),
                "--input-dir",
                str(filtered_output),
                "--output-dir",
                str(paths_output),
                "--mapping-file",
                str(mapping_file),
                "--fragment",
                str(int(thresholds.get("fragment", 6))),
                "--min-path-points",
                str(int(thresholds.get("min_path_points", 5))),
                "--overlap-iou-threshold",
                str(float(thresholds.get("overlap_iou_threshold", 0.1))),
                "--workers",
                str(int(e2e.get("path_workers", 20))),
            ]
            if bool(e2e.get("path_strict", True)):
                path_command.append("--strict")
            path_metric = run_stage(
                run_id=run_id,
                stage="build_paths",
                repeat=repeat,
                pair_stats=pair_stats,
                gpu_count=gpu_count,
                devices=devices,
                command=path_command,
                stage_monitor_dir=run_dir / "monitor_build_paths",
                measured_output_dir=paths_output,
                repo_root=repo_root,
                env=env,
                monitor_interval=monitor_interval,
                timeout_seconds=timeout_seconds,
            )
            path_summary_path = paths_output / "path_building_summary.json"
            path_summary = read_json(path_summary_path) if path_summary_path.is_file() else {}
            path_errors = []
            if path_metric["returncode"] != 0:
                path_errors.append(f"return code {path_metric['returncode']}")
            if int(path_summary.get("failed_files", -1)) != 0:
                path_errors.append(f"failed_files={path_summary.get('failed_files')}")
            path_metric.update(
                {
                    "validation_status": "passed" if not path_errors else "failed",
                    "validation_errors": path_errors,
                    "path_summary_json": str(path_summary_path),
                    "path_filtered_pk_files": path_summary.get("filtered_pk_files"),
                    "clean_paths": path_summary.get("clean_paths", 0),
                    "unclean_paths": path_summary.get("unclean_paths", 0),
                    "gap_paths": path_summary.get("gap_paths", 0),
                    "path_failed_files": path_summary.get("failed_files"),
                    "path_target_error_count": path_summary.get("target_error_count"),
                }
            )
            write_json(metrics_dir / "build_paths.json", path_metric)
            stage_metrics.append(path_metric)
            if path_errors:
                write_json(summary_path, {"status": "failed", "stage": "build_paths", "errors": path_errors})
                raise RuntimeError(f"End-to-end path stage failed validation for {run_id}: {path_errors}")

            clean_csv = paths_output / "global_high_score_summary.csv"
            if not clean_csv.is_file() or count_csv_rows(clean_csv) == 0:
                reason = "Path construction produced zero clean paths for this sampled workload."
                rescore_metric = skipped_metric(
                    run_id, "rescore_paths", repeat, pair_stats, gpu_count, devices, reason
                )
                write_json(metrics_dir / "rescore_paths.json", rescore_metric)
                stage_metrics.append(rescore_metric)
                validation_warnings.append(reason)
                final_path_count = 0
            else:
                final_csv = final_output / "final_paths.csv"
                rescored_all_csv = final_output / "all_paths_rescored.csv"
                score_plot = final_output / "score_distribution_before_after.png"
                summary_json = final_output / "final_paths.summary.json"
                rescore_command = [
                    sys.executable,
                    str(rescore_script),
                    "--input-csv",
                    str(clean_csv),
                    "--coverage-dir",
                    str(coverage_dir),
                    "--output-csv",
                    str(final_csv),
                    "--rescored-all-csv",
                    str(rescored_all_csv),
                    "--summary-json",
                    str(summary_json),
                    "--coverage-threshold",
                    str(float(thresholds.get("coverage_threshold", 0.85))),
                    "--score-threshold",
                    str(float(thresholds.get("score_threshold", 1.2))),
                    "--missing-coverage-default",
                    str(float(thresholds.get("missing_coverage_default", 0.0))),
                ]
                if bool(e2e.get("make_score_distribution_plot", True)):
                    rescore_command.extend(["--score-distribution-png", str(score_plot)])
                rescore_metric = run_stage(
                    run_id=run_id,
                    stage="rescore_paths",
                    repeat=repeat,
                    pair_stats=pair_stats,
                    gpu_count=gpu_count,
                    devices=devices,
                    command=rescore_command,
                    stage_monitor_dir=run_dir / "monitor_rescore_paths",
                    measured_output_dir=final_output,
                    repo_root=repo_root,
                    env=env,
                    monitor_interval=monitor_interval,
                    timeout_seconds=timeout_seconds,
                )
                rescore_summary = read_json(summary_json) if summary_json.is_file() else {}
                rescore_errors = []
                if rescore_metric["returncode"] != 0:
                    rescore_errors.append(f"return code {rescore_metric['returncode']}")
                if not final_csv.is_file():
                    rescore_errors.append("final_paths.csv was not created")
                if int(rescore_summary.get("parse_failures", -1)) != 0:
                    rescore_errors.append(
                        f"parse_failures={rescore_summary.get('parse_failures')}"
                    )
                if int(rescore_summary.get("coverage_records_missing_or_failed", -1)) != 0:
                    rescore_errors.append(
                        "coverage_records_missing_or_failed="
                        f"{rescore_summary.get('coverage_records_missing_or_failed')}"
                    )
                final_path_count = count_csv_rows(final_csv)
                expected_final = rescore_summary.get("rows_after_final_score_filter")
                if expected_final is not None and int(expected_final) != final_path_count:
                    rescore_errors.append(
                        f"final CSV rows={final_path_count}, summary rows={expected_final}"
                    )
                rescore_metric.update(
                    {
                        "validation_status": "passed" if not rescore_errors else "failed",
                        "validation_errors": rescore_errors,
                        "rescore_summary_json": str(summary_json),
                        "rescore_input_rows": rescore_summary.get("input_rows"),
                        "rescore_parse_failures": rescore_summary.get("parse_failures"),
                        "coverage_records_loaded": rescore_summary.get("coverage_records_loaded"),
                        "coverage_records_missing_or_failed": rescore_summary.get(
                            "coverage_records_missing_or_failed"
                        ),
                        "rows_after_coverage_filter": rescore_summary.get("rows_after_coverage_filter"),
                        "final_path_count": final_path_count,
                        "final_paths_csv": str(final_csv),
                    }
                )
                write_json(metrics_dir / "rescore_paths.json", rescore_metric)
                stage_metrics.append(rescore_metric)
                if rescore_errors:
                    write_json(
                        summary_path,
                        {"status": "failed", "stage": "rescore_paths", "errors": rescore_errors},
                    )
                    raise RuntimeError(
                        f"End-to-end rescoring stage failed validation for {run_id}: {rescore_errors}"
                    )

        cleanup_results: Dict[str, Any] = {}
        cleanup_config = config.get("cleanup", {})
        if bool(cleanup_config.get("end_to_end_cpp_matrices_after_success", True)):
            cleanup_results["cpp_matrices"] = cleanup_globs(cpp_output, ["kl_*.npy"])
        if bool(cleanup_config.get("end_to_end_filtered_pickles_after_success", True)):
            cleanup_results["filtered_pickles"] = cleanup_globs(filtered_output, ["kl_*_filtered.pk"])

        total_wall = sum(float(metric.get("wall_seconds", 0.0) or 0.0) for metric in stage_metrics)
        overall_status = "completed" if final_path_count > 0 else "completed_no_final_paths"
        run_summary = {
            "status": overall_status,
            "run_id": run_id,
            "run_type": "end_to_end",
            "repeat": repeat,
            "pair_count": pair_count,
            "window_cells": pair_stats["window_cells"],
            "unique_families": pair_stats["unique_families"],
            "gpu_count": gpu_count,
            "gpu_device_ids": devices,
            "total_measured_stage_wall_seconds": total_wall,
            "final_path_count": final_path_count,
            "validation_warnings": validation_warnings,
            "cleanup": cleanup_results,
            "stage_metric_files": [str(metrics_dir / f"{metric['stage']}.json") for metric in stage_metrics],
            "completed_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        write_json(summary_path, run_summary)
        completed_now += 1
        print(
            f"  total measured stage time={total_wall:.3f}s, final paths={final_path_count}, "
            f"status={overall_status}"
        )

    print(
        f"End-to-end benchmark finished: {completed_now} new run(s), "
        f"{skipped_existing} existing run(s) skipped."
    )
    print(f"Run root: {output_root / 'runs' / 'end_to_end'}")


if __name__ == "__main__":
    main()
