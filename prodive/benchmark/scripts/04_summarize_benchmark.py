#!/usr/bin/env python3
"""Aggregate ProDive benchmark measurements into tables, figures, and a report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_common import load_config, read_json, resolved, write_json


STAGE_ORDER = ["cpp_kl", "filter_kl", "build_paths", "rescore_paths"]
STAGE_LABELS = {
    "cpp_kl": "C++/CUDA KL",
    "filter_kl": "KL filtering",
    "build_paths": "Path construction",
    "rescore_paths": "Coverage rescoring",
}


def flatten_metric(metric: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in metric.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            flat[key] = value
    return flat


def read_stage_metrics(output_root: Path) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for path in sorted((output_root / "runs" / "core").glob("*/stage_metrics.json")):
        try:
            metrics.append(read_json(path))
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}")
    for path in sorted((output_root / "runs" / "end_to_end").glob("*/metrics/*.json")):
        try:
            metrics.append(read_json(path))
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}")
    for path in sorted(
        (output_root / "runs" / "cpu_gpu_validation").glob("*/metrics/*.json")
    ):
        try:
            metrics.append(read_json(path))
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}")
    return metrics


def read_e2e_summaries(output_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((output_root / "runs" / "end_to_end").glob("*/run_summary.json")):
        try:
            row = read_json(path)
            row["run_summary_json"] = str(path)
            rows.append(row)
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}")
    return rows


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def summary_stats(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if array.size == 0:
        return {"mean": None, "std": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def make_core_summary(core: pd.DataFrame) -> pd.DataFrame:
    valid = core[
        (core.get("status") == "completed")
        & (core.get("validation_status") == "passed")
    ].copy()
    if valid.empty:
        return pd.DataFrame()
    if "sample_kind" not in valid.columns:
        valid["sample_kind"] = "stratified_workload"
    valid["sample_kind"] = valid["sample_kind"].fillna("stratified_workload")
    numeric_columns = [
        "wall_seconds",
        "peak_cpu_rss_mib",
        "gpu_peak_memory_used_mib_sum",
        "gpu_peak_incremental_memory_mib_sum",
        "gpu_mean_utilization_percent",
        "gpu_energy_wh_estimate",
        "external_pairs_per_second",
        "external_window_cells_per_second",
        "cpp_cpu_matrix_inversion_seconds",
        "cpp_reported_computation_seconds",
        "matrix_output_bytes",
    ]
    for column in numeric_columns + ["pair_count", "gpu_count", "window_cells", "unique_families"]:
        if column in valid:
            valid[column] = numeric(valid[column])

    rows: List[Dict[str, Any]] = []
    for (sample_kind, pair_count, gpu_count), group in valid.groupby(
        ["sample_kind", "pair_count", "gpu_count"], dropna=False
    ):
        row: Dict[str, Any] = {
            "sample_kind": str(sample_kind),
            "pair_count": int(pair_count),
            "gpu_count": int(gpu_count),
            "repeat_count": int(len(group)),
            "window_cells": int(group["window_cells"].dropna().iloc[0]),
            "unique_families": int(group["unique_families"].dropna().iloc[0]),
        }
        for column in numeric_columns:
            stats = summary_stats(group[column]) if column in group else summary_stats([])
            for stat_name, value in stats.items():
                row[f"{column}_{stat_name}"] = value
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["sample_kind", "pair_count", "gpu_count"]).reset_index(drop=True)
    baseline = {
        (str(row.sample_kind), int(row.pair_count)): float(row.wall_seconds_median)
        for row in result.itertuples()
        if int(row.gpu_count) == 1 and pd.notna(row.wall_seconds_median)
    }
    result["speedup_vs_1_gpu"] = [
        baseline.get((str(row.sample_kind), int(row.pair_count)), np.nan) / float(row.wall_seconds_median)
        if (str(row.sample_kind), int(row.pair_count)) in baseline and float(row.wall_seconds_median) > 0
        else np.nan
        for row in result.itertuples()
    ]
    result["parallel_efficiency_vs_1_gpu"] = result["speedup_vs_1_gpu"] / result["gpu_count"]
    return result


def make_e2e_stage_summary(e2e: pd.DataFrame) -> pd.DataFrame:
    if e2e.empty:
        return pd.DataFrame()
    rows = []
    for stage in STAGE_ORDER:
        group = e2e[e2e["stage"] == stage]
        if group.empty:
            continue
        completed = group[group["status"] == "completed"]
        row: Dict[str, Any] = {
            "stage": stage,
            "stage_label": STAGE_LABELS[stage],
            "runs_observed": int(len(group)),
            "completed_runs": int(len(completed)),
            "skipped_runs": int((group["status"] == "skipped").sum()),
            "failed_runs": int((group["status"].isin(["failed", "timed_out"])).sum()),
        }
        for column in [
            "wall_seconds",
            "peak_cpu_rss_mib",
            "gpu_peak_incremental_memory_mib_sum",
            "stage_output_bytes",
        ]:
            stats = summary_stats(completed[column]) if column in completed else summary_stats([])
            for stat_name, value in stats.items():
                row[f"{column}_{stat_name}"] = value
        for count_column in [
            "filter_total_input_points",
            "filter_total_raw_pass_points",
            "filter_total_second_pass_points",
            "clean_paths",
            "unclean_paths",
            "gap_paths",
            "rows_after_coverage_filter",
            "final_path_count",
        ]:
            row[f"{count_column}_median"] = summary_stats(completed.get(count_column, []))["median"]
        rows.append(row)
    return pd.DataFrame(rows)


def make_cpu_gpu_validation_summary(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for run_id, group in validation.groupby("run_id"):
        by_stage = {str(row["stage"]): row for _, row in group.iterrows()}
        cpu = by_stage.get("cpu_raw_kl")
        gpu = by_stage.get("gpu_raw_kl")
        comparison = by_stage.get("raw_kl_comparison")
        if cpu is None or gpu is None or comparison is None:
            continue

        def value(row: pd.Series, key: str) -> Any:
            result = row.get(key)
            return None if pd.isna(result) else result

        cpu_wall = float(value(cpu, "wall_seconds"))
        gpu_wall = float(value(gpu, "wall_seconds"))
        cpu_compute = value(cpu, "cpu_internal_matrix_compute_seconds")
        gpu_compute = value(gpu, "cpp_reported_computation_seconds")
        cpu_compute = float(cpu_compute) if cpu_compute is not None else None
        gpu_compute = float(gpu_compute) if gpu_compute is not None else None
        rows.append(
            {
                "run_id": str(run_id),
                "pair_count": int(value(cpu, "pair_count")),
                "window_cells": int(value(cpu, "window_cells")),
                "cpu_mode": value(cpu, "cpu_mode"),
                "cpu_wall_seconds": cpu_wall,
                "gpu_wall_seconds": gpu_wall,
                "gpu_speedup_vs_serial_cpu": (
                    cpu_wall / gpu_wall if gpu_wall > 0 else None
                ),
                "cpu_internal_matrix_compute_seconds": cpu_compute,
                "gpu_reported_computation_seconds": gpu_compute,
                "gpu_raw_compute_speedup_vs_serial_cpu": (
                    cpu_compute / gpu_compute
                    if cpu_compute is not None
                    and gpu_compute is not None
                    and gpu_compute > 0
                    else None
                ),
                "cpu_internal_matrix_inversion_seconds": value(
                    cpu, "cpu_internal_matrix_inversion_seconds"
                ),
                "gpu_cpp_matrix_inversion_seconds": value(
                    gpu, "cpp_cpu_matrix_inversion_seconds"
                ),
                "cpu_peak_rss_mib": value(cpu, "peak_cpu_rss_mib"),
                "gpu_stage_peak_cpu_rss_mib": value(gpu, "peak_cpu_rss_mib"),
                "gpu_peak_incremental_memory_mib_sum": value(
                    gpu, "gpu_peak_incremental_memory_mib_sum"
                ),
                "rtol": value(comparison, "rtol"),
                "atol": value(comparison, "atol"),
                "allclose_pair_fraction": value(
                    comparison, "allclose_pair_fraction"
                ),
                "pooled_mismatch_fraction": value(
                    comparison, "pooled_mismatch_fraction"
                ),
                "maximum_pair_absolute_error": value(
                    comparison, "maximum_pair_absolute_error"
                ),
                "pooled_mean_absolute_error": value(
                    comparison, "pooled_mean_absolute_error"
                ),
                "pooled_rmse": value(comparison, "pooled_rmse"),
                "pooled_pearson_r": value(comparison, "pooled_pearson_r"),
                "numerical_acceptance_passed": value(
                    comparison, "numerical_acceptance_passed"
                ),
                "comparison_scope": "raw symmetric KL only",
            }
        )
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True) if rows else pd.DataFrame()


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def projected_full_scale(core: pd.DataFrame, full_workload: Mapping[str, Any]) -> pd.DataFrame:
    valid = core[
        (core.get("status") == "completed")
        & (core.get("validation_status") == "passed")
    ].copy()
    if valid.empty:
        return pd.DataFrame()
    if "sample_kind" in valid.columns:
        valid = valid[
            valid["sample_kind"].fillna("stratified_workload") == "stratified_workload"
        ].copy()
    if valid.empty:
        return pd.DataFrame()
    for column in ["pair_count", "gpu_count", "external_window_cells_per_second"]:
        valid[column] = numeric(valid[column])
    full_cells = float(full_workload["window_cells_unordered_excluding_self"])
    full_pairs = int(full_workload["unordered_pairs_excluding_self"])
    rows = []
    for gpu_count, gpu_group in valid.groupby("gpu_count"):
        largest_pair_count = int(gpu_group["pair_count"].max())
        largest = gpu_group[gpu_group["pair_count"] == largest_pair_count]
        throughput = numeric(largest["external_window_cells_per_second"])
        throughput = throughput[np.isfinite(throughput) & (throughput > 0)]
        if throughput.empty:
            continue
        seconds = full_cells / throughput.to_numpy(dtype=float)
        rows.append(
            {
                "gpu_count": int(gpu_count),
                "benchmark_pair_count_used": largest_pair_count,
                "repeat_count": int(len(seconds)),
                "full_unordered_nonself_pairs": full_pairs,
                "full_unordered_nonself_window_cells": int(full_cells),
                "observed_window_cells_per_second_median": float(np.median(throughput)),
                "projected_full_seconds_median": float(np.median(seconds)),
                "projected_full_seconds_q25": float(np.percentile(seconds, 25)),
                "projected_full_seconds_q75": float(np.percentile(seconds, 75)),
                "projected_full_days_median": float(np.median(seconds) / 86400.0),
                "estimate_type": "rough throughput extrapolation; not a full-scale measurement",
                "limitations": (
                    "Assumes the largest sampled run's end-to-end C++-stage window-cell throughput "
                    "persists at full scale. Batch setup, file-system contention, output-file count, "
                    "and scheduling can make the actual full run differ substantially."
                ),
            }
        )
    return pd.DataFrame(rows)


def save_core_runtime_chart(core_summary: pd.DataFrame, output: Path) -> None:
    if core_summary.empty:
        return
    core_summary = core_summary[
        core_summary["sample_kind"] == "stratified_workload"
    ].copy()
    if core_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for gpu_count, group in core_summary.groupby("gpu_count"):
        group = group.sort_values("window_cells")
        x = group["window_cells"].to_numpy(dtype=float)
        y = group["wall_seconds_median"].to_numpy(dtype=float)
        lower = y - group["wall_seconds_q25"].to_numpy(dtype=float)
        upper = group["wall_seconds_q75"].to_numpy(dtype=float) - y
        ax.errorbar(x, y, yerr=np.vstack([lower, upper]), marker="o", capsize=3, label=f"{int(gpu_count)} GPU")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sampled window cells, Σ(nfragᵢ × nfragⱼ)")
    ax.set_ylabel("Whole C++ stage wall time (s)")
    ax.set_title("ProDive C++/CUDA runtime scaling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_throughput_chart(core_summary: pd.DataFrame, output: Path) -> None:
    if core_summary.empty:
        return
    core_summary = core_summary[
        core_summary["sample_kind"] == "stratified_workload"
    ].copy()
    if core_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for gpu_count, group in core_summary.groupby("gpu_count"):
        group = group.sort_values("window_cells")
        ax.plot(
            group["window_cells"],
            group["external_window_cells_per_second_median"],
            marker="o",
            label=f"{int(gpu_count)} GPU",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sampled window cells, Σ(nfragᵢ × nfragⱼ)")
    ax.set_ylabel("Window cells per second")
    ax.set_title("ProDive observed throughput")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_memory_chart(core_summary: pd.DataFrame, output: Path) -> None:
    if core_summary.empty:
        return
    coverage = core_summary[core_summary["sample_kind"] == "full_family_coverage"]
    if not coverage.empty:
        largest = coverage.sort_values("gpu_count")
        title = "Peak memory in the full-family coverage sample"
    else:
        stratified = core_summary[core_summary["sample_kind"] == "stratified_workload"]
        largest = stratified[stratified["pair_count"] == stratified["pair_count"].max()].sort_values("gpu_count")
        title = f"Peak memory at {int(largest['pair_count'].max()):,} sampled pairs"
    if largest.empty:
        return
    labels = [f"{int(value)} GPU" for value in largest["gpu_count"]]
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - width / 2, largest["peak_cpu_rss_mib_median"], width, label="CPU process-tree RSS")
    ax.bar(
        x + width / 2,
        largest["gpu_peak_incremental_memory_mib_sum_median"],
        width,
        label="Incremental GPU memory (sum)",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Peak memory (MiB)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_e2e_chart(e2e_summary: pd.DataFrame, output: Path) -> None:
    if e2e_summary.empty:
        return
    completed = e2e_summary[e2e_summary["completed_runs"] > 0].copy()
    if completed.empty:
        return
    completed["order"] = completed["stage"].map({name: idx for idx, name in enumerate(STAGE_ORDER)})
    completed = completed.sort_values("order")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(completed["stage_label"], completed["wall_seconds_median"])
    ax.set_ylabel("Median wall time (s)")
    ax.set_title("ProDive sampled end-to-end stage breakdown")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_cpu_gpu_runtime_chart(validation_summary: pd.DataFrame, output: Path) -> None:
    if validation_summary.empty:
        return
    row = validation_summary.iloc[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    labels = ["Serial CPU", "GPU stage"]
    values = [float(row["cpu_wall_seconds"]), float(row["gpu_wall_seconds"])]
    bars = ax.bar(labels, values)
    ax.set_ylabel("Whole-stage wall time (s)")
    ax.set_title(
        f"Raw symmetric-KL comparison ({int(row['pair_count'])} matched pairs)"
    )
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value_ in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value_:.3g}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    if abs(number) >= 100000:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_No completed measurements are available._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    config: Mapping[str, Any],
    hardware: Mapping[str, Any],
    sampling: Mapping[str, Any],
    full_workload: Mapping[str, Any],
    core_summary: pd.DataFrame,
    cpu_gpu_validation_summary: pd.DataFrame,
    e2e_stage_summary: pd.DataFrame,
    e2e_runs: pd.DataFrame,
    estimates: pd.DataFrame,
) -> str:
    gpus = hardware.get("gpus", [])
    gpu_text = ", ".join(
        f"GPU {row.get('index')}: {row.get('name')} ({format_value(row.get('memory_total_mib'), 0)} MiB)"
        for row in gpus
    ) or "GPU inventory unavailable"
    pair_counts = sampling.get("pair_counts", [])
    repeats = int(config.get("core", {}).get("repeats", 0))
    interval = float(config.get("monitor", {}).get("interval_seconds", 0.5))

    core_rows = []
    if not core_summary.empty:
        for row in core_summary.itertuples():
            core_rows.append(
                [
                    "Full-family coverage" if row.sample_kind == "full_family_coverage" else "Stratified scaling",
                    f"{int(row.pair_count):,}",
                    int(row.gpu_count),
                    int(row.repeat_count),
                    format_value(row.wall_seconds_median),
                    format_value(row.external_window_cells_per_second_median),
                    format_value(row.peak_cpu_rss_mib_median),
                    format_value(row.gpu_peak_incremental_memory_mib_sum_median),
                    format_value(row.speedup_vs_1_gpu),
                ]
            )

    e2e_rows = []
    if not e2e_stage_summary.empty:
        for row in e2e_stage_summary.itertuples():
            e2e_rows.append(
                [
                    row.stage_label,
                    int(row.completed_runs),
                    int(row.skipped_runs),
                    format_value(row.wall_seconds_median),
                    format_value(row.peak_cpu_rss_mib_median),
                    format_value(row.gpu_peak_incremental_memory_mib_sum_median),
                ]
            )

    final_rows = []
    if not e2e_runs.empty:
        for row in e2e_runs.itertuples():
            final_rows.append(
                [
                    row.run_id,
                    row.status,
                    format_value(row.total_measured_stage_wall_seconds),
                    int(row.final_path_count),
                ]
            )

    estimate_rows = []
    if not estimates.empty:
        for row in estimates.itertuples():
            estimate_rows.append(
                [
                    int(row.gpu_count),
                    f"{int(row.benchmark_pair_count_used):,}",
                    format_value(row.observed_window_cells_per_second_median),
                    format_value(row.projected_full_days_median),
                ]
            )

    validation_rows = []
    if not cpu_gpu_validation_summary.empty:
        for row in cpu_gpu_validation_summary.itertuples():
            validation_rows.append(
                [
                    row.run_id,
                    f"{int(row.pair_count):,}",
                    f"{int(row.window_cells):,}",
                    format_value(row.cpu_wall_seconds),
                    format_value(row.gpu_wall_seconds),
                    format_value(row.gpu_speedup_vs_serial_cpu),
                    format_value(row.cpu_internal_matrix_compute_seconds),
                    format_value(row.gpu_reported_computation_seconds),
                    format_value(row.gpu_raw_compute_speedup_vs_serial_cpu),
                    format_value(row.cpu_peak_rss_mib),
                    format_value(row.gpu_peak_incremental_memory_mib_sum),
                    format_value(row.pooled_mismatch_fraction, 6),
                    format_value(row.pooled_rmse, 6),
                    format_value(row.pooled_pearson_r, 6),
                    str(row.numerical_acceptance_passed),
                ]
            )

    python_text = str(hardware.get("python_version") or "unknown").split()[0]

    return f"""# ProDive computational benchmark report

## Scope

This report measures the unchanged production workflow on deterministic sampled family pairs.  It reports the complete C++/CUDA stage (packed-database loading, CPU matrix inversion, GPU calculation, and NPY writing) separately from filtering, path construction, and coverage rescoring.  HHM parsing and packed-database construction are preprocessing and are not included in the timed end-to-end path.

## Hardware and software

- Host: `{hardware.get('hostname', 'unknown')}`
- CPU: {hardware.get('cpu_model') or 'unavailable'}
- Logical CPUs: {hardware.get('logical_cpu_count')}
- RAM: {format_value((hardware.get('total_ram_bytes') or 0) / (1024 ** 3), 2)} GiB
- GPU(s): {gpu_text}
- Python: {python_text}
- Monitor interval: {interval:g} s

## Sampling design

- Fixed random seed: `{sampling.get('seed')}`
- Pair counts: {', '.join(f'{int(value):,}' for value in pair_counts)}
- Repeats per core configuration: {repeats}
- Candidate pairs: {int(sampling.get('candidate_pool_size', 0)):,}
- Workload strata: {int(sampling.get('workload_strata', 0))}
- Sampling unit: unordered, non-self family pair
- Workload unit: `nfrag_i × nfrag_j` KL-matrix cells per family pair
- Full database families: {int(full_workload.get('family_count', 0)):,}
- Full unordered non-self pairs: {int(full_workload.get('unordered_pairs_excluding_self', 0)):,}
- Full unordered non-self window cells: {int(full_workload.get('window_cells_unordered_excluding_self', 0)):,}

The sampled pair lists are nested prefixes.  Candidate pairs were divided into quantile strata according to `nfrag_i × nfrag_j`, shuffled with the fixed seed, and interleaved across strata.  This avoids treating one small-family pair as computationally equivalent to one large-family pair.

## Core C++/CUDA results

{markdown_table(
    ['Sample', 'Pairs', 'GPUs', 'Repeats', 'Wall time median (s)', 'Window cells/s median', 'Peak CPU RSS (MiB)', 'Incremental GPU memory (MiB)', 'Speedup vs 1 GPU'],
    core_rows,
)}

The wall time above is measured outside the executable and therefore includes packed-data loading, CPU matrix inversion, GPU computation, synchronization, and NPY output.  The raw resource samples and the executable's own `computation_seconds` and matrix-inversion timing are retained separately.  The full-family coverage sample uses approximately half as many pairs as families and touches every packed family at least once; it estimates the full-family loading/inversion memory requirement without all-pairs computation and is not used for runtime-scaling extrapolation.

## Serial CPU versus GPU raw-KL validation

{markdown_table(
    ['Run', 'Pairs', 'Window cells', 'CPU wall (s)', 'GPU wall (s)', 'Whole-stage speedup', 'CPU raw compute (s)', 'GPU reported compute (s)', 'Raw-compute speedup', 'CPU peak RSS (MiB)', 'GPU incremental memory (MiB)', 'Mismatch fraction', 'RMSE', 'Pearson r', 'Acceptance'],
    validation_rows,
)}

This comparison uses exactly the same deterministic family pairs and compares only `0.5 × (KL(1||2) + KL(2||1))` in the GPU-compatible matrix orientation.  It does **not** divide by `(3 × fragment + 2)` and does not include background normalization, thresholding, path construction, or coverage rescoring.  The CPU implementation is forced to one BLAS/OpenMP/Numba thread.  The GPU wall time includes the complete unchanged C++/CUDA stage; the CPU wall time includes HHM loading, fragment construction, CPU inversion, raw-KL calculation, and NPY writing, so the comparison must be described with that boundary.

## End-to-end stage results

{markdown_table(
    ['Stage', 'Completed repeats', 'Skipped repeats', 'Wall time median (s)', 'Peak CPU RSS (MiB)', 'Incremental GPU memory (MiB)'],
    e2e_rows,
)}

### Final path outputs by repeat

{markdown_table(['Run', 'Status', 'Measured stage time (s)', 'Final paths'], final_rows)}

If a sampled workload produces no filtered pickle or no clean path, the downstream stage is recorded as skipped rather than assigned a fabricated runtime.  Increase `end_to_end.pair_count` if this occurs in all repeats.

## Full-scale workload projection

{markdown_table(['GPUs', 'Sample pairs used', 'Observed window cells/s', 'Projected days'], estimate_rows)}

These values are throughput extrapolations, not full-scale measurements. Full-run batching, file-system contention, scheduling, setup costs, and output-file count can change the realized runtime.

## Reporting cautions

- Report medians and the interquartile range across repeats; do not select the fastest repeat.
- State whether GPU memory is total used memory or incremental memory above the pre-run baseline.  This report provides both in the raw metrics.
- Report window cells per second alongside pairs per second.
- Describe the full-scale duration only as an extrapolation unless a full run was actually measured.
- Run on reserved GPUs and an otherwise idle host; external GPU jobs contaminate memory and utilization measurements.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ProDive benchmark outputs.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = resolved(config, "output_root")
    results_dir = output_root / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_stage_metrics(output_root)
    flattened = [flatten_metric(metric) for metric in metrics]
    raw_df = pd.DataFrame(flattened)
    write_dataframe(raw_df, tables_dir / "all_stage_metrics.csv")

    if raw_df.empty:
        core_df = pd.DataFrame()
        e2e_df = pd.DataFrame()
        validation_df = pd.DataFrame()
    else:
        core_df = raw_df[raw_df["run_type"] == "core"].copy()
        e2e_df = raw_df[raw_df["run_type"] == "end_to_end"].copy()
        validation_df = raw_df[raw_df["run_type"] == "cpu_gpu_validation"].copy()

    core_summary = make_core_summary(core_df) if not core_df.empty else pd.DataFrame()
    e2e_stage_summary = make_e2e_stage_summary(e2e_df) if not e2e_df.empty else pd.DataFrame()
    cpu_gpu_validation_summary = (
        make_cpu_gpu_validation_summary(validation_df)
        if not validation_df.empty
        else pd.DataFrame()
    )
    write_dataframe(core_summary, tables_dir / "core_benchmark_summary.csv")
    write_dataframe(e2e_stage_summary, tables_dir / "end_to_end_stage_summary.csv")
    write_dataframe(
        cpu_gpu_validation_summary,
        tables_dir / "cpu_gpu_raw_kl_validation_summary.csv",
    )

    e2e_run_rows = read_e2e_summaries(output_root)
    e2e_runs = pd.DataFrame([flatten_metric(row) for row in e2e_run_rows])
    write_dataframe(e2e_runs, tables_dir / "end_to_end_run_summary.csv")

    full_workload_path = output_root / "design" / "full_workload.json"
    sampling_path = output_root / "design" / "sampling_manifest.json"
    hardware_path = output_root / "hardware_inventory.json"
    full_workload = read_json(full_workload_path) if full_workload_path.is_file() else {}
    sampling = read_json(sampling_path) if sampling_path.is_file() else {}
    hardware = read_json(hardware_path) if hardware_path.is_file() else {}
    estimates = projected_full_scale(core_df, full_workload) if not core_df.empty and full_workload else pd.DataFrame()
    write_dataframe(estimates, tables_dir / "full_scale_throughput_projection.csv")

    save_core_runtime_chart(core_summary, figures_dir / "runtime_vs_window_cells.png")
    save_throughput_chart(core_summary, figures_dir / "throughput_vs_window_cells.png")
    save_memory_chart(core_summary, figures_dir / "peak_memory_largest_sample.png")
    save_e2e_chart(e2e_stage_summary, figures_dir / "end_to_end_stage_breakdown.png")
    save_cpu_gpu_runtime_chart(
        cpu_gpu_validation_summary,
        figures_dir / "cpu_gpu_raw_kl_runtime.png",
    )

    report = build_report(
        config=config,
        hardware=hardware,
        sampling=sampling,
        full_workload=full_workload,
        core_summary=core_summary,
        cpu_gpu_validation_summary=cpu_gpu_validation_summary,
        e2e_stage_summary=e2e_stage_summary,
        e2e_runs=e2e_runs,
        estimates=estimates,
    )
    report_path = results_dir / "benchmark_report.md"
    report_path.write_text(report, encoding="utf-8")

    result_manifest = {
        "raw_stage_metrics_csv": str(tables_dir / "all_stage_metrics.csv"),
        "core_summary_csv": str(tables_dir / "core_benchmark_summary.csv"),
        "cpu_gpu_raw_kl_validation_summary_csv": str(
            tables_dir / "cpu_gpu_raw_kl_validation_summary.csv"
        ),
        "end_to_end_stage_summary_csv": str(tables_dir / "end_to_end_stage_summary.csv"),
        "end_to_end_run_summary_csv": str(tables_dir / "end_to_end_run_summary.csv"),
        "full_scale_projection_csv": str(tables_dir / "full_scale_throughput_projection.csv"),
        "report_markdown": str(report_path),
        "figures": [str(path) for path in sorted(figures_dir.glob("*.png"))],
        "completed_core_measurements": int(len(core_df)),
        "completed_cpu_gpu_validation_measurements": int(len(validation_df)),
        "completed_end_to_end_stage_measurements": int(len(e2e_df)),
    }
    write_json(results_dir / "result_manifest.json", result_manifest)

    print(f"Raw measurements: {tables_dir / 'all_stage_metrics.csv'}")
    print(f"Core summary:     {tables_dir / 'core_benchmark_summary.csv'}")
    print(
        "CPU/GPU raw KL:   "
        f"{tables_dir / 'cpu_gpu_raw_kl_validation_summary.csv'}"
    )
    print(f"End-to-end table: {tables_dir / 'end_to_end_stage_summary.csv'}")
    print(f"Report:           {report_path}")
    if raw_df.empty:
        print("[WARN] No completed benchmark metrics were found; the report contains the design only.")


if __name__ == "__main__":
    main()
