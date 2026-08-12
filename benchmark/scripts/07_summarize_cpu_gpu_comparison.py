#!/usr/bin/env python3
"""Summarize fair packed-input CPU/GPU performance measurements."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_common import load_config, read_json, resolved, write_json


def load_metrics(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(paths):
        try:
            row = read_json(path)
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}")
            continue
        row["metric_path"] = str(path)
        rows.append(row)
    return rows


def numeric(values: Iterable[Any]) -> np.ndarray:
    converted = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    return converted[np.isfinite(converted)].to_numpy(dtype=float)


def stats(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    array = numeric(values)
    if not array.size:
        return {"median": None, "q1": None, "q3": None, "min": None, "max": None}
    return {
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_backend(rows: pd.DataFrame, backend: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    if rows.empty:
        return output
    device_column = "cpu_process_count" if backend == "CPU" else "gpu_count"
    for (pair_count, device_count), group in rows.groupby(["pair_count", device_column]):
        wall = stats(group["wall_seconds"])
        throughput = stats(group["external_window_cells_per_second"])
        rss = stats(group["peak_cpu_rss_mib"])
        gpu_memory = stats(group.get("gpu_peak_incremental_memory_mib_sum", []))
        internal_key = (
            "cpu_parallel_pair_stage_wall_seconds"
            if backend == "CPU"
            else "cpp_reported_computation_seconds"
        )
        internal = stats(group.get(internal_key, []))
        window_cells = int(pd.to_numeric(group["window_cells"]).median())
        output.append(
            {
                "backend": backend,
                "device_count": int(device_count),
                "device_unit": "CPU processes" if backend == "CPU" else "GPUs",
                "pair_count": int(pair_count),
                "window_cells": window_cells,
                "repeat_count": int(len(group)),
                "wall_seconds_median": wall["median"],
                "wall_seconds_q1": wall["q1"],
                "wall_seconds_q3": wall["q3"],
                "window_cells_per_second_median": throughput["median"],
                "window_cells_per_second_q1": throughput["q1"],
                "window_cells_per_second_q3": throughput["q3"],
                "peak_host_rss_mib_median": rss["median"],
                "peak_gpu_incremental_memory_mib_sum_median": gpu_memory["median"],
                "internal_pair_or_gpu_stage_seconds_median": internal["median"],
            }
        )
    return output


def make_speedups(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty:
        return pd.DataFrame()
    cpu = performance[performance["backend"] == "CPU"]
    gpu = performance[performance["backend"] == "GPU"]
    rows: List[Dict[str, Any]] = []
    for _, cpu_row in cpu.iterrows():
        matches = gpu[gpu["pair_count"] == cpu_row["pair_count"]]
        for _, gpu_row in matches.iterrows():
            cpu_wall = float(cpu_row["wall_seconds_median"])
            gpu_wall = float(gpu_row["wall_seconds_median"])
            rows.append(
                {
                    "pair_count": int(cpu_row["pair_count"]),
                    "window_cells": int(cpu_row["window_cells"]),
                    "cpu_process_count": int(cpu_row["device_count"]),
                    "gpu_count": int(gpu_row["device_count"]),
                    "cpu_wall_seconds_median": cpu_wall,
                    "gpu_wall_seconds_median": gpu_wall,
                    "gpu_speedup_vs_cpu": cpu_wall / gpu_wall if gpu_wall > 0 else None,
                    "cpu_window_cells_per_second_median": cpu_row[
                        "window_cells_per_second_median"
                    ],
                    "gpu_window_cells_per_second_median": gpu_row[
                        "window_cells_per_second_median"
                    ],
                    "comparison_boundary": (
                        "complete launched raw-KL stage from the same packed database, "
                        "including selected-family loading, inversion, computation, and NPY output"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["pair_count", "cpu_process_count", "gpu_count"]
    ).reset_index(drop=True) if rows else pd.DataFrame()


def linear_fits(performance: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if performance.empty:
        return pd.DataFrame()
    for (backend, device_count), group in performance.groupby(["backend", "device_count"]):
        group = group.sort_values("window_cells")
        x = pd.to_numeric(group["window_cells"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group["wall_seconds_median"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if len(np.unique(x)) < 2:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        prediction = intercept + slope * x
        residual = float(np.sum((y - prediction) ** 2))
        total = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - residual / total if total > 0 else 1.0
        rows.append(
            {
                "backend": backend,
                "device_count": int(device_count),
                "point_count": int(len(x)),
                "intercept_seconds": float(intercept),
                "seconds_per_window_cell": float(slope),
                "asymptotic_window_cells_per_second": (
                    float(1.0 / slope) if slope > 0 else None
                ),
                "r_squared": float(r_squared),
                "interpretation": "descriptive fit over measured sampled workloads only",
            }
        )
    return pd.DataFrame(rows)


def save_runtime_chart(performance: pd.DataFrame, path: Path) -> None:
    if performance.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for (backend, count), group in performance.groupby(["backend", "device_count"]):
        group = group.sort_values("window_cells")
        label = f"{backend} ({int(count)} {'processes' if backend == 'CPU' else 'GPU'})"
        marker = "o" if backend == "CPU" else "s"
        ax.plot(
            group["window_cells"],
            group["wall_seconds_median"],
            marker=marker,
            linewidth=1.8,
            label=label,
        )
        ax.fill_between(
            group["window_cells"].to_numpy(dtype=float),
            group["wall_seconds_q1"].to_numpy(dtype=float),
            group["wall_seconds_q3"].to_numpy(dtype=float),
            alpha=0.13,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Raw-KL matrix cells")
    ax.set_ylabel("Complete stage wall time (s)")
    ax.set_title("CPU and GPU packed-input raw-KL runtime")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_speedup_chart(speedups: pd.DataFrame, path: Path) -> None:
    if speedups.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for (cpu_count, gpu_count), group in speedups.groupby(
        ["cpu_process_count", "gpu_count"]
    ):
        group = group.sort_values("window_cells")
        ax.plot(
            group["window_cells"],
            group["gpu_speedup_vs_cpu"],
            marker="o",
            linewidth=1.8,
            label=f"{int(gpu_count)} GPU vs {int(cpu_count)} CPU process(es)",
        )
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Raw-KL matrix cells")
    ax.set_ylabel("GPU speedup (CPU time / GPU time)")
    ax.set_title("CPU/GPU speedup across measured workloads")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def build_report(
    config: Mapping[str, Any],
    performance: pd.DataFrame,
    speedups: pd.DataFrame,
    fits: pd.DataFrame,
) -> str:
    comparison = config.get("cpu_gpu_comparison", {})
    lines = [
        "# CPU/GPU raw-KL performance comparison",
        "",
        "## Scope",
        "",
        "CPU and GPU start from the same packed float32 fragment database and use the same deterministic pair lists. The measured quantity is the raw symmetric KL matrix before background normalization, thresholds, path construction, or coverage rescoring.",
        "",
        "Complete-stage wall time includes selected-family loading, CPU matrix inversion, raw-KL calculation, and NPY output. CPU numerical-library threads are fixed to one per process.",
        "",
        "## Design",
        "",
        f"- Pair counts: {comparison.get('pair_counts', [])}",
        f"- CPU process counts across the size series: {comparison.get('cpu_process_counts', [1])}",
        f"- GPU counts: {comparison.get('gpu_counts', [])}",
        f"- Repeats: {comparison.get('repeats', 3)}",
        "",
        "## Measured performance",
        "",
        "| Backend | Devices/processes | Pairs | Matrix cells | Median wall (s) | Median cells/s | Median host RSS (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if performance.empty:
        lines.append("| No completed measurements | | | | | | |")
    else:
        for row in performance.sort_values(
            ["pair_count", "backend", "device_count"]
        ).itertuples():
            lines.append(
                f"| {row.backend} | {row.device_count} | {row.pair_count:,} | "
                f"{row.window_cells:,} | {fmt(row.wall_seconds_median)} | "
                f"{fmt(row.window_cells_per_second_median, 1)} | "
                f"{fmt(row.peak_host_rss_mib_median, 1)} |"
            )
    lines.extend(
        [
            "",
            "## Speedup",
            "",
            "| Pairs | Matrix cells | CPU processes | GPUs | GPU speedup |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    if speedups.empty:
        lines.append("| No matched CPU/GPU measurements | | | | |")
    else:
        for row in speedups.itertuples():
            lines.append(
                f"| {row.pair_count:,} | {row.window_cells:,} | "
                f"{row.cpu_process_count} | {row.gpu_count} | "
                f"{fmt(row.gpu_speedup_vs_cpu, 2)}x |"
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Pair count is not the computational workload; use total matrix cells when interpreting scaling.",
            "- CPU and GPU wall times are approximately linear only over the measured range. Linear fits are descriptive and are not substitutes for a full all-pairs run.",
            "- GPU speedup may rise while fixed setup costs are amortized, then approach a plateau after device utilization saturates.",
            "- Multi-process host RSS is the summed process-tree RSS and may count shared copy-on-write pages more than once.",
        ]
    )
    if not fits.empty:
        lines.extend(
            [
                "",
                "## Descriptive linear fits",
                "",
                "| Backend | Devices/processes | Seconds per cell | R² |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in fits.itertuples():
            lines.append(
                f"| {row.backend} | {row.device_count} | "
                f"{row.seconds_per_window_cell:.6e} | {fmt(row.r_squared, 4)} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize packed-input CPU/GPU comparison measurements."
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    comparison = config.get("cpu_gpu_comparison", {})
    output_root = resolved(config, "output_root")
    results_dir = output_root / "results" / "cpu_gpu_comparison"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cpu_rows = load_metrics(
        (output_root / "runs" / "cpu_gpu_comparison").glob("*/stage_metrics.json")
    )
    gpu_rows = load_metrics((output_root / "runs" / "core").glob("*/stage_metrics.json"))
    pair_counts = {int(value) for value in comparison.get("pair_counts", [])}
    gpu_counts = {int(value) for value in comparison.get("gpu_counts", [])}
    cpu_df = pd.DataFrame(cpu_rows)
    gpu_df = pd.DataFrame(gpu_rows)
    if not cpu_df.empty:
        cpu_df = cpu_df[
            (cpu_df.get("status") == "completed")
            & (cpu_df.get("validation_status") == "passed")
            & cpu_df["pair_count"].astype(int).isin(pair_counts)
        ].copy()
    if not gpu_df.empty:
        gpu_df = gpu_df[
            (gpu_df.get("status") == "completed")
            & (gpu_df.get("validation_status") == "passed")
            & (gpu_df.get("sample_kind") == "stratified_workload")
            & gpu_df["pair_count"].astype(int).isin(pair_counts)
            & gpu_df["gpu_count"].astype(int).isin(gpu_counts)
        ].copy()

    rows = aggregate_backend(cpu_df, "CPU") + aggregate_backend(gpu_df, "GPU")
    performance = pd.DataFrame(rows)
    if not performance.empty:
        performance = performance.sort_values(
            ["pair_count", "backend", "device_count"]
        ).reset_index(drop=True)
    speedups = make_speedups(performance)
    fits = linear_fits(performance)
    performance.to_csv(tables_dir / "cpu_gpu_performance_summary.csv", index=False)
    speedups.to_csv(tables_dir / "cpu_gpu_speedup_summary.csv", index=False)
    fits.to_csv(tables_dir / "cpu_gpu_linear_fit_summary.csv", index=False)

    save_runtime_chart(performance, figures_dir / "cpu_gpu_runtime_scaling.png")
    save_speedup_chart(speedups, figures_dir / "cpu_gpu_speedup_scaling.png")
    report_path = results_dir / "cpu_gpu_comparison_report.md"
    report_path.write_text(
        build_report(config, performance, speedups, fits), encoding="utf-8"
    )
    manifest = {
        "performance_summary_csv": str(
            tables_dir / "cpu_gpu_performance_summary.csv"
        ),
        "speedup_summary_csv": str(tables_dir / "cpu_gpu_speedup_summary.csv"),
        "linear_fit_summary_csv": str(
            tables_dir / "cpu_gpu_linear_fit_summary.csv"
        ),
        "report_markdown": str(report_path),
        "figures": [str(path) for path in sorted(figures_dir.glob("*.png"))],
        "completed_cpu_measurements": int(len(cpu_df)),
        "completed_gpu_measurements": int(len(gpu_df)),
    }
    write_json(results_dir / "result_manifest.json", manifest)
    print(f"CPU/GPU performance table: {manifest['performance_summary_csv']}")
    print(f"CPU/GPU speedup table:     {manifest['speedup_summary_csv']}")
    print(f"CPU/GPU report:            {report_path}")


if __name__ == "__main__":
    main()
