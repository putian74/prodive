#!/usr/bin/env python3
"""Validate a ProDive benchmark configuration before any timed run."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmark_common import (
    collect_hardware_inventory,
    is_placeholder,
    load_config,
    query_gpus,
    read_packed_index,
    resolved,
    write_json,
)


PACKED_REQUIRED = [
    "index.csv",
    "metadata.json",
    "pi_all.float32.bin",
    "A_all.float32.bin",
    "B_all.float32.bin",
]


def add_check(checks: List[Dict[str, Any]], name: str, ok: bool, detail: str, level: str = "error") -> None:
    checks.append({"name": name, "ok": bool(ok), "level": level, "detail": detail})


def count_coverage_files(coverage_dir: Path) -> int:
    if not coverage_dir.is_dir():
        return 0
    return sum(1 for _ in coverage_dir.rglob("*_coverage.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ProDive benchmark inputs and hardware.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["gpu", "compare", "all", "core", "validation"],
        default="all",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = resolved(config, "repo_root")
    packed_db = resolved(config, "packed_db")
    output_root = resolved(config, "output_root")
    cpp_executable = resolved(config, "cpp_executable")
    checks: List[Dict[str, Any]] = []

    add_check(checks, "repository root", repo_root.is_dir(), str(repo_root))
    add_check(checks, "packed database directory", packed_db.is_dir(), str(packed_db))
    for name in PACKED_REQUIRED:
        path = packed_db / name
        add_check(checks, f"packed database file: {name}", path.is_file(), str(path))

    index_records = []
    if (packed_db / "index.csv").is_file():
        try:
            index_records = read_packed_index(packed_db / "index.csv")
            add_check(
                checks,
                "packed index schema",
                True,
                f"{len(index_records):,} valid family rows",
            )
        except Exception as exc:
            add_check(checks, "packed index schema", False, repr(exc))

    metadata = config.get("_packed_metadata", {})
    expected_fragment = int(config.get("thresholds", {}).get("fragment", 6))
    metadata_fragment = metadata.get("fragment")
    if metadata_fragment is not None:
        add_check(
            checks,
            "fragment length",
            int(metadata_fragment) == expected_fragment,
            f"metadata={metadata_fragment}, benchmark={expected_fragment}",
        )

    executable_ok = cpp_executable.is_file() and os.access(cpp_executable, os.X_OK)
    add_check(checks, "C++/CUDA executable", executable_ok, str(cpp_executable))
    if executable_ok:
        try:
            result = subprocess.run(
                [str(cpp_executable), "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10.0,
                check=False,
            )
            help_ok = result.returncode == 0 and "--pair_list" in result.stdout and "--packed_db" in result.stdout
            add_check(checks, "C++ benchmark CLI compatibility", help_ok, "--pair_list and --packed_db detected")
        except Exception as exc:
            add_check(checks, "C++ benchmark CLI compatibility", False, repr(exc))

    cpu_validation_enabled = bool(
        config.get("cpu_gpu_validation", {}).get("enabled", True)
    ) and args.mode in {"compare", "validation", "all"}
    cpu_comparison_requested = bool(
        config.get("cpu_gpu_comparison", {}).get("enabled", True)
    ) and args.mode in {"compare", "all"}
    packages = ["numpy", "pandas", "matplotlib"]
    if (
        cpu_validation_enabled
        and bool(config.get("cpu_gpu_validation", {}).get("require_numba", True))
    ) or (
        cpu_comparison_requested
        and bool(config.get("cpu_gpu_comparison", {}).get("require_numba", True))
    ):
        packages.append("numba")
    for package in packages:
        try:
            module = importlib.import_module(package)
            add_check(checks, f"Python package: {package}", True, str(getattr(module, "__version__", "unknown")))
        except Exception as exc:
            add_check(checks, f"Python package: {package}", False, repr(exc))

    gpu_ids = [int(x) for x in config.get("core", {}).get("gpu_device_ids", [])]
    gpu_counts = [int(x) for x in config.get("core", {}).get("gpu_counts", [])]
    gpus = query_gpus()
    available_ids = {int(row["index"]) for row in gpus}
    add_check(checks, "nvidia-smi GPU query", bool(gpus), f"detected physical GPU IDs: {sorted(available_ids)}")
    add_check(
        checks,
        "configured GPU IDs",
        bool(gpu_ids) and set(gpu_ids).issubset(available_ids),
        f"configured={gpu_ids}, detected={sorted(available_ids)}",
    )
    gpu_profile_requested = args.mode in {"gpu", "core", "all"}
    if gpu_profile_requested:
        add_check(
            checks,
            "GPU performance count settings",
            bool(gpu_counts)
            and min(gpu_counts) >= 1
            and max(gpu_counts) <= len(gpu_ids),
            f"gpu_counts={gpu_counts}, configured devices={len(gpu_ids)}",
        )
    full_family = config.get("core", {}).get("full_family_coverage", {})
    if gpu_profile_requested and bool(full_family.get("enabled", False)):
        full_family_gpu_counts = [int(x) for x in full_family.get("gpu_counts", [])]
        add_check(
            checks,
            "full-family coverage GPU settings",
            bool(full_family_gpu_counts)
            and min(full_family_gpu_counts) >= 1
            and max(full_family_gpu_counts) <= len(gpu_ids),
            f"gpu_counts={full_family_gpu_counts}, configured devices={len(gpu_ids)}",
        )

    sampling_pair_counts = [
        int(x) for x in config.get("sampling", {}).get("pair_counts", [])
    ]
    core_pair_counts = [
        int(x)
        for x in config.get("core", {}).get(
            "pair_counts", sampling_pair_counts
        )
    ]
    comparison = config.get("cpu_gpu_comparison", {})
    comparison_pair_counts = (
        [int(x) for x in comparison.get("pair_counts", [])]
        if bool(comparison.get("enabled", True))
        else []
    )
    pair_counts = sorted(
        set(sampling_pair_counts)
        | set(core_pair_counts)
        | set(comparison_pair_counts)
    )
    total_pairs = len(index_records) * (len(index_records) - 1) // 2 if index_records else 0
    add_check(
        checks,
        "pair-count design",
        bool(pair_counts) and pair_counts == sorted(set(pair_counts)) and min(pair_counts) > 0 and max(pair_counts) <= total_pairs,
        f"pair_counts={pair_counts}, available unordered non-self pairs={total_pairs:,}",
    )
    if (
        args.mode in {"gpu", "all"}
        and config.get("end_to_end", {}).get("enabled", True)
    ):
        e2e_pairs = int(config.get("end_to_end", {}).get("pair_count", 0))
        add_check(checks, "end-to-end pair count", e2e_pairs in pair_counts, f"pair_count={e2e_pairs}")

    if cpu_validation_enabled:
        cpu_validation = config.get("cpu_gpu_validation", {})
        cpu_pair_count = int(cpu_validation.get("pair_count", 0))
        cpu_source_pair_count = int(cpu_validation.get("source_pair_count", 0))
        cpu_max_cells = int(cpu_validation.get("max_window_cells_per_pair", 0))
        cpu_gpu_count = int(cpu_validation.get("gpu_count", 1))
        cpu_design_ok = (
            cpu_pair_count > 0
            and cpu_source_pair_count >= cpu_pair_count
            and bool(pair_counts)
            and cpu_source_pair_count <= max(pair_counts)
            and cpu_max_cells > 0
            and 1 <= cpu_gpu_count <= len(gpu_ids)
        )
        add_check(
            checks,
            "CPU/GPU raw-KL validation design",
            cpu_design_ok,
            (
                f"pair_count={cpu_pair_count}, source_pair_count={cpu_source_pair_count}, "
                f"max_window_cells_per_pair={cpu_max_cells}, gpu_count={cpu_gpu_count}"
            ),
        )
        cpu_reference = repo_root / "src" / "cpu_reference" / "cpu_kl_reference.py"
        cpu_batch_runner = repo_root / "benchmark" / "scripts" / "cpu_batch_runner.py"
        add_check(
            checks,
            "serial CPU raw-KL reference",
            cpu_reference.is_file() and cpu_batch_runner.is_file(),
            f"{cpu_reference}; {cpu_batch_runner}",
        )
        checked_hhm_paths = []
        missing_hhm_paths = []
        metadata_input_dir = config.get("_packed_metadata", {}).get("input_dir")
        for record in index_records[: min(20, len(index_records))]:
            raw = Path(str(record.get("hmm_file", "")))
            candidates = [raw]
            if raw and not raw.is_absolute():
                candidates.append(packed_db / raw)
                if metadata_input_dir:
                    candidates.append(Path(str(metadata_input_dir)) / raw)
            resolved_hhm = next((path for path in candidates if path.is_file()), None)
            if resolved_hhm is None:
                missing_hhm_paths.append(str(raw))
            else:
                checked_hhm_paths.append(str(resolved_hhm))
        add_check(
            checks,
            "CPU HHM source files (first 20 packed-index rows)",
            bool(checked_hhm_paths) and not missing_hhm_paths,
            (
                f"resolved={len(checked_hhm_paths)}, missing={len(missing_hhm_paths)}"
                + (f"; first missing={missing_hhm_paths[0]}" if missing_hhm_paths else "")
            ),
        )

    comparison_enabled = bool(comparison.get("enabled", True)) and args.mode in {
        "compare",
        "all",
    }
    if comparison_enabled:
        comparison_gpu_counts = [
            int(value) for value in comparison.get("gpu_counts", [])
        ]
        base_cpu_process_counts = [
            int(value) for value in comparison.get("cpu_process_counts", [1])
        ]
        parallel = comparison.get("parallel_scaling", {})
        parallel_process_counts = (
            [int(value) for value in parallel.get("process_counts", [])]
            if bool(parallel.get("enabled", True))
            else []
        )
        all_process_counts = sorted(
            set(base_cpu_process_counts) | set(parallel_process_counts)
        )
        logical_cpus = int(os.cpu_count() or 1)
        comparison_ok = (
            bool(comparison_pair_counts)
            and comparison_pair_counts == sorted(set(comparison_pair_counts))
            and min(comparison_pair_counts) > 0
            and bool(comparison_gpu_counts)
            and min(comparison_gpu_counts) >= 1
            and max(comparison_gpu_counts) <= len(gpu_ids)
            and bool(all_process_counts)
            and min(all_process_counts) >= 1
            and max(all_process_counts) <= logical_cpus
        )
        add_check(
            checks,
            "packed-input CPU/GPU comparison design",
            comparison_ok,
            (
                f"pair_counts={comparison_pair_counts}, "
                f"gpu_counts={comparison_gpu_counts}, "
                f"cpu_process_counts={all_process_counts}, "
                f"logical_cpus={logical_cpus}"
            ),
        )
        cpu_packed_runner = (
            repo_root / "benchmark" / "scripts" / "cpu_packed_runner.py"
        )
        cpu_comparison_runner = (
            repo_root
            / "benchmark"
            / "scripts"
            / "06_run_cpu_gpu_comparison.py"
        )
        add_check(
            checks,
            "packed-input CPU performance runner",
            cpu_packed_runner.is_file() and cpu_comparison_runner.is_file(),
            f"{cpu_packed_runner}; {cpu_comparison_runner}",
        )

    if args.mode in {"gpu", "all"}:
        mapping_file = resolved(config, "mapping_file")
        background_json = resolved(config, "background_json")
        coverage_dir = resolved(config, "coverage_dir")
        add_check(
            checks,
            "mapping file",
            mapping_file.is_file(),
            f"{mapping_file} (resolved from packed metadata when configured as AUTO)",
        )
        add_check(
            checks,
            "background KL JSON",
            not is_placeholder(str(background_json)) and background_json.is_file(),
            str(background_json),
        )
        # Coverage files may be flat or nested as
        # coverage_dir/PF00001/PF00001_coverage.csv.  The production rescoring
        # stage supports both layouts, so preflight must search recursively too.
        coverage_file_count = count_coverage_files(coverage_dir)
        add_check(
            checks,
            "coverage directory",
            not is_placeholder(str(coverage_dir))
            and coverage_dir.is_dir()
            and coverage_file_count > 0,
            f"{coverage_dir} ({coverage_file_count:,} recursive *_coverage.csv files)",
        )

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".prodive_benchmark_write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        add_check(checks, "benchmark output directory", True, str(output_root))
    except Exception as exc:
        add_check(checks, "benchmark output directory", False, f"{output_root}: {exc}")

    hardware = collect_hardware_inventory(output_root, packed_db)
    write_json(output_root / "hardware_inventory.json", hardware)

    failures = [item for item in checks if not item["ok"] and item["level"] == "error"]
    warnings = [item for item in checks if not item["ok"] and item["level"] == "warning"]
    report = {
        "status": "failed" if failures else "passed",
        "mode": args.mode,
        "config_path": str(args.config.resolve()),
        "checked_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "checks": checks,
        "error_count": len(failures),
        "warning_count": len(warnings),
        "hardware_inventory": str(output_root / "hardware_inventory.json"),
    }
    write_json(output_root / "preflight_report.json", report)

    for item in checks:
        marker = "PASS" if item["ok"] else ("WARN" if item["level"] == "warning" else "FAIL")
        print(f"[{marker}] {item['name']}: {item['detail']}")
    print(f"Preflight report: {output_root / 'preflight_report.json'}")
    if failures:
        print(f"Preflight failed with {len(failures)} blocking error(s).", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
