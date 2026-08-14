#!/usr/bin/env python3
"""Shared utilities for the external ProDive benchmark harness.

This module intentionally treats the existing C++/CUDA executable and Python
post-processing programs as black boxes.  It does not import or modify their
implementation, command-line interface, or output formats.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PLACEHOLDER_MARKERS = ("EDIT_ME", "/path/to/", "CHANGE_ME")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_placeholder(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return any(marker in value for marker in PLACEHOLDER_MARKERS)


def resolve_path(value: str, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load the JSON configuration and add resolved absolute paths."""
    config_path = config_path.resolve()
    raw = read_json(config_path)
    config_dir = config_path.parent

    repo_root = resolve_path(str(raw.get("repo_root", "..")), config_dir)
    packed_db = resolve_path(str(raw["packed_db"]), config_dir)
    output_root = resolve_path(str(raw["output_root"]), config_dir)

    cpp_value = str(raw.get("cpp_executable", "src/cpp_cuda/kl_divergence"))
    cpp_executable = resolve_path(cpp_value, repo_root)

    metadata_path = packed_db / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = read_json(metadata_path)
        except Exception:
            metadata = {}

    mapping_value = str(raw.get("mapping_file", "AUTO_FROM_PACKED_METADATA"))
    if mapping_value == "AUTO_FROM_PACKED_METADATA":
        metadata_mapping = metadata.get("mapping_file")
        mapping_file = (
            resolve_path(str(metadata_mapping), packed_db)
            if metadata_mapping
            else Path("AUTO_FROM_PACKED_METADATA_NOT_FOUND")
        )
    else:
        mapping_file = resolve_path(mapping_value, config_dir)

    background_value = str(raw.get("background_json", "EDIT_ME"))
    background_json = (
        Path(background_value)
        if is_placeholder(background_value)
        else resolve_path(background_value, config_dir)
    )
    coverage_value = str(raw.get("coverage_dir", "EDIT_ME"))
    coverage_dir = (
        Path(coverage_value)
        if is_placeholder(coverage_value)
        else resolve_path(coverage_value, config_dir)
    )

    raw["_resolved"] = {
        "config_path": str(config_path),
        "config_dir": str(config_dir),
        "repo_root": str(repo_root),
        "packed_db": str(packed_db),
        "output_root": str(output_root),
        "cpp_executable": str(cpp_executable),
        "mapping_file": str(mapping_file),
        "background_json": str(background_json),
        "coverage_dir": str(coverage_dir),
    }
    raw["_packed_metadata"] = metadata
    return raw


def resolved(config: Mapping[str, Any], name: str) -> Path:
    return Path(str(config["_resolved"][name]))


def read_packed_index(index_path: Path) -> List[Dict[str, Any]]:
    required = {
        "family_id",
        "hmm_length",
        "fragment",
        "nfrag",
        "pi_offset",
        "pi_count",
        "A_offset",
        "A_count",
        "B_offset",
        "B_count",
    }
    records: List[Dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{index_path} is missing required columns: {missing}")
        for line_no, row in enumerate(reader, start=2):
            try:
                record = dict(row)
                for key in required:
                    record[key] = int(row[key])
            except Exception as exc:
                raise ValueError(f"Invalid packed-index row at line {line_no}: {exc}") from exc
            if record["family_id"] < 0 or record["nfrag"] <= 0:
                raise ValueError(
                    f"Invalid family_id/nfrag at {index_path}:{line_no}: "
                    f"{record['family_id']}/{record['nfrag']}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"Packed index contains no records: {index_path}")
    family_ids = [int(row["family_id"]) for row in records]
    if len(set(family_ids)) != len(family_ids):
        raise ValueError(f"Packed index contains duplicate family_id values: {index_path}")
    return records


def directory_stats(path: Path, patterns: Optional[Sequence[str]] = None) -> Dict[str, int]:
    if not path.exists():
        return {"file_count": 0, "bytes": 0}
    files: Iterable[Path]
    if patterns:
        collected: List[Path] = []
        for pattern in patterns:
            collected.extend(item for item in path.rglob(pattern) if item.is_file())
        files = set(collected)
    else:
        files = (item for item in path.rglob("*") if item.is_file())
    total_bytes = 0
    count = 0
    for item in files:
        try:
            total_bytes += item.stat().st_size
            count += 1
        except FileNotFoundError:
            continue
    return {"file_count": int(count), "bytes": int(total_bytes)}


def count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        count = sum(1 for _ in handle)
    return max(0, count - 1)


def read_pair_list_stats(path: Path) -> Dict[str, int]:
    pair_count = 0
    total_cells = 0
    families = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"i", "j", "window_cells"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pair list {path} is missing columns: {sorted(missing)}")
        for row in reader:
            i = int(row["i"])
            j = int(row["j"])
            pair_count += 1
            total_cells += int(row["window_cells"])
            families.add(i)
            families.add(j)
    return {
        "pair_count": int(pair_count),
        "window_cells": int(total_cells),
        "unique_families": int(len(families)),
    }


def _run_text(command: Sequence[str], timeout: float = 10.0) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "available": True,
            "returncode": int(completed.returncode),
            "output": completed.stdout.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "returncode": None, "output": str(exc)}


def query_gpus(device_ids: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []

    allowed = None if device_ids is None else {int(x) for x in device_ids}
    rows: List[Dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        if allowed is not None and index not in allowed:
            continue

        def number(value: str) -> Optional[float]:
            match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
            return float(match.group(0)) if match else None

        rows.append(
            {
                "index": index,
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": number(parts[3]),
                "memory_used_mib": number(parts[4]),
                "utilization_gpu_percent": number(parts[5]),
                "utilization_memory_percent": number(parts[6]),
                "power_draw_w": number(parts[7]),
            }
        )
    return rows


def _proc_children(pid: int) -> List[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    if not text:
        return []
    children: List[int] = []
    for token in text.split():
        try:
            children.append(int(token))
        except ValueError:
            continue
    return children


def process_tree_pids(root_pid: int) -> List[int]:
    seen = set()
    stack = [int(root_pid)]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(_proc_children(pid))
    return sorted(seen)


def process_tree_rss_mib(root_pid: int, exclude_root: bool = False) -> float:
    total_kib = 0
    for pid in process_tree_pids(root_pid):
        if exclude_root and pid == int(root_pid):
            continue
        status = Path(f"/proc/{pid}/status")
        try:
            text = status.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        match = re.search(r"^VmRSS:\s+(\d+)\s+kB", text, flags=re.MULTILINE)
        if match:
            total_kib += int(match.group(1))
    return total_kib / 1024.0


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def _max(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(clean) if clean else None


def run_monitored_command(
    command: Sequence[str],
    stage_dir: Path,
    cwd: Path,
    env: Mapping[str, str],
    gpu_device_ids: Sequence[int],
    monitor_interval_seconds: float,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a command and record wall time, process-tree RSS, and GPU samples."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = stage_dir / "stdout.log"
    stderr_path = stage_dir / "stderr.log"
    command_path = stage_dir / "command.json"
    gnu_time_path = stage_dir / "gnu_time_verbose.txt"
    resource_wrapper_path = stage_dir / "resource_wrapper.json"
    gnu_time = Path("/usr/bin/time")
    executed_command = list(command)
    if gnu_time.is_file():
        executed_command = [str(gnu_time), "-v", "-o", str(gnu_time_path)] + executed_command
    else:
        wrapper_script = Path(__file__).resolve().with_name("resource_wrapper.py")
        executed_command = [
            sys.executable,
            str(wrapper_script),
            "--output",
            str(resource_wrapper_path),
            "--",
        ] + executed_command
    write_json(
        command_path,
        {
            "argv": list(command),
            "shell_rendering_for_reference_only": shlex.join(list(command)),
            "monitored_argv": executed_command,
            "cwd": str(cwd),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", ""),
        },
    )

    baseline_rows = query_gpus(gpu_device_ids)
    baseline_memory = {
        int(row["index"]): float(row["memory_used_mib"] or 0.0)
        for row in baseline_rows
    }
    samples: List[Dict[str, Any]] = []
    start_wall = time.perf_counter()
    timed_out = False

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            executed_command,
            cwd=str(cwd),
            env=dict(env),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        try:
            while True:
                elapsed = time.perf_counter() - start_wall
                # The launched root is a lightweight GNU-time or Python resource
                # supervisor.  Excluding it prevents monitor overhead from being
                # attributed to the production command while retaining all command
                # descendants in the process-tree sum.
                rss = process_tree_rss_mib(process.pid, exclude_root=True)
                gpu_rows = query_gpus(gpu_device_ids)
                samples.append(
                    {
                        "elapsed_seconds": elapsed,
                        "cpu_rss_mib": rss,
                        "gpus": gpu_rows,
                    }
                )
                returncode = process.poll()
                if returncode is not None:
                    break
                if timeout_seconds is not None and elapsed > timeout_seconds:
                    timed_out = True
                    try:
                        os.killpg(process.pid, 15)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, 9)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    break
                time.sleep(max(0.05, float(monitor_interval_seconds)))
        except KeyboardInterrupt:
            try:
                os.killpg(process.pid, 15)
            except ProcessLookupError:
                pass
            process.wait()
            raise

    wall_seconds = time.perf_counter() - start_wall
    returncode = int(process.returncode if process.returncode is not None else -1)

    flattened_samples: List[Dict[str, Any]] = []
    for sample in samples:
        flat: Dict[str, Any] = {
            "elapsed_seconds": round(float(sample["elapsed_seconds"]), 6),
            "cpu_rss_mib": round(float(sample["cpu_rss_mib"]), 6),
        }
        gpu_by_index = {int(row["index"]): row for row in sample["gpus"]}
        for device_id in gpu_device_ids:
            row = gpu_by_index.get(int(device_id), {})
            prefix = f"gpu_{device_id}"
            flat[f"{prefix}_memory_used_mib"] = row.get("memory_used_mib")
            flat[f"{prefix}_utilization_gpu_percent"] = row.get("utilization_gpu_percent")
            flat[f"{prefix}_utilization_memory_percent"] = row.get("utilization_memory_percent")
            flat[f"{prefix}_power_draw_w"] = row.get("power_draw_w")
        flattened_samples.append(flat)

    samples_path = stage_dir / "resource_samples.csv"
    if flattened_samples:
        fields = list(flattened_samples[0].keys())
        with samples_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flattened_samples)

    per_device: Dict[str, Dict[str, Optional[float]]] = {}
    for device_id in gpu_device_ids:
        relevant = []
        for sample in samples:
            relevant.extend(
                row for row in sample["gpus"] if int(row["index"]) == int(device_id)
            )
        peak_memory = _max(row.get("memory_used_mib") for row in relevant)
        baseline = baseline_memory.get(int(device_id), 0.0)
        per_device[str(device_id)] = {
            "baseline_memory_used_mib": baseline,
            "peak_memory_used_mib": peak_memory,
            "peak_incremental_memory_mib": (
                max(0.0, float(peak_memory) - baseline) if peak_memory is not None else None
            ),
            "mean_gpu_utilization_percent": _mean(
                row.get("utilization_gpu_percent") for row in relevant
            ),
            "peak_gpu_utilization_percent": _max(
                row.get("utilization_gpu_percent") for row in relevant
            ),
            "mean_memory_utilization_percent": _mean(
                row.get("utilization_memory_percent") for row in relevant
            ),
            "mean_power_draw_w": _mean(row.get("power_draw_w") for row in relevant),
            "peak_power_draw_w": _max(row.get("power_draw_w") for row in relevant),
        }

    sampled_peak_cpu = _max(sample.get("cpu_rss_mib") for sample in samples) or 0.0
    gnu_time_peak_cpu: Optional[float] = None
    if gnu_time_path.is_file():
        try:
            gnu_time_text = gnu_time_path.read_text(encoding="utf-8", errors="replace")
            match = re.search(
                r"Maximum resident set size \(kbytes\):\s*(\d+)",
                gnu_time_text,
            )
            if match:
                gnu_time_peak_cpu = int(match.group(1)) / 1024.0
        except OSError:
            pass
    resource_wrapper_peak_cpu: Optional[float] = None
    resource_wrapper_payload: Dict[str, Any] = {}
    if resource_wrapper_path.is_file():
        try:
            resource_wrapper_payload = read_json(resource_wrapper_path)
            platform_units = float(resource_wrapper_payload["max_rss_platform_units"])
            if str(resource_wrapper_payload.get("platform", "")).startswith("darwin"):
                resource_wrapper_peak_cpu = platform_units / (1024.0 * 1024.0)
            else:
                resource_wrapper_peak_cpu = platform_units / 1024.0
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            resource_wrapper_payload = {}
    peak_cpu = max(
        float(sampled_peak_cpu),
        float(gnu_time_peak_cpu or 0.0),
        float(resource_wrapper_peak_cpu or 0.0),
    )
    peak_gpu_sum = sum(
        float(values["peak_memory_used_mib"] or 0.0) for values in per_device.values()
    )
    peak_increment_sum = sum(
        float(values["peak_incremental_memory_mib"] or 0.0) for values in per_device.values()
    )
    mean_power_sum = sum(
        float(values["mean_power_draw_w"] or 0.0) for values in per_device.values()
    )
    mean_util = _mean(
        values.get("mean_gpu_utilization_percent") for values in per_device.values()
    )

    return {
        "status": "timed_out" if timed_out else ("completed" if returncode == 0 else "failed"),
        "returncode": returncode,
        "timed_out": bool(timed_out),
        "wall_seconds": float(wall_seconds),
        "peak_cpu_rss_mib": float(peak_cpu),
        "sampled_process_tree_peak_cpu_rss_mib": float(sampled_peak_cpu),
        "gnu_time_peak_cpu_rss_mib": gnu_time_peak_cpu,
        "resource_wrapper_peak_cpu_rss_mib": resource_wrapper_peak_cpu,
        "resource_wrapper_usage": resource_wrapper_payload,
        "cpu_memory_measurement_mode": (
            "process_tree_sampling_with_gnu_time_fallback"
            if sampled_peak_cpu > 0
            else (
                "gnu_time_max_rss_fallback"
                if gnu_time_peak_cpu is not None
                else (
                    "per_invocation_child_rusage_fallback"
                    if resource_wrapper_peak_cpu is not None
                    else "unavailable"
                )
            )
        ),
        "gpu_monitor_available": bool(baseline_rows or any(sample["gpus"] for sample in samples)),
        "gpu_device_ids": [int(x) for x in gpu_device_ids],
        "gpu_peak_memory_used_mib_sum": float(peak_gpu_sum),
        "gpu_peak_incremental_memory_mib_sum": float(peak_increment_sum),
        "gpu_mean_utilization_percent": mean_util,
        "gpu_mean_power_draw_w_sum": float(mean_power_sum),
        "gpu_energy_wh_estimate": float(mean_power_sum * wall_seconds / 3600.0),
        "gpu_per_device": per_device,
        "monitor_interval_seconds": float(monitor_interval_seconds),
        "resource_sample_count": int(len(samples)),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "resource_samples_csv": str(samples_path),
        "gnu_time_verbose": str(gnu_time_path) if gnu_time_path.is_file() else "",
        "resource_wrapper_json": (
            str(resource_wrapper_path) if resource_wrapper_path.is_file() else ""
        ),
    }


def parse_cpp_stdout(stdout_path: Path) -> Dict[str, Optional[float]]:
    try:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {}
    patterns = {
        "cpp_cpu_matrix_inversion_seconds": r"CPU matrix inversion completed in\s+([0-9.]+)\s+ms",
        "cpp_reported_average_pairs_per_second": r"Average throughput:\s+([0-9.eE+-]+)\s+pairs/second",
        "cpp_reported_pairs_completed": r"Total pairs computed:\s+(\d+)",
    }
    result: Dict[str, Optional[float]] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            result[name] = None
            continue
        value = float(match.group(1))
        if name == "cpp_cpu_matrix_inversion_seconds":
            value /= 1000.0
        result[name] = value
    return result


def collect_hardware_inventory(output_root: Path, packed_db: Path) -> Dict[str, Any]:
    cpu_model = ""
    logical_cpus = os.cpu_count()
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, flags=re.MULTILINE)
        if match:
            cpu_model = match.group(1).strip()
    except FileNotFoundError:
        pass

    total_ram_bytes: Optional[int] = None
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB", meminfo, flags=re.MULTILINE)
        if match:
            total_ram_bytes = int(match.group(1)) * 1024
    except FileNotFoundError:
        pass

    try:
        disk = shutil.disk_usage(output_root if output_root.exists() else output_root.parent)
        disk_info = {
            "path": str(output_root),
            "total_bytes": int(disk.total),
            "used_bytes": int(disk.used),
            "free_bytes": int(disk.free),
        }
    except FileNotFoundError:
        disk_info = {"path": str(output_root), "total_bytes": None, "used_bytes": None, "free_bytes": None}

    package_versions: Dict[str, Optional[str]] = {}
    for package_name in ["numpy", "pandas", "matplotlib"]:
        try:
            module = __import__(package_name)
            package_versions[package_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            package_versions[package_name] = None

    db_files = {}
    for name in [
        "index.csv",
        "metadata.json",
        "summary.json",
        "pi_all.float32.bin",
        "A_all.float32.bin",
        "B_all.float32.bin",
    ]:
        path = packed_db / name
        db_files[name] = int(path.stat().st_size) if path.is_file() else None

    return {
        "collected_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "cpu_model": cpu_model,
        "logical_cpu_count": logical_cpus,
        "total_ram_bytes": total_ram_bytes,
        "gpus": query_gpus(),
        "nvidia_smi": _run_text(["nvidia-smi"]),
        "nvcc_version": _run_text(["nvcc", "--version"]),
        "cxx_version": _run_text(["g++", "--version"]),
        "package_versions": package_versions,
        "disk": disk_info,
        "packed_database": {
            "path": str(packed_db),
            "files": db_files,
            "total_known_bytes": int(sum(value or 0 for value in db_files.values())),
        },
    }


def stage_metric_base(
    run_id: str,
    run_type: str,
    stage: str,
    repeat: int,
    pair_stats: Mapping[str, int],
    gpu_count: int,
    gpu_device_ids: Sequence[int],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "run_type": run_type,
        "stage": stage,
        "repeat": int(repeat),
        "pair_count": int(pair_stats.get("pair_count", 0)),
        "window_cells": int(pair_stats.get("window_cells", 0)),
        "unique_families": int(pair_stats.get("unique_families", 0)),
        "gpu_count": int(gpu_count),
        "gpu_device_ids": [int(x) for x in gpu_device_ids],
    }


def command_environment(gpu_device_ids: Sequence[int]) -> Dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(x)) for x in gpu_device_ids)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def cleanup_globs(root: Path, patterns: Sequence[str]) -> Dict[str, int]:
    removed_count = 0
    removed_bytes = 0
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            try:
                removed_bytes += path.stat().st_size
                path.unlink()
                removed_count += 1
            except FileNotFoundError:
                continue
    return {"removed_file_count": removed_count, "removed_bytes": removed_bytes}
