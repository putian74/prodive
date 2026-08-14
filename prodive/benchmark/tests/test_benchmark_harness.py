#!/usr/bin/env python3
"""Small tests for the external benchmark harness (no CUDA required)."""

from __future__ import annotations

import importlib.util
import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = BENCHMARK_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from benchmark_common import command_environment, run_monitored_command  # noqa: E402
from raw_kl_compare import run_comparison  # noqa: E402


def load_script_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SamplingTests(unittest.TestCase):
    def test_unique_pair_codes_are_nonself_and_deterministic(self) -> None:
        import numpy as np

        prepare = load_script_module("01_prepare_benchmark.py", "prepare_benchmark_test")
        first = prepare.unique_random_pair_codes(np.random.default_rng(123), n=100, target=500)
        second = prepare.unique_random_pair_codes(np.random.default_rng(123), n=100, target=500)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), len(np.unique(first)))
        left = first // 100
        right = first % 100
        self.assertTrue(np.all(left < right))

    def test_cpu_validation_subset_is_bounded_and_deterministic(self) -> None:
        prepare = load_script_module("01_prepare_benchmark.py", "prepare_cpu_subset_test")
        rows = [
            {
                "i": value,
                "j": value + 100,
                "nfrag_i": 10,
                "nfrag_j": value + 1,
                "window_cells": 10 * (value + 1),
                "workload_stratum": f"Q{value % 5 + 1}",
                "sample_rank": value + 1,
            }
            for value in range(10)
        ]
        selected = prepare.select_cpu_validation_rows(
            rows, pair_count=3, max_window_cells_per_pair=50
        )
        self.assertEqual([row["i"] for row in selected], [0, 1, 2])
        self.assertTrue(all(row["window_cells"] <= 50 for row in selected))
        self.assertEqual([row["sample_rank"] for row in selected], [1, 2, 3])

    def test_prepare_and_summarize_integration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packed = root / "packed"
            packed.mkdir()
            fields = [
                "family_id",
                "pfam",
                "hmm_file",
                "hmm_length",
                "fragment",
                "nfrag",
                "pi_offset",
                "pi_count",
                "pi_shape",
                "A_offset",
                "A_count",
                "A_shape",
                "B_offset",
                "B_count",
                "B_shape",
            ]
            with (packed / "index.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for family_id in range(100):
                    nfrag = 2 + family_id % 17
                    writer.writerow(
                        {
                            "family_id": family_id,
                            "pfam": f"PF{family_id:05d}",
                            "hmm_file": f"PF{family_id:05d}.hhm",
                            "hmm_length": nfrag + 5,
                            "fragment": 6,
                            "nfrag": nfrag,
                            "pi_offset": 0,
                            "pi_count": nfrag * 19,
                            "pi_shape": "",
                            "A_offset": 0,
                            "A_count": nfrag * 19 * 19,
                            "A_shape": "",
                            "B_offset": 0,
                            "B_count": nfrag * 19 * 21,
                            "B_shape": "",
                        }
                    )
            (packed / "metadata.json").write_text(
                json.dumps({"fragment": 6, "mapping_file": str(root / "mapping.txt")}),
                encoding="utf-8",
            )
            for name in ["pi_all.float32.bin", "A_all.float32.bin", "B_all.float32.bin"]:
                (packed / name).write_bytes(b"")

            output_root = root / "output"
            config = {
                "repo_root": str(root),
                "packed_db": str(packed),
                "output_root": str(output_root),
                "cpp_executable": str(root / "not_used"),
                "mapping_file": "AUTO_FROM_PACKED_METADATA",
                "background_json": "EDIT_ME",
                "coverage_dir": "EDIT_ME",
                "sampling": {
                    "seed": 1234,
                    "pair_counts": [10, 50],
                    "candidate_pool_size": 1000,
                    "workload_strata": 5,
                    "exclude_self_pairs": True,
                },
                "core": {
                    "gpu_device_ids": [0, 1],
                    "gpu_counts": [1, 2],
                    "repeats": 1,
                },
                "end_to_end": {"pair_count": 50},
                "monitor": {"interval_seconds": 0.5},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            prepared = subprocess.run(
                [sys.executable, str(SCRIPTS / "01_prepare_benchmark.py"), "--config", str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(prepared.returncode, 0, msg=prepared.stdout)
            with (output_root / "design/pair_lists/pairs_10.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                pairs_10 = list(csv.DictReader(handle))
            with (output_root / "design/pair_lists/pairs_50.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                pairs_50 = list(csv.DictReader(handle))
            self.assertEqual(pairs_10, pairs_50[:10])
            self.assertEqual(len({(row["i"], row["j"]) for row in pairs_50}), 50)

            core_root = output_root / "runs" / "core"
            for gpu_count, wall in [(1, 10.0), (2, 6.0)]:
                run_dir = core_root / f"core_p50_g{gpu_count}_r01"
                run_dir.mkdir(parents=True)
                metric = {
                    "run_id": run_dir.name,
                    "run_type": "core",
                    "stage": "cpp_kl",
                    "repeat": 1,
                    "pair_count": 50,
                    "window_cells": 5000,
                    "unique_families": 60,
                    "gpu_count": gpu_count,
                    "status": "completed",
                    "validation_status": "passed",
                    "wall_seconds": wall,
                    "peak_cpu_rss_mib": 200,
                    "gpu_peak_memory_used_mib_sum": 1000 * gpu_count,
                    "gpu_peak_incremental_memory_mib_sum": 800 * gpu_count,
                    "gpu_mean_utilization_percent": 70,
                    "gpu_energy_wh_estimate": 1.0,
                    "external_pairs_per_second": 50 / wall,
                    "external_window_cells_per_second": 5000 / wall,
                    "cpp_cpu_matrix_inversion_seconds": 1.0,
                    "cpp_reported_computation_seconds": wall - 2,
                    "matrix_output_bytes": 20000,
                }
                (run_dir / "stage_metrics.json").write_text(json.dumps(metric), encoding="utf-8")

            e2e_run = output_root / "runs/end_to_end/e2e_p50_g2_r01"
            (e2e_run / "metrics").mkdir(parents=True)
            for index, stage in enumerate(["cpp_kl", "filter_kl", "build_paths", "rescore_paths"]):
                metric = {
                    "run_id": e2e_run.name,
                    "run_type": "end_to_end",
                    "stage": stage,
                    "repeat": 1,
                    "pair_count": 50,
                    "window_cells": 5000,
                    "unique_families": 60,
                    "gpu_count": 2,
                    "status": "completed",
                    "validation_status": "passed",
                    "wall_seconds": float(index + 1),
                    "peak_cpu_rss_mib": 100 + index,
                    "gpu_peak_incremental_memory_mib_sum": 500 if stage == "cpp_kl" else 0,
                    "stage_output_bytes": 1000,
                    "clean_paths": 4 if stage == "build_paths" else None,
                    "final_path_count": 2 if stage == "rescore_paths" else None,
                }
                (e2e_run / "metrics" / f"{stage}.json").write_text(json.dumps(metric), encoding="utf-8")
            (e2e_run / "run_summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "run_id": e2e_run.name,
                        "total_measured_stage_wall_seconds": 10,
                        "final_path_count": 2,
                    }
                ),
                encoding="utf-8",
            )

            summarized = subprocess.run(
                [sys.executable, str(SCRIPTS / "04_summarize_benchmark.py"), "--config", str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(summarized.returncode, 0, msg=summarized.stdout)
            self.assertTrue((output_root / "results/benchmark_report.md").is_file())
            self.assertTrue((output_root / "results/tables/core_benchmark_summary.csv").is_file())
            self.assertTrue((output_root / "results/figures/runtime_vs_window_cells.png").is_file())


class MonitorTests(unittest.TestCase):
    def test_process_tree_rss_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage_dir = Path(tmp) / "monitor"
            result = run_monitored_command(
                command=[
                    sys.executable,
                    "-c",
                    "import time; x=bytearray(8*1024*1024); time.sleep(0.2); print(len(x))",
                ],
                stage_dir=stage_dir,
                cwd=Path(tmp),
                env=command_environment([]),
                gpu_device_ids=[],
                monitor_interval_seconds=0.05,
                timeout_seconds=5.0,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["returncode"], 0)
            self.assertGreater(result["peak_cpu_rss_mib"], 1.0)
            self.assertTrue((stage_dir / "resource_samples.csv").is_file())


class PreflightTests(unittest.TestCase):
    def test_nested_coverage_files_are_detected_recursively(self) -> None:
        preflight = load_script_module("00_preflight.py", "preflight_recursive_coverage_test")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "PF00001"
            nested.mkdir()
            (nested / "PF00001_coverage.csv").write_text("x\n", encoding="utf-8")
            self.assertEqual(preflight.count_coverage_files(root), 1)


class RawKLComparisonTests(unittest.TestCase):
    def test_raw_npy_comparison_reports_cell_errors_without_normalization(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpu_dir = root / "cpu"
            gpu_dir = root / "gpu"
            output_dir = root / "comparison"
            cpu_dir.mkdir()
            gpu_dir.mkdir()
            pair_list = root / "pairs.csv"
            pair_list.write_text("i,j\n1,2\n", encoding="utf-8")
            cpu = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            gpu = np.asarray([[1.0, 2.0], [3.0, 4.2]], dtype=np.float32)
            np.save(cpu_dir / "kl_0001_0002.npy", cpu)
            np.save(gpu_dir / "kl_0001_0002.npy", gpu)
            summary = run_comparison(
                pair_list=pair_list,
                cpu_dir=cpu_dir,
                gpu_dir=gpu_dir,
                output_dir=output_dir,
                rtol=0.0,
                atol=0.05,
            )
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(summary["normalization_or_downstream_filtering_included"])
            self.assertEqual(summary["total_compared_cells"], 4)
            self.assertEqual(summary["total_mismatch_cells"], 1)
            self.assertAlmostEqual(summary["pooled_mismatch_fraction"], 0.25)
            self.assertTrue((output_dir / "raw_kl_pair_comparison.csv").is_file())


class PackedCPUPerformanceRunnerTests(unittest.TestCase):
    def test_serial_and_multiprocess_packed_runs_match(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packed = root / "packed"
            packed.mkdir()
            fragment = 1
            m_states = 4
            n_obs = 3
            nfrag = 2
            pi_blocks = []
            A_blocks = []
            B_blocks = []
            index_rows = []
            pi_offset = A_offset = B_offset = 0
            for family_id in range(3):
                pi = np.asarray(
                    [
                        [0.40, 0.30, 0.20, 0.10],
                        [0.35, 0.25, 0.25, 0.15],
                    ],
                    dtype=np.float32,
                )
                pi = np.roll(pi, family_id, axis=1)
                A = np.empty((nfrag, m_states, m_states), dtype=np.float32)
                for window in range(nfrag):
                    A[window] = np.eye(m_states, dtype=np.float32) * (
                        0.08 + 0.01 * family_id
                    )
                    A[window] += np.float32(0.01 + 0.002 * window)
                B = np.asarray(
                    [
                        [0.50, 0.30, 0.20],
                        [0.20, 0.50, 0.30],
                        [0.30, 0.20, 0.50],
                        [0.40, 0.35, 0.25],
                    ],
                    dtype=np.float32,
                )
                B = np.stack(
                    [
                        np.roll(B, family_id, axis=1),
                        np.roll(B, family_id + 1, axis=1),
                    ]
                )
                pi_blocks.append(pi.reshape(-1))
                A_blocks.append(A.reshape(-1))
                B_blocks.append(B.reshape(-1))
                index_rows.append(
                    {
                        "family_id": family_id,
                        "fragment": fragment,
                        "nfrag": nfrag,
                        "pi_offset": pi_offset,
                        "pi_count": int(pi.size),
                        "A_offset": A_offset,
                        "A_count": int(A.size),
                        "B_offset": B_offset,
                        "B_count": int(B.size),
                    }
                )
                pi_offset += int(pi.size)
                A_offset += int(A.size)
                B_offset += int(B.size)

            np.concatenate(pi_blocks).astype(np.float32).tofile(
                packed / "pi_all.float32.bin"
            )
            np.concatenate(A_blocks).astype(np.float32).tofile(
                packed / "A_all.float32.bin"
            )
            np.concatenate(B_blocks).astype(np.float32).tofile(
                packed / "B_all.float32.bin"
            )
            with (packed / "index.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
                writer.writeheader()
                writer.writerows(index_rows)
            (packed / "metadata.json").write_text(
                json.dumps(
                    {
                        "fragment": fragment,
                        "m_states": m_states,
                        "n_obs": n_obs,
                        "num_datasets": 3,
                    }
                ),
                encoding="utf-8",
            )
            pair_list = root / "pairs.csv"
            with pair_list.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["i", "j", "window_cells", "sample_rank"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"i": 0, "j": 1, "window_cells": 4, "sample_rank": 1},
                        {"i": 1, "j": 2, "window_cells": 4, "sample_rank": 2},
                    ]
                )

            outputs = []
            for processes in (1, 2):
                output = root / f"cpu_{processes}"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS / "cpu_packed_runner.py"),
                        "--pair-list",
                        str(pair_list),
                        "--packed-db",
                        str(packed),
                        "--output-dir",
                        str(output),
                        "--fragment",
                        str(fragment),
                        "--acids-number",
                        str(n_obs),
                        "--processes",
                        str(processes),
                        "--save-matrices",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=60,
                )
                self.assertEqual(completed.returncode, 0, msg=completed.stdout)
                summary = json.loads(
                    (output / "cpu_packed_summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["pair_count"], 2)
                self.assertEqual(summary["window_cells"], 8)
                self.assertEqual(summary["cpu_process_count"], processes)
                outputs.append(output)

            for name in ("kl_0000_0001.npy", "kl_0001_0002.npy"):
                serial = np.load(outputs[0] / name)
                parallel = np.load(outputs[1] / name)
                self.assertEqual(serial.shape, (2, 2))
                self.assertTrue(np.isfinite(serial).all())
                np.testing.assert_allclose(serial, parallel, rtol=0, atol=0)


class CPUGPUComparisonTests(unittest.TestCase):
    def test_comparison_plan_and_summary(self) -> None:
        runner = load_script_module(
            "06_run_cpu_gpu_comparison.py", "cpu_gpu_comparison_runner_test"
        )
        plan = runner.configured_plan(
            {
                "cpu_gpu_comparison": {
                    "pair_counts": [20, 50],
                    "cpu_process_counts": [1],
                    "repeats": 2,
                    "parallel_scaling": {
                        "enabled": True,
                        "pair_count": 50,
                        "process_counts": [1, 2],
                    },
                }
            }
        )
        self.assertEqual(len(plan), 6)
        self.assertIn((20, 1, 1), plan)
        self.assertIn((50, 2, 2), plan)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packed = root / "packed"
            packed.mkdir()
            (packed / "metadata.json").write_text("{}\n", encoding="utf-8")
            output_root = root / "output"
            config = {
                "repo_root": str(root),
                "packed_db": str(packed),
                "output_root": str(output_root),
                "cpp_executable": str(root / "unused"),
                "mapping_file": "AUTO_FROM_PACKED_METADATA",
                "background_json": "EDIT_ME",
                "coverage_dir": "EDIT_ME",
                "cpu_gpu_comparison": {
                    "pair_counts": [20, 50],
                    "gpu_counts": [1],
                    "cpu_process_counts": [1],
                    "repeats": 2,
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            for pair_count, cells in ((20, 2000), (50, 5000)):
                for repeat in (1, 2):
                    cpu_dir = (
                        output_root
                        / "runs"
                        / "cpu_gpu_comparison"
                        / f"cpu_p{pair_count}_c1_r{repeat:02d}"
                    )
                    gpu_dir = (
                        output_root
                        / "runs"
                        / "core"
                        / f"core_p{pair_count}_g1_r{repeat:02d}"
                    )
                    cpu_dir.mkdir(parents=True)
                    gpu_dir.mkdir(parents=True)
                    cpu_wall = pair_count / 10.0 + repeat * 0.01
                    gpu_wall = pair_count / 40.0 + repeat * 0.01
                    cpu_metric = {
                        "status": "completed",
                        "validation_status": "passed",
                        "pair_count": pair_count,
                        "window_cells": cells,
                        "cpu_process_count": 1,
                        "wall_seconds": cpu_wall,
                        "external_window_cells_per_second": cells / cpu_wall,
                        "peak_cpu_rss_mib": 100.0,
                        "cpu_parallel_pair_stage_wall_seconds": cpu_wall * 0.8,
                    }
                    gpu_metric = {
                        "status": "completed",
                        "validation_status": "passed",
                        "sample_kind": "stratified_workload",
                        "pair_count": pair_count,
                        "window_cells": cells,
                        "gpu_count": 1,
                        "wall_seconds": gpu_wall,
                        "external_window_cells_per_second": cells / gpu_wall,
                        "peak_cpu_rss_mib": 80.0,
                        "gpu_peak_incremental_memory_mib_sum": 500.0,
                        "cpp_reported_computation_seconds": gpu_wall * 0.6,
                    }
                    (cpu_dir / "stage_metrics.json").write_text(
                        json.dumps(cpu_metric), encoding="utf-8"
                    )
                    (gpu_dir / "stage_metrics.json").write_text(
                        json.dumps(gpu_metric), encoding="utf-8"
                    )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "07_summarize_cpu_gpu_comparison.py"),
                    "--config",
                    str(config_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stdout)
            result_root = output_root / "results" / "cpu_gpu_comparison"
            self.assertTrue((result_root / "cpu_gpu_comparison_report.md").is_file())
            speedup_path = result_root / "tables" / "cpu_gpu_speedup_summary.csv"
            with speedup_path.open("r", encoding="utf-8", newline="") as handle:
                speedups = list(csv.DictReader(handle))
            self.assertEqual(len(speedups), 2)
            self.assertTrue(all(float(row["gpu_speedup_vs_cpu"]) > 3.0 for row in speedups))
            self.assertTrue(
                (result_root / "figures" / "cpu_gpu_runtime_scaling.png").is_file()
            )


class CoreRunnerTests(unittest.TestCase):
    def test_core_runner_accepts_unchanged_cli_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "benchmark_output"
            pair_dir = output_root / "design" / "pair_lists"
            pair_dir.mkdir(parents=True)
            (pair_dir / "pairs_2.csv").write_text(
                "i,j,nfrag_i,nfrag_j,window_cells,workload_stratum,sample_rank\n"
                "0,1,4,5,20,Q1,1\n"
                "1,2,5,6,30,Q2,2\n",
                encoding="utf-8",
            )
            (pair_dir / "pairs_full_family_coverage.csv").write_text(
                "i,j,nfrag_i,nfrag_j,window_cells,workload_stratum,sample_rank\n"
                "0,2,4,6,24,full_family_coverage,1\n"
                "1,3,5,7,35,full_family_coverage,2\n",
                encoding="utf-8",
            )
            packed = root / "packed"
            packed.mkdir()
            (packed / "metadata.json").write_text("{}\n", encoding="utf-8")
            executable = root / "mock_kl.py"
            executable.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import argparse, csv, json
                    from pathlib import Path
                    p=argparse.ArgumentParser()
                    p.add_argument('--pair_list'); p.add_argument('--packed_db')
                    p.add_argument('--num_streams'); p.add_argument('--max_gpus')
                    p.add_argument('--outdir'); p.add_argument('--memory_efficient', action='store_true')
                    a=p.parse_args(); out=Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
                    rows=list(csv.DictReader(open(a.pair_list, encoding='utf-8')))
                    for row in rows:
                        (out / f"kl_{int(row['i']):04d}_{int(row['j']):04d}.npy").write_bytes(b'NPY')
                    (out/'prodive_computed_pairs.csv').write_text('i,j\\n' + ''.join(f"{r['i']},{r['j']}\\n" for r in rows))
                    json.dump({'status':'completed','planned_pair_count':len(rows),'missing_output_count':0,'pairs_completed':len(rows),'computation_seconds':1}, open(out/'prodive_run_manifest.json','w'))
                    (out/'RUN_COMPLETE').write_text('completed\\n')
                    print('CPU matrix inversion completed in 25 ms')
                    print(f'Total pairs computed: {len(rows)}')
                    print('Average throughput: 2 pairs/second')
                    """
                ),
                encoding="utf-8",
            )
            executable.chmod(0o755)
            config = {
                "repo_root": str(root),
                "packed_db": str(packed),
                "output_root": str(output_root),
                "cpp_executable": str(executable),
                "mapping_file": "AUTO_FROM_PACKED_METADATA",
                "background_json": "EDIT_ME",
                "coverage_dir": "EDIT_ME",
                "sampling": {"seed": 7, "pair_counts": [2]},
                "core": {
                    "gpu_device_ids": [0],
                    "gpu_counts": [1],
                    "repeats": 1,
                    "num_streams": 8,
                    "memory_efficient": True,
                    "randomize_run_order": True,
                    "full_family_coverage": {
                        "enabled": True,
                        "gpu_counts": [1],
                        "repeats": 1,
                    },
                    "timeout_seconds": 5,
                },
                "cpu_gpu_comparison": {
                    "pair_counts": [2],
                    "gpu_counts": [1],
                    "repeats": 1,
                },
                "monitor": {"interval_seconds": 0.05},
                "cleanup": {"core_cpp_matrices_after_success": True},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "02_run_core_benchmark.py"),
                    "--config",
                    str(config_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout)
            metrics_path = output_root / "runs" / "core" / "core_p2_g1_r01" / "stage_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["validation_status"], "passed")
            self.assertEqual(metrics["matrix_output_file_count"], 2)
            self.assertEqual(metrics["cleanup"]["removed_file_count"], 2)
            coverage_metrics_path = (
                output_root
                / "runs"
                / "core"
                / "core_fullfamilies_p2_g1_r01"
                / "stage_metrics.json"
            )
            coverage_metrics = json.loads(coverage_metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(coverage_metrics["sample_kind"], "full_family_coverage")

            # The compare profile intentionally reuses an already validated
            # GPU point with the same pair count/GPU count/repeat.  This is
            # how existing 500-pair results remain compatible after upgrade.
            reused = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "02_run_core_benchmark.py"),
                    "--config",
                    str(config_path),
                    "--profile",
                    "compare",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=20,
            )
            self.assertEqual(reused.returncode, 0, msg=reused.stdout)
            self.assertIn("already completed; skipping", reused.stdout)


class EndToEndRunnerTests(unittest.TestCase):
    def test_real_postprocessing_scripts_reach_final_paths(self) -> None:
        """Use a tiny mock KL producer, then run the real three Python stages."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "output"
            pair_dir = output_root / "design" / "pair_lists"
            pair_dir.mkdir(parents=True)
            rows = [
                {
                    "i": i,
                    "j": j,
                    "nfrag_i": 12,
                    "nfrag_j": 12,
                    "window_cells": 144,
                    "workload_stratum": "Q1",
                    "sample_rank": rank,
                }
                for rank, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)], start=1)
            ]
            with (pair_dir / "pairs_3.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            packed = root / "packed"
            packed.mkdir()
            (packed / "metadata.json").write_text("{}\n", encoding="utf-8")
            mapping = root / "mapping.txt"
            mapping.write_text("PF00000: 0\nPF00001: 1\nPF00002: 2\n", encoding="utf-8")
            background = root / "background.json"
            background.write_text(
                json.dumps({f"PF{index:05d}": [1.0] * 12 for index in range(3)}),
                encoding="utf-8",
            )
            coverage = root / "coverage"
            coverage.mkdir()
            for index in range(3):
                with (coverage / f"PF{index:05d}_coverage.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "Match_State",
                            "MSA_Column",
                            "Residue_Count",
                            "Total_Seqs",
                            "Completeness_Percent",
                        ]
                    )
                    writer.writerows([[position, position, 10, 10, 100] for position in range(1, 40)])

            executable = root / "mock_diagonal_kl.py"
            executable.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import argparse, csv, json
                    from pathlib import Path
                    import numpy as np
                    p=argparse.ArgumentParser()
                    p.add_argument('--pair_list'); p.add_argument('--packed_db')
                    p.add_argument('--num_streams'); p.add_argument('--max_gpus')
                    p.add_argument('--outdir'); p.add_argument('--memory_efficient', action='store_true')
                    a=p.parse_args(); out=Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
                    rows=list(csv.DictReader(open(a.pair_list, encoding='utf-8')))
                    for row in rows:
                        matrix=np.full((int(row['nfrag_i']), int(row['nfrag_j'])), 3.0, dtype=np.float32)
                        np.fill_diagonal(matrix, 1.0)
                        np.save(out / f"kl_{int(row['i']):04d}_{int(row['j']):04d}.npy", matrix)
                    with open(out/'prodive_computed_pairs.csv', 'w', newline='') as handle:
                        writer=csv.writer(handle); writer.writerow(['i','j'])
                        writer.writerows([(row['i'], row['j']) for row in rows])
                    json.dump({'status':'completed','mode':'pair_list','completion_marker':'RUN_COMPLETE','computed_pairs_file':'prodive_computed_pairs.csv','planned_pair_count':len(rows),'missing_output_count':0,'pairs_completed':len(rows),'computation_seconds':1}, open(out/'prodive_run_manifest.json','w'))
                    (out/'RUN_COMPLETE').write_text('completed\\n')
                    print('CPU matrix inversion completed in 10 ms')
                    print(f'Total pairs computed: {len(rows)}')
                    print('Average throughput: 3 pairs/second')
                    """
                ),
                encoding="utf-8",
            )
            executable.chmod(0o755)

            config = {
                "repo_root": str(BENCHMARK_ROOT.parent),
                "packed_db": str(packed),
                "cpp_executable": str(executable),
                "mapping_file": str(mapping),
                "background_json": str(background),
                "coverage_dir": str(coverage),
                "output_root": str(output_root),
                "sampling": {"seed": 1, "pair_counts": [3]},
                "core": {"gpu_device_ids": [0], "gpu_counts": [1], "repeats": 1},
                "end_to_end": {
                    "enabled": True,
                    "pair_count": 3,
                    "gpu_count": 1,
                    "repeats": 1,
                    "num_streams": 8,
                    "path_workers": 1,
                    "path_strict": True,
                    "make_score_distribution_plot": False,
                    "timeout_seconds_per_stage": 20,
                },
                "thresholds": {
                    "fragment": 6,
                    "raw_threshold": 2.0,
                    "filter_threshold": 13.0,
                    "epsilon": 0.0001,
                    "row_block_size": 1024,
                    "min_path_points": 5,
                    "overlap_iou_threshold": 0.1,
                    "coverage_threshold": 0.85,
                    "score_threshold": 1.2,
                    "missing_coverage_default": 0.0,
                },
                "monitor": {"interval_seconds": 0.05},
                "cleanup": {
                    "end_to_end_cpp_matrices_after_success": False,
                    "end_to_end_filtered_pickles_after_success": False,
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "03_run_end_to_end_benchmark.py"),
                    "--config",
                    str(config_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout)
            summary_path = output_root / "runs/end_to_end/e2e_p3_g1_r01/run_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["final_path_count"], 3)
            self.assertTrue(
                (output_root / "runs/end_to_end/e2e_p3_g1_r01/08_final_paths/final_paths.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
