# ProDive benchmark

This module measures GPU performance, CPU/GPU numerical agreement, CPU/GPU scaling, and the sampled end-to-end workflow.

## Setup

```bash
python3 -m pip install -r benchmark/requirements.txt
make -C src/cpp_cuda -j
cp benchmark/config.example.json benchmark/config.json
```

Edit these fields in `benchmark/config.json`:

```json
{
  "packed_db": "/path/to/kl_fragment_6_packed",
  "background_json": "/path/to/background_kl_fragment_6.json",
  "coverage_dir": "/path/to/coverage_tables",
  "output_root": "/path/to/benchmark_output"
}
```

Also set `gpu_device_ids`, `gpu_counts`, and `end_to_end.gpu_count` for the available GPUs.

Check the configuration before running:

```bash
python3 benchmark/scripts/00_preflight.py \
  --config benchmark/config.json \
  --mode all
```

## Run modes

```bash
bash benchmark/run_server_once.sh MODE
```

| Mode | Operations |
|---|---|
| `gpu` | GPU scaling, full-family memory sample, and end-to-end timing |
| `compare` | CPU/GPU raw-KL validation and packed-input scaling |
| `all` | run `gpu` and `compare` |
| `core` | GPU KL benchmark only |
| `validation` | five-pair CPU/GPU numerical validation only |
| `summary` | rebuild reports, tables, and figures from existing results |

Completed and validated runs are skipped automatically.

## Script reference

| Script | Function | Main output |
|---|---|---|
| `run_server_once.sh` | check the local configuration, compile the GPU program when needed, and launch one mode | selected benchmark workflow |
| `run_benchmark.sh` | dispatch individual benchmark stages by mode | stage-specific outputs |
| `scripts/00_preflight.py` | validate input files, Python packages, executable, GPU IDs, and sample sizes | `preflight_report.json`, hardware inventory |
| `scripts/01_prepare_benchmark.py` | generate deterministic nested pair lists and workload summaries | `design/` |
| `scripts/02_run_core_benchmark.py` | run sampled C++/CUDA KL calculations and monitor resources | `runs/core/` |
| `scripts/03_run_end_to_end_benchmark.py` | run KL calculation, filtering, path building, and rescoring | `runs/end_to_end/` |
| `scripts/04_summarize_benchmark.py` | aggregate core, validation, and end-to-end measurements | main report, tables, and figures |
| `scripts/05_run_cpu_gpu_validation.py` | calculate the same bounded pairs on CPU and GPU and compare raw KL matrices | `runs/cpu_gpu_validation/` |
| `scripts/06_run_cpu_gpu_comparison.py` | measure CPU-process and GPU-count scaling on identical packed inputs | `runs/cpu_gpu_comparison/` |
| `scripts/07_summarize_cpu_gpu_comparison.py` | aggregate CPU/GPU runtime, throughput, speedup, and linear fits | comparison report, tables, and figures |
| `scripts/cpu_batch_runner.py` | calculate sampled raw-KL matrices from HHM files on one CPU core | CPU NPY matrices and timing JSON |
| `scripts/cpu_packed_runner.py` | calculate raw-KL matrices directly from the packed database on CPU | CPU NPY matrices and timing JSON |
| `scripts/raw_kl_compare.py` | compare corresponding CPU and GPU NPY matrices | error statistics JSON |
| `scripts/resource_wrapper.py` | measure wall time, process-tree RSS, and GPU resource use for one command | stage metrics and raw samples |
| `scripts/benchmark_common.py` | shared configuration, path, validation, and sampling utilities | imported by benchmark scripts |
| `tests/test_benchmark_harness.py` | test sampling, monitoring, comparison, restart behavior, and end-to-end execution | unit-test results |

Use `python3 benchmark/scripts/<script>.py --help` for stage-specific arguments.

## Staged execution

Prepare pair lists:

```bash
python3 benchmark/scripts/01_prepare_benchmark.py \
  --config benchmark/config.json
```

Run one GPU pilot:

```bash
python3 benchmark/scripts/02_run_core_benchmark.py \
  --config benchmark/config.json \
  --profile gpu \
  --pair-count 500 \
  --gpu-count 1 \
  --repeat 1
```

Rebuild summaries:

```bash
python3 benchmark/scripts/04_summarize_benchmark.py \
  --config benchmark/config.json

python3 benchmark/scripts/07_summarize_cpu_gpu_comparison.py \
  --config benchmark/config.json
```

## Measurement definitions

| Metric | Definition |
|---|---|
| wall time | elapsed time of the complete launched stage |
| peak CPU RSS | maximum summed resident memory of the process tree |
| incremental GPU memory | peak GPU memory above the pre-stage baseline |
| pair throughput | family pairs divided by stage wall time |
| cell throughput | sum of `nfrag_i * nfrag_j` divided by stage wall time |
| mismatch fraction | fraction of raw-KL cells outside `atol + rtol * abs(GPU)` |

The numerical comparison uses raw symmetric KL only. It excludes background normalization, thresholding, path building, and coverage rescoring.

## Output layout

```text
<output_root>/
├── preflight_report.json
├── hardware_inventory.json
├── design/
├── runs/
│   ├── core/
│   ├── cpu_gpu_validation/
│   ├── cpu_gpu_comparison/
│   └── end_to_end/
└── results/
    ├── tables/
    ├── figures/
    └── cpu_gpu_comparison/
```

`benchmark/reference_results/` contains the exact pair lists and the generated reports, tables, and figures from the reference run. Full raw logs and intermediate matrices are not included.

## Tests

```bash
python3 -m unittest discover -s benchmark/tests -v
```
