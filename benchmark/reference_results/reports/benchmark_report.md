# ProDive computational benchmark report

## Scope

This report measures the unchanged production workflow on deterministic sampled family pairs.  It reports the complete C++/CUDA stage (packed-database loading, CPU matrix inversion, GPU calculation, and NPY writing) separately from filtering, path construction, and coverage rescoring.  HHM parsing and packed-database construction are preprocessing and are not included in the timed end-to-end path.

## Hardware and software

- Host: reference workstation (hostname removed)
- CPU: Intel(R) Xeon(R) CPU E5-2696 v3 @ 2.30GHz
- Logical CPUs: 72
- RAM: 62.75 GiB
- GPU(s): GPU 0: NVIDIA GeForce RTX 3090 (24,576 MiB), GPU 1: NVIDIA GeForce RTX 3090 (24,576 MiB)
- Python: 3.12.2
- Monitor interval: 0.2 s

## Sampling design

- Fixed random seed: `20260712`
- Pair counts: 20, 50, 100, 200, 500, 2,000, 10,000
- Repeats per core configuration: 3
- Candidate pairs: 500,000
- Workload strata: 5
- Sampling unit: unordered, non-self family pair
- Workload unit: `nfrag_i × nfrag_j` KL-matrix cells per family pair
- Full database families: 25,545
- Full unordered non-self pairs: 326,260,740
- Full unordered non-self window cells: 7,877,409,782,701

The sampled pair lists are nested prefixes.  Candidate pairs were divided into quantile strata according to `nfrag_i × nfrag_j`, shuffled with the fixed seed, and interleaved across strata.  This avoids treating one small-family pair as computationally equivalent to one large-family pair.

## Core C++/CUDA results

| Sample | Pairs | GPUs | Repeats | Wall time median (s) | Window cells/s median | Peak CPU RSS (MiB) | Incremental GPU memory (MiB) | Speedup vs 1 GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full-family coverage | 12,773 | 2 | 1 | 324.366 | 945,772 | 17,462.582 | 28,410.000 | NA |
| Stratified scaling | 20 | 1 | 3 | 0.925 | 526,341 | 127.902 | 14,165.000 | 1.000 |
| Stratified scaling | 20 | 2 | 3 | 0.846 | 575,367 | 210.488 | 28,332.000 | 1.093 |
| Stratified scaling | 50 | 1 | 3 | 1.702 | 641,622 | 164.977 | 14,165.000 | 1.000 |
| Stratified scaling | 50 | 2 | 3 | 1.308 | 834,776 | 247.980 | 28,400.000 | 1.301 |
| Stratified scaling | 100 | 1 | 3 | 2.845 | 873,994 | 237.602 | 14,173.000 | 1.000 |
| Stratified scaling | 100 | 2 | 3 | 2.401 | 1,035,881 | 320.273 | 28,348.000 | 1.185 |
| Stratified scaling | 200 | 1 | 3 | 5.733 | 873,778 | 370.398 | 14,269.000 | 1.000 |
| Stratified scaling | 200 | 2 | 3 | 4.037 | 1,240,901 | 453.395 | 28,348.000 | 1.420 |
| Stratified scaling | 500 | 1 | 3 | 13.785 | 893,159 | 769.676 | 14,175.000 | 1.000 |
| Stratified scaling | 500 | 2 | 3 | 9.580 | 1,285,228 | 852.770 | 28,350.000 | 1.439 |
| Stratified scaling | 2,000 | 1 | 3 | 53.235 | 904,139 | 2,594.324 | 14,177.000 | 1.000 |
| Stratified scaling | 2,000 | 2 | 3 | 35.604 | 1,351,845 | 2,678.594 | 28,354.000 | 1.495 |
| Stratified scaling | 10,000 | 1 | 3 | 244.349 | 988,569 | 9,506.750 | 14,205.000 | 1.000 |
| Stratified scaling | 10,000 | 2 | 3 | 153.467 | 1,573,991 | 9,589.555 | 28,410.000 | 1.592 |

The wall time above is measured outside the executable and therefore includes packed-data loading, CPU matrix inversion, GPU computation, synchronization, and NPY output.  The raw resource samples and the executable's own `computation_seconds` and matrix-inversion timing are retained separately.  The full-family coverage sample uses approximately half as many pairs as families and touches every packed family at least once; it estimates the full-family loading/inversion memory requirement without all-pairs computation and is not used for runtime-scaling extrapolation.

## Serial CPU versus GPU raw-KL validation

| Run | Pairs | Window cells | CPU wall (s) | GPU wall (s) | Whole-stage speedup | CPU raw compute (s) | GPU reported compute (s) | Raw-compute speedup | CPU peak RSS (MiB) | GPU incremental memory (MiB) | Mismatch fraction | RMSE | Pearson r | Acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rawkl_cpu_gpu_p5_g1_r01 | 5 | 27,733 | 5.294 | 0.540 | 9.809 | 0.697 | 0.000 | NA | 271.820 | 14,159.000 | 0.000000 | 0.000001 | 1.000000 | True |

This comparison uses exactly the same deterministic family pairs and compares only `0.5 × (KL(1||2) + KL(2||1))` in the GPU-compatible matrix orientation.  It does **not** divide by `(3 × fragment + 2)` and does not include background normalization, thresholding, path construction, or coverage rescoring.  The CPU implementation is forced to one BLAS/OpenMP/Numba thread.  The GPU wall time includes the complete unchanged C++/CUDA stage; the CPU wall time includes HHM loading, fragment construction, CPU inversion, raw-KL calculation, and NPY writing, so the comparison must be described with that boundary.

## End-to-end stage results

| Stage | Completed repeats | Skipped repeats | Wall time median (s) | Peak CPU RSS (MiB) | Incremental GPU memory (MiB) |
| --- | --- | --- | --- | --- | --- |
| C++/CUDA KL | 3 | 0 | 149.869 | 9,589.391 | 28,410.000 |
| KL filtering | 3 | 0 | 7.504 | 250.555 | 0.000 |
| Path construction | 3 | 0 | 17.991 | 1,092.691 | 0.000 |
| Coverage rescoring | 3 | 0 | 1.949 | 188.363 | 0.000 |

### Final path outputs by repeat

| Run | Status | Measured stage time (s) | Final paths |
| --- | --- | --- | --- |
| e2e_p10000_g2_r01 | completed | 181.983 | 14 |
| e2e_p10000_g2_r02 | completed | 176.229 | 14 |
| e2e_p10000_g2_r03 | completed | 177.584 | 14 |

If a sampled workload produces no filtered pickle or no clean path, the downstream stage is recorded as skipped rather than assigned a fabricated runtime.  Increase `end_to_end.pair_count` if this occurs in all repeats.

## Full-scale workload projection

| GPUs | Sample pairs used | Observed window cells/s | Projected days |
| --- | --- | --- | --- |
| 1 | 10,000 | 988,569 | 92.228 |
| 2 | 10,000 | 1,573,991 | 57.925 |

These values are throughput extrapolations, not full-scale measurements. Full-run batching, file-system contention, scheduling, setup costs, and output-file count can change the realized runtime.

## Reporting cautions

- Report medians and the interquartile range across repeats; do not select the fastest repeat.
- State whether GPU memory is total used memory or incremental memory above the pre-run baseline.  This report provides both in the raw metrics.
- Report window cells per second alongside pairs per second.
- Describe the full-scale duration only as an extrapolation unless a full run was actually measured.
- Run on reserved GPUs and an otherwise idle host; external GPU jobs contaminate memory and utilization measurements.
