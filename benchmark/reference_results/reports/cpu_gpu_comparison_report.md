# CPU/GPU raw-KL performance comparison

## Scope

CPU and GPU start from the same packed float32 fragment database and use the same deterministic pair lists. The measured quantity is the raw symmetric KL matrix before background normalization, thresholds, path construction, or coverage rescoring.

Complete-stage wall time includes selected-family loading, CPU matrix inversion, raw-KL calculation, and NPY output. CPU numerical-library threads are fixed to one per process.

## Design

- Pair counts: [20, 50, 100, 200, 500]
- CPU process counts across the size series: [1]
- GPU counts: [1, 2]
- Repeats: 3

## Measured performance

| Backend | Devices/processes | Pairs | Matrix cells | Median wall (s) | Median cells/s | Median host RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 1 | 20 | 487,045 | 12.975 | 37535.8 | 263.6 |
| GPU | 1 | 20 | 487,045 | 0.925 | 526341.1 | 127.9 |
| GPU | 2 | 20 | 487,045 | 0.846 | 575366.9 | 210.5 |
| CPU | 1 | 50 | 1,091,934 | 29.048 | 37590.4 | 301.5 |
| GPU | 1 | 50 | 1,091,934 | 1.702 | 641622.2 | 165.0 |
| GPU | 2 | 50 | 1,091,934 | 1.308 | 834775.6 | 248.0 |
| CPU | 1 | 100 | 2,486,781 | 61.566 | 40391.9 | 371.0 |
| GPU | 1 | 100 | 2,486,781 | 2.845 | 873993.7 | 237.6 |
| GPU | 2 | 100 | 2,486,781 | 2.401 | 1035880.7 | 320.3 |
| CPU | 1 | 200 | 5,009,275 | 128.329 | 39034.5 | 500.6 |
| GPU | 1 | 200 | 5,009,275 | 5.733 | 873778.4 | 370.4 |
| GPU | 2 | 200 | 5,009,275 | 4.037 | 1240901.3 | 453.4 |
| CPU | 1 | 500 | 12,311,926 | 325.193 | 37860.3 | 881.9 |
| CPU | 2 | 500 | 12,311,926 | 177.078 | 69528.1 | 1555.9 |
| CPU | 4 | 500 | 12,311,926 | 97.117 | 126774.1 | 1897.9 |
| CPU | 8 | 500 | 12,311,926 | 54.169 | 227287.0 | 2573.8 |
| CPU | 16 | 500 | 12,311,926 | 32.441 | 379512.8 | 3930.9 |
| GPU | 1 | 500 | 12,311,926 | 13.785 | 893159.0 | 769.7 |
| GPU | 2 | 500 | 12,311,926 | 9.580 | 1285228.3 | 852.8 |

## Speedup

| Pairs | Matrix cells | CPU processes | GPUs | GPU speedup |
|---:|---:|---:|---:|---:|
| 20 | 487,045 | 1 | 1 | 14.02x |
| 20 | 487,045 | 1 | 2 | 15.33x |
| 50 | 1,091,934 | 1 | 1 | 17.07x |
| 50 | 1,091,934 | 1 | 2 | 22.21x |
| 100 | 2,486,781 | 1 | 1 | 21.64x |
| 100 | 2,486,781 | 1 | 2 | 25.65x |
| 200 | 5,009,275 | 1 | 1 | 22.38x |
| 200 | 5,009,275 | 1 | 2 | 31.79x |
| 500 | 12,311,926 | 1 | 1 | 23.59x |
| 500 | 12,311,926 | 1 | 2 | 33.95x |
| 500 | 12,311,926 | 2 | 1 | 12.85x |
| 500 | 12,311,926 | 2 | 2 | 18.49x |
| 500 | 12,311,926 | 4 | 1 | 7.05x |
| 500 | 12,311,926 | 4 | 2 | 10.14x |
| 500 | 12,311,926 | 8 | 1 | 3.93x |
| 500 | 12,311,926 | 8 | 2 | 5.65x |
| 500 | 12,311,926 | 16 | 1 | 2.35x |
| 500 | 12,311,926 | 16 | 2 | 3.39x |

## Interpretation limits

- Pair count is not the computational workload; use total matrix cells when interpreting scaling.
- CPU and GPU wall times are approximately linear only over the measured range. Linear fits are descriptive and are not substitutes for a full all-pairs run.
- GPU speedup may rise while fixed setup costs are amortized, then approach a plateau after device utilization saturates.
- Multi-process host RSS is the summed process-tree RSS and may count shared copy-on-write pages more than once.

## Descriptive linear fits

| Backend | Devices/processes | Seconds per cell | R² |
|---|---:|---:|---:|
| CPU | 1 | 2.644926e-05 | 0.9997 |
| GPU | 1 | 1.087454e-06 | 0.9993 |
| GPU | 2 | 7.353543e-07 | 0.9995 |
