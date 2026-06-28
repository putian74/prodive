# Figure generation from RMSD validation outputs

This submodule generates RMSD validation figures and real-vs-random statistical summaries from the structural-validation output tables.

## Inputs

| Input | Source |
|---|---|
| RMSD validation tables | Output of `../structural_validation/` or released precomputed tables |
| Figure configuration CSVs | Local files under `configs/`, edited to point to the relevant tables |

## Workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/plot_fig2_and_s3_panels.py` | Config CSV listing RMSD tables | RMSD validation and supplementary density panels |
| 2 | `scripts/plot_real_vs_random_core_boxplot.py` | Real-vs-random statistics | Random-control boxplots |
| 3 | `scripts/welch_ttest_from_stats.py` | Real-vs-random statistics | Welch t-test summary table |
| 4 | `scripts/run_figure_reproduction.py` | Figure configuration files | Complete figure-reproduction run |

The example configuration files in `configs/` define the expected column layout and should be edited to local paths before execution.
