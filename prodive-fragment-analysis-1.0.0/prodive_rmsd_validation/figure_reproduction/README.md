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
| 1 | `../structural_validation/scripts/07_plot_rmsd_density_panels.py` | Config CSV listing RMSD tables | RMSD validation and supplementary density panels |
| 2 | `../structural_validation/scripts/10_compare_real_random_core.py` | Real and random RMSD tables | Random-control boxplots and summary tables |
| 3 | `../structural_validation/scripts/11_welch_ttest_from_stats.py` | Real-versus-random summary tables | Welch t-test table |
| 4 | `scripts/run_figure_reproduction.py` | Figure configuration files | Complete figure-reproduction run |

The example configuration files in `configs/` define the expected column layout and should be copied and edited to use local or released RMSD tables before execution. The wrapper reuses the structural-validation plotting and statistics scripts instead of maintaining duplicate implementations.
