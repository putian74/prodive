# RMSD validation and HHsearch comparison

This module parses HHsearch outputs, constructs ProDive/HHsearch comparison classes, computes local structural RMSD values, builds random controls, and generates RMSD validation figures.

## Inputs

| Input | Source |
|---|---|
| HHsearch `.hhr` files | `$PRODIVE_DATA_ROOT/rmsd/hhsearch_hhr/` |
| Pfam-Pfam ProDive result | `$PRODIVE_DATA_ROOT/shared/global_high_score_summary_fin.csv` |
| de novo-Pfam ProDive result | `$PRODIVE_DATA_ROOT/shared/denovo_global_high_score_summary_fin.csv` |
| Pfam seed HHM/STO/FAS files | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` |
| Pfam structure runtime directory | `$PRODIVE_PFAM_RUNTIME_ROOT` |
| de novo structures and FASTA | `$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/` |

## Workflow stages

| Stage | Directory | Main output |
|---:|---|---|
| 1 | `hhsearch_comparison/` | Parsed HHsearch tables, percentile subsets, overlap reports, and ProDive/HHsearch task tables |
| 2 | `structural_validation/` | RMSD tables for ProDive-only, HHsearch-only, shared-boundary comparisons, de novo-Pfam validation, and random controls |
| 3 | `figure_reproduction/` | RMSD validation figures and real-versus-random statistical summaries |

Set the common paths before running the workflow:

```bash
export PRODIVE_DATA_ROOT=/path/to/ProDive_release/data
export PRODIVE_WORK_ROOT=/path/to/local/output
export PRODIVE_PFAM_RUNTIME_ROOT=/path/to/PfamA_seed_runtime
```

`PRODIVE_PFAM_RUNTIME_ROOT` must expose the profile/alignment files from `data/shared/PfamA_seed/` and the reconstructed structure files from `PfamA_seed_structure/` under the same `PFxxxxx/` family directories.

Run the stages in order:

```bash
bash hhsearch_comparison/run_hhsearch_preprocessing.sh
bash structural_validation/run_structural_validation.sh
python3 figure_reproduction/scripts/run_figure_reproduction.py \
  --fig2-config /path/to/fig2_datasets.csv \
  --random-config /path/to/random_control_datasets.csv \
  --output-dir /path/to/figures
```

The released task tables, RMSD tables, random controls, and figures are available under `data/rmsd/` and `precomputed_results/rmsd/` when recomputation is unnecessary.
