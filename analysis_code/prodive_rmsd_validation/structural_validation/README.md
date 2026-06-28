# Structural validation and random controls

This submodule computes local RMSD values for ProDive/HHsearch comparison classes, de novo-Pfam pairs, and length-stratified random controls.

## Inputs

| Input | Source |
|---|---|
| HHsearch/ProDive task tables | Output of `../hhsearch_comparison/` |
| Pfam seed HHM/STO/FAS files | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` |
| Pfam structure runtime directory | `$PRODIVE_PFAM_RUNTIME_ROOT` |
| de novo-Pfam result | `$PRODIVE_DATA_ROOT/shared/denovo_global_high_score_summary_fin.csv` |
| de novo structures | `$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/downloaded_denovo_pdbs/` |
| de novo FASTA | `$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/all_1927_sequences.fasta` |

## Workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/01_rmsd_shared_prodive_segments.py` | Shared ProDive-boundary task table | Shared ProDive-boundary RMSD table |
| 2 | `scripts/02_rmsd_hhsearch_only.py` | HHsearch-only task table | HHsearch-only RMSD table |
| 3 | `scripts/03_rmsd_shared_hhsearch_segments.py` | Shared HHsearch-boundary task table | Shared HHsearch-boundary RMSD table |
| 4 | `scripts/04_rmsd_prodive_only_pipeline.py` | ProDive-only task table | ProDive-only RMSD table |
| 5 | `scripts/05_rmsd_denovo_pfam.py` | de novo-Pfam result and structures | de novo-Pfam RMSD table |
| 6 | `scripts/06_filter_rmsd_by_hhsearch_probability.py` | HHsearch CSVs and RMSD tables | Probability-filtered RMSD tables |
| 7 | `scripts/07_plot_rmsd_density_panels.py` | RMSD tables | RMSD density panels |
| 8 | `scripts/08_sample_random_pfam_pairs.py` | Pfam runtime directory | Random Pfam-Pfam RMSD table |
| 9 | `scripts/09_sample_random_denovo_pfam_pairs.py` | de novo structures and Pfam runtime directory | Random de novo-Pfam RMSD table |
| 10 | `scripts/10_compare_real_random_core.py` | Real and random RMSD tables | Core-region real-vs-random summaries and plots |
| 11 | `scripts/11_welch_ttest_from_stats.py` | Step 10 summary tables | Welch t-test table |

Structure alignment scripts require PyMOL in the runtime environment.
