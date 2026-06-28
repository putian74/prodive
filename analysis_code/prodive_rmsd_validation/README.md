# RMSD validation and HHsearch comparison

This module reproduces the structural validation workflow for ProDive. It parses HHsearch outputs, constructs ProDive/HHsearch comparison tasks, computes local structural RMSD values, builds random controls, and generates RMSD validation figures.

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
| 3 | `figure_reproduction/` | RMSD validation figures and real-vs-random statistical summaries |

Run Stage 1, then Stage 2, then Stage 3. Detailed commands are documented in the README files inside each subdirectory.
