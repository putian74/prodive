# Secondary structure and RSA analysis

This module annotates ProDive fragments with secondary-structure class and relative solvent accessibility (RSA), then compares the observed distributions with matched random controls.

## Inputs

| Input | Source |
|---|---|
| Pfam-Pfam ProDive result | `$PRODIVE_DATA_ROOT/shared/global_high_score_summary_fin.csv` |
| de novo-Pfam ProDive result | `$PRODIVE_DATA_ROOT/shared/denovo_global_high_score_summary_fin.csv` |
| Pfam seed HHM/STO/FAS files | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` |
| Pfam structure runtime directory | `$PRODIVE_PFAM_RUNTIME_ROOT` |
| de novo structures | `$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/downloaded_denovo_pdbs/` |
| de novo sequence FASTA | `$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/all_1927_sequences.fasta` |

DSSP or mkdssp must be available for structure annotation.

## Pfam-Pfam workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/pfam/01_compute_pfam_real_ss_rsa.py` | Pfam-Pfam result and Pfam runtime directory | Real ProDive fragment SS/RSA table |
| 2 | `scripts/pfam/02_build_pfam_random_background_sqlite.py` | Step 1 output and Pfam runtime directory | SQLite random-background database |
| 3 | `scripts/pfam/03_plot_pfam_real_vs_random.py` | Random-background SQLite database | Pfam real-vs-random SS/RSA figures |
| 4 | `scripts/pfam/04_plot_pfam_top_score_subsets.py` | Real SS/RSA table and ProDive score subsets | Top-score subset SS/RSA figures |

Run template:

```bash
cd prodive_secondary_structure_rsa
bash run_templates/run_pfam_secondary_structure_rsa.sh
```

## de novo-Pfam workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/denovo/01_compute_denovo_real_ss_rsa.py` | de novo-Pfam result, de novo structures, and FASTA | Real de novo-side SS/RSA table |
| 2 | `scripts/denovo/02_build_denovo_random_controls.py` | de novo-Pfam result, de novo structures, and FASTA | Matched random-control table |
| 3 | `scripts/denovo/03_plot_denovo_real_vs_random.py` | Real and random-control tables | de novo real-vs-random SS/RSA figure |

Run template:

```bash
cd prodive_secondary_structure_rsa
bash run_templates/run_denovo_secondary_structure_rsa.sh
```
