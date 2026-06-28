# Interface-context analysis

This module evaluates whether de novo-side ProDive fragments are located in protein-interface contexts and compares the observed distribution with same-chain random controls.

## Inputs

| Input | Source |
|---|---|
| de novo-Pfam ProDive result | `$PRODIVE_DATA_ROOT/shared/denovo_global_high_score_summary_fin.csv` |
| de novo structures | `$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/downloaded_denovo_pdbs/` |
| de novo sequence FASTA | `$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/all_1927_sequences.fasta` |

## Workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/01_denovo_interface_random_control.py` | de novo-Pfam ProDive result and de novo structures | Interface classification tables and matched random-control summaries |

Run template:

```bash
cd prodive_interface_analysis
bash run_templates/run_denovo_interface_analysis.sh
```
