# ESM2 entropy analysis

This module prepares representative sequences, runs masked-token ESM2 inference, converts probability outputs to entropy, compares ESM2 entropy with Pfam seed MSA conservation, and stratifies entropy by secondary structure and solvent accessibility.

## Inputs

| Input | Source |
|---|---|
| Pfam-Pfam ProDive result | `$PRODIVE_DATA_ROOT/shared/global_high_score_summary_fin.csv` |
| de novo-Pfam ProDive result | `$PRODIVE_DATA_ROOT/shared/denovo_global_high_score_summary_fin.csv` |
| Pfam seed HHM/STO/FAS files | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` |
| Representative-fragment FASTA or fragment-range tables | Released ProDive data tables or outputs from the preparation scripts |
| Structure annotations | Outputs from `prodive_secondary_structure_rsa` when structural stratification is required |

## Workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/01_prepare_pfam_representatives.py` | Pfam-Pfam ProDive result and Pfam seed database | Pfam representative sequence and fragment-range tables |
| 2 | `scripts/01b_prepare_denovo_representatives.py` | de novo-Pfam ProDive result and de novo sequence data | de novo representative sequence and fragment-range tables |
| 3 | `scripts/02_split_fragdom_record_ranges.py` | Representative range table | Batched range files for ESM2 inference |
| 4 | `scripts/03_run_masked_esm2_inference.py` | Representative FASTA and range files | Masked-token probability outputs |
| 5 | `scripts/04_calculate_entropy_from_probabilities.py` | ESM2 probability outputs | Per-position ESM2 entropy tables |
| 6 | `scripts/05_compare_esm2_vs_msa_conservation.py` | ESM2 entropy and Pfam seed MSA conservation | Correlation and overlap summaries |
| 7 | `scripts/06_fragment_vs_background_after_msa_filtering.py` | Entropy/conservation tables and fragment intervals | Fragment-vs-background entropy summaries |
| 8 | `scripts/07_check_structure_availability.py` | Representative tables and structure directories | Structure-availability report |
| 9 | `scripts/08_entropy_by_secondary_structure_rsa.py` | Entropy and SS/RSA tables | Entropy summaries stratified by structural context |

Run templates:

```bash
cd prodive_esm2
bash run_templates/run_pfam_esm2_pipeline.sh
bash run_templates/run_denovo_esm2_pipeline.sh
```

## Notes

ESM2 inference can be computationally demanding. For large batches, use a GPU-enabled environment and set batch sizes according to available memory.
