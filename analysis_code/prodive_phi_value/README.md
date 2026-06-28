# Phi-value overlap analysis

This module maps folding phi-value residues from PFDB-related resources to Pfam seed coordinates and extracts ProDive fragment intervals that overlap the mapped residues.

## Inputs

| Input | Source |
|---|---|
| PFDB/PDB residue mapping resources | `$PRODIVE_DATA_ROOT/phi_value/` |
| Pfam seed HHM/STO/FAS files | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` |
| Pfam-Pfam ProDive result | `$PRODIVE_DATA_ROOT/shared/global_high_score_summary_fin.csv` |

## Workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/01_match_pfdb_seed_sifts_residues.py` | PFDB/PDB resources, SIFTS mapping, and Pfam seed database | PFDB residue-to-Pfam seed mapping tables |
| 2 | `scripts/02_check_phi_pfam_ids_in_missing_list.py` | Step 1 mapping results and Pfam ID lists | Coverage and missing-family reports |
| 3 | `scripts/03_extract_global_score_rows_for_phi_pfams.py` | ProDive result and mapped Pfam IDs | ProDive rows involving phi-value-associated Pfam families |
| 4 | `scripts/04_map_phi_related_fragments_to_seed_intervals.py` | Step 1 and Step 3 outputs | ProDive fragment intervals mapped to phi-value-related seed coordinates |

Run template:

```bash
cd prodive_phi_value
bash run_templates/run_phi_value_pipeline.sh
```
