# DisProt disorder-overlap analysis

This module maps DisProt annotated regions to Pfam seed coordinates, maps ProDive HMM segments to seed residue intervals, and compares observed disorder overlap against randomized expectations.

## Inputs

| Input | Source |
|---|---|
| DisProt TSV | `$PRODIVE_DATA_ROOT/disorder_overlap/DisProt_current_IDPO.tsv` |
| Pfam seed HHM/STO/FAS database | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` |
| Pfam-Pfam ProDive result | `$PRODIVE_DATA_ROOT/shared/global_high_score_summary_fin.csv` |

## Workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/01_compare_disprot_regions_to_pfam_seed.py` | DisProt annotations and Pfam seed database | DisProt-to-seed match tables |
| 2 | `scripts/02_map_hmm_segments_to_disprot_regions.py` | Step 1 tables, ProDive result, and Pfam seed database | ProDive segment-to-disorder overlap tables |
| 3 | `scripts/03_compute_pfam_disorder_baseline.py` | DisProt annotations, seed matches, and mapped segments | Pfam-level disorder baseline tables |
| 4 | `scripts/04_pfam_disorder_observed_expected.py` | DisProt annotations, seed matches, and mapped segments | Observed/expected disorder-overlap summaries |

Run template:

```bash
cd prodive_disorder_overlap
bash run_templates/run_disprot_overlap_pipeline.sh
```
