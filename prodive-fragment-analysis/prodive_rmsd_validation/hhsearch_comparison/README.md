# HHsearch comparison preprocessing

This submodule parses raw HHsearch `.hhr` files, generates ProDive percentile subsets, evaluates overlap between ProDive and HHsearch family pairs, and constructs task tables for downstream structural validation.

## Inputs

| Input | Source |
|---|---|
| HHsearch raw `.hhr` files | `$PRODIVE_DATA_ROOT/rmsd/hhsearch_hhr/` |
| ProDive Pfam-Pfam result | `$PRODIVE_DATA_ROOT/shared/global_high_score_summary_fin.csv` |

## Workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/01_parse_hhr_to_csv.py` | HHsearch `.hhr` directory | `hhsuite20.csv` and `hhsuite70.csv` |
| 2 | `scripts/02_make_percentile_subsets.py` | ProDive result table | Percentile-filtered ProDive subset tables |
| 3 | `scripts/03_check_hhsearch_overlap.py` | HHsearch CSVs and ProDive subsets | ProDive/HHsearch overlap reports |
| 4 | `scripts/04_build_hhsearch_class_tasks.py` | HHsearch CSVs and overlap reports | ProDive-only, HHsearch-only, and shared comparison task tables |

Run template:

```bash
cd prodive_rmsd_validation/hhsearch_comparison
bash run_hhsearch_preprocessing.sh
```
