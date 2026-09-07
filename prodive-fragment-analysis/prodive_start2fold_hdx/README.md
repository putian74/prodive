# Start2Fold and HDX overlap analysis

This module parses Start2Fold XML records, maps Start2Fold and HDX annotations to Pfam seed intervals, maps ProDive fragments to those intervals, generates matched random windows, and performs multitype real-vs-random analyses.

Start2Fold residue indices are treated as UniProt full-length residue coordinates.

## Inputs

| Input | Source |
|---|---|
| Start2Fold XML files | `$PRODIVE_DATA_ROOT/start2fold_hdx/start2fold_xml/` |
| Pfam seed HHM/STO/FAS database | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` |
| Pfam-Pfam ProDive result | `$PRODIVE_DATA_ROOT/shared/global_high_score_summary_fin.csv` |

## Workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/02_parse_start2fold_xml.py` | Start2Fold XML directory | Parsed Start2Fold annotation table |
| 2 | `scripts/03_match_start2fold_to_pfam_seed.py` | Parsed annotation table and Pfam seed database | Start2Fold-to-Pfam seed match tables |
| 3 | `scripts/04_map_prodive_segments_to_start2fold_and_random.py` | Step 2 outputs, ProDive result, and Pfam seed database | Real ProDive overlap table and matched random windows |
| 4 | `scripts/05_multitype_real_vs_random_analysis.py` | Real and random overlap tables | Multitype real-vs-random summaries |
| 5 | `scripts/06_plot_start2fold_hdx_results.py` | Step 3 and Step 4 outputs | Main Start2Fold/HDX figures |
| 6 | `scripts/07_plot_selected_stf_fragment_intervals.py` | Mapped-segment tables | Selected fragment-interval figures |
| 7 | `scripts/08_plot_exact_combo_bubbles.py` | Mapped-segment table | Exact-combination bubble plot |

Optional XML download script: `scripts/01_download_start2fold_xml.py`.

Set `PRODIVE_DATA_ROOT` and `PRODIVE_WORK_ROOT`, then run from the repository root:

```bash
bash prodive_start2fold_hdx/run_templates/run_start2fold_hdx_pipeline.sh
```

The template uses local XML files and runs steps 02–08. It requests 1,000 random windows per segment (`RANDOM_SAMPLES_PER_SEGMENT`), with seed `20260409` (`RANDOM_SEED`), and uses an annotated-residue count threshold of two for multitype analysis. Step 07 plots `STF0006` and `STF0025` by default; edit `--target-stf-ids` to select other records. Outputs are written under `$PRODIVE_WORK_ROOT/start2fold_hdx/`.
