# Parameter sensitivity

This module evaluates fragment length, raw-KL thresholds, normalized-score thresholds, path formation, and the final adjusted-score cutoff.

## Requirements

```bash
python3 -m pip install -r validation/parameter_sensitivity/requirements.txt
```

Set the external data, completed-result, work, and GPU executable paths:

```bash
export PRODIVE_DATA_ROOT=/path/to/ProDive_release/data
export PRODIVE_PRECOMPUTED_ROOT=/path/to/ProDive_release/precomputed_results
export PRODIVE_WORK_ROOT=/path/to/parameter_work
export KL_CPP_EXE=/path/to/kl_divergence
```

The released inputs used by this module are:

| Input | Path |
|---|---|
| Pfam profiles and alignments | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` |
| Pfam numeric mapping | `$PRODIVE_DATA_ROOT/shared/pfam_mapping_seed_new.txt` |
| fragment-6 background JSON | `$PRODIVE_DATA_ROOT/shared/result_kl_all_pfam.json` |
| fixed fragment-length pair list | `$PRODIVE_DATA_ROOT/parameter_sensitivity/fragment_length/sampled_pairlist.csv` |
| completed sensitivity results | `$PRODIVE_PRECOMPUTED_ROOT/parameter_sensitivity/` |

Dense GPU matrices, sparse threshold-scan matrices, distributed family sets, path grids, and score-cutoff curves are local intermediates. New outputs are written under `PRODIVE_WORK_ROOT`.

## Fragment-length sensitivity

| Script | Function |
|---|---|
| `01_pack_fixed_length_hhm_fragments.py` | pack HHM windows for one fragment length |
| `02_generate_multi_fragment_packed_db.sh` | call script 01 for every value in `FRAGMENTS` |
| `03_generate_fixed_pairlist.py` | generate an optional deterministic replacement pair list |
| `04_compute_background_kl_for_fragment_lengths.py` | calculate background JSON files for multiple fragment lengths |
| `05_run_pairlist_all_fragment_lengths.sh` | run the GPU executable on the same pairs at every fragment length |
| `06_analyze_fragment_length_sensitivity.py` | calculate retained-cell and path summaries across fragment lengths |
| `07_plot_fragment_length_sensitivity.py` | plot the CSV summaries produced by script 06 |

Run the workflow with the released fixed pair list:

```bash
bash validation/parameter_sensitivity/run_templates/run_fragment_length_sensitivity_pipeline.sh
```

Override the tested fragment lengths when required:

```bash
FRAGMENTS="2 3 4 5 6 7 8 9 10" \
bash validation/parameter_sensitivity/run_templates/run_fragment_length_sensitivity_pipeline.sh
```

To generate a new deterministic pair list instead of using the released one:

```bash
REGENERATE_PAIRLIST=1 \
SAMPLE_N=50000 \
RANDOM_SEED=20260511 \
bash validation/parameter_sensitivity/run_templates/run_fragment_length_sensitivity_pipeline.sh
```

An alternative pair-list file can be supplied directly:

```bash
PAIR_LIST=/path/to/pairs.csv \
bash validation/parameter_sensitivity/run_templates/run_fragment_length_sensitivity_pipeline.sh
```

## Threshold scan

| Script | Function |
|---|---|
| `01_convert_raw_cpp_to_sparse_npz_monitor.py` | convert completed dense GPU NPY blocks to sparse NPZ files |
| `02_apply_first_second_threshold_grid.py` | apply raw and normalized-score threshold combinations |
| `03_extract_family_sets_by_threshold.py` | collect covered family IDs for each first/second-threshold combination |
| `04_scan_path_threshold_grid.py` | build paths, apply coverage rescoring, and summarize each threshold combination |
| `05_collect_third_score_thresholds.py` | merge distributed path results and count retained families across final-score cutoffs |
| `06_plot_global_threshold_grid.py` | merge distributed family sets and plot the first/second-threshold response surface |
| `07_logistic_second_derivative_extrema.py` | fit score-cutoff response curves and extract second-derivative extrema |

Convert local dense GPU output before running scripts 02-04:

```bash
python3 validation/parameter_sensitivity/scripts/threshold_scan/01_convert_raw_cpp_to_sparse_npz_monitor.py \
  --folder /path/to/raw_npy \
  --target-dir /path/to/sparse_npz \
  --mapping-file "$PRODIVE_DATA_ROOT/shared/pfam_mapping_seed_new.txt" \
  --max-i 25544
```

Run scripts 02-04 with the resulting sparse directory:

```bash
export RAW_SPARSE_DIR=/path/to/sparse_npz
bash validation/parameter_sensitivity/run_templates/run_threshold_scan_pipeline.sh
```

`COVERAGE_ROOT` defaults to `$PRODIVE_DATA_ROOT/shared/PfamA_seed`, which contains the nested per-family coverage CSV files. It can be overridden for another dataset.

Scripts 05-07 consume distributed intermediates. Supply their three input roots explicitly:

```bash
export THIRD_SCORE_INPUT_ROOT=/path/to/distributed_full_grid_results
export FAMILY_SET_INPUT_ROOT=/path/to/distributed_family_set_results
export SCORE_CURVE_INPUT_ROOT=/path/to/per_combo_score_cutoff_curves

bash validation/parameter_sensitivity/run_templates/run_parameter_threshold_extra.sh
```

The extra workflow derives its Pfam ID list from the released mapping. Set `PFAM_IDS_FILE` only when a different family set is required.

## Configuration files

| File | Contents |
|---|---|
| `configs/fragment_lengths.example.txt` | one fragment length per line |
| `configs/threshold_grid.example.env` | first-, second-, and final-threshold values |

## Tests

```bash
python3 -m unittest discover \
  -s validation/parameter_sensitivity/tests \
  -v
```

Use `python3 <script> --help` for all arguments.
