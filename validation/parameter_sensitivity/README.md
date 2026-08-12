# Parameter sensitivity

This module evaluates fragment length, raw-KL thresholds, normalized-score thresholds, path formation, and the final adjusted-score cutoff.

## Requirements

```bash
python3 -m pip install -r validation/parameter_sensitivity/requirements.txt
```

Set the data, output, and GPU executable paths:

```bash
export PRODIVE_DATA_ROOT=/path/to/ProDive_release/data
export PRODIVE_WORK_ROOT=/path/to/parameter_work
export KL_CPP_EXE=/path/to/kl_divergence
```

## Fragment-length scripts

| Script | Function | How to run |
|---|---|---|
| `01_pack_fixed_length_hhm_fragments.py` | pack HHM windows for one fragment length | `python3 <script> --input_dir DIR --output_dir DIR --mapping FILE --fragment K` |
| `02_generate_multi_fragment_packed_db.sh` | call script 01 for every value in `FRAGMENTS` | set `INPUT_DIR`, `MAPPING_FILE`, `OUT_ROOT`, and `FRAGMENTS`, then run with `bash` |
| `03_generate_fixed_pairlist.py` | generate one deterministic pair list reused at every fragment length | `python3 <script> --mapping-file FILE --out-dir DIR --sample-n 50000 --seed 20260511` |
| `04_compute_background_kl_for_fragment_lengths.py` | calculate background JSON files for multiple fragment lengths | `python3 <script> --pfam-list-file FILE --pfam-seed-dir DIR --out-dir DIR --fragment-lengths 2,3,4,5,6` |
| `05_run_pairlist_all_fragment_lengths.sh` | run the GPU executable on the same pairs at every fragment length | set `CPP_EXE`, `PAIR_LIST`, `PACKED_DB_ROOT`, `OUT_ROOT`, and `FRAGMENTS`, then run with `bash` |
| `06_analyze_fragment_length_sensitivity.py` | calculate retained-cell and path summaries across fragment lengths | `python3 <script> --input-root DIR --output-dir DIR --mapping-file FILE --background-json-pattern TEMPLATE` |
| `07_plot_fragment_length_sensitivity.py` | plot the CSV summaries produced by script 06 | `python3 <script> --result-dir DIR` |

Run the complete fragment-length workflow:

```bash
bash validation/parameter_sensitivity/run_templates/run_fragment_length_sensitivity_pipeline.sh
```

Override the default fragment series when needed:

```bash
FRAGMENTS="2 3 4 5 6 7 8 9 10" \
bash validation/parameter_sensitivity/run_templates/run_fragment_length_sensitivity_pipeline.sh
```

## Threshold-scan scripts

| Script | Function | How to run |
|---|---|---|
| `01_convert_raw_cpp_to_sparse_npz_monitor.py` | monitor dense GPU NPY files and convert completed blocks to sparse NPZ | `python3 <script> --folder DIR --target-dir DIR --mapping-file FILE --max-i N` |
| `02_apply_first_second_threshold_grid.py` | apply raw and normalized-score threshold combinations | `python3 <script> --input-dir DIR --output-root DIR --mapping-file FILE --background-json FILE` |
| `03_extract_family_sets_by_threshold.py` | collect covered family IDs for each first/second-threshold combination | `python3 <script> --preprocessed-root DIR --mapping-file FILE --output-dir DIR` |
| `04_scan_path_threshold_grid.py` | build paths, apply coverage rescoring, and summarize each threshold combination | `python3 <script> --preprocessed-root DIR --output-root DIR --mapping-file FILE --coverage-root DIR` |
| `05_collect_third_score_thresholds.py` | merge distributed path results and count retained families across final-score cutoffs | `python3 <script> --root-dir DIR --pfam-ids-file FILE --output-root DIR` |
| `06_plot_global_threshold_grid.py` | merge family sets and plot the first/second-threshold response surface | `python3 <script> --root-dir DIR --output-dir DIR` |
| `07_logistic_second_derivative_extrema.py` | fit score-cutoff response curves and extract second-derivative extrema | `python3 <script> --input-curve-root DIR --output-root DIR` |
| `tests/test_threshold_scan.py` | verify that path statistics retain distinct length, count, IoU, and similarity fields | run with the unit-test command below |

If dense GPU matrices have not yet been converted, run script 01 first:

```bash
python3 validation/parameter_sensitivity/scripts/threshold_scan/01_convert_raw_cpp_to_sparse_npz_monitor.py \
  --folder /path/to/raw_npy \
  --target-dir /path/to/sparse_npz \
  --mapping-file /path/to/hhm_mapping.txt \
  --max-i 25544
```

Run scripts 02–04:

```bash
bash validation/parameter_sensitivity/run_templates/run_threshold_scan_pipeline.sh
```

Run scripts 05–07 on completed distributed results:

```bash
bash validation/parameter_sensitivity/run_templates/run_parameter_threshold_extra.sh
```

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

Use `python3 <script> --help` for all optional arguments.
