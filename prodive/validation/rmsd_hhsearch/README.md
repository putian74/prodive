# RMSD and HHsearch validation

This module parses HHsearch results, classifies ProDive/HHsearch family pairs, calculates local structural RMSD, generates random controls, and plots validation results.

## Requirements

```bash
python3 -m pip install -r validation/rmsd_hhsearch/requirements.txt
```

RMSD calculation requires PyMOL. A typical conda installation is:

```bash
conda install -c conda-forge pymol-open-source
```

Set the input, completed-result, structure, and output roots:

```bash
export PRODIVE_DATA_ROOT=/path/to/ProDive_release/data
export PRODIVE_PRECOMPUTED_ROOT=/path/to/ProDive_release/precomputed_results
export PRODIVE_STRUCTURE_ROOT=/path/to/PfamA_seed_structure
export PRODIVE_WORK_ROOT=/path/to/validation_work
export PRODIVE_PFAM_RUNTIME_ROOT=/path/to/PfamA_seed_runtime
```

Released resources used by this module are:

| Resource | Path |
|---|---|
| HHsearch `.hhr` files | `$PRODIVE_DATA_ROOT/rmsd/hhsearch_hhr/` |
| parsed HHsearch and task tables | `$PRODIVE_DATA_ROOT/rmsd/rmsd_task_tables/` |
| Pfam HHMs and alignments | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` |
| Pfam structure collection | `$PRODIVE_STRUCTURE_ROOT/` |
| completed RMSD tables and figures | `$PRODIVE_PRECOMPUTED_ROOT/rmsd/` |

Structural scripts that accept `--pfam-dir` require a merged runtime view in which each `PFxxxxx/` directory exposes both the released HHM/alignment files and the reconstructed structure files. Build `PRODIVE_PFAM_RUNTIME_ROOT` by following the `PfamA_seed_structure/README.md` instructions supplied with the structure manifest archive.

## HHsearch comparison

| Script | Function | How to run |
|---|---|---|
| `01_parse_hhr_to_csv.py` | parse `.hhr` files and retain hits above a probability threshold | `python3 <script> --input-dir DIR --output-csv FILE --prob-threshold 20` |
| `02_make_percentile_subsets.py` | create score-ranked ProDive subsets | `python3 <script> --input-csv FILE --output-dir DIR` |
| `03_check_hhsearch_overlap.py` | classify each ProDive pair as present or absent in HHsearch | `python3 <script> --base-dir DIR --hhsuite-csv FILE --hhsuite-threshold 20` |
| `04_build_hhsearch_class_tasks.py` | create shared, ProDive-only, and HHsearch-only task tables | `python3 <script> --prodive-overlap-csv FILE --hhsuite-csv FILE --output-dir DIR` |
| `run_hhsearch_preprocessing.sh` | run scripts 01–04 for probability thresholds 20 and 70 | set the environment variables above and run with `bash` |

Run the complete preprocessing workflow:

```bash
bash validation/rmsd_hhsearch/hhsearch_comparison/run_hhsearch_preprocessing.sh
```

Main outputs are parsed HHsearch tables, percentile subsets, overlap classifications, and RMSD task tables under `$PRODIVE_WORK_ROOT/rmsd/hhsearch_comparison`.

## Structural RMSD

| Script/module | Function | How to run |
|---|---|---|
| `01_rmsd_shared_prodive_segments.py` | calculate RMSD for shared pairs using ProDive segment boundaries | `python3 <script> --ref-csv FILE --pfam-dir DIR --output-csv FILE` |
| `02_rmsd_hhsearch_only.py` | calculate RMSD for HHsearch-only task rows | `python3 <script> --task-csv FILE --pfam-dir DIR --output-csv FILE` |
| `03_rmsd_shared_hhsearch_segments.py` | calculate RMSD for shared pairs using HHsearch boundaries | `python3 <script> --ref-csv FILE --pfam-dir DIR --output-csv FILE` |
| `04_rmsd_prodive_only_pipeline.py` | select representative structures, calculate ProDive-only RMSD, and standardize output | `python3 <script> --input-csv FILE --pfam-dir DIR --output-dir DIR` |
| `06_filter_rmsd_by_hhsearch_probability.py` | filter RMSD tables by matched HHsearch probability | `python3 <script> --hhsuite-csv FILE --overlap-csv FILE --hhsuite-only-rmsd FILE --shared-prodive-rmsd FILE --shared-hh-rmsd FILE --output-dir DIR` |
| `07_plot_rmsd_density_panels.py` | draw RMSD-versus-length density panels from configured datasets | `python3 <script> --dataset-config FILE --output-dir DIR` |
| `08_sample_random_pfam_pairs.py` | generate length-stratified random Pfam-Pfam RMSD controls | `python3 <script> --pfam-dir DIR --output-csv FILE --per-length-quota 25000` |
| `10_compare_real_random_core.py` | compare real and random RMSD distributions by aligned length | `python3 <script> --real-csv FILE --random-csv FILE --output-stats-csv FILE --output-plot FILE` |
| `11_welch_ttest_from_stats.py` | calculate Welch tests from the grouped statistics | `python3 <script> --stats-csv FILE --output-csv FILE` |
| `prodive_rmsd/structural_utils.py` | shared sequence, structure, alignment, and RMSD functions | imported by scripts 01–04 |

Example ProDive-only calculation:

```bash
python3 validation/rmsd_hhsearch/structural_validation/scripts/04_rmsd_prodive_only_pipeline.py \
  --input-csv /path/to/prodive_overlap.csv \
  --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT" \
  --output-dir "$PRODIVE_WORK_ROOT/rmsd/prodive_only" \
  --num-workers 20
```

Example random control and comparison:

```bash
python3 validation/rmsd_hhsearch/structural_validation/scripts/08_sample_random_pfam_pairs.py \
  --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT" \
  --output-csv "$PRODIVE_WORK_ROOT/rmsd/random_pfam.csv" \
  --aligned-lengths 8,9,10,11,12,13 \
  --per-length-quota 25000 \
  --coverage-threshold 0.8 \
  --workers 20

python3 validation/rmsd_hhsearch/structural_validation/scripts/10_compare_real_random_core.py \
  --real-csv /path/to/prodive_only_rmsd_standard.csv \
  --random-csv "$PRODIVE_WORK_ROOT/rmsd/random_pfam.csv" \
  --output-stats-csv "$PRODIVE_WORK_ROOT/rmsd/real_random_summary.csv" \
  --output-plot "$PRODIVE_WORK_ROOT/rmsd/real_random_boxplot.png"
```

## Figure generation

| Script | Function | How to run |
|---|---|---|
| `plot_fig2_and_s3_panels.py` | generate configured RMSD density panels | `python3 <script> --dataset-config FILE --output-dir DIR` |
| `plot_real_vs_random_core_boxplot.py` | generate real-versus-random summary statistics and a boxplot | `python3 <script> --real-csv FILE --random-csv FILE --output-stats-csv FILE --output-plot FILE` |
| `welch_ttest_from_stats.py` | calculate Welch tests from the real/random summary | `python3 <script> --stats-csv FILE --output-csv FILE` |
| `run_figure_reproduction.py` | execute all datasets listed in the two configuration CSVs | `python3 <script> --fig2-config FILE --random-config FILE --output-dir DIR` |

Edit the paths in these templates before running:

```text
figure_reproduction/configs/fig2_datasets.example.csv
figure_reproduction/configs/random_control_datasets.example.csv
```

Example:

```bash
cd validation/rmsd_hhsearch/figure_reproduction

python3 scripts/run_figure_reproduction.py \
  --fig2-config configs/fig2_datasets.example.csv \
  --random-config configs/random_control_datasets.example.csv \
  --output-dir outputs
```

Use `python3 <script> --help` for filtering, coverage, structure-quality, plotting, and parallelization options.
