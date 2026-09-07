# De novo-Pfam structural validation

Run commands from the repository root after setting the paths in the [module README](../README.md).

| Script | Function |
|---|---|
| `01_rmsd_denovo_pfam.py` | Map de novo and Pfam fragments to structures, superpose C-alpha atoms with PyMOL, and write RMSD and coverage. |
| `02_sample_random_denovo_pfam_pairs.py` | Sample random de novo-Pfam structural windows, grouped by aligned length. |
| `03_plot_rmsd_density_panels.py` | Plot RMSD against aligned length from a dataset configuration CSV. |
| `04_compare_real_random_core.py` | Compare real and random RMSD by aligned length; write summary statistics and a boxplot. |
| `05_welch_ttest_from_stats.py` | Calculate two-sided Welch tests from grouped summary statistics. |

## Observed fragments

```bash
python3 prodive_rmsd_validation/structural_validation/scripts/01_rmsd_denovo_pfam.py \
  --input-csv "$PRODIVE_DATA_ROOT/shared/denovo_global_high_score_summary_fin.csv" \
  --denovo-pdb-dir "$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/downloaded_denovo_pdbs" \
  --fasta-mapping-file "$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/all_1927_sequences.fasta" \
  --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT" \
  --output-csv "$PRODIVE_WORK_ROOT/rmsd/denovo_rmsd.csv"
```

`--segment-mode first` uses the first segment pair per input row; `--segment-mode all` processes all semicolon-separated pairs. Coverage is the aligned C-alpha count divided by `Main_Segment_Len`, or by the de novo query segment length when that column is unavailable. Output rows record successful calculations; failure counts are printed to the terminal.

## Random controls and comparison

```bash
python3 prodive_rmsd_validation/structural_validation/scripts/02_sample_random_denovo_pfam_pairs.py \
  --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT" \
  --denovo-pdb-dir "$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/downloaded_denovo_pdbs" \
  --output-csv "$PRODIVE_WORK_ROOT/rmsd/denovo_random.csv" \
  --aligned-lengths 8,9,10,11,12,13 --per-length-quota 10000 \
  --coverage-threshold 0.8 --workers 20

python3 prodive_rmsd_validation/structural_validation/scripts/04_compare_real_random_core.py \
  --real-csv "$PRODIVE_WORK_ROOT/rmsd/denovo_rmsd.csv" \
  --random-csv "$PRODIVE_WORK_ROOT/rmsd/denovo_random.csv" \
  --output-stats-csv "$PRODIVE_WORK_ROOT/rmsd/denovo_summary.csv" \
  --output-plot "$PRODIVE_WORK_ROOT/rmsd/denovo_real_random.png"

python3 prodive_rmsd_validation/structural_validation/scripts/05_welch_ttest_from_stats.py \
  --stats-csv "$PRODIVE_WORK_ROOT/rmsd/denovo_summary.csv" \
  --output-csv "$PRODIVE_WORK_ROOT/rmsd/denovo_welch.csv"
```

The random sampler selects a Pfam family, one of its structures, and a de novo structure, then samples equal-length windows from their first chains. Input lengths range from 8 to 20; accepted pairs must satisfy the requested aligned-length bins and coverage. This is a length-stratified structural background, not a same-protein control for each observed fragment. The existing seed option includes process/time offsets and does not guarantee identical draws across runs.

For density panels, use the [figure workflow](../figure_reproduction/README.md). Run any script with `--help` for additional options.
