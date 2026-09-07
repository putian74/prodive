# De novo-Pfam RMSD validation

Calculate structural RMSD for ProDive de novo-Pfam fragment correspondences, generate length-stratified random controls, and plot the completed results.

## Setup

```bash
python3 -m pip install -r prodive_rmsd_validation/requirements.txt
conda install -c conda-forge pymol-open-source

export PRODIVE_DATA_ROOT=/path/to/ProDive_release/data
export PRODIVE_WORK_ROOT=/path/to/output
export PRODIVE_PFAM_RUNTIME_ROOT=/path/to/PfamA_seed_runtime
```

Run Python and PyMOL in the same environment. The Pfam runtime directory must contain each family's HHM, FAS/STO alignment, and structure files together.

## Inputs

| Input | Default path |
|---|---|
| ProDive de novo-Pfam correspondences | `$PRODIVE_DATA_ROOT/shared/denovo_global_high_score_summary_fin.csv` |
| De novo PDB structures | `$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/downloaded_denovo_pdbs/` |
| FASTA with de novo chain annotations | `$PRODIVE_DATA_ROOT/shared/structures/denovo_structures/all_1927_sequences.fasta` |
| Pfam profiles, alignments, and structures | `$PRODIVE_PFAM_RUNTIME_ROOT/PFxxxxx/` |

The correspondence CSV requires `Main_HMM` (de novo query ID), `Sub_HMM` (Pfam family), and `Sub_Segments_Details` (for example, `1-11 -> 75-85`). `Main_Segment_Len` and `Score` are optional. De novo PDB filenames must match the query ID or its prefix before the first underscore. Pfam seed headers must include sequence coordinates such as `P12345/1-100`.

## Run

From the repository root:

```bash
bash prodive_rmsd_validation/structural_validation/run_structural_validation.sh
```

This runs the observed de novo-Pfam calculation and writes `denovo_vs_pfam_rmsd_results_with_coverage.csv` under `$PRODIVE_WORK_ROOT/rmsd/structural_validation/rmsd_result_tables/`.

To include random controls, a comparison plot, summary statistics, and Welch tests:

```bash
RUN_RANDOM_CONTROLS=1 RUN_COMPARISON=1 WORKERS=20 \
  bash prodive_rmsd_validation/structural_validation/run_structural_validation.sh
```

Random controls default to 10,000 accepted pairs for each aligned length from 8 to 13 C-alpha atoms, with coverage at least 0.8. The sampler uses Pfam filenames containing `model` by default and resumes an existing random CSV. Set `PFAM_PDB_SUBSTRING=''` to include all Pfam PDB files. Use a new `RANDOM_CSV` path when changing sampling settings.

| Environment variable | Purpose |
|---|---|
| `DENOVO_INPUT_CSV`, `DENOVO_PDB_DIR`, `DENOVO_FASTA` | Override input paths. Set `DENOVO_FASTA=''` to use automatic chain selection. |
| `RESULT_ROOT` | Override the result directory. |
| `SEGMENT_MODE` | `first` (default) or `all` segment pairs per input row. |
| `RUN_RANDOM_CONTROLS`, `RUN_COMPARISON` | Enable optional steps with `1`; both default to `0`. |
| `RANDOM_CSV` | Supply an existing random-control table or choose a new output file. |
| `ALIGNED_LENGTHS`, `PER_LENGTH_QUOTA`, `COVERAGE_THRESHOLD` | Random/control-comparison settings; defaults `8,9,10,11,12,13`, `10000`, `0.8`. |
| `INPUT_WINDOW_MIN`, `INPUT_WINDOW_MAX` | Random input-window lengths; defaults `8`, `20`. |
| `WORKERS` | Random-control worker count; default `20`. |

The observed RMSD CSV is rewritten when the launcher runs. The random sampler resumes its existing CSV. See [structural_validation/README.md](structural_validation/README.md) for individual commands and [figure_reproduction/README.md](figure_reproduction/README.md) for plotting completed results.
