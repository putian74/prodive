
# ProDive analysis modules

This repository contains analysis modules used to reproduce the validation, characterization, and sensitivity analyses for ProDive, a profile-HMM-based method for detecting cross-family protein fragment similarity.

## Data availability

All external data archives, precomputed result tables, analysis-ready resources, and the Pfam structure manifest required by these modules are available from Zenodo:

https://doi.org/10.5281/zenodo.20838915

The Zenodo record provides the main `ProDive_release` data archive and a `PfamA_seed_structure` manifest archive. The full Pfam-organized structure directory can be reconstructed from the manifest and download scripts distributed with that record.

## Repository organization

| Module | Purpose |
|---|---|
| `prodive_rmsd_validation` | HHsearch comparison, structural RMSD validation, random controls, and figure reproduction. |
| `prodive_secondary_structure_rsa` | Secondary-structure and relative solvent-accessibility analyses for Pfam-Pfam and de novo-Pfam fragments. |
| `prodive_interface_analysis` | Interface-context analysis for de novo-side ProDive fragments and matched random controls. |
| `prodive_clustering` | Leiden clustering of ProDive fragment correspondences and community-level structural summaries. |
| `prodive_esm2` | ESM2 masked-token entropy analysis, MSA conservation comparison, and structural stratification. |
| `prodive_contact_order` | Fragment contact-order analysis, matched random controls, and high-rFCO follow-up analyses. |
| `prodive_phi_value` | Mapping of folding phi-value residues to Pfam seed coordinates and ProDive fragment intervals. |
| `prodive_disorder_overlap` | DisProt-to-Pfam mapping and observed/expected disorder-overlap analysis. |
| `prodive_start2fold_hdx` | Start2Fold and HDX overlap analysis with matched random-window controls. |
| `prodive_parameter_sensitivity` | Fragment-length, threshold, derivative, and large-dataset structural-context analyses. |

Each module contains a `README.md` file with module-specific inputs, outputs, and run templates.

## External data layout

After downloading and extracting the Zenodo archives, a typical working directory has the following layout:

```text
parent_directory/
├── ProDive_release/
│   ├── data/
│   └── precomputed_results/
└── PfamA_seed_structure/
    ├── pfam_structure_manifest.tsv
    ├── 02_download_pfam_structure_archive.py
    └── PFxxxxx/                 # generated after structure reconstruction
```

Set the following environment variables before running the analysis modules:

```bash
export PRODIVE_RELEASE_ROOT=/path/to/parent_directory/ProDive_release
export PRODIVE_DATA_ROOT=$PRODIVE_RELEASE_ROOT/data
export PRODIVE_PRECOMPUTED_ROOT=$PRODIVE_RELEASE_ROOT/precomputed_results
export PRODIVE_STRUCTURE_ROOT=/path/to/parent_directory/PfamA_seed_structure
export PRODIVE_WORK_ROOT=/path/to/local/output
export PYTHON=python3
```

`PRODIVE_DATA_ROOT` points to the main external data archive. `PRODIVE_PRECOMPUTED_ROOT` points to released result tables and figures. `PRODIVE_STRUCTURE_ROOT` points to the Pfam-organized structure directory reconstructed from the Zenodo manifest. `PRODIVE_WORK_ROOT` should be a writable output location for newly generated results.

## Pfam runtime directory for structure-dependent modules

Several structure-dependent workflows require Pfam profile/alignment files and Pfam structure files to be accessible under a common family-level directory tree. A merged runtime view can be created from the released data archives as follows:

```bash
export PRODIVE_PFAM_RUNTIME_ROOT=$PRODIVE_WORK_ROOT/PfamA_seed_runtime
mkdir -p "$PRODIVE_PFAM_RUNTIME_ROOT"

rsync -a   --include='PF*/'   --include='PF*/*.hhm'   --include='PF*/*.sto'   --include='PF*/*.fas'   --include='PF*/*.fasta'   --exclude='*'   "$PRODIVE_DATA_ROOT/shared/PfamA_seed/"   "$PRODIVE_PFAM_RUNTIME_ROOT/"

rsync -a   "$PRODIVE_STRUCTURE_ROOT/"   "$PRODIVE_PFAM_RUNTIME_ROOT/"
```

For structure-dependent command-line options such as `--pfam-dir`, `--pfam-root`, or `--pfam-seed-dir`, use `$PRODIVE_PFAM_RUNTIME_ROOT` unless the script provides separate options for sequence/profile inputs and structure inputs. For seed-only mapping steps, `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` is sufficient.

## Running the analyses

Most modules provide shell templates under `run_templates/`. These templates are intended as reproducible command examples and should be edited to match the local data and output locations.

A typical workflow is:

```bash
cd prodive_secondary_structure_rsa
bash run_templates/run_pfam_secondary_structure_rsa.sh
```

For modules containing multiple subworkflows, follow the step order described in the corresponding module README.

## Software requirements

Each module includes a `requirements.txt` file when Python package requirements are module-specific. Structure-dependent workflows may additionally require external programs such as DSSP/mkdssp or PyMOL, depending on the analysis.

Recommended baseline Python packages include:

```text
numpy
pandas
scipy
matplotlib
biopython
tqdm
```

## Reproducibility notes

The analysis modules are organized around released input tables, precomputed resources, and run templates. For exact figure reproduction, use the released Zenodo data together with the plotting and comparison scripts in the corresponding module directories.
