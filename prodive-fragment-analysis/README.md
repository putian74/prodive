# ProDive fragment analysis

This repository contains downstream workflows for analyzing ProDive Pfam–Pfam and de novo–Pfam fragment correspondences. Analysis inputs and released results are available from Zenodo:

<https://doi.org/10.5281/zenodo.22643519>
## Repository layout

| Directory | Function |
|---|---|
| `prodive_clustering/` | Construct weighted fragment graphs, run CPM-Leiden clustering, and scan the resolution parameter. |
| `prodive_rmsd_validation/` | Calculate de novo–Pfam RMSD, generate length-stratified random controls, and plot validation results. |
| `prodive_esm2/` | Calculate residue-level ESM2 entropy and compare fragment positions with MSA-filtered background positions. |
| `prodive_secondary_structure_rsa/` | Annotate secondary structure and relative solvent accessibility for Pfam–Pfam and de novo–Pfam fragments. |
| `prodive_3did_interface/` | Map Pfam–Pfam fragments to 3did interface residues and generate matched seed-domain random controls and threshold-sensitivity statistics. |
| `prodive_disorder_overlap/` | Map DisProt regions, run matched randomization, and plot observed-versus-random overlap. |
| `prodive_phi_value/` | Map phi-value systems to Pfam seed coordinates and compare real fragments with matched random fragments. |
| `prodive_start2fold_hdx/` | Map Start2Fold folding and stability annotations to fragments and compare overlap with matched random windows. |
| `prodive_contact_order/` | Calculate fragment contact order and rFCO, compare matched random windows, and inspect high-rFCO cases. |

Each module provides analysis scripts, usage documentation or run templates, and a module-specific `requirements.txt`. All Python entry scripts support `--help`.

## Installation

Python 3.10 or later is recommended. Install only the requirements needed for the selected module:

```bash
python3 -m pip install -r prodive_clustering/requirements.txt
python3 -m pip install -r prodive_3did_interface/requirements.txt
python3 -m pip install -r prodive_disorder_overlap/requirements.txt
python3 -m pip install -r prodive_esm2/requirements.txt
python3 -m pip install -r prodive_phi_value/requirements.txt
python3 -m pip install -r prodive_rmsd_validation/requirements.txt
python3 -m pip install -r prodive_secondary_structure_rsa/requirements.txt
python3 -m pip install -r prodive_start2fold_hdx/requirements.txt
python3 -m pip install -r prodive_contact_order/requirements.txt
```

Additional runtime requirements are:

| Workflow | External requirement |
|---|---|
| RMSD calculation | PyMOL |
| Secondary structure and RSA | DSSP or `mkdssp` |
| ESM2 inference | Local Transformers checkpoint for `facebook/esm2_t36_3B_UR50D` |
| Phi-value and contact-order SIFTS mapping | PDBe API access or a populated local API cache |
| Start2Fold | Local XML records; network access is needed only for optional downloads |

## Data setup

Download and extract the data archive and structure manifest from Zenodo. Set:

```bash
export PRODIVE_RELEASE_ROOT=/path/to/ProDive_release
export PRODIVE_DATA_ROOT="$PRODIVE_RELEASE_ROOT/data"
export PRODIVE_PRECOMPUTED_ROOT="$PRODIVE_RELEASE_ROOT/precomputed_results"
export PRODIVE_WORK_ROOT=/path/to/local/output

export PRODIVE_STRUCTURE_ROOT=/path/to/PfamA_seed_structure
export PFAM_STRUCTURE_ROOT="$PRODIVE_STRUCTURE_ROOT"
export PRODIVE_PFAM_RUNTIME_ROOT=/path/to/PfamA_seed_runtime
```

The run templates expect the following local input layout; supply the Start2Fold XML records separately if they are absent from your data archive:

| Resource | Path relative to `$PRODIVE_DATA_ROOT` |
|---|---|
| Pfam-Pfam result CSV | `shared/global_high_score_summary_fin.csv` |
| De novo-Pfam result CSV | `shared/denovo_global_high_score_summary_fin.csv` |
| Score subsets | `shared/score_percentile_subsets/` |
| Pfam profiles and alignments | `shared/PfamA_seed/` |
| De novo structures and FASTA | `shared/structures/denovo_structures/` |
| 3did interface records | `3did/3did_flat.gz` |
| DisProt annotations | `disorder_overlap/DisProt_current_IDPO.tsv` |
| ESM2 inputs | `esm2/` |
| Phi-value inputs | `phi_value/Final_2Sm.csv`, `phi_value/phi_sites_plot_preserved_long.csv` |
| Start2Fold XML | `start2fold_hdx/start2fold_xml/` |

### Pfam structure paths

`$PRODIVE_DATA_ROOT/shared/PfamA_seed/` contains HHM and alignment files. The reconstructed `$PRODIVE_STRUCTURE_ROOT/` contains experimental and AlphaFold structures. Reconstruction commands and the expected family-level layout are documented in `PfamA_seed_structure/README.md` in the structure-manifest package.

- The 3did script accepts the seed and structure roots separately.
- RMSD, contact-order, and Pfam SS/RSA scripts expect HHM/alignment and structure files in the same family-level `--pfam-dir`. Use `$PRODIVE_PFAM_RUNTIME_ROOT` for these options after creating a merged runtime view.
- The RMSD structural-validation template requires `$PRODIVE_PFAM_RUNTIME_ROOT`. The Pfam SS/RSA template should be adapted to use the same runtime root when the archives are stored separately.

## Code-to-data mapping

| Module | Main inputs | Released results |
|---|---|---|
| Clustering | `data/shared/score_percentile_subsets/`; an SS/RSA table is required only for the optional structural summary | `precomputed_results/clustering/` |
| RMSD validation | `data/shared/denovo_global_high_score_summary_fin.csv`, de novo PDBs/FASTA, and the merged Pfam runtime directory | `precomputed_results/rmsd/` |
| ESM2 entropy | `data/shared/PfamA_seed/` and `data/esm2/` | `precomputed_results/esm2/` |
| Secondary structure/RSA | `data/shared/` and Pfam structures | `precomputed_results/secondary_structure_rsa/` |
| 3did interface | `data/3did/`, `data/shared/PfamA_seed/`, and experimental Pfam structures | `precomputed_results/interface_analysis/` |
| DisProt overlap | `data/disorder_overlap/` and `data/shared/` | `precomputed_results/disorder_overlap/` |
| Phi-value overlap | `data/phi_value/` and `data/shared/` | `precomputed_results/phi_value/` |

Additional module inputs and new output locations:

| Module | Main inputs | New outputs |
|---|---|---|
| Start2Fold/HDX | `data/start2fold_hdx/start2fold_xml/` and `data/shared/` | `$PRODIVE_WORK_ROOT/start2fold_hdx/` |
| Contact order | Pfam-Pfam result CSV, merged Pfam runtime, and optional SIFTS cache | `$PRODIVE_WORK_ROOT/contact_order/` |

## Leiden clustering

| Script | Function | Main outputs |
|---|---|---|
| `build_leiden_clusters.py` | Bin segment starts, merge nearby intervals with DBSCAN, construct an undirected weighted graph, and run CPM-Leiden. Repeated merged edges retain the maximum `Score`. | Community membership, graph summaries, node/edge tables, and family GraphML files. |
| `summarize_leiden_struct_stats.py` | Add intra-community SS/RSA summaries to completed clustering results without rebuilding the graph or rerunning Leiden. | Per-community SS/RSA summary, long-format counts, unique-node table, and report. |
| `leiden_gamma_sensitivity.py` | Rebuild the same graph and repeat clustering across supplied gamma values. | `gamma_all_runs.csv`, `gamma_selected_summary.csv`, and `gamma_selected_size_distribution.csv`. |

Default graph and clustering settings are:

```text
bin size                 8
DBSCAN eps               3
DBSCAN min_samples       2
DBSCAN metric            Chebyshev
Leiden objective         CPM
resolution gamma         0.05
beta                     0.01
Leiden iterations        4
runs                     3
base random seed         42
minimum community size   2
```

For each setting, three runs are evaluated and the partition with the highest weighted NetworkX modularity is retained. The standard clustering script writes family GraphML files by default; use `--no-family-graphml` to suppress them. The gamma-sensitivity script writes CSV tables only.

The default sensitivity scan tests gamma values `0.01`, `0.02`, `0.03`, `0.04`, `0.05`, `0.06`, `0.08`, `0.10`, `0.15`, and `0.20`.

Run all score subsets:

```bash
bash prodive_clustering/run_templates/run_leiden_clustering.sh
```

Add the optional structural summary after clustering:

```bash
SS_RSA_CSV="${PRODIVE_PRECOMPUTED_ROOT}/secondary_structure_rsa/pfam_pairs_ss_rsa_calculated_true_fin_hmmlen.csv" \
  bash prodive_clustering/run_templates/run_leiden_struct_stats.sh
```

Select a sensitivity subset with `PERCENTILE=5`, `10`, `20`, `50`, or `100`:

```bash
PERCENTILE=5 bash prodive_clustering/run_templates/run_gamma_sensitivity.sh

for percentile in 5 10 20 50 100; do
  PERCENTILE="$percentile" \
    bash prodive_clustering/run_templates/run_gamma_sensitivity.sh
done
```

Override the tested gamma values with `GAMMAS="0.02 0.05 0.08 0.10"`.

## De novo-Pfam RMSD validation

The RMSD module contains five scripts: observed de novo-Pfam RMSD, random de novo-Pfam controls, density plotting, real-versus-random comparison, and Welch tests. The calculation starts directly from the de novo ProDive result CSV and local structures.

```bash
bash prodive_rmsd_validation/structural_validation/run_structural_validation.sh
```

To include random controls and comparison statistics:

```bash
RUN_RANDOM_CONTROLS=1 RUN_COMPARISON=1 WORKERS=20 \
  bash prodive_rmsd_validation/structural_validation/run_structural_validation.sh
```

The default command runs only the observed calculation. Both commands require `PRODIVE_PFAM_RUNTIME_ROOT`; all inputs and optional settings are listed in [prodive_rmsd_validation/README.md](prodive_rmsd_validation/README.md). See the [figure README](prodive_rmsd_validation/figure_reproduction/README.md) to plot completed de novo results.

## ESM2 entropy

| Script | Function |
|---|---|
| `01_prepare_pfam_representatives.py` | Select representative seed sequences and map fragment ranges. |
| `02_split_fragdom_record_ranges.py` | Report record lengths and inference batch boundaries. |
| `03_run_masked_esm2_inference.py` | Calculate masked-token probabilities for a selected record range. |
| `04_calculate_entropy_from_probabilities.py` | Convert probability files to per-position entropy. |
| `05_compare_esm2_vs_msa_conservation.py` | Map representative positions to Pfam seed MSAs and compare conservation measures. |
| `06_fragment_vs_background_after_msa_filtering.py` | Compare fragment and background entropy after MSA-conservation filtering. |
| `fragdom_utils.py` | Parse the three-line fragment-domain record format. |

The released probability files were generated with the 36-layer, 3-billion-parameter [`facebook/esm2_t36_3B_UR50D`](https://huggingface.co/facebook/esm2_t36_3B_UR50D) checkpoint. `03_run_masked_esm2_inference.py` requires a local Transformers copy of this checkpoint and validates its layer count and hidden size.

Generate the two released inference batches, then run downstream steps from the completed probability directory:

```bash
export ESM2_MODEL_DIR=/path/to/esm2_t36_3B_UR50D

RECORD_RANGE=1:50 INFERENCE_ONLY=1 DEVICE=cuda \
  bash prodive_esm2/run_templates/run_pfam_esm2_pipeline.sh
RECORD_RANGE=51:100 INFERENCE_ONLY=1 DEVICE=cuda \
  bash prodive_esm2/run_templates/run_pfam_esm2_pipeline.sh

ESM2_PROBABILITY_DIR="${PRODIVE_WORK_ROOT}/esm2/pfam/02_inference" \
  bash prodive_esm2/run_templates/run_pfam_esm2_pipeline.sh
```

To skip model inference and use the released files directly:

```bash
ESM2_PROBABILITY_DIR="${PRODIVE_DATA_ROOT}/esm2/esm2_probability_outputs_optional" \
  bash prodive_esm2/run_templates/run_pfam_esm2_pipeline.sh
```

Use `DEVICE=cpu` only when sufficient host memory and runtime are available.

## Secondary structure and RSA

### Pfam–Pfam

| Script | Function |
|---|---|
| `pfam/01_compute_pfam_real_ss_rsa.py` | Map both fragment sides to Pfam structures and run DSSP. |
| `pfam/02_build_pfam_random_background_sqlite.py` | Sample same-chain, same-length windows and store the random background in SQLite. |
| `pfam/03_plot_pfam_real_vs_random.py` | Compare real and random SS/RSA distributions. |

Use `$PRODIVE_PFAM_RUNTIME_ROOT` for both `--pfam-dir` options before running the adapted template:

```bash
bash prodive_secondary_structure_rsa/run_templates/run_pfam_secondary_structure_rsa.sh
```

### De novo–Pfam

| Script | Function |
|---|---|
| `denovo/01_compute_denovo_real_ss_rsa.py` | Calculate SS/RSA for mapped de novo fragments. |
| `denovo/02_build_denovo_random_controls.py` | Sample same-chain, same-length de novo windows. |
| `denovo/03_plot_denovo_real_vs_random.py` | Compare real and random SS/RSA distributions. |

```bash
bash prodive_secondary_structure_rsa/run_templates/run_denovo_secondary_structure_rsa.sh
```

## 3did interface overlap

The revised workflow maps fragments through HHM, seed, and experimental PDB coordinates. Real and matched same-seed-domain random windows use the same union across experimental structures; random sampling excludes the observed window.

```bash
bash prodive_3did_interface/run_templates/run_3did_interface_analysis.sh
```

The template uses sequence/HMM length ratio 0.8–1.2, minimum structure coverage 0.8, search depth 200, and 200 random samples per assessable side. Interface categories retain cutoffs 0.30 and 0.50. Additional comparisons evaluate thresholds 0.10, 0.20, 0.30, 0.40, and 0.50 and continuous real-minus-random differences, with confidence intervals accounting for the two sides of each input row.

Outputs include the row-level CSV, side-level CSV, text summary, threshold-sensitivity CSV, and continuous-comparison CSV. See [prodive_3did_interface/README.md](prodive_3did_interface/README.md) for filenames, interaction flags, and parameter overrides.

## DisProt overlap

| Script | Function |
|---|---|
| `01_compare_disprot_regions_to_pfam_seed.py` | Match coordinate-defined DisProt regions to Pfam seed sequences. |
| `02_map_hmm_segments_to_disprot_regions.py` | Project ProDive HMM segments onto matched seed sequences. |
| `03_pfam_disorder_observed_expected.py` | Randomize same-length fragments within the same protein and Pfam-covered interval. |
| `04_plot_real_vs_random.py` | Recalculate statistics from result tables and draw observed-versus-random figures. |

```bash
bash prodive_disorder_overlap/run_templates/run_disprot_overlap_pipeline.sh
```

The default randomization uses 1,000 permutations and seed `20260331`. Override them with `N_PERMUTATIONS` and `RANDOM_SEED`.

## Phi-value overlap

| Script | Function |
|---|---|
| `01_match_pfdb_seed_sifts_residues.py` | Match PFDB systems to Pfam seed sequences through PDBe/SIFTS under author and residue numbering. |
| `02_extract_global_score_rows_for_phi_pfams.py` | Extract ProDive rows containing matched Pfam families. |
| `03_map_phi_related_fragments_to_seed_intervals.py` | Map HMM fragment coordinates to seed and UniProt intervals. |
| `04_compare_real_random_phi.py` | Sample same-case, same-UniProt, same-interval, same-length random fragments and compare phi-site coverage. |

```bash
bash prodive_phi_value/run_templates/run_phi_value_pipeline.sh
```

The randomization defaults to 10,000 iterations, seed `20260709`, and high-phi threshold `0.5`. Override these with `RANDOM_ITERATIONS`, `RANDOM_SEED`, and `HIGH_PHI_THRESHOLD`. Set `PHI_LONG_CSV` to override the default long-format phi table.

Step 1 caches PDBe/SIFTS responses under its output directory. The released cache is available under `precomputed_results/phi_value/pfam_seed_sifts_output_2sm_dual/sifts_api_cache/`.

## Start2Fold and HDX overlap

Parse Start2Fold XML records, match annotated proteins to Pfam seed sequences, map ProDive fragments, and compare annotated-residue overlap with matched random windows. The template interprets residue indices as UniProt coordinates, requests 1,000 random windows per segment, and uses a count threshold of two annotated residues for multitype summaries.

```bash
bash prodive_start2fold_hdx/run_templates/run_start2fold_hdx_pipeline.sh
```

The workflow writes parsed annotations, seed matches, real/random overlap tables, multitype summaries, overview plots, selected interval plots, and a combination bubble plot under `$PRODIVE_WORK_ROOT/start2fold_hdx/`. XML download is optional. See [prodive_start2fold_hdx/README.md](prodive_start2fold_hdx/README.md) for scripts and inputs.

## Contact order

Calculate fragment contact order and relative fragment contact order (rFCO), compare same-chain/same-length random controls, and inspect the high-rFCO tail.

```bash
bash prodive_contact_order/run_templates/run_full_contact_order_pipeline.sh
bash prodive_contact_order/run_templates/run_high_rfco_followup.sh
```

The main template requires `PRODIVE_PFAM_RUNTIME_ROOT`, uses an 8-angstrom contact cutoff and 1,000 random samples, and writes results under `$PRODIVE_WORK_ROOT/contact_order/`. It keeps realized-sequence/HMM length ratios from 0.8 to 1.2. Set `CONTACT_ORDER_CACHE` to reuse an existing cache; otherwise a cache is created under the output directory.

The follow-up generates high-rFCO selections, contact-topology figures, removal-sensitivity tables, and ProDive score annotations. See [prodive_contact_order/README.md](prodive_contact_order/README.md) for the six scripts and their outputs.

## Released results

All released tables, SQLite databases, and figures are under:

```text
$PRODIVE_PRECOMPUTED_ROOT/
```

Use these files for inspection or downstream plotting when a full rerun is unnecessary. New runs should write to `$PRODIVE_WORK_ROOT`.

