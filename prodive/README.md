# ProDive

ProDive detects local similarities between profile hidden Markov models. It packs fixed-length HHM fragments, calculates symmetric KL-divergence matrices on GPUs, filters significant window pairs, builds diagonal paths, and applies coverage-based rescoring.

External data and precomputed results are available from <https://doi.org/10.5281/zenodo.20838915>.

## Directory layout

| Directory | Contents |
|---|---|
| `data_preparation/` | HHM mapping, fragment packing, background values, and coverage tables |
| `src/cpp_cuda/` | C++/CUDA KL-divergence implementation |
| `src/cpu_reference/` | serial CPU implementation for numerical validation |
| `src/path_extraction/` | KL filtering, path construction, and final rescoring |
| `benchmark/` | performance and CPU/GPU validation framework |
| `validation/parameter_sensitivity/` | fragment-length and threshold scans |
| `validation/rmsd_hhsearch/` | HHsearch comparison and structural RMSD validation |
| `configs/` | reference dataset, production, and benchmark parameters |

## Requirements

Python 3.10 or later is recommended.

```bash
python3 -m pip install -r data_preparation/requirements.txt
python3 -m pip install -r src/path_extraction/requirements.txt
```

The GPU program requires CUDA, a C++14 compiler, Eigen, and zlib. The default Makefile targets NVIDIA Ampere GPUs (`sm_86`).

```bash
make -C src/cpp_cuda -j
```

## Data setup

Extract the external data archive, then define the data, result, and work roots:

```bash
export PRODIVE_RELEASE_ROOT=/path/to/ProDive_release
export PRODIVE_DATA_ROOT="$PRODIVE_RELEASE_ROOT/data"
export PRODIVE_PRECOMPUTED_ROOT="$PRODIVE_RELEASE_ROOT/precomputed_results"
export PRODIVE_WORK_ROOT=/path/to/prodive_work
```

Set `PRODIVE_STRUCTURE_ROOT` only for structure-dependent validation:

```bash
export PRODIVE_STRUCTURE_ROOT=/path/to/PfamA_seed_structure
```

Released resources used by this repository are:

| Resource | Path |
|---|---|
| Pfam HHMs, alignments, and coverage tables | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` |
| Pfam numeric mapping | `$PRODIVE_DATA_ROOT/shared/pfam_mapping_seed_new.txt` |
| fragment-6 background values | `$PRODIVE_DATA_ROOT/shared/result_kl_all_pfam.json` |
| final Pfam-Pfam table | `$PRODIVE_DATA_ROOT/shared/global_high_score_summary_fin.csv` |
| final de novo-Pfam table | `$PRODIVE_DATA_ROOT/shared/denovo_global_high_score_summary_fin.csv` |
| fixed fragment-length pair list | `$PRODIVE_DATA_ROOT/parameter_sensitivity/fragment_length/sampled_pairlist.csv` |
| HHsearch and RMSD inputs | `$PRODIVE_DATA_ROOT/rmsd/` |
| completed analysis outputs | `$PRODIVE_PRECOMPUTED_ROOT/` |

Packed databases, dense GPU KL matrices, filtered pickle files, and path-building intermediates are generated locally under `PRODIVE_WORK_ROOT`.

## Script reference

### Data preparation

| Script | Function | Main input | Main output |
|---|---|---|---|
| `01_generate_pfam_hhm_mapping.py` | assign stable integer IDs to HHM files | HHM directory | `ID: integer` mapping file |
| `02_pack_hhm_fragments.py` | convert sliding HHM windows to packed float32 arrays | HHMs and mapping | binary arrays, `index.csv`, `metadata.json` |
| `03_generate_background_kl.py` | calculate background values for each HHM window | HHMs and mapping | `background_kl_fragment_<k>.json` |
| `04_generate_hhm_coverage.py` | calculate match-state coverage from alignments | HHM and alignment files | `*_coverage.csv` |

### KL calculation

| Program/module | Function | Usage |
|---|---|---|
| `src/cpp_cuda/kl_divergence` | calculate dense symmetric-KL matrices on one or more GPUs | execute after compiling with `make` |
| `cpu_kl_reference.py` | calculate one family pair on a single CPU core and optionally compare it with GPU output | run directly for validation |
| `pfam_selfhmm_read_hhm.py` | parse HHM transition and emission values | imported by the CPU reference |
| `splits_of_hhm.py` | construct fragment transition and emission matrices | imported by the CPU reference |
| `tools.py` | locate HHM metadata records | imported by the CPU reference |

### Path extraction

| Script | Function | Main input | Main output |
|---|---|---|---|
| `01_filter_cpp_kl_results.py` | apply raw-KL and background-normalized filters | GPU NPY matrices | filtered pickle files |
| `02_build_paths_from_filtered_pk.py` | join retained cells into diagonal paths | filtered pickle files | path summary CSV files |
| `03_rescore_paths_by_coverage.py` | apply coverage and adjusted-score filters | path summary and coverage tables | final path CSV |

Use `python3 <script> --help` to list all arguments.

## Main workflow

Use the released mapping, background, and coverage resources while writing all new intermediates to a separate work directory:

```bash
export HHM_ROOT="$PRODIVE_DATA_ROOT/shared/PfamA_seed"
export MAPPING_FILE="$PRODIVE_DATA_ROOT/shared/pfam_mapping_seed_new.txt"
export BACKGROUND_JSON="$PRODIVE_DATA_ROOT/shared/result_kl_all_pfam.json"
export COVERAGE_ROOT="$PRODIVE_DATA_ROOT/shared/PfamA_seed"
export WORK_ROOT="$PRODIVE_WORK_ROOT/main"
export FRAGMENT=6

mkdir -p "$WORK_ROOT"/{mapping,packed,background,coverage,kl,filtered,paths,final}
```

### 1. Generate a mapping when required

The released mapping can be used directly. To generate a new mapping for another HHM collection:

```bash
python3 data_preparation/scripts/01_generate_pfam_hhm_mapping.py \
  --root_dir "$HHM_ROOT" \
  --output "$WORK_ROOT/mapping/hhm_mapping.txt"

export MAPPING_FILE="$WORK_ROOT/mapping/hhm_mapping.txt"
```

### 2. Pack HHM fragments

```bash
python3 data_preparation/scripts/02_pack_hhm_fragments.py \
  --input_dir "$HHM_ROOT" \
  --output_dir "$WORK_ROOT/packed/fragment_${FRAGMENT}_packed" \
  --mapping "$MAPPING_FILE" \
  --fragment "$FRAGMENT" \
  --jobs 20 \
  --overwrite
```

### 3. Generate background values when required

The released fragment-6 background JSON can be used directly. For another fragment length or HHM collection:

```bash
python3 data_preparation/scripts/03_generate_background_kl.py \
  --input_dir "$HHM_ROOT" \
  --mapping "$MAPPING_FILE" \
  --output_dir "$WORK_ROOT/background" \
  --fragments "$FRAGMENT"

export BACKGROUND_JSON="$WORK_ROOT/background/background_kl_fragment_${FRAGMENT}.json"
```

### 4. Generate coverage tables when required

The released Pfam directory already contains nested `PFxxxxx_coverage.csv` files. For another HHM collection:

```bash
python3 data_preparation/scripts/04_generate_hhm_coverage.py batch \
  --input-dir "$HHM_ROOT" \
  --output-dir "$WORK_ROOT/coverage" \
  --jobs 20

export COVERAGE_ROOT="$WORK_ROOT/coverage"
```

### 5. Calculate GPU KL matrices

Pair-list mode:

```bash
src/cpp_cuda/kl_divergence \
  --pair_list /path/to/pairs.csv \
  --packed_db "$WORK_ROOT/packed/fragment_${FRAGMENT}_packed" \
  --outdir "$WORK_ROOT/kl" \
  --memory_efficient \
  --num_streams 8 \
  --max_gpus 2
```

`pairs.csv` contains numeric IDs from the mapping file:

```text
i,j
0,1
2,5
```

Range mode is also supported:

```bash
src/cpp_cuda/kl_divergence \
  --start_i 0 --end_i 100 \
  --start_j 0 --end_j 100 \
  --packed_db "$WORK_ROOT/packed/fragment_${FRAGMENT}_packed" \
  --outdir "$WORK_ROOT/kl" \
  --memory_efficient --num_streams 8 --max_gpus 2
```

### 6. Filter KL matrices

```bash
python3 src/path_extraction/scripts/01_filter_cpp_kl_results.py \
  --input-dir "$WORK_ROOT/kl" \
  --output-dir "$WORK_ROOT/filtered" \
  --mapping-file "$MAPPING_FILE" \
  --background-json "$BACKGROUND_JSON" \
  --fragment "$FRAGMENT" \
  --raw-threshold 2.0 \
  --filter-threshold 13.0
```

### 7. Build paths

```bash
python3 src/path_extraction/scripts/02_build_paths_from_filtered_pk.py \
  --input-dir "$WORK_ROOT/filtered" \
  --output-dir "$WORK_ROOT/paths" \
  --mapping-file "$MAPPING_FILE" \
  --fragment "$FRAGMENT" \
  --min-path-points 5 \
  --workers 20
```

### 8. Rescore paths

```bash
python3 src/path_extraction/scripts/03_rescore_paths_by_coverage.py \
  --input-csv "$WORK_ROOT/paths/global_high_score_summary.csv" \
  --coverage-dir "$COVERAGE_ROOT" \
  --output-csv "$WORK_ROOT/final/global_high_score_summary_fin.csv" \
  --coverage-threshold 0.85 \
  --score-threshold 1.2
```

## Default filters

| Step | Rule |
|---|---|
| raw KL | `round(raw, 4) < 2.0` |
| normalized score | `(q_bg + t_bg) / (raw4 + 0.0001) * (3*k + 2) >= 13.0` |
| path size | at least 5 linked window pairs |
| coverage | both sides `>= 0.85` |
| final score | `log10(Avg_Similarity * ((Coverage_Main + Coverage_Sub)/2)^2) > 1.2` |

## Additional workflows

- Performance tests: [`benchmark/README.md`](benchmark/README.md)
- Parameter scans: [`validation/parameter_sensitivity/README.md`](validation/parameter_sensitivity/README.md)
- RMSD and HHsearch validation: [`validation/rmsd_hhsearch/README.md`](validation/rmsd_hhsearch/README.md)
