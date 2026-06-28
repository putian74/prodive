# ProDive main pipeline

ProDive is a computational pipeline for detecting fragment-level similarity between protein-family profile hidden Markov models (profile HMMs). The workflow converts HHM profiles into fixed-length fragment representations, computes pairwise fragment-level KL-divergence matrices with a C++/CUDA backend, filters background-normalized fragment matches, assembles diagonal paths, and applies coverage-based rescoring to produce final ProDive path calls.

## Data availability

The external data archive associated with this repository is available at:

https://doi.org/10.5281/zenodo.20838915

The Zenodo record provides the `ProDive_release` data archive and the `PfamA_seed_structure` manifest archive. These resources contain the input data, precomputed resources, documentation, and structure-manifest files required to reproduce the ProDive analyses. The source code in this repository is designed to operate on those external resources or on user-supplied HHM/profile-HMM datasets organized in the same format.

## Repository layout

```text
ProDive/
├── README.md
├── data_preparation/
│   ├── README.md
│   ├── requirements.txt
│   └── scripts/
│       ├── 01_generate_pfam_hhm_mapping.py
│       ├── 02_pack_hhm_fragments.py
│       ├── 03_generate_background_kl.py
│       └── 04_generate_hhm_coverage.py
└── src/
    ├── cpp_cuda/
    │   ├── README.md
    │   ├── Makefile
    │   └── *.cpp / *.cu / *.h
    └── path_extraction/
        ├── README.md
        ├── requirements.txt
        └── scripts/
            ├── 01_filter_cpp_kl_results.py
            ├── 02_build_paths_from_filtered_pk.py
            └── 03_rescore_paths_by_coverage.py
```

The main workflow is organized into three functional components:

| Component | Location | Purpose |
|---|---|---|
| Data preparation | `data_preparation/` | Build HHM mappings, pack fixed-length HHM fragments, compute background KL values, and generate coverage tables. |
| KL computation | `src/cpp_cuda/` | Compute pairwise fragment-level KL-divergence matrices using the packed HHM fragment database. |
| Path extraction | `src/path_extraction/` | Filter KL matrices, build diagonal paths, and apply coverage-based rescoring. |

## Expected data organization

For Pfam-scale reproduction, the HHM and alignment files are expected to be provided by the external data archive under `ProDive_release/data`. A typical configuration is:

```bash
export PRODIVE_DATA_ROOT=/path/to/ProDive_release/data
export PRODIVE_WORK_ROOT=/path/to/prodive_work
export HHM_ROOT=$PRODIVE_DATA_ROOT/shared/PfamA_seed
export FRAGMENT=6
export JOBS=40
```

The HHM root directory should contain one subdirectory per protein family:

```text
$HHM_ROOT/
├── PF00001/
│   ├── PF00001.hhm
│   ├── PF00001.sto
│   └── PF00001.fas
├── PF00002/
│   ├── PF00002.hhm
│   ├── PF00002.sto
│   └── PF00002.fas
└── ...
```

Pipeline outputs are written to the user-defined working directory specified by `PRODIVE_WORK_ROOT`.

## Software requirements

Python 3.8 or later is recommended for the Python stages.

```bash
python3 -m pip install -r data_preparation/requirements.txt
python3 -m pip install -r src/path_extraction/requirements.txt
```

The C++/CUDA stage requires:

```text
CUDA toolkit
C++ compiler with C++14 support
Eigen headers
zlib
```

The bundled Makefile uses `-arch=sm_86`, which targets NVIDIA Ampere GPUs. For other GPU architectures, adjust `NVCC_FLAGS` in `src/cpp_cuda/Makefile` before compilation.

## Pipeline overview

| Stage | Script or program | Main input | Main output |
|---:|---|---|---|
| 1 | `01_generate_pfam_hhm_mapping.py` | HHM root directory | `hhm_mapping.txt` |
| 2 | `02_pack_hhm_fragments.py` | HHM root, mapping file | packed fragment database |
| 3 | `03_generate_background_kl.py` | HHM root, mapping file | `background_kl_fragment_6.json` |
| 4 | `04_generate_hhm_coverage.py` | HHM and alignment files | `*_coverage.csv` tables |
| 5 | `src/cpp_cuda/kl_divergence` | packed fragment database | `kl_<i>_<j>.npy` matrices |
| 6 | `01_filter_cpp_kl_results.py` | C++ KL matrices, mapping, background JSON | `kl_<query>_filtered.pk` |
| 7 | `02_build_paths_from_filtered_pk.py` | filtered pickle files | `global_high_score_summary.csv` |
| 8 | `03_rescore_paths_by_coverage.py` | path table, coverage tables | final rescored path table |

Stages 1--3 and Stage 5 are required for KL computation. Stage 4 is required for the final coverage-based rescoring step.

## Stage 1: generate HHM mapping

```bash
mkdir -p $PRODIVE_WORK_ROOT/01_mapping

python3 data_preparation/scripts/01_generate_pfam_hhm_mapping.py   --root_dir $HHM_ROOT   --output $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt
```

This produces a mapping file assigning one numeric record ID to each HHM record. The C++/CUDA stage uses these numeric IDs in output filenames.

For datasets with repeated filename stems, relative-path IDs can be used:

```bash
python3 data_preparation/scripts/01_generate_pfam_hhm_mapping.py   --root_dir $HHM_ROOT   --output $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt   --id_source relative_path
```

## Stage 2: pack fixed-length HHM fragments

```bash
mkdir -p $PRODIVE_WORK_ROOT/02_packed_fragments

python3 data_preparation/scripts/02_pack_hhm_fragments.py   --input_dir $HHM_ROOT   --output_dir $PRODIVE_WORK_ROOT/02_packed_fragments/fragment_${FRAGMENT}_packed   --mapping $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt   --fragment $FRAGMENT   --jobs $JOBS   --overwrite
```

Expected output:

```text
$PRODIVE_WORK_ROOT/02_packed_fragments/fragment_6_packed/
├── pi_all.float32.bin
├── A_all.float32.bin
├── B_all.float32.bin
├── index.csv
├── metadata.json
└── summary.json
```

The C++/CUDA implementation reads `float32` packed arrays.

## Stage 3: generate background KL values

```bash
mkdir -p $PRODIVE_WORK_ROOT/03_background_kl

python3 data_preparation/scripts/03_generate_background_kl.py   --input_dir $HHM_ROOT   --mapping $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt   --output_dir $PRODIVE_WORK_ROOT/03_background_kl   --fragments $FRAGMENT
```

Expected output:

```text
$PRODIVE_WORK_ROOT/03_background_kl/background_kl_fragment_6.json
$PRODIVE_WORK_ROOT/03_background_kl/background_kl_generation_summary.json
```

Each JSON entry stores the background/window-information vector for one HHM record. The vector length corresponds to the number of sliding windows in the packed fragment database.

## Stage 4: generate match-state coverage tables

```bash
mkdir -p $PRODIVE_WORK_ROOT/04_coverage_tables

python3 data_preparation/scripts/04_generate_hhm_coverage.py batch   --input-dir $HHM_ROOT   --output-dir $PRODIVE_WORK_ROOT/04_coverage_tables   --jobs 20
```

Expected output:

```text
$PRODIVE_WORK_ROOT/04_coverage_tables/*_coverage.csv
```

Coverage is computed from the explicit `Match_State -> MSA_Column` mapping recorded in each HHM file.

## Stage 5: compile and run C++/CUDA KL computation

Build the executable:

```bash
cd src/cpp_cuda
make clean
make -j
cd ../..
```

Run KL computation in range mode:

```bash
mkdir -p $PRODIVE_WORK_ROOT/05_cpp_kl

src/cpp_cuda/kl_divergence   --start_i 0 --end_i 100   --start_j 0 --end_j 100   --packed_db $PRODIVE_WORK_ROOT/02_packed_fragments/fragment_${FRAGMENT}_packed   --memory_efficient   --num_streams 8   --max_gpus 4   --outdir $PRODIVE_WORK_ROOT/05_cpp_kl
```

For selected pairs, use pair-list mode:

```bash
src/cpp_cuda/kl_divergence   --pair_list /path/to/pairs.csv   --packed_db $PRODIVE_WORK_ROOT/02_packed_fragments/fragment_${FRAGMENT}_packed   --memory_efficient   --num_streams 8   --max_gpus 4   --outdir $PRODIVE_WORK_ROOT/05_cpp_kl
```

Pair-list input format:

```text
i,j
0,1
2,5
```

Expected output:

```text
$PRODIVE_WORK_ROOT/05_cpp_kl/
├── kl_0000_0001.npy
├── prodive_run_manifest.json
├── prodive_computed_ranges.csv       # range mode
├── prodive_computed_pairs.csv        # pair-list mode
└── RUN_COMPLETE
```

## Stage 6: filter C++ KL matrices

```bash
mkdir -p $PRODIVE_WORK_ROOT/06_filtered_pk

python3 src/path_extraction/scripts/01_filter_cpp_kl_results.py   --input-dir $PRODIVE_WORK_ROOT/05_cpp_kl   --output-dir $PRODIVE_WORK_ROOT/06_filtered_pk   --mapping-file $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt   --background-json $PRODIVE_WORK_ROOT/03_background_kl/background_kl_fragment_${FRAGMENT}.json   --fragment $FRAGMENT   --raw-threshold 2.0   --filter-threshold 13.0   --epsilon 0.0001
```

Expected output:

```text
$PRODIVE_WORK_ROOT/06_filtered_pk/
├── kl_<query_record_name>_filtered.pk
└── filter_cpp_kl_run_summary.json
```

Filtering rule:

```text
raw4 = round(raw_score, 4)
keep raw4 < 2.0
linear = (query_background[row] + target_background[col]) / (raw4 + 0.0001) * (3 * fragment + 2)
keep linear >= 13.0
```

The saved score is `round(linear, 4)`.

## Stage 7: build diagonal paths

```bash
mkdir -p $PRODIVE_WORK_ROOT/07_paths

python3 src/path_extraction/scripts/02_build_paths_from_filtered_pk.py   --input-dir $PRODIVE_WORK_ROOT/06_filtered_pk   --output-dir $PRODIVE_WORK_ROOT/07_paths   --mapping-file $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt   --fragment $FRAGMENT   --min-path-points 5   --workers 20
```

Expected output:

```text
$PRODIVE_WORK_ROOT/07_paths/
├── global_high_score_summary.csv
├── global_UNCL_overlap_report.csv
├── global_jump_gap_report.csv
└── path_building_summary.json
```

`global_high_score_summary.csv` is the standard input for final rescoring.

## Stage 8: coverage-based completeness rescoring

```bash
mkdir -p $PRODIVE_WORK_ROOT/08_final_paths

python3 src/path_extraction/scripts/03_rescore_paths_by_coverage.py   --input-csv $PRODIVE_WORK_ROOT/07_paths/global_high_score_summary.csv   --coverage-dir $PRODIVE_WORK_ROOT/04_coverage_tables   --output-csv $PRODIVE_WORK_ROOT/08_final_paths/final_paths.csv   --rescored-all-csv $PRODIVE_WORK_ROOT/08_final_paths/all_paths_rescored.csv   --coverage-threshold 0.85   --score-threshold 1.2   --score-distribution-png $PRODIVE_WORK_ROOT/08_final_paths/score_distribution_before_after.png
```

Expected output:

```text
$PRODIVE_WORK_ROOT/08_final_paths/
├── final_paths.csv
├── final_paths.summary.json
├── all_paths_rescored.csv
└── score_distribution_before_after.png
```

Rows are retained when:

```text
Coverage_Main >= 0.85
Coverage_Sub  >= 0.85
Rescored_Score > 1.2
```

The adjusted score is:

```text
Rescored_Score = log10(Avg_Similarity * (((Coverage_Main + Coverage_Sub) / 2)^2))
```

## Large-scale runs

Full Pfam-scale computation is typically divided into numeric ID ranges or explicit pair lists according to available GPU memory, storage capacity, and runtime constraints. Each completed C++/CUDA output directory contains a run manifest and a `RUN_COMPLETE` marker that can be used by downstream filtering steps to verify completion.
