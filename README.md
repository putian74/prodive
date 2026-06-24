# ProDive main pipeline

This repository contains the executable code for the main ProDive pipeline, from profile-HMM preprocessing to fragment-level KL-divergence computation, background-normalized filtering, diagonal-path extraction, and final completeness-based rescoring.

The repository is code-only. Raw input data, large intermediate files, structure files, and precomputed analysis results are distributed separately in the external data archive. Do not place the external `data/` or `precomputed_results/` directories inside this GitHub repository.

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

The three post-CUDA scripts are kept together in `src/path_extraction/` because their execution order is sequential and should not be inferred from the older cross-directory numbering.

## External data and output locations

Set the external data and working directories before running the pipeline:

```bash
export PRODIVE_DATA_ROOT=/path/to/ProDive_release/data
export PRODIVE_WORK_ROOT=/path/to/prodive_work
export HHM_ROOT=$PRODIVE_DATA_ROOT/shared/PfamA_seed
export FRAGMENT=6
export JOBS=40
```

The external data archive should provide the HHM/alignment directory used as `HHM_ROOT`. For Pfam-scale runs, each family directory is expected to contain the corresponding HHM and alignment files, for example:

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

All generated files should be written under `$PRODIVE_WORK_ROOT`. The pipeline does not require write access to the external data archive.

## Software requirements

Python 3.8+ is recommended for the Python stages.

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

The bundled Makefile targets NVIDIA Ampere GPUs through `-arch=sm_86`. Modify `NVCC_FLAGS` in `src/cpp_cuda/Makefile` for other GPU architectures.

## Pipeline summary

| Stage | Script or program | Main input | Main output |
|---:|---|---|---|
| 1 | `01_generate_pfam_hhm_mapping.py` | HHM root directory | `hhm_mapping.txt` |
| 2 | `02_pack_hhm_fragments.py` | HHM root, mapping file | packed fragment database |
| 3 | `03_generate_background_kl.py` | HHM root, mapping file | `background_kl_fragment_6.json` |
| 4 | `04_generate_hhm_coverage.py` | HHM and alignment files | `*_coverage.csv` tables |
| 5 | `src/cpp_cuda/kl_divergence` | packed fragment database | `kl_<i>_<j>.npy` matrices |
| 6 | `01_filter_cpp_kl_results.py` | C++ KL matrices, mapping, background JSON | `kl_<query>_filtered.pk` |
| 7 | `02_build_paths_from_filtered_pk.py` | filtered PK files | `global_high_score_summary.csv` |
| 8 | `03_rescore_paths_by_coverage.py` | path table, coverage tables | final rescored path table |

Stages 1--3 and Stage 5 are required for KL computation. Stage 4 is required for the final coverage-based rescoring step.

## Stage 1: generate HHM mapping

Input:

```text
$HHM_ROOT/                 recursive directory containing *.hhm files
```

Command:

```bash
mkdir -p $PRODIVE_WORK_ROOT/01_mapping

python3 data_preparation/scripts/01_generate_pfam_hhm_mapping.py \
  --root_dir $HHM_ROOT \
  --output $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt
```

Output:

```text
$PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt
```

The mapping file assigns one numeric record ID to each HHM record. The C++/CUDA stage uses these numeric IDs in output filenames.

For datasets with repeated filename stems, use relative-path IDs:

```bash
python3 data_preparation/scripts/01_generate_pfam_hhm_mapping.py \
  --root_dir $HHM_ROOT \
  --output $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt \
  --id_source relative_path
```

## Stage 2: pack fixed-length HHM fragments

Input:

```text
$HHM_ROOT/
$PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt
```

Command:

```bash
mkdir -p $PRODIVE_WORK_ROOT/02_packed_fragments

python3 data_preparation/scripts/02_pack_hhm_fragments.py \
  --input_dir $HHM_ROOT \
  --output_dir $PRODIVE_WORK_ROOT/02_packed_fragments/fragment_${FRAGMENT}_packed \
  --mapping $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt \
  --fragment $FRAGMENT \
  --jobs $JOBS \
  --overwrite
```

Output:

```text
$PRODIVE_WORK_ROOT/02_packed_fragments/fragment_6_packed/
├── pi_all.float32.bin
├── A_all.float32.bin
├── B_all.float32.bin
├── index.csv
├── metadata.json
└── summary.json
```

The current C++/CUDA implementation reads `float32` packed arrays. Do not pass `float64` packed files to the C++/CUDA stage.

## Stage 3: generate background KL JSON

Input:

```text
$HHM_ROOT/
$PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt
```

Command:

```bash
mkdir -p $PRODIVE_WORK_ROOT/03_background_kl

python3 data_preparation/scripts/03_generate_background_kl.py \
  --input_dir $HHM_ROOT \
  --mapping $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt \
  --output_dir $PRODIVE_WORK_ROOT/03_background_kl \
  --fragments $FRAGMENT
```

Output:

```text
$PRODIVE_WORK_ROOT/03_background_kl/background_kl_fragment_6.json
$PRODIVE_WORK_ROOT/03_background_kl/background_kl_generation_summary.json
```

Each JSON entry stores the background/window-information vector for one HHM record. The vector length must match the number of sliding windows in the packed fragment database.

## Stage 4: generate match-state coverage tables

Input:

```text
$HHM_ROOT/                 HHM files and corresponding alignment files
```

Command:

```bash
mkdir -p $PRODIVE_WORK_ROOT/04_coverage_tables

python3 data_preparation/scripts/04_generate_hhm_coverage.py batch \
  --input-dir $HHM_ROOT \
  --output-dir $PRODIVE_WORK_ROOT/04_coverage_tables \
  --jobs 20
```

Output:

```text
$PRODIVE_WORK_ROOT/04_coverage_tables/*_coverage.csv
```

Coverage is computed from the explicit `Match_State -> MSA_Column` mapping recorded in each HHM file. The script does not re-infer match columns using a residue-fraction rule.

## Stage 5: compile and run C++/CUDA KL computation

Build:

```bash
cd src/cpp_cuda
make clean
make -j
cd ../..
```

Input:

```text
$PRODIVE_WORK_ROOT/02_packed_fragments/fragment_6_packed/
```

Range-mode command:

```bash
mkdir -p $PRODIVE_WORK_ROOT/05_cpp_kl

src/cpp_cuda/kl_divergence \
  --start_i 0 --end_i 100 \
  --start_j 0 --end_j 100 \
  --packed_db $PRODIVE_WORK_ROOT/02_packed_fragments/fragment_${FRAGMENT}_packed \
  --memory_efficient \
  --num_streams 8 \
  --max_gpus 4 \
  --outdir $PRODIVE_WORK_ROOT/05_cpp_kl
```

Pair-list mode can be used for selected pairs:

```bash
src/cpp_cuda/kl_divergence \
  --pair_list /path/to/pairs.csv \
  --packed_db $PRODIVE_WORK_ROOT/02_packed_fragments/fragment_${FRAGMENT}_packed \
  --memory_efficient \
  --num_streams 8 \
  --max_gpus 4 \
  --outdir $PRODIVE_WORK_ROOT/05_cpp_kl
```

Pair-list input format:

```text
i,j
0,1
2,5
```

Output:

```text
$PRODIVE_WORK_ROOT/05_cpp_kl/
├── kl_0000_0001.npy
├── prodive_run_manifest.json
├── prodive_computed_ranges.csv       # range mode
├── prodive_computed_pairs.csv        # pair-list mode
└── RUN_COMPLETE                      # written only after successful completion
```

The `RUN_COMPLETE` marker and manifest are used by the filtering stage to validate C++ completion.

## Stage 6: filter C++ KL matrices

Input:

```text
$PRODIVE_WORK_ROOT/05_cpp_kl/
$PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt
$PRODIVE_WORK_ROOT/03_background_kl/background_kl_fragment_6.json
```

Command:

```bash
mkdir -p $PRODIVE_WORK_ROOT/06_filtered_pk

python3 src/path_extraction/scripts/01_filter_cpp_kl_results.py \
  --input-dir $PRODIVE_WORK_ROOT/05_cpp_kl \
  --output-dir $PRODIVE_WORK_ROOT/06_filtered_pk \
  --mapping-file $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt \
  --background-json $PRODIVE_WORK_ROOT/03_background_kl/background_kl_fragment_${FRAGMENT}.json \
  --fragment $FRAGMENT \
  --raw-threshold 2.0 \
  --filter-threshold 13.0 \
  --epsilon 0.0001
```

Output:

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

Input:

```text
$PRODIVE_WORK_ROOT/06_filtered_pk/
$PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt
```

Command:

```bash
mkdir -p $PRODIVE_WORK_ROOT/07_paths

python3 src/path_extraction/scripts/02_build_paths_from_filtered_pk.py \
  --input-dir $PRODIVE_WORK_ROOT/06_filtered_pk \
  --output-dir $PRODIVE_WORK_ROOT/07_paths \
  --mapping-file $PRODIVE_WORK_ROOT/01_mapping/hhm_mapping.txt \
  --fragment $FRAGMENT \
  --min-path-points 5 \
  --workers 20
```

Output:

```text
$PRODIVE_WORK_ROOT/07_paths/
├── global_high_score_summary.csv
├── global_UNCL_overlap_report.csv
├── global_jump_gap_report.csv
└── path_building_summary.json
```

`global_high_score_summary.csv` is the clean one-to-one path table and is the normal input for final rescoring.

## Stage 8: coverage-based completeness rescoring

Input:

```text
$PRODIVE_WORK_ROOT/07_paths/global_high_score_summary.csv
$PRODIVE_WORK_ROOT/04_coverage_tables/*_coverage.csv
```

Command:

```bash
mkdir -p $PRODIVE_WORK_ROOT/08_final_paths

python3 src/path_extraction/scripts/03_rescore_paths_by_coverage.py \
  --input-csv $PRODIVE_WORK_ROOT/07_paths/global_high_score_summary.csv \
  --coverage-dir $PRODIVE_WORK_ROOT/04_coverage_tables \
  --output-csv $PRODIVE_WORK_ROOT/08_final_paths/final_paths.csv \
  --rescored-all-csv $PRODIVE_WORK_ROOT/08_final_paths/all_paths_rescored.csv \
  --coverage-threshold 0.85 \
  --score-threshold 1.2 \
  --score-distribution-png $PRODIVE_WORK_ROOT/08_final_paths/score_distribution_before_after.png
```

Output:

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

## Notes for large runs

Full Pfam-scale runs are computationally large. Split C++/CUDA work into ranges or explicit pair lists according to available GPUs and storage. The downstream filtering stage uses the C++ manifest and `RUN_COMPLETE` marker when available, so each C++ output directory should be treated as an atomic completed run.

Do not commit generated data or results to the GitHub repository. Keep the code repository and the external data archive separate.
