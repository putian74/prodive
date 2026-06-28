# Data preparation

This directory contains the Python preprocessing scripts used before C++/CUDA KL-divergence computation. These scripts prepare HHM records, fixed-length HHM fragments, background KL values, and match-state coverage tables for downstream ProDive analysis.

## Scripts

| Script | Purpose | Main input | Main output |
|---|---|---|---|
| `01_generate_pfam_hhm_mapping.py` | Assign numeric IDs to HHM records | recursive HHM root | mapping file |
| `02_pack_hhm_fragments.py` | Convert HHM windows into packed binary arrays | HHM root, mapping file | packed fragment database |
| `03_generate_background_kl.py` | Compute per-window background information | HHM root, mapping file | background KL JSON |
| `04_generate_hhm_coverage.py` | Compute match-state coverage from alignments | HHM and alignment files | `*_coverage.csv` tables |

## Requirements

```bash
python3 -m pip install -r requirements.txt
```

Python 3.8 or later is recommended. The `tqdm` package is optional; if it is unavailable, the scripts run without progress bars.

## Step 1: generate HHM mapping

The mapping file assigns an integer record ID to each HHM file. These numeric IDs are used by the packed fragment database and the C++/CUDA KL computation stage.

```bash
python3 scripts/01_generate_pfam_hhm_mapping.py   --root_dir /path/to/hhm_root   --output /path/to/hhm_mapping.txt
```

Default mapping format:

```text
PF00001: 0
PF00002: 1
```

The default record ID is the filename stem. For datasets in which filename stems may repeat, use relative-path IDs:

```bash
python3 scripts/01_generate_pfam_hhm_mapping.py   --root_dir /path/to/hhm_root   --output /path/to/hhm_mapping.txt   --id_source relative_path
```

Available ID modes are:

```text
stem
filename
relative_path
```

Record IDs must not contain the colon character `:` because the mapping file uses the `ID: integer` format.

## Step 2: pack fixed-length HHM fragments

This step converts each HHM into fixed-length sliding-window fragments and stores the transition, emission, and initial-state arrays in packed binary format.

```bash
python3 scripts/02_pack_hhm_fragments.py   --input_dir /path/to/hhm_root   --output_dir /path/to/fragment_6_packed   --mapping /path/to/hhm_mapping.txt   --fragment 6   --jobs 40   --overwrite
```

Expected output:

```text
fragment_6_packed/
├── pi_all.float32.bin
├── A_all.float32.bin
├── B_all.float32.bin
├── index.csv
├── metadata.json
├── summary.json
└── failed_files.csv        # written when failed files are present
```

`index.csv` stores element offsets rather than byte offsets. The binary arrays are flattened in C order. The C++/CUDA reader expects `float32` packed arrays.

Optional exclusion patterns can be supplied when needed:

```bash
python3 scripts/02_pack_hhm_fragments.py   --input_dir /path/to/hhm_root   --output_dir /path/to/fragment_6_packed   --mapping /path/to/hhm_mapping.txt   --exclude_pattern '*_filtered.hhm'   --overwrite
```

## Step 3: generate background KL JSON

This step computes the per-window background information used by the downstream normalized filtering stage.

```bash
python3 scripts/03_generate_background_kl.py   --input_dir /path/to/hhm_root   --mapping /path/to/hhm_mapping.txt   --output_dir /path/to/background_kl   --fragments 6
```

Expected output:

```text
background_kl/
├── background_kl_fragment_6.json
└── background_kl_generation_summary.json
```

Each JSON array has length `HMM_length - fragment + 1`, matching the number of sliding windows used by the packed fragment database and the C++/CUDA KL computation.

The background value for each window is the mean of per-column `KL(match-state emission || null background)` values over the window. Stored values are rounded to four decimal places by default.

Multiple fragment lengths can be generated in one run:

```bash
python3 scripts/03_generate_background_kl.py   --input_dir /path/to/hhm_root   --mapping /path/to/hhm_mapping.txt   --output_dir /path/to/background_kl   --fragments 2,3,4,5,6,7,8,9,10
```

Use `--strict` to terminate the run on missing or failed HHM records.

## Step 4: generate match-state coverage tables

Coverage-based rescoring requires per-match-state completeness values from the alignment used to build each HHM.

Single-file mode:

```bash
python3 scripts/04_generate_hhm_coverage.py single   --hhm /path/to/PF00001.hhm   --alignment /path/to/PF00001.sto   --output-csv /path/to/PF00001_coverage.csv
```

Batch mode:

```bash
python3 scripts/04_generate_hhm_coverage.py batch   --input-dir /path/to/hhm_root   --output-dir /path/to/coverage_tables   --jobs 20
```

Expected output schema:

```text
Match_State,MSA_Column,Residue_Count,Total_Seqs,Completeness_Percent
```

The script reads the explicit match-state to MSA-column mapping recorded in each HHM file. It does not infer match columns by residue-fraction thresholding.

Alignment files are searched by matching filename stem. Supported extensions include:

```text
.sto, .stockholm, .fas, .fasta, .fa, .a3m, .afa
```

When alignments are stored in a separate directory, provide:

```bash
--alignment-root /path/to/alignment_root
```
