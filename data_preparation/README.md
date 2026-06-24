# Data preparation

This directory contains the Python preprocessing scripts used before C++/CUDA KL-divergence computation.

The scripts are command-line programs. They do not require installation as a Python package.

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

Python 3.8+ is recommended. `tqdm` is optional; without it, scripts still run without progress bars.

## Step 1: generate HHM mapping

Input:

```text
HHM root directory containing recursive *.hhm files
```

Command:

```bash
python3 scripts/01_generate_pfam_hhm_mapping.py \
  --root_dir /path/to/hhm_root \
  --output /path/to/hhm_mapping.txt
```

Output:

```text
hhm_mapping.txt
```

Default mapping format:

```text
PF00001: 0
PF00002: 1
```

The default record ID is the filename stem. For datasets with repeated filenames or repeated stems, use relative-path IDs:

```bash
python3 scripts/01_generate_pfam_hhm_mapping.py \
  --root_dir /path/to/hhm_root \
  --output /path/to/hhm_mapping.txt \
  --id_source relative_path
```

Record IDs must not contain the colon character `:` because the mapping format uses `ID: integer`.

## Step 2: pack fixed-length HHM fragments

Input:

```text
HHM root directory
hhm_mapping.txt
```

Command:

```bash
python3 scripts/02_pack_hhm_fragments.py \
  --input_dir /path/to/hhm_root \
  --output_dir /path/to/fragment_6_packed \
  --mapping /path/to/hhm_mapping.txt \
  --fragment 6 \
  --jobs 40 \
  --overwrite
```

Output:

```text
fragment_6_packed/
├── pi_all.float32.bin
├── A_all.float32.bin
├── B_all.float32.bin
├── index.csv
├── metadata.json
├── summary.json
└── failed_files.csv        # written only when failed files exist
```

`index.csv` stores element offsets, not byte offsets. The binary arrays are flattened in C order. The current C++/CUDA reader expects `float32` packed arrays.

Optional exclusion patterns can be supplied when needed:

```bash
python3 scripts/02_pack_hhm_fragments.py \
  --input_dir /path/to/hhm_root \
  --output_dir /path/to/fragment_6_packed \
  --mapping /path/to/hhm_mapping.txt \
  --exclude_pattern '*_filtered.hhm' \
  --overwrite
```

## Step 3: generate background KL JSON

Input:

```text
HHM root directory
hhm_mapping.txt
```

Command:

```bash
python3 scripts/03_generate_background_kl.py \
  --input_dir /path/to/hhm_root \
  --mapping /path/to/hhm_mapping.txt \
  --output_dir /path/to/background_kl \
  --fragments 6
```

Output:

```text
background_kl/
├── background_kl_fragment_6.json
└── background_kl_generation_summary.json
```

Each JSON array length is `HMM_length - fragment + 1`, matching the number of sliding windows used by the packed fragment database and C++/CUDA KL computation.

The background value for each window is the mean of per-column `KL(match-state emission || NULL background)` values over the window. Stored values are rounded to four decimals by default.

Multiple fragment lengths can be generated in one run:

```bash
python3 scripts/03_generate_background_kl.py \
  --input_dir /path/to/hhm_root \
  --mapping /path/to/hhm_mapping.txt \
  --output_dir /path/to/background_kl \
  --fragments 2,3,4,5,6,7,8,9,10
```

Use `--strict` to terminate the run on missing or failed HHM records.

## Step 4: generate match-state coverage tables

Coverage-based rescoring requires per-match-state completeness values from the alignment used to build each HHM.

Single-file mode:

```bash
python3 scripts/04_generate_hhm_coverage.py single \
  --hhm /path/to/PF00001.hhm \
  --alignment /path/to/PF00001.sto \
  --output-csv /path/to/PF00001_coverage.csv
```

Batch mode:

```bash
python3 scripts/04_generate_hhm_coverage.py batch \
  --input-dir /path/to/hhm_root \
  --output-dir /path/to/coverage_tables \
  --jobs 20
```

Output schema:

```text
Match_State,MSA_Column,Residue_Count,Total_Seqs,Completeness_Percent
```

The script reads the explicit match-state to MSA-column mapping recorded in each HHM file. It does not re-infer match columns using a residue-fraction rule.

Alignment files are searched by same stem. Supported extensions include:

```text
.sto, .stockholm, .fas, .fasta, .fa, .a3m, .afa
```

When alignments are stored separately from HHM files, provide:

```bash
--alignment-root /path/to/alignment_root
```
