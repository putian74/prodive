# Contact-order analysis

This module maps ProDive fragments onto protein structures, computes fragment-related contact order and relative fragment contact order (rFCO), constructs same-chain and same-length random controls, and performs follow-up analyses for high-rFCO fragments.

## Inputs

| Input | Source |
|---|---|
| Pfam-Pfam ProDive result | `$PRODIVE_DATA_ROOT/shared/global_high_score_summary_fin.csv` |
| Pfam seed HHM/STO/FAS files | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/` |
| Experimental Pfam PDB structures | `$PRODIVE_PFAM_RUNTIME_ROOT/` (merged with HHM/alignment files) |
| SIFTS/PDBe cache | `$CONTACT_ORDER_CACHE`, default `$PRODIVE_WORK_ROOT/contact_order/cache/` |

Set `PRODIVE_DATA_ROOT`, `PRODIVE_WORK_ROOT`, and `PRODIVE_PFAM_RUNTIME_ROOT` before running. The runtime must expose profiles, alignments, and structures in each Pfam family directory. The `--cache-dir` argument points to the cache root; its `sifts/` subdirectory stores API responses. Network access is required for uncached PDBe/SIFTS lookups.

## Workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/01_compute_contact_order_and_random.py` | ProDive result, Pfam runtime directory, and SIFTS cache | Actual and random contact-order tables plus summary text |
| 2 | `scripts/02_compare_actual_vs_random_dedup.py` | Actual and random tables from Step 1 | Deduplicated real-vs-random summaries and plots |
| 3 | `scripts/03_select_high_rfco_tail.py` | Step 2 summary table | High-rFCO tail tables |
| 4 | `scripts/04_draw_high_rfco_contact_topology.py` | Step 1 and Step 2 outputs | Contact-topology figures for selected high-rFCO cases |
| 5 | `scripts/05_remove_high_rfco_sensitivity.py` | Step 2 summary table | Sensitivity tables after high-rFCO removal |
| 6 | `scripts/06_merge_high_rfco_with_prodive_score.py` | High-rFCO cases and the ProDive result table | High-rFCO cases annotated with original ProDive scores |

Run templates from the repository root:

```bash
bash prodive_contact_order/run_templates/run_full_contact_order_pipeline.sh
bash prodive_contact_order/run_templates/run_high_rfco_followup.sh
```

## Outputs

The workflow produces CSV summary tables and figures describing the rFCO distribution of ProDive fragments relative to matched random controls.

The default main run uses an 8-angstrom contact cutoff, excludes sequence neighbors within two residues, and requests 1,000 random windows per fragment. Step 2 deduplicates identical structural intervals. Override workload settings with `JOBS`, `PREPROCESS_JOBS`, `CHUNK_SIZE`, and `RANDOM_SAMPLES`. Outputs are written under `$PRODIVE_WORK_ROOT/contact_order/`.
