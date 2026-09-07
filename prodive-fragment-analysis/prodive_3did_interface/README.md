# 3did interface overlap

Map ProDive Pfam-Pfam fragments through local HHM states and seed alignment coordinates onto experimental PDB chains, then measure overlap with known 3did interface residues.

## Inputs and execution

| Input | Path |
|---|---|
| ProDive Pfam-Pfam results | `$PRODIVE_DATA_ROOT/shared/global_high_score_summary_fin.csv` |
| Seed HHM and FAS/STO files | `$PRODIVE_DATA_ROOT/shared/PfamA_seed/PFxxxxx/` |
| Experimental structures | `$PFAM_STRUCTURE_ROOT/PFxxxxx/<UniProt>_exp_<PDBID>_<CHAIN>.pdb` |
| Interface records | `$PRODIVE_DATA_ROOT/3did/3did_flat.gz` |

From the repository root:

```bash
bash prodive_3did_interface/run_templates/run_3did_interface_analysis.sh
```

The template requires `PRODIVE_DATA_ROOT`, `PRODIVE_WORK_ROOT`, and `PFAM_STRUCTURE_ROOT`. It uses a realized-sequence/HMM length ratio of 0.8–1.2, minimum structure coverage 0.8, and maximum search depth 200.

Random windows have the observed fragment's realized sequence length and are sampled within the same selected seed-domain interval, excluding the observed start. Real and random windows use the same union of mapped residues and interface contacts across matched experimental structures. Structure coverage is checked against the HMM fragment length. Each assessable side requests 200 accepted random windows by default; `Random_N` records the number obtained.

| Variable | Default |
|---|---|
| `RANDOM_SAMPLES` | `200` |
| `RANDOM_SEED` | `20260601` |
| `WORKERS` | `20` |
| `INTERFACE_PARTIAL_THRESHOLD` | `0.30` |
| `INTERFACE_MAJOR_THRESHOLD` | `0.50` |
| `SENSITIVITY_THRESHOLDS` | `0.10,0.20,0.30,0.40,0.50` |
| `CONFIDENCE_LEVEL` | `0.95` |

## Outputs

Files are written under `$PRODIVE_WORK_ROOT/3did_interface/`.

| File | Contents |
|---|---|
| `pfam_pfam_3did_interface.csv` | Original rows with Main/Sub interface annotations and random-control statistics. |
| `pfam_pfam_3did_interface.side_level.csv` | One record per fragment side, including status, interface fraction, and interaction-type flags. |
| `pfam_pfam_3did_interface.summary.txt` | Mapping success, interface categories, and comparison summaries. |
| `pfam_pfam_3did_interface.threshold_sensitivity.csv` | Real and random proportions at or above each threshold, paired differences, confidence intervals, and directional/two-sided tests. |
| `pfam_pfam_3did_interface.continuous_random_comparison.csv` | Continuous real-minus-random interface-fraction comparisons. |

`NonInterface` means interface fraction below 0.30, `InterfacePartial` means 0.30 to below 0.50, and `InterfaceMajor` means at least 0.50. The partial and major cutoffs are always included in the sensitivity analysis.

Group statistics cover all assessable sides and subsets defined by interchain/intrachain and same/different Pfam contacts. Confidence intervals and tests account for Main/Sub sides belonging to the same input `Row_ID`; comparison statistics use sides with valid random controls. Interaction flags describe domain/chain relationships and are not a definitive classification of protein identity.

For normalized interface tables, use `--complex-tsv` instead of, or together with, `--three-did-flat`. The `--exclude-same-pfam-pairs` option applies to 3did input. Run `python3 prodive_3did_interface/scripts/01_map_fragments_to_3did_interfaces.py --help` for direct CLI options.
