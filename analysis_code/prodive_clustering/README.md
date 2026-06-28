# Leiden clustering of ProDive fragment correspondences

This module constructs weighted graphs from ProDive fragment correspondences, merges near-identical segment nodes, performs Leiden community detection, and summarizes structural annotations within the resulting communities.

## Inputs

| Input | Source |
|---|---|
| ProDive score-percentile subsets | `$PRODIVE_DATA_ROOT/shared/score_percentile_subsets/` |
| Pfam secondary-structure/RSA table | Output of `prodive_secondary_structure_rsa/scripts/pfam/01_compute_pfam_real_ss_rsa.py` or the corresponding released table |

## Workflow

| Step | Script | Main input | Main output |
|---:|---|---|---|
| 1 | `scripts/build_leiden_clusters_with_struct_stats.py` | Top-score ProDive subset CSVs and Pfam SS/RSA annotations | Leiden community tables, node and edge tables, raw-to-merged maps, GraphML files, and community-level structural summaries |

Run template:

```bash
cd prodive_clustering
bash run_templates/run_leiden_clustering.sh
```

## Outputs

Generated outputs include cluster membership tables, graph files, merged-node maps, and summary statistics for structural composition within communities. Output paths are controlled by the arguments in the run template.
