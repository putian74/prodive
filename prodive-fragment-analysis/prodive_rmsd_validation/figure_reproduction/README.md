# De novo-Pfam RMSD figures

Generate density panels and real-versus-random plots from completed de novo-Pfam RMSD tables.

Edit the input paths in:

- `configs/denovo_datasets.example.csv`
- `configs/denovo_random_control_datasets.example.csv`

Use absolute input paths, or paths relative to `prodive_rmsd_validation/figure_reproduction/`. Relative output paths in the random-control configuration are resolved under `--output-dir`.

From the repository root:

```bash
python3 prodive_rmsd_validation/figure_reproduction/scripts/run_figure_reproduction.py \
  --density-config configs/denovo_datasets.example.csv \
  --random-config configs/denovo_random_control_datasets.example.csv \
  --output-dir "$PRODIVE_WORK_ROOT/rmsd/denovo_figures"
```

Use `--skip-random` to plot only the observed RMSD density, or `--skip-density` to generate only random-control comparisons. Density panels are written under `density_panels/`; comparison outputs follow the filenames in the random-control configuration.

The example configurations contain only de novo-Pfam datasets. The wrapper reads existing results and does not recalculate RMSD.
