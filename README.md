# ProDive

ProDive is a computational framework for detecting local similarities between profile hidden Markov models (profile HMMs). This repository provides a common entry point for the ProDive method implementation and the downstream analyses of the detected local fragments.

## Repository structure

| Directory | Scope |
| --- | --- |
| [`prodive/`](prodive/) | Core ProDive method, including profile-HMM preprocessing, GPU-accelerated symmetric KL-divergence calculations, multi-stage filtering, path construction, parameter-sensitivity analyses, structural validation, and computational benchmarking. |
| [`prodive-fragment-analysis/`](prodive-fragment-analysis/) | Downstream characterization of ProDive-detected fragments, including community detection, structural comparisons, secondary-structure and solvent-accessibility analyses, disorder analyses, 3did interface comparisons, ESM2 analyses, and phi-value comparisons. |

Each directory is a self-contained workflow with its own `README.md`, software requirements, input descriptions, execution commands, parameters, and output documentation.

## Data availability

The input resources and precomputed results for the two workflows are available from their corresponding Zenodo records:

| Workflow | Code directory | Data record |
| --- | --- | --- |
| ProDive method and validation | [`prodive/`](prodive/) | [10.5281/zenodo.22009469](https://doi.org/10.5281/zenodo.22009469) |
| Fragment-level analyses | [`prodive-fragment-analysis/`](prodive-fragment-analysis/) | [10.5281/zenodo.21932666](https://doi.org/10.5281/zenodo.21932666) |

Download the data record corresponding to the workflow that you intend to run. The detailed directory layout and required environment variables are documented in the relevant code directory.

Large generated intermediates, including packed profile databases, dense GPU KL-divergence matrices, and temporary path-building files, are not stored in this GitHub repository and must be generated locally when required.

## Getting started

1. Select the relevant workflow.
2. Open its detailed documentation:
   - [ProDive method documentation](prodive/README.md)
   - [Fragment-analysis documentation](prodive-fragment-analysis/README.md)
3. Download and extract the matching Zenodo data record.
4. Install the dependencies and configure the data and output paths described in the selected workflow.
5. Run the documented commands or use the released precomputed results.

## Workflow relationship

The `prodive/` workflow detects and scores local profile-HMM similarities. Its final filtered segment correspondences provide the starting point for the biological and structural analyses in `prodive-fragment-analysis/`.

The released precomputed results allow individual validation or analysis modules to be examined without repeating the complete GPU-intensive detection workflow.

## Reproducibility notes

- Default parameters and exact command-line interfaces are documented in the README within each code directory.
- Full regeneration of the ProDive search requires substantial GPU computation and local storage for intermediate matrices.
- Structure-dependent analyses require the structure resources described in the relevant module documentation.
- New outputs should be written to separate working directories rather than into the released data directories.

