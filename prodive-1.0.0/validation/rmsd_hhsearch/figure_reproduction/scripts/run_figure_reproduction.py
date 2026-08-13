#!/usr/bin/env python3
"""Run figure-level reproduction from deposited RMSD and random-control CSV files.

This wrapper draws the Fig. 2 / Fig. S3 RMSD density panels and the Fig. S4 / Fig. S5
real-versus-random boxplots from configuration files. It does not recompute RMSD.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List


def resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def run(cmd: List[str], cwd: Path) -> None:
    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Fig. 2/S3 and random-control figures from deposited CSV files.")
    parser.add_argument("--fig2-config", type=Path, default=Path("configs/fig2_datasets.example.csv"), help="Dataset config for Fig. 2 and Fig. S3 panels.")
    parser.add_argument("--random-config", type=Path, default=Path("configs/random_control_datasets.example.csv"), help="Dataset config for Fig. S4/S5 random-control panels.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--skip-fig2", action="store_true", help="Skip Fig. 2/S3 density panels.")
    parser.add_argument("--skip-random", action="store_true", help="Skip random-control boxplots and t-tests.")
    parser.add_argument("--workers", type=int, default=0, help="Worker count for density-panel generation. Use 0 for automatic selection.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    package_dir = Path(__file__).resolve().parents[1]
    scripts_dir = package_dir / "scripts"
    base_dir = package_dir
    out_dir = args.output_dir if args.output_dir.is_absolute() else (package_dir / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig2_config = args.fig2_config if args.fig2_config.is_absolute() else (package_dir / args.fig2_config)
    random_config = args.random_config if args.random_config.is_absolute() else (package_dir / args.random_config)

    if not args.skip_fig2:
        cmd = [
            sys.executable,
            str(scripts_dir / "plot_fig2_and_s3_panels.py"),
            "--dataset-config", str(fig2_config),
            "--output-dir", str(out_dir / "fig2_and_s3_panels"),
        ]
        if args.workers > 0:
            cmd.extend(["--workers", str(args.workers)])
        run(cmd, package_dir)

    if not args.skip_random:
        if not random_config.exists():
            raise FileNotFoundError(f"Random-control config does not exist: {random_config}")
        with random_config.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row or not row.get("analysis_key"):
                    continue
                stats_csv = resolve_path(base_dir, row["output_stats_csv"])
                plot_path = resolve_path(base_dir, row["output_plot"])
                cmd = [
                    sys.executable,
                    str(scripts_dir / "plot_real_vs_random_core_boxplot.py"),
                    "--real-csv", resolve_path(base_dir, row["real_csv"]),
                    "--random-csv", resolve_path(base_dir, row["random_csv"]),
                    "--output-stats-csv", stats_csv,
                    "--output-plot", plot_path,
                    "--real-label", row.get("real_label") or "Real signal",
                    "--random-label", row.get("random_label") or "Random noise",
                    "--plot-title", row.get("plot_title") or "Core region comparison: real vs. random",
                ]
                run(cmd, package_dir)

                ttest_csv = row.get("ttest_csv", "").strip()
                if ttest_csv:
                    run([
                        sys.executable,
                        str(scripts_dir / "welch_ttest_from_stats.py"),
                        "--stats-csv", stats_csv,
                        "--output-csv", resolve_path(base_dir, ttest_csv),
                    ], package_dir)


if __name__ == "__main__":
    main()
