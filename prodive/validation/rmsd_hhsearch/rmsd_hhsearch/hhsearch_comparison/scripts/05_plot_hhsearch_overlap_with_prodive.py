#!/usr/bin/env python3
"""
Generate the HHsearch-centric companion to the ProDive overlap figure.

For each ProDive score-percentile subset, all HHsearch family-pair hits are
partitioned into:

1. Shared with ProDive
2. HHsearch-only

The script reads the detailed overlap tables produced by
03_check_hhsearch_overlap.py. A HHsearch family pair is shared when at least
one ProDive path for that pair has Dual_Min_Coverage greater than or equal to
the requested coverage threshold.

Default inputs
--------------
1. hhsuite20.csv
2. subset_data_Top_05%_overlap_check_dual_20.csv
3. subset_data_Top_10%_overlap_check_dual_20.csv
4. subset_data_Top_20%_overlap_check_dual_20.csv
5. subset_data_Top_50%_overlap_check_dual_20.csv
6. subset_data_Top_100%_overlap_check_dual_20.csv

Outputs
-------
1. hhsearch_overlap_with_prodive_summary.csv
2. hhsearch_overlap_with_prodive.png
3. hhsearch_overlap_with_prodive.pdf
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


Pair = Tuple[str, str]

DEFAULT_PERCENTILE_TAGS = ["05", "10", "20", "50", "100"]
DEFAULT_DISPLAY_LABELS = ["Top 5%", "Top 10%", "Top 20%", "Top 50%", "Top 100%"]


def require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def normalize_family_id(value: object, column: str, row_number: int) -> str:
    if pd.isna(value):
        raise ValueError(f"Missing {column} value at data row {row_number:,}.")

    family_id = str(value).strip()
    if not family_id:
        raise ValueError(f"Empty {column} value at data row {row_number:,}.")

    return family_id


def canonical_pair(main_hmm: object, sub_hmm: object, row_number: int) -> Pair:
    main_id = normalize_family_id(main_hmm, "Main_HMM", row_number)
    sub_id = normalize_family_id(sub_hmm, "Sub_HMM", row_number)
    return tuple(sorted((main_id, sub_id)))


def build_pair_set(df: pd.DataFrame) -> Set[Pair]:
    pairs: Set[Pair] = set()
    for row_number, row in enumerate(
        df[["Main_HMM", "Sub_HMM"]].itertuples(index=False), start=2
    ):
        pairs.add(canonical_pair(row.Main_HMM, row.Sub_HMM, row_number))
    return pairs


def threshold_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).rstrip("0").rstrip(".")


def load_hhsearch_pairs(
    csv_path: Path,
    probability_threshold: float,
    encoding: str,
) -> Tuple[Set[Pair], int, int]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"HHsearch CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path, encoding=encoding)
    require_columns(df, ["Main_HMM", "Sub_HMM", "Prob"], "HHsearch CSV")

    probability = pd.to_numeric(df["Prob"], errors="coerce")
    invalid_probability = int(probability.isna().sum())
    if invalid_probability:
        raise ValueError(
            f"HHsearch CSV contains {invalid_probability:,} rows with a non-numeric Prob value."
        )

    filtered = df.loc[probability >= probability_threshold, ["Main_HMM", "Sub_HMM"]].copy()
    pairs = build_pair_set(filtered)

    if not pairs:
        raise ValueError(
            f"No HHsearch family pairs remain after applying Prob >= {probability_threshold:g}."
        )

    return pairs, len(df), len(filtered)


def default_overlap_files(probability_threshold: float) -> List[str]:
    tag = threshold_tag(probability_threshold)
    return [
        f"subset_data_Top_{percentile}%_overlap_check_dual_{tag}.csv"
        for percentile in DEFAULT_PERCENTILE_TAGS
    ]


def summarize_overlap_file(
    csv_path: Path,
    label: str,
    hhsearch_pairs: Set[Pair],
    coverage_threshold: float,
    encoding: str,
) -> dict:
    if not csv_path.is_file():
        raise FileNotFoundError(f"ProDive overlap CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path, encoding=encoding)
    require_columns(
        df,
        ["Main_HMM", "Sub_HMM", "Dual_Min_Coverage"],
        f"ProDive overlap CSV ({csv_path.name})",
    )

    coverage = pd.to_numeric(df["Dual_Min_Coverage"], errors="coerce")
    invalid_coverage = int(coverage.isna().sum())
    if invalid_coverage:
        raise ValueError(
            f"{csv_path.name} contains {invalid_coverage:,} rows with a non-numeric "
            "Dual_Min_Coverage value."
        )

    shared_path_mask = coverage >= coverage_threshold
    shared_path_rows = df.loc[shared_path_mask, ["Main_HMM", "Sub_HMM"]]
    shared_candidate_pairs = build_pair_set(shared_path_rows)
    shared_pairs = shared_candidate_pairs.intersection(hhsearch_pairs)
    hhsearch_only_pairs = hhsearch_pairs.difference(shared_pairs)
    unexpected_shared_pairs = shared_candidate_pairs.difference(hhsearch_pairs)

    total = len(hhsearch_pairs)
    shared = len(shared_pairs)
    hhsearch_only = len(hhsearch_only_pairs)

    if shared + hhsearch_only != total:
        raise RuntimeError(
            f"Internal counting error for {label}: {shared} + {hhsearch_only} != {total}."
        )

    return {
        "Dataset": label,
        "Input_File": csv_path.name,
        "ProDive_Path_Rows": len(df),
        "Shared_ProDive_Path_Rows": int(shared_path_mask.sum()),
        "HHsearch_Pairs_Total": total,
        "Shared_With_ProDive_Pairs": shared,
        "HHsearch_Only_Pairs": hhsearch_only,
        "Shared_Percent": 100.0 * shared / total,
        "HHsearch_Only_Percent": 100.0 * hhsearch_only / total,
        "Shared_Candidate_Pairs_Not_In_HHsearch": len(unexpected_shared_pairs),
    }


def compact_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def add_segment_labels(
    ax: plt.Axes,
    bars: Iterable,
    total: int,
    position: str,
    fontsize: float,
) -> None:
    for bar in bars:
        height = int(round(bar.get_height()))
        if height <= 0:
            continue

        if position == "above":
            y = bar.get_y() + bar.get_height() + total * 0.008
            vertical_alignment = "bottom"
            text_color = "black"
        elif position == "center":
            y = bar.get_y() + bar.get_height() / 2
            vertical_alignment = "center"
            text_color = "white"
        else:
            raise ValueError(f"Unsupported label position: {position}")

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            compact_count(height),
            ha="center",
            va=vertical_alignment,
            color=text_color,
            fontsize=fontsize,
            fontweight="bold",
        )


def plot_summary(
    summary_df: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    probability_threshold: float,
    coverage_threshold: float,
    dpi: int,
) -> None:
    labels = summary_df["Dataset"].tolist()
    shared = summary_df["Shared_With_ProDive_Pairs"].astype(int).tolist()
    hhsearch_only = summary_df["HHsearch_Only_Pairs"].astype(int).tolist()
    total = int(summary_df["HHsearch_Pairs_Total"].iloc[0])

    x = list(range(len(labels)))
    width = 0.62

    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=dpi)

    shared_bars = ax.bar(
        x,
        shared,
        width,
        label="Shared with ProDive",
        color="#4c72b0",
        edgecolor="white",
        linewidth=0.8,
    )
    hhsearch_only_bars = ax.bar(
        x,
        hhsearch_only,
        width,
        bottom=shared,
        label="HHsearch-only",
        color="#dd8452",
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_title(
        "HHsearch hits partitioned by overlap with ProDive\n"
        f"(Prob >= {probability_threshold:g}%, dual coverage threshold = {coverage_threshold:g})",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("ProDive dataset (score percentile)", fontsize=12)
    ax.set_ylabel("Number of unique Pfam family pairs", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=5))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.set_ylim(0, total * 1.09)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=11,
    )

    add_segment_labels(ax, shared_bars, total, position="center", fontsize=8)
    add_segment_labels(ax, hhsearch_only_bars, total, position="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def check_expected_trend(summary_df: pd.DataFrame) -> None:
    shared = summary_df["Shared_With_ProDive_Pairs"].astype(int).tolist()
    if any(current < previous for previous, current in zip(shared, shared[1:])):
        print(
            "Warning: shared HHsearch-pair counts are not non-decreasing across the "
            "ProDive percentile subsets. Check that the supplied files are nested and "
            "listed from Top 5% to Top 100%.",
            file=sys.stderr,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all HHsearch family-pair hits as shared with ProDive or HHsearch-only "
            "across ProDive score-percentile subsets."
        )
    )
    parser.add_argument(
        "--hhsuite-csv",
        type=Path,
        required=True,
        help="Parsed HHsearch CSV, for example hhsuite20.csv.",
    )
    parser.add_argument(
        "--overlap-dir",
        type=Path,
        required=True,
        help="Directory containing subset_data_Top_*_overlap_check_dual_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: the value of --overlap-dir.",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=20.0,
        help="Minimum HHsearch probability and default filename tag. Default: 20.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.2,
        help="Minimum Dual_Min_Coverage for a shared hit. Default: 0.2.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help=(
            "Overlap CSV filenames under --overlap-dir. Default: Top 5%%, 10%%, 20%%, "
            "50%%, and 100%% files matching --probability-threshold."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display labels corresponding to --files. Default: Top 5%% ... Top 100%%.",
    )
    parser.add_argument(
        "--output-prefix",
        default="hhsearch_overlap_with_prodive",
        help="Prefix for the summary CSV, PNG, and PDF. Default: hhsearch_overlap_with_prodive.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV input/output encoding. Default: utf-8-sig.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300.",
    )
    return parser


def validate_args(args: argparse.Namespace, files: Sequence[str], labels: Sequence[str]) -> None:
    if not math.isfinite(args.probability_threshold):
        raise ValueError("--probability-threshold must be finite.")
    if not math.isfinite(args.coverage_threshold) or not 0 <= args.coverage_threshold <= 1:
        raise ValueError("--coverage-threshold must be a finite value in [0, 1].")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero.")
    if len(files) != len(labels):
        raise ValueError(
            f"The number of --files ({len(files)}) must equal the number of "
            f"--labels ({len(labels)})."
        )
    if not files:
        raise ValueError("At least one overlap CSV must be supplied.")


def main() -> None:
    args = build_arg_parser().parse_args()

    files = args.files or default_overlap_files(args.probability_threshold)
    if args.labels is not None:
        labels = args.labels
    elif args.files is None:
        labels = DEFAULT_DISPLAY_LABELS
    else:
        labels = [Path(filename).stem for filename in files]

    validate_args(args, files, labels)

    output_dir = args.output_dir or args.overlap_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hhsearch_pairs, raw_hhsearch_rows, filtered_hhsearch_rows = load_hhsearch_pairs(
        csv_path=args.hhsuite_csv,
        probability_threshold=args.probability_threshold,
        encoding=args.encoding,
    )

    summaries = []
    for filename, label in zip(files, labels):
        summaries.append(
            summarize_overlap_file(
                csv_path=args.overlap_dir / filename,
                label=label,
                hhsearch_pairs=hhsearch_pairs,
                coverage_threshold=args.coverage_threshold,
                encoding=args.encoding,
            )
        )

    summary_df = pd.DataFrame(summaries)
    check_expected_trend(summary_df)

    summary_path = output_dir / f"{args.output_prefix}_summary.csv"
    png_path = output_dir / f"{args.output_prefix}.png"
    pdf_path = output_dir / f"{args.output_prefix}.pdf"

    summary_df.to_csv(summary_path, index=False, encoding=args.encoding)
    plot_summary(
        summary_df=summary_df,
        png_path=png_path,
        pdf_path=pdf_path,
        probability_threshold=args.probability_threshold,
        coverage_threshold=args.coverage_threshold,
        dpi=args.dpi,
    )

    print(f"HHsearch CSV rows before probability filtering: {raw_hhsearch_rows:,}")
    print(f"HHsearch CSV rows after probability filtering:  {filtered_hhsearch_rows:,}")
    print(f"Unique HHsearch family pairs used:             {len(hhsearch_pairs):,}")
    print("\nHHsearch-centric overlap summary:")
    print(
        summary_df[
            [
                "Dataset",
                "HHsearch_Pairs_Total",
                "Shared_With_ProDive_Pairs",
                "HHsearch_Only_Pairs",
                "Shared_Percent",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved PNG:     {png_path}")
    print(f"Saved PDF:     {pdf_path}")


if __name__ == "__main__":
    main()
