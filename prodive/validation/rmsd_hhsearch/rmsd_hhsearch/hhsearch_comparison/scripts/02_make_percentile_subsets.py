#!/usr/bin/env python3
"""
Generate score-percentile subsets from a ProDive high-score summary.

Default output files:
    subset_data_Top_100%.csv
    subset_data_Top_50%.csv
    subset_data_Top_20%.csv
    subset_data_Top_10%.csv
    subset_data_Top_05%.csv
    percentile_thresholds.csv
    family_trends_stats.csv
    trend_line_chart.html                 optional, requires plotly
    trend_parallel_coordinates.html       optional, requires plotly
    trend_retention_heatmap.html          optional, requires plotly
"""

import argparse
import os
from typing import Dict, List

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


DEFAULT_PERCENTILES = [1.0, 0.5, 0.2, 0.1, 0.05]
DEFAULT_LABELS = ["Top_100%", "Top_50%", "Top_20%", "Top_10%", "Top_05%"]


def validate_percentiles_and_labels(percentiles: List[float], labels: List[str]) -> None:
    if len(percentiles) != len(labels):
        raise ValueError("The number of percentiles must match the number of labels.")
    for p in percentiles:
        if not (0 < p <= 1):
            raise ValueError(f"Invalid percentile value: {p}. Each value must be in (0, 1].")


def generate_family_trend_plots(
    stats_df: pd.DataFrame,
    output_dir: str,
    labels: List[str],
    top_n: int,
) -> None:
    """Generate the same three Plotly trend plots as the original script."""
    top_df = stats_df.head(top_n).copy()
    if top_df.empty:
        print("Warning: family statistics are empty. Skip plotting.")
        return

    print(f"Generating trend plots for top {top_n} families...")

    plot_data = top_df.melt(
        id_vars=["Family_ID"],
        value_vars=labels,
        var_name="Percentile_Stage",
        value_name="Count",
    )

    fig_line = px.line(
        plot_data,
        x="Percentile_Stage",
        y="Count",
        color="Family_ID",
        markers=True,
        title=f"Family Count Evolution - Top {top_n}",
        labels={"Percentile_Stage": "Filter Stage", "Count": "Number of Matches"},
    )
    fig_line.write_html(os.path.join(output_dir, "trend_line_chart.html"))

    top_df["Family_ID_Code"] = top_df["Family_ID"].astype("category").cat.codes
    max_val = top_df[labels[0]].max()

    par_coords_dims = []
    for label in labels:
        par_coords_dims.append(dict(range=[0, max_val], label=label, values=top_df[label]))

    fig_par = go.Figure(
        data=go.Parcoords(
            line=dict(color=top_df["Family_ID_Code"], colorscale="Turbo"),
            dimensions=par_coords_dims,
        )
    )
    fig_par.update_layout(title=f"Parallel Coordinates: Family Counts - Top {top_n}")
    fig_par.write_html(os.path.join(output_dir, "trend_parallel_coordinates.html"))

    retention_df = top_df.copy()
    base_col = labels[0]

    for label in labels:
        if label != base_col:
            retention_df[label] = retention_df.apply(
                lambda row: (row[label] / row[base_col]) if row[base_col] > 0 else 0,
                axis=1,
            )
    retention_df[base_col] = 1.0

    heatmap_data = retention_df.melt(
        id_vars=["Family_ID"],
        value_vars=labels,
        var_name="Stage",
        value_name="Retention_Rate",
    )

    fig_heat = px.density_heatmap(
        heatmap_data,
        x="Stage",
        y="Family_ID",
        z="Retention_Rate",
        title=f"Family Retention Rate - Top {top_n}",
        labels={"Retention_Rate": "Retention (0-1)"},
        color_continuous_scale="RdYlGn",
        text_auto=".1%",
    )
    fig_heat.write_html(os.path.join(output_dir, "trend_retention_heatmap.html"))

    print("Plotly trend plots saved.")


def generate_percentile_subsets(
    input_csv_path: str,
    output_dir: str,
    percentiles: List[float],
    labels: List[str],
    top_n: int,
    skip_plots: bool,
) -> None:
    """Generate score percentile subsets."""
    validate_percentiles_and_labels(percentiles, labels)

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data: {input_csv_path}")
    df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
    print(f"Original rows loaded: {len(df):,}")

    # Preserve the original compatibility behavior.
    if "File" not in df.columns and len(df.columns) > 0:
        df.rename(columns={df.columns[0]: "File"}, inplace=True)

    required_cols = ["Main_HMM", "Sub_HMM", "Score"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input CSV is missing required columns: {missing_cols}\n"
            f"Current columns: {list(df.columns)}"
        )

    score_values = pd.to_numeric(df["Score"], errors="coerce")
    bad_score_count = int(score_values.isna().sum())
    if bad_score_count > 0:
        raise ValueError(
            f"Score column contains {bad_score_count:,} non-numeric rows. "
            "To avoid silently deleting rows, the script stops here."
        )

    all_hmms = pd.concat([df["Main_HMM"], df["Sub_HMM"]]).dropna().unique()
    family_stats: Dict[str, Dict[str, int]] = {hmm: {} for hmm in all_hmms}
    threshold_values: Dict[str, float] = {}

    print("\nCalculating score thresholds and saving subset files:")
    for p, label in zip(percentiles, labels):
        if p == 1.0:
            threshold = float(score_values.min())
        else:
            threshold = float(score_values.quantile(1 - p))

        threshold_values[label] = threshold
        subset = df[score_values >= threshold].copy()

        subset_filename = f"subset_data_{label}.csv"
        subset_path = os.path.join(output_dir, subset_filename)
        subset.to_csv(subset_path, index=False, encoding="utf-8-sig")

        print(
            f"  [{label}] Score >= {threshold:.4f} | "
            f"rows: {len(subset):,} -> {subset_filename}"
        )

        current_counts = pd.concat([subset["Main_HMM"], subset["Sub_HMM"]]).value_counts()
        for hmm in all_hmms:
            family_stats[hmm][label] = int(current_counts.get(hmm, 0))

    threshold_df = pd.DataFrame(
        [{"Percentile_Stage": label, "Score_Threshold": val}
         for label, val in threshold_values.items()]
    )
    threshold_csv_path = os.path.join(output_dir, "percentile_thresholds.csv")
    threshold_df.to_csv(threshold_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved thresholds: {threshold_csv_path}")

    stats_df = pd.DataFrame.from_dict(family_stats, orient="index")
    stats_df.index.name = "Family_ID"
    stats_df.reset_index(inplace=True)

    sort_cols = labels[::-1]
    stats_df.sort_values(by=sort_cols, ascending=False, inplace=True)

    stats_csv_path = os.path.join(output_dir, "family_trends_stats.csv")
    stats_df.to_csv(stats_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved family statistics: {stats_csv_path}")

    if skip_plots:
        print("Skipping Plotly trend plots because --skip-plots was used.")
    elif HAS_PLOTLY:
        generate_family_trend_plots(stats_df, output_dir, labels, top_n)
    else:
        print("Warning: plotly is not installed. Skipping interactive trend plots.")

    print(f"Done. Outputs saved to: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create ProDive score-percentile subsets."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input ProDive global high-score summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for percentile subset CSVs and trend statistics.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=DEFAULT_PERCENTILES,
        help="Percentile fractions to retain. Default: 1.0 0.5 0.2 0.1 0.05",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Labels corresponding to percentiles. Default: Top_100%% Top_50%% Top_20%% Top_10%% Top_05%%",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top families shown in trend plots. Default: 20.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip Plotly HTML trend plots.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    generate_percentile_subsets(
        input_csv_path=args.input_csv,
        output_dir=args.output_dir,
        percentiles=args.percentiles,
        labels=args.labels,
        top_n=args.top_n,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
