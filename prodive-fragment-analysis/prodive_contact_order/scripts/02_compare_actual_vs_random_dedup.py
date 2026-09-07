#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare actual contact-order results against random controls.

This version additionally removes duplicated actual fragments that correspond
to the same real interval / same mapped fragment, before matching to random rows.

Example
-------
python3 CHANGE_ME \
  --actual-csv CHANGE_ME \
  --random-csv CHANGE_ME \
  --outdir CHANGE_ME \
  --complete-mapping-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEY_COLS = ["global_row_index", "pfam_id", "pfam_side", "segment_rank"]
ACTUAL_REQUIRED = [
    "status",
    "target_seq_len",
    "mapped_pdb_residue_count",
    "fragment_size_mapped",
    *KEY_COLS,
    "rfco",
    "lr_contact_fraction",
    "n_fragment_contacts",
]

ACTUAL_OPTIONAL = [
    "mean_sequence_separation",
    "chain_length",
    "pdb_id",
    "chain_id",
    "pdb_path",
    "accession",
    "target_hmm_start",
    "target_hmm_end",
    "target_seq_start",
    "target_seq_end",
    "mapped_pdb_residues",
]

RANDOM_NEEDED = KEY_COLS + [
    "random_rfco",
    "random_lr_contact_fraction",
    "random_n_fragment_contacts",
]


class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        self.n += int(arr.size)
        self.sum += float(arr.sum())
        self.sumsq += float(np.square(arr).sum())
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        if amin < self.min:
            self.min = amin
        if amax > self.max:
            self.max = amax

    def mean(self) -> float:
        return np.nan if self.n == 0 else self.sum / self.n

    def std(self) -> float:
        if self.n <= 1:
            return np.nan
        mu = self.mean()
        var = (self.sumsq / self.n) - mu * mu
        return float(np.sqrt(max(var, 0.0)))


def safe_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_columns(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def get_default_interval_dedup_cols(df: pd.DataFrame) -> List[str]:
    """
    These columns are intended to identify the same real fragment interval,
    regardless of which partner HMM caused that fragment to be extracted.
    """
    preferred = [
        "pfam_id",
        "accession",
        "pdb_id",
        "chain_id",
        "target_hmm_start",
        "target_hmm_end",
        "target_seq_start",
        "target_seq_end",
        "mapped_pdb_residues",
    ]
    cols = [c for c in preferred if c in df.columns]

    if len(cols) == 0:
        raise ValueError(
            "No suitable columns found for identical-interval deduplication. "
            "Need at least some of: "
            "pfam_id, accession, pdb_id, chain_id, target_hmm_start, target_hmm_end, "
            "target_seq_start, target_seq_end, mapped_pdb_residues"
        )
    return cols


def deduplicate_identical_intervals(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    dedup_cols = get_default_interval_dedup_cols(df)

    # Normalize string columns
    for c in dedup_cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].fillna("").astype(str)

    dup_mask = df.duplicated(subset=dedup_cols, keep="first")
    removed = df.loc[dup_mask].copy()
    kept = df.loc[~dup_mask].copy()

    return kept, removed, dedup_cols


def filter_actual(
    df: pd.DataFrame,
    complete_mapping_only: bool,
    length: int | None
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    ensure_columns(df, ACTUAL_REQUIRED)

    df = safe_numeric(df, [
        "target_seq_len",
        "mapped_pdb_residue_count",
        "fragment_size_mapped",
        "rfco",
        "lr_contact_fraction",
        "n_fragment_contacts",
        "mean_sequence_separation",
        "chain_length",
        "global_row_index",
        "segment_rank",
        "target_hmm_start",
        "target_hmm_end",
        "target_seq_start",
        "target_seq_end",
    ])

    stats: Dict[str, object] = {
        "rows_raw_input": len(df),
        "rows_after_ok": None,
        "rows_after_complete_mapping": None,
        "rows_after_length_filter": None,
        "rows_after_interval_dedup": None,
        "interval_duplicates_removed": None,
        "dedup_cols": None,
    }

    df = df[df["status"].astype(str) == "OK"].copy()
    df = df.dropna(subset=["global_row_index", "segment_rank", "rfco", "lr_contact_fraction", "n_fragment_contacts"])
    df["global_row_index"] = df["global_row_index"].astype(int)
    df["segment_rank"] = df["segment_rank"].astype(int)
    df["pfam_id"] = df["pfam_id"].astype(str)
    df["pfam_side"] = df["pfam_side"].astype(str)

    # Normalize optional string columns if present
    for c in ["accession", "pdb_id", "chain_id", "pdb_path", "mapped_pdb_residues"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    stats["rows_after_ok"] = len(df)

    if complete_mapping_only:
        df = df.dropna(subset=["target_seq_len", "mapped_pdb_residue_count", "fragment_size_mapped"])
        for c in ["target_seq_len", "mapped_pdb_residue_count", "fragment_size_mapped"]:
            df[c] = df[c].astype(int)
        df = df[
            (df["target_seq_len"] == df["mapped_pdb_residue_count"]) &
            (df["target_seq_len"] == df["fragment_size_mapped"])
        ].copy()

    stats["rows_after_complete_mapping"] = len(df)

    if length is not None:
        if "target_seq_len" not in df.columns:
            raise ValueError("--length requested but target_seq_len column is absent")
        df = df[df["target_seq_len"] == length].copy()

    stats["rows_after_length_filter"] = len(df)

    # Deduplicate identical actual intervals here
    df_dedup, removed_duplicates, dedup_cols = deduplicate_identical_intervals(df.copy())

    stats["rows_after_interval_dedup"] = len(df_dedup)
    stats["interval_duplicates_removed"] = len(removed_duplicates)
    stats["dedup_cols"] = dedup_cols

    return df_dedup, removed_duplicates, stats


def plot_hist(series, title: str, xlabel: str, outpath: Path, bins=40, density=False, ylabel=None):
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if s.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.hist(s, bins=bins, density=density)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel is not None else ("Density" if density else "Count"))
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_overlay_hist_density(actual, random_vals, title: str, xlabel: str, outpath: Path, bins=40,
                              label1="Actual", label2="Random"):
    a = pd.to_numeric(pd.Series(actual), errors="coerce").dropna()
    r = pd.to_numeric(pd.Series(random_vals), errors="coerce").dropna()
    if a.empty or r.empty:
        return
    edges = np.histogram_bin_edges(np.concatenate([a.values, r.values]), bins=bins)
    plt.figure(figsize=(7, 5))
    plt.hist(a, bins=edges, density=True, alpha=0.5, label=label1)
    plt.hist(r, bins=edges, density=True, alpha=0.5, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_scatter(x, y, title: str, xlabel: str, ylabel: str, outpath: Path):
    d = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if d.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(d["x"], d["y"], s=10, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_box_by_length(df: pd.DataFrame, metric: str, outpath: Path, min_count=20, max_lengths=20):
    if "target_seq_len" not in df.columns or metric not in df.columns:
        return
    d = df[["target_seq_len", metric]].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna()
    if d.empty:
        return
    counts = d["target_seq_len"].value_counts()
    lengths = sorted([int(k) for k, v in counts.items() if v >= min_count])
    if not lengths:
        return
    if len(lengths) > max_lengths:
        keep = counts[counts >= min_count].sort_values(ascending=False).head(max_lengths).index.tolist()
        lengths = sorted([int(x) for x in keep])
    data = [d.loc[d["target_seq_len"] == L, metric].values for L in lengths]
    plt.figure(figsize=(max(8, len(lengths) * 0.5), 5))
    plt.boxplot(data, tick_labels=[str(L) for L in lengths], showfliers=False)
    plt.title(f"{metric} by fragment length")
    plt.xlabel("Fragment length")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare actual contact-order results against random controls using density-normalized overlay histograms, after removing duplicated identical actual intervals."
    )
    p.add_argument("--actual-csv", required=True, help="Path to contact_order_results.csv")
    p.add_argument("--random-csv", required=True, help="Path to contact_order_random_results.csv")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--complete-mapping-only", action="store_true",
                   help="Keep only rows where target_seq_len = mapped_pdb_residue_count = fragment_size_mapped")
    p.add_argument("--length", type=int, default=None, help="Optional: restrict to one fragment length")
    p.add_argument("--chunksize", type=int, default=1_000_000, help="Chunk size for streaming random CSV")
    p.add_argument("--bins", type=int, default=40, help="Number of bins for histograms")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    actual_raw = pd.read_csv(args.actual_csv, low_memory=False)
    actual_df, removed_duplicates_df, filter_stats = filter_actual(
        actual_raw,
        complete_mapping_only=args.complete_mapping_only,
        length=args.length,
    )

    actual_df.to_csv(outdir / "actual_filtered_rows.csv", index=False)
    removed_duplicates_df.to_csv(outdir / "removed_identical_interval_duplicates.csv", index=False)

    if actual_df.empty:
        msg = textwrap.dedent(
            """
            No actual rows remained after filtering.
            Check your filter settings and input file.
            """
        ).strip()
        (outdir / "summary.txt").write_text(msg, encoding="utf-8")
        return

    actual_df["_key"] = list(zip(
        actual_df["global_row_index"],
        actual_df["pfam_id"].astype(str),
        actual_df["pfam_side"].astype(str),
        actual_df["segment_rank"],
    ))
    actual_keys = set(actual_df["_key"])

    # Incremental aggregation of random results per remaining actual key.
    per_key: Dict[Tuple, Dict[str, object]] = {}
    random_rfco_global = RunningStats()
    random_lr_global = RunningStats()

    usecols = RANDOM_NEEDED
    dtype_map = {
        "global_row_index": "int32",
        "pfam_id": "string",
        "pfam_side": "string",
        "segment_rank": "int16",
        "random_rfco": "float32",
        "random_lr_contact_fraction": "float32",
        "random_n_fragment_contacts": "float32",
    }

    for chunk in pd.read_csv(args.random_csv, chunksize=args.chunksize, usecols=usecols, dtype=dtype_map, low_memory=False):
        chunk = chunk.dropna(subset=["global_row_index", "segment_rank"])
        chunk["pfam_id"] = chunk["pfam_id"].astype(str)
        chunk["pfam_side"] = chunk["pfam_side"].astype(str)

        chunk["_key"] = list(zip(
            chunk["global_row_index"].astype(int),
            chunk["pfam_id"],
            chunk["pfam_side"],
            chunk["segment_rank"].astype(int),
        ))
        chunk = chunk[chunk["_key"].isin(actual_keys)]
        if chunk.empty:
            continue

        rf = pd.to_numeric(chunk["random_rfco"], errors="coerce").dropna().to_numpy(dtype=float)
        lr = pd.to_numeric(chunk["random_lr_contact_fraction"], errors="coerce").dropna().to_numpy(dtype=float)
        random_rfco_global.update(rf)
        random_lr_global.update(lr)

        grouped = chunk.groupby("_key", sort=False)
        for key, g in grouped:
            if key not in per_key:
                per_key[key] = {
                    "rfco_sum": 0.0,
                    "rfco_sumsq": 0.0,
                    "rfco_vals": [],
                    "lr_sum": 0.0,
                    "lr_sumsq": 0.0,
                    "lr_vals": [],
                    "nfc_sum": 0.0,
                    "nfc_sumsq": 0.0,
                    "nfc_vals": [],
                    "n": 0,
                }
            state = per_key[key]
            rfco_vals = pd.to_numeric(g["random_rfco"], errors="coerce").dropna().to_numpy(dtype=float)
            lr_vals = pd.to_numeric(g["random_lr_contact_fraction"], errors="coerce").dropna().to_numpy(dtype=float)
            nfc_vals = pd.to_numeric(g["random_n_fragment_contacts"], errors="coerce").dropna().to_numpy(dtype=float)

            if rfco_vals.size:
                state["rfco_sum"] += float(rfco_vals.sum())
                state["rfco_sumsq"] += float(np.square(rfco_vals).sum())
                state["rfco_vals"].append(rfco_vals)
            if lr_vals.size:
                state["lr_sum"] += float(lr_vals.sum())
                state["lr_sumsq"] += float(np.square(lr_vals).sum())
                state["lr_vals"].append(lr_vals)
            if nfc_vals.size:
                state["nfc_sum"] += float(nfc_vals.sum())
                state["nfc_sumsq"] += float(np.square(nfc_vals).sum())
                state["nfc_vals"].append(nfc_vals)
            state["n"] += int(len(g))

    rows = []
    for _, r in actual_df.iterrows():
        key = r["_key"]
        state = per_key.get(key)
        if not state or state["n"] == 0:
            continue
        n = int(state["n"])
        rfco_vals = np.concatenate(state["rfco_vals"]) if state["rfco_vals"] else np.array([], dtype=float)
        lr_vals = np.concatenate(state["lr_vals"]) if state["lr_vals"] else np.array([], dtype=float)
        nfc_vals = np.concatenate(state["nfc_vals"]) if state["nfc_vals"] else np.array([], dtype=float)

        rfco_mean = state["rfco_sum"] / n if n else np.nan
        lr_mean = state["lr_sum"] / n if n else np.nan
        nfc_mean = state["nfc_sum"] / n if n else np.nan
        rfco_std = np.sqrt(max(state["rfco_sumsq"] / n - rfco_mean ** 2, 0.0)) if n else np.nan
        lr_std = np.sqrt(max(state["lr_sumsq"] / n - lr_mean ** 2, 0.0)) if n else np.nan
        nfc_std = np.sqrt(max(state["nfc_sumsq"] / n - nfc_mean ** 2, 0.0)) if n else np.nan

        actual_rfco = float(r["rfco"])
        actual_lr = float(r["lr_contact_fraction"])
        actual_nfc = float(r["n_fragment_contacts"])

        rows.append({
            **{c: r[c] for c in actual_df.columns if c != "_key"},
            "random_n": n,
            "random_rfco_mean": rfco_mean,
            "random_rfco_std": rfco_std,
            "random_rfco_p05": np.nan if rfco_vals.size == 0 else float(np.quantile(rfco_vals, 0.05)),
            "random_rfco_p25": np.nan if rfco_vals.size == 0 else float(np.quantile(rfco_vals, 0.25)),
            "random_rfco_p50": np.nan if rfco_vals.size == 0 else float(np.quantile(rfco_vals, 0.50)),
            "random_rfco_p75": np.nan if rfco_vals.size == 0 else float(np.quantile(rfco_vals, 0.75)),
            "random_rfco_p95": np.nan if rfco_vals.size == 0 else float(np.quantile(rfco_vals, 0.95)),
            "random_lr_mean": lr_mean,
            "random_lr_std": lr_std,
            "random_lr_p50": np.nan if lr_vals.size == 0 else float(np.quantile(lr_vals, 0.50)),
            "random_nfc_mean": nfc_mean,
            "random_nfc_std": nfc_std,
            "rfco_delta_vs_random_mean": actual_rfco - rfco_mean,
            "lr_delta_vs_random_mean": actual_lr - lr_mean,
            "nfc_delta_vs_random_mean": actual_nfc - nfc_mean,
            "rfco_empirical_p_le": np.nan if rfco_vals.size == 0 else float(np.mean(rfco_vals <= actual_rfco)),
            "lr_empirical_p_le": np.nan if lr_vals.size == 0 else float(np.mean(lr_vals <= actual_lr)),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(outdir / "actual_vs_random_summary.csv", index=False)

    lines = []
    lines.append("Actual vs Random comparison summary")
    lines.append("=" * 40)
    lines.append(f"actual_rows_raw_input = {filter_stats['rows_raw_input']}")
    lines.append(f"actual_rows_after_ok = {filter_stats['rows_after_ok']}")
    lines.append(f"actual_rows_after_complete_mapping = {filter_stats['rows_after_complete_mapping']}")
    lines.append(f"actual_rows_after_length_filter = {filter_stats['rows_after_length_filter']}")
    lines.append(f"actual_rows_after_identical_interval_dedup = {filter_stats['rows_after_interval_dedup']}")
    lines.append(f"identical_interval_duplicates_removed = {filter_stats['interval_duplicates_removed']}")
    lines.append(f"identical_interval_dedup_cols = {filter_stats['dedup_cols']}")
    lines.append("")
    lines.append(f"actual_rows_with_random = {len(summary_df)}")
    lines.append(f"random_rfco_global_n = {random_rfco_global.n}")
    lines.append(f"random_rfco_global_mean = {random_rfco_global.mean():.6f}")
    lines.append(f"random_rfco_global_std = {random_rfco_global.std():.6f}")
    lines.append(f"random_lr_global_n = {random_lr_global.n}")
    lines.append(f"random_lr_global_mean = {random_lr_global.mean():.6f}")
    lines.append(f"random_lr_global_std = {random_lr_global.std():.6f}")

    if not summary_df.empty:
        for col in [
            "rfco",
            "random_rfco_mean",
            "rfco_delta_vs_random_mean",
            "lr_contact_fraction",
            "random_lr_mean",
            "lr_delta_vs_random_mean",
        ]:
            s = pd.to_numeric(summary_df[col], errors="coerce").dropna()
            if not s.empty:
                lines.append("")
                lines.append(f"{col}:")
                lines.append(f"  n={len(s)}")
                lines.append(f"  mean={s.mean():.6f}")
                lines.append(f"  median={s.median():.6f}")
                lines.append(f"  std={s.std():.6f}")
                lines.append(f"  min={s.min():.6f}")
                lines.append(f"  max={s.max():.6f}")

    (outdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    if summary_df.empty:
        return

    plot_hist(summary_df["rfco"], "Actual rFCO distribution", "rFCO",
              outdir / "actual_rfco_hist.png", bins=args.bins)
    plot_hist(summary_df["lr_contact_fraction"], "Actual long-range fraction distribution",
              "Long-range contact fraction", outdir / "actual_lr_hist.png", bins=args.bins)

    plot_overlay_hist_density(
        summary_df["rfco"],
        summary_df["random_rfco_mean"],
        "Actual vs random-mean rFCO (density)",
        "rFCO",
        outdir / "actual_vs_random_rfco_overlay_density.png",
        bins=args.bins
    )
    plot_overlay_hist_density(
        summary_df["lr_contact_fraction"],
        summary_df["random_lr_mean"],
        "Actual vs random-mean long-range fraction (density)",
        "Long-range contact fraction",
        outdir / "actual_vs_random_lr_overlay_density.png",
        bins=args.bins
    )

    plot_scatter(
        summary_df["random_rfco_mean"],
        summary_df["rfco"],
        "Actual rFCO vs random mean rFCO",
        "Random mean rFCO",
        "Actual rFCO",
        outdir / "actual_vs_random_mean_rfco_scatter.png"
    )
    plot_scatter(
        summary_df["random_lr_mean"],
        summary_df["lr_contact_fraction"],
        "Actual LR fraction vs random mean LR fraction",
        "Random mean LR fraction",
        "Actual LR fraction",
        outdir / "actual_vs_random_mean_lr_scatter.png"
    )

    plot_hist(
        summary_df["rfco_delta_vs_random_mean"],
        "Actual rFCO - random mean rFCO",
        "Delta rFCO",
        outdir / "rfco_delta_vs_random_mean_hist.png",
        bins=args.bins
    )
    plot_hist(
        summary_df["lr_delta_vs_random_mean"],
        "Actual LR fraction - random mean LR fraction",
        "Delta long-range fraction",
        outdir / "lr_delta_vs_random_mean_hist.png",
        bins=args.bins
    )
    plot_hist(
        summary_df["rfco_empirical_p_le"],
        "Empirical P(random rFCO <= actual rFCO)",
        "Empirical p",
        outdir / "rfco_empirical_p_le_hist.png",
        bins=args.bins
    )
    plot_hist(
        summary_df["lr_empirical_p_le"],
        "Empirical P(random LR <= actual LR)",
        "Empirical p",
        outdir / "lr_empirical_p_le_hist.png",
        bins=args.bins
    )

    if "target_seq_len" in summary_df.columns:
        plot_box_by_length(summary_df, "rfco", outdir / "actual_rfco_by_length_boxplot.png")
        plot_box_by_length(summary_df, "rfco_delta_vs_random_mean", outdir / "rfco_delta_by_length_boxplot.png")
        plot_box_by_length(summary_df, "lr_contact_fraction", outdir / "actual_lr_by_length_boxplot.png")
        plot_box_by_length(summary_df, "lr_delta_vs_random_mean", outdir / "lr_delta_by_length_boxplot.png")


if __name__ == "__main__":
    main()