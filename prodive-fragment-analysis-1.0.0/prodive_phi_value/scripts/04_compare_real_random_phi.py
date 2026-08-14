#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare mapped fragments with matched random fragments at phi-value sites.

Random fragments preserve the case, UniProt sequence, Pfam-mapped interval,
and fragment length. The default statistic is the density of unique positions
with phi >= 0.5.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Default analysis settings. Command-line arguments override these.
# =========================================================

DEFAULT_RANDOM_N = 10000
DEFAULT_RANDOM_SEED = 20260709
DEFAULT_HIGH_PHI_THRESHOLD = 0.5
DEFAULT_MAIN_METRIC = "high_phi_position_density"
DEFAULT_GROUP_LEVEL = "series"  # series or series_condition

PREFER_ALL_MAPPED_INTERVALS = True
DROP_EXACT_DUPLICATE_ROWS = True
SAVE_RANDOM_FRAGMENT_LEVEL_DEFAULT = False

METRIC_COLUMNS = [
    "any_phi_point",
    "any_high_phi_point",
    "phi_point_count",
    "high_phi_point_count",
    "phi_position_count",
    "high_phi_position_count",
    "all_phi_point_density",
    "high_phi_point_density",
    "all_phi_position_density",
    "high_phi_position_density",
    "phi_weighted_point_density_raw",
    "phi_weighted_point_density_clipped_0_1",
    "phi_weighted_position_mean_density_raw",
    "phi_weighted_position_mean_density_clipped_0_1",
]


# =========================================================
# Helper functions
# =========================================================

def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_pdb(x) -> str:
    return norm_str(x).upper()


def norm_pfam(x) -> str:
    return norm_str(x).upper()


def norm_uniprot(x) -> str:
    return norm_str(x).split(".")[0].upper()


def ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def prepare_input_root(input_path: Path, out_dir: Path) -> Path:
    """Return an extracted directory if input_path is a zip; otherwise return input_path."""
    input_path = input_path.expanduser().resolve()
    if input_path.is_dir():
        return input_path
    if not input_path.is_file():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.suffix.lower() != ".zip":
        raise ValueError(f"Input must be a directory or .zip file: {input_path}")

    extract_dir = out_dir / "_extracted_input"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def find_mapped_interval_files(root: Path) -> List[Path]:
    """
    Find mapped interval files.

    Server-friendly behavior:
    1) If the input directory itself contains ALL_mapped_intervals.csv, use it.
    2) Otherwise, search recursively for ALL_mapped_intervals.csv and prefer it.
    3) If no ALL file exists, use all case-level *_mapped_intervals.csv files.

    """
    root = root.resolve()

    direct_all = root / "ALL_mapped_intervals.csv"
    if PREFER_ALL_MAPPED_INTERVALS and direct_all.is_file():
        return [direct_all]

    recursive_all = sorted(root.rglob("ALL_mapped_intervals.csv"))
    if PREFER_ALL_MAPPED_INTERVALS and recursive_all:
        return [recursive_all[0]]

    files = [p for p in root.rglob("*_mapped_intervals.csv") if p.name != "ALL_mapped_intervals.csv"]
    if not files:
        raise FileNotFoundError(f"No *_mapped_intervals.csv or ALL_mapped_intervals.csv files found under {root}")
    return sorted(files)


def load_phi_long(phi_csv: Path) -> pd.DataFrame:
    phi = pd.read_csv(phi_csv)
    ensure_columns(
        phi,
        ["case", "pdb", "pfam", "uniprot", "uniprot_pos", "series_label", "phi_value"],
        "phi long CSV",
    )

    phi = phi.copy()
    phi["case"] = phi["case"].map(norm_str)
    phi["pdb"] = phi["pdb"].map(norm_pdb)
    phi["pfam"] = phi["pfam"].map(norm_pfam)
    phi["uniprot"] = phi["uniprot"].map(norm_uniprot)
    phi["uniprot_pos"] = pd.to_numeric(phi["uniprot_pos"], errors="coerce")
    phi["phi_value"] = pd.to_numeric(phi["phi_value"], errors="coerce")

    if "construct_or_condition" not in phi.columns:
        phi["construct_or_condition"] = ""
    phi["construct_or_condition"] = phi["construct_or_condition"].map(norm_str)
    phi["series_label"] = phi["series_label"].map(norm_str)
    if "mutation" not in phi.columns:
        phi["mutation"] = ""
    phi["mutation"] = phi["mutation"].astype(str)

    phi = phi.dropna(subset=["uniprot_pos", "phi_value"]).copy()
    phi["uniprot_pos"] = phi["uniprot_pos"].astype(int)
    phi = phi.reset_index(drop=True)
    return phi


def make_case_key_table(phi: pd.DataFrame) -> pd.DataFrame:
    keys = phi[["case", "pdb", "pfam", "uniprot"]].drop_duplicates().copy()
    return keys.sort_values(["case", "pdb", "pfam", "uniprot"]).reset_index(drop=True)


def load_mapped_intervals(root: Path, phi: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    files = find_mapped_interval_files(root)
    frames = []
    for p in files:
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    required = [
        "pdb_id", "pfam_id", "seed_ac_base", "seed_start", "seed_end",
        "Target_Seq_Start", "Target_Seq_End",
    ]
    ensure_columns(raw, required, "mapped intervals")

    if "Map_Status" in raw.columns:
        before = len(raw)
        raw = raw[raw["Map_Status"].astype(str).str.upper().eq("OK")].copy()
        warnings.append(f"Map_Status filter: kept {len(raw)} / {before} rows with Map_Status == OK.")

    raw["pdb_norm"] = raw["pdb_id"].map(norm_pdb)
    raw["pfam_norm"] = raw["pfam_id"].map(norm_pfam)
    raw["uniprot_norm"] = raw["seed_ac_base"].map(norm_uniprot)

    key_table = make_case_key_table(phi)
    key_to_case = {
        (r.pdb, r.pfam, r.uniprot): r.case
        for r in key_table.itertuples(index=False)
    }

    raw["case"] = [
        key_to_case.get((pdb, pfam, uni), "")
        for pdb, pfam, uni in zip(raw["pdb_norm"], raw["pfam_norm"], raw["uniprot_norm"])
    ]
    filtered = raw[raw["case"] != ""].copy()

    if filtered.empty:
        raise ValueError(
            "No mapped interval rows matched the case/pdb/pfam/uniprot keys in the phi CSV. "
            "Check whether phi CSV and mapped interval files use the same identifiers."
        )

    # Remove only exact duplicate analytical rows. source_csv is excluded so that the same row
    # duplicated in both all_data and case-specific folders is removed.
    if DROP_EXACT_DUPLICATE_ROWS:
        before = len(filtered)
        dup_cols = [c for c in filtered.columns if c != "source_csv"]
        filtered = filtered.drop_duplicates(subset=dup_cols).copy()
        warnings.append(f"Exact duplicate removal: kept {len(filtered)} / {before} mapped rows.")

    for col in ["seed_start", "seed_end", "Target_Seq_Start", "Target_Seq_End"]:
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce")
    filtered = filtered.dropna(subset=["seed_start", "seed_end", "Target_Seq_Start", "Target_Seq_End"]).copy()

    filtered["seed_start"] = filtered["seed_start"].astype(int)
    filtered["seed_end"] = filtered["seed_end"].astype(int)
    filtered["real_start"] = filtered["Target_Seq_Start"].astype(int)
    filtered["real_end"] = filtered["Target_Seq_End"].astype(int)

    bad_order = filtered["real_end"] < filtered["real_start"]
    if bad_order.any():
        raise ValueError(f"Found {bad_order.sum()} mapped intervals with Target_Seq_End < Target_Seq_Start.")

    filtered["real_length"] = filtered["real_end"] - filtered["real_start"] + 1
    filtered["bound_length_from_row"] = filtered["seed_end"] - filtered["seed_start"] + 1

    if "seed_seq_name" not in filtered.columns:
        filtered["seed_seq_name"] = ""

    # Stable ID. Do not use only coordinates because repeated intervals from different family pairs are meaningful.
    filtered = filtered.reset_index(drop=True)
    filtered["real_fragment_id"] = [f"real_{i:06d}" for i in range(len(filtered))]

    # Keep useful display columns if present.
    for c in ["File", "Main_HMM", "Sub_HMM", "Target_Pfam_Side", "Score", "Segment_Rank"]:
        if c not in filtered.columns:
            filtered[c] = np.nan

    return filtered


def derive_seed_bounds(real: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    bounds = (
        real.groupby(["case", "pdb_norm", "pfam_norm", "uniprot_norm"], as_index=False)
        .agg(
            seed_region_start=("seed_start", "min"),
            seed_region_end=("seed_end", "max"),
            n_real_rows=("real_fragment_id", "count"),
            real_start_min=("real_start", "min"),
            real_end_max=("real_end", "max"),
            max_real_length=("real_length", "max"),
            seed_seq_examples=("seed_seq_name", lambda s: ";".join(map(str, pd.unique(s))) if "seed_seq_name" in real.columns else ""),
        )
        .rename(columns={"pdb_norm": "pdb", "pfam_norm": "pfam", "uniprot_norm": "uniprot"})
    )
    bounds["seed_region_length"] = bounds["seed_region_end"] - bounds["seed_region_start"] + 1
    bounds["bound_source"] = "min(seed_start) to max(seed_end) from mapped_intervals"

    too_long = bounds["max_real_length"] > bounds["seed_region_length"]
    if too_long.any():
        raise ValueError(
            "At least one real fragment is longer than the derived Pfam seed region. "
            "This should not happen if seed_start/seed_end are correct. Problem rows:\n"
            + bounds.loc[too_long].to_string(index=False)
        )

    outside = bounds[(bounds["real_start_min"] < bounds["seed_region_start"]) | (bounds["real_end_max"] > bounds["seed_region_end"])]
    if not outside.empty:
        warnings.append(
            "Some real fragments extend outside the seed_start/seed_end region. "
            "The random background still uses seed_start/seed_end. Problem cases:\n"
            + outside.to_string(index=False)
        )

    return bounds.sort_values(["case", "uniprot"]).reset_index(drop=True)


def add_group_columns(phi: pd.DataFrame, group_level: str) -> pd.DataFrame:
    phi = phi.copy()
    if group_level == "series":
        phi["analysis_group"] = phi["series_label"]
    elif group_level == "series_condition":
        phi["analysis_group"] = phi["series_label"] + " | " + phi["construct_or_condition"].fillna("").astype(str)
    else:
        raise ValueError("group_level must be 'series' or 'series_condition'.")
    return phi


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(p)
    pv = p[valid]
    if len(pv) == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    raw_q = ranked * n / np.arange(1, n + 1)
    raw_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    raw_q = np.minimum(raw_q, 1.0)
    q_valid = np.empty_like(raw_q)
    q_valid[order] = raw_q
    q[valid] = q_valid
    return q


# =========================================================
# Metric calculation
# =========================================================

def empty_metrics() -> Dict[str, float]:
    return {c: 0.0 for c in METRIC_COLUMNS}


def interval_metrics(
    start: int,
    end: int,
    phi_pos: np.ndarray,
    phi_values: np.ndarray,
    high_threshold: float,
) -> Dict[str, float]:
    length = end - start + 1
    if length <= 0:
        raise ValueError(f"Invalid interval: {start}-{end}")

    mask = (phi_pos >= start) & (phi_pos <= end)
    if not np.any(mask):
        return empty_metrics()

    pos = phi_pos[mask]
    val = phi_values[mask]
    high_mask = val >= high_threshold

    point_count = int(len(val))
    high_point_count = int(np.sum(high_mask))
    unique_positions = np.unique(pos)
    position_count = int(len(unique_positions))
    high_position_count = int(len(np.unique(pos[high_mask]))) if np.any(high_mask) else 0

    clipped = np.clip(val, 0.0, 1.0)

    # Position-mean weighted score: within the same residue position, average multiple
    # values first, then sum over residue positions. This avoids collapsing series into max,
    # but also avoids unlimited duplicate inflation at the same residue.
    pos_mean_raw_sum = 0.0
    pos_mean_clipped_sum = 0.0
    for upos in unique_positions:
        vv = val[pos == upos]
        pos_mean_raw_sum += float(np.mean(vv))
        pos_mean_clipped_sum += float(np.mean(np.clip(vv, 0.0, 1.0)))

    return {
        "any_phi_point": float(point_count > 0),
        "any_high_phi_point": float(high_point_count > 0),
        "phi_point_count": float(point_count),
        "high_phi_point_count": float(high_point_count),
        "phi_position_count": float(position_count),
        "high_phi_position_count": float(high_position_count),
        "all_phi_point_density": float(point_count / length),
        "high_phi_point_density": float(high_point_count / length),
        "all_phi_position_density": float(position_count / length),
        "high_phi_position_density": float(high_position_count / length),
        "phi_weighted_point_density_raw": float(np.sum(val) / length),
        "phi_weighted_point_density_clipped_0_1": float(np.sum(clipped) / length),
        "phi_weighted_position_mean_density_raw": float(pos_mean_raw_sum / length),
        "phi_weighted_position_mean_density_clipped_0_1": float(pos_mean_clipped_sum / length),
    }


def metrics_to_prefixed_row(metrics: Dict[str, float], prefix: str = "") -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def precompute_metrics_for_possible_starts(
    bound_start: int,
    bound_end: int,
    length: int,
    phi_pos: np.ndarray,
    phi_values: np.ndarray,
    high_threshold: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Precompute interval metrics for all possible random starts for a fixed length."""
    max_start = bound_end - length + 1
    if max_start < bound_start:
        raise ValueError(f"Random region too short: bound={bound_start}-{bound_end}, length={length}")
    starts = np.arange(bound_start, max_start + 1, dtype=int)
    metric_arrays = {c: np.zeros(len(starts), dtype=float) for c in METRIC_COLUMNS}
    for i, s in enumerate(starts):
        m = interval_metrics(int(s), int(s + length - 1), phi_pos, phi_values, high_threshold)
        for c in METRIC_COLUMNS:
            metric_arrays[c][i] = m[c]
    return starts, metric_arrays



# =========================================================
# Main analysis
# =========================================================

def get_bound_for_row(row: pd.Series, bounds: pd.DataFrame) -> Tuple[int, int]:
    sub = bounds[
        (bounds["case"] == row["case"]) &
        (bounds["pdb"] == row["pdb_norm"]) &
        (bounds["pfam"] == row["pfam_norm"]) &
        (bounds["uniprot"] == row["uniprot_norm"])
    ]
    if len(sub) != 1:
        raise ValueError(
            f"Cannot find unique seed bound for {row['case']} / {row['pdb_norm']} / "
            f"{row['pfam_norm']} / {row['uniprot_norm']}. Found {len(sub)} rows."
        )
    return int(sub.iloc[0]["seed_region_start"]), int(sub.iloc[0]["seed_region_end"])


def build_analysis_groups(phi: pd.DataFrame) -> pd.DataFrame:
    groups = (
        phi[["case", "pdb", "pfam", "uniprot", "analysis_group"]]
        .drop_duplicates()
        .sort_values(["case", "analysis_group"])
        .reset_index(drop=True)
    )
    groups["analysis_group_id"] = [f"group_{i:03d}" for i in range(len(groups))]
    return groups


def run_analysis(
    real: pd.DataFrame,
    phi: pd.DataFrame,
    bounds: pd.DataFrame,
    out_dir: Path,
    random_n: int,
    random_seed: int,
    high_phi_threshold: float,
    main_metric: str,
    save_random_fragment_level: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if main_metric not in METRIC_COLUMNS:
        raise ValueError(f"Unknown main metric: {main_metric}. Allowed: {METRIC_COLUMNS}")

    rng = np.random.default_rng(random_seed)
    groups = build_analysis_groups(phi)

    # Attach the seed-derived random bounds to each real fragment once.
    b2 = bounds[["case", "pdb", "pfam", "uniprot", "seed_region_start", "seed_region_end"]].copy()
    real2 = real.merge(
        b2,
        left_on=["case", "pdb_norm", "pfam_norm", "uniprot_norm"],
        right_on=["case", "pdb", "pfam", "uniprot"],
        how="left",
        suffixes=("", "_bound"),
    )
    if real2["seed_region_start"].isna().any():
        bad = real2[real2["seed_region_start"].isna()][["case", "pdb_norm", "pfam_norm", "uniprot_norm"]].drop_duplicates()
        raise ValueError("Some real fragments have no seed-derived random bounds:\n" + bad.to_string(index=False))
    real2["seed_region_start"] = real2["seed_region_start"].astype(int)
    real2["seed_region_end"] = real2["seed_region_end"].astype(int)

    real_rows = []
    random_iter_frames = []
    random_frag_rows = []

    for g in groups.itertuples(index=False):
        group_phi = phi[
            (phi["case"] == g.case) &
            (phi["pdb"] == g.pdb) &
            (phi["pfam"] == g.pfam) &
            (phi["uniprot"] == g.uniprot) &
            (phi["analysis_group"] == g.analysis_group)
        ].copy()

        phi_pos = group_phi["uniprot_pos"].to_numpy(dtype=int)
        phi_values = group_phi["phi_value"].to_numpy(dtype=float)

        group_real = real2[
            (real2["case"] == g.case) &
            (real2["pdb_norm"] == g.pdb) &
            (real2["pfam_norm"] == g.pfam) &
            (real2["uniprot_norm"] == g.uniprot)
        ].copy()

        if group_real.empty:
            continue

        # Real metrics.
        for r in group_real.itertuples(index=False):
            m = interval_metrics(
                int(r.real_start), int(r.real_end), phi_pos, phi_values, high_phi_threshold
            )
            real_rows.append({
                "analysis_group_id": g.analysis_group_id,
                "case": g.case,
                "pdb": g.pdb,
                "pfam": g.pfam,
                "uniprot": g.uniprot,
                "analysis_group": g.analysis_group,
                "real_fragment_id": r.real_fragment_id,
                "real_start": int(r.real_start),
                "real_end": int(r.real_end),
                "real_length": int(r.real_length),
                "seed_start": int(r.seed_start),
                "seed_end": int(r.seed_end),
                "random_bound_start": int(r.seed_region_start),
                "random_bound_end": int(r.seed_region_end),
                "File": r.File,
                "Main_HMM": r.Main_HMM,
                "Sub_HMM": r.Sub_HMM,
                "Target_Pfam_Side": r.Target_Pfam_Side,
                "Score": r.Score,
                "Segment_Rank": r.Segment_Rank,
                **m,
            })

        # Vectorized random iteration-level matched sampling.
        sums = {c: np.zeros(random_n, dtype=float) for c in METRIC_COLUMNS}
        n_matched = len(group_real)

        # Cache possible-start metrics for each unique (bound_start, bound_end, length).
        lookup_cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}

        for r in group_real.itertuples(index=False):
            b_start = int(r.seed_region_start)
            b_end = int(r.seed_region_end)
            length = int(r.real_length)
            key = (b_start, b_end, length)
            if key not in lookup_cache:
                lookup_cache[key] = precompute_metrics_for_possible_starts(
                    b_start, b_end, length, phi_pos, phi_values, high_phi_threshold
                )
            starts, metric_arrays = lookup_cache[key]
            sampled_idx = rng.integers(0, len(starts), size=random_n)
            sampled_starts = starts[sampled_idx]

            for c in METRIC_COLUMNS:
                sums[c] += metric_arrays[c][sampled_idx]

            if save_random_fragment_level:
                # This can be large: n_real_fragments * random_n * n_groups.
                for it, s0 in enumerate(sampled_starts):
                    row = {
                        "analysis_group_id": g.analysis_group_id,
                        "case": g.case,
                        "pdb": g.pdb,
                        "pfam": g.pfam,
                        "uniprot": g.uniprot,
                        "analysis_group": g.analysis_group,
                        "random_iter": it,
                        "matched_real_fragment_id": r.real_fragment_id,
                        "random_start": int(s0),
                        "random_end": int(s0 + length - 1),
                        "random_length": length,
                    }
                    for c in METRIC_COLUMNS:
                        row[c] = float(metric_arrays[c][sampled_idx[it]])
                    random_frag_rows.append(row)

        frame = pd.DataFrame({
            "analysis_group_id": g.analysis_group_id,
            "case": g.case,
            "pdb": g.pdb,
            "pfam": g.pfam,
            "uniprot": g.uniprot,
            "analysis_group": g.analysis_group,
            "random_iter": np.arange(random_n, dtype=int),
            "n_matched_fragments": n_matched,
            **{c: sums[c] / n_matched for c in METRIC_COLUMNS},
        })
        random_iter_frames.append(frame)

    real_metrics = pd.DataFrame(real_rows)
    random_iter_metrics = pd.concat(random_iter_frames, ignore_index=True) if random_iter_frames else pd.DataFrame()

    real_metrics.to_csv(out_dir / "real_fragment_metrics_by_series.csv", index=False)
    random_iter_metrics.to_csv(out_dir / "random_iteration_metrics_by_series.csv", index=False)

    if save_random_fragment_level:
        pd.DataFrame(random_frag_rows).to_csv(out_dir / "random_fragment_metrics_by_series.csv", index=False)

    summary_long = summarize_all_metrics(real_metrics, random_iter_metrics)
    summary_long.to_csv(out_dir / "summary_by_series_all_metrics_long.csv", index=False)

    main_summary = summary_long[summary_long["metric"] == main_metric].copy()
    main_summary["fdr_bh_q_right"] = bh_fdr(main_summary["empirical_p_right"].to_numpy())
    main_summary = main_summary.sort_values(["case", "analysis_group"]).reset_index(drop=True)
    main_summary.to_csv(out_dir / "summary_by_series_main_metric.csv", index=False)

    case_summary = summarize_case_level(summary_long, random_iter_metrics, main_metric)
    case_summary.to_csv(out_dir / "summary_by_case_main_metric.csv", index=False)

    return real_metrics, random_iter_metrics, main_summary


def summarize_all_metrics(real_metrics: pd.DataFrame, random_iter_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["analysis_group_id", "case", "pdb", "pfam", "uniprot", "analysis_group"]

    for group_key, real_sub in real_metrics.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, group_key))
        rand_sub = random_iter_metrics
        for c, v in key_dict.items():
            rand_sub = rand_sub[rand_sub[c] == v]

        for metric in METRIC_COLUMNS:
            real_values = real_sub[metric].to_numpy(dtype=float)
            rand_values = rand_sub[metric].to_numpy(dtype=float)
            obs = float(np.mean(real_values))
            p_right = float((1 + np.sum(rand_values >= obs)) / (len(rand_values) + 1))
            p_left = float((1 + np.sum(rand_values <= obs)) / (len(rand_values) + 1))

            rows.append({
                **key_dict,
                "metric": metric,
                "n_real_fragments": int(real_sub["real_fragment_id"].nunique()),
                "real_mean": obs,
                "real_median": float(np.median(real_values)),
                "random_mean": float(np.mean(rand_values)),
                "random_median": float(np.median(rand_values)),
                "random_q025": float(np.quantile(rand_values, 0.025)),
                "random_q975": float(np.quantile(rand_values, 0.975)),
                "real_minus_random_mean": float(obs - np.mean(rand_values)),
                "empirical_p_right": p_right,
                "empirical_p_left": p_left,
            })

    return pd.DataFrame(rows)


def summarize_case_level(
    summary_long: pd.DataFrame,
    random_iter_metrics: pd.DataFrame,
    main_metric: str,
) -> pd.DataFrame:
    """Case-level summary by equally averaging the series-level values within each case."""
    main = summary_long[summary_long["metric"] == main_metric].copy()
    rows = []

    for case, sub in main.groupby("case", dropna=False):
        real_case_mean = float(sub["real_mean"].mean())

        rand_case_iter = (
            random_iter_metrics[random_iter_metrics["case"] == case]
            .groupby("random_iter", as_index=False)[main_metric]
            .mean()
        )
        rv = rand_case_iter[main_metric].to_numpy(dtype=float)
        p_right = float((1 + np.sum(rv >= real_case_mean)) / (len(rv) + 1))

        rows.append({
            "case": case,
            "metric": main_metric,
            "n_analysis_groups": int(sub["analysis_group"].nunique()),
            "analysis_groups": ";".join(map(str, sub["analysis_group"].tolist())),
            "real_case_mean_equal_series_weight": real_case_mean,
            "random_case_mean": float(np.mean(rv)),
            "random_case_median": float(np.median(rv)),
            "random_case_q025": float(np.quantile(rv, 0.025)),
            "random_case_q975": float(np.quantile(rv, 0.975)),
            "real_minus_random_mean": float(real_case_mean - np.mean(rv)),
            "empirical_p_right": p_right,
        })

    out = pd.DataFrame(rows)
    out["fdr_bh_q_right"] = bh_fdr(out["empirical_p_right"].to_numpy())
    return out.sort_values("case").reset_index(drop=True)


# =========================================================
# Plotting
# =========================================================

def plot_series_null_boxplot(
    random_iter_metrics: pd.DataFrame,
    main_summary: pd.DataFrame,
    out_png: Path,
    metric: str,
) -> None:
    labels = []
    data = []
    real_values = []

    for row in main_summary.itertuples(index=False):
        labels.append(f"{row.case}\n{row.analysis_group}")
        sub = random_iter_metrics[
            (random_iter_metrics["case"] == row.case) &
            (random_iter_metrics["analysis_group"] == row.analysis_group)
        ]
        data.append(sub[metric].to_numpy(dtype=float))
        real_values.append(float(row.real_mean))

    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(labels)), 5.2), dpi=300)
    ax.boxplot(data, showfliers=False)
    ax.scatter(np.arange(1, len(labels) + 1), real_values, marker="D", s=35, label="Real")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(metric)
    ax.set_title("Real fragments against matched random fragments by phi series")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_case_bar(case_summary: pd.DataFrame, out_png: Path) -> None:
    labels = case_summary["case"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    real = case_summary["real_case_mean_equal_series_weight"].to_numpy(dtype=float)
    rand = case_summary["random_case_mean"].to_numpy(dtype=float)
    lower = rand - case_summary["random_case_q025"].to_numpy(dtype=float)
    upper = case_summary["random_case_q975"].to_numpy(dtype=float) - rand

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.bar(x - width / 2, real, width, label="Real fragments")
    ax.bar(x + width / 2, rand, width, label="Random fragments")
    ax.errorbar(x + width / 2, rand, yerr=[lower, upper], fmt="none", capsize=4, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(case_summary["metric"].iloc[0])
    ax.set_title("Case-level phi-site overlap: real vs random")
    ax.legend(frameon=False)

    for i, row in enumerate(case_summary.itertuples(index=False)):
        top = max(float(row.real_case_mean_equal_series_weight), float(row.random_case_q975))
        y = top * 1.08 if top > 0 else 0.02
        ax.text(i, y, f"p={row.empirical_p_right:.3g}", ha="center", va="bottom", fontsize=8)

    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def compute_real_coverage(real_sub: pd.DataFrame, start: int, end: int) -> np.ndarray:
    positions = np.arange(start, end + 1)
    cov = np.zeros(len(positions), dtype=float)
    for r in real_sub.itertuples(index=False):
        s = max(int(r.real_start), start)
        e = min(int(r.real_end), end)
        if e >= s:
            cov[(s - start):(e - start + 1)] += 1
    return cov


def plot_site_maps(
    real: pd.DataFrame,
    phi: pd.DataFrame,
    bounds: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for b in bounds.itertuples(index=False):
        case = b.case
        pdb = b.pdb
        pfam = b.pfam
        uniprot = b.uniprot
        start = int(b.seed_region_start)
        end = int(b.seed_region_end)
        positions = np.arange(start, end + 1)

        real_sub = real[
            (real["case"] == case) &
            (real["pdb_norm"] == pdb) &
            (real["pfam_norm"] == pfam) &
            (real["uniprot_norm"] == uniprot)
        ].copy()
        phi_sub = phi[
            (phi["case"] == case) &
            (phi["pdb"] == pdb) &
            (phi["pfam"] == pfam) &
            (phi["uniprot"] == uniprot)
        ].copy()

        if real_sub.empty or phi_sub.empty:
            continue

        real_cov = compute_real_coverage(real_sub, start, end)

        fig, ax1 = plt.subplots(figsize=(12, 4.8), dpi=300)
        ax1.bar(positions, real_cov, width=1.0, alpha=0.45, label="Fragment coverage")
        ax1.set_xlabel(f"UniProt residue position in {uniprot}")
        ax1.set_ylabel("Fragment coverage count")
        ax1.set_title(f"{case}: Pfam seed range, fragment coverage, and phi values")

        ax2 = ax1.twinx()
        for label, sub in phi_sub.groupby("series_label", dropna=False):
            ax2.scatter(sub["uniprot_pos"], sub["phi_value"], s=28, label=str(label))
            for rr in sub.itertuples(index=False):
                # Label only points inside/near the x-axis range to avoid excessive text outside the domain.
                if start <= int(rr.uniprot_pos) <= end:
                    ax2.text(int(rr.uniprot_pos), float(rr.phi_value), str(int(rr.uniprot_pos)), fontsize=6,
                             ha="left", va="bottom")
        ax2.set_ylabel("phi value")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(
            h1 + h2,
            l1 + l2,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )

        safe = f"{case}_{pdb}_{pfam}_{uniprot}".replace("/", "_").replace(" ", "_").replace("κ", "k")
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe}_site_map_seedrange.png", bbox_inches="tight")
        plt.close(fig)


def plot_all(real_metrics: pd.DataFrame, random_iter_metrics: pd.DataFrame, main_summary: pd.DataFrame,
             case_summary: pd.DataFrame, real: pd.DataFrame, phi: pd.DataFrame, bounds: pd.DataFrame,
             out_dir: Path, metric: str) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_series_null_boxplot(random_iter_metrics, main_summary, fig_dir / f"Fig1_series_null_boxplot_{metric}.png", metric)
    plot_case_bar(case_summary, fig_dir / f"Fig2_case_bar_{metric}.png")
    plot_site_maps(real, phi, bounds, fig_dir / "Fig3_site_maps_seedrange")


# =========================================================
# CLI and execution
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare real cross-family fragments with random same-sequence fragments for phi-value site overlap."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Input phi-value package directory or zip file.")
    parser.add_argument("--phi", type=Path, required=True,
                        help="Corrected long-format phi CSV, usually phi_sites_plot_preserved_long.csv.")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output directory.")
    parser.add_argument("--random-n", type=int, default=DEFAULT_RANDOM_N,
                        help="Number of random iterations.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help="Random seed.")
    parser.add_argument("--high-phi-threshold", type=float, default=DEFAULT_HIGH_PHI_THRESHOLD,
                        help="Threshold for high phi sites.")
    parser.add_argument("--main-metric", type=str, default=DEFAULT_MAIN_METRIC,
                        choices=METRIC_COLUMNS,
                        help="Main metric used in summary and plots.")
    parser.add_argument("--group-level", type=str, default=DEFAULT_GROUP_LEVEL,
                        choices=["series", "series_condition"],
                        help="Group phi values by series only, or by series plus construct/condition.")
    parser.add_argument("--save-random-fragment-level", action="store_true",
                        help="Save every random fragment metric. This can create a large CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.outdir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    input_root = prepare_input_root(args.input, out_dir)

    phi = load_phi_long(args.phi.expanduser().resolve())
    phi = add_group_columns(phi, args.group_level)

    real = load_mapped_intervals(input_root, phi, warnings)
    bounds = derive_seed_bounds(real, warnings)

    # Save prepared inputs for reproducibility.
    phi.to_csv(out_dir / "phi_sites_used_long.csv", index=False)
    real.to_csv(out_dir / "real_fragments_used_from_mapped_intervals.csv", index=False)
    bounds.to_csv(out_dir / "sequence_bounds_from_pfam_seed.csv", index=False)

    real_metrics, random_iter_metrics, main_summary = run_analysis(
        real=real,
        phi=phi,
        bounds=bounds,
        out_dir=out_dir,
        random_n=args.random_n,
        random_seed=args.seed,
        high_phi_threshold=args.high_phi_threshold,
        main_metric=args.main_metric,
        save_random_fragment_level=args.save_random_fragment_level,
    )

    case_summary = pd.read_csv(out_dir / "summary_by_case_main_metric.csv")
    plot_all(
        real_metrics=real_metrics,
        random_iter_metrics=random_iter_metrics,
        main_summary=main_summary,
        case_summary=case_summary,
        real=real,
        phi=phi,
        bounds=bounds,
        out_dir=out_dir,
        metric=args.main_metric,
    )

    # Write a compact run report.
    report = []
    report.append("Phi random overlap analysis completed.\n")
    report.append(f"Input root: {input_root}\n")
    report.append(f"Phi CSV: {args.phi.expanduser().resolve()}\n")
    report.append(f"Output directory: {out_dir}\n")
    report.append(f"Random iterations: {args.random_n}\n")
    report.append(f"Random seed: {args.seed}\n")
    report.append(f"Group level: {args.group_level}\n")
    report.append(f"Main metric: {args.main_metric}\n")
    report.append(f"High phi threshold: {args.high_phi_threshold}\n")
    report.append("\nSequence bounds from Pfam seed mapping:\n")
    report.append(bounds.to_string(index=False))
    report.append("\n\nMain summary by series:\n")
    report.append(main_summary.to_string(index=False))
    report.append("\n\nMain summary by case:\n")
    report.append(case_summary.to_string(index=False))
    if warnings:
        report.append("\n\nWarnings:\n")
        report.append("\n\n".join(warnings))

    (out_dir / "run_report.txt").write_text("".join(report), encoding="utf-8")
    (out_dir / "warnings.txt").write_text("\n\n".join(warnings), encoding="utf-8")

    print("Done.")
    print(f"Output directory: {out_dir}")
    print("\nSequence bounds:")
    print(bounds.to_string(index=False))
    print("\nMain summary by series:")
    print(main_summary.to_string(index=False))
    print("\nMain summary by case:")
    print(case_summary.to_string(index=False))
    if warnings:
        print("\nWarnings written to warnings.txt")


if __name__ == "__main__":
    main()
