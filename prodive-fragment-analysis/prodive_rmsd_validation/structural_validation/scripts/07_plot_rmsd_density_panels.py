#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw Fig. 2-style RMSD density panels from already-computed RMSD CSV files.

The script reads a dataset configuration CSV rather than hard-coding local paths.
Each dataset is drawn as an independent PNG. It supports the 0-30 panels used in
Fig. 2 and the 0-100 full-range panels used for supplementary HHsearch-boundary
visualization.

Required input CSV columns per RMSD table:
  - RMSD
  - Aligned_Atoms
  - one of: Coverage, Target_Len, HMM_Len, HMM_Len_A+HMM_Len_B

Dataset config columns:
  key,path,file_stem,xlim_min,xlim_max,ylim_min,ylim_max,show_rmsd_lines,
  show_core_box,show_band_percentages,show_broad_count,title
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from tqdm import tqdm


def str_to_bool(x: Any, default: bool = True) -> bool:
    if pd.isna(x):
        return default
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return default


def read_dataset_config(config_csv: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(config_csv)
    df.columns = [c.strip() for c in df.columns]
    required = ["key", "path", "file_stem", "xlim_min", "xlim_max", "ylim_min", "ylim_max"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset config missing columns: {missing}. Available: {list(df.columns)}")

    datasets: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        datasets.append({
            "key": str(row["key"]).strip(),
            "path": str(row["path"]).strip(),
            "file_stem": str(row["file_stem"]).strip(),
            "title": "" if "title" not in df.columns or pd.isna(row.get("title")) else str(row.get("title")),
            "xlim": (float(row["xlim_min"]), float(row["xlim_max"])),
            "ylim": (float(row["ylim_min"]), float(row["ylim_max"])),
            "show_rmsd_lines": str_to_bool(row.get("show_rmsd_lines", True), True),
            "show_core_box": str_to_bool(row.get("show_core_box", True), True),
            "show_band_percentages": str_to_bool(row.get("show_band_percentages", True), True),
            "show_broad_count": str_to_bool(row.get("show_broad_count", True), True),
        })
    return datasets


def load_and_filter_csv(path: str, coverage_threshold: float, rmsd_limit_for_filter: float, min_atoms_for_filter: int, len_min: int, len_max: int, rmsd_core_limit: float, rmsd_bands: List[Tuple[float, float]]) -> Dict[str, Any]:
    df_raw = pd.read_csv(path, low_memory=False)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    raw_count = len(df_raw)

    for col in ["Aligned_Atoms", "RMSD", "Coverage", "Target_Len", "HMM_Len", "HMM_Len_A", "HMM_Len_B"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    if "Coverage" in df_raw.columns and df_raw["Coverage"].notna().any():
        pass
    elif "Target_Len" in df_raw.columns and df_raw["Target_Len"].notna().any():
        df_raw["Coverage"] = df_raw["Aligned_Atoms"] / df_raw["Target_Len"]
    elif "HMM_Len" in df_raw.columns and df_raw["HMM_Len"].notna().any():
        df_raw["Target_Len"] = df_raw["HMM_Len"]
        df_raw["Coverage"] = df_raw["Aligned_Atoms"] / df_raw["Target_Len"]
    elif "HMM_Len_A" in df_raw.columns and "HMM_Len_B" in df_raw.columns:
        df_raw["Target_Len"] = df_raw[["HMM_Len_A", "HMM_Len_B"]].min(axis=1)
        df_raw["Coverage"] = df_raw["Aligned_Atoms"] / df_raw["Target_Len"]
    else:
        raise ValueError(f"Cannot compute Coverage for {path}. Need Coverage, Target_Len, HMM_Len, or HMM_Len_A/HMM_Len_B.")

    if "Target_Len" in df_raw.columns:
        df_raw.loc[df_raw["Target_Len"] <= 0, "Coverage"] = np.nan

    df = df_raw[
        (df_raw["Coverage"] >= coverage_threshold) &
        (df_raw["RMSD"] < rmsd_limit_for_filter) &
        (df_raw["Aligned_Atoms"] >= min_atoms_for_filter)
    ].copy()

    broad = df[df["RMSD"] < rmsd_core_limit]
    core = df[(df["RMSD"] < rmsd_core_limit) & (df["Aligned_Atoms"] >= len_min) & (df["Aligned_Atoms"] <= len_max)]

    denom = float(len(df)) if len(df) > 0 else 1.0
    band_stats = []
    for lo, hi in rmsd_bands:
        cnt = int(((df["RMSD"] >= lo) & (df["RMSD"] < hi)).sum())
        band_stats.append({"lo": lo, "hi": hi, "count": cnt, "pct": 100.0 * cnt / denom if len(df) > 0 else 0.0})

    return {
        "df": df,
        "raw_count": raw_count,
        "filtered_count": len(df),
        "broad_count": len(broad),
        "core_count": len(core),
        "band_stats": band_stats,
    }


def compute_density(x: np.ndarray, y: np.ndarray, batch_size: int = 2000) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], bool]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 5 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return x, y, None, False
    try:
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        z = np.empty(len(x), dtype=float)
        for i in range(0, len(x), batch_size):
            j = min(i + batch_size, len(x))
            z[i:j] = kde(xy[:, i:j])
        order = np.argsort(z)
        return x[order], y[order], z[order], True
    except Exception:
        return x, y, None, False


def maybe_sample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def process_one_dataset(item: Dict[str, Any], args_dict: Dict[str, Any]) -> Dict[str, Any]:
    bands = [(float(x.split("-")[0]), float(x.split("-")[1])) for x in args_dict["rmsd_bands"].split(",")]
    loaded = load_and_filter_csv(
        item["path"],
        coverage_threshold=args_dict["coverage_threshold"],
        rmsd_limit_for_filter=args_dict["rmsd_filter_max"],
        min_atoms_for_filter=args_dict["min_atoms_for_filter"],
        len_min=args_dict["len_min"],
        len_max=args_dict["len_max"],
        rmsd_core_limit=args_dict["rmsd_core_limit"],
        rmsd_bands=bands,
    )
    df_kde = maybe_sample(loaded["df"], args_dict["sample_n_per_file"], args_dict["sample_random_seed"])
    x = df_kde["Aligned_Atoms"].to_numpy(dtype=float)
    y = df_kde["RMSD"].to_numpy(dtype=float)
    x, y, z, has_density = compute_density(x, y, batch_size=args_dict["kde_batch_size"])
    return {**item, **{k: v for k, v in loaded.items() if k != "df"}, "x": x, "y": y, "z": z, "has_density": has_density}


def nice_colorbar_max(v: float) -> float:
    if v <= 0:
        return 1.0
    decimals = 1 if v < 10 else 0
    factor = 10 ** decimals
    return math.ceil(v * factor) / factor


def make_colorbar_ticks(vmax: float) -> List[float]:
    step = 0.1 if vmax <= 0.8 else 0.2 if vmax <= 1.6 else 0.5 if vmax <= 4 else 1.0
    ticks = np.arange(0, vmax + step * 0.5, step)
    ticks = np.round(ticks, 3)
    if len(ticks) == 0 or ticks[-1] < vmax:
        ticks = np.append(ticks, vmax)
    elif ticks[-1] > vmax:
        ticks[-1] = vmax
    return ticks.tolist()


def draw_one(result: Dict[str, Any], norm: Optional[Normalize], ticks: Optional[List[float]], args: argparse.Namespace) -> None:
    x, y, z = result["x"], result["y"], result["z"]
    xlim, ylim = result["xlim"], result["ylim"]
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    ax.set_title(result.get("title", ""), fontsize=args.title_fontsize, pad=6)

    visible = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    x_vis, y_vis = x[visible], y[visible]
    z_vis = z[visible] if z is not None and len(z) == len(x) else None

    if len(x_vis) == 0:
        ax.text(0.5, 0.5, "No data after filtering", transform=ax.transAxes, ha="center", va="center")
    elif result["has_density"] and norm is not None and z_vis is not None:
        sc = ax.scatter(x_vis, y_vis, c=z_vis, s=args.point_size, cmap=args.cmap, norm=norm, alpha=args.point_alpha, edgecolor="none")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.6%", pad=0.16)
        cb = fig.colorbar(sc, cax=cax, ticks=ticks)
        cb.set_label("Local density (KDE)", fontsize=10)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    else:
        ax.scatter(x_vis, y_vis, color="steelblue", s=args.point_size, alpha=args.point_alpha, edgecolor="none")

    if result["show_rmsd_lines"]:
        for yy in [args.rmsd_core_limit] + args.aux_rmsd_lines:
            ax.axhline(y=yy, color="red", linestyle="--", linewidth=1.5, zorder=1)

    if result["show_core_box"]:
        x0 = args.len_min - args.core_box_xpad
        x1 = args.len_max + args.core_box_xpad
        y0 = args.core_box_ymin
        y1 = args.rmsd_core_limit - args.core_box_ypad
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.8, edgecolor="red", facecolor="none", zorder=6, clip_on=False)
        ax.add_patch(rect)

    info = [f"raw = {result['raw_count']}", f"kept = {result['filtered_count']}"]
    if result.get("show_broad_count", True):
        info.append(f"RMSD < {args.rmsd_core_limit:g} A: {result['broad_count']}")
    info.extend([f"core(8-13): {result['core_count']}", f"Coverage >= {int(args.coverage_threshold * 100)}%"])
    ax.text(0.03, 0.97, "\n".join(info), transform=ax.transAxes, fontsize=args.info_fontsize, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="gray", alpha=0.88))

    if result["show_band_percentages"]:
        for band in result["band_stats"]:
            ax.text(0.8, 0.5 * (band["lo"] + band["hi"]), f"{band['pct']:.1f}%", fontsize=args.band_text_fontsize, ha="left", va="center", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75), zorder=7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Aligned CA Atoms (Count)", fontsize=args.axis_label_fontsize)
    ax.set_ylabel("C-alpha-RMSD (A)", fontsize=args.axis_label_fontsize)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle=":", alpha=0.5)
    out = Path(args.output_dir) / f"{result['file_stem']}.png"
    plt.tight_layout()
    fig.savefig(out, dpi=args.output_dpi)
    plt.close(fig)
    print(f"[OUT] {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Draw Fig. 2-style RMSD density panels from RMSD CSV files.")
    p.add_argument("--dataset-config", required=True, help="CSV specifying datasets and plot options")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--coverage-threshold", type=float, default=0.8)
    p.add_argument("--rmsd-core-limit", type=float, default=1.0)
    p.add_argument("--rmsd-filter-max", type=float, default=10.0)
    p.add_argument("--len-min", type=int, default=8)
    p.add_argument("--len-max", type=int, default=13)
    p.add_argument("--min-atoms-for-filter", type=int, default=4)
    p.add_argument("--rmsd-bands", default="0-1,1-2,2-3,3-4")
    p.add_argument("--sample-n-per-file", type=int, default=0, help="0 means no sampling")
    p.add_argument("--sample-random-seed", type=int, default=20260314)
    p.add_argument("--kde-batch-size", type=int, default=2000)
    p.add_argument("--workers", type=int, default=0, help="0 means min(number of datasets, CPU count)")
    p.add_argument("--point-size", type=float, default=26)
    p.add_argument("--point-alpha", type=float, default=0.82)
    p.add_argument("--cmap", default="Spectral_r")
    p.add_argument("--fig-width", type=float, default=8.2)
    p.add_argument("--fig-height", type=float, default=6.2)
    p.add_argument("--output-dpi", type=int, default=300)
    p.add_argument("--title-fontsize", type=int, default=15)
    p.add_argument("--axis-label-fontsize", type=int, default=13)
    p.add_argument("--info-fontsize", type=int, default=10)
    p.add_argument("--band-text-fontsize", type=int, default=10)
    p.add_argument("--core-box-xpad", type=float, default=0.40)
    p.add_argument("--core-box-ymin", type=float, default=0.04)
    p.add_argument("--core-box-ypad", type=float, default=0.03)
    p.add_argument("--aux-rmsd-lines", type=float, nargs="*", default=[2.0, 3.0, 4.0])
    return p


def main() -> None:
    args = build_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    datasets = read_dataset_config(args.dataset_config)
    args_dict = vars(args).copy()
    if args_dict["sample_n_per_file"] <= 0:
        args_dict["sample_n_per_file"] = None

    workers = args.workers or min(len(datasets), os.cpu_count() or 1)
    results_map: Dict[str, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one_dataset, item, args_dict): item["key"] for item in datasets}
        for fut in as_completed(futs):
            key = futs[fut]
            results_map[key] = fut.result()
            print(f"[DONE] {key}")
    results = [results_map[d["key"]] for d in datasets]

    all_z = [r["z"] for r in results if r["has_density"] and r["z"] is not None and len(r["z"]) > 0]
    if all_z:
        vals = np.concatenate(all_z)
        vmax = nice_colorbar_max(float(np.max(vals)))
        norm = Normalize(vmin=0.0, vmax=vmax)
        ticks = make_colorbar_ticks(vmax)
    else:
        norm = None
        ticks = None

    sns.set(style="ticks", context="talk", font_scale=1.0)
    for r in results:
        draw_one(r, norm, ticks, args)


if __name__ == "__main__":
    main()
