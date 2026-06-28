#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Only 1 output:
- ALL(null) vs ALL(real hit) comparison in ONE figure
  2x2 grid (row=dataset, col=SS/RSA)

Data source: SQLite
- hit_obs  : real mapped segments (hit_avg_rsa, hit_ss_class)
- null_win : random windows (win_avg_rsa, win_ss_class)

Notes
-----
- "Location" is derived from Avg_RSA thresholds:
    <0.2      -> Buried (Internal)
    0.2-0.5   -> Intermediate
    >=0.5     -> Exposed (Surface)
- RSA histogram uses stat="probability"
"""

import os
import argparse
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Config
# =========================

DB_PATH = "CHANGE_ME"
OUT_DIR = "CHANGE_ME"

# Only 1 output
OUT_ALL_REAL = os.path.join(OUT_DIR, "04_all_null_vs_all_real.png")

# Filters
KEEP_ONLY_OK_HIT = True
KEEP_ONLY_OK_WIN = True

# Plot control
BINS = 40
USE_KDE = False
DPI = 900

# Categories order
SS_ORDER  = ["Helix_Dominant", "Sheet_Dominant", "Coil/Loop_Dominant", "Mixed"]
LOC_ORDER = ["Buried (Internal)", "Intermediate", "Exposed (Surface)"]

# Colors for Location stack
CUSTOM_COLORS = ["#4575b4", "#fee090", "#d73027"]  # buried/intermediate/exposed

# RSA thresholds
THR_BURIED = 0.2
THR_EXPOSED = 0.5


# =========================
# Helpers
# =========================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def add_location(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.to_numeric(df["Avg_RSA"], errors="coerce")
    conds = [
        x < THR_BURIED,
        (x >= THR_BURIED) & (x < THR_EXPOSED),
        x >= THR_EXPOSED
    ]
    choices = [LOC_ORDER[0], LOC_ORDER[1], LOC_ORDER[2]]
    df["Location"] = np.select(conds, choices, default=np.nan)
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Avg_RSA"] = pd.to_numeric(df["Avg_RSA"], errors="coerce")
    df = df[df["Avg_RSA"].notna()].copy()
    df = df[df["SS_Class"].isin(SS_ORDER)].copy()
    df = add_location(df)
    df = df[df["Location"].isin(LOC_ORDER)].copy()

    df["SS_Class"] = pd.Categorical(df["SS_Class"], categories=SS_ORDER, ordered=True)
    df["Location"] = pd.Categorical(df["Location"], categories=LOC_ORDER, ordered=True)
    return df

def plot_two_rows(df_top: pd.DataFrame, df_bot: pd.DataFrame,
                  row1_name: str, row2_name: str,
                  title: str, out_png: str):
    """
    2x2 figure:
      row1: df_top  (left SS, right RSA)
      row2: df_bot  (left SS, right RSA)
    """
    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    plt.subplots_adjust(wspace=0.22, hspace=0.25)
    def plot_ss(ax, df, label):
        if len(df) == 0:
            ax.set_title(f"{label} | SS: NO DATA")
            ax.axis("off")
            return

        # Raw count table.
        ct = (
            pd.crosstab(df["SS_Class"], df["Location"])
            .reindex(SS_ORDER)
            .reindex(LOC_ORDER, axis=1)
            .fillna(0)
        )

        # Per-class totals for N labels.
        totals = ct.sum(axis=1)

        # Convert counts to proportions of the grand total.
        grand_total = totals.sum()
        if grand_total > 0:
            ct_plot = ct / grand_total
        else:
            ct_plot = ct.copy()

        ct_plot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=CUSTOM_COLORS,
            edgecolor="black",
            width=0.7
        )

        # Keep the class total N above each bar.
        y_sums = ct_plot.sum(axis=1)
        y_offset = 0.01
        for i, ss_name in enumerate(SS_ORDER):
            count = int(totals.loc[ss_name]) if ss_name in totals.index else 0
            y = float(y_sums.loc[ss_name]) if ss_name in y_sums.index else 0.0
            ax.text(
                i, min(y + y_offset, 0.99), f"N={count}",
                ha="center", va="bottom",
                fontsize=18, fontweight="bold"
            )

        ax.set_title("", fontsize=18, pad=15)
        ax.set_xlabel("SS Class", fontsize=18, fontweight="bold")
        ax.set_ylabel("Proportion of total", fontsize=18, fontweight="bold")
        ax.tick_params(axis="x", rotation=10)

        # Fix y-axis range from 0 to 1.
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))

        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.legend(title="Location", frameon=False)
        sns.despine(ax=ax)

    def plot_rsa(ax, df, label):
        if len(df) == 0:
            ax.set_title(f"{label} | RSA: NO DATA")
            ax.axis("off")
            return

        sns.histplot(
            data=df,
            x="Avg_RSA",
            bins=BINS,
            kde=USE_KDE,
            ax=ax,
            alpha=0.65,
            edgecolor=None,
            stat="probability",
            color="#666666"
        )
        ax.axvline(THR_BURIED, color=CUSTOM_COLORS[0], linestyle="--", alpha=0.8,
                   label=f"Buried threshold ({THR_BURIED})")
        ax.axvline(THR_EXPOSED, color=CUSTOM_COLORS[2], linestyle="--", alpha=0.8,
                   label=f"Exposed threshold ({THR_EXPOSED})")

        ax.set_title(f"",
                     fontsize=18, pad=15)
        ax.set_xlabel("Avg RSA", fontsize=18, fontweight="bold")
        ax.set_ylabel("Probability", fontsize=18, fontweight="bold")
        ax.legend(frameon=False)
        sns.despine(ax=ax)

    # Row 1: real hits
    plot_ss(axes[0, 0], df_top, row1_name)
    plot_rsa(axes[0, 1], df_top, row1_name)

    # Row 2: null windows
    plot_ss(axes[1, 0], df_bot, row2_name)
    plot_rsa(axes[1, 1], df_bot, row2_name)

    fig.suptitle(title, fontsize=18, y=0.98, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"[OUT] {out_png}")


# =========================
# SQL loaders
# =========================

def load_null_all(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    ALL null_win (no grp split)
    """
    where = "1=1"
    if KEEP_ONLY_OK_WIN:
        where += " AND status = 'OK'"

    q = f"""
    SELECT
      win_ss_class AS SS_Class,
      win_avg_rsa  AS Avg_RSA
    FROM null_win
    WHERE {where}
    """
    return pd.read_sql_query(q, conn)

def load_hit_all(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    ALL hit_obs (no grp split)
    """
    where = "1=1"
    if KEEP_ONLY_OK_HIT:
        where += " AND status = 'OK'"

    q = f"""
    SELECT
      hit_ss_class AS SS_Class,
      hit_avg_rsa  AS Avg_RSA
    FROM hit_obs
    WHERE {where}
    """
    return pd.read_sql_query(q, conn)


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Pfam-Pfam all-real vs all-random SS/RSA distributions from the SQLite background database."
    )
    parser.add_argument("--db", default=DB_PATH, help="Input SQLite database generated by 02_build_pfam_random_background_sqlite.py.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory used when --out is not provided.")
    parser.add_argument("--out", default=None, help="Output PNG path. Default: <out-dir>/04_all_null_vs_all_real.png")
    parser.add_argument("--bins", type=int, default=BINS, help="Number of RSA histogram bins.")
    parser.add_argument("--dpi", type=int, default=DPI, help="Output figure DPI.")
    parser.add_argument("--use-kde", action="store_true", default=USE_KDE, help="Enable seaborn KDE for RSA histograms.")
    parser.add_argument("--keep-non-ok-hit", action="store_true", help="Do not restrict hit_obs rows to status == OK.")
    parser.add_argument("--keep-non-ok-win", action="store_true", help="Do not restrict null_win rows to status == OK.")
    return parser.parse_args()

def apply_args(args):
    global DB_PATH, OUT_DIR, OUT_ALL_REAL, BINS, DPI, USE_KDE, KEEP_ONLY_OK_HIT, KEEP_ONLY_OK_WIN
    DB_PATH = args.db
    OUT_DIR = args.out_dir
    OUT_ALL_REAL = args.out if args.out is not None else os.path.join(OUT_DIR, "04_all_null_vs_all_real.png")
    BINS = args.bins
    DPI = args.dpi
    USE_KDE = args.use_kde
    KEEP_ONLY_OK_HIT = not args.keep_non_ok_hit
    KEEP_ONLY_OK_WIN = not args.keep_non_ok_win

def main():
    args = parse_args()
    apply_args(args)

    ensure_dir(os.path.dirname(OUT_ALL_REAL) or OUT_DIR)

    if not os.path.exists(DB_PATH):
        print(f"[FATAL] DB not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    print("[INFO] Load ALL real(hit_obs) ...")
    df_all_real = clean_df(load_hit_all(conn))

    print("[INFO] Load ALL null(null_win) ...")
    df_all_null = clean_df(load_null_all(conn))

    plot_two_rows(
        df_top=df_all_real,
        df_bot=df_all_null,
        row1_name="ALL (Real hits)",
        row2_name="ALL (Null windows)",
        title="",
        out_png=OUT_ALL_REAL
    )

    conn.close()
    print("[DONE] Output saved to:", OUT_ALL_REAL)


if __name__ == "__main__":
    main()