#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Configuration =================
STRUCT_CSV = "CHANGE_ME"
SCORE_CSV  = "<PRODIVE_DATA_ROOT>/shared/global_high_score_summary_fin.csv"

OUT_DIR = "CHANGE_ME"
os.makedirs(OUT_DIR, exist_ok=True)

# Score column name in SCORE_CSV.
SCORE_COL_IN_FILE = "Score"   # Must exactly match the CSV column name, including case.

# Merge keys.
KEY_COLS = ["File", "Main_Segment", "Sub_Segments_Details"]

TOP_PCTS = [5, 10, 20, 50, 100]

# Rename the score column before merge to avoid collisions.
RANK_COL = "Rank_Score"


def normalize_keys(df: pd.DataFrame, key_cols):
    df = df.copy()
    for c in key_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].str.replace(r"\s+", " ", regex=True)

    if "Sub_Segments_Details" in df.columns:
        df["Sub_Segments_Details"] = df["Sub_Segments_Details"].astype(str)
        df["Sub_Segments_Details"] = df["Sub_Segments_Details"].str.replace(
            r"\s*->\s*", " -> ", regex=True
        ).str.strip()
    return df


def reshape_data(df):
    """
    Reshape Main and Sub side features into one long table.
    RSA is plotted as one combined Main+Sub distribution, so Type is not used for plotting.
    """
    df = df.copy()
    df["Main_Avg_RSA"] = pd.to_numeric(df.get("Main_Avg_RSA"), errors="coerce")
    df["Sub_Avg_RSA"]  = pd.to_numeric(df.get("Sub_Avg_RSA"), errors="coerce")

    df_main = df[["Main_Status", "Main_SS_Class", "Main_Location", "Main_Avg_RSA"]].copy()
    df_main.columns = ["Status", "SS_Class", "Location", "Avg_RSA"]
    df_main["Type"] = "Main"

    df_sub = df[["Sub_Status", "Sub_SS_Class", "Sub_Location", "Sub_Avg_RSA"]].copy()
    df_sub.columns = ["Status", "SS_Class", "Location", "Avg_RSA"]
    df_sub["Type"] = "Sub"

    return pd.concat([df_main, df_sub], ignore_index=True)


def plot_stacked_distribution(df_long, output_path, title_suffix):
    print(f"[INFO] Plotting: {title_suffix} | long-table rows: {len(df_long)}")

    df_valid = df_long[
        (df_long["Avg_RSA"].notna()) &
        (df_long["SS_Class"] != "Struct_Not_Found")
    ].copy()

    if len(df_valid) == 0:
        print(f"[WARN] {title_suffix}: no valid data; skipped.")
        return

    ss_order  = ["Helix_Dominant", "Sheet_Dominant", "Coil/Loop_Dominant", "Mixed"]
    loc_order = ["Buried (Internal)", "Intermediate", "Exposed (Surface)"]
    custom_colors = ["#4575b4", "#fee090", "#d73027"]

    df_clean = df_valid[
        df_valid["SS_Class"].isin(ss_order) &
        df_valid["Location"].isin(loc_order)
    ].copy()

    if len(df_clean) == 0:
        print(f"[WARN] {title_suffix}: no rows after standard class filtering; skipped.")
        return

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    plt.subplots_adjust(wspace=0.25)

    # Left panel: SS x Location stacked bar plot as a proportion of total fragments.
    ax1 = axes[0]

    # Raw count table.
    ct_count = (
        pd.crosstab(df_clean["SS_Class"], df_clean["Location"])
        .reindex(ss_order)
        .reindex(loc_order, axis=1)
        .fillna(0)
    )

    total_n = ct_count.to_numpy().sum()
    if total_n <= 0:
        print(f"[WARN] {title_suffix}: total_n <= 0; skipped.")
        return

    # Proportion of the grand total, not within-class normalization.
    ct_prop_total = ct_count / total_n

    ct_prop_total.plot(
        kind="bar",
        stacked=True,
        ax=ax1,
        color=custom_colors,
        edgecolor="black",
        width=0.7
    )

    # Label each secondary-structure class with its total N.
    totals = ct_count.sum(axis=1)
    y_offset = max(ct_prop_total.sum(axis=1).max() * 0.02, 0.005)

    for i, ss_name in enumerate(ss_order):
        count = totals.get(ss_name, 0)
        bar_height = ct_prop_total.loc[ss_name].sum() if ss_name in ct_prop_total.index else 0
        ax1.text(
            i, bar_height + y_offset, f"N={int(count)}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )

    ax1.set_title(f"Secondary structure distribution ({title_suffix})", fontsize=15, pad=20)
    ax1.set_xlabel("Secondary Structure Class", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Proportion of Total Fragments", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="x", rotation=10)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(title="Location", frameon=False, loc="upper right")
    sns.despine(ax=ax1)

    # Right panel: one combined RSA distribution for Main + Sub.
    ax2 = axes[1]
    rsa_all = df_clean["Avg_RSA"].dropna()

    sns.histplot(
        rsa_all,
        bins=40,
        kde=True,
        ax=ax2,
        color="#666666",
        alpha=0.65,
        edgecolor=None,
        stat="probability"
    )

    ax2.axvline(0.2, color="#4575b4", linestyle="--", alpha=0.7, label="Buried threshold (0.2)")
    ax2.axvline(0.5, color="#d73027", linestyle="--", alpha=0.7, label="Exposed threshold (0.5)")

    ax2.set_title(f"RSA distribution ({title_suffix})", fontsize=15, pad=20)
    ax2.set_xlabel("Average RSA", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Probability", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax2)

    # Save.
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OUT] {output_path}")


def parse_top_pcts(value: str):
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Pfam-Pfam SS/RSA distributions for score-ranked top-percentile subsets."
    )
    parser.add_argument("--struct-csv", default=STRUCT_CSV, help="Input Pfam-Pfam SS/RSA CSV.")
    parser.add_argument("--score-csv", default=SCORE_CSV, help="Input high-score summary CSV containing ranking scores.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory for top-percentile plots.")
    parser.add_argument("--score-col", default=SCORE_COL_IN_FILE, help="Score column in --score-csv.")
    parser.add_argument("--rank-col", default=RANK_COL, help="Internal ranking column name used after merge.")
    parser.add_argument("--top-pcts", default=",".join(map(str, TOP_PCTS)), help="Comma-separated top percentages, e.g. 5,10,20,50,100.")
    return parser.parse_args()

def apply_args(args):
    global STRUCT_CSV, SCORE_CSV, OUT_DIR, SCORE_COL_IN_FILE, RANK_COL, TOP_PCTS
    STRUCT_CSV = args.struct_csv
    SCORE_CSV = args.score_csv
    OUT_DIR = args.out_dir
    SCORE_COL_IN_FILE = args.score_col
    RANK_COL = args.rank_col
    TOP_PCTS = parse_top_pcts(args.top_pcts)

def main():
    args = parse_args()
    apply_args(args)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(STRUCT_CSV):
        print(f"[ERR] STRUCT_CSV does not exist: {STRUCT_CSV}")
        return
    if not os.path.exists(SCORE_CSV):
        print(f"[ERR] SCORE_CSV does not exist: {SCORE_CSV}")
        return

    print("[INFO] 1) Reading STRUCT_CSV ...")
    df_struct = pd.read_csv(STRUCT_CSV, low_memory=False)
    df_struct = normalize_keys(df_struct, KEY_COLS)

    print("[INFO] 2) Reading SCORE_CSV ...")
    df_score = pd.read_csv(SCORE_CSV, low_memory=False)

    missing = [c for c in KEY_COLS + [SCORE_COL_IN_FILE] if c not in df_score.columns]
    if missing:
        print(f"[ERR] SCORE_CSV is missing columns: {missing}")
        print("      Inspect columns with: python -c \"import pandas as pd;print(pd.read_csv(r'{}',nrows=1).columns)\"".format(SCORE_CSV))
        return

    df_score = df_score[KEY_COLS + [SCORE_COL_IN_FILE]].copy()
    df_score = normalize_keys(df_score, KEY_COLS)

    df_score[RANK_COL] = pd.to_numeric(df_score[SCORE_COL_IN_FILE], errors="coerce")
    df_score.drop(columns=[SCORE_COL_IN_FILE], inplace=True)

    before = len(df_score)
    df_score = df_score[df_score[RANK_COL].notna()].copy()
    print(f"[INFO] Valid SCORE rows: {len(df_score)} / {before}")

    dup = df_score.duplicated(subset=KEY_COLS).sum()
    if dup > 0:
        print(f"[WARN] SCORE_CSV has duplicated fragment keys: {dup} rows. Keeping max {RANK_COL} per key.")
        df_score = df_score.groupby(KEY_COLS, as_index=False)[RANK_COL].max()

    print("[INFO] 3) Merge STRUCT + SCORE (inner join, fragment level) ...")
    df_merged = pd.merge(df_struct, df_score, on=KEY_COLS, how="inner")
    print(f"[INFO] Merged fragment count: {len(df_merged)}")

    if len(df_merged) == 0:
        print("[ERR] Empty merge result. This usually means key mismatch in spacing, arrow formatting, or field content.")
        return

    df_merged = df_merged.sort_values(RANK_COL, ascending=False).reset_index(drop=True)

    n = len(df_merged)
    for pct in TOP_PCTS:
        k = max(1, int(round(n * (pct / 100.0))))
        df_top = df_merged.iloc[:k].copy()

        df_long = reshape_data(df_top)

        out_png = os.path.join(OUT_DIR, f"pfam_dist_Top{pct}pct_by_{RANK_COL}.png")
        title = f"Top {pct}% score-ranked fragment pairs"
        plot_stacked_distribution(df_long, out_png, title)

    print("\n[DONE] All plots completed.")


if __name__ == "__main__":
    main()