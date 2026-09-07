#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# =========================================================
#  ()
# =========================================================
INPUT_CSV = Path("CHANGE_ME")
OUTPUT_PLOT = Path("CHANGE_ME")

# Start2Fold 
CLASS_KEYS = [
    "Fold_EARLY", "Fold_INTER", "Fold_LATE",
    "Stab_STRONG", "Stab_MEDIUM", "Stab_WEAK"
]

#  Exact Combos
TARGET_COMBOS = [
    "Stab_MEDIUM",
    "Fold_EARLY;Fold_INTER;Stab_MEDIUM"
]

def get_exact_combo(row, threshold=2):
    """
     (Exact Class Combo)
    """
    hit_classes = []
    for cls in CLASS_KEYS:
        col = f"{cls}_segment_overlap_count"
        if col in row.index:
            try:
                val = float(row[col])
                if val >= threshold:
                    hit_classes.append(cls)
            except:
                pass
    if not hit_classes:
        return "no_hit"
    return ";".join(hit_classes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot selected exact Start2Fold class combinations as length-vs-overlap bubble plots.")
    p.add_argument("--input-csv", type=Path, default=INPUT_CSV, help="all_mapped_segments_with_start2fold_metrics.csv")
    p.add_argument("--output-plot", type=Path, default=OUTPUT_PLOT, help="Output PNG path.")
    p.add_argument("--target-combos", nargs="+", default=TARGET_COMBOS, help="Exact class combinations to plot.")
    p.add_argument("--threshold", type=float, default=2.0, help="Per-class overlap-count threshold used to define exact combos.")
    return p.parse_args()

def main():
    global INPUT_CSV, OUTPUT_PLOT, TARGET_COMBOS
    args = parse_args()
    INPUT_CSV = args.input_csv
    OUTPUT_PLOT = args.output_plot
    TARGET_COMBOS = args.target_combos
    combo_threshold = args.threshold
    if not INPUT_CSV.exists():
        print(f"[ERROR] : {INPUT_CSV}")
        return

    print("[INFO] ...")
    df = pd.read_csv(INPUT_CSV, dtype=str)

    # 1. ：
    df = df[df["Map_Status"] == "OK"].copy()

    # 
    df["Target_Seq_Len"] = pd.to_numeric(df["Target_Seq_Len"], errors="coerce").fillna(0)
    df["Any_Start2Fold_Overlap_Count"] = pd.to_numeric(df["Any_Start2Fold_Overlap_Count"], errors="coerce").fillna(0)

    print("[INFO]  Exact Class Combo (Threshold >= 2)...")
    # 2. 
    df["Exact_Combo"] = df.apply(lambda r: get_exact_combo(r, threshold=combo_threshold), axis=1)

    # 
    max_len = df["Target_Seq_Len"].max()
    x_vals = np.array([0, max_len + 2])

    # 3.  1x2 
    fig, axes = plt.subplots(nrows=1, ncols=max(len(TARGET_COMBOS), 1), figsize=(7 * max(len(TARGET_COMBOS), 1), 6), dpi=300)
    axes = np.array([axes]).flatten()

    print("[INFO] ...")

    for i, combo in enumerate(TARGET_COMBOS):
        ax = axes[i]
        
        #  Exact Combo 
        subset = df[df["Exact_Combo"] == combo].copy()

        if subset.empty:
            ax.set_title(f"{combo} (n=0)", fontsize=13)
            continue

        # 
        plot_data = subset.groupby(["Target_Seq_Len", "Any_Start2Fold_Overlap_Count"]).size().reset_index(name="Segment_Count")

        # 
        sns.scatterplot(
            data=plot_data,
            x="Target_Seq_Len",
            y="Any_Start2Fold_Overlap_Count",
            size="Segment_Count",
            sizes=(50, 800), 
            alpha=0.75,
            color="#2b8cbe" if i == 0 else "#e6550d",
            edgecolor="white",
            linewidth=1.2,
            ax=ax
        )

        # 
        ax.plot(x_vals, x_vals * 1.0, color='red', linestyle='--', alpha=0.5, label="100% Overlap")
        ax.plot(x_vals, x_vals * 0.5, color='orange', linestyle='--', alpha=0.5, label="50% Overlap")
        ax.plot(x_vals, x_vals * 0.2, color='green', linestyle='--', alpha=0.5, label="20% Overlap")

        # 
        ax.set_xlim(left=0, right=max_len + 1)
        ax.set_ylim(bottom=-0.5, top=max_len + 1)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 
        display_title = combo.replace(";", "\n+") 
        ax.set_title(f"Exact Combo: {display_title}\n(n={len(subset)})", fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Fragment Length (aa)", fontsize=11)
        if i == 0:
            ax.set_ylabel("Total HDX Hit Count", fontsize=11)
        else:
            ax.set_ylabel("")
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left', fontsize=9)

    plt.tight_layout()

    # 4. 
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, bbox_inches="tight")
    print(f"[SUCCESS] : {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()