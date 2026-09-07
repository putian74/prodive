#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

DEFAULT_SUMMARY_CSV = "CHANGE_ME"
DEFAULT_OUT_DIR = "CHANGE_ME"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select high-rFCO tail subsets from actual_vs_random_summary.csv."
    )
    p.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV, help="Path to actual_vs_random_summary.csv.")
    p.add_argument("--outdir", default=DEFAULT_OUT_DIR, help="Output directory.")

    p.add_argument("--loose-rfco", type=float, default=0.35)

    p.add_argument("--strict-rfco", type=float, default=0.40)
    p.add_argument("--strict-delta", type=float, default=0.10)
    p.add_argument("--strict-emp-p", type=float, default=0.99)
    p.add_argument("--strict-min-contacts", type=int, default=10)

    p.add_argument("--extreme-rfco", type=float, default=0.50)
    p.add_argument("--extreme-delta", type=float, default=0.15)
    p.add_argument("--extreme-emp-p", type=float, default=0.995)
    p.add_argument("--extreme-min-contacts", type=int, default=10)
    return p.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"summary CSV missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv, low_memory=False)
    require_columns(
        df,
        ["rfco", "rfco_delta_vs_random_mean", "rfco_empirical_p_le", "n_fragment_contacts"],
    )

    for c in ["rfco", "rfco_delta_vs_random_mean", "rfco_empirical_p_le", "n_fragment_contacts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    tail_loose = df[df["rfco"] >= args.loose_rfco].copy()

    tail_strict = df[
        (df["rfco"] >= args.strict_rfco)
        & (df["rfco_delta_vs_random_mean"] >= args.strict_delta)
        & (df["rfco_empirical_p_le"] >= args.strict_emp_p)
        & (df["n_fragment_contacts"] >= args.strict_min_contacts)
    ].copy()

    tail_extreme = df[
        (df["rfco"] >= args.extreme_rfco)
        & (df["rfco_delta_vs_random_mean"] >= args.extreme_delta)
        & (df["rfco_empirical_p_le"] >= args.extreme_emp_p)
        & (df["n_fragment_contacts"] >= args.extreme_min_contacts)
    ].copy()

    tail_loose.to_csv(outdir / "tail_loose_rfco_ge_0.35.csv", index=False)
    tail_strict.to_csv(outdir / "tail_strict_rfco_ge_0.40.csv", index=False)
    tail_extreme.to_csv(outdir / "tail_extreme_rfco_ge_0.50.csv", index=False)

    lines = [
        f"loose: {len(tail_loose)}",
        f"strict: {len(tail_strict)}",
        f"extreme: {len(tail_extreme)}",
    ]
    (outdir / "high_rfco_tail_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
