#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare phi-related Pfam IDs against a missing-Pfam list."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PFAM_CSV = Path("CHANGE_ME")
DEFAULT_MISSING_TXT = Path("CHANGE_ME")
DEFAULT_OUT_DIR = Path("CHANGE_ME")


def read_id_list(path: Path) -> list[str]:
    ids = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            x = line.strip().upper()
            if x:
                ids.append(x)
    return sorted(set(ids))


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pfam_csv)
    if "pfam_id" not in df.columns:
        raise ValueError(f"Input CSV has no pfam_id column. Available columns: {list(df.columns)}")

    pfam_ids = df["pfam_id"].astype(str).str.strip().str.upper()
    pfam_ids = sorted(set(x for x in pfam_ids if x and x != "NAN"))
    missing_ids = read_id_list(args.missing_txt)

    pfam_set = set(pfam_ids)
    missing_set = set(missing_ids)

    matched = sorted(pfam_set & missing_set)
    not_in_missing = sorted(pfam_set - missing_set)
    missing_but_not_in_residue = sorted(missing_set - pfam_set)

    (args.out_dir / "matched_pfam_ids.txt").write_text("\n".join(matched) + ("\n" if matched else ""), encoding="utf-8")
    (args.out_dir / "pfam_ids_not_in_missing.txt").write_text("\n".join(not_in_missing) + ("\n" if not_in_missing else ""), encoding="utf-8")
    (args.out_dir / "missing_but_not_in_residue.txt").write_text("\n".join(missing_but_not_in_residue) + ("\n" if missing_but_not_in_residue else ""), encoding="utf-8")

    pd.DataFrame({"pfam_id": matched}).to_csv(args.out_dir / "matched_pfam_ids.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"pfam_id": not_in_missing}).to_csv(args.out_dir / "pfam_ids_not_in_missing.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"pfam_id": missing_but_not_in_residue}).to_csv(args.out_dir / "missing_but_not_in_residue.csv", index=False, encoding="utf-8-sig")

    summary = [
        f"pfam_residue_summary unique pfam_id: {len(pfam_set)}",
        f"missing_pfam_ids unique IDs: {len(missing_set)}",
        f"intersection count: {len(matched)}",
        f"only in pfam_residue_summary: {len(not_in_missing)}",
        f"only in missing_pfam_ids: {len(missing_but_not_in_residue)}",
    ]
    (args.out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))
    print(f"\nOutput directory: {args.out_dir}")
    print(f"Intersection file: {args.out_dir / 'matched_pfam_ids.txt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare phi-related Pfam IDs against a missing-Pfam list.")
    parser.add_argument("--pfam-csv", type=Path, default=DEFAULT_PFAM_CSV, help="CSV containing a pfam_id column.")
    parser.add_argument("--missing-txt", type=Path, default=DEFAULT_MISSING_TXT, help="Text file containing one missing Pfam ID per line.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
