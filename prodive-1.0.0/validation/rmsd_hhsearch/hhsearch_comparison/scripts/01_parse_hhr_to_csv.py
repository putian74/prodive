#!/usr/bin/env python3
"""
Parse HHsearch .hhr files into a CSV table used by downstream ProDive/HHsearch comparison scripts.

This script intentionally preserves the original output schema:
    Main_HMM, Sub_HMM, Sub_Segments_Details, Prob

Original behavior preserved:
    1. Keep hits with Prob >= threshold.
    2. Remove self hits where Main_HMM == Sub_HMM.
    3. Remove undirected duplicated family pairs, e.g. A-B and B-A.
    4. Store HHsearch segment range as "query_range -> target_range" in Sub_Segments_Details.
"""

import argparse
import os
import re
from typing import Dict, List, Any

import pandas as pd
from tqdm import tqdm


def parse_single_hhr(filepath: str, prob_threshold: float) -> List[Dict[str, Any]]:
    """
    Parse a single .hhr file.

    Parameters
    ----------
    filepath : str
        Path to one HHsearch .hhr output file.
    prob_threshold : float
        Minimum HHsearch probability threshold. Original behavior is Prob >= threshold.

    Returns
    -------
    list of dict
        Rows with columns Main_HMM, Sub_HMM, Sub_Segments_Details, Prob.
    """
    extracted_rows: List[Dict[str, Any]] = []
    main_hmm = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    start_parsing_table = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Query ID.
        if line.startswith("Query"):
            parts = line.split()
            if len(parts) >= 2:
                main_hmm = parts[1].replace(".hhm", "").replace(".hhs", "")
            continue

        # HHsearch summary table header.
        if "No Hit" in line and "Prob" in line:
            start_parsing_table = True
            continue

        if start_parsing_table:
            # Summary table rows start with rank numbers.
            if not line[0].isdigit():
                if line.startswith("No 1"):
                    break
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            try:
                prob_val = float(parts[2])
                if prob_val < prob_threshold:
                    continue

                sub_hmm = parts[1]
                ranges = re.findall(r"(\d+-\d+)", line)

                if len(ranges) >= 2:
                    q_range = ranges[-2]
                    t_range = ranges[-1]
                    details_str = f"{q_range} -> {t_range}"

                    extracted_rows.append({
                        "Main_HMM": main_hmm,
                        "Sub_HMM": sub_hmm,
                        "Sub_Segments_Details": details_str,
                        "Prob": prob_val,
                    })
            except Exception:
                # Keep original tolerant behavior: skip malformed summary rows.
                continue

    return extracted_rows


def parse_hhr_directory(
    input_dir: str,
    output_csv: str,
    prob_threshold: float,
    file_extension: str,
) -> None:
    """Parse all .hhr files in a directory and save one CSV."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_parent = os.path.dirname(os.path.abspath(output_csv))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(file_extension)]
    files.sort()

    print(f"Found {len(files)} files in: {input_dir}")
    print(f"Current filter: Prob >= {prob_threshold}")
    print(f"Output CSV: {output_csv}")

    all_data: List[Dict[str, Any]] = []
    seen_pairs = set()

    for filename in tqdm(files, desc="Parsing HHR files"):
        filepath = os.path.join(input_dir, filename)
        rows = parse_single_hhr(filepath, prob_threshold)

        for row in rows:
            main_id = row["Main_HMM"]
            sub_id = row["Sub_HMM"]

            # Remove self hits.
            if main_id == sub_id:
                continue

            # Remove undirected duplicates.
            pair_key = tuple(sorted([main_id, sub_id]))
            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)
            all_data.append(row)

    df = pd.DataFrame(all_data, columns=["Main_HMM", "Sub_HMM", "Sub_Segments_Details", "Prob"])
    df.to_csv(output_csv, index=False)

    print(f"Done. Extracted {len(df):,} valid records.")
    if not df.empty:
        print(df.head())
    else:
        print("Warning: output table is empty. Check input directory, file extension, or probability threshold.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse HHsearch .hhr files into hhsuite{threshold}.csv for ProDive comparison."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing HHsearch .hhr files.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path, for example hhsuite20.csv.",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=20.0,
        help="Minimum HHsearch probability threshold. Original behavior: Prob >= threshold. Default: 20.",
    )
    parser.add_argument(
        "--file-extension",
        default=".hhr",
        help="HHsearch result file extension. Default: .hhr",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    parse_hhr_directory(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        prob_threshold=args.prob_threshold,
        file_extension=args.file_extension,
    )


if __name__ == "__main__":
    main()
