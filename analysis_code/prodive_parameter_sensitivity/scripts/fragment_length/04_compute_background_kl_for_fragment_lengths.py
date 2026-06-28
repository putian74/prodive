#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ==============================================================================
# Configuration
# ==============================================================================

PFAM_LIST_FILE = Path("CHANGE_ME")
PFAM_SEED_DIR = Path("CHANGE_ME")

OUT_DIR = Path("CHANGE_ME")

FRAGMENT_LENGTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

ROUND_DIGITS = 4


# ==============================================================================
# HHM parsing and background KL calculation
# ==============================================================================

def parse_hhm_column_kl_values(hhm_file_path: Path):
    """
    Parse one HHsuite .hhm file.

    Returns:
        list[float]
            Per-match-state KL divergence against the NULL background.

    The returned list length should be the HMM length.
    """
    if not hhm_file_path.exists():
        return None

    background_probs = None
    column_kls = []

    try:
        with hhm_file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                # NULL background row
                if parts[0] == "NULL":
                    raw_scores = np.array([float(p) for p in parts[1:21]], dtype=np.float64)
                    background_probs = 2.0 ** (-raw_scores / 1000.0)

                    bg_sum = background_probs.sum()
                    if bg_sum > 0:
                        background_probs = background_probs / bg_sum
                    continue

                # Match-state emission row.
                # Original rule from your code:
                # len(parts) > 20 and parts[0].isalpha() and parts[0].isupper() and parts[1].isdigit()
                if (
                    len(parts) > 20
                    and parts[0].isalpha()
                    and parts[0].isupper()
                    and len(parts) > 1
                    and parts[1].isdigit()
                ):
                    if background_probs is None:
                        continue

                    raw_emission_scores = np.array([float(p) for p in parts[2:22]], dtype=np.float64)

                    with np.errstate(over="ignore", invalid="ignore"):
                        emission_probs = 2.0 ** (-raw_emission_scores / 1000.0)

                    emission_sum = emission_probs.sum()

                    if not np.isfinite(emission_sum) or emission_sum <= 0:
                        continue

                    emission_probs = emission_probs / emission_sum

                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.divide(
                            emission_probs,
                            background_probs,
                            out=np.ones_like(emission_probs),
                            where=background_probs > 0,
                        )
                        terms = emission_probs * np.log(ratio)

                    kl_divergence = float(np.nansum(terms))

                    if np.isfinite(kl_divergence):
                        column_kls.append(kl_divergence)

        return column_kls

    except Exception as e:
        print(f"[WARN] Failed to parse {hhm_file_path}: {e}")
        return None


def sliding_window_mean(values, fragment_length: int, round_digits: int = 4):
    """
    Convert per-column KL values into per-window background values.
    """
    if values is None:
        return None

    n = len(values)

    if n < fragment_length:
        return []

    arr = np.asarray(values, dtype=np.float64)

    # cumulative sum for efficient window mean
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    win_sum = csum[fragment_length:] - csum[:-fragment_length]
    win_mean = win_sum / float(fragment_length)

    return [round(float(x), round_digits) for x in win_mean]


def load_pfam_list(path: Path):
    pfams = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                pfams.append(x)

    return pfams


def atomic_write_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)




def parse_fragment_lengths(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(',') if x.strip()]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate background KL JSON files for multiple fragment lengths.")
    parser.add_argument("--pfam-list-file", type=Path, default=PFAM_LIST_FILE, help="Text file containing one Pfam ID per line.")
    parser.add_argument("--pfam-seed-dir", type=Path, default=PFAM_SEED_DIR, help="PfamA_seed root directory.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory.")
    parser.add_argument("--fragment-lengths", default=','.join(map(str, FRAGMENT_LENGTHS)), help="Comma-separated fragment lengths.")
    parser.add_argument("--round-digits", type=int, default=ROUND_DIGITS, help="Decimal digits for saved background values.")
    return parser.parse_args()

def apply_runtime_args(args: argparse.Namespace) -> None:
    global PFAM_LIST_FILE, PFAM_SEED_DIR, OUT_DIR, FRAGMENT_LENGTHS, ROUND_DIGITS
    PFAM_LIST_FILE = args.pfam_list_file
    PFAM_SEED_DIR = args.pfam_seed_dir
    OUT_DIR = args.out_dir
    FRAGMENT_LENGTHS = parse_fragment_lengths(args.fragment_lengths)
    ROUND_DIGITS = int(args.round_digits)

# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    apply_runtime_args(args)
    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pfam_ids = load_pfam_list(PFAM_LIST_FILE)

    print("=" * 100)
    print("Generate background KL JSON files for multiple fragment lengths")
    print(f"PFAM_LIST_FILE: {PFAM_LIST_FILE}")
    print(f"PFAM_SEED_DIR:  {PFAM_SEED_DIR}")
    print(f"OUT_DIR:        {OUT_DIR}")
    print(f"FRAGMENT_LENGTHS: {FRAGMENT_LENGTHS}")
    print(f"Pfam count: {len(pfam_ids)}")
    print("=" * 100)

    # First parse each HHM once.
    per_family_column_kls = {}
    missing_files = []
    parse_failed = []

    for pfam_id in tqdm(pfam_ids, desc="Parsing HHM files", unit="family"):
        hhm_file = PFAM_SEED_DIR / pfam_id / f"{pfam_id}.hhm"

        if not hhm_file.exists():
            missing_files.append(str(hhm_file))
            continue

        kls = parse_hhm_column_kl_values(hhm_file)

        if kls is None:
            parse_failed.append(str(hhm_file))
            continue

        per_family_column_kls[pfam_id] = kls

    print(f"[INFO] Parsed families: {len(per_family_column_kls)}")
    print(f"[INFO] Missing HHM files: {len(missing_files)}")
    print(f"[INFO] Parse failed: {len(parse_failed)}")

    # Generate one JSON per fragment length.
    summary_rows = []

    for fragment in FRAGMENT_LENGTHS:
        result = {}
        empty_count = 0

        for pfam_id, column_kls in tqdm(
            per_family_column_kls.items(),
            desc=f"Generating k={fragment}",
            unit="family",
        ):
            bg = sliding_window_mean(column_kls, fragment, ROUND_DIGITS)

            if bg is None:
                continue

            if len(bg) == 0:
                empty_count += 1

            result[pfam_id] = bg

        out_json = OUT_DIR / f"result_kl_all_pfam_fragment_{fragment}.json"
        atomic_write_json(result, out_json)

        total_windows = sum(len(v) for v in result.values())

        summary_rows.append({
            "fragment": fragment,
            "families": len(result),
            "empty_families": empty_count,
            "total_windows": total_windows,
            "output_json": str(out_json),
        })

        print(
            f"[DONE] k={fragment}: families={len(result)}, "
            f"empty={empty_count}, total_windows={total_windows}, output={out_json}"
        )

    summary_path = OUT_DIR / "background_kl_generation_summary.json"
    atomic_write_json(
        {
            "pfam_list_file": str(PFAM_LIST_FILE),
            "pfam_seed_dir": str(PFAM_SEED_DIR),
            "out_dir": str(OUT_DIR),
            "fragment_lengths": FRAGMENT_LENGTHS,
            "parsed_families": len(per_family_column_kls),
            "missing_files": missing_files,
            "parse_failed": parse_failed,
            "outputs": summary_rows,
            "round_digits": ROUND_DIGITS,
            "definition": "window_background_K = mean per-column KL(emission || NULL background)",
        },
        summary_path,
    )

    dt = time.time() - t0

    print("=" * 100)
    print("[DONE] All background JSON files generated")
    print(f"Summary: {summary_path}")
    print(f"Total time: {dt:.1f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()