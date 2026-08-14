#!/usr/bin/env python3
"""Generate length-stratified random Pfam-Pfam structural controls.

The script samples random C-alpha windows from two different Pfam families,
superposes the windows with PyMOL, and retains accepted pairs until the target
quota is reached for each aligned C-alpha length. The default bins are aligned
lengths 8-13, matching the compact-core region used in structural validation.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

FIELDNAMES = [
    "Family_A", "PDB_A", "Range_A",
    "Family_B", "PDB_B", "Range_B",
    "RMSD", "Aligned_Atoms", "Target_Len", "Coverage",
]


def parse_length_bins(text: str) -> List[int]:
    values: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("At least one aligned-length bin is required.")
    return sorted(set(values))


def parse_pdb_info(file_path: str) -> Tuple[str, str]:
    path = Path(file_path)
    base = path.stem
    parts = base.split("_")
    if len(parts) >= 2 and parts[0].startswith("PF"):
        identifier = f"{parts[1]}_{parts[2]}" if len(parts) > 2 else parts[1]
        return parts[0], identifier
    parent = path.parent.name
    return (parent, base) if parent.startswith("PF") else ("Unknown", base)


def scan_pfam_structures(root_dir: Path, require_substring: str = "model") -> Dict[str, List[str]]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Pfam structure directory does not exist: {root_dir}")

    family_map: Dict[str, List[str]] = {}
    for root, _, files in os.walk(root_dir):
        family_id = Path(root).name
        if not family_id.startswith("PF"):
            continue
        for filename in files:
            if not filename.endswith(".pdb"):
                continue
            if require_substring and require_substring not in filename:
                continue
            full_path = str(Path(root) / filename)
            family_map.setdefault(family_id, []).append(full_path)

    return {family: paths for family, paths in family_map.items() if paths}


def get_existing_counts(csv_path: Path, bins: Sequence[int], coverage_threshold: float) -> Dict[int, int]:
    counts = {int(k): 0 for k in bins}
    if not csv_path.exists():
        return counts
    try:
        df = pd.read_csv(csv_path, usecols=["Aligned_Atoms", "Coverage"])
        df["Aligned_Atoms"] = pd.to_numeric(df["Aligned_Atoms"], errors="coerce")
        df["Coverage"] = pd.to_numeric(df["Coverage"], errors="coerce")
        valid = df[df["Coverage"] >= coverage_threshold]
        observed = valid["Aligned_Atoms"].value_counts().to_dict()
        for k in counts:
            counts[k] = int(observed.get(k, 0))
    except Exception as exc:
        print(f"Warning: could not read existing output for resume mode: {exc}")
    return counts


def get_first_chain_ca_residues(cmd, object_name: str) -> Tuple[Optional[str], List[int]]:
    chains = cmd.get_chains(object_name)
    if not chains:
        return None, []
    chain = chains[0]
    model = cmd.get_model(f"{object_name} and chain {chain} and name CA")
    residues = [int(atom.resi) for atom in model.atom]
    residues = sorted(set(residues))
    return chain, residues


def choose_window(residue_ids: Sequence[int], input_len: int) -> Optional[Tuple[int, int]]:
    if len(residue_ids) < input_len:
        return None
    start_idx = random.randint(0, len(residue_ids) - input_len)
    return int(residue_ids[start_idx]), int(residue_ids[start_idx + input_len - 1])


def worker_loop(
    family_map: Dict[str, List[str]],
    result_queue: mp.Queue,
    bins: Sequence[int],
    input_window_min: int,
    input_window_max: int,
    coverage_threshold: float,
    seed: Optional[int],
) -> None:
    import pymol
    from pymol import cmd

    if seed is not None:
        random.seed(seed + os.getpid() + int(time.time()))

    pymol.finish_launching(["pymol", "-c", "-q"])
    families = list(family_map.keys())
    wanted = set(int(x) for x in bins)

    while True:
        try:
            input_len = random.randint(input_window_min, input_window_max)
            fam_a, fam_b = random.sample(families, 2)
            path_a = random.choice(family_map[fam_a])
            path_b = random.choice(family_map[fam_b])

            cmd.reinitialize()
            cmd.load(path_a, "obj_a")
            chain_a, residues_a = get_first_chain_ca_residues(cmd, "obj_a")
            if chain_a is None:
                continue
            win_a = choose_window(residues_a, input_len)
            if win_a is None:
                continue

            cmd.load(path_b, "obj_b")
            chain_b, residues_b = get_first_chain_ca_residues(cmd, "obj_b")
            if chain_b is None:
                continue
            win_b = choose_window(residues_b, input_len)
            if win_b is None:
                continue

            start_a, end_a = win_a
            start_b, end_b = win_b
            sel_a = f"obj_a and chain {chain_a} and resi {start_a}-{end_a} and name CA"
            sel_b = f"obj_b and chain {chain_b} and resi {start_b}-{end_b} and name CA"

            if cmd.count_atoms(sel_a) != input_len or cmd.count_atoms(sel_b) != input_len:
                continue

            alignment = cmd.super(sel_a, sel_b, object="aln")
            rmsd = float(alignment[0])
            aligned_atoms = int(alignment[1])
            if aligned_atoms not in wanted:
                continue

            coverage = aligned_atoms / float(input_len)
            if coverage < coverage_threshold:
                continue

            family_a, pdb_a = parse_pdb_info(path_a)
            family_b, pdb_b = parse_pdb_info(path_b)
            result_queue.put({
                "Family_A": family_a,
                "PDB_A": pdb_a,
                "Range_A": f"{start_a}-{end_a}",
                "Family_B": family_b,
                "PDB_B": pdb_b,
                "Range_B": f"{start_b}-{end_b}",
                "RMSD": round(rmsd, 4),
                "Aligned_Atoms": aligned_atoms,
                "Target_Len": input_len,
                "Coverage": round(coverage, 4),
            })
        except Exception:
            continue


def ensure_output_header(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not output_csv.exists():
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()


def run(args: argparse.Namespace) -> None:
    family_map = scan_pfam_structures(args.pfam_dir, args.require_substring)
    if len(family_map) < 2:
        raise RuntimeError("At least two Pfam families with structure files are required.")

    target_counts = {int(k): int(args.per_length_quota) for k in args.aligned_lengths}
    current_counts = get_existing_counts(args.output_csv, args.aligned_lengths, args.coverage_threshold)
    remaining = sum(max(0, target_counts[k] - current_counts.get(k, 0)) for k in target_counts)

    print(f"Pfam families with structures: {len(family_map)}")
    print(f"Current counts: {current_counts}")
    print(f"Target counts: {target_counts}")
    print(f"Remaining accepted pairs required: {remaining}")

    if remaining == 0:
        print("Target quotas are already satisfied.")
        return

    ensure_output_header(args.output_csv)

    queue: mp.Queue = mp.Queue(maxsize=args.workers * 20)
    processes: List[mp.Process] = []
    for worker_id in range(args.workers):
        seed = None if args.seed is None else args.seed + worker_id * 100000
        process = mp.Process(
            target=worker_loop,
            args=(
                family_map,
                queue,
                args.aligned_lengths,
                args.input_window_min,
                args.input_window_max,
                args.coverage_threshold,
                seed,
            ),
        )
        process.daemon = True
        process.start()
        processes.append(process)

    try:
        with args.output_csv.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            with tqdm(total=remaining, desc="Accepted random controls") as pbar:
                while remaining > 0:
                    row = queue.get()
                    aligned_len = int(row["Aligned_Atoms"])
                    if current_counts.get(aligned_len, 0) >= target_counts.get(aligned_len, 0):
                        continue
                    writer.writerow(row)
                    current_counts[aligned_len] += 1
                    remaining -= 1
                    pbar.update(1)
                    if remaining % args.flush_every == 0:
                        handle.flush()
    except KeyboardInterrupt:
        print("Interrupted; terminating workers.")
    finally:
        for process in processes:
            process.terminate()
            process.join()

    print(f"Final counts: {current_counts}")
    print(f"Output written to: {args.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample length-stratified random Pfam-Pfam structural controls.")
    parser.add_argument("--pfam-dir", type=Path, required=True, help="Root directory containing Pfam family structure folders.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV for accepted random controls.")
    parser.add_argument("--aligned-lengths", type=parse_length_bins, default=parse_length_bins("8,9,10,11,12,13"), help="Comma-separated accepted aligned C-alpha lengths.")
    parser.add_argument("--per-length-quota", type=int, default=25000, help="Number of accepted random pairs to collect for each aligned length.")
    parser.add_argument("--coverage-threshold", type=float, default=0.8, help="Minimum aligned_atoms/input_window_length coverage.")
    parser.add_argument("--input-window-min", type=int, default=8, help="Minimum random input window length.")
    parser.add_argument("--input-window-max", type=int, default=20, help="Maximum random input window length.")
    parser.add_argument("--require-substring", default="model", help="Only include Pfam PDB filenames containing this substring. Use an empty string to include all PDB files.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="Number of PyMOL worker processes.")
    parser.add_argument("--flush-every", type=int, default=50, help="Flush output after this many remaining accepted records.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed base.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
