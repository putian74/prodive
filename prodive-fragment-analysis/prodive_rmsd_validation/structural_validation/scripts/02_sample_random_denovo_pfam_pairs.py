#!/usr/bin/env python3
"""Generate length-stratified random de novo-Pfam structural controls.

The script samples one random C-alpha window from a Pfam structure and one from
a de novo structure, superposes them with PyMOL, and retains accepted pairs until
the target quota is reached for each aligned C-alpha length.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import random
import time
from queue import Empty
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

FIELDNAMES = [
    "Family_A", "PDB_A", "Range_A",
    "Family_B", "PDB_B", "Range_B",
    "RMSD", "Aligned_Atoms", "Target_Len", "Coverage",
]


def parse_length_bins(text: str) -> List[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one aligned-length bin is required.")
    return sorted(set(values))


def parse_pdb_info(file_path: str) -> Tuple[str, str]:
    path = Path(file_path)
    base = path.stem
    if base.startswith("PF"):
        parts = base.split("_")
        if len(parts) >= 2:
            identifier = f"{parts[1]}_{parts[2]}" if len(parts) > 2 else parts[1]
            return parts[0], identifier
    return path.parent.name, base


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
            family_map.setdefault(family_id, []).append(str(Path(root) / filename))
    return {family: paths for family, paths in family_map.items() if paths}


def scan_denovo_structures(root_dir: Path) -> List[str]:
    if not root_dir.exists():
        raise FileNotFoundError(f"de novo structure directory does not exist: {root_dir}")
    paths: List[str] = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".pdb"):
                paths.append(str(Path(root) / filename))
    return paths


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
    residues = sorted(set(int(atom.resi) for atom in model.atom))
    return chain, residues


def choose_window(residue_ids: Sequence[int], input_len: int) -> Optional[Tuple[int, int]]:
    if len(residue_ids) < input_len:
        return None
    idx = random.randint(0, len(residue_ids) - input_len)
    return int(residue_ids[idx]), int(residue_ids[idx + input_len - 1])


def worker_loop(
    pfam_map: Dict[str, List[str]],
    denovo_paths: Sequence[str],
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

    pymol.finish_launching(["pymol", "-c", "-q", "-k"])
    pfam_families = list(pfam_map.keys())
    wanted = set(int(x) for x in bins)

    while True:
        try:
            input_len = random.randint(input_window_min, input_window_max)
            family_a = random.choice(pfam_families)
            path_a = random.choice(pfam_map[family_a])
            path_b = random.choice(denovo_paths)

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

            family_a_label, pdb_a = parse_pdb_info(path_a)
            denovo_group, denovo_id = parse_pdb_info(path_b)
            result_queue.put({
                "Family_A": family_a_label,
                "PDB_A": pdb_a,
                "Range_A": f"{start_a}-{end_a}",
                "Family_B": f"DeNovo_{denovo_group}",
                "PDB_B": denovo_id,
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
    if args.workers < 1 or args.flush_every < 1 or args.per_length_quota < 1:
        raise ValueError("Workers, flush interval, and per-length quota must be positive.")
    if not 0 < args.coverage_threshold <= 1:
        raise ValueError("Coverage threshold must be in (0, 1].")
    if not 1 <= args.input_window_min <= args.input_window_max:
        raise ValueError("Require 1 <= input-window-min <= input-window-max.")
    if any(k < 1 or k > args.input_window_max for k in args.aligned_lengths):
        raise ValueError("Aligned lengths must be positive and cannot exceed input-window-max.")
    pfam_map = scan_pfam_structures(args.pfam_dir, args.require_substring)
    denovo_paths = scan_denovo_structures(args.denovo_pdb_dir)
    if not pfam_map:
        raise RuntimeError("No Pfam structures were found.")
    if not denovo_paths:
        raise RuntimeError("No de novo structures were found.")

    target_counts = {int(k): int(args.per_length_quota) for k in args.aligned_lengths}
    current_counts = get_existing_counts(args.output_csv, args.aligned_lengths, args.coverage_threshold)
    remaining = sum(max(0, target_counts[k] - current_counts.get(k, 0)) for k in target_counts)

    print(f"Pfam families with structures: {len(pfam_map)}")
    print(f"de novo structures: {len(denovo_paths)}")
    print(f"Current counts: {current_counts}")
    print(f"Target counts: {target_counts}")
    print(f"Remaining accepted pairs required: {remaining}")

    if remaining == 0:
        print("Target quotas are already satisfied.")
        return

    try:
        import pymol
    except ImportError as exc:
        raise RuntimeError("PyMOL is required for random RMSD calculation; install pymol-open-source in the active environment.") from exc
    ensure_output_header(args.output_csv)
    queue: mp.Queue = mp.Queue(maxsize=args.workers * 20)
    processes: List[mp.Process] = []
    for worker_id in range(args.workers):
        seed = None if args.seed is None else args.seed + worker_id * 100000
        process = mp.Process(
            target=worker_loop,
            args=(
                pfam_map,
                denovo_paths,
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
                    try:
                        row = queue.get(timeout=5)
                    except Empty:
                        if not any(process.is_alive() for process in processes):
                            raise RuntimeError("All random-control workers exited before the quotas were reached.")
                        continue
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
    parser = argparse.ArgumentParser(description="Sample length-stratified random de novo-Pfam structural controls.")
    parser.add_argument("--pfam-dir", type=Path, required=True, help="Root directory containing Pfam family structure folders.")
    parser.add_argument("--denovo-pdb-dir", type=Path, required=True, help="Directory containing de novo PDB structures.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV for accepted random controls.")
    parser.add_argument("--aligned-lengths", type=parse_length_bins, default=parse_length_bins("8,9,10,11,12,13"), help="Comma-separated accepted aligned C-alpha lengths.")
    parser.add_argument("--per-length-quota", type=int, default=10000, help="Number of accepted random pairs to collect for each aligned length.")
    parser.add_argument("--coverage-threshold", type=float, default=0.8, help="Minimum aligned_atoms/input_window_length coverage.")
    parser.add_argument("--input-window-min", type=int, default=8, help="Minimum random input window length.")
    parser.add_argument("--input-window-max", type=int, default=20, help="Maximum random input window length.")
    parser.add_argument("--require-substring", default="model", help="Only include Pfam PDB filenames containing this substring. Use an empty string to include all PDB files.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="Number of PyMOL worker processes.")
    parser.add_argument("--flush-every", type=int, default=10, help="Flush output after this many remaining accepted records.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed base.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
