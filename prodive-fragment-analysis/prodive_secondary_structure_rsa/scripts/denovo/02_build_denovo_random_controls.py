#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import re
import random
import warnings
import argparse
from typing import Dict, List, Tuple, Optional

import pandas as pd
from Bio import SeqIO, BiopythonWarning
from Bio.PDB import PDBParser, DSSP
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= Configuration =================
INPUT_POSITIVE_CSV = ""
PDB_DIR = ""
FASTA_PATH = ""
OUTPUT_CONTROL_CSV = ""

NUM_WORKERS = 40
NEG_PER_POS = 30
RANDOM_SEED = 20260129

MIN_FOUND_RATIO = 0.8
EXCLUDE_OVERLAP_WITH_ALL_POS = True
MAX_TRIES_PER_CONTROL = 200

# Number of rows buffered before each flush.
WRITE_BUFFER = 500

# Failure log. Group-level failures are recorded without stopping the full run.
FAIL_LOG = ""

SEG_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def parse_segment(seg: str) -> Tuple[int, int]:
    m = SEG_RE.match(str(seg))
    if not m:
        raise ValueError(f"Bad segment format: {seg}")
    a = int(m.group(1))
    b = int(m.group(2))
    if a > b:
        a, b = b, a
    return a, b


def interval_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def safe_float(x) -> Optional[float]:
    """
    Robustly parse DSSP relative accessibility. Return None when unavailable.
    """
    if x is None:
        return None
    try:
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s.upper() == "NA":
                return None
            v = float(s)
        else:
            v = float(x)

        # NaN
        if v != v:
            return None
        return v
    except Exception:
        return None


def choose_random_segment(
    seq_len: int,
    seg_len: int,
    forbidden: List[Tuple[int, int]],
    rng: random.Random,
    max_tries: int = 200
) -> Tuple[int, int]:
    seg_len = max(1, min(seg_len, seq_len))
    if seq_len - seg_len + 1 <= 0:
        return 1, seq_len

    for _ in range(max_tries):
        s = rng.randint(1, seq_len - seg_len + 1)
        e = s + seg_len - 1
        cand = (s, e)
        if forbidden:
            ok = True
            for f in forbidden:
                if interval_overlap(cand, f):
                    ok = False
                    break
            if not ok:
                continue
        return cand

    # Fallback: avoid only the exact same interval.
    for _ in range(max_tries):
        s = rng.randint(1, seq_len - seg_len + 1)
        e = s + seg_len - 1
        cand = (s, e)
        if cand not in forbidden:
            return cand

    return 1, seg_len


def get_chain_mapping_and_len(fasta_path: str) -> Tuple[Dict[str, str], Dict[str, int]]:
    print("[INFO] Loading chain mapping and sequence lengths from FASTA...")
    chain_map = {}
    len_map = {}
    if not os.path.exists(fasta_path):
        return chain_map, len_map

    for record in SeqIO.parse(fasta_path, "fasta"):
        desc = record.description
        parts = desc.split('|')
        denovo_id = parts[0].strip() if parts else record.id.strip()
        len_map[denovo_id] = len(str(record.seq))

        if len(parts) >= 2:
            c_part = parts[1].replace("Chains", "").replace("Chain", "").strip()
            first_c = c_part.split(',')[0].strip().split('[')[0].strip()
            chain_map[denovo_id] = first_c
        else:
            chain_map[denovo_id] = "A"

    return chain_map, len_map


def compute_features_for_group(args):
    """
    Worker function for all rows from one de novo ID.
    DSSP is computed once per ID and then reused for all segments.
    """
    warnings.simplefilter('ignore', BiopythonWarning)
    warnings.filterwarnings("ignore")

    denovo_id, rows, chain_id, pdb_path = args

    def fail_row(r, status):
        return {**r, **{
            'SS_Sequence': '',
            'SS_Class': 'Unknown',
            'Avg_RSA': float('nan'),
            'Location': 'Unknown',
            'Status': status,
            'Found_Ratio': 0.0,
        }}

    if not os.path.exists(pdb_path):
        return [fail_row(r, "PDB_Not_Found") for r in rows]

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('struct', pdb_path)
        model = structure[0]
        dssp = DSSP(model, pdb_path)  # compute once per de novo structure
    except Exception as e:
        return [fail_row(r, f"DSSP_Fail: {str(e)}") for r in rows]

    out = []

    for r in rows:
        seg = r.get("Main_Segment", "")
        try:
            start_res, end_res = parse_segment(seg)
        except Exception:
            out.append(fail_row(r, "Format_Error"))
            continue

        ss_list = []
        rsa_list: List[float] = []
        found = 0
        total = end_res - start_res + 1

        for i in range(start_res, end_res + 1):
            key = (chain_id, (' ', i, ' '))
            if key in dssp:
                ss_code = dssp[key][2]
                rel_rsa_raw = dssp[key][3]

                ss_list.append(ss_code if ss_code != " " else "-")
                found += 1

                rel_rsa = safe_float(rel_rsa_raw)
                if rel_rsa is not None:
                    rsa_list.append(rel_rsa)
            else:
                ss_list.append("-")

        ss_seq = "".join(ss_list)
        found_ratio = (found / total) if total > 0 else 0.0

        # Mapping failure is determined by DSSP key coverage.
        if found == 0 or found_ratio < MIN_FOUND_RATIO:
            out.append({**r, **{
                'SS_Sequence': ss_seq,
                'SS_Class': 'Unknown',
                'Avg_RSA': float('nan'),
                'Location': 'Unknown',
                'Status': 'Segment_Not_Mapped',
                'Found_Ratio': round(found_ratio, 4),
            }})
            continue

        # Secondary-structure classification does not depend on RSA.
        length = len(ss_seq)
        h_cnt = sum(1 for c in ss_seq if c in ['H', 'G', 'I'])
        s_cnt = sum(1 for c in ss_seq if c in ['E', 'B'])

        if length > 0:
            if h_cnt / length > 0.5:
                ss_class = "Helix_Dominant"
            elif s_cnt / length > 0.5:
                ss_class = "Sheet_Dominant"
            elif (h_cnt + s_cnt) / length < 0.4:
                ss_class = "Coil/Loop_Dominant"
            else:
                ss_class = "Mixed"
        else:
            ss_class = "Unknown"

        # Do not coerce all-NA RSA segments to zero.
        if len(rsa_list) == 0:
            avg_rsa = float('nan')
            loc = "Unknown"
            status = "OK_RSA_NA"
        else:
            avg_rsa = sum(rsa_list) / len(rsa_list)
            if avg_rsa < 0.20:
                loc = "Buried (Internal)"
            elif avg_rsa < 0.50:
                loc = "Intermediate"
            else:
                loc = "Exposed (Surface)"
            status = "OK"

        out.append({**r, **{
            'SS_Sequence': ss_seq,
            'SS_Class': ss_class,
            'Avg_RSA': round(avg_rsa, 4) if avg_rsa == avg_rsa else float('nan'),
            'Location': loc,
            'Status': status,
            'Found_Ratio': round(found_ratio, 4),
        }})

    return out


def load_processed_keys_fast(path: str) -> set:
    """
    Resume a large output file by reading only the unique_key column when possible.
    """
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, usecols=["unique_key"], dtype=str)
        return set(df["unique_key"].dropna().values)
    except Exception:
        # Fallback: line-by-line CSV scan.
        keys = set()
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or "unique_key" not in header:
                return set()
            idx = header.index("unique_key")
            for row in reader:
                if idx < len(row) and row[idx]:
                    keys.add(row[idx])
        return keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate same-length same-chain random controls for de novo-Pfam SS/RSA analysis."
    )
    parser.add_argument("--input-positive-csv", required=True, help="Input de novo-Pfam positive CSV.")
    parser.add_argument("--pdb-dir", required=True, help="Directory containing de novo PDB files.")
    parser.add_argument("--fasta", required=True, help="FASTA used to map de novo IDs to chains and sequence lengths.")
    parser.add_argument("--output-control-csv", required=True, help="Output random-control CSV. The output column order is unchanged.")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes.")
    parser.add_argument("--neg-per-pos", type=int, default=NEG_PER_POS, help="Number of random controls per positive row.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--min-found-ratio", type=float, default=MIN_FOUND_RATIO, help="Minimum DSSP residue coverage required for a segment.")
    parser.add_argument("--max-tries-per-control", type=int, default=MAX_TRIES_PER_CONTROL, help="Maximum random-sampling attempts per control window.")
    parser.add_argument("--write-buffer", type=int, default=WRITE_BUFFER, help="Number of output rows buffered before writing.")
    parser.add_argument("--fail-log", default=None, help="Failure log path. Default: <output-control-csv>.failures.log")
    parser.add_argument("--allow-overlap-with-positive", action="store_true", help="Allow random windows to overlap positive intervals.")
    return parser.parse_args()

def apply_args(args):
    global INPUT_POSITIVE_CSV, PDB_DIR, FASTA_PATH, OUTPUT_CONTROL_CSV
    global NUM_WORKERS, NEG_PER_POS, RANDOM_SEED, MIN_FOUND_RATIO, MAX_TRIES_PER_CONTROL
    global WRITE_BUFFER, FAIL_LOG, EXCLUDE_OVERLAP_WITH_ALL_POS
    INPUT_POSITIVE_CSV = args.input_positive_csv
    PDB_DIR = args.pdb_dir
    FASTA_PATH = args.fasta
    OUTPUT_CONTROL_CSV = args.output_control_csv
    NUM_WORKERS = args.workers
    NEG_PER_POS = args.neg_per_pos
    RANDOM_SEED = args.random_seed
    MIN_FOUND_RATIO = args.min_found_ratio
    MAX_TRIES_PER_CONTROL = args.max_tries_per_control
    WRITE_BUFFER = args.write_buffer
    FAIL_LOG = args.fail_log or f"{OUTPUT_CONTROL_CSV}.failures.log"
    EXCLUDE_OVERLAP_WITH_ALL_POS = not args.allow_overlap_with_positive

def main():
    args = parse_args()
    apply_args(args)

    rng = random.Random(RANDOM_SEED)

    if not os.path.exists(INPUT_POSITIVE_CSV):
        print(f"[FATAL] Input positive CSV not found: {INPUT_POSITIVE_CSV}")
        return

    print(f"[INFO] Reading positive CSV: {INPUT_POSITIVE_CSV}")
    df_pos = pd.read_csv(INPUT_POSITIVE_CSV)

    for need in ["Main_HMM", "Main_Segment"]:
        if need not in df_pos.columns:
            print(f"[FATAL] Missing required column: {need}")
            print(f"Available columns: {list(df_pos.columns)}")
            return

    chain_map, len_map = get_chain_mapping_and_len(FASTA_PATH)

    # Precompute all positive intervals per de novo ID to avoid overlap.
    denovo_pos_intervals: Dict[str, List[Tuple[int, int]]] = {}
    for _, row in df_pos.iterrows():
        denovo_id = str(row["Main_HMM"])
        try:
            a, b = parse_segment(row.get("Main_Segment", ""))
        except Exception:
            continue
        denovo_pos_intervals.setdefault(denovo_id, []).append((a, b))

    # Generate random-control rows.
    print(f"[INFO] Generate random controls NEG_PER_POS={NEG_PER_POS} seed={RANDOM_SEED}")
    controls = []

    for idx, row in df_pos.iterrows():
        denovo_id = str(row["Main_HMM"])
        seg_str = str(row["Main_Segment"])

        try:
            a, b = parse_segment(seg_str)
            seg_len = b - a + 1
        except Exception:
            continue

        seq_len = len_map.get(denovo_id, None)
        if not seq_len:
            continue

        forbidden = denovo_pos_intervals.get(denovo_id, []) if EXCLUDE_OVERLAP_WITH_ALL_POS else []

        for k in range(NEG_PER_POS):
            rs, re_ = choose_random_segment(seq_len, seg_len, forbidden, rng, MAX_TRIES_PER_CONTROL)

            new_row = row.to_dict()
            sub_hmm = str(row["Sub_HMM"]) if "Sub_HMM" in df_pos.columns else ""
            base_file = str(row["File"]) if "File" in df_pos.columns else f"{denovo_id}_{sub_hmm}"
            new_row["File"] = f"{base_file}__RND{k+1}__idx{idx}"
            new_row["Main_Segment"] = f"{rs}-{re_}"
            new_row["Is_Positive"] = 0
            new_row["Control_Of_Index"] = int(idx)
            new_row["Control_Mode"] = "RandomSegment_SameLen_SameDenovo_SamePfam"
            new_row["denovo_id_extracted"] = denovo_id

            controls.append(new_row)

    if not controls:
        print("[FATAL] No control rows were generated. FASTA may be missing Main_HMM de novo IDs.")
        return

    df_ctrl = pd.DataFrame(controls)

    # unique_key for resumable execution.
    df_ctrl["unique_key"] = (
        df_ctrl["File"].astype(str) + "_" +
        df_ctrl["Main_Segment"].astype(str) + "_" +
        df_ctrl["Control_Of_Index"].astype(str)
    )

    processed_keys = load_processed_keys_fast(OUTPUT_CONTROL_CSV)
    write_header = not os.path.exists(OUTPUT_CONTROL_CSV) or len(processed_keys) == 0

    if processed_keys:
        df_to_process = df_ctrl[~df_ctrl["unique_key"].isin(processed_keys)].copy()
        print(f"[INFO] Existing output found. Completed rows={len(processed_keys)}; remaining rows={len(df_to_process)}")
    else:
        df_to_process = df_ctrl.copy()
        print(f"[INFO] Full control run rows: {len(df_to_process)}")

    if len(df_to_process) == 0:
        print("[DONE] All control rows are already completed.")
        return

    # Output header.
    feature_cols = ["SS_Sequence", "SS_Class", "Avg_RSA", "Location", "Status", "Found_Ratio"]
    final_headers = list(df_ctrl.columns) + feature_cols

    out_dir = os.path.dirname(OUTPUT_CONTROL_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if write_header:
        with open(OUTPUT_CONTROL_CSV, "w", newline="") as f:
            csv.writer(f).writerow(final_headers)

    # Group by de novo ID for parallel processing.
    groups: Dict[str, List[dict]] = {}
    for r in df_to_process.to_dict("records"):
        denovo_id = r.get("denovo_id_extracted", "")
        groups.setdefault(denovo_id, []).append(r)

    tasks = []
    for denovo_id, rows in groups.items():
        chain_id = chain_map.get(denovo_id, "A")
        pdb_code = denovo_id.split("_")[0]
        pdb_path = os.path.join(PDB_DIR, f"{pdb_code}.pdb")
        tasks.append((denovo_id, rows, chain_id, pdb_path))

    print(f"[INFO] Parallel de novo groups: {len(tasks)}; total segments: {len(df_to_process)}; workers={NUM_WORKERS}")

    # Multiprocessing execution.
    buffer = []
    fail_dir = os.path.dirname(FAIL_LOG)
    if fail_dir:
        os.makedirs(fail_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_denovo = {}
        for t in tasks:
            fu = executor.submit(compute_features_for_group, t)
            future_to_denovo[fu] = t[0]

        with tqdm(total=len(df_to_process), unit="seg", desc="Grouped DSSP (MP)") as pbar:
            for fu in as_completed(future_to_denovo):
                denovo_id = future_to_denovo[fu]
                try:
                    res_rows = fu.result()  # list of dict
                except Exception as e:
                    # Continue the full run and mark all rows from this group as Crash.
                    with open(FAIL_LOG, "a") as lf:
                        lf.write(f"{denovo_id}\tCrash\t{str(e)}\n")
                    res_rows = []
                    for r in groups.get(denovo_id, []):
                        res_rows.append({**r, **{
                            'SS_Sequence': '',
                            'SS_Class': 'Unknown',
                            'Avg_RSA': float('nan'),
                            'Location': 'Unknown',
                            'Status': f"Crash: {str(e)}",
                            'Found_Ratio': 0.0,
                        }})

                pbar.update(len(res_rows))

                buffer.extend(res_rows)
                if len(buffer) >= WRITE_BUFFER:
                    with open(OUTPUT_CONTROL_CSV, "a", newline="") as f:
                        w = csv.writer(f)
                        for rr in buffer:
                            w.writerow([rr.get(h, "") for h in final_headers])
                    buffer = []

    # flush last
    if buffer:
        with open(OUTPUT_CONTROL_CSV, "a", newline="") as f:
            w = csv.writer(f)
            for rr in buffer:
                w.writerow([rr.get(h, "") for h in final_headers])

    print("\n[DONE] All control tasks completed.")
    print(f"[OUT] {OUTPUT_CONTROL_CSV}")
    print(f"[FAIL_LOG] {FAIL_LOG}")


if __name__ == "__main__":
    main()
