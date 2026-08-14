#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a packed fixed-length HHM fragment database from Pfam HHM files.

This is a single-file version. It no longer depends on:
    tools.py
    pfam_selfhmm_read_hhm.py
    splits_of_hhm.py

Default output:
    pi_all.float32.bin
    A_all.float32.bin
    B_all.float32.bin
    index.csv
    metadata.json
    summary.json
    failed_files.csv

Optional compatibility output:
    --save_small_npy writes old-style pi_<id>.npy / A_<id>.npy / B_<id>.npy
    under output_dir/small_npy/.

Example:
    python3 01_pack_fixed_length_hhm_fragments.py \
      --input_dir /path/to/PfamA_seed \
      --output_dir /path/to/packed_db \
      --mapping /path/to/pfam_mapping_seed_new.txt \
      --fragment 6 \
      --jobs 40 \
      --overwrite
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import json
import os
import shutil
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ==============================================================================
# Inlined tools.py
# ==============================================================================


def get_hmm_length(hmm_file_path: str) -> str:
    """Return the HHM length from the LENG line."""
    with open(hmm_file_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.strip()
            if line.startswith("LENG"):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
    raise ValueError(f"No LENG line found in {hmm_file_path}")


def get_hmm_number(hmm_file_path: str) -> int:
    """Return 1-based line number of the HMM line."""
    with open(hmm_file_path, "r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if line.startswith("HMM  "):
                return line_number
    raise ValueError(f"No HMM line found in {hmm_file_path}")


def mkdir_hmm(folder_path: str, fragment: int) -> None:
    os.makedirs(os.path.join(folder_path, str(fragment)), exist_ok=True)


def get_null_number(hmm_file_path: str) -> int:
    """Return 1-based line number of the NULL line, preserving original script convention."""
    with open(hmm_file_path, "r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if line.startswith("NULL   "):
                return line_number
    raise ValueError(f"No NULL line found in {hmm_file_path}")


# ==============================================================================
# Inlined pfam_selfhmm_read_hhm.py
# ==============================================================================


def _to_float_or_inf(x: str) -> float:
    """Mimic pandas.to_numeric(errors='coerce').fillna(np.inf)."""
    try:
        return float(x)
    except Exception:
        return float("inf")


def _tokens_to_float_list(tokens: List[str]) -> List[float]:
    return [_to_float_or_inf(x) for x in tokens]


def hmm_read(address: str):
    """
    Read one HHsuite .hhm file.

    Returns:
        transition_probability_data,
        emission_probability_m_data,
        emission_probability_i_data

    This keeps the indexing and slicing behavior of the original
    pfam_selfhmm_read_hhm.py as closely as possible.
    """
    null_line = int(get_null_number(address))

    with open(address, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    transition_probability_data = []

    # First transition row after NULL/HMM header area.
    split_line = lines[null_line + 2].split()
    transition_probability_data.append(_tokens_to_float_list(split_line)[:-3])

    # Remaining transition rows.
    for row in range(null_line + 4, len(lines), 3):
        split_line = lines[row].split()
        cleaned = _tokens_to_float_list(split_line)[:-3]
        transition_probability_data.append(cleaned)

    # Insert emission probability row from NULL line area.
    split_line = lines[null_line - 1].split()
    emission_probability_i_data = _tokens_to_float_list(split_line[1:])

    # Match emission probabilities.
    emission_probability_m_data = []
    for row in range(null_line + 3, len(lines), 3):
        value = lines[row].split()
        values = value[2:-1]
        if values != []:
            emission_probability_m_data.append(_tokens_to_float_list(values))

    return transition_probability_data, emission_probability_m_data, emission_probability_i_data


# ==============================================================================
# Inlined splits_of_hhm.py
# ==============================================================================


def p_mapping(row_A: int):
    """Build mapping matrix for transition placement."""
    mapping_ = []
    for i in range(row_A):
        if i > 0 and i < row_A - 1:
            row_B = 2 + (i - 1) * 3
            col_B = 5 + (i - 1) * 3
            new_mapping = [
                [row_B, col_B],
                [row_B, col_B - 1],
                [row_B, col_B + 1],
                [row_B + 2, col_B],
                [row_B + 2, col_B - 1],
                [row_B + 1, col_B],
                [row_B + 1, col_B + 1],
            ]
            mapping_.append(new_mapping)
    return mapping_


def split_emission(start_1: int, end_1: int, am, ai):
    """Split emission probabilities for one fixed-length HHM fragment."""
    am = np.asarray(am)
    ai = np.asarray(ai)

    M_emssion_ = am[start_1:end_1 - 1]
    I_emission = np.power(2, -ai / 1000)
    I_emission = np.append(I_emission, 0).tolist()

    D_emssion = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]

    M_emssion = np.power(2, -M_emssion_ / 1000).tolist()

    emssion_matrix = []
    emssion_matrix.append(I_emission)

    for i in range(end_1 - start_1 - 1):
        row = M_emssion[i] + [0.0]
        emssion_matrix.append(row)
        emssion_matrix.append(D_emssion)
        emssion_matrix.append(I_emission)

    return emssion_matrix


def split_transition_and_emission(start_1: int, end_1: int, at, am, ai):
    """
    Split transition and emission for one fragment.

    Original convention:
        start_1 corresponds to M(start_1 + 1)
        end_1 corresponds to M(end_1 - 1)
        effective fragment length = end_1 - start_1 - 1
    """
    at = np.array(at)
    am = np.array(am)
    ai = np.array(ai)

    A_ = at[start_1:end_1]
    filter_matrix = np.power(2, -A_ / 1000)
    len_filter_matrix = len(filter_matrix) - 1

    start_matrix_1 = np.zeros((1, (end_1 - start_1 - 1) * 3 + 1))
    start_matrix_1[0, 0] = 0
    start_matrix_1[0, 1] = 1
    start_matrix_1[0, 2] = 0

    # Original special start-row normalization logic.
    total = filter_matrix[0, 4] + filter_matrix[0, 3]

    if total > 0:
        if filter_matrix[0, 3] == 0:
            filter_matrix[0, 3] = 0.01
            total = filter_matrix[0, 4] + filter_matrix[0, 3]
        filter_matrix[0] = [
            filter_matrix[0, 4] / total,
            filter_matrix[0, 3] / total,
            0, 0, 0, 0, 0,
        ]
    else:
        filter_matrix[0] = [0, 0, 0, 0, 0, 0, 0]

    trans_matrix_1 = np.zeros((1 + 3 * len_filter_matrix, 1 + 3 * len_filter_matrix))

    mapping = p_mapping(len(filter_matrix))

    for i in range(filter_matrix.shape[0] - 2):
        for j in range(filter_matrix.shape[1]):
            row, col = mapping[i][j]
            trans_matrix_1[row - 1, col - 1] = filter_matrix[i + 1, j]

    for i in range(2):
        trans_matrix_1[0, i] = filter_matrix[0, i]

    trans_matrix_1[2 + 3 * len_filter_matrix - 4, 2 + 3 * len_filter_matrix - 2] = filter_matrix[len_filter_matrix][1]
    trans_matrix_1[2 + 3 * len_filter_matrix - 2, 2 + 3 * len_filter_matrix - 2] = filter_matrix[len_filter_matrix][4]

    emssion_matrix_1 = split_emission(start_1, end_1, am, ai)

    start_matrix_1 = np.array(start_matrix_1)
    trans_matrix_1 = np.array(trans_matrix_1)
    emssion_matrix_1 = np.array(emssion_matrix_1)

    return start_matrix_1, trans_matrix_1, emssion_matrix_1


# ==============================================================================
# Mapping and scanning
# ==============================================================================


def load_mapping(file_path: str) -> Dict[str, int]:
    """
    Load mapping file.

    Expected format:
        PF00001:0
        PF00002:1

    Keys may be either full file names or PFxxxxx stems. Both are supported later.
    """
    mapping: Dict[str, int] = {}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mapping file not found: {file_path}")

    print(f"Loading mapping file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            try:
                mapping[key] = int(val)
            except ValueError:
                print(f"[WARN] bad mapping value at line {line_no}: {line}")

    if not mapping:
        raise RuntimeError(f"No valid mapping records loaded from {file_path}")

    print(f"Loaded mapping records: {len(mapping):,}")
    return mapping


def get_family_id_from_path(hmm_file_path: str, pfam_mapping: Dict[str, int]) -> Optional[int]:
    """
    Return numeric family id from mapping.

    The function first tries the complete filename, then the stem.
    Example:
        /xxx/PF00001/PF00001.hhm -> filename PF00001.hhm, stem PF00001
    """
    path_obj = Path(hmm_file_path)
    filename = path_obj.name
    stem = path_obj.stem

    if filename in pfam_mapping:
        return pfam_mapping[filename]

    if stem in pfam_mapping:
        return pfam_mapping[stem]

    return None


def scan_hhm_files(input_dir: str) -> List[str]:
    """
    Scan HHM files using the same selection rule as the old script:
        name matches PF[0-9]*.hhm
        and filename does not contain '_'
    """
    print("Scanning .hhm files...")

    raw_files = glob.glob(os.path.join(input_dir, "**", "*.hhm"), recursive=True)
    selected: List[str] = []

    for file_path in raw_files:
        filename = os.path.basename(file_path)
        if fnmatch.fnmatch(filename, "PF[0-9]*.hhm") and "_" not in filename:
            selected.append(file_path)

    selected = sorted(selected)

    print(f"Found selected HHM files: {len(selected):,} / raw scanned: {len(raw_files):,}")
    return selected


# ==============================================================================
# Per-family processing
# ==============================================================================


def build_fragment_arrays(hmm_file_path: str, fragment_value: int):
    """
    Read one HHM and split it into fixed-length profile-HMM fragments.

    Returns:
        hmm_length, arr_start, arr_trans, arr_emit

    For fragment=6, expected shapes are normally:
        arr_start: (nfrag, 1, 19)
        arr_trans: (nfrag, 19, 19)
        arr_emit : (nfrag, 19, 21)

    where nfrag = hmm_length - fragment + 1.
    """
    length_hmm = int(get_hmm_length(hmm_file_path))

    if length_hmm < fragment_value:
        raise ValueError(f"HMM length {length_hmm} < fragment {fragment_value}")

    transition_data, emit_m_data, emit_i_data = hmm_read(hmm_file_path)

    profile_start = []
    profile_trans = []
    profile_emit = []

    for start_idx in range(0, length_hmm - fragment_value + 1):
        # Preserve the binary layout convention used by the packed-database reader.
        # For fragment=k, end_idx = start_idx + k + 1,
        # and the effective fragment length is end_idx - start_idx - 1 = k.
        end_idx = start_idx + fragment_value + 1

        p_start, p_trans, p_emit = split_transition_and_emission(
            start_idx,
            end_idx,
            transition_data,
            emit_m_data,
            emit_i_data,
        )

        profile_start.append(p_start)
        profile_trans.append(p_trans)
        profile_emit.append(p_emit)

    arr_start = np.asarray(profile_start, dtype=np.float32, order="C")
    arr_trans = np.asarray(profile_trans, dtype=np.float32, order="C")
    arr_emit = np.asarray(profile_emit, dtype=np.float32, order="C")

    return length_hmm, arr_start, arr_trans, arr_emit


def process_single_file(
    hmm_file_path: str,
    fragment_value: int,
    pfam_mapping: Dict[str, int],
):
    """
    Worker function.

    Return a dict. Status values:
        ok
        ignored
        error
    """
    try:
        family_id = get_family_id_from_path(hmm_file_path, pfam_mapping)

        if family_id is None:
            return {
                "status": "ignored",
                "reason": "no_mapping",
                "file": hmm_file_path,
            }

        length_hmm = int(get_hmm_length(hmm_file_path))
        if length_hmm < fragment_value:
            return {
                "status": "ignored",
                "reason": "too_short",
                "file": hmm_file_path,
                "family_id": family_id,
                "hmm_length": length_hmm,
            }

        length_hmm, arr_start, arr_trans, arr_emit = build_fragment_arrays(
            hmm_file_path=hmm_file_path,
            fragment_value=fragment_value,
        )

        nfrag = int(arr_start.shape[0])

        if arr_trans.shape[0] != nfrag or arr_emit.shape[0] != nfrag:
            raise RuntimeError(
                f"inconsistent nfrag: pi={arr_start.shape}, A={arr_trans.shape}, B={arr_emit.shape}"
            )

        return {
            "status": "ok",
            "file": hmm_file_path,
            "family_id": int(family_id),
            "pfam": Path(hmm_file_path).stem,
            "hmm_length": int(length_hmm),
            "fragment": int(fragment_value),
            "nfrag": int(nfrag),
            "pi": np.ascontiguousarray(arr_start, dtype=np.float32),
            "A": np.ascontiguousarray(arr_trans, dtype=np.float32),
            "B": np.ascontiguousarray(arr_emit, dtype=np.float32),
        }

    except Exception as e:
        return {
            "status": "error",
            "file": hmm_file_path,
            "error": repr(e),
        }


# ==============================================================================
# Packed writer
# ==============================================================================


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not overwrite:
            raise RuntimeError(
                f"Output directory already exists and is not empty: {output_dir}\n"
                f"Use --overwrite if you want to replace it."
            )
        if overwrite:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def write_manifest(output_dir: Path, manifest: dict) -> None:
    path = output_dir / "metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def write_small_npy(result: dict, small_dir: Path) -> None:
    """
    Optional compatibility output with the old C++ loader.
    """
    family_id = result["family_id"]
    small_dir.mkdir(parents=True, exist_ok=True)

    np.save(small_dir / f"pi_{family_id}.npy", result["pi"])
    np.save(small_dir / f"A_{family_id}.npy", result["A"])
    np.save(small_dir / f"B_{family_id}.npy", result["B"])


def append_float32_array(fh, arr: np.ndarray, current_float_offset: int) -> Tuple[int, int]:
    """
    Append a float32 array to an open binary file.

    Returns:
        offset_float, count_float
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    offset = int(current_float_offset)
    count = int(arr.size)
    fh.write(arr.tobytes(order="C"))
    return offset, count


def write_failed_csv(output_dir: Path, failed_rows: List[dict]) -> None:
    if not failed_rows:
        return

    path = output_dir / "failed_files.csv"
    fieldnames = sorted({k for row in failed_rows for k in row.keys()})

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_rows)


def run_packed_generation(
    input_dir: str,
    output_dir: str,
    mapping_file: str,
    fragment: int,
    jobs: int,
    overwrite: bool,
    save_small_npy: bool,
    max_files: Optional[int],
) -> None:
    output_path = Path(output_dir)
    prepare_output_dir(output_path, overwrite=overwrite)

    mapping = load_mapping(mapping_file)
    all_files = scan_hhm_files(input_dir)

    if max_files is not None and max_files > 0:
        all_files = all_files[:max_files]
        print(f"[TEST MODE] only processing first {len(all_files):,} files")

    if not all_files:
        raise RuntimeError("No HHM files found to process.")

    # Basic metadata known before processing.
    # Final metadata.json is written after all families are processed, because
    # values such as num_datasets, min_frag, max_frag, m_states, and n_obs
    # should be recorded explicitly for the C++ reader.
    base_metadata = {
        "format": "packed_hhm_fragment_database_v2",
        "script": "np_save_packed_single_metadata.py",
        "dependency_mode": "single_file_no_local_module_imports",
        "dtype": "float32",
        "byte_order": "native_little_endian_expected_on_x86_64",
        "input_dir": str(input_dir),
        "mapping_file": str(mapping_file),
        "fragment": int(fragment),
        "expected_m_states": int(3 * fragment + 1),
        "jobs": int(jobs),
        "n_input_hhm_files": int(len(all_files)),
        "files": {
            "pi": "pi_all.float32.bin",
            "A": "A_all.float32.bin",
            "B": "B_all.float32.bin",
            "index": "index.csv",
        },
        "offset_unit": "float32_element_offset_not_byte_offset",
        "notes": [
            "Arrays are flattened in C order when written to the .bin files.",
            "index.csv records float offsets and float counts for pi/A/B arrays.",
            "metadata.json stores all runtime constants needed by the C++ reader.",
            "C++ should read m_states, n_obs, num_datasets, min_frag, and max_frag from metadata.json instead of hard-coding them.",
            "This single-file script inlines tools.py, pfam_selfhmm_read_hhm.py, and splits_of_hhm.py.",
        ],
    }

    pi_bin_path = output_path / "pi_all.float32.bin"
    A_bin_path = output_path / "A_all.float32.bin"
    B_bin_path = output_path / "B_all.float32.bin"
    index_path = output_path / "index.csv"
    small_dir = output_path / "small_npy"

    worker = partial(
        process_single_file,
        fragment_value=fragment,
        pfam_mapping=mapping,
    )

    count_ok = 0
    count_ignored = 0
    count_error = 0
    failed_rows: List[dict] = []

    pi_float_offset = 0
    A_float_offset = 0
    B_float_offset = 0

    # Runtime metadata to be collected while writing the packed database.
    ok_family_ids: List[int] = []
    nfrag_values: List[int] = []
    hmm_length_values: List[int] = []
    pi_shape_set = set()
    A_shape_set = set()
    B_shape_set = set()
    m_states_set = set()
    n_obs_set = set()

    start_time = time.time()

    print("=" * 100)
    print("Packed HHM fragment database generation")
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_path}")
    print(f"Mapping   : {mapping_file}")
    print(f"Fragment  : {fragment}")
    print(f"Jobs      : {jobs}")
    print(f"Save old small npy files: {save_small_npy}")
    print("=" * 100)

    index_fields = [
        "family_id",
        "pfam",
        "hmm_file",
        "hmm_length",
        "fragment",
        "nfrag",
        "pi_offset",
        "pi_count",
        "pi_shape",
        "A_offset",
        "A_count",
        "A_shape",
        "B_offset",
        "B_count",
        "B_shape",
    ]

    pool = None
    with open(pi_bin_path, "wb") as f_pi, \
            open(A_bin_path, "wb") as f_A, \
            open(B_bin_path, "wb") as f_B, \
            open(index_path, "w", encoding="utf-8-sig", newline="") as f_index:

        index_writer = csv.DictWriter(f_index, fieldnames=index_fields)
        index_writer.writeheader()

        if jobs <= 1:
            iterator: Iterable[dict] = map(worker, all_files)
        else:
            pool = Pool(processes=jobs)
            iterator = pool.imap_unordered(worker, all_files, chunksize=1)

        if HAS_TQDM:
            iterator = tqdm(iterator, total=len(all_files), unit="file", dynamic_ncols=True)

        try:
            for result in iterator:
                status = result.get("status")

                if status == "ok":
                    count_ok += 1

                    pi_offset, pi_count = append_float32_array(
                        f_pi, result["pi"], pi_float_offset
                    )
                    A_offset, A_count = append_float32_array(
                        f_A, result["A"], A_float_offset
                    )
                    B_offset, B_count = append_float32_array(
                        f_B, result["B"], B_float_offset
                    )

                    pi_float_offset += pi_count
                    A_float_offset += A_count
                    B_float_offset += B_count

                    # Collect shape/stat metadata for C++ runtime configuration.
                    family_id = int(result["family_id"])
                    hmm_length = int(result["hmm_length"])
                    nfrag = int(result["nfrag"])
                    pi_shape = tuple(int(x) for x in result["pi"].shape)
                    A_shape = tuple(int(x) for x in result["A"].shape)
                    B_shape = tuple(int(x) for x in result["B"].shape)

                    ok_family_ids.append(family_id)
                    hmm_length_values.append(hmm_length)
                    nfrag_values.append(nfrag)
                    pi_shape_set.add(pi_shape)
                    A_shape_set.add(A_shape)
                    B_shape_set.add(B_shape)

                    if len(A_shape) != 3 or A_shape[1] != A_shape[2]:
                        raise RuntimeError(f"Bad A shape for family {family_id}: {A_shape}")
                    if len(B_shape) != 3:
                        raise RuntimeError(f"Bad B shape for family {family_id}: {B_shape}")
                    if A_shape[0] != nfrag or B_shape[0] != nfrag:
                        raise RuntimeError(
                            f"nfrag mismatch for family {family_id}: nfrag={nfrag}, A={A_shape}, B={B_shape}"
                        )

                    m_states_set.add(int(A_shape[1]))
                    n_obs_set.add(int(B_shape[2]))

                    index_writer.writerow({
                        "family_id": result["family_id"],
                        "pfam": result["pfam"],
                        "hmm_file": result["file"],
                        "hmm_length": result["hmm_length"],
                        "fragment": result["fragment"],
                        "nfrag": result["nfrag"],
                        "pi_offset": pi_offset,
                        "pi_count": pi_count,
                        "pi_shape": "x".join(str(x) for x in result["pi"].shape),
                        "A_offset": A_offset,
                        "A_count": A_count,
                        "A_shape": "x".join(str(x) for x in result["A"].shape),
                        "B_offset": B_offset,
                        "B_count": B_count,
                        "B_shape": "x".join(str(x) for x in result["B"].shape),
                    })

                    if save_small_npy:
                        write_small_npy(result, small_dir)

                    del result

                elif status == "ignored":
                    count_ignored += 1
                    failed_rows.append(result)

                else:
                    count_error += 1
                    failed_rows.append(result)

        finally:
            if pool is not None:
                pool.close()
                pool.join()

    write_failed_csv(output_path, failed_rows)

    duration = time.time() - start_time

    if count_ok == 0:
        raise RuntimeError("No family was successfully processed; packed database is empty.")

    if len(m_states_set) != 1:
        raise RuntimeError(f"Inconsistent m_states across families: {sorted(m_states_set)}")
    if len(n_obs_set) != 1:
        raise RuntimeError(f"Inconsistent n_obs across families: {sorted(n_obs_set)}")

    m_states = int(next(iter(m_states_set)))
    n_obs = int(next(iter(n_obs_set)))
    expected_m_states = int(3 * fragment + 1)
    if m_states != expected_m_states:
        raise RuntimeError(
            f"m_states mismatch: observed {m_states}, expected 3*fragment+1 = {expected_m_states}"
        )

    mapping_ids = list(mapping.values())
    mapping_min_id = int(min(mapping_ids)) if mapping_ids else 0
    mapping_max_id = int(max(mapping_ids)) if mapping_ids else -1
    num_datasets = int(mapping_max_id + 1)

    final_summary = {
        "ok_families": int(count_ok),
        "ignored_files": int(count_ignored),
        "error_files": int(count_error),
        "pi_total_float32_count": int(pi_float_offset),
        "A_total_float32_count": int(A_float_offset),
        "B_total_float32_count": int(B_float_offset),
        "pi_total_bytes": int(pi_float_offset * 4),
        "A_total_bytes": int(A_float_offset * 4),
        "B_total_bytes": int(B_float_offset * 4),
        "duration_seconds": float(duration),
    }

    final_metadata = dict(base_metadata)
    final_metadata.update({
        "status": "complete",
        "fragment": int(fragment),
        "m_states": int(m_states),
        "n_obs": int(n_obs),
        "num_datasets": int(num_datasets),
        "mapping_records": int(len(mapping)),
        "mapping_min_id": int(mapping_min_id),
        "mapping_max_id": int(mapping_max_id),
        "available_family_count": int(count_ok),
        "available_min_family_id": int(min(ok_family_ids)),
        "available_max_family_id": int(max(ok_family_ids)),
        "min_frag": int(min(nfrag_values)),
        "max_frag": int(max(nfrag_values)),
        "mean_frag": float(sum(nfrag_values) / len(nfrag_values)),
        "min_hmm_length": int(min(hmm_length_values)),
        "max_hmm_length": int(max(hmm_length_values)),
        "mean_hmm_length": float(sum(hmm_length_values) / len(hmm_length_values)),
        "pi_shape_examples": ["x".join(map(str, x)) for x in sorted(pi_shape_set)[:10]],
        "A_shape_examples": ["x".join(map(str, x)) for x in sorted(A_shape_set)[:10]],
        "B_shape_examples": ["x".join(map(str, x)) for x in sorted(B_shape_set)[:10]],
        "per_fragment_shapes": {
            "pi": [1, int(m_states)],
            "A": [int(m_states), int(m_states)],
            "B": [int(m_states), int(n_obs)],
        },
        "per_fragment_float_counts": {
            "pi": int(m_states),
            "A": int(m_states * m_states),
            "B": int(m_states * n_obs),
        },
        "index_columns": index_fields,
        "summary": final_summary,
    })

    # metadata.json is the authoritative runtime-configuration file for C++.
    # C++ should read m_states, n_obs, num_datasets, min_frag, and max_frag from here.
    write_manifest(output_path, final_metadata)

    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("DONE")
    print(f"OK families     : {count_ok:,}")
    print(f"Ignored files   : {count_ignored:,}")
    print(f"Error files     : {count_error:,}")
    print(f"pi floats       : {pi_float_offset:,}")
    print(f"A floats        : {A_float_offset:,}")
    print(f"B floats        : {B_float_offset:,}")
    print(f"Output index    : {index_path}")
    print(f"Output pi bin   : {pi_bin_path}")
    print(f"Output A bin    : {A_bin_path}")
    print(f"Output B bin    : {B_bin_path}")
    print(f"Summary         : {output_path / 'summary.json'}")
    print(f"Time            : {duration:.2f} seconds")
    print("=" * 100)


# ==============================================================================
# Main
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a packed fixed-length HHM fragment database for C++ KL computation. Single-file version."
    )

    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="Input PfamA_seed / HHM root directory.")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Output packed database directory.")
    parser.add_argument("--mapping", "-m", type=str, required=True,
                        help="Mapping file, format PFxxxxx:int_id.")
    parser.add_argument("--fragment", "-f", type=int, default=6,
                        help="Fragment length. Default: 6.")
    parser.add_argument("--jobs", "-j", type=int, default=10,
                        help="Number of worker processes. Default: 10.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output directory if it already exists.")
    parser.add_argument("--save_small_npy", action="store_true",
                        help="Also save old per-family pi_<id>.npy/A_<id>.npy/B_<id>.npy files under output/small_npy.")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Optional test mode: process only the first N selected HHM files.")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    if args.fragment <= 0:
        raise ValueError("--fragment must be positive")

    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")

    run_packed_generation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mapping_file=args.mapping,
        fragment=args.fragment,
        jobs=args.jobs,
        overwrite=args.overwrite,
        save_small_npy=args.save_small_npy,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
