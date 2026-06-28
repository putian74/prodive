#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import json
import shutil
import warnings
import tempfile
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import PDB, BiopythonWarning
from Bio.PDB.DSSP import DSSP
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= Configuration =================
INPUT_CSV = "<PRODIVE_DATA_ROOT>/shared/global_high_score_summary_fin.csv"
PFAM_DIR = "<PRODIVE_DATA_ROOT>/shared/PfamA_seed/"
OUTPUT_CSV = "CHANGE_ME"

NUM_WORKERS = 60

# Search and quality-control parameters
MAX_SEARCH_DEPTH = 200
MAX_LEN_DIFF = 5          # Maximum allowed difference from the HHM segment length
MIN_MEAN_PLDDT = 70.0
MAX_HEAD_TAIL_PAE = 10.0
MIN_FOUND_RATIO = 0.8      # Minimum fraction of residues that must be found in the structure

# Output control
OVERWRITE_OUTPUT = True
WRITE_CHUNK_SIZE = 1000
MAX_INFLIGHT = NUM_WORKERS * 4

# DSSP executable
DSSP_EXE = shutil.which("mkdssp") or shutil.which("dssp")

# Fixed appended output columns
EXTRA_OUTPUT_COLUMNS = [
    # Main representative sequence and interval fields
    "Main_Selected_UID",
    "Main_MSA_Start",
    "Main_MSA_End",
    "Main_Selected_Seq_Start",
    "Main_Selected_Seq_End",
    "Main_Selected_Seq_Len",
    "Main_Len_Diff",

    # Main structure, SS, and RSA fields
    "Main_PDB_Type",
    "Main_PDB_ID",
    "Main_SS_Class",
    "Main_Avg_RSA",
    "Main_Location",
    "Main_Status",
    "Main_DSSP_Mode",
    "Main_Chain_Used",

    # Sub representative sequence and interval fields
    "Sub_Selected_UID",
    "Sub_MSA_Start",
    "Sub_MSA_End",
    "Sub_Selected_Seq_Start",
    "Sub_Selected_Seq_End",
    "Sub_Selected_Seq_Len",
    "Sub_Len_Diff",

    # Sub structure, SS, and RSA fields
    "Sub_PDB_Type",
    "Sub_PDB_ID",
    "Sub_SS_Class",
    "Sub_Avg_RSA",
    "Sub_Location",
    "Sub_Status",
    "Sub_DSSP_Mode",
    "Sub_Chain_Used",
]


# ================= Helper functions =================

def extract_hmm_to_msa_map(hhm_path):
    mapping = {}
    if not os.path.exists(hhm_path):
        return {}
    try:
        with open(hhm_path, "r", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 10 and parts[1].isdigit():
                    try:
                        hmm_idx = int(parts[1])
                        msa_col_str = parts[-1]
                        if msa_col_str.isdigit():
                            mapping[hmm_idx] = int(msa_col_str)
                    except Exception:
                        continue
        return mapping
    except Exception:
        return {}


def get_candidates_from_alignment(aln_path, msa_start_col, msa_end_col, expected_len):
    """
    Extract candidate sequence intervals from a FASTA or Stockholm alignment.
    The fragment is taken from the full MSA interval, while expected_len is
    the HHM segment length rather than the raw MSA span.
    """
    if not os.path.exists(aln_path):
        return []

    seqs = {}
    current_header = None

    try:
        with open(aln_path, "r", errors="ignore") as f:
            first_line = f.readline()
            f.seek(0)
            is_fasta = first_line.startswith(">")

            if is_fasta:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        current_header = line[1:].strip()
                        seqs[current_header] = ""
                    elif current_header:
                        seqs[current_header] += line
            else:  # Stockholm
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("//"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    sid, sfrag = parts[0], parts[1]
                    seqs[sid] = seqs.get(sid, "") + sfrag
    except Exception:
        return []

    candidates = []
    idx_start = msa_start_col - 1
    idx_end = msa_end_col

    for header, seq in seqs.items():
        if len(seq) < idx_end:
            continue

        match = re.search(r"\/(\d+)-(\d+)", header)
        if not match:
            continue
        global_offset = int(match.group(1))

        uid_match = re.search(
            r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})",
            header,
        )
        uid = uid_match.group(1) if uid_match else header.split("/")[0].split()[0]

        frag_raw = seq[idx_start:idx_end]
        gaps = frag_raw.count(".") + frag_raw.count("-")
        frag_pure = frag_raw.replace(".", "").replace("-", "")
        if len(frag_pure) < 3:
            continue

        prefix = seq[:idx_start]
        local_res_before = len(prefix.replace(".", "").replace("-", ""))
        pdb_start = global_offset + local_res_before
        pdb_end = pdb_start + len(frag_pure) - 1

        candidates.append(
            {
                "uid": uid,
                "msa_start": msa_start_col,
                "msa_end": msa_end_col,
                "pdb_start": pdb_start,
                "pdb_end": pdb_end,
                "seq_len": len(frag_pure),
                "gaps": gaps,
                "len_diff": abs(len(frag_pure) - expected_len),
            }
        )

    candidates.sort(key=lambda x: (x["len_diff"], x["gaps"]))
    return candidates[:MAX_SEARCH_DEPTH]


def _first_chain_id(structure):
    model = structure[0]
    chains = list(model.get_chains())
    return chains[0].id if chains else None


def check_af_quality(pdb_path, start_res, end_res):
    try:
        parser = PDB.PDBParser(PERMISSIVE=1, QUIET=True)
        structure = parser.get_structure("X", pdb_path)
        model = structure[0]
        chain_id = "A" if "A" in model else _first_chain_id(structure)
        if chain_id is None:
            return False, 0.0

        plddt_vals = []
        for res in model[chain_id]:
            if start_res <= res.id[1] <= end_res:
                if "CA" in res:
                    plddt_vals.append(res["CA"].get_bfactor())
        if not plddt_vals:
            return False, 0.0
        mean = float(np.mean(plddt_vals))
        if mean < MIN_MEAN_PLDDT:
            return False, mean
    except Exception:
        return False, 0.0

    json_path = pdb_path.replace(".pdb", ".json")
    if not os.path.exists(json_path):
        base = pdb_path.replace(".pdb", "")
        jsons = glob.glob(f"{base}*.json")
        json_path = jsons[0] if jsons else None

    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            pae = data[0]["predicted_aligned_error"]
            idx_s = max(0, start_res - 1)
            idx_e = min(len(pae) - 1, end_res - 1)
            if pae[idx_s][idx_e] > MAX_HEAD_TAIL_PAE:
                return False, mean
        except Exception:
            pass

    return True, mean


def parse_exp_chain(filename):
    m = re.search(r"_([A-Za-z0-9]+)\.pdb$", filename)
    return m.group(1) if m else "A"


def find_best_structure_hybrid(family_dir, candidates, max_len_diff=10):
    """
    Hybrid structure search. EXP structures are preferred over AlphaFold.
    Only candidates with len_diff <= max_len_diff are searched.
    Candidates are pre-sorted by len_diff, so the loop stops after the threshold is exceeded.
    """
    for cand in candidates:
        if cand["len_diff"] > max_len_diff:
            break

        uid = cand["uid"]
        s = cand["pdb_start"]
        e = cand["pdb_end"]

        exp_files = sorted(glob.glob(os.path.join(family_dir, f"{uid}_exp_*.pdb")))
        if exp_files:
            best_pdb = exp_files[0]
            chain = parse_exp_chain(best_pdb)
            return {
                "uid": uid,
                "msa_start": cand["msa_start"],
                "msa_end": cand["msa_end"],
                "seq_start": s,
                "seq_end": e,
                "seq_len": cand["seq_len"],
                "len_diff": cand["len_diff"],
                "path": best_pdb,
                "type": "EXP",
                "chain": chain,
                "start": s,
                "end": e,
            }

        af_files = sorted(
            glob.glob(os.path.join(family_dir, f"AF-{uid}-F1-model*.pdb")),
            reverse=True,
        )
        if af_files:
            af_pdb = af_files[0]
            passed, _ = check_af_quality(af_pdb, s, e)
            if passed:
                return {
                    "uid": uid,
                    "msa_start": cand["msa_start"],
                    "msa_end": cand["msa_end"],
                    "seq_start": s,
                    "seq_end": e,
                    "seq_len": cand["seq_len"],
                    "len_diff": cand["len_diff"],
                    "path": af_pdb,
                    "type": "AF",
                    "chain": "A",
                    "start": s,
                    "end": e,
                }

    return None


# ================= Robust DSSP wrapper =================

def _call_dssp(model, pdb_path):
    if DSSP_EXE is None:
        raise RuntimeError("mkdssp/dssp not found in PATH. Please install mkdssp.")
    try:
        return DSSP(model, pdb_path, dssp=DSSP_EXE)
    except TypeError:
        return DSSP(model, pdb_path)


def build_dssp_safely(pdb_path):
    parser = PDB.PDBParser(PERMISSIVE=1, QUIET=True)
    try:
        structure = parser.get_structure("X", pdb_path)
    except Exception:
        raise RuntimeError("Biopython Parse Failed")

    model = structure[0]

    try:
        dssp = _call_dssp(model, pdb_path)
        chain_ids = {k[0] for k in dssp.keys()}
        if len(dssp) > 0:
            return dssp, chain_ids, "Original"
    except Exception:
        pass

    fd, tmp_pdb = tempfile.mkstemp(prefix="dssp_clean_", suffix=".pdb")
    os.close(fd)

    try:
        io = PDB.PDBIO()
        io.set_structure(structure)
        io.save(tmp_pdb)

        with open(tmp_pdb, "r") as f:
            lines = f.readlines()

        has_cryst1 = any(line.startswith("CRYST1") for line in lines)
        if not has_cryst1:
            dummy_cryst1 = "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n"
            with open(tmp_pdb, "w") as f:
                f.write(dummy_cryst1)
                f.writelines(lines)

        dssp = _call_dssp(model, tmp_pdb)
        chain_ids = {k[0] for k in dssp.keys()}
        return dssp, chain_ids, "Cleaned_PDBIO"

    except Exception as e:
        raise RuntimeError(f"DSSP failed on both Original and Cleaned: {e}")
    finally:
        if os.path.exists(tmp_pdb):
            try:
                os.remove(tmp_pdb)
            except Exception:
                pass


def calculate_single_structure_features(struct_info):
    if not struct_info:
        return {
            "PDB_Type": "",
            "PDB_ID": "",
            "SS_Class": "Struct_Not_Found",
            "Avg_RSA": 0.0,
            "Status": "No_Struct",
            "Location": "Unknown",
            "DSSP_Mode": "NA",
            "Chain_Used": "NA",
        }

    pdb_path = struct_info["path"]
    chain_id = str(struct_info["chain"])
    start_res = int(struct_info["start"])
    end_res = int(struct_info["end"])

    warnings.simplefilter("ignore", BiopythonWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")

    try:
        dssp, chain_ids, mode = build_dssp_safely(pdb_path)
    except Exception as e:
        return {
            "PDB_Type": struct_info.get("type", ""),
            "PDB_ID": os.path.basename(struct_info["path"]) if struct_info.get("path") else "",
            "SS_Class": "Error",
            "Avg_RSA": 0.0,
            "Status": f"DSSP_Fail_{e}",
            "Location": "Error",
            "DSSP_Mode": "FAIL",
            "Chain_Used": "NA",
        }

    chain_used = chain_id
    if chain_used not in chain_ids:
        if len(chain_ids) == 1:
            chain_used = list(chain_ids)[0]
        elif "A" in chain_ids:
            chain_used = "A"
        else:
            chain_used = list(chain_ids)[0]

    ss_list = []
    rsa_list = []
    found = 0
    total = (end_res - start_res + 1)

    for i in range(start_res, end_res + 1):
        key = (chain_used, (" ", i, " "))
        if key in dssp:
            ss_code = dssp[key][2]
            rel_rsa = float(dssp[key][3])

            if rel_rsa > 1.0:
                rel_rsa = 1.0
            if rel_rsa < 0.0:
                rel_rsa = 0.0

            ss_list.append(ss_code if ss_code != " " else "-")
            rsa_list.append(rel_rsa)
            found += 1
        else:
            ss_list.append("-")

    if total > 0 and (found / total) < MIN_FOUND_RATIO:
        return {
            "PDB_Type": struct_info.get("type", ""),
            "PDB_ID": os.path.basename(struct_info["path"]) if struct_info.get("path") else "",
            "SS_Class": "Unknown",
            "Avg_RSA": 0.0,
            "Status": f"Low_Coverage({found}/{total})",
            "Location": "Unknown",
            "DSSP_Mode": mode,
            "Chain_Used": chain_used,
        }

    if found == 0:
        return {
            "PDB_Type": struct_info.get("type", ""),
            "PDB_ID": os.path.basename(struct_info["path"]) if struct_info.get("path") else "",
            "SS_Class": "Unknown",
            "Avg_RSA": 0.0,
            "Status": "Residues_Not_Found",
            "Location": "Unknown",
            "DSSP_Mode": mode,
            "Chain_Used": chain_used,
        }

    ss_seq = "".join(ss_list)
    length = len(ss_seq)
    h_cnt = sum(1 for c in ss_seq if c in ["H", "G", "I"])
    s_cnt = sum(1 for c in ss_seq if c in ["E", "B"])

    if h_cnt / length > 0.5:
        ss_class = "Helix_Dominant"
    elif s_cnt / length > 0.5:
        ss_class = "Sheet_Dominant"
    elif (h_cnt + s_cnt) / length < 0.4:
        ss_class = "Coil/Loop_Dominant"
    else:
        ss_class = "Mixed"

    avg_rsa = float(sum(rsa_list) / len(rsa_list))

    if avg_rsa < 0.20:
        location = "Buried (Internal)"
    elif avg_rsa < 0.50:
        location = "Intermediate"
    else:
        location = "Exposed (Surface)"

    return {
        "PDB_Type": struct_info.get("type", ""),
        "PDB_ID": os.path.basename(struct_info["path"]) if struct_info.get("path") else "",
        "SS_Class": ss_class,
        "Avg_RSA": round(avg_rsa, 4),
        "Status": "OK",
        "Location": location,
        "DSSP_Mode": mode,
        "Chain_Used": chain_used,
    }


def make_base_result_row(row):
    result_row = dict(row)

    # Main defaults
    result_row["Main_Selected_UID"] = ""
    result_row["Main_MSA_Start"] = np.nan
    result_row["Main_MSA_End"] = np.nan
    result_row["Main_Selected_Seq_Start"] = np.nan
    result_row["Main_Selected_Seq_End"] = np.nan
    result_row["Main_Selected_Seq_Len"] = np.nan
    result_row["Main_Len_Diff"] = np.nan

    result_row["Main_PDB_Type"] = ""
    result_row["Main_PDB_ID"] = ""
    result_row["Main_SS_Class"] = "Unknown"
    result_row["Main_Avg_RSA"] = 0.0
    result_row["Main_Location"] = "Unknown"
    result_row["Main_Status"] = "Not_Processed"
    result_row["Main_DSSP_Mode"] = "NA"
    result_row["Main_Chain_Used"] = "NA"

    # Sub defaults
    result_row["Sub_Selected_UID"] = ""
    result_row["Sub_MSA_Start"] = np.nan
    result_row["Sub_MSA_End"] = np.nan
    result_row["Sub_Selected_Seq_Start"] = np.nan
    result_row["Sub_Selected_Seq_End"] = np.nan
    result_row["Sub_Selected_Seq_Len"] = np.nan
    result_row["Sub_Len_Diff"] = np.nan

    result_row["Sub_PDB_Type"] = ""
    result_row["Sub_PDB_ID"] = ""
    result_row["Sub_SS_Class"] = "Unknown"
    result_row["Sub_Avg_RSA"] = 0.0
    result_row["Sub_Location"] = "Unknown"
    result_row["Sub_Status"] = "Not_Processed"
    result_row["Sub_DSSP_Mode"] = "NA"
    result_row["Sub_Chain_Used"] = "NA"

    return result_row


def process_row(row):
    result_row = make_base_result_row(row)

    try:
        main_hmm = row.get("Main_HMM")
        sub_hmm = row.get("Sub_HMM")
        details = str(row.get("Sub_Segments_Details", ""))

        match = re.search(r"(\d+)-(\d+)\s*->\s*(\d+)-(\d+)", details)
        if not match:
            result_row["Main_Status"] = "Parse_Fail"
            result_row["Sub_Status"] = "Parse_Fail"
            return result_row

        main_seg = [int(match.group(1)), int(match.group(2))]
        sub_seg = [int(match.group(3)), int(match.group(4))]

        # Use the HHM segment length as expected_len
        hmm_len_a = main_seg[1] - main_seg[0] + 1
        hmm_len_b = sub_seg[1] - sub_seg[0] + 1

        # -------- Main --------
        struct_a = None
        map_a = extract_hmm_to_msa_map(os.path.join(PFAM_DIR, main_hmm, f"{main_hmm}.hhm"))
        if map_a and main_seg[0] in map_a and main_seg[1] in map_a:
            msa_start_a = map_a[main_seg[0]]
            msa_end_a = map_a[main_seg[1]]

            f_a = os.path.join(PFAM_DIR, main_hmm, f"{main_hmm}.fas")
            if not os.path.exists(f_a):
                f_a = os.path.join(PFAM_DIR, main_hmm, f"{main_hmm}.sto")

            cands_a = get_candidates_from_alignment(f_a, msa_start_a, msa_end_a, hmm_len_a)
            struct_a = find_best_structure_hybrid(os.path.join(PFAM_DIR, main_hmm), cands_a, MAX_LEN_DIFF)

            result_row["Main_MSA_Start"] = msa_start_a
            result_row["Main_MSA_End"] = msa_end_a

        if struct_a:
            result_row["Main_Selected_UID"] = struct_a.get("uid", "")
            result_row["Main_Selected_Seq_Start"] = struct_a.get("seq_start", np.nan)
            result_row["Main_Selected_Seq_End"] = struct_a.get("seq_end", np.nan)
            result_row["Main_Selected_Seq_Len"] = struct_a.get("seq_len", np.nan)
            result_row["Main_Len_Diff"] = struct_a.get("len_diff", np.nan)

        res_a = calculate_single_structure_features(struct_a)
        result_row["Main_PDB_Type"] = res_a["PDB_Type"]
        result_row["Main_PDB_ID"] = res_a["PDB_ID"]
        result_row["Main_SS_Class"] = res_a["SS_Class"]
        result_row["Main_Avg_RSA"] = res_a["Avg_RSA"]
        result_row["Main_Location"] = res_a["Location"]
        result_row["Main_Status"] = res_a["Status"]
        result_row["Main_DSSP_Mode"] = res_a["DSSP_Mode"]
        result_row["Main_Chain_Used"] = res_a["Chain_Used"]

        # -------- Sub --------
        struct_b = None
        map_b = extract_hmm_to_msa_map(os.path.join(PFAM_DIR, sub_hmm, f"{sub_hmm}.hhm"))
        if map_b and sub_seg[0] in map_b and sub_seg[1] in map_b:
            msa_start_b = map_b[sub_seg[0]]
            msa_end_b = map_b[sub_seg[1]]

            f_b = os.path.join(PFAM_DIR, sub_hmm, f"{sub_hmm}.fas")
            if not os.path.exists(f_b):
                f_b = os.path.join(PFAM_DIR, sub_hmm, f"{sub_hmm}.sto")

            cands_b = get_candidates_from_alignment(f_b, msa_start_b, msa_end_b, hmm_len_b)
            struct_b = find_best_structure_hybrid(os.path.join(PFAM_DIR, sub_hmm), cands_b, MAX_LEN_DIFF)

            result_row["Sub_MSA_Start"] = msa_start_b
            result_row["Sub_MSA_End"] = msa_end_b

        if struct_b:
            result_row["Sub_Selected_UID"] = struct_b.get("uid", "")
            result_row["Sub_Selected_Seq_Start"] = struct_b.get("seq_start", np.nan)
            result_row["Sub_Selected_Seq_End"] = struct_b.get("seq_end", np.nan)
            result_row["Sub_Selected_Seq_Len"] = struct_b.get("seq_len", np.nan)
            result_row["Sub_Len_Diff"] = struct_b.get("len_diff", np.nan)

        res_b = calculate_single_structure_features(struct_b)
        result_row["Sub_PDB_Type"] = res_b["PDB_Type"]
        result_row["Sub_PDB_ID"] = res_b["PDB_ID"]
        result_row["Sub_SS_Class"] = res_b["SS_Class"]
        result_row["Sub_Avg_RSA"] = res_b["Avg_RSA"]
        result_row["Sub_Location"] = res_b["Location"]
        result_row["Sub_Status"] = res_b["Status"]
        result_row["Sub_DSSP_Mode"] = res_b["DSSP_Mode"]
        result_row["Sub_Chain_Used"] = res_b["Chain_Used"]

        return result_row

    except Exception as e:
        result_row["Main_Status"] = f"Exception_{e}"
        result_row["Sub_Status"] = f"Exception_{e}"
        return result_row


def write_chunk(results_buf, output_csv, output_columns):
    if not results_buf:
        return

    df_chunk = pd.DataFrame(results_buf)

    for col in output_columns:
        if col not in df_chunk.columns:
            df_chunk[col] = np.nan

    df_chunk = df_chunk[output_columns]

    mode = "w" if not os.path.exists(output_csv) else "a"
    header = (mode == "w")
    df_chunk.to_csv(output_csv, mode=mode, header=header, index=False)


# ================= Main program =================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute secondary-structure class and average RSA for Pfam-Pfam fragment pairs."
    )
    parser.add_argument("--input-csv", default=INPUT_CSV, help="Input ProDive high-score summary CSV.")
    parser.add_argument("--pfam-dir", default=PFAM_DIR, help="PfamA_seed root directory containing HHM, alignment, and structure files.")
    parser.add_argument("--output-csv", default=OUTPUT_CSV, help="Output CSV path. The output column order is unchanged.")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes.")
    parser.add_argument("--max-search-depth", type=int, default=MAX_SEARCH_DEPTH, help="Maximum number of candidate sequences to try per side.")
    parser.add_argument("--max-len-diff", type=int, default=MAX_LEN_DIFF, help="Maximum allowed difference from HHM segment length.")
    parser.add_argument("--min-mean-plddt", type=float, default=MIN_MEAN_PLDDT, help="Minimum segment-mean AlphaFold pLDDT.")
    parser.add_argument("--max-head-tail-pae", type=float, default=MAX_HEAD_TAIL_PAE, help="Maximum endpoint PAE allowed for AlphaFold models.")
    parser.add_argument("--min-found-ratio", type=float, default=MIN_FOUND_RATIO, help="Minimum mapped-residue fraction required for a segment.")
    parser.add_argument("--write-chunk-size", type=int, default=WRITE_CHUNK_SIZE, help="Number of rows per output write chunk.")
    parser.add_argument("--max-inflight", type=int, default=None, help="Maximum submitted futures. Default: workers * 4.")
    parser.add_argument("--dssp-exe", default=DSSP_EXE, help="Path to mkdssp/dssp. Default: first executable found in PATH.")
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_OUTPUT, help="Overwrite an existing output CSV before running.")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Append to an existing output CSV instead of deleting it first.")
    return parser.parse_args()

def apply_args(args):
    global INPUT_CSV, PFAM_DIR, OUTPUT_CSV, NUM_WORKERS
    global MAX_SEARCH_DEPTH, MAX_LEN_DIFF, MIN_MEAN_PLDDT, MAX_HEAD_TAIL_PAE, MIN_FOUND_RATIO
    global OVERWRITE_OUTPUT, WRITE_CHUNK_SIZE, MAX_INFLIGHT, DSSP_EXE
    INPUT_CSV = args.input_csv
    PFAM_DIR = args.pfam_dir
    OUTPUT_CSV = args.output_csv
    NUM_WORKERS = args.workers
    MAX_SEARCH_DEPTH = args.max_search_depth
    MAX_LEN_DIFF = args.max_len_diff
    MIN_MEAN_PLDDT = args.min_mean_plddt
    MAX_HEAD_TAIL_PAE = args.max_head_tail_pae
    MIN_FOUND_RATIO = args.min_found_ratio
    OVERWRITE_OUTPUT = args.overwrite
    WRITE_CHUNK_SIZE = args.write_chunk_size
    MAX_INFLIGHT = args.max_inflight if args.max_inflight is not None else NUM_WORKERS * 4
    DSSP_EXE = args.dssp_exe

def main():
    args = parse_args()
    apply_args(args)
    if DSSP_EXE is None:
        print("[FATAL] mkdssp/dssp not found in PATH. Please install DSSP/mkdssp.")
        return

    if not os.path.exists(INPUT_CSV):
        print(f"[FATAL] Input CSV not found: {INPUT_CSV}")
        return

    if OVERWRITE_OUTPUT and os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    print(f"[INFO] Reading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    input_columns = list(df.columns)
    output_columns = input_columns + EXTRA_OUTPUT_COLUMNS

    rows_data = df.to_dict("records")
    total_tasks = len(rows_data)

    print(f"[INFO] Start parallel processing (workers={NUM_WORKERS}, DSSP={DSSP_EXE}, MAX_LEN_DIFF={MAX_LEN_DIFF})...")

    results_buf = []
    submitted = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        inflight = {}

        def submit_one(r):
            nonlocal submitted
            fut = executor.submit(process_row, r)
            inflight[fut] = 1
            submitted += 1

        it = iter(rows_data)
        for _ in range(min(MAX_INFLIGHT, total_tasks)):
            try:
                submit_one(next(it))
            except StopIteration:
                break

        with tqdm(total=total_tasks, desc="Pfam SS/RSA Calc") as pbar:
            while inflight:
                for fut in as_completed(list(inflight.keys()), timeout=None):
                    inflight.pop(fut, None)
                    res = fut.result()
                    results_buf.append(res)
                    pbar.update(1)

                    try:
                        submit_one(next(it))
                    except StopIteration:
                        pass

                    if len(results_buf) >= WRITE_CHUNK_SIZE:
                        write_chunk(results_buf, OUTPUT_CSV, output_columns)
                        results_buf = []

                    break

    if results_buf:
        write_chunk(results_buf, OUTPUT_CSV, output_columns)

    print(f"\n[DONE] Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()