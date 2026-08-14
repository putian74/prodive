import pandas as pd
import os
import warnings
import csv
import argparse
from Bio import SeqIO, BiopythonWarning
from Bio.PDB import PDBParser, DSSP
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= Configuration =================
INPUT_CSV_PATH = ""
PDB_DIR = ""
FASTA_PATH = ""
OUTPUT_CSV_PATH = ""
NUM_WORKERS = 20

# Minimum fraction of segment residues that must be found by DSSP.
MIN_FOUND_RATIO = 0.8

def process_single_row(row_data, chain_map_subset):
    warnings.simplefilter('ignore', BiopythonWarning)
    warnings.filterwarnings("ignore")

    result = {
        'SS_Sequence': '',
        'SS_Class': 'Error',
        'Avg_RSA': 0.0,
        'Location': 'Error',
        'Status': 'Init',
        'Found_Ratio': 0.0,
    }

    try:
        denovo_id = row_data.get('denovo_id_extracted')
        segment_str = row_data.get('Main_Segment')

        if not denovo_id or not isinstance(segment_str, str) or '-' not in segment_str:
            result['Status'] = 'Format_Error'
            return result

        pdb_code = denovo_id.split('_')[0]
        pdb_path = os.path.join(PDB_DIR, f"{pdb_code}.pdb")

        chain_id = chain_map_subset.get(denovo_id, 'A')

        start_res = int(segment_str.split('-')[0])
        end_res = int(segment_str.split('-')[1])
        if start_res > end_res:
            start_res, end_res = end_res, start_res

        if not os.path.exists(pdb_path):
            result['Status'] = 'PDB_Not_Found'
            return result

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('struct', pdb_path)
        model = structure[0]

        # DSSP: dssp[key] = (dssp_index, aa, ss, rel_acc, phi, psi, ...)
        dssp = DSSP(model, pdb_path)

        ss_list = []
        rsa_list = []
        found = 0
        total = (end_res - start_res + 1)

        for i in range(start_res, end_res + 1):
            key = (chain_id, (' ', i, ' '))
            if key in dssp:
                ss_code = dssp[key][2]      # secondary structure code
                rel_rsa = dssp[key][3]      # relative ASA (0..1)
                ss_list.append(ss_code if ss_code != " " else "-")
                rsa_list.append(float(rel_rsa))
                found += 1
            else:
                ss_list.append('-')

        ss_seq = "".join(ss_list)
        result['SS_Sequence'] = ss_seq
        result['Found_Ratio'] = round(found / total, 4) if total > 0 else 0.0

        # Treat low DSSP residue coverage as mapping failure.
        if found == 0 or (found / total) < MIN_FOUND_RATIO:
            result['SS_Class'] = 'Unknown'
            result['Avg_RSA'] = 0.0
            result['Location'] = 'Unknown'
            result['Status'] = 'Segment_Not_Mapped'
            return result

        # Secondary-structure classification.
        length = len(ss_seq)
        h_cnt = sum(1 for c in ss_seq if c in ['H', 'G', 'I'])
        s_cnt = sum(1 for c in ss_seq if c in ['E', 'B'])

        if length > 0:
            if h_cnt / length > 0.5:
                result['SS_Class'] = "Helix_Dominant"
            elif s_cnt / length > 0.5:
                result['SS_Class'] = "Sheet_Dominant"
            elif (h_cnt + s_cnt) / length < 0.4:
                result['SS_Class'] = "Coil/Loop_Dominant"
            else:
                result['SS_Class'] = "Mixed"
        else:
            result['SS_Class'] = "Unknown"

        # RSA and exposure class. Values are DSSP relative ASA in [0, 1].
        avg_rsa = sum(rsa_list) / len(rsa_list)
        result['Avg_RSA'] = round(avg_rsa, 4)
        if avg_rsa < 0.20:
            result['Location'] = "Buried (Internal)"
        elif avg_rsa < 0.50:
            result['Location'] = "Intermediate"
        else:
            result['Location'] = "Exposed (Surface)"

        result['Status'] = 'OK'
        return result

    except Exception as e:
        result['Status'] = f"Error: {str(e)}"
        return result


def get_chain_mapping(fasta_path):
    print("[INFO] Loading chain mapping from FASTA...")
    mapping = {}
    if not os.path.exists(fasta_path):
        return mapping
    for record in SeqIO.parse(fasta_path, "fasta"):
        parts = record.description.split('|')
        if len(parts) >= 2:
            denovo_id = parts[0].strip()
            c_part = parts[1].replace("Chains", "").replace("Chain", "").strip()
            first_c = c_part.split(',')[0].strip().split('[')[0].strip()
            mapping[denovo_id] = first_c
    return mapping


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute secondary-structure class and average RSA for de novo-Pfam real fragments."
    )
    parser.add_argument("--input-csv", required=True, help="Input de novo-Pfam high-score summary CSV.")
    parser.add_argument("--pdb-dir", required=True, help="Directory containing de novo PDB files.")
    parser.add_argument("--fasta", required=True, help="FASTA used to map de novo IDs to chains.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path. The output column order is unchanged.")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes.")
    parser.add_argument("--min-found-ratio", type=float, default=MIN_FOUND_RATIO, help="Minimum DSSP residue coverage required for a segment.")
    return parser.parse_args()

def apply_args(args):
    global INPUT_CSV_PATH, PDB_DIR, FASTA_PATH, OUTPUT_CSV_PATH, NUM_WORKERS, MIN_FOUND_RATIO
    INPUT_CSV_PATH = args.input_csv
    PDB_DIR = args.pdb_dir
    FASTA_PATH = args.fasta
    OUTPUT_CSV_PATH = args.output_csv
    NUM_WORKERS = args.workers
    MIN_FOUND_RATIO = args.min_found_ratio

def main():
    args = parse_args()
    apply_args(args)

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"[FATAL] Input CSV not found: {INPUT_CSV_PATH}")
        return

    print(f"[INFO] Reading input CSV: {INPUT_CSV_PATH}")
    df_input = pd.read_csv(INPUT_CSV_PATH)

    # Build a stable key for resumable execution.
    df_input['unique_key'] = df_input['File'].astype(str) + "_" + df_input['Main_Segment'].astype(str)

    processed_keys = set()
    write_header = True

    if os.path.exists(OUTPUT_CSV_PATH):
        print(f"[INFO] Existing output found: {OUTPUT_CSV_PATH}. Checking completed tasks...")
        try:
            df_done = pd.read_csv(OUTPUT_CSV_PATH)
            if 'File' in df_done.columns and 'Main_Segment' in df_done.columns:
                df_done['unique_key'] = df_done['File'].astype(str) + "_" + df_done['Main_Segment'].astype(str)
                processed_keys = set(df_done['unique_key'].values)
                print(f"[INFO] Completed rows already present: {len(processed_keys)}")
                write_header = False
            else:
                print("[WARN] Existing output format does not match. Restarting.")
        except Exception as e:
            print(f"[WARN] Failed to read existing output ({e}). Restarting.")

    if processed_keys:
        df_to_process = df_input[~df_input['unique_key'].isin(processed_keys)].copy()
        print(f"[INFO] Remaining rows after resume filtering: {len(df_to_process)}")
    else:
        df_to_process = df_input.copy()
        print(f"[INFO] Full run rows: {len(df_to_process)}")

    if len(df_to_process) == 0:
        print("[DONE] All rows are already completed.")
        return

    # Prefer the named Main_HMM column; use the positional field as a fallback.
    if 'Main_HMM' in df_to_process.columns:
        df_to_process['denovo_id_extracted'] = df_to_process['Main_HMM'].astype(str)
    else:
        df_to_process['denovo_id_extracted'] = df_to_process.iloc[:, 4].astype(str)

    chain_map = get_chain_mapping(FASTA_PATH)

    new_cols = ['SS_Sequence', 'SS_Class', 'Avg_RSA', 'Location', 'Status', 'Found_Ratio']
    original_cols = [c for c in df_input.columns if c != 'unique_key']
    final_headers = original_cols + new_cols

    out_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if write_header:
        with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(final_headers)

    print(f"[INFO] Start multiprocessing (workers={NUM_WORKERS})...")

    rows_data = df_to_process.to_dict('records')
    total_tasks = len(rows_data)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_row = {
            executor.submit(process_single_row, row, chain_map): row
            for row in rows_data
        }

        with tqdm(total=total_tasks, unit="seq", desc="Parallel DSSP") as pbar:
            for future in as_completed(future_to_row):
                original_row = future_to_row[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        'SS_Sequence': '',
                        'SS_Class': 'Error',
                        'Avg_RSA': 0.0,
                        'Location': 'Error',
                        'Status': f"Crash: {exc}",
                        'Found_Ratio': 0.0,
                    }

                final_row_dict = {**original_row, **result}

                with open(OUTPUT_CSV_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row_values = [final_row_dict.get(h, '') for h in final_headers]
                    writer.writerow(row_values)

                pbar.update(1)

    print("\n[DONE] All tasks completed.")


if __name__ == "__main__":
    main()
