#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map phi-related ProDive HMM segments to target Pfam seed-sequence intervals."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SIFTS_CSV = Path("CHANGE_ME")
DEFAULT_PFAM_DIR = Path("<PRODIVE_DATA_ROOT>/shared/PfamA_seed")
DEFAULT_MATCHES_DIR = Path("CHANGE_ME")
DEFAULT_OUTPUT_ROOT = Path("CHANGE_ME")


def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_int(x, default=None):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def sanitize_filename(text: str) -> str:
    text = safe_str(text)
    text = re.sub(r"[^\w.\-]+", "_", text)
    return text[:200] if len(text) > 200 else text


def ungap(seq: str) -> str:
    return seq.replace(".", "").replace("-", "")


def extract_hmm_to_msa_map(hhm_path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if not hhm_path.exists():
        raise FileNotFoundError(f"HHM file not found: {hhm_path}")

    with hhm_path.open("r", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) > 10 and parts[1].isdigit():
                try:
                    hmm_idx = int(parts[1])
                    msa_col_str = parts[-1]
                    if msa_col_str.isdigit():
                        mapping[hmm_idx] = int(msa_col_str)
                except Exception:
                    continue

    if not mapping:
        raise RuntimeError(f"No HMM-to-MSA mapping was parsed from: {hhm_path}")
    return mapping


def read_alignment(aln_path: Path) -> dict[str, str]:
    if not aln_path.exists():
        raise FileNotFoundError(f"Alignment file not found: {aln_path}")

    seqs: dict[str, str] = {}
    current_header = None

    with aln_path.open("r", errors="ignore") as handle:
        first_line = handle.readline()
        handle.seek(0)
        is_fasta = first_line.startswith(">")

        if is_fasta:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    current_header = line[1:].strip()
                    seqs[current_header] = ""
                elif current_header is not None:
                    seqs[current_header] += line
        else:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                sid, sfrag = parts[0], parts[1]
                seqs[sid] = seqs.get(sid, "") + sfrag

    if not seqs:
        raise RuntimeError(f"Alignment file is empty or could not be parsed: {aln_path}")
    return seqs


def choose_alignment_path(family_dir: Path, pfam_id: str) -> Path:
    fas_path = family_dir / f"{pfam_id}.fas"
    sto_path = family_dir / f"{pfam_id}.sto"
    if fas_path.exists():
        return fas_path
    if sto_path.exists():
        return sto_path
    raise FileNotFoundError(f"No {pfam_id}.fas or {pfam_id}.sto found under {family_dir}")


def normalize_header_text(text: str) -> str:
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def build_target_candidates(row) -> list[str]:
    candidates = []
    seed_seq_name = safe_str(row.get("seed_seq_name", ""))
    seed_seq_id = safe_str(row.get("seed_seq_id", ""))
    seed_ac_raw = safe_str(row.get("seed_ac_raw", ""))
    seed_ac_base = safe_str(row.get("seed_ac_base", ""))
    sifts_uniprot_ac_base = safe_str(row.get("sifts_uniprot_ac_base", ""))

    for value in [seed_seq_name, seed_seq_id, seed_ac_base, seed_ac_raw]:
        if value:
            candidates.append(value)
    if seed_ac_raw and "." in seed_ac_raw:
        candidates.append(seed_ac_raw.split(".")[0])
    if sifts_uniprot_ac_base:
        candidates.append(sifts_uniprot_ac_base)

    seen = set()
    uniq = []
    for c in candidates:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def find_target_sequence(seqs: dict[str, str], row):
    candidates = build_target_candidates(row)
    headers = list(seqs.keys())
    normalized = {h: normalize_header_text(h) for h in headers}

    for c in candidates:
        c_norm = normalize_header_text(c)
        for h in headers:
            if normalized[h] == c_norm:
                return h, seqs[h], f"exact:{c}"

    for c in candidates:
        c_norm = normalize_header_text(c)
        for h in headers:
            if normalized[h].startswith(c_norm):
                return h, seqs[h], f"startswith:{c}"

    for c in candidates:
        c_plain = re.escape(c.split(".")[0])
        pattern = re.compile(rf"(?<![A-Za-z0-9]){c_plain}(?![A-Za-z0-9])")
        for h in headers:
            if pattern.search(normalized[h]):
                return h, seqs[h], f"token:{c}"

    for c in candidates:
        c0 = c.split(".")[0]
        for h in headers:
            if c0 in normalized[h]:
                return h, seqs[h], f"contains:{c}"

    raise RuntimeError(f"Target sequence not found in alignment. Candidate identifiers: {candidates[:10]}")


def parse_header_global_range(header: str):
    m = re.search(r"/(\d+)-(\d+)", header)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def map_msa_interval_to_real_seq(aligned_seq: str, global_start: int, msa_start_col: int, msa_end_col: int):
    idx_start = msa_start_col - 1
    idx_end = msa_end_col

    if len(aligned_seq) < idx_end:
        return {
            "msa_start": msa_start_col,
            "msa_end": msa_end_col,
            "seq_start": np.nan,
            "seq_end": np.nan,
            "seq_len": 0,
            "frag_raw": "",
            "frag_pure": "",
            "gaps": np.nan,
            "status": "MSA_Out_Of_Range",
        }

    frag_raw = aligned_seq[idx_start:idx_end]
    frag_pure = ungap(frag_raw)
    gaps = frag_raw.count(".") + frag_raw.count("-")

    if len(frag_pure) == 0:
        return {
            "msa_start": msa_start_col,
            "msa_end": msa_end_col,
            "seq_start": np.nan,
            "seq_end": np.nan,
            "seq_len": 0,
            "frag_raw": frag_raw,
            "frag_pure": frag_pure,
            "gaps": gaps,
            "status": "Only_Gaps_On_Target",
        }

    prefix = aligned_seq[:idx_start]
    real_res_before = len(ungap(prefix))
    seq_start = global_start + real_res_before
    seq_end = seq_start + len(frag_pure) - 1

    return {
        "msa_start": msa_start_col,
        "msa_end": msa_end_col,
        "seq_start": seq_start,
        "seq_end": seq_end,
        "seq_len": len(frag_pure),
        "frag_raw": frag_raw,
        "frag_pure": frag_pure,
        "gaps": gaps,
        "status": "OK",
    }


def parse_main_segment(text: str) -> list[tuple[int, int]]:
    text = safe_str(text)
    pairs = re.findall(r"(\d+)-(\d+)", text)
    return [(int(a), int(b)) for a, b in pairs]


def parse_sub_segments_details(text: str) -> list[tuple[int, int, int, int]]:
    text = safe_str(text)
    pairs = re.findall(r"(\d+)-(\d+)\s*->\s*(\d+)-(\d+)", text)
    return [(int(a), int(b), int(c), int(d)) for a, b, c, d in pairs]


def extract_target_hmm_segments(match_row, target_pfam: str):
    segments = []
    main_hmm = safe_str(match_row.get("Main_HMM", ""))
    sub_hmm = safe_str(match_row.get("Sub_HMM", ""))
    main_segment = safe_str(match_row.get("Main_Segment", ""))
    details = safe_str(match_row.get("Sub_Segments_Details", ""))
    detail_pairs = parse_sub_segments_details(details)

    if main_hmm == target_pfam:
        main_pairs = parse_main_segment(main_segment)
        if main_pairs:
            for s, e in main_pairs:
                segments.append(("main", s, e))
        else:
            for a, b, _, _ in detail_pairs:
                segments.append(("main", a, b))

    if sub_hmm == target_pfam:
        for _, _, c, d in detail_pairs:
            segments.append(("sub", c, d))

    return segments


def build_position_coverage(mapped_df: pd.DataFrame, real_start: int, real_end: int) -> pd.DataFrame:
    positions = np.arange(real_start, real_end + 1)
    coverage = pd.DataFrame({"Position": positions, "Count": np.zeros(len(positions), dtype=int)}).set_index("Position")
    ok_df = mapped_df[mapped_df["Map_Status"] == "OK"].copy()

    for _, row in ok_df.iterrows():
        s = safe_int(row["Target_Seq_Start"])
        e = safe_int(row["Target_Seq_End"])
        if s is None or e is None:
            continue
        if s > e:
            s, e = e, s
        coverage.loc[s:e, "Count"] += 1

    return coverage.reset_index()


def plot_coverage(coverage_df: pd.DataFrame, out_png: Path, title: str) -> None:
    plt.figure(figsize=(16, 4))
    plt.bar(coverage_df["Position"], coverage_df["Count"], width=1.0)
    plt.xlabel("Residue position")
    plt.ylabel("Hit count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def get_cached_pfam_data(pfam_id: str, pfam_cache: dict, pfam_dir: Path, matches_dir: Path):
    if pfam_id in pfam_cache:
        return pfam_cache[pfam_id]

    family_dir = pfam_dir / pfam_id
    hhm_path = family_dir / f"{pfam_id}.hhm"
    aln_path = choose_alignment_path(family_dir, pfam_id)
    matches_csv = matches_dir / f"{pfam_id}_matches.csv"

    if not matches_csv.exists():
        raise FileNotFoundError(f"Matching CSV not found: {matches_csv}")

    data = {
        "family_dir": family_dir,
        "hhm_path": hhm_path,
        "aln_path": aln_path,
        "matches_csv": matches_csv,
        "hhm_map": extract_hmm_to_msa_map(hhm_path),
        "seqs": read_alignment(aln_path),
        "matches_df": pd.read_csv(matches_csv),
    }
    pfam_cache[pfam_id] = data
    return data


def build_task_name(row) -> str:
    pdb_id = safe_str(row.get("pdb_id", ""))
    pdb_raw = safe_str(row.get("pdb_raw", ""))
    pfdb_chain = safe_str(row.get("pfdb_chain", ""))
    sifts_chain = safe_str(row.get("sifts_chain", ""))
    used_auth_pdb_start = safe_int(row.get("used_auth_pdb_start"), None)
    used_auth_pdb_end = safe_int(row.get("used_auth_pdb_end"), None)
    pfam_id = safe_str(row.get("pfam_id", ""))
    seed_ac_base = safe_str(row.get("seed_ac_base", ""))
    seed_seq_name = safe_str(row.get("seed_seq_name", ""))

    chain_tag = pfdb_chain if pfdb_chain else sifts_chain
    pdb_tag = pdb_id if pdb_id else pdb_raw
    pdb_range_tag = f"{used_auth_pdb_start}-{used_auth_pdb_end}" if used_auth_pdb_start is not None and used_auth_pdb_end is not None else "full"

    return sanitize_filename(
        f"{pdb_tag}"
        f"{'_' + chain_tag if chain_tag else ''}"
        f"__{pdb_range_tag}"
        f"__{pfam_id}"
        f"__{seed_ac_base or seed_seq_name}"
    )


def process_one_sifts_row(row, pfam_cache: dict, pfam_dir: Path, matches_dir: Path, output_root: Path, make_plots: bool):
    pfdb_row_id = safe_str(row.get("pfdb_row_id", ""))
    pfam_id = safe_str(row.get("pfam_id", ""))
    seed_ac_base = safe_str(row.get("seed_ac_base", ""))
    seed_seq_name = safe_str(row.get("seed_seq_name", ""))
    seed_start = safe_int(row.get("seed_start"), None)
    seed_end = safe_int(row.get("seed_end"), None)
    task_name = build_task_name(row)

    summary = {
        "task_name": task_name,
        "pfdb_row_id": pfdb_row_id,
        "pdb_id": safe_str(row.get("pdb_id", "")),
        "pdb_raw": safe_str(row.get("pdb_raw", "")),
        "pfdb_chain": safe_str(row.get("pfdb_chain", "")),
        "sifts_chain": safe_str(row.get("sifts_chain", "")),
        "pfam_id": pfam_id,
        "seed_ac_base": seed_ac_base,
        "seed_seq_name": seed_seq_name,
        "seed_start": seed_start,
        "seed_end": seed_end,
        "status": "INIT",
        "message": "",
        "target_header": "",
        "match_mode": "",
        "num_match_rows": 0,
        "num_segments_total": 0,
        "num_segments_ok": 0,
    }

    records = []

    try:
        if not pfam_id:
            raise RuntimeError("pfam_id is empty")

        pfam_data = get_cached_pfam_data(pfam_id, pfam_cache, pfam_dir, matches_dir)
        hhm_map = pfam_data["hhm_map"]
        seqs = pfam_data["seqs"]
        matches_df = pfam_data["matches_df"]

        target_header, target_aln_seq, match_mode = find_target_sequence(seqs, row)
        summary["target_header"] = target_header
        summary["match_mode"] = match_mode

        header_start, header_end = parse_header_global_range(target_header)
        if seed_start is not None and seed_end is not None:
            global_start = seed_start
            global_end_expected = seed_end
        elif header_start is not None and header_end is not None:
            global_start = header_start
            global_end_expected = header_end
            seed_start = header_start
            seed_end = header_end
        else:
            raise RuntimeError(f"No usable seed_start/seed_end and no parseable header range: {target_header}")

        real_len_from_alignment = len(ungap(target_aln_seq))
        derived_end = global_start + real_len_from_alignment - 1

        length_note = ""
        if global_end_expected is not None:
            expected_len = global_end_expected - global_start + 1
            if expected_len != real_len_from_alignment:
                length_note = f"LengthMismatch(expected={expected_len}, alignment={real_len_from_alignment})"

        summary["num_match_rows"] = len(matches_df)

        for idx, mrow in matches_df.iterrows():
            row_segments = extract_target_hmm_segments(mrow, pfam_id)
            if not row_segments:
                continue

            for seg_rank, (pfam_side, hmm_start, hmm_end) in enumerate(row_segments, start=1):
                summary["num_segments_total"] += 1
                base_rec = {
                    "task_name": task_name,
                    "pfdb_row_id": pfdb_row_id,
                    "pdb_id": safe_str(row.get("pdb_id", "")),
                    "pdb_raw": safe_str(row.get("pdb_raw", "")),
                    "pfdb_chain": safe_str(row.get("pfdb_chain", "")),
                    "sifts_chain": safe_str(row.get("sifts_chain", "")),
                    "pfam_id": pfam_id,
                    "seed_ac_base": seed_ac_base,
                    "seed_seq_name": seed_seq_name,
                    "seed_start": seed_start,
                    "seed_end": seed_end,
                    "target_header": target_header,
                    "match_mode": match_mode,
                    "length_note": length_note,
                    "Match_Row_Index": idx,
                    "Segment_Rank": seg_rank,
                    "File": mrow.get("File", ""),
                    "Score": mrow.get("Score", np.nan),
                    "Main_HMM": mrow.get("Main_HMM", ""),
                    "Sub_HMM": mrow.get("Sub_HMM", ""),
                    "Target_Pfam_Side": pfam_side,
                    "Target_HMM_Start": hmm_start,
                    "Target_HMM_End": hmm_end,
                }

                if hmm_start not in hhm_map or hmm_end not in hhm_map:
                    rec = dict(base_rec)
                    rec.update({
                        "Target_MSA_Start": np.nan,
                        "Target_MSA_End": np.nan,
                        "Target_Seq_Start": np.nan,
                        "Target_Seq_End": np.nan,
                        "Target_Seq_Len": 0,
                        "Target_Frag": "",
                        "Target_Aligned_Frag": "",
                        "Gap_Count_In_MSA_Window": np.nan,
                        "Map_Status": "HMM_Position_Not_In_HHM_Map",
                    })
                    records.append(rec)
                    continue

                msa_start = hhm_map[hmm_start]
                msa_end = hhm_map[hmm_end]
                mapped = map_msa_interval_to_real_seq(target_aln_seq, global_start, msa_start, msa_end)

                rec = dict(base_rec)
                rec.update({
                    "Target_MSA_Start": mapped["msa_start"],
                    "Target_MSA_End": mapped["msa_end"],
                    "Target_Seq_Start": mapped["seq_start"],
                    "Target_Seq_End": mapped["seq_end"],
                    "Target_Seq_Len": mapped["seq_len"],
                    "Target_Frag": mapped["frag_pure"],
                    "Target_Aligned_Frag": mapped["frag_raw"],
                    "Gap_Count_In_MSA_Window": mapped["gaps"],
                    "Map_Status": mapped["status"],
                })
                records.append(rec)

                if mapped["status"] == "OK":
                    summary["num_segments_ok"] += 1

        mapped_df = pd.DataFrame(records)
        if mapped_df.empty:
            raise RuntimeError("No target-Pfam intervals were extracted from the matches file")

        task_dir = output_root / task_name
        ensure_dir(task_dir)

        interval_csv = task_dir / f"{task_name}_mapped_intervals.csv"
        coverage_csv = task_dir / f"{task_name}_position_coverage.csv"
        plot_png = task_dir / f"{task_name}_position_distribution.png"

        mapped_df.to_csv(interval_csv, index=False)
        coverage_df = build_position_coverage(
            mapped_df=mapped_df,
            real_start=global_start,
            real_end=(global_end_expected if global_end_expected is not None else derived_end),
        )
        coverage_df.to_csv(coverage_csv, index=False)

        if make_plots:
            plot_coverage(coverage_df, plot_png, task_name)

        summary["status"] = "OK"
        summary["message"] = length_note if length_note else "Success"
        return summary, mapped_df

    except Exception as exc:
        summary["status"] = "FAIL"
        summary["message"] = f"{type(exc).__name__}: {exc}"
        return summary, pd.DataFrame()


def run(args: argparse.Namespace) -> None:
    ensure_dir(args.output_root)

    sifts_df = pd.read_csv(args.sifts_csv)
    if args.row_limit is not None:
        sifts_df = sifts_df.head(args.row_limit)

    pfam_cache = {}
    all_summaries = []
    all_intervals = []

    total = len(sifts_df)
    print(f"[INFO] rows to process: {total}")

    for i, (_, row) in enumerate(sifts_df.iterrows(), start=1):
        task_name = build_task_name(row)
        pfam_id = safe_str(row.get("pfam_id", ""))
        seed_ac_base = safe_str(row.get("seed_ac_base", ""))
        print(f"[{i}/{total}] Processing: {task_name} | pfam_id={pfam_id} | seed={seed_ac_base}")

        summary, mapped_df = process_one_sifts_row(
            row=row,
            pfam_cache=pfam_cache,
            pfam_dir=args.pfam_dir,
            matches_dir=args.matches_dir,
            output_root=args.output_root,
            make_plots=not args.no_plots,
        )
        all_summaries.append(summary)
        if not mapped_df.empty:
            all_intervals.append(mapped_df)

    summary_df = pd.DataFrame(all_summaries)
    summary_csv = args.output_root / "ALL_tasks_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    if all_intervals:
        merged_intervals_df = pd.concat(all_intervals, ignore_index=True)
    else:
        merged_intervals_df = pd.DataFrame()

    merged_intervals_csv = args.output_root / "ALL_mapped_intervals.csv"
    merged_intervals_df.to_csv(merged_intervals_csv, index=False)

    ok_n = int((summary_df["status"] == "OK").sum()) if not summary_df.empty else 0
    fail_n = int((summary_df["status"] == "FAIL").sum()) if not summary_df.empty else 0

    print("\n==================== DONE ====================")
    print(f"[OK] Summary table: {summary_csv}")
    print(f"[OK] Combined interval table: {merged_intervals_csv}")
    print(f"[INFO] Success: {ok_n}")
    print(f"[INFO] Failed: {fail_n}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map phi-related ProDive HMM segments to Pfam seed-sequence intervals.")
    parser.add_argument("--sifts-csv", type=Path, default=DEFAULT_SIFTS_CSV, help="Best-hit SIFTS/Pfam CSV from step 01.")
    parser.add_argument("--pfam-dir", type=Path, default=DEFAULT_PFAM_DIR, help="PfamA_seed root directory.")
    parser.add_argument("--matches-dir", type=Path, default=DEFAULT_MATCHES_DIR, help="Per-Pfam match CSV directory from step 03.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root directory.")
    parser.add_argument("--row-limit", type=int, default=None, help="Optional number of SIFTS rows to process for testing.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV outputs but skip coverage PNG generation.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
