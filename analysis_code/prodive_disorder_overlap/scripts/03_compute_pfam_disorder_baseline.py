#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute fair disorder baselines within Pfam-covered sequence space.

Main idea
---------
Use the same analyzable sequence universe as the fragment analysis, rather than
full protein length by default.

Inputs
------
Required:
- DisProt TSV with residue-level annotations
- seed-match detail CSV (contains seed intervals mapped to proteins)

Recommended:
- all_mapped_segments_with_overlap_status.csv
  Used to define the active protein universe as proteins that actually entered
  the segment-mapping stage.

Outputs
-------
1) per_protein_pfam_disorder_baseline.csv
   One row per protein with Pfam-covered length and disorder/all-term coverage.
2) global_pfam_disorder_baseline_summary.csv
   Global summary across proteins.
3) merged_intervals_debug.csv
   Optional debug table of merged interval blocks.

The script can read either extracted files or a project zip bundle.


python3 CHANGE_ME \
  --disprot-tsv CHANGE_ME \
  --detail-csv CHANGE_ME \
  --mapped-csv CHANGE_ME \
  --scope mapped \
  --out-dir CHANGE_ME
"""

from __future__ import annotations

import argparse
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

Interval = Tuple[int, int]


def normalize_ac(x: object) -> str:
    s = str(x).strip()
    if not s or s.lower() == 'nan':
        return ''
    return s.split('.')[0].upper()


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    cleaned: List[Interval] = []
    for a, b in intervals:
        if pd.isna(a) or pd.isna(b):
            continue
        a = int(a)
        b = int(b)
        if a > b:
            a, b = b, a
        cleaned.append((a, b))
    if not cleaned:
        return []
    cleaned.sort()
    merged = [cleaned[0]]
    for a, b in cleaned[1:]:
        la, lb = merged[-1]
        if a <= lb + 1:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def intervals_len(intervals: Sequence[Interval]) -> int:
    return sum(b - a + 1 for a, b in intervals)


def intersect_two_lists(a_list: Sequence[Interval], b_list: Sequence[Interval]) -> List[Interval]:
    out: List[Interval] = []
    i = j = 0
    a_list = list(a_list)
    b_list = list(b_list)
    while i < len(a_list) and j < len(b_list):
        a1, a2 = a_list[i]
        b1, b2 = b_list[j]
        start = max(a1, b1)
        end = min(a2, b2)
        if start <= end:
            out.append((start, end))
        if a2 < b2:
            i += 1
        else:
            j += 1
    return out


@dataclass
class Inputs:
    disprot_tsv: pd.DataFrame
    detail_df: pd.DataFrame
    mapped_df: Optional[pd.DataFrame]


class Reader:
    def __init__(self, bundle_zip: Optional[Path]):
        self.bundle_zip = bundle_zip
        self.zf: Optional[zipfile.ZipFile] = None
        if bundle_zip is not None:
            self.zf = zipfile.ZipFile(bundle_zip)

    def read_csv(self, path: Optional[Path], inner_name: Optional[str]) -> Optional[pd.DataFrame]:
        if path is not None:
            path_str = str(path)
            if path_str.endswith('.tsv'):
                return pd.read_csv(path, sep='\t')
            return pd.read_csv(path)

        if self.zf is None or inner_name is None:
            return None

        with self.zf.open(inner_name) as f:
            if inner_name.endswith('.tsv'):
                return pd.read_csv(f, sep='\t')
            return pd.read_csv(f)


def load_inputs(args: argparse.Namespace) -> Inputs:
    reader = Reader(args.bundle_zip)

    disprot_tsv = reader.read_csv(args.disprot_tsv, 'DisProt_current_IDPO.tsv')
    detail_df = reader.read_csv(args.detail_csv, 'disprot_seed_match_output/disprot_seed_match_detail.csv')
    mapped_df = reader.read_csv(args.mapped_csv, 'global_hmm_to_disprot_region_overlap/all_mapped_segments_with_overlap_status.csv')

    if disprot_tsv is None:
        raise ValueError('DisProt TSV is required.')
    if detail_df is None:
        raise ValueError('Seed-match detail CSV is required.')

    return Inputs(disprot_tsv=disprot_tsv, detail_df=detail_df, mapped_df=mapped_df)


def build_active_protein_set(inputs: Inputs, scope: str) -> set[str]:
    detail = inputs.detail_df.copy()
    detail['uniprot_ac_base'] = detail['uniprot_ac_base'].map(normalize_ac)

    if scope == 'detail_all':
        return set(x for x in detail['uniprot_ac_base'] if x)

    if scope == 'detail_covered_only':
        mask = detail['overlap_len'].fillna(0).astype(float) > 0
        return set(x for x in detail.loc[mask, 'uniprot_ac_base'] if x)

    if scope == 'mapped' and inputs.mapped_df is not None:
        mapped = inputs.mapped_df.copy()
        mapped['uniprot_ac_base'] = mapped['uniprot_ac_base'].map(normalize_ac)
        return set(x for x in mapped['uniprot_ac_base'] if x)

    raise ValueError(
        "scope='mapped' requires all_mapped_segments_with_overlap_status.csv; "
        "use --mapped-csv or --bundle-zip."
    )


def collect_pfam_intervals(detail_df: pd.DataFrame, active_proteins: set[str]) -> Dict[str, List[Interval]]:
    df = detail_df.copy()
    df['uniprot_ac_base'] = df['uniprot_ac_base'].map(normalize_ac)
    df = df[df['uniprot_ac_base'].isin(active_proteins)].copy()

    # Deduplicate seed intervals to avoid repeated counts from multiple region rows.
    keep_cols = ['uniprot_ac_base', 'pfam_id', 'seed_seq_name', 'seed_start', 'seed_end']
    df = df.drop_duplicates(subset=keep_cols)

    by_protein: Dict[str, List[Interval]] = {}
    for ac, sub in df.groupby('uniprot_ac_base'):
        intervals = [(int(s), int(e)) for s, e in zip(sub['seed_start'], sub['seed_end']) if pd.notna(s) and pd.notna(e)]
        by_protein[ac] = merge_intervals(intervals)
    return by_protein


def collect_disprot_intervals(disprot_tsv: pd.DataFrame, active_proteins: set[str]) -> Tuple[Dict[str, int], Dict[str, List[Interval]], Dict[str, List[Interval]]]:
    df = disprot_tsv.copy()
    df['uniprot_ac_base'] = df['UniProt ACC'].map(normalize_ac)
    df = df[df['uniprot_ac_base'].isin(active_proteins)].copy()

    protein_lengths: Dict[str, int] = {}
    disorder_intervals: Dict[str, List[Interval]] = {}
    all_term_intervals: Dict[str, List[Interval]] = {}

    for ac, sub in df.groupby('uniprot_ac_base'):
        lengths = sub['Sequence length'].dropna().astype(int)
        protein_lengths[ac] = int(lengths.iloc[0]) if len(lengths) else 0

        all_iv = [(int(s), int(e)) for s, e in zip(sub['Start'], sub['End']) if pd.notna(s) and pd.notna(e)]
        disorder_sub = sub[sub['Term name'].astype(str).str.strip().str.lower() == 'disorder']
        dis_iv = [(int(s), int(e)) for s, e in zip(disorder_sub['Start'], disorder_sub['End']) if pd.notna(s) and pd.notna(e)]

        disorder_intervals[ac] = merge_intervals(dis_iv)
        all_term_intervals[ac] = merge_intervals(all_iv)

    for ac in active_proteins:
        protein_lengths.setdefault(ac, 0)
        disorder_intervals.setdefault(ac, [])
        all_term_intervals.setdefault(ac, [])

    return protein_lengths, disorder_intervals, all_term_intervals


def compute_baseline_table(inputs: Inputs, scope: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    active = build_active_protein_set(inputs, scope=scope)
    pfam_intervals = collect_pfam_intervals(inputs.detail_df, active)
    protein_lengths, disorder_iv, allterm_iv = collect_disprot_intervals(inputs.disprot_tsv, active)

    rows = []
    debug_rows = []

    for ac in sorted(active):
        pf = pfam_intervals.get(ac, [])
        dis = disorder_iv.get(ac, [])
        allv = allterm_iv.get(ac, [])

        dis_in_pf = intersect_two_lists(pf, dis)
        all_in_pf = intersect_two_lists(pf, allv)

        pf_len = intervals_len(pf)
        dis_len = intervals_len(dis_in_pf)
        all_len = intervals_len(all_in_pf)
        protein_len = protein_lengths.get(ac, 0)

        rows.append({
            'uniprot_ac': ac,
            'protein_length': protein_len,
            'Pfam_covered_length': pf_len,
            'Disorder_length_within_Pfam': dis_len,
            'AllTerm_length_within_Pfam': all_len,
            'Disorder_fraction_within_Pfam': (dis_len / pf_len) if pf_len else pd.NA,
            'AllTerm_fraction_within_Pfam': (all_len / pf_len) if pf_len else pd.NA,
            'Disorder_fraction_full_length': (dis_len / protein_len) if protein_len else pd.NA,
            'AllTerm_fraction_full_length': (all_len / protein_len) if protein_len else pd.NA,
            'n_Pfam_blocks': len(pf),
            'n_disorder_blocks_within_Pfam': len(dis_in_pf),
            'n_allterm_blocks_within_Pfam': len(all_in_pf),
        })

        for label, intervals in [
            ('Pfam_covered', pf),
            ('Disorder_within_Pfam', dis_in_pf),
            ('AllTerm_within_Pfam', all_in_pf),
        ]:
            for idx, (s, e) in enumerate(intervals, start=1):
                debug_rows.append({
                    'uniprot_ac': ac,
                    'interval_type': label,
                    'interval_index': idx,
                    'start': s,
                    'end': e,
                    'length': e - s + 1,
                })

    per_protein = pd.DataFrame(rows)
    debug_df = pd.DataFrame(debug_rows)
    return per_protein, debug_df


def summarize(per_protein: pd.DataFrame, scope: str) -> pd.DataFrame:
    n_proteins = len(per_protein)
    pf_len = int(per_protein['Pfam_covered_length'].fillna(0).sum())
    dis_len = int(per_protein['Disorder_length_within_Pfam'].fillna(0).sum())
    all_len = int(per_protein['AllTerm_length_within_Pfam'].fillna(0).sum())

    summary = {
        'protein_scope': scope,
        'n_proteins': n_proteins,
        'total_Pfam_covered_length': pf_len,
        'total_Disorder_length_within_Pfam': dis_len,
        'total_AllTerm_length_within_Pfam': all_len,
        'global_Disorder_fraction_within_Pfam': (dis_len / pf_len) if pf_len else pd.NA,
        'global_AllTerm_fraction_within_Pfam': (all_len / pf_len) if pf_len else pd.NA,
        'median_Disorder_fraction_within_Pfam': per_protein['Disorder_fraction_within_Pfam'].dropna().median(),
        'median_AllTerm_fraction_within_Pfam': per_protein['AllTerm_fraction_within_Pfam'].dropna().median(),
        'mean_Disorder_fraction_within_Pfam': per_protein['Disorder_fraction_within_Pfam'].dropna().mean(),
        'mean_AllTerm_fraction_within_Pfam': per_protein['AllTerm_fraction_within_Pfam'].dropna().mean(),
    }
    return pd.DataFrame([summary])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compute disorder baseline within Pfam-covered sequence space.')
    p.add_argument('--bundle-zip', type=Path, default=None, help='Project zip bundle containing the required CSV/TSV files.')
    p.add_argument('--disprot-tsv', type=Path, default=None, help='Path to DisProt TSV.')
    p.add_argument('--detail-csv', type=Path, default=None, help='Path to seed-match detail CSV.')
    p.add_argument('--mapped-csv', type=Path, default=None, help='Path to all_mapped_segments_with_overlap_status.csv.')
    p.add_argument('--scope', choices=['mapped', 'detail_all', 'detail_covered_only'], default='mapped',
                   help='Protein universe. mapped = proteins that entered segment mapping (recommended).')
    p.add_argument('--out-dir', type=Path, required=True, help='Output directory.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args)
    per_protein, debug_df = compute_baseline_table(inputs, scope=args.scope)
    summary_df = summarize(per_protein, scope=args.scope)

    per_path = args.out_dir / 'per_protein_pfam_disorder_baseline.csv'
    sum_path = args.out_dir / 'global_pfam_disorder_baseline_summary.csv'
    dbg_path = args.out_dir / 'merged_intervals_debug.csv'

    per_protein.to_csv(per_path, index=False, encoding='utf-8-sig')
    summary_df.to_csv(sum_path, index=False, encoding='utf-8-sig')
    debug_df.to_csv(dbg_path, index=False, encoding='utf-8-sig')

    print(f'[INFO] Protein scope: {args.scope}')
    print(f'[INFO] Proteins analyzed: {len(per_protein)}')
    print(f'[INFO] Per-protein baseline file: {per_path}')
    print(f'[INFO] Global summary file: {sum_path}')
    print(f'[INFO] Debug intervals file: {dbg_path}')
    if len(summary_df):
        row = summary_df.iloc[0]
        print(f"[INFO] global_Disorder_fraction_within_Pfam: {row['global_Disorder_fraction_within_Pfam']}")
        print(f"[INFO] global_AllTerm_fraction_within_Pfam: {row['global_AllTerm_fraction_within_Pfam']}")


if __name__ == '__main__':
    main()
