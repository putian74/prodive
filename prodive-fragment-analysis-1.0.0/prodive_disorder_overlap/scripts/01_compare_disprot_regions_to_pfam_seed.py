#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

DISPROT_TSV: Path
PFAM_SEED_DIR: Path
OUT_DIR: Path



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_text(x) -> str:
    if pd.isna(x):
        return ''
    return str(x).replace('\xa0', ' ').strip()


def strip_version(ac: str) -> str:
    s = clean_text(ac)
    return s.split('.', 1)[0] if s else ''


def normalize_term_name(x) -> str:
    return clean_text(x).casefold()


def interval_overlap(a1: int, a2: int, b1: int, b2: int) -> Optional[Tuple[int, int]]:
    lo = max(a1, b1)
    hi = min(a2, b2)
    if lo <= hi:
        return lo, hi
    return None


def read_disprot_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', dtype=str)
    required_cols = ['UniProt ACC', 'DisProt ID', 'Region ID', 'Start', 'End']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in DisProt file: {missing}')

    for col in df.columns:
        df[col] = df[col].map(clean_text)

    df['uniprot_ac_raw'] = df['UniProt ACC']
    df['uniprot_ac_base'] = df['uniprot_ac_raw'].map(strip_version)
    df['region_start'] = pd.to_numeric(df['Start'], errors='coerce')
    df['region_end'] = pd.to_numeric(df['End'], errors='coerce')
    if 'Term name' in df.columns:
        df['term_name_norm'] = df['Term name'].map(normalize_term_name)
    else:
        df['term_name_norm'] = ''

    df = df[df['uniprot_ac_base'] != ''].copy()
    df = df[df['region_start'].notna() & df['region_end'].notna()].copy()
    df['region_start'] = df['region_start'].astype(int)
    df['region_end'] = df['region_end'].astype(int)

    swap_mask = df['region_start'] > df['region_end']
    if swap_mask.any():
        tmp = df.loc[swap_mask, 'region_start'].copy()
        df.loc[swap_mask, 'region_start'] = df.loc[swap_mask, 'region_end']
        df.loc[swap_mask, 'region_end'] = tmp

    return df.reset_index(drop=True)


GS_AC_RE = re.compile(r'^#=GS\s+(\S+)\s+AC\s+(\S+)')
SEQ_RANGE_RE = re.compile(r'^(?P<seqid>.+?)/(\d+)-(\d+)$')


def parse_sto_ac_rows(sto_path: Path) -> List[dict]:
    pfam_id = sto_path.parent.name
    rows: List[dict] = []
    with sto_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = GS_AC_RE.match(line.strip())
            if not m:
                continue
            seq_name = m.group(1).strip()
            ac = m.group(2).strip()
            ac_base = strip_version(ac)

            seq_id = seq_name
            seed_start = None
            seed_end = None
            m2 = SEQ_RANGE_RE.match(seq_name)
            if m2:
                seq_id = m2.group('seqid')
                seed_start = int(m2.group(2))
                seed_end = int(m2.group(3))

            rows.append({
                'pfam_id': pfam_id,
                'sto_file': str(sto_path),
                'seed_seq_name': seq_name,
                'seed_seq_id': seq_id,
                'seed_ac_raw': ac,
                'seed_ac_base': ac_base,
                'seed_start': seed_start,
                'seed_end': seed_end,
            })
    return rows


def scan_all_seed_ac(seed_root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    sto_files = sorted(seed_root.glob('PF*/PF*.sto'))
    if not sto_files:
        raise FileNotFoundError(f'No sto files found under: {seed_root}')

    for i, sto in enumerate(sto_files, start=1):
        if i % 1000 == 0 or i == len(sto_files):
            print(f'[INFO] scanned sto files: {i}/{len(sto_files)}')
        rows.extend(parse_sto_ac_rows(sto))

    seed_df = pd.DataFrame(rows)
    if seed_df.empty:
        raise RuntimeError('No AC rows parsed from sto files.')
    return seed_df


def match_disprot_to_seed(disprot_df: pd.DataFrame, seed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_by_ac: Dict[str, List[dict]] = defaultdict(list)
    for row in seed_df.to_dict('records'):
        seed_by_ac[row['seed_ac_base']].append(row)

    detail_rows: List[dict] = []
    matched_region_count = 0
    exact_ac_region_count = 0
    overlap_region_count = 0

    for idx, drow in disprot_df.iterrows():
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(disprot_df):
            print(f'[INFO] processed DisProt rows: {idx+1}/{len(disprot_df)}')

        ac = drow['uniprot_ac_base']
        candidates = seed_by_ac.get(ac, [])
        if candidates:
            exact_ac_region_count += 1

        row_has_any_overlap = False
        for srow in candidates:
            seed_start = srow.get('seed_start')
            seed_end = srow.get('seed_end')
            overlap_start = None
            overlap_end = None
            overlap_len = 0
            has_seed_range = seed_start is not None and seed_end is not None

            if has_seed_range:
                ov = interval_overlap(int(drow['region_start']), int(drow['region_end']), int(seed_start), int(seed_end))
                if ov is not None:
                    overlap_start, overlap_end = ov
                    overlap_len = overlap_end - overlap_start + 1
                    if overlap_len > 0:
                        row_has_any_overlap = True
            detail_rows.append({
                'disprot_row_index_1based': idx + 1,
                'disprot_id': drow['DisProt ID'],
                'region_id': drow['Region ID'],
                'uniprot_ac_raw': drow['uniprot_ac_raw'],
                'uniprot_ac_base': ac,
                'protein_name': drow.get('Protein name', ''),
                'gene_name': drow.get('Gene name', ''),
                'organism': drow.get('Organism', ''),
                'term_namespace': drow.get('Term namespace', ''),
                'term_id': drow.get('Term ID', ''),
                'term_name': drow.get('Term name', ''),
                'term_name_norm': drow.get('term_name_norm', ''),
                'region_start': int(drow['region_start']),
                'region_end': int(drow['region_end']),
                'region_len': int(drow['region_end']) - int(drow['region_start']) + 1,
                'region_sequence': drow.get('Region sequence', ''),
                'pfam_id': srow['pfam_id'],
                'sto_file': srow['sto_file'],
                'seed_seq_name': srow['seed_seq_name'],
                'seed_seq_id': srow['seed_seq_id'],
                'seed_ac_raw': srow['seed_ac_raw'],
                'seed_ac_base': srow['seed_ac_base'],
                'seed_start': seed_start,
                'seed_end': seed_end,
                'seed_has_range': has_seed_range,
                'overlap_start': overlap_start,
                'overlap_end': overlap_end,
                'overlap_len': overlap_len,
            })

        if candidates:
            matched_region_count += 1
        if row_has_any_overlap:
            overlap_region_count += 1

    detail_df = pd.DataFrame(detail_rows)

    ac_match_df = detail_df.copy() if not detail_df.empty else pd.DataFrame()
    overlap_df = detail_df[detail_df['overlap_len'].fillna(0) > 0].copy() if not detail_df.empty else pd.DataFrame()

    region_summary_rows = []
    for idx, drow in disprot_df.iterrows():
        rid = drow['Region ID']
        sub_all = ac_match_df[ac_match_df['region_id'] == rid] if not ac_match_df.empty else pd.DataFrame()
        sub_hit = overlap_df[overlap_df['region_id'] == rid] if not overlap_df.empty else pd.DataFrame()
        region_summary_rows.append({
            'disprot_row_index_1based': idx + 1,
            'disprot_id': drow['DisProt ID'],
            'region_id': rid,
            'uniprot_ac_raw': drow['uniprot_ac_raw'],
            'uniprot_ac_base': drow['uniprot_ac_base'],
            'protein_name': drow.get('Protein name', ''),
            'gene_name': drow.get('Gene name', ''),
            'organism': drow.get('Organism', ''),
            'term_namespace': drow.get('Term namespace', ''),
            'term_id': drow.get('Term ID', ''),
            'term_name': drow.get('Term name', ''),
            'term_name_norm': drow.get('term_name_norm', ''),
            'region_start': int(drow['region_start']),
            'region_end': int(drow['region_end']),
            'region_len': int(drow['region_end']) - int(drow['region_start']) + 1,
            'num_seed_ac_matches': int(len(sub_all)),
            'num_seed_overlap_hits': int(len(sub_hit)),
            'has_seed_ac_match': int(len(sub_all) > 0),
            'has_seed_overlap_hit': int(len(sub_hit) > 0),
            'matched_pfams': ';'.join(sorted(sub_hit['pfam_id'].astype(str).unique().tolist())) if not sub_hit.empty else '',
            'matched_seed_seq_names': ';'.join(sorted(sub_hit['seed_seq_name'].astype(str).unique().tolist())) if not sub_hit.empty else '',
            'best_overlap_len': int(sub_hit['overlap_len'].max()) if not sub_hit.empty else 0,
        })

    region_summary_df = pd.DataFrame(region_summary_rows)

    pfam_summary_df = pd.DataFrame()
    if not overlap_df.empty:
        pfam_summary_df = (
            overlap_df.groupby('pfam_id', as_index=False)
            .agg(
                num_overlap_hits=('region_id', 'count'),
                num_unique_regions=('region_id', pd.Series.nunique),
                num_unique_uniprot=('uniprot_ac_base', pd.Series.nunique),
                max_overlap_len=('overlap_len', 'max'),
            )
            .sort_values(['num_unique_regions', 'num_overlap_hits', 'pfam_id'], ascending=[False, False, True])
        )

    print(f'[INFO] DisProt regions total: {len(disprot_df)}')
    print(f'[INFO] Regions with AC match in seed: {exact_ac_region_count}')
    print(f'[INFO] Regions with residue overlap in seed: {overlap_region_count}')

    return detail_df, region_summary_df, pfam_summary_df


def build_covered_region_table(region_summary_df: pd.DataFrame) -> pd.DataFrame:
    if region_summary_df.empty:
        return pd.DataFrame()

    covered_df = region_summary_df[
        (region_summary_df['num_seed_ac_matches'] > 0) |
        (region_summary_df['num_seed_overlap_hits'] > 0)
    ].copy()

    if covered_df.empty:
        return covered_df

    covered_df['coverage_basis'] = covered_df.apply(
        lambda r: 'residue_overlap' if int(r['num_seed_overlap_hits']) > 0 else 'ac_match_only',
        axis=1,
    )
    covered_df['is_term_name_disorder'] = covered_df['term_name_norm'].eq('disorder').astype(int)

    ordered_cols = [
        'disprot_row_index_1based', 'disprot_id', 'region_id', 'uniprot_ac_raw', 'uniprot_ac_base',
        'protein_name', 'gene_name', 'organism', 'term_namespace', 'term_id', 'term_name',
        'is_term_name_disorder', 'region_start', 'region_end', 'region_len',
        'num_seed_ac_matches', 'num_seed_overlap_hits', 'best_overlap_len',
        'coverage_basis', 'matched_pfams', 'matched_seed_seq_names',
    ]
    keep_cols = [c for c in ordered_cols if c in covered_df.columns]
    covered_df = covered_df[keep_cols].sort_values(
        ['num_seed_overlap_hits', 'num_seed_ac_matches', 'best_overlap_len', 'disprot_id', 'region_id'],
        ascending=[False, False, False, True, True],
    )
    return covered_df.reset_index(drop=True)


def build_disorder_only_region_summary(region_summary_df: pd.DataFrame) -> pd.DataFrame:
    if region_summary_df.empty:
        return pd.DataFrame()
    out = region_summary_df[region_summary_df['term_name_norm'] == 'disorder'].copy()
    return out.reset_index(drop=True)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match DisProt annotated regions to Pfam seed sequences by UniProt accession "
            "and residue-level interval overlap."
        )
    )
    parser.add_argument('--disprot-tsv', type=Path, required=True,
                        help='Path to DisProt_current_IDPO.tsv.')
    parser.add_argument('--pfam-seed-dir', type=Path, required=True,
                        help='Root directory containing PFxxxxx/PFxxxxx.sto seed alignments.')
    parser.add_argument('--out-dir', type=Path, required=True,
                        help='Output directory.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global DISPROT_TSV, PFAM_SEED_DIR, OUT_DIR
    DISPROT_TSV = args.disprot_tsv
    PFAM_SEED_DIR = args.pfam_seed_dir
    OUT_DIR = args.out_dir

    ensure_dir(OUT_DIR)

    if not DISPROT_TSV.exists():
        raise FileNotFoundError(f'Missing DisProt TSV: {DISPROT_TSV}')
    if not PFAM_SEED_DIR.exists():
        raise FileNotFoundError(f'Missing Pfam seed directory: {PFAM_SEED_DIR}')

    print('[INFO] Reading DisProt TSV...')
    disprot_df = read_disprot_tsv(DISPROT_TSV)
    print(f'[INFO] Parsed DisProt rows: {len(disprot_df)}')
    print(f'[INFO] Unique UniProt AC: {disprot_df["uniprot_ac_base"].nunique()}')

    print('[INFO] Scanning sto files...')
    seed_df = scan_all_seed_ac(PFAM_SEED_DIR)
    print(f'[INFO] Parsed seed AC rows: {len(seed_df)}')
    print(f'[INFO] Unique seed AC: {seed_df["seed_ac_base"].nunique()}')
    print(f'[INFO] Unique Pfam families: {seed_df["pfam_id"].nunique()}')

    detail_df, region_summary_df, pfam_summary_df = match_disprot_to_seed(disprot_df, seed_df)
    disorder_only_region_summary_df = build_disorder_only_region_summary(region_summary_df)
    covered_region_df = build_covered_region_table(region_summary_df)

    detail_path = OUT_DIR / 'disprot_seed_match_detail.csv'
    region_summary_path = OUT_DIR / 'disprot_region_summary.csv'
    pfam_summary_path = OUT_DIR / 'disprot_pfam_summary.csv'
    disorder_only_region_summary_path = OUT_DIR / 'disprot_region_summary_term_disorder_only.csv'
    covered_region_path = OUT_DIR / 'disprot_regions_covered_in_seed.csv'

    detail_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
    region_summary_df.to_csv(region_summary_path, index=False, encoding='utf-8-sig')
    pfam_summary_df.to_csv(pfam_summary_path, index=False, encoding='utf-8-sig')
    disorder_only_region_summary_df.to_csv(disorder_only_region_summary_path, index=False, encoding='utf-8-sig')
    covered_region_df.to_csv(covered_region_path, index=False, encoding='utf-8-sig')

    summary_lines = [
        f'DisProt rows parsed: {len(disprot_df)}',
        f'Unique DisProt UniProt AC: {disprot_df["uniprot_ac_base"].nunique()}',
        f'Seed AC rows parsed: {len(seed_df)}',
        f'Unique seed AC: {seed_df["seed_ac_base"].nunique()}',
        f'Unique Pfam families parsed: {seed_df["pfam_id"].nunique()}',
        f'Regions with any seed AC match: {(region_summary_df["num_seed_ac_matches"] > 0).sum()}',
        f'Regions with any residue overlap hit: {(region_summary_df["num_seed_overlap_hits"] > 0).sum()}',
        f'Term name = disorder rows: {len(disorder_only_region_summary_df)}',
        f'Covered region rows (AC match or residue overlap): {len(covered_region_df)}',
        f'Detail file: {detail_path}',
        f'Region summary file: {region_summary_path}',
        f'Pfam summary file: {pfam_summary_path}',
        f'Disorder-only region summary file: {disorder_only_region_summary_path}',
        f'Covered regions file: {covered_region_path}',
    ]
    (OUT_DIR / 'summary.txt').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    for line in summary_lines:
        print('[INFO]', line)


if __name__ == '__main__':
    main()
