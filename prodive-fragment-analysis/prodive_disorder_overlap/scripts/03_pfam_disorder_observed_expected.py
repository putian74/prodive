#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare observed DisProt overlap with matched randomized fragments."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

Interval = Tuple[int, int]


def read_table(path: Path) -> pd.DataFrame:
    path_str = str(path).lower()
    if path_str.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def norm_col(s: str) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    norm_map = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        c = norm_map.get(norm_col(cand))
        if c is not None:
            return c
    if required:
        raise KeyError(f"Could not find required column among candidates={candidates}. Available columns={list(df.columns)}")
    return None


def merge_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    ints = sorted((int(a), int(b)) for a, b in intervals if pd.notna(a) and pd.notna(b) and int(a) <= int(b))
    if not ints:
        return []
    merged: List[Interval] = [ints[0]]
    for s, e in ints[1:]:
        ls, le = merged[-1]
        if s <= le + 1:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def intervals_length(intervals: Sequence[Interval]) -> int:
    return sum(e - s + 1 for s, e in intervals)


def intersect_two_lists(a: Sequence[Interval], b: Sequence[Interval]) -> List[Interval]:
    out: List[Interval] = []
    i = j = 0
    a = sorted(a)
    b = sorted(b)
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s = max(s1, s2)
        e = min(e1, e2)
        if s <= e:
            out.append((s, e))
        if e1 < e2:
            i += 1
        else:
            j += 1
    return merge_intervals(out)


@dataclass
class LoadedData:
    disprot: pd.DataFrame
    detail: pd.DataFrame
    mapped: pd.DataFrame


def load_inputs(args: argparse.Namespace) -> LoadedData:
    disprot = read_table(Path(args.disprot_tsv))
    detail = read_table(Path(args.detail_csv))
    mapped = read_table(Path(args.mapped_csv))
    return LoadedData(disprot=disprot, detail=detail, mapped=mapped)


def prepare_disprot(df: pd.DataFrame) -> pd.DataFrame:
    ac_col = find_column(df, ["UniProt ACC", "uniprot acc", "uniprot_ac", "uniprot"])
    start_col = find_column(df, ["Start", "region_start", "start"])
    end_col = find_column(df, ["End", "region_end", "end"])
    term_col = find_column(df, ["Term name", "term_name", "term name"])

    out = df[[ac_col, start_col, end_col, term_col]].copy()
    out.columns = ["uniprot_ac_base", "region_start", "region_end", "term_name"]
    out["uniprot_ac_base"] = out["uniprot_ac_base"].astype(str).str.strip()
    out["term_name"] = out["term_name"].astype(str).str.strip()
    out["region_start"] = pd.to_numeric(out["region_start"], errors="coerce").astype("Int64")
    out["region_end"] = pd.to_numeric(out["region_end"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["uniprot_ac_base", "region_start", "region_end"])
    out = out[out["region_start"] <= out["region_end"]].copy()
    return out


def prepare_detail(df: pd.DataFrame) -> pd.DataFrame:
    ac_col = find_column(df, ["uniprot_ac_base", "UniProt ACC", "uniprot acc"])
    seed_start_col = find_column(df, ["seed_start", "Seed_Start", "seed start"])
    seed_end_col = find_column(df, ["seed_end", "Seed_End", "seed end"])

    out = df[[ac_col, seed_start_col, seed_end_col]].copy()
    out.columns = ["uniprot_ac_base", "seed_start", "seed_end"]
    out["uniprot_ac_base"] = out["uniprot_ac_base"].astype(str).str.strip()
    out["seed_start"] = pd.to_numeric(out["seed_start"], errors="coerce").astype("Int64")
    out["seed_end"] = pd.to_numeric(out["seed_end"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["uniprot_ac_base", "seed_start", "seed_end"])
    out = out[out["seed_start"] <= out["seed_end"]].copy()
    return out


def prepare_mapped(df: pd.DataFrame) -> pd.DataFrame:
    ac_col = find_column(df, ["uniprot_ac_base", "UniProt ACC", "uniprot acc"])
    start_col = find_column(df, ["Target_Seq_Start", "target_seq_start", "target seq start"])
    end_col = find_column(df, ["Target_Seq_End", "target_seq_end", "target seq end"])
    map_status_col = find_column(df, ["Map_Status", "map_status", "map status"], required=False)

    cols = [ac_col, start_col, end_col]
    if map_status_col is not None:
        cols.append(map_status_col)

    out = df[cols].copy()
    rename = {ac_col: "uniprot_ac_base", start_col: "target_seq_start", end_col: "target_seq_end"}
    if map_status_col is not None:
        rename[map_status_col] = "map_status"
    out = out.rename(columns=rename)

    out["uniprot_ac_base"] = out["uniprot_ac_base"].astype(str).str.strip()
    out["target_seq_start"] = pd.to_numeric(out["target_seq_start"], errors="coerce").astype("Int64")
    out["target_seq_end"] = pd.to_numeric(out["target_seq_end"], errors="coerce").astype("Int64")

    if "map_status" in out.columns:
        out["map_status"] = out["map_status"].astype(str).str.strip()
        out = out[out["map_status"].str.upper() == "OK"].copy()

    out = out.dropna(subset=["uniprot_ac_base", "target_seq_start", "target_seq_end"])
    out = out[out["target_seq_start"] <= out["target_seq_end"]].copy()
    return out


def group_intervals(df: pd.DataFrame, protein_col: str, start_col: str, end_col: str) -> Dict[str, List[Interval]]:
    out: Dict[str, List[Interval]] = {}
    for ac, sub in df.groupby(protein_col):
        intervals = list(zip(sub[start_col].astype(int), sub[end_col].astype(int)))
        out[ac] = merge_intervals(intervals)
    return out


def build_annotation_intervals(disprot_df: pd.DataFrame, proteins: Sequence[str]) -> Tuple[Dict[str, List[Interval]], Dict[str, List[Interval]]]:
    sub = disprot_df[disprot_df["uniprot_ac_base"].isin(proteins)].copy()
    disorder_df = sub[sub["term_name"].str.lower() == "disorder"].copy()
    allterm_df = sub.copy()
    disorder_ints = group_intervals(disorder_df, "uniprot_ac_base", "region_start", "region_end")
    allterm_ints = group_intervals(allterm_df, "uniprot_ac_base", "region_start", "region_end")
    return disorder_ints, allterm_ints


def build_pfam_intervals(detail_df: pd.DataFrame, proteins: Sequence[str]) -> Dict[str, List[Interval]]:
    sub = detail_df[detail_df["uniprot_ac_base"].isin(proteins)].copy()
    return group_intervals(sub, "uniprot_ac_base", "seed_start", "seed_end")


def build_observed_mapped_intervals(mapped_df: pd.DataFrame, proteins: Sequence[str]) -> Tuple[Dict[str, List[Interval]], Dict[str, List[int]]]:
    sub = mapped_df[mapped_df["uniprot_ac_base"].isin(proteins)].copy()
    interval_dict = group_intervals(sub, "uniprot_ac_base", "target_seq_start", "target_seq_end")
    lengths: Dict[str, List[int]] = {}
    for ac, group in sub.groupby("uniprot_ac_base"):
        lengths[ac] = (group["target_seq_end"].astype(int) - group["target_seq_start"].astype(int) + 1).tolist()
    return interval_dict, lengths


def compute_residue_level_observed(
    proteins: Sequence[str],
    pfam_ints: Dict[str, List[Interval]],
    disorder_ints: Dict[str, List[Interval]],
    allterm_ints: Dict[str, List[Interval]],
    mapped_observed_ints: Dict[str, List[Interval]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_rows = []
    debug_rows = []
    total_mapped_union = 0
    total_obs_dis = 0
    total_obs_all = 0

    for ac in sorted(proteins):
        pfam_cov = pfam_ints.get(ac, [])
        disorder = disorder_ints.get(ac, [])
        allterm = allterm_ints.get(ac, [])
        mapped = mapped_observed_ints.get(ac, [])

        mapped_union = merge_intervals(mapped)
        mapped_union_len = intervals_length(mapped_union)
        obs_dis_ints = intersect_two_lists(mapped_union, disorder)
        obs_all_ints = intersect_two_lists(mapped_union, allterm)
        obs_dis_len = intervals_length(obs_dis_ints)
        obs_all_len = intervals_length(obs_all_ints)

        total_mapped_union += mapped_union_len
        total_obs_dis += obs_dis_len
        total_obs_all += obs_all_len

        per_rows.append({
            "uniprot_ac": ac,
            "Pfam_covered_length": intervals_length(pfam_cov),
            "Mapped_residue_union_length": mapped_union_len,
            "Observed_Disorder_overlap_residues": obs_dis_len,
            "Observed_AllTerm_overlap_residues": obs_all_len,
            "Observed_Disorder_fraction": (obs_dis_len / mapped_union_len) if mapped_union_len > 0 else math.nan,
            "Observed_AllTerm_fraction": (obs_all_len / mapped_union_len) if mapped_union_len > 0 else math.nan,
        })

        for tag, intervals in [
            ("Pfam_covered", pfam_cov),
            ("Mapped_union", mapped_union),
            ("Observed_Disorder_overlap", obs_dis_ints),
            ("Observed_AllTerm_overlap", obs_all_ints),
        ]:
            for s, e in intervals:
                debug_rows.append({
                    "uniprot_ac": ac,
                    "interval_type": tag,
                    "start": s,
                    "end": e,
                    "length": e - s + 1,
                })

    per_df = pd.DataFrame(per_rows)
    summary_df = pd.DataFrame([{
        "n_proteins": len(proteins),
        "total_Mapped_residue_union_length": total_mapped_union,
        "total_Observed_Disorder_overlap_residues": total_obs_dis,
        "total_Observed_AllTerm_overlap_residues": total_obs_all,
        "global_Observed_Disorder_fraction": (total_obs_dis / total_mapped_union) if total_mapped_union > 0 else math.nan,
        "global_Observed_AllTerm_fraction": (total_obs_all / total_mapped_union) if total_mapped_union > 0 else math.nan,
    }])
    debug_df = pd.DataFrame(debug_rows)
    return per_df, summary_df, debug_df


def interval_can_fit(interval: Interval, seg_len: int) -> bool:
    s, e = interval
    return (e - s + 1) >= seg_len


def sample_one_segment_within_intervals(intervals: Sequence[Interval], seg_len: int, rng: random.Random) -> Optional[Interval]:
    fit_blocks = [(s, e) for s, e in intervals if interval_can_fit((s, e), seg_len)]
    if not fit_blocks:
        return None
    weights = []
    total_positions = 0
    for s, e in fit_blocks:
        npos = (e - s + 1) - seg_len + 1
        weights.append((s, e, npos))
        total_positions += npos
    pick = rng.randint(1, total_positions)
    cum = 0
    for s, e, npos in weights:
        cum += npos
        if pick <= cum:
            start_offset = pick - (cum - npos) - 1
            start = s + start_offset
            end = start + seg_len - 1
            return (start, end)
    return None


def randomize_segments_for_protein(pfam_intervals: Sequence[Interval], seg_lengths: Sequence[int], rng: random.Random) -> List[Interval]:
    sampled = []
    for L in seg_lengths:
        seg = sample_one_segment_within_intervals(pfam_intervals, int(L), rng)
        if seg is not None:
            sampled.append(seg)
    return merge_intervals(sampled)


def compute_randomized_control(
    proteins: Sequence[str],
    pfam_ints: Dict[str, List[Interval]],
    disorder_ints: Dict[str, List[Interval]],
    allterm_ints: Dict[str, List[Interval]],
    observed_seg_lengths: Dict[str, List[int]],
    global_observed_disorder_fraction: float,
    global_observed_allterm_fraction: float,
    n_permutations: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    dist_rows = []

    for perm_ix in range(1, n_permutations + 1):
        total_union = 0
        total_dis = 0
        total_all = 0

        for ac in proteins:
            pfam_cov = pfam_ints.get(ac, [])
            seg_lengths = observed_seg_lengths.get(ac, [])
            if not pfam_cov or not seg_lengths:
                continue
            sampled_union = randomize_segments_for_protein(pfam_cov, seg_lengths, rng)
            dis = intersect_two_lists(sampled_union, disorder_ints.get(ac, []))
            allterm = intersect_two_lists(sampled_union, allterm_ints.get(ac, []))
            total_union += intervals_length(sampled_union)
            total_dis += intervals_length(dis)
            total_all += intervals_length(allterm)

        dist_rows.append({
            "permutation": perm_ix,
            "Random_union_length": total_union,
            "Random_Disorder_overlap_residues": total_dis,
            "Random_AllTerm_overlap_residues": total_all,
            "Random_Disorder_fraction": (total_dis / total_union) if total_union > 0 else math.nan,
            "Random_AllTerm_fraction": (total_all / total_union) if total_union > 0 else math.nan,
        })

    dist_df = pd.DataFrame(dist_rows)

    def summarize(label: str, obs: float, col: str) -> Dict[str, object]:
        vals = dist_df[col].dropna().tolist()
        mean = float(pd.Series(vals).mean()) if vals else math.nan
        sd = float(pd.Series(vals).std(ddof=1)) if len(vals) > 1 else math.nan
        p_enrich = ((sum(v >= obs for v in vals) + 1) / (len(vals) + 1)) if vals else math.nan
        p_deplete = ((sum(v <= obs for v in vals) + 1) / (len(vals) + 1)) if vals else math.nan
        return {
            "label": label,
            "Observed_fraction": obs,
            "Expected_fraction_mean": mean,
            "Expected_fraction_sd": sd,
            "Observed_over_Expected": (obs / mean) if mean and not math.isnan(mean) and mean != 0 else math.nan,
            "P_enrichment": p_enrich,
            "P_depletion": p_deplete,
            "n_permutations": n_permutations,
        }

    summary_rows = [
        summarize("disorder", global_observed_disorder_fraction, "Random_Disorder_fraction"),
        summarize("all_term", global_observed_allterm_fraction, "Random_AllTerm_fraction"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    return dist_df, summary_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute residue-level observed overlap and randomized expected overlap within Pfam-covered space.")
    p.add_argument("--disprot-tsv", required=True, help="Path to DisProt_current_IDPO.tsv")
    p.add_argument("--detail-csv", required=True, help="Path to disprot_seed_match_detail.csv")
    p.add_argument("--mapped-csv", required=True, help="Path to all_mapped_segments_with_overlap_status.csv")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--n-permutations", type=int, default=1000, help="Number of random permutations")
    p.add_argument("--seed", type=int, default=20260331, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_inputs(args)
    disprot = prepare_disprot(loaded.disprot)
    detail = prepare_detail(loaded.detail)
    mapped = prepare_mapped(loaded.mapped)

    proteins = sorted(set(mapped["uniprot_ac_base"].astype(str)))
    if not proteins:
        raise ValueError("No proteins found in mapped CSV after filtering Map_Status == OK.")

    pfam_ints = build_pfam_intervals(detail, proteins)
    disorder_ints, allterm_ints = build_annotation_intervals(disprot, proteins)
    mapped_observed_ints, observed_seg_lengths = build_observed_mapped_intervals(mapped, proteins)

    per_df, obs_summary_df, obs_debug_df = compute_residue_level_observed(
        proteins=proteins,
        pfam_ints=pfam_ints,
        disorder_ints=disorder_ints,
        allterm_ints=allterm_ints,
        mapped_observed_ints=mapped_observed_ints,
    )

    per_df.to_csv(out_dir / "per_protein_residue_level_observed.csv", index=False, encoding="utf-8-sig")
    obs_summary_df.to_csv(out_dir / "global_residue_level_observed_summary.csv", index=False, encoding="utf-8-sig")
    obs_debug_df.to_csv(out_dir / "residue_level_observed_intervals_debug.csv", index=False, encoding="utf-8-sig")

    global_observed_dis = float(obs_summary_df.loc[0, "global_Observed_Disorder_fraction"])
    global_observed_all = float(obs_summary_df.loc[0, "global_Observed_AllTerm_fraction"])

    dist_df, rand_summary_df = compute_randomized_control(
        proteins=proteins,
        pfam_ints=pfam_ints,
        disorder_ints=disorder_ints,
        allterm_ints=allterm_ints,
        observed_seg_lengths=observed_seg_lengths,
        global_observed_disorder_fraction=global_observed_dis,
        global_observed_allterm_fraction=global_observed_all,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    dist_df.to_csv(out_dir / "randomized_segment_control_distribution.csv", index=False, encoding="utf-8-sig")
    rand_summary_df.to_csv(out_dir / "observed_vs_expected_randomized_summary.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Done.")
    print(f"[INFO] Proteins analyzed: {len(proteins)}")
    print(f"[INFO] Global observed disorder fraction: {global_observed_dis:.6f}")
    print(f"[INFO] Global observed all-term fraction: {global_observed_all:.6f}")
    print(f"[INFO] Output dir: {out_dir}")


if __name__ == "__main__":
    main()
