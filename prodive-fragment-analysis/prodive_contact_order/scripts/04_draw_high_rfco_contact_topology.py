#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Draw structure-position / contact-topology plots for high-rFCO cases.

Input
-----
1. summary-csv: actual_vs_random_summary.csv
   Must contain at least:
     global_row_index, pfam_id, pfam_side, segment_rank,
     target_hmm_start, target_hmm_end,
     rfco, rfco_delta_vs_random_mean, rfco_empirical_p_le,
     lr_contact_fraction, n_fragment_contacts

2. actual-csv: contact_order_results*.csv
   Used only when summary-csv does not already contain:
     pdb_path, chain_id, mapped_pdb_residues, atom_mode,
     cutoff, exclude_near, long_range_threshold

Output
------
One PNG per selected fragment, with 4 panels:
  A. info panel
  B. linear chain position plot
  C. fragment-to-partner topology lines
  D. contact separation distribution

Example
-------
python3 CHANGE_ME \
  --summary-csv CHANGE_ME \
  --actual-csv CHANGE_ME \
  --outdir CHANGE_ME \
  --mode extreme
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEY_COLS = ["global_row_index", "pfam_id", "pfam_side", "segment_rank"]

STRUCTURE_NEEDED_COLS = [
    "pdb_path",
    "chain_id",
    "mapped_pdb_residues",
    "atom_mode",
    "cutoff",
    "exclude_near",
    "long_range_threshold",
]

SUMMARY_REQUIRED_COLS = KEY_COLS + [
    "target_hmm_start",
    "target_hmm_end",
    "rfco",
    "rfco_delta_vs_random_mean",
    "rfco_empirical_p_le",
    "lr_contact_fraction",
    "n_fragment_contacts",
]


# =========================================================
# helpers
# =========================================================

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def sanitize_filename(s: str) -> str:
    bad = '\\/:*?"<>| '
    out = []
    for ch in str(s):
        out.append("_" if ch in bad else ch)
    return "".join(out)


# =========================================================
# PDB parsing and contact computation
# =========================================================

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


class Residue:
    def __init__(self, chain_id: str, resseq: int, icode: str, resname: str):
        self.chain_id = chain_id
        self.resseq = resseq
        self.icode = icode
        self.resname = resname
        self.atoms: Dict[str, Tuple[float, float, float]] = {}

    @property
    def pdb_token(self) -> str:
        return f"{self.resseq}{self.icode}".strip()


def parse_pdb_atom_line(line: str):
    rec = line[0:6].strip()
    if rec not in {"ATOM", "HETATM"}:
        return None
    atom_name = line[12:16].strip()
    altloc = line[16:17].strip()
    resname = line[17:20].strip().upper()
    chain_id = line[21:22].strip()
    try:
        resseq = int(line[22:26].strip())
        icode = line[26:27].strip()
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
    except ValueError:
        return None
    return rec, atom_name, altloc, resname, chain_id, resseq, icode, x, y, z


def altloc_ok(altloc: str) -> bool:
    return altloc in {"", "A", "1"}


def load_chain_residues(pdb_path: Path, chain_id: str, include_hetatm: bool = False) -> List[Residue]:
    residues: Dict[Tuple[int, str], Residue] = {}
    with pdb_path.open("r", errors="ignore") as f:
        for line in f:
            rec = line[0:6].strip()
            if rec == "ENDMDL":
                break
            parsed = parse_pdb_atom_line(line)
            if parsed is None:
                continue
            rec, atom_name, altloc, resname, atom_chain_id, resseq, icode, x, y, z = parsed
            if atom_chain_id != chain_id:
                continue
            if rec == "HETATM" and not include_hetatm:
                continue
            if resname not in AA3_TO_1:
                continue
            if not altloc_ok(altloc):
                continue
            key = (resseq, icode)
            if key not in residues:
                residues[key] = Residue(atom_chain_id, resseq, icode, resname)
            residues[key].atoms[atom_name] = (x, y, z)

    ordered = [residues[k] for k in sorted(residues.keys(), key=lambda t: (t[0], t[1]))]
    if not ordered:
        raise RuntimeError(f"No usable residues found in {pdb_path} chain {chain_id}")
    return ordered


def sqdist(a, b) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def residue_rep_coord(res: Residue, atom_mode: str):
    if atom_mode == "alpha":
        return res.atoms.get("CA")
    if atom_mode == "beta":
        if res.resname == "GLY":
            return res.atoms.get("CA")
        return res.atoms.get("CB")
    raise ValueError("heavy mode does not use a single representative atom")


def heavy_atom_coords(res: Residue):
    out = []
    for atom_name, xyz in res.atoms.items():
        if atom_name.startswith("H"):
            continue
        out.append(xyz)
    return out


def residue_pair_distance(res1: Residue, res2: Residue, atom_mode: str):
    if atom_mode == "heavy":
        c1 = heavy_atom_coords(res1)
        c2 = heavy_atom_coords(res2)
        if not c1 or not c2:
            return None
        best = math.inf
        for a in c1:
            for b in c2:
                d2 = sqdist(a, b)
                if d2 < best:
                    best = d2
        return math.sqrt(best) if math.isfinite(best) else None

    c1 = residue_rep_coord(res1, atom_mode)
    c2 = residue_rep_coord(res2, atom_mode)
    if c1 is None or c2 is None:
        return None
    return math.sqrt(sqdist(c1, c2))


def build_contacts(
    residues: List[Residue],
    atom_mode: str,
    cutoff: float,
    exclude_near: int
) -> List[Tuple[int, int]]:
    n = len(residues)
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) <= exclude_near:
                continue
            d = residue_pair_distance(residues[i], residues[j], atom_mode=atom_mode)
            if d is not None and d <= cutoff:
                contacts.append((i, j))
    return contacts


# =========================================================
# input loading
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Draw structure-position / contact-topology plots for high-rFCO cases.")
    p.add_argument("--summary-csv", required=True, help="actual_vs_random_summary.csv")
    p.add_argument("--actual-csv", required=True, help="contact_order_results*.csv (actual results)")
    p.add_argument("--outdir", required=True)

    p.add_argument("--mode", choices=["loose", "strict", "extreme", "custom"], default="extreme")
    p.add_argument("--max-plots", type=int, default=None)
    p.add_argument("--sort-by", choices=["rfco", "rfco_delta_vs_random_mean", "rfco_empirical_p_le"], default="rfco")

    # thresholds
    p.add_argument("--loose-rfco", type=float, default=0.35)

    p.add_argument("--strict-rfco", type=float, default=0.40)
    p.add_argument("--strict-delta", type=float, default=0.10)
    p.add_argument("--strict-emp-p", type=float, default=0.99)
    p.add_argument("--strict-min-contacts", type=int, default=10)

    p.add_argument("--extreme-rfco", type=float, default=0.50)
    p.add_argument("--extreme-delta", type=float, default=0.15)
    p.add_argument("--extreme-emp-p", type=float, default=0.995)
    p.add_argument("--extreme-min-contacts", type=int, default=10)

    p.add_argument("--custom-rfco", type=float, default=None)
    p.add_argument("--custom-delta", type=float, default=None)
    p.add_argument("--custom-emp-p", type=float, default=None)
    p.add_argument("--custom-min-contacts", type=int, default=None)

    return p.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in SUMMARY_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"summary CSV missing required columns: {missing}")

    num_cols = [
        "global_row_index", "segment_rank",
        "target_hmm_start", "target_hmm_end",
        "rfco", "rfco_delta_vs_random_mean", "rfco_empirical_p_le",
        "lr_contact_fraction", "n_fragment_contacts"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=SUMMARY_REQUIRED_COLS).copy()
    df["global_row_index"] = df["global_row_index"].astype(int)
    df["segment_rank"] = df["segment_rank"].astype(int)
    df["target_hmm_start"] = df["target_hmm_start"].astype(int)
    df["target_hmm_end"] = df["target_hmm_end"].astype(int)
    df["pfam_id"] = df["pfam_id"].astype(str)
    df["pfam_side"] = df["pfam_side"].astype(str)

    # normalize possible structure columns if already present
    for c in STRUCTURE_NEEDED_COLS:
        if c in df.columns:
            if c in ["cutoff"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            elif c in ["exclude_near", "long_range_threshold"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = df[c].fillna("").astype(str)

    return df


def select_subset(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.mode == "loose":
        out = df[df["rfco"] >= args.loose_rfco].copy()

    elif args.mode == "strict":
        out = df[
            (df["rfco"] >= args.strict_rfco) &
            (df["rfco_delta_vs_random_mean"] >= args.strict_delta) &
            (df["rfco_empirical_p_le"] >= args.strict_emp_p) &
            (df["n_fragment_contacts"] >= args.strict_min_contacts)
        ].copy()

    elif args.mode == "extreme":
        out = df[
            (df["rfco"] >= args.extreme_rfco) &
            (df["rfco_delta_vs_random_mean"] >= args.extreme_delta) &
            (df["rfco_empirical_p_le"] >= args.extreme_emp_p) &
            (df["n_fragment_contacts"] >= args.extreme_min_contacts)
        ].copy()

    else:
        out = df.copy()
        if args.custom_rfco is not None:
            out = out[out["rfco"] >= args.custom_rfco]
        if args.custom_delta is not None:
            out = out[out["rfco_delta_vs_random_mean"] >= args.custom_delta]
        if args.custom_emp_p is not None:
            out = out[out["rfco_empirical_p_le"] >= args.custom_emp_p]
        if args.custom_min_contacts is not None:
            out = out[out["n_fragment_contacts"] >= args.custom_min_contacts]
        out = out.copy()

    out = out.sort_values(args.sort_by, ascending=False).reset_index(drop=True)

    if args.max_plots is not None:
        out = out.head(args.max_plots).copy()

    return out


def load_actual(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    req = KEY_COLS + STRUCTURE_NEEDED_COLS
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"actual CSV missing required columns: {missing}")

    for c in ["global_row_index", "segment_rank", "cutoff", "exclude_near", "long_range_threshold"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["global_row_index", "segment_rank", "cutoff", "exclude_near", "long_range_threshold"]).copy()
    df["global_row_index"] = df["global_row_index"].astype(int)
    df["segment_rank"] = df["segment_rank"].astype(int)
    df["exclude_near"] = df["exclude_near"].astype(int)
    df["long_range_threshold"] = df["long_range_threshold"].astype(int)
    df["pfam_id"] = df["pfam_id"].astype(str)
    df["pfam_side"] = df["pfam_side"].astype(str)

    for c in ["pdb_path", "chain_id", "mapped_pdb_residues", "atom_mode"]:
        df[c] = df[c].fillna("").astype(str)

    return df


def ensure_structure_cols(summary_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    """
    If summary_df already contains needed structure columns, use them directly.
    Otherwise merge from actual_df.
    """
    summary_has_all = all(c in summary_df.columns for c in STRUCTURE_NEEDED_COLS)

    if summary_has_all:
        out = summary_df.copy()
    else:
        needed_from_actual = KEY_COLS + STRUCTURE_NEEDED_COLS
        actual_sub = actual_df[needed_from_actual].copy()

        out = summary_df.merge(
            actual_sub,
            on=KEY_COLS,
            how="left",
            suffixes=("", "_act")
        )

    # normalize structure columns
    for c in STRUCTURE_NEEDED_COLS:
        if c not in out.columns and f"{c}_act" in out.columns:
            out[c] = out[f"{c}_act"]

    # final validation
    missing = [c for c in STRUCTURE_NEEDED_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"After merge, still missing structure columns: {missing}")

    # normalize types
    out["pdb_path"] = out["pdb_path"].fillna("").astype(str)
    out["chain_id"] = out["chain_id"].fillna("").astype(str)
    out["mapped_pdb_residues"] = out["mapped_pdb_residues"].fillna("").astype(str)
    out["atom_mode"] = out["atom_mode"].fillna("").astype(str)
    out["cutoff"] = pd.to_numeric(out["cutoff"], errors="coerce")
    out["exclude_near"] = pd.to_numeric(out["exclude_near"], errors="coerce")
    out["long_range_threshold"] = pd.to_numeric(out["long_range_threshold"], errors="coerce")

    return out


def parse_fragment_tokens(text: str) -> List[str]:
    s = safe_str(text)
    if s == "":
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


# =========================================================
# plotting
# =========================================================

def draw_case(
    row: pd.Series,
    residues: List[Residue],
    contacts: List[Tuple[int, int]],
    outpath: Path,
    rank_i: int,
    total_n: int,
):
    token_to_index = {r.pdb_token: i for i, r in enumerate(residues)}

    frag_tokens = parse_fragment_tokens(row["mapped_pdb_residues"])
    frag_idx = sorted({token_to_index[t] for t in frag_tokens if t in token_to_index})
    frag_set = set(frag_idx)

    long_thr = int(row["long_range_threshold"])

    # collect fragment-related contacts
    frag_contacts = []
    partner_idx_all = set()
    partner_idx_long = set()
    separations = []

    for i, j in contacts:
        if i in frag_set or j in frag_set:
            if i in frag_set and j in frag_set:
                fi, pj = i, j
            elif i in frag_set:
                fi, pj = i, j
            else:
                fi, pj = j, i

            sep = abs(j - i)
            is_long = sep >= long_thr
            frag_contacts.append((fi, pj, sep, is_long))
            partner_idx_all.add(pj)
            if is_long:
                partner_idx_long.add(pj)
            separations.append(sep)

    chain_len = len(residues)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_info, ax_linear, ax_links, ax_sep = axes.ravel()

    # A. info
    ax_info.axis("off")
    lines = []
    lines.append(f"[{rank_i}/{total_n}]")
    lines.append(f"PFAM: {safe_str(row['pfam_id'])}")
    lines.append(f"Side: {safe_str(row['pfam_side'])}")
    lines.append(f"HMM interval: {int(row['target_hmm_start'])}-{int(row['target_hmm_end'])}")
    lines.append(f"global_row_index: {int(row['global_row_index'])}")
    lines.append(f"segment_rank: {int(row['segment_rank'])}")
    lines.append("")
    lines.append(f"rFCO = {float(row['rfco']):.6f}")
    lines.append(f"delta = {float(row['rfco_delta_vs_random_mean']):.6f}")
    lines.append(f"empirical_p = {float(row['rfco_empirical_p_le']):.6f}")
    lines.append(f"LR fraction = {float(row['lr_contact_fraction']):.6f}")
    lines.append(f"n_fragment_contacts = {float(row['n_fragment_contacts']):.0f}")
    lines.append("")
    lines.append(f"PDB path: {safe_str(row['pdb_path'])}")
    lines.append(f"Chain: {safe_str(row['chain_id'])}")
    lines.append(f"Atom mode: {safe_str(row['atom_mode'])}")
    lines.append(f"cutoff = {float(row['cutoff'])}")
    lines.append(f"exclude_near = {int(row['exclude_near'])}")
    lines.append(f"long_range_threshold = {int(row['long_range_threshold'])}")
    lines.append(f"chain_length = {chain_len}")
    lines.append(f"mapped_fragment_residue_n = {len(frag_idx)}")
    lines.append(f"all_partner_n = {len(partner_idx_all)}")
    lines.append(f"long_partner_n = {len(partner_idx_long)}")

    ax_info.text(
        0.0, 1.0, "\n".join(lines),
        va="top", ha="left",
        fontsize=10, family="monospace"
    )

    # B. linear chain position plot
    ax_linear.set_title("Linear chain position map")
    ax_linear.set_xlim(0, chain_len + 1)
    ax_linear.set_ylim(0, 1)
    ax_linear.set_yticks([])
    ax_linear.set_xlabel("Residue index in structure chain")
    ax_linear.hlines(0.5, 1, chain_len, color="lightgray", linewidth=2)

    if frag_idx:
        x_frag = np.array(frag_idx) + 1
        ax_linear.scatter(x_frag, np.full_like(x_frag, 0.5, dtype=float), s=50, color="red", label="fragment")

    if partner_idx_all:
        x_part = np.array(sorted(partner_idx_all)) + 1
        ax_linear.scatter(x_part, np.full_like(x_part, 0.62, dtype=float), s=18, color="royalblue", label="all partners")

    if partner_idx_long:
        x_long = np.array(sorted(partner_idx_long)) + 1
        ax_linear.scatter(x_long, np.full_like(x_long, 0.38, dtype=float), s=22, color="navy", label="long-range partners")

    ax_linear.legend(loc="upper right")

    # C. topology lines
    ax_links.set_title("Fragment-to-partner contact topology")
    ax_links.set_xlim(0, chain_len + 1)
    ax_links.set_ylim(0, 1)
    ax_links.set_xlabel("Residue index in structure chain")
    ax_links.set_yticks([0.75, 0.25])
    ax_links.set_yticklabels(["fragment", "partner"])

    ax_links.hlines(0.75, 1, chain_len, color="lightgray", linewidth=1.5)
    ax_links.hlines(0.25, 1, chain_len, color="lightgray", linewidth=1.5)

    if frag_idx:
        x_frag = np.array(frag_idx) + 1
        ax_links.scatter(x_frag, np.full_like(x_frag, 0.75, dtype=float), s=40, color="red", zorder=3)

    if partner_idx_all:
        x_part = np.array(sorted(partner_idx_all)) + 1
        ax_links.scatter(x_part, np.full_like(x_part, 0.25, dtype=float), s=18, color="royalblue", zorder=3)

    for fi, pj, sep, is_long in frag_contacts:
        color = "navy" if is_long else "skyblue"
        alpha = 0.75 if is_long else 0.35
        ax_links.plot([fi + 1, pj + 1], [0.75, 0.25], color=color, alpha=alpha, linewidth=1)

    # D. separation distribution
    ax_sep.set_title("Sequence separation of fragment-related contacts")
    if separations:
        ax_sep.hist(separations, bins=30, color="lightgray", edgecolor="black")
        ax_sep.axvline(long_thr, color="red", linestyle="--", linewidth=2, label=f"long-range cutoff = {long_thr}")
        ax_sep.set_xlabel("Sequence separation |i-j|")
        ax_sep.set_ylabel("Count")
        ax_sep.legend()

        sep_arr = np.array(separations, dtype=float)
        txt = (
            f"n = {len(sep_arr)}\n"
            f"mean = {sep_arr.mean():.2f}\n"
            f"median = {np.median(sep_arr):.2f}\n"
            f"max = {sep_arr.max():.0f}"
        )
        ax_sep.text(
            0.98, 0.95, txt,
            transform=ax_sep.transAxes,
            ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
        )
    else:
        ax_sep.text(0.5, 0.5, "No fragment-related contacts found", ha="center", va="center")
        ax_sep.set_xlabel("Sequence separation |i-j|")
        ax_sep.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# =========================================================
# main
# =========================================================

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary(Path(args.summary_csv))
    subset_df = select_subset(summary_df, args)
    subset_df.to_csv(outdir / "selected_subset.csv", index=False)

    if subset_df.empty:
        (outdir / "summary.txt").write_text("No rows selected.\n", encoding="utf-8")
        return

    actual_df = load_actual(Path(args.actual_csv))
    merged = ensure_structure_cols(subset_df, actual_df)

    missing_actual = (
        merged["pdb_path"].eq("") |
        merged["chain_id"].eq("") |
        merged["mapped_pdb_residues"].eq("") |
        merged["atom_mode"].eq("") |
        merged["cutoff"].isna() |
        merged["exclude_near"].isna() |
        merged["long_range_threshold"].isna()
    ).sum()

    lines = []
    lines.append(f"selected_rows = {len(subset_df)}")
    lines.append(f"rows_missing_structure_info = {missing_actual}")
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_rows = merged[
        (merged["pdb_path"] != "") &
        (merged["chain_id"] != "") &
        (merged["mapped_pdb_residues"] != "") &
        (merged["atom_mode"] != "") &
        (~merged["cutoff"].isna()) &
        (~merged["exclude_near"].isna()) &
        (~merged["long_range_threshold"].isna())
    ].copy()

    total_n = len(plot_rows)
    for idx, row in plot_rows.reset_index(drop=True).iterrows():
        pdb_path = Path(str(row["pdb_path"]))
        if not pdb_path.exists():
            print(f"[WARN] missing pdb file: {pdb_path}")
            continue

        try:
            residues = load_chain_residues(
                pdb_path=pdb_path,
                chain_id=str(row["chain_id"]),
                include_hetatm=False,
            )
            contacts = build_contacts(
                residues=residues,
                atom_mode=str(row["atom_mode"]),
                cutoff=float(row["cutoff"]),
                exclude_near=int(row["exclude_near"]),
            )

            fname = (
                f"{idx+1:04d}_"
                f"{sanitize_filename(row['pfam_id'])}_"
                f"{int(row['target_hmm_start'])}-{int(row['target_hmm_end'])}_"
                f"row{int(row['global_row_index'])}_seg{int(row['segment_rank'])}.png"
            )

            draw_case(
                row=row,
                residues=residues,
                contacts=contacts,
                outpath=outdir / fname,
                rank_i=idx + 1,
                total_n=total_n,
            )

        except Exception as e:
            print(f"[WARN] failed for row {idx+1}: {e}")


if __name__ == "__main__":
    main()