#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse Start2Fold XML files into a single residue-class table."""

from __future__ import annotations

import argparse
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

DEFAULT_DATASET_DIR = Path("CHANGE_ME")
CLASS_KEYS = ["Fold_EARLY", "Fold_INTER", "Fold_LATE", "Stab_STRONG", "Stab_MEDIUM", "Stab_WEAK"]


def parse_start2fold_xml_dir(directory: Path) -> pd.DataFrame:
    xml_files = sorted(glob.glob(str(directory / "*.xml")))
    print(f"[INFO] XML files found: {len(xml_files)}")
    all_results = []

    for xml_path in xml_files:
        xml_path = Path(xml_path)
        stf_id = xml_path.stem
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            protein_node = root.find("protein")
            if protein_node is None:
                continue

            uniprot_id = protein_node.get("uniprot_id")
            uniprot_range = protein_node.get("uniprot_range")
            data_dict = {key: set() for key in CLASS_KEYS}

            for experiment in protein_node.findall("experiment"):
                method_node = experiment.find("method")
                protection_node = experiment.find("protection")
                if method_node is None or protection_node is None:
                    continue

                exp_type = method_node.get("type")
                level = protection_node.get("protection_level")

                for residue in experiment.findall("residue"):
                    idx = int(residue.get("index"))
                    if exp_type == "folding":
                        if level == "EARLY":
                            data_dict["Fold_EARLY"].add(idx)
                        elif level == "INTERMEDIATE":
                            data_dict["Fold_INTER"].add(idx)
                        elif level == "LATE":
                            data_dict["Fold_LATE"].add(idx)
                    elif exp_type == "stability":
                        if level == "STRONG":
                            data_dict["Stab_STRONG"].add(idx)
                        elif level == "MEDIUM":
                            data_dict["Stab_MEDIUM"].add(idx)
                        elif level == "WEAK":
                            data_dict["Stab_WEAK"].add(idx)

            row_data = {"STF_ID": stf_id, "UniProt_ID": uniprot_id, "UniProt_Range": uniprot_range}
            for key in CLASS_KEYS:
                row_data[f"{key}_Count"] = len(data_dict[key])
                row_data[f"{key}_List"] = str(sorted(data_dict[key]))
            all_results.append(row_data)
        except Exception as exc:
            print(f"[WARN] failed to parse {xml_path.name}: {exc}")

    df = pd.DataFrame(all_results)
    if not df.empty:
        df = df.sort_values(by="STF_ID").reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse Start2Fold XML files into parsed_start2fold_all_classes.csv.")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Directory containing Start2Fold XML files.")
    p.add_argument("--output-csv", type=Path, default=None, help="Output CSV path. Default: <dataset-dir>/parsed_start2fold_all_classes.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv or (args.dataset_dir / "parsed_start2fold_all_classes.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stf_df = parse_start2fold_xml_dir(args.dataset_dir)
    stf_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("=" * 60)
    print(f"[INFO] Parsed table saved to: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
