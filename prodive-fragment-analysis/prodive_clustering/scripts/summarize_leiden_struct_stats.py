#!/usr/bin/env python3
"""Add intra-community SS/RSA summaries to completed Leiden results.

The script reuses the raw-to-merged mapping and node membership tables written
by ``build_leiden_clusters.py``. It does not rebuild the graph or rerun Leiden.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd
from tqdm import tqdm

from build_leiden_clusters import (
    Node,
    RawPair,
    compute_intra_struct_stats,
    estimate_chunks,
    norm_seg,
    parse_details_string,
    parse_input_specs,
)


def _int_or_default(value, default: int = -1) -> int:
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def load_subset_pairs(input_csv: str, chunk_size: int) -> Set[RawPair]:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input subset does not exist: {input_csv}")
    head = pd.read_csv(input_csv, nrows=1)
    required = ["Main_HMM", "Sub_HMM", "Sub_Segments_Details"]
    missing = [column for column in required if column not in head.columns]
    if missing:
        raise ValueError(
            f"Input subset is missing required columns: {missing}; "
            f"current columns: {list(head.columns)}"
        )

    pairs: Set[RawPair] = set()
    reader = pd.read_csv(input_csv, chunksize=chunk_size)
    for chunk in tqdm(
        reader,
        total=estimate_chunks(input_csv, chunk_size),
        desc="Load subset pairs",
    ):
        for row in chunk.itertuples(index=False):
            main_hmm = str(getattr(row, "Main_HMM"))
            sub_hmm = str(getattr(row, "Sub_HMM"))
            for main_seg, sub_seg in parse_details_string(
                getattr(row, "Sub_Segments_Details")
            ):
                pairs.add(
                    (
                        main_hmm,
                        norm_seg(main_seg),
                        sub_hmm,
                        norm_seg(sub_seg),
                    )
                )
    return pairs


def load_clustering_tables(
    case_output_dir: str,
    label: str,
) -> Tuple[
    nx.Graph,
    Dict[Node, Node],
    Dict[Node, int],
    Dict[Node, int],
]:
    mapping_csv = os.path.join(case_output_dir, f"{label}_raw_to_merged_map.csv")
    nodes_csv = os.path.join(case_output_dir, f"{label}_nodes_LEIDEN.csv")
    for path in [mapping_csv, nodes_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required clustering output does not exist: {path}. "
                "Run build_leiden_clusters.py with node-table export enabled."
            )

    mapping_df = pd.read_csv(
        mapping_csv,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    mapping_required = ["HMM_ID", "Raw_Segment", "Merged_Segment"]
    mapping_missing = [c for c in mapping_required if c not in mapping_df.columns]
    if mapping_missing:
        raise ValueError(f"Raw-to-merged table is missing columns: {mapping_missing}")

    original_to_merged_map: Dict[Node, Node] = {}
    for row in mapping_df.itertuples(index=False):
        hmm_id = str(row.HMM_ID).strip()
        raw_seg = norm_seg(row.Raw_Segment)
        merged_seg = norm_seg(row.Merged_Segment)
        original_to_merged_map[(hmm_id, raw_seg)] = (hmm_id, merged_seg)

    nodes_df = pd.read_csv(
        nodes_csv,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    node_required = [
        "HMM_ID",
        "Merged_Segment_Range",
        "Kept_Family_ID",
        "Kept_Family_Size",
    ]
    node_missing = [c for c in node_required if c not in nodes_df.columns]
    if node_missing:
        raise ValueError(f"Leiden node table is missing columns: {node_missing}")

    graph = nx.Graph()
    node_to_kept_id: Dict[Node, int] = {}
    node_to_kept_size: Dict[Node, int] = {}
    for row in nodes_df.itertuples(index=False):
        node = (str(row.HMM_ID).strip(), norm_seg(row.Merged_Segment_Range))
        graph.add_node(node)
        node_to_kept_id[node] = _int_or_default(row.Kept_Family_ID)
        node_to_kept_size[node] = _int_or_default(row.Kept_Family_Size)

    return graph, original_to_merged_map, node_to_kept_id, node_to_kept_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate optional intra-community SS/RSA summaries from completed Leiden outputs."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        action="append",
        help="Input subset in LABEL=/path/file.csv format. Can be repeated.",
    )
    input_group.add_argument(
        "--input-file-table",
        help="TSV/CSV with columns Label and Path.",
    )
    parser.add_argument(
        "--clustering-root",
        required=True,
        help="Output root previously produced by build_leiden_clusters.py.",
    )
    parser.add_argument(
        "--struct-csv",
        required=True,
        help="Pfam SS/RSA annotation table.",
    )
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--node-id-sep", default="||")
    parser.add_argument(
        "--allow-multi-mapping",
        action="store_true",
        help="Use the first parsed mapping when a structure row contains multiple mappings.",
    )
    parser.add_argument(
        "--struct-dedup-unit",
        default="merged_node",
        choices=["merged_node"],
    )
    parser.add_argument(
        "--struct-representative-status-policy",
        default="ok_if_any_else_majority",
        choices=["ok_if_any_else_majority", "majority"],
    )
    parser.add_argument(
        "--struct-representative-label-policy",
        default="majority_among_OK",
        choices=["majority_among_OK"],
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.struct_csv):
        raise FileNotFoundError(f"SS/RSA table does not exist: {args.struct_csv}")
    args.strict_one_to_one = not args.allow_multi_mapping
    inputs = parse_input_specs(args.input, args.input_file_table)

    for label, input_csv in inputs:
        print("=" * 100)
        print(f"Structural summary: {label}")
        case_output_dir = os.path.join(args.clustering_root, label)
        graph, mapping, node_to_kept_id, node_to_kept_size = load_clustering_tables(
            case_output_dir,
            label,
        )
        subset_pairs = load_subset_pairs(input_csv, args.chunk_size)
        compute_intra_struct_stats(
            case_output_dir,
            label,
            graph,
            subset_pairs,
            mapping,
            node_to_kept_id,
            node_to_kept_size,
            args,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
