#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build ProDive fragment-community graphs and run Leiden clustering.

Intra-community SS/RSA summaries are generated separately by
``summarize_leiden_struct_stats.py`` and are not required for clustering.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community.quality import modularity as nx_modularity
from sklearn.cluster import DBSCAN
from tqdm import tqdm

Node = Tuple[str, str]
RawPair = Tuple[str, str, str, str]

COL_MAIN_HMM = "Main_HMM"
COL_SUB_HMM = "Sub_HMM"
COL_DETAILS = "Sub_Segments_Details"
COL_MAIN_STATUS = "Main_Status"
COL_SUB_STATUS = "Sub_Status"
COL_MAIN_SS = "Main_SS_Class"
COL_SUB_SS = "Sub_SS_Class"
COL_MAIN_RSA = "Main_Avg_RSA"
COL_SUB_RSA = "Sub_Avg_RSA"
COL_MAIN_LOC = "Main_Location"
COL_SUB_LOC = "Sub_Location"
COL_MAIN_UID = "Main_Selected_UID"
COL_SUB_UID = "Sub_Selected_UID"
COL_MAIN_MSA_START = "Main_MSA_Start"
COL_MAIN_MSA_END = "Main_MSA_End"
COL_SUB_MSA_START = "Sub_MSA_Start"
COL_SUB_MSA_END = "Sub_MSA_End"
COL_MAIN_SEQ_START = "Main_Selected_Seq_Start"
COL_MAIN_SEQ_END = "Main_Selected_Seq_End"
COL_MAIN_SEQ_LEN = "Main_Selected_Seq_Len"
COL_SUB_SEQ_START = "Sub_Selected_Seq_Start"
COL_SUB_SEQ_END = "Sub_Selected_Seq_End"
COL_SUB_SEQ_LEN = "Sub_Selected_Seq_Len"
COL_MAIN_PDB_TYPE = "Main_PDB_Type"
COL_MAIN_PDB_ID = "Main_PDB_ID"
COL_MAIN_CHAIN = "Main_Chain_Used"
COL_SUB_PDB_TYPE = "Sub_PDB_Type"
COL_SUB_PDB_ID = "Sub_PDB_ID"
COL_SUB_CHAIN = "Sub_Chain_Used"


def safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_float_nan(x) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def safe_int_nan(x) -> float:
    try:
        if x is None:
            return np.nan
        s = str(x).strip()
        if s == "" or s.upper() == "NA":
            return np.nan
        return int(float(s))
    except Exception:
        return np.nan


def normalize_label(x, default: str = "NA") -> str:
    s = str(x).strip()
    return s if s else default


def normalize_optional_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_interval(s) -> Optional[Tuple[int, int]]:
    try:
        txt = str(s).strip().replace(" ", "")
        m = re.match(r"^(\d+)-(\d+)$", txt)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return (a, b) if a <= b else (b, a)
        m = re.match(r"^(\d+)$", txt)
        if m:
            v = int(m.group(1))
            return v, v
    except Exception:
        return None
    return None


def norm_seg(seg_str: str) -> str:
    iv = parse_interval(seg_str)
    if iv is None:
        return str(seg_str).strip().replace(" ", "")
    return f"{iv[0]}-{iv[1]}"


def parse_details_string(details_str) -> List[Tuple[str, str]]:
    if pd.isna(details_str):
        return []
    text = str(details_str).strip()
    if not text:
        return []
    out: List[Tuple[str, str]] = []
    for piece in re.split(r"\s*,\s*", text):
        m = re.match(r"^(\d+(?:-\d+)?)\s*->\s*(\d+(?:-\d+)?)$", piece.strip())
        if m:
            out.append((norm_seg(m.group(1)), norm_seg(m.group(2))))
    return out


def get_binned_node_name(hmm_id, segment_str, bin_size: int) -> Node:
    start_pos = 0
    iv = parse_interval(segment_str)
    if iv:
        start_pos = iv[0]
    bin_id = (start_pos // bin_size) * bin_size
    return str(hmm_id), f"bin_{bin_id}"


def merge_intervals_logic(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[int, int]] = []
    curr_start, curr_end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    return merged


def segment_sort_key(item: Node):
    hmm_id, seg_str = item
    iv = parse_interval(seg_str)
    if iv is None:
        return str(hmm_id), 10**18, 10**18
    return str(hmm_id), iv[0], iv[1]


def consolidate_family_nodes(family_nodes: Iterable[Node]) -> List[Node]:
    hmm_groups: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for hmm_id, seg_str in family_nodes:
        iv = parse_interval(seg_str)
        if iv:
            hmm_groups[str(hmm_id)].append(iv)
    consolidated: List[Node] = []
    for hmm_id, intervals in hmm_groups.items():
        for start, end in merge_intervals_logic(list(set(intervals))):
            consolidated.append((hmm_id, f"{start}-{end}"))
    consolidated.sort(key=segment_sort_key)
    return consolidated


def update_edge_max_score(G: nx.Graph, node_a: Node, node_b: Node, score_val: float) -> None:
    if G.has_edge(node_a, node_b):
        old_w = safe_float(G[node_a][node_b].get("weight", 0.0), 0.0)
        if score_val > old_w:
            G[node_a][node_b]["weight"] = score_val
    else:
        G.add_edge(node_a, node_b, weight=score_val)


def compute_smart_mappings(segment_set: Iterable[str], eps: int, min_samples: int, metric: str) -> Dict[str, str]:
    coords: List[List[int]] = []
    original_strings: List[str] = []
    for s in segment_set:
        iv = parse_interval(s)
        if iv:
            coords.append([iv[0], iv[1]])
            original_strings.append(str(s))
    if not coords:
        return {}
    X = np.array(coords, dtype=float)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(X)
    labels = db.labels_
    mapping: Dict[str, str] = {}
    for label in set(labels):
        idx = np.where(labels == label)[0]
        cluster_intervals = X[idx]
        if label == -1:
            for i in idx:
                s, e = map(int, X[i])
                mapping[original_strings[i]] = f"{s}-{e}"
        else:
            min_start = int(np.min(cluster_intervals[:, 0]))
            max_end = int(np.max(cluster_intervals[:, 1]))
            merged_str = f"{min_start}-{max_end}"
            for i in idx:
                mapping[original_strings[i]] = merged_str
    return mapping


def estimate_chunks(csv_path: str, chunk_size: int) -> int:
    total_lines = 0
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            total_lines += 1
    total_data_rows = max(total_lines - 1, 0)
    return max((total_data_rows + chunk_size - 1) // chunk_size, 1)


def clean_final_graph(G: nx.Graph, args) -> Dict[str, int]:
    removed_self_loops = 0
    removed_weak_edges = 0
    removed_isolates = 0
    if args.remove_self_loops:
        loops = list(nx.selfloop_edges(G))
        removed_self_loops = len(loops)
        if loops:
            G.remove_edges_from(loops)
    if args.min_edge_weight_to_keep is not None:
        weak_edges = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if safe_float(d.get("weight", 0.0), 0.0) < args.min_edge_weight_to_keep
        ]
        removed_weak_edges = len(weak_edges)
        if weak_edges:
            G.remove_edges_from(weak_edges)
    if args.remove_isolates_before_cluster:
        isolates = list(nx.isolates(G))
        removed_isolates = len(isolates)
        if isolates:
            G.remove_nodes_from(isolates)
    return {
        "removed_self_loops": removed_self_loops,
        "removed_weak_edges": removed_weak_edges,
        "removed_isolates": removed_isolates,
    }


def split_communities_by_connectivity(G: nx.Graph, communities: Iterable[Iterable[Node]]) -> List[List[Node]]:
    out: List[List[Node]] = []
    for comm in communities:
        nodes = list(comm)
        if not nodes:
            continue
        subg = G.subgraph(nodes)
        for cc in nx.connected_components(subg):
            out.append(list(cc))
    out.sort(key=len, reverse=True)
    return out


def summarize_family(G: nx.Graph, family_nodes: Iterable[Node]) -> Dict[str, float]:
    subg = G.subgraph(family_nodes)
    node_count = subg.number_of_nodes()
    edge_count = subg.number_of_edges()
    density = nx.density(subg) if node_count > 1 else 0.0
    avg_degree = (2.0 * edge_count) / node_count if node_count > 0 else 0.0
    total_weight = 0.0
    max_edge_weight = 0.0
    for _, _, data in subg.edges(data=True):
        w = safe_float(data.get("weight", 0.0), 0.0)
        total_weight += w
        max_edge_weight = max(max_edge_weight, w)
    avg_edge_weight = total_weight / edge_count if edge_count > 0 else 0.0
    hmm_ids = [str(n[0]) for n in subg.nodes()]
    unique_hmm_count = len(set(hmm_ids))
    repeated_hmm_nodes = node_count - unique_hmm_count
    repeated_hmm_fraction = repeated_hmm_nodes / node_count if node_count > 0 else 0.0
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": density,
        "avg_degree": avg_degree,
        "total_weight": total_weight,
        "avg_edge_weight": avg_edge_weight,
        "max_edge_weight": max_edge_weight,
        "unique_hmm_count": unique_hmm_count,
        "repeated_hmm_nodes": repeated_hmm_nodes,
        "repeated_hmm_fraction": repeated_hmm_fraction,
    }


def node_to_str(node: Node, sep: str) -> str:
    return f"{node[0]}{sep}{node[1]}"


def counter_mode(counter: Counter, default="NA"):
    if not counter:
        return default
    return counter.most_common(1)[0][0]


def counter_mode_and_fraction(counter: Counter, default="NA"):
    if not counter:
        return default, 0.0
    k, v = counter.most_common(1)[0]
    tot = sum(counter.values())
    return k, float(v) / tot if tot > 0 else 0.0


def representative_status_from_counter(counter: Counter, policy: str) -> str:
    if not counter:
        return "NA"
    if policy == "ok_if_any_else_majority":
        if counter.get("OK", 0) > 0:
            return "OK"
        return counter_mode(counter, default="NA")
    if policy == "majority":
        return counter_mode(counter, default="NA")
    raise ValueError(f"Unsupported representative status policy: {policy}")


def representative_label_from_counter(counter: Counter, policy: str):
    if not counter:
        return "NA", 0.0
    if policy == "majority_among_OK":
        return counter_mode_and_fraction(counter, default="NA")
    raise ValueError(f"Unsupported representative label policy: {policy}")


def build_kept_size_map(node_to_kept_id: Dict[Node, int], node_to_kept_size: Dict[Node, int]) -> Dict[int, int]:
    kept_size_map: Dict[int, int] = {}
    for node, kid in node_to_kept_id.items():
        kid = int(kid)
        if kid == -1:
            continue
        kept_size_map[kid] = int(node_to_kept_size.get(node, -1))
    return kept_size_map


def build_common_graph(csv_input_file: str, args):
    if not os.path.exists(csv_input_file):
        raise FileNotFoundError(f"Input file does not exist: {csv_input_file}")
    head = pd.read_csv(csv_input_file, nrows=1)
    required_cols = ["Main_HMM", "Sub_HMM", "Sub_Segments_Details", args.score_column]
    missing = [c for c in required_cols if c not in head.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}; current columns: {list(head.columns)}")
    pbar_total = estimate_chunks(csv_input_file, args.chunk_size)

    print("\n[COMMON PASS 1] Building coarse binned graph...")
    G_binned = nx.Graph()
    reader = pd.read_csv(csv_input_file, chunksize=args.chunk_size)
    for chunk in tqdm(reader, total=pbar_total, desc="[COMMON PASS 1] read CSV"):
        for row in chunk.itertuples(index=False):
            try:
                parsed_pairs = parse_details_string(getattr(row, "Sub_Segments_Details"))
                if not parsed_pairs:
                    continue
                main_hmm = str(getattr(row, "Main_HMM"))
                sub_hmm = str(getattr(row, "Sub_HMM"))
                for main_seg, sub_seg in parsed_pairs:
                    main_seg = norm_seg(main_seg)
                    sub_seg = norm_seg(sub_seg)
                    node_a = get_binned_node_name(main_hmm, main_seg, args.bin_size)
                    node_b = get_binned_node_name(sub_hmm, sub_seg, args.bin_size)
                    if node_a not in G_binned:
                        G_binned.add_node(node_a, raw_segs=set())
                    if node_b not in G_binned:
                        G_binned.add_node(node_b, raw_segs=set())
                    G_binned.nodes[node_a]["raw_segs"].add(main_seg)
                    G_binned.nodes[node_b]["raw_segs"].add(sub_seg)
                    if node_a != node_b:
                        G_binned.add_edge(node_a, node_b)
            except Exception:
                continue
    pass1_nodes = G_binned.number_of_nodes()
    pass1_edges = G_binned.number_of_edges()
    print(f"[COMMON PASS 1] Done: nodes={pass1_nodes}, edges={pass1_edges}")

    print(f"\n[COMMON PASS 1.5] Computing raw-to-merged segment map with DBSCAN metric={args.dbscan_metric}...")
    original_to_merged_map: Dict[Node, Node] = {}
    segments_buffer: Dict[str, Set[str]] = defaultdict(set)
    for nodes_in_component in tqdm(nx.connected_components(G_binned), desc="[COMMON PASS 1.5] components"):
        segments_buffer.clear()
        for node in nodes_in_component:
            hmm_id = str(node[0])
            segments_buffer[hmm_id].update(G_binned.nodes[node].get("raw_segs", set()))
        for hmm_id, seg_set in segments_buffer.items():
            mapping_dict = compute_smart_mappings(seg_set, args.dbscan_eps, args.dbscan_min_samples, args.dbscan_metric)
            for raw_seg, merged_seg in mapping_dict.items():
                raw_seg = norm_seg(raw_seg)
                merged_seg = norm_seg(merged_seg)
                original_to_merged_map[(hmm_id, raw_seg)] = (hmm_id, merged_seg)
    mapping_count = len(original_to_merged_map)
    print(f"[COMMON PASS 1.5] Mapping entries: {mapping_count}")
    del G_binned

    print("\n[COMMON PASS 2] Building final weighted graph...")
    G_final = nx.Graph()
    subset_pairs: Set[RawPair] = set()
    reader = pd.read_csv(csv_input_file, chunksize=args.chunk_size)
    for chunk in tqdm(reader, total=pbar_total, desc="[COMMON PASS 2] add edges"):
        for row in chunk.itertuples(index=False):
            try:
                pairs = parse_details_string(getattr(row, "Sub_Segments_Details"))
                if not pairs:
                    continue
                main_hmm = str(getattr(row, "Main_HMM"))
                sub_hmm = str(getattr(row, "Sub_HMM"))
                score_val = safe_float(getattr(row, args.score_column), default=0.0)
                for main_seg, sub_seg in pairs:
                    main_seg = norm_seg(main_seg)
                    sub_seg = norm_seg(sub_seg)
                    subset_pairs.add((main_hmm, main_seg, sub_hmm, sub_seg))
                    merged_node_a = original_to_merged_map.get((main_hmm, main_seg), (main_hmm, main_seg))
                    merged_node_b = original_to_merged_map.get((sub_hmm, sub_seg), (sub_hmm, sub_seg))
                    if merged_node_a == merged_node_b:
                        continue
                    update_edge_max_score(G_final, merged_node_a, merged_node_b, score_val)
            except Exception:
                continue
    raw_final_nodes = G_final.number_of_nodes()
    raw_final_edges = G_final.number_of_edges()
    print(f"[COMMON PASS 2] Before cleaning: nodes={raw_final_nodes}, edges={raw_final_edges}")
    clean_info = clean_final_graph(G_final, args)
    final_nodes = G_final.number_of_nodes()
    final_edges = G_final.number_of_edges()
    total_weight = sum(safe_float(d.get("weight", 0.0), 0.0) for _, _, d in G_final.edges(data=True))
    print(f"[COMMON PASS 2] After cleaning: nodes={final_nodes}, edges={final_edges}, total_weight={total_weight:.6f}")
    print(f"[COMMON PASS 2] Cleaning info: {clean_info}")
    if final_nodes == 0 or final_edges == 0:
        raise RuntimeError("Final graph is empty; Leiden cannot be run.")
    graph_info = {
        "score_column": args.score_column,
        "chunk_size": args.chunk_size,
        "bin_size": args.bin_size,
        "dbscan_eps": args.dbscan_eps,
        "dbscan_min_samples": args.dbscan_min_samples,
        "dbscan_metric": args.dbscan_metric,
        "remove_self_loops": args.remove_self_loops,
        "remove_isolates_before_cluster": args.remove_isolates_before_cluster,
        "min_edge_weight_to_keep": args.min_edge_weight_to_keep,
        "pass1_nodes": pass1_nodes,
        "pass1_edges": pass1_edges,
        "mapping_count": mapping_count,
        "raw_final_nodes": raw_final_nodes,
        "raw_final_edges": raw_final_edges,
        "cleaned_final_nodes": final_nodes,
        "cleaned_final_edges": final_edges,
        "cleaned_final_total_weight": total_weight,
        "removed_self_loops": clean_info["removed_self_loops"],
        "removed_weak_edges": clean_info["removed_weak_edges"],
        "removed_isolates": clean_info["removed_isolates"],
        "subset_pairs_count": len(subset_pairs),
    }
    return G_final, graph_info, original_to_merged_map, subset_pairs


def _call_igraph_leiden_compat(ig_graph, args):
    call_variants = [
        {"objective_function": args.leiden_objective, "weights": "weight", "resolution": args.community_resolution, "beta": args.leiden_beta, "n_iterations": args.leiden_n_iterations},
        {"objective_function": args.leiden_objective, "weights": "weight", "resolution_parameter": args.community_resolution, "beta": args.leiden_beta, "n_iterations": args.leiden_n_iterations},
        {"edge_weights": "weight", "resolution": args.community_resolution, "beta": args.leiden_beta, "n_iterations": args.leiden_n_iterations},
        {"edge_weights": "weight", "resolution_parameter": args.community_resolution, "beta": args.leiden_beta, "n_iterations": args.leiden_n_iterations},
    ]
    last_error = None
    for kwargs in call_variants:
        try:
            return ig_graph.community_leiden(**kwargs)
        except TypeError as e:
            last_error = e
    if last_error is not None:
        raise last_error
    raise RuntimeError("igraph community_leiden call failed")


def detect_communities_leiden(G: nx.Graph, args):
    try:
        import igraph as ig
    except ImportError as exc:
        raise ImportError("python-igraph is required: python -m pip install python-igraph") from exc
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edge_list = list(G.edges())
    ig_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in edge_list]
    ig_weights = [safe_float(G[u][v].get("weight", 1.0), 1.0) for u, v in edge_list]
    ig_graph = ig.Graph(n=len(nodes), edges=ig_edges, directed=False)
    if ig_edges:
        ig_graph.es["weight"] = ig_weights
    best_partition = None
    best_modularity = -np.inf
    for run_idx in range(args.community_runs):
        trial_seed = args.community_seed + run_idx
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        clustering = _call_igraph_leiden_compat(ig_graph, args)
        grouped: Dict[int, Set[Node]] = defaultdict(set)
        for idx, cid in enumerate(clustering.membership):
            grouped[int(cid)].add(nodes[idx])
        communities = list(grouped.values())
        q = nx_modularity(G, communities, weight="weight", resolution=args.community_resolution)
        if q > best_modularity:
            best_modularity = q
            best_partition = [set(c) for c in communities]
    return best_partition, best_modularity


def ensure_case_dirs(case_output_dir: str) -> None:
    os.makedirs(case_output_dir, exist_ok=True)
    os.makedirs(os.path.join(case_output_dir, "graphml_LEIDEN"), exist_ok=True)


def export_common_graph_summary(case_output_dir: str, label: str, input_csv: str, graph_info: dict) -> None:
    out_csv = os.path.join(case_output_dir, f"{label}_common_graph_summary.csv")
    row = {"Label": label, "input_csv": input_csv}
    row.update(graph_info)
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"[export] Common graph summary: {out_csv}")


def build_node_membership_maps(final_components: List[List[Node]], min_community_size: int):
    node_to_comm_id: Dict[Node, int] = {}
    node_to_comm_size: Dict[Node, int] = {}
    node_to_kept_id: Dict[Node, int] = {}
    node_to_kept_size: Dict[Node, int] = {}
    kept_id_to_nodes: Dict[int, List[Node]] = defaultdict(list)
    kept_counter = 0
    for comm_idx, family_nodes in enumerate(final_components, start=1):
        comm_size = len(family_nodes)
        kept_id = None
        if comm_size >= min_community_size:
            kept_counter += 1
            kept_id = kept_counter
        for n in family_nodes:
            node_to_comm_id[n] = comm_idx
            node_to_comm_size[n] = comm_size
            if kept_id is None:
                node_to_kept_id[n] = -1
                node_to_kept_size[n] = -1
            else:
                node_to_kept_id[n] = kept_id
                node_to_kept_size[n] = comm_size
                kept_id_to_nodes[kept_id].append(n)
    return node_to_comm_id, node_to_comm_size, node_to_kept_id, node_to_kept_size, kept_id_to_nodes


def export_raw_to_merged_map(case_output_dir: str, label: str, original_to_merged_map: Dict[Node, Node]) -> str:
    out_csv = os.path.join(case_output_dir, f"{label}_raw_to_merged_map.csv")
    rows = [{"HMM_ID": hmm, "Raw_Segment": raw_seg, "Merged_Segment": merged_seg}
            for (hmm, raw_seg), (_, merged_seg) in original_to_merged_map.items()]
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[export] Raw-to-merged map: {out_csv}")
    return out_csv


def export_global_node_edge_tables(case_output_dir: str, label: str, G: nx.Graph, final_components: List[List[Node]], same_graph_modularity: float,
                                  node_to_comm_id: Dict[Node, int], node_to_comm_size: Dict[Node, int],
                                  node_to_kept_id: Dict[Node, int], node_to_kept_size: Dict[Node, int], args):
    nodes_csv = os.path.join(case_output_dir, f"{label}_nodes_LEIDEN.csv")
    edges_csv = os.path.join(case_output_dir, f"{label}_edges_LEIDEN.csv")
    node_rows = []
    for n in G.nodes():
        hmm_id, seg = n
        node_id = node_to_str(n, args.node_id_sep)
        deg = int(G.degree(n))
        wdeg = float(G.degree(n, weight="weight"))
        comm_id = int(node_to_comm_id.get(n, -1))
        comm_size = int(node_to_comm_size.get(n, 1))
        kept_id = int(node_to_kept_id.get(n, -1))
        kept_size = int(node_to_kept_size.get(n, -1))
        node_rows.append({
            "Id": node_id,
            "Label": node_id,
            "HMM_ID": str(hmm_id),
            "Merged_Segment_Range": str(seg),
            "Community_ID": comm_id,
            "Community_Size": comm_size,
            "Kept_Family_ID": kept_id if kept_id != -1 else "",
            "Kept_Family_Size": kept_size if kept_id != -1 else "",
            "Is_Kept_Family": 1 if kept_id != -1 else 0,
            "Degree": deg,
            "Weighted_Degree": wdeg,
            "Label_Set": label,
            "Method": "LEIDEN",
            "Modularity": float(same_graph_modularity),
        })
    pd.DataFrame(node_rows).to_csv(nodes_csv, index=False)
    edge_rows = []
    for u, v, d in G.edges(data=True):
        cu = int(node_to_comm_id.get(u, -1))
        cv = int(node_to_comm_id.get(v, -1))
        edge_rows.append({
            "Source": node_to_str(u, args.node_id_sep),
            "Target": node_to_str(v, args.node_id_sep),
            "Type": "Undirected",
            "Weight": safe_float(d.get("weight", 0.0), 0.0),
            "Source_HMM_ID": str(u[0]),
            "Source_Segment": str(u[1]),
            "Target_HMM_ID": str(v[0]),
            "Target_Segment": str(v[1]),
            "Source_Community_ID": cu,
            "Target_Community_ID": cv,
            "Intra_Community": 1 if cu == cv else 0,
            "Label_Set": label,
            "Method": "LEIDEN",
        })
    pd.DataFrame(edge_rows).to_csv(edges_csv, index=False)
    print(f"[export] Global node table: {nodes_csv}")
    print(f"[export] Global edge table: {edges_csv}")
    return nodes_csv, edges_csv, len(node_rows), len(edge_rows)


def export_global_graphml(case_output_dir: str, label: str, G: nx.Graph, node_to_comm_id: Dict[Node, int], same_graph_modularity: float, args) -> str:
    out_graphml = os.path.join(case_output_dir, f"{label}_global_graph_LEIDEN.graphml")
    H = G.copy()
    for n in list(H.nodes()):
        H.nodes[n]["HMM_ID"] = str(n[0])
        H.nodes[n]["Merged_Segment_Range"] = str(n[1])
        H.nodes[n]["Community_ID"] = int(node_to_comm_id.get(n, -1))
        H.nodes[n]["Degree"] = int(H.degree(n))
        H.nodes[n]["Weighted_Degree"] = float(H.degree(n, weight="weight"))
        H.nodes[n]["Label_Set"] = label
        H.nodes[n]["Method"] = "LEIDEN"
        H.nodes[n]["Modularity"] = float(same_graph_modularity)
    relabel = {n: node_to_str(n, args.node_id_sep) for n in H.nodes()}
    nx.write_graphml(nx.relabel_nodes(H, relabel), out_graphml)
    print(f"[export] Global GraphML: {out_graphml}")
    return out_graphml


def to_long(counter_map: dict, colname: str, label: str) -> pd.DataFrame:
    rec = []
    for kid, cnt in counter_map.items():
        total = sum(cnt.values())
        for k, v in cnt.items():
            rec.append({
                "Label_Set": label,
                "Kept_Family_ID": kid,
                colname: k,
                "Count": int(v),
                "Fraction_within_family": float(v) / total if total > 0 else 0.0,
            })
    return pd.DataFrame(rec)


def top_item(counter: Counter):
    if not counter:
        return "", 0, 0.0
    k, v = counter.most_common(1)[0]
    tot = sum(counter.values())
    return k, int(v), float(v) / tot if tot > 0 else 0.0


def _array_or_empty(chunk: pd.DataFrame, column: str) -> np.ndarray:
    if column in chunk.columns:
        return chunk[column].to_numpy()
    return np.array([""] * len(chunk), dtype=object)


def compute_intra_struct_stats(case_output_dir: str, label: str, G_final: nx.Graph, subset_pairs: Set[RawPair],
                               original_to_merged_map: Dict[Node, Node], node_to_kept_id: Dict[Node, int],
                               node_to_kept_size: Dict[Node, int], args) -> str:
    if args.struct_dedup_unit != "merged_node":
        raise ValueError("Only struct_dedup_unit='merged_node' is supported")
    out_summary = os.path.join(case_output_dir, f"{label}_intra_struct_summary.csv")
    out_ss_long = os.path.join(case_output_dir, f"{label}_intra_struct_ss_counts_long.csv")
    out_loc_long = os.path.join(case_output_dir, f"{label}_intra_struct_location_counts_long.csv")
    out_status_long = os.path.join(case_output_dir, f"{label}_intra_struct_status_counts_long.unique_nodes.csv")
    out_unique_nodes = os.path.join(case_output_dir, f"{label}_intra_struct_unique_nodes_dedup.csv")
    out_report = os.path.join(case_output_dir, f"{label}_intra_struct_report.txt")
    if not os.path.exists(args.struct_csv):
        pd.DataFrame().to_csv(out_summary, index=False)
        pd.DataFrame().to_csv(out_ss_long, index=False)
        pd.DataFrame().to_csv(out_loc_long, index=False)
        pd.DataFrame().to_csv(out_status_long, index=False)
        pd.DataFrame().to_csv(out_unique_nodes, index=False)
        with open(out_report, "w", encoding="utf-8") as f:
            f.write(f"STRUCT_CSV: {args.struct_csv}\n")
            f.write("Status: skipped because STRUCT_CSV was not found.\n")
        print(f"[export] Struct stats skipped; STRUCT_CSV not found: {args.struct_csv}")
        return out_summary

    intra_rows = Counter()
    intra_rows_any_ok = Counter()
    intra_ok_sides_row_level = Counter()
    intra_unique_nodes: Dict[int, Set[Node]] = defaultdict(set)
    intra_unique_ok_nodes: Dict[int, Set[Node]] = defaultdict(set)
    node_status_counter: Dict[Node, Counter] = defaultdict(Counter)
    node_ss_counter: Dict[Node, Counter] = defaultdict(Counter)
    node_loc_counter: Dict[Node, Counter] = defaultdict(Counter)
    node_rsa_sum: Dict[Node, float] = defaultdict(float)
    node_rsa_cnt = Counter()
    node_obs_in_intra = Counter()
    node_ok_obs_in_intra = Counter()
    node_uid_counter: Dict[Node, Counter] = defaultdict(Counter)
    node_pdb_type_counter: Dict[Node, Counter] = defaultdict(Counter)
    node_pdb_id_counter: Dict[Node, Counter] = defaultdict(Counter)
    node_chain_counter: Dict[Node, Counter] = defaultdict(Counter)
    node_msa_start_counter: Dict[Node, Counter] = defaultdict(Counter)
    node_msa_end_counter: Dict[Node, Counter] = defaultdict(Counter)
    node_seq_start_counter: Dict[Node, Counter] = defaultdict(Counter)
    node_seq_end_counter: Dict[Node, Counter] = defaultdict(Counter)
    node_seq_len_counter: Dict[Node, Counter] = defaultdict(Counter)
    ss_count: Dict[int, Counter] = defaultdict(Counter)
    loc_count: Dict[int, Counter] = defaultdict(Counter)
    status_count: Dict[int, Counter] = defaultdict(Counter)
    rsa_sum: Dict[int, float] = defaultdict(float)
    rsa_cnt = Counter()
    total_struct_rows = 0
    parsed_pairs_rows = 0
    skipped_multi_mapping = 0
    skipped_not_in_subset = 0
    mapped_both_nodes = 0
    mapped_both_kept = 0
    intra_total = 0
    kept_size_map = build_kept_size_map(node_to_kept_id, node_to_kept_size)
    hmm_in_graph = {str(n[0]) for n in G_final.nodes()}

    reader = pd.read_csv(args.struct_csv, dtype=str, keep_default_na=False, low_memory=False, chunksize=args.chunk_size)
    for chunk in reader:
        total_struct_rows += len(chunk)
        for c in [COL_MAIN_HMM, COL_SUB_HMM, COL_DETAILS]:
            if c not in chunk.columns:
                raise RuntimeError(f"STRUCT_CSV is missing required column: {c}; current columns: {list(chunk.columns)}")
        main_hmms = chunk[COL_MAIN_HMM].astype(str).str.strip().to_numpy()
        sub_hmms = chunk[COL_SUB_HMM].astype(str).str.strip().to_numpy()
        details = chunk[COL_DETAILS].to_numpy()
        main_status = _array_or_empty(chunk, COL_MAIN_STATUS)
        sub_status = _array_or_empty(chunk, COL_SUB_STATUS)
        main_ss = _array_or_empty(chunk, COL_MAIN_SS)
        sub_ss = _array_or_empty(chunk, COL_SUB_SS)
        main_loc = _array_or_empty(chunk, COL_MAIN_LOC)
        sub_loc = _array_or_empty(chunk, COL_SUB_LOC)
        main_rsa = _array_or_empty(chunk, COL_MAIN_RSA)
        sub_rsa = _array_or_empty(chunk, COL_SUB_RSA)
        main_uid = _array_or_empty(chunk, COL_MAIN_UID)
        sub_uid = _array_or_empty(chunk, COL_SUB_UID)
        main_msa_start = _array_or_empty(chunk, COL_MAIN_MSA_START)
        main_msa_end = _array_or_empty(chunk, COL_MAIN_MSA_END)
        sub_msa_start = _array_or_empty(chunk, COL_SUB_MSA_START)
        sub_msa_end = _array_or_empty(chunk, COL_SUB_MSA_END)
        main_seq_start = _array_or_empty(chunk, COL_MAIN_SEQ_START)
        main_seq_end = _array_or_empty(chunk, COL_MAIN_SEQ_END)
        main_seq_len = _array_or_empty(chunk, COL_MAIN_SEQ_LEN)
        sub_seq_start = _array_or_empty(chunk, COL_SUB_SEQ_START)
        sub_seq_end = _array_or_empty(chunk, COL_SUB_SEQ_END)
        sub_seq_len = _array_or_empty(chunk, COL_SUB_SEQ_LEN)
        main_pdb_type = _array_or_empty(chunk, COL_MAIN_PDB_TYPE)
        main_pdb_id = _array_or_empty(chunk, COL_MAIN_PDB_ID)
        main_chain = _array_or_empty(chunk, COL_MAIN_CHAIN)
        sub_pdb_type = _array_or_empty(chunk, COL_SUB_PDB_TYPE)
        sub_pdb_id = _array_or_empty(chunk, COL_SUB_PDB_ID)
        sub_chain = _array_or_empty(chunk, COL_SUB_CHAIN)
        for i in range(len(chunk)):
            mh = main_hmms[i]
            sh = sub_hmms[i]
            if mh not in hmm_in_graph and sh not in hmm_in_graph:
                continue
            pairs = parse_details_string(details[i])
            if not pairs:
                continue
            if args.strict_one_to_one and len(pairs) != 1:
                skipped_multi_mapping += 1
                continue
            parsed_pairs_rows += 1
            main_seg_raw, sub_seg_raw = pairs[0]
            if (mh, main_seg_raw, sh, sub_seg_raw) not in subset_pairs:
                skipped_not_in_subset += 1
                continue
            merged_a = original_to_merged_map.get((mh, main_seg_raw), (mh, main_seg_raw))
            merged_b = original_to_merged_map.get((sh, sub_seg_raw), (sh, sub_seg_raw))
            if merged_a not in node_to_kept_id or merged_b not in node_to_kept_id:
                continue
            mapped_both_nodes += 1
            kid_a = int(node_to_kept_id.get(merged_a, -1))
            kid_b = int(node_to_kept_id.get(merged_b, -1))
            if kid_a == -1 or kid_b == -1:
                continue
            mapped_both_kept += 1
            if kid_a != kid_b:
                continue
            kid = kid_a
            intra_total += 1
            intra_rows[kid] += 1
            any_ok_this_row = False
            side_payloads = [
                (merged_a, normalize_label(main_status[i]), normalize_label(main_ss[i]), normalize_label(main_loc[i]), main_rsa[i],
                 normalize_optional_str(main_uid[i]), normalize_optional_str(main_pdb_type[i]), normalize_optional_str(main_pdb_id[i]), normalize_optional_str(main_chain[i]),
                 normalize_optional_str(main_msa_start[i]), normalize_optional_str(main_msa_end[i]), normalize_optional_str(main_seq_start[i]), normalize_optional_str(main_seq_end[i]), normalize_optional_str(main_seq_len[i])),
                (merged_b, normalize_label(sub_status[i]), normalize_label(sub_ss[i]), normalize_label(sub_loc[i]), sub_rsa[i],
                 normalize_optional_str(sub_uid[i]), normalize_optional_str(sub_pdb_type[i]), normalize_optional_str(sub_pdb_id[i]), normalize_optional_str(sub_chain[i]),
                 normalize_optional_str(sub_msa_start[i]), normalize_optional_str(sub_msa_end[i]), normalize_optional_str(sub_seq_start[i]), normalize_optional_str(sub_seq_end[i]), normalize_optional_str(sub_seq_len[i])),
            ]
            for merged_node, st, ss, loc, rsa_val, uid_val, pdb_type_val, pdb_id_val, chain_val, msa_start_val, msa_end_val, seq_start_val, seq_end_val, seq_len_val in side_payloads:
                intra_unique_nodes[kid].add(merged_node)
                node_obs_in_intra[merged_node] += 1
                node_status_counter[merged_node][st] += 1
                if st == "OK":
                    any_ok_this_row = True
                    intra_ok_sides_row_level[kid] += 1
                    intra_unique_ok_nodes[kid].add(merged_node)
                    node_ok_obs_in_intra[merged_node] += 1
                    node_ss_counter[merged_node][ss] += 1
                    node_loc_counter[merged_node][loc] += 1
                    r = safe_float_nan(rsa_val)
                    if np.isfinite(r):
                        node_rsa_sum[merged_node] += float(r)
                        node_rsa_cnt[merged_node] += 1
                    for counter_map, val in [
                        (node_uid_counter, uid_val), (node_pdb_type_counter, pdb_type_val), (node_pdb_id_counter, pdb_id_val), (node_chain_counter, chain_val),
                        (node_msa_start_counter, msa_start_val), (node_msa_end_counter, msa_end_val), (node_seq_start_counter, seq_start_val),
                        (node_seq_end_counter, seq_end_val), (node_seq_len_counter, seq_len_val),
                    ]:
                        if val:
                            counter_map[merged_node][val] += 1
            if any_ok_this_row:
                intra_rows_any_ok[kid] += 1

    unique_node_rows = []
    for kid in sorted(intra_unique_nodes.keys(), key=lambda k: (-len(intra_unique_nodes[k]), k)):
        for node in sorted(list(intra_unique_nodes[kid]), key=segment_sort_key):
            rep_status = representative_status_from_counter(node_status_counter.get(node, Counter()), args.struct_representative_status_policy)
            status_count[kid][rep_status] += 1
            rep_ss = "NA"
            rep_loc = "NA"
            rep_ss_frac = 0.0
            rep_loc_frac = 0.0
            mean_rsa_node = np.nan
            rep_uid = rep_pdb_type = rep_pdb_id = rep_chain = ""
            rep_msa_start = rep_msa_end = rep_seq_start = rep_seq_end = rep_seq_len = np.nan
            if node_ok_obs_in_intra.get(node, 0) > 0:
                rep_ss, rep_ss_frac = representative_label_from_counter(node_ss_counter.get(node, Counter()), args.struct_representative_label_policy)
                rep_loc, rep_loc_frac = representative_label_from_counter(node_loc_counter.get(node, Counter()), args.struct_representative_label_policy)
                ss_count[kid][rep_ss] += 1
                loc_count[kid][rep_loc] += 1
                if node_rsa_cnt.get(node, 0) > 0:
                    mean_rsa_node = node_rsa_sum[node] / node_rsa_cnt[node]
                    rsa_sum[kid] += float(mean_rsa_node)
                    rsa_cnt[kid] += 1
                rep_uid = counter_mode(node_uid_counter.get(node, Counter()), default="")
                rep_pdb_type = counter_mode(node_pdb_type_counter.get(node, Counter()), default="")
                rep_pdb_id = counter_mode(node_pdb_id_counter.get(node, Counter()), default="")
                rep_chain = counter_mode(node_chain_counter.get(node, Counter()), default="")
                rep_msa_start = safe_int_nan(counter_mode(node_msa_start_counter.get(node, Counter()), default=""))
                rep_msa_end = safe_int_nan(counter_mode(node_msa_end_counter.get(node, Counter()), default=""))
                rep_seq_start = safe_int_nan(counter_mode(node_seq_start_counter.get(node, Counter()), default=""))
                rep_seq_end = safe_int_nan(counter_mode(node_seq_end_counter.get(node, Counter()), default=""))
                rep_seq_len = safe_int_nan(counter_mode(node_seq_len_counter.get(node, Counter()), default=""))
            unique_node_rows.append({
                "Label_Set": label,
                "Kept_Family_ID": kid,
                "Kept_Family_Size": int(kept_size_map.get(kid, -1)),
                "Node_ID": node_to_str(node, args.node_id_sep),
                "HMM_ID": str(node[0]),
                "Merged_Segment_Range": str(node[1]),
                "Representative_Status": rep_status,
                "Has_any_OK_obs": 1 if node_ok_obs_in_intra.get(node, 0) > 0 else 0,
                "Representative_SS_if_OK": rep_ss,
                "Representative_SS_Fraction_if_OK": rep_ss_frac,
                "Representative_Location_if_OK": rep_loc,
                "Representative_Location_Fraction_if_OK": rep_loc_frac,
                "Mean_RSA_if_OK": mean_rsa_node,
                "Representative_UID_if_OK": rep_uid,
                "Representative_PDB_Type_if_OK": rep_pdb_type,
                "Representative_PDB_ID_if_OK": rep_pdb_id,
                "Representative_Chain_if_OK": rep_chain,
                "Representative_MSA_Start_if_OK": rep_msa_start,
                "Representative_MSA_End_if_OK": rep_msa_end,
                "Representative_Seq_Start_if_OK": rep_seq_start,
                "Representative_Seq_End_if_OK": rep_seq_end,
                "Representative_Seq_Len_if_OK": rep_seq_len,
                "Struct_Obs_in_Intra": int(node_obs_in_intra.get(node, 0)),
                "Struct_OK_Obs_in_Intra": int(node_ok_obs_in_intra.get(node, 0)),
            })
    pd.DataFrame(unique_node_rows).to_csv(out_unique_nodes, index=False)

    summary_rows = []
    for kid in sorted(intra_rows.keys(), key=lambda k: (-intra_rows[k], k)):
        n_intra = int(intra_rows[kid])
        n_any_ok = int(intra_rows_any_ok.get(kid, 0))
        n_ok_sides_row = int(intra_ok_sides_row_level.get(kid, 0))
        n_unique_nodes = int(len(intra_unique_nodes.get(kid, set())))
        n_unique_ok_nodes = int(len(intra_unique_ok_nodes.get(kid, set())))
        mean_rsa = rsa_sum[kid] / rsa_cnt[kid] if rsa_cnt.get(kid, 0) > 0 else np.nan
        ss_top, ss_top_n, ss_top_p = top_item(ss_count.get(kid, Counter()))
        loc_top, loc_top_n, loc_top_p = top_item(loc_count.get(kid, Counter()))
        status_top, status_top_n, status_top_p = top_item(status_count.get(kid, Counter()))
        summary_rows.append({
            "Label_Set": label,
            "Kept_Family_ID": kid,
            "Kept_Family_Size": int(kept_size_map.get(kid, -1)),
            "Struct_Dedup_Unit": args.struct_dedup_unit,
            "Intra_Rows": n_intra,
            "Intra_Rows_with_any_OK_side": n_any_ok,
            "OK_Sides_in_Intra_Rows": n_ok_sides_row,
            "AnyOK_Row_Fraction_in_Intra": n_any_ok / n_intra if n_intra > 0 else 0.0,
            "Unique_Merged_Nodes_in_Intra": n_unique_nodes,
            "Unique_Merged_Nodes_with_any_OK": n_unique_ok_nodes,
            "Unique_OK_Node_Fraction": n_unique_ok_nodes / n_unique_nodes if n_unique_nodes > 0 else 0.0,
            "Row_to_UniqueNode_Ratio": n_intra / n_unique_nodes if n_unique_nodes > 0 else np.nan,
            "OKSideRow_to_UniqueOKNode_Ratio": n_ok_sides_row / n_unique_ok_nodes if n_unique_ok_nodes > 0 else np.nan,
            "Mean_RSA_OK_unique_nodes": mean_rsa,
            "RSA_OK_Unique_Node_Count": int(rsa_cnt.get(kid, 0)),
            "Top_Status_unique_nodes": status_top,
            "Top_Status_Count": status_top_n,
            "Top_Status_Fraction_within_unique_nodes": status_top_p,
            "Top_SS_unique_nodes": ss_top,
            "Top_SS_Count": ss_top_n,
            "Top_SS_Fraction_within_unique_OK_nodes": ss_top_p,
            "Top_Location_unique_nodes": loc_top,
            "Top_Location_Count": loc_top_n,
            "Top_Location_Fraction_within_unique_OK_nodes": loc_top_p,
            "Total_unique_OK_nodes_used_for_SS": int(sum(ss_count.get(kid, Counter()).values())),
            "Total_unique_OK_nodes_used_for_Location": int(sum(loc_count.get(kid, Counter()).values())),
        })
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    to_long(ss_count, "SS_Class", label).to_csv(out_ss_long, index=False)
    to_long(loc_count, "Location", label).to_csv(out_loc_long, index=False)
    to_long(status_count, "Status", label).to_csv(out_status_long, index=False)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(f"STRUCT_CSV: {args.struct_csv}\n")
        f.write(f"Label: {label}\n")
        f.write(f"subset_pairs_count (from input): {len(subset_pairs):,}\n")
        f.write(f"Total STRUCT rows scanned: {total_struct_rows:,}\n")
        f.write(f"Rows with parsable mapping pairs: {parsed_pairs_rows:,}\n")
        f.write(f"Skipped multi-mapping rows (STRICT_ONE_TO_ONE): {skipped_multi_mapping:,}\n")
        f.write(f"Skipped rows not in this Top subset (pair not in subset_pairs): {skipped_not_in_subset:,}\n")
        f.write(f"Rows where both endpoints mapped to nodes in G_final: {mapped_both_nodes:,}\n")
        f.write(f"Rows where both endpoints mapped to kept families: {mapped_both_kept:,}\n")
        f.write(f"Intra rows (both endpoints same kept family): {intra_total:,}\n")
        f.write(f"Struct dedup unit: {args.struct_dedup_unit}\n")
        f.write(f"Representative status policy: {args.struct_representative_status_policy}\n")
        f.write(f"Representative SS/Location policy: {args.struct_representative_label_policy}\n")
        f.write("\nNotes:\n")
        f.write("- The analysis uses the exact raw-to-merged mapping from PASS 1.5.\n")
        f.write("- Only intra-community observations are counted.\n")
        f.write("- Row-level fields are retained for auditability.\n")
        f.write("- SS, Location, and RSA are summarized after unique merged-node deduplication.\n")
        f.write("- UID, PDB_ID, Chain, MSA interval, and sequence interval representatives are derived from OK observations.\n")
    print(f"[export] Intra SS/RSA summary: {out_summary}")
    return out_summary


def export_leiden_outputs(case_output_dir: str, label: str, G: nx.Graph, communities, same_graph_modularity: float,
                         graph_info: dict, original_to_merged_map: Dict[Node, Node], args) -> dict:
    families_csv = os.path.join(case_output_dir, f"{label}_families_LEIDEN.csv")
    stats_csv = os.path.join(case_output_dir, f"{label}_families_LEIDEN.stats.csv")
    final_components = split_communities_by_connectivity(G, communities)
    node_to_comm_id, node_to_comm_size, node_to_kept_id, node_to_kept_size, _ = build_node_membership_maps(final_components, args.min_community_size)
    export_raw_to_merged_map(case_output_dir, label, original_to_merged_map)
    nodes_csv = edges_csv = ""
    node_table_rows = edge_table_rows = 0
    if args.export_node_edge_table:
        nodes_csv, edges_csv, node_table_rows, edge_table_rows = export_global_node_edge_tables(
            case_output_dir, label, G, final_components, same_graph_modularity,
            node_to_comm_id, node_to_comm_size, node_to_kept_id, node_to_kept_size, args
        )
    global_graphml = ""
    if args.export_global_graphml:
        try:
            global_graphml = export_global_graphml(case_output_dir, label, G, node_to_comm_id, same_graph_modularity, args)
        except Exception as e:
            print(f"[warning] Global GraphML export failed for {label}: {e}")
    kept_family_count = 0
    written_rows = 0
    family_stats_rows = []
    family_graphml_dir = os.path.join(case_output_dir, "graphml_LEIDEN")
    with open(families_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Family_ID", "HMM_ID", "Merged_Segment_Range"])
        for family_nodes in tqdm(final_components, desc=f"[export {label} LEIDEN]"):
            if len(family_nodes) < args.min_community_size:
                continue
            kept_family_count += 1
            family_id = kept_family_count
            stats = summarize_family(G, family_nodes)
            stats["Label"] = label
            stats["Family_ID"] = family_id
            stats["method"] = "LEIDEN"
            stats["same_graph_modularity"] = same_graph_modularity
            stats["community_resolution"] = args.community_resolution
            stats["community_runs"] = args.community_runs
            stats["graph_nodes_common"] = graph_info["cleaned_final_nodes"]
            stats["graph_edges_common"] = graph_info["cleaned_final_edges"]
            stats["graph_total_weight_common"] = graph_info["cleaned_final_total_weight"]
            family_stats_rows.append(stats)
            for hmm_id, segment in consolidate_family_nodes(family_nodes):
                writer.writerow([family_id, hmm_id, segment])
                written_rows += 1
            if args.export_family_graphml:
                try:
                    subgraph = G.subgraph(family_nodes).copy()
                    for n in list(subgraph.nodes()):
                        subgraph.nodes[n]["HMM_ID"] = str(n[0])
                        subgraph.nodes[n]["Merged_Segment_Range"] = str(n[1])
                        subgraph.nodes[n]["Family_ID"] = family_id
                        subgraph.nodes[n]["Degree"] = int(subgraph.degree(n))
                        subgraph.nodes[n]["Weighted_Degree"] = float(subgraph.degree(n, weight="weight"))
                    relabel = {n: node_to_str(n, args.node_id_sep) for n in subgraph.nodes()}
                    out_graphml = os.path.join(family_graphml_dir, f"{label}_family_{family_id}.graphml")
                    nx.write_graphml(nx.relabel_nodes(subgraph, relabel), out_graphml)
                except Exception as e:
                    print(f"[warning] Family GraphML export failed for {label} family {family_id}: {e}")
    if family_stats_rows:
        df_stats = pd.DataFrame(family_stats_rows)
        ordered_cols = [
            "Label", "Family_ID", "method", "same_graph_modularity", "community_resolution", "community_runs",
            "graph_nodes_common", "graph_edges_common", "graph_total_weight_common", "node_count", "edge_count", "density",
            "avg_degree", "total_weight", "avg_edge_weight", "max_edge_weight", "unique_hmm_count", "repeated_hmm_nodes", "repeated_hmm_fraction",
        ]
        df_stats = df_stats[ordered_cols].sort_values(by=["node_count", "total_weight"], ascending=[False, False])
        df_stats.to_csv(stats_csv, index=False)
    else:
        pd.DataFrame().to_csv(stats_csv, index=False)
    sizes = np.array([len(c) for c in final_components if len(c) >= args.min_community_size], dtype=int)
    summary = {
        "Label": label,
        "method": "LEIDEN",
        "community_count_after_connectivity_split": int(len(final_components)),
        "community_count_kept_min_size": int(len(sizes)),
        "output_rows_consolidated": int(written_rows),
        "largest_cluster_size": int(sizes.max()) if len(sizes) else 0,
        "median_cluster_size": float(np.median(sizes)) if len(sizes) else 0.0,
        "mean_cluster_size": float(np.mean(sizes)) if len(sizes) else 0.0,
        "same_graph_modularity": float(same_graph_modularity),
        "common_graph_nodes": int(graph_info["cleaned_final_nodes"]),
        "common_graph_edges": int(graph_info["cleaned_final_edges"]),
        "common_graph_total_weight": float(graph_info["cleaned_final_total_weight"]),
        "node_table_rows": int(node_table_rows),
        "edge_table_rows": int(edge_table_rows),
        "nodes_csv": nodes_csv,
        "edges_csv": edges_csv,
        "families_csv": families_csv,
        "stats_csv": stats_csv,
        "global_graphml": global_graphml,
    }
    print(f"[export] LEIDEN families: {families_csv}")
    print(f"[export] LEIDEN stats   : {stats_csv}")
    return summary


def run_one_case(label: str, csv_input_file: str, args) -> dict:
    print("\n" + "=" * 120)
    print(f"Processing {label}")
    print(f"Input file: {csv_input_file}")
    print("=" * 120)
    case_output_dir = os.path.join(args.output_root, label)
    ensure_case_dirs(case_output_dir)
    G_final, graph_info, original_to_merged_map, _subset_pairs = build_common_graph(csv_input_file, args)
    export_common_graph_summary(case_output_dir, label, csv_input_file, graph_info)
    print("\n" + "-" * 120)
    print(f"[{label}] Running Leiden on G_final...")
    leiden_partition, leiden_q = detect_communities_leiden(G_final, args)
    print(f"[{label}] Leiden same-graph modularity = {leiden_q:.10f}")
    leiden_summary = export_leiden_outputs(
        case_output_dir, label, G_final, leiden_partition, leiden_q, graph_info,
        original_to_merged_map, args
    )
    print("\n" + "=" * 120)
    print(f"{label} finished")
    print(f"Common graph nodes/edges : {graph_info['cleaned_final_nodes']} / {graph_info['cleaned_final_edges']}")
    print(f"Leiden modularity        : {leiden_q:.10f}")
    print(f"Largest Leiden cluster   : {leiden_summary['largest_cluster_size']}")
    print(f"Kept Leiden clusters     : {leiden_summary['community_count_kept_min_size']}")
    print(f"Output directory         : {case_output_dir}")
    print("=" * 120)
    return {"label": label, "input_csv": csv_input_file, "graph_info": graph_info, "leiden_summary": leiden_summary}


def export_batch_overall_summary(all_results: List[dict], args) -> None:
    summary_rows = []
    graph_rows = []
    for item in all_results:
        graph_row = {"Label": item["label"], "input_csv": item["input_csv"]}
        graph_row.update(item["graph_info"])
        graph_rows.append(graph_row)
        summary_rows.append(item["leiden_summary"])
    batch_summary_csv = os.path.join(args.output_root, "batch_leiden_summary.csv")
    batch_graph_csv = os.path.join(args.output_root, "batch_graph_summary.csv")
    pd.DataFrame(summary_rows).to_csv(batch_summary_csv, index=False)
    pd.DataFrame(graph_rows).to_csv(batch_graph_csv, index=False)
    print("\n" + "=" * 120)
    print("Batch summary exported")
    print(f"Leiden summary: {batch_summary_csv}")
    print(f"Graph summary : {batch_graph_csv}")
    print("=" * 120)


def parse_input_specs(input_specs: Optional[List[str]], input_file_table: Optional[str]) -> List[Tuple[str, str]]:
    if input_specs:
        out = []
        for spec in input_specs:
            if "=" not in spec:
                raise ValueError(f"Invalid --input value: {spec}. Expected LABEL=/path/to/file.csv")
            label, path = spec.split("=", 1)
            out.append((label.strip(), path.strip()))
        return out
    if input_file_table:
        df = pd.read_csv(input_file_table, sep=None, engine="python")
        if "Label" not in df.columns or "Path" not in df.columns:
            raise ValueError("Input table must contain columns: Label, Path")
        return [(str(r.Label), str(r.Path)) for r in df.itertuples(index=False)]
    raise ValueError("Provide --input at least once or use --input-file-table.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build ProDive fragment graphs and run CPM-Leiden clustering.")
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", action="append", help="Input subset in LABEL=/path/file.csv format. Can be repeated.")
    input_group.add_argument("--input-file-table", help="TSV/CSV with columns Label and Path.")
    p.add_argument("--output-root", required=True, help="Batch output root directory.")
    p.add_argument("--chunk-size", type=int, default=100000)
    p.add_argument("--bin-size", type=int, default=8)
    p.add_argument("--dbscan-eps", type=int, default=3)
    p.add_argument("--dbscan-min-samples", type=int, default=2)
    p.add_argument("--dbscan-metric", default="chebyshev")
    p.add_argument("--score-column", default="Score")
    p.add_argument("--min-edge-weight-to-keep", type=float, default=None)
    p.add_argument("--keep-self-loops", action="store_true", help="Do not remove self loops before clustering.")
    p.add_argument("--keep-isolates", action="store_true", help="Do not remove isolated nodes before clustering.")
    p.add_argument("--community-resolution", type=float, default=0.05)
    p.add_argument("--community-runs", type=int, default=3)
    p.add_argument("--community-seed", type=int, default=42)
    p.add_argument("--leiden-objective", default="CPM", choices=["CPM", "modularity"])
    p.add_argument("--leiden-beta", type=float, default=0.01)
    p.add_argument("--leiden-n-iterations", type=int, default=4)
    p.add_argument("--min-community-size", type=int, default=2)
    p.add_argument("--no-family-graphml", action="store_true")
    p.add_argument("--export-global-graphml", action="store_true")
    p.add_argument("--no-node-edge-table", action="store_true")
    p.add_argument("--node-id-sep", default="||")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.remove_self_loops = not args.keep_self_loops
    args.remove_isolates_before_cluster = not args.keep_isolates
    args.export_family_graphml = not args.no_family_graphml
    args.export_node_edge_table = not args.no_node_edge_table
    input_files = parse_input_specs(args.input, args.input_file_table)
    print("=" * 120)
    print("Running batch Leiden clustering")
    print(f"Output root        : {args.output_root}")
    print("=" * 120)
    os.makedirs(args.output_root, exist_ok=True)
    all_results = []
    for label, csv_input_file in input_files:
        all_results.append(run_one_case(label, csv_input_file, args))
    export_batch_overall_summary(all_results, args)
    print("\n" + "=" * 120)
    print("All percentile subsets finished")
    print(f"Batch output root: {args.output_root}")
    print("=" * 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
