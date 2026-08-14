#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gamma sensitivity analysis for a ProDive fragment graph.

The graph construction and community-detection settings match the standard
clustering workflow:

1. Build coarse segment nodes with BIN_SIZE = 8.
2. Merge nearby intervals within each HMM using DBSCAN.
3. Build an undirected weighted graph, retaining the maximum ProDive Score
   when the same merged edge occurs more than once.
4. Remove self-loops and isolated nodes.
5. For each gamma, run weighted CPM-Leiden three times.
6. Select the run with the highest weighted NetworkX modularity calculated
   with the same gamma.

Only sensitivity-analysis tables are written. No GraphML or graph-plot files
are generated.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community.quality import modularity as nx_modularity
from sklearn.cluster import DBSCAN
from tqdm import tqdm


Node = Tuple[str, str]


# -----------------------------------------------------------------------------
# Analysis settings
# -----------------------------------------------------------------------------

DEFAULT_GAMMAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20]

CHUNK_SIZE = 100_000
BIN_SIZE = 8
DBSCAN_EPS = 3
DBSCAN_MIN_SAMPLES = 2
DBSCAN_METRIC = "chebyshev"
SCORE_COLUMN = "Score"

COMMUNITY_RUNS = 3
COMMUNITY_SEED = 42
LEIDEN_OBJECTIVE = "CPM"
LEIDEN_BETA = 0.01
LEIDEN_N_ITERATIONS = 4
MIN_COMMUNITY_SIZE = 2

REQUIRED_COLUMNS = ["Main_HMM", "Sub_HMM", "Sub_Segments_Details", SCORE_COLUMN]


def safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except Exception:
        return default


def parse_interval(value) -> Optional[Tuple[int, int]]:
    try:
        text = str(value).strip().replace(" ", "")
        match = re.match(r"^(\d+)-(\d+)$", text)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return (start, end) if start <= end else (end, start)

        match = re.match(r"^(\d+)$", text)
        if match:
            position = int(match.group(1))
            return position, position
    except Exception:
        return None
    return None


def normalize_segment(segment) -> str:
    interval = parse_interval(segment)
    if interval is None:
        return str(segment).strip().replace(" ", "")
    return f"{interval[0]}-{interval[1]}"


def parse_details_string(details) -> List[Tuple[str, str]]:
    if pd.isna(details):
        return []

    text = str(details).strip()
    if not text:
        return []

    parsed: List[Tuple[str, str]] = []
    for piece in re.split(r"\s*,\s*", text):
        match = re.match(
            r"^(\d+(?:-\d+)?)\s*->\s*(\d+(?:-\d+)?)$",
            piece.strip(),
        )
        if match:
            parsed.append(
                (
                    normalize_segment(match.group(1)),
                    normalize_segment(match.group(2)),
                )
            )
    return parsed


def binned_node(hmm_id, segment, bin_size: int) -> Node:
    interval = parse_interval(segment)
    start = interval[0] if interval is not None else 0
    bin_id = (start // bin_size) * bin_size
    return str(hmm_id), f"bin_{bin_id}"


def compute_smart_mappings(
    segment_set: Iterable[str],
    eps: int,
    min_samples: int,
    metric: str,
) -> Dict[str, str]:
    coordinates: List[List[int]] = []
    original_segments: List[str] = []

    for segment in segment_set:
        interval = parse_interval(segment)
        if interval is not None:
            coordinates.append([interval[0], interval[1]])
            original_segments.append(str(segment))

    if not coordinates:
        return {}

    matrix = np.asarray(coordinates, dtype=float)
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    ).fit(matrix).labels_

    mapping: Dict[str, str] = {}
    for label in set(labels):
        indices = np.where(labels == label)[0]
        cluster_intervals = matrix[indices]

        if label == -1:
            for index in indices:
                start, end = map(int, matrix[index])
                mapping[original_segments[index]] = f"{start}-{end}"
        else:
            merged_start = int(np.min(cluster_intervals[:, 0]))
            merged_end = int(np.max(cluster_intervals[:, 1]))
            merged_segment = f"{merged_start}-{merged_end}"
            for index in indices:
                mapping[original_segments[index]] = merged_segment

    return mapping


def update_edge_max_score(
    graph: nx.Graph,
    node_a: Node,
    node_b: Node,
    score: float,
) -> None:
    if graph.has_edge(node_a, node_b):
        previous = safe_float(graph[node_a][node_b].get("weight", 0.0), 0.0)
        if score > previous:
            graph[node_a][node_b]["weight"] = score
    else:
        graph.add_edge(node_a, node_b, weight=score)


def read_csv_chunks(csv_path: Path, chunk_size: int):
    return pd.read_csv(
        csv_path,
        usecols=REQUIRED_COLUMNS,
        chunksize=chunk_size,
    )


def build_fragment_graph(
    csv_path: Path,
    chunk_size: int,
    bin_size: int,
    dbscan_eps: int,
    dbscan_min_samples: int,
    dbscan_metric: str,
) -> Tuple[nx.Graph, Dict[str, float]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Available columns: {columns}"
        )

    print("\n[PASS 1/3] Building coarse binned graph...")
    coarse_graph = nx.Graph()
    valid_mapping_pairs_pass1 = 0

    for chunk in tqdm(read_csv_chunks(csv_path, chunk_size), desc="PASS 1 CSV chunks"):
        for row in chunk.itertuples(index=False):
            pairs = parse_details_string(row.Sub_Segments_Details)
            if not pairs:
                continue

            main_hmm = str(row.Main_HMM)
            sub_hmm = str(row.Sub_HMM)

            for main_segment, sub_segment in pairs:
                main_segment = normalize_segment(main_segment)
                sub_segment = normalize_segment(sub_segment)
                node_a = binned_node(main_hmm, main_segment, bin_size)
                node_b = binned_node(sub_hmm, sub_segment, bin_size)

                if node_a not in coarse_graph:
                    coarse_graph.add_node(node_a, raw_segs=set())
                if node_b not in coarse_graph:
                    coarse_graph.add_node(node_b, raw_segs=set())

                coarse_graph.nodes[node_a]["raw_segs"].add(main_segment)
                coarse_graph.nodes[node_b]["raw_segs"].add(sub_segment)
                valid_mapping_pairs_pass1 += 1

                if node_a != node_b:
                    coarse_graph.add_edge(node_a, node_b)

    coarse_nodes = coarse_graph.number_of_nodes()
    coarse_edges = coarse_graph.number_of_edges()
    print(
        f"[PASS 1/3] Coarse graph: nodes={coarse_nodes:,}, "
        f"edges={coarse_edges:,}"
    )

    print("\n[PASS 2/3] Computing DBSCAN raw-to-merged segment mapping...")
    raw_to_merged: Dict[Node, Node] = {}
    segments_by_hmm: Dict[str, Set[str]] = defaultdict(set)

    for component in tqdm(
        nx.connected_components(coarse_graph),
        desc="PASS 2 components",
    ):
        segments_by_hmm.clear()
        for node in component:
            hmm_id = str(node[0])
            segments_by_hmm[hmm_id].update(
                coarse_graph.nodes[node].get("raw_segs", set())
            )

        for hmm_id, segment_set in segments_by_hmm.items():
            mapping = compute_smart_mappings(
                segment_set,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                metric=dbscan_metric,
            )
            for raw_segment, merged_segment in mapping.items():
                raw_node = (hmm_id, normalize_segment(raw_segment))
                merged_node = (hmm_id, normalize_segment(merged_segment))
                raw_to_merged[raw_node] = merged_node

    mapping_count = len(raw_to_merged)
    print(f"[PASS 2/3] Mapping entries: {mapping_count:,}")
    del coarse_graph

    print("\n[PASS 3/3] Building final weighted graph...")
    final_graph = nx.Graph()
    valid_mapping_pairs_pass3 = 0

    for chunk in tqdm(read_csv_chunks(csv_path, chunk_size), desc="PASS 3 CSV chunks"):
        for row in chunk.itertuples(index=False):
            pairs = parse_details_string(row.Sub_Segments_Details)
            if not pairs:
                continue

            main_hmm = str(row.Main_HMM)
            sub_hmm = str(row.Sub_HMM)
            score = safe_float(getattr(row, SCORE_COLUMN), default=0.0)

            for main_segment, sub_segment in pairs:
                main_segment = normalize_segment(main_segment)
                sub_segment = normalize_segment(sub_segment)
                merged_a = raw_to_merged.get(
                    (main_hmm, main_segment),
                    (main_hmm, main_segment),
                )
                merged_b = raw_to_merged.get(
                    (sub_hmm, sub_segment),
                    (sub_hmm, sub_segment),
                )
                valid_mapping_pairs_pass3 += 1

                if merged_a != merged_b:
                    update_edge_max_score(final_graph, merged_a, merged_b, score)

    self_loops = list(nx.selfloop_edges(final_graph))
    final_graph.remove_edges_from(self_loops)
    isolates = list(nx.isolates(final_graph))
    final_graph.remove_nodes_from(isolates)

    final_nodes = final_graph.number_of_nodes()
    final_edges = final_graph.number_of_edges()
    total_weight = sum(
        safe_float(data.get("weight", 0.0), 0.0)
        for _, _, data in final_graph.edges(data=True)
    )

    if final_nodes == 0 or final_edges == 0:
        raise RuntimeError("The final graph is empty; Leiden cannot be run.")

    print(
        f"[PASS 3/3] Final graph: nodes={final_nodes:,}, "
        f"edges={final_edges:,}, total_weight={total_weight:.6f}"
    )

    graph_info = {
        "input_csv": str(csv_path.resolve()),
        "valid_mapping_pairs_pass1": valid_mapping_pairs_pass1,
        "valid_mapping_pairs_pass3": valid_mapping_pairs_pass3,
        "coarse_nodes": coarse_nodes,
        "coarse_edges": coarse_edges,
        "raw_to_merged_mapping_count": mapping_count,
        "removed_self_loops": len(self_loops),
        "removed_isolates": len(isolates),
        "graph_nodes": final_nodes,
        "graph_edges": final_edges,
        "graph_total_weight": total_weight,
    }
    return final_graph, graph_info


def make_igraph(graph: nx.Graph):
    try:
        import igraph as ig
    except ImportError as exc:
        raise ImportError(
            "python-igraph is required. Install it with: "
            "python3 -m pip install python-igraph"
        ) from exc

    nodes = list(graph.nodes())
    node_to_index = {node: index for index, node in enumerate(nodes)}
    nx_edges = list(graph.edges())
    ig_edges = [(node_to_index[u], node_to_index[v]) for u, v in nx_edges]
    weights = [
        safe_float(graph[u][v].get("weight", 1.0), 1.0)
        for u, v in nx_edges
    ]

    ig_graph = ig.Graph(n=len(nodes), edges=ig_edges, directed=False)
    ig_graph.es["weight"] = weights
    return ig_graph, nodes


def call_weighted_cpm_leiden(ig_graph, gamma: float):
    call_variants = [
        {
            "objective_function": LEIDEN_OBJECTIVE,
            "weights": "weight",
            "resolution": gamma,
            "beta": LEIDEN_BETA,
            "n_iterations": LEIDEN_N_ITERATIONS,
        },
        {
            "objective_function": LEIDEN_OBJECTIVE,
            "weights": "weight",
            "resolution_parameter": gamma,
            "beta": LEIDEN_BETA,
            "n_iterations": LEIDEN_N_ITERATIONS,
        },
        {
            "edge_weights": "weight",
            "resolution": gamma,
            "beta": LEIDEN_BETA,
            "n_iterations": LEIDEN_N_ITERATIONS,
        },
        {
            "edge_weights": "weight",
            "resolution_parameter": gamma,
            "beta": LEIDEN_BETA,
            "n_iterations": LEIDEN_N_ITERATIONS,
        },
    ]

    last_error = None
    for kwargs in call_variants:
        try:
            return ig_graph.community_leiden(**kwargs)
        except TypeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    raise RuntimeError("igraph community_leiden call failed.")


def membership_to_communities(membership, nodes: Sequence[Node]) -> List[Set[Node]]:
    grouped: Dict[int, Set[Node]] = defaultdict(set)
    for node_index, community_id in enumerate(membership):
        grouped[int(community_id)].add(nodes[node_index])
    return list(grouped.values())


def split_by_connectivity(
    graph: nx.Graph,
    communities: Iterable[Iterable[Node]],
) -> List[Set[Node]]:
    split_communities: List[Set[Node]] = []
    for community in communities:
        subgraph = graph.subgraph(list(community))
        for component in nx.connected_components(subgraph):
            split_communities.append(set(component))
    split_communities.sort(key=len, reverse=True)
    return split_communities


def summarize_partition(
    graph: nx.Graph,
    communities: Sequence[Set[Node]],
) -> Dict[str, float]:
    split_communities = split_by_connectivity(graph, communities)
    sizes_all = np.asarray([len(community) for community in split_communities], dtype=int)
    sizes_kept = sizes_all[sizes_all >= MIN_COMMUNITY_SIZE]
    singleton_count = int(np.sum(sizes_all == 1))
    graph_nodes = graph.number_of_nodes()

    return {
        "community_count_before_connectivity_split": len(communities),
        "community_count_after_connectivity_split": len(split_communities),
        "community_count_size_ge_2": int(np.sum(sizes_all >= 2)),
        "community_count_size_ge_5": int(np.sum(sizes_all >= 5)),
        "community_count_size_ge_10": int(np.sum(sizes_all >= 10)),
        "community_count_size_ge_50": int(np.sum(sizes_all >= 50)),
        "community_count_size_ge_100": int(np.sum(sizes_all >= 100)),
        "singleton_count": singleton_count,
        "singleton_fraction_of_communities": (
            singleton_count / len(sizes_all) if len(sizes_all) else 0.0
        ),
        "singleton_fraction_of_nodes": (
            singleton_count / graph_nodes if graph_nodes else 0.0
        ),
        "largest_community_size": int(sizes_all.max()) if len(sizes_all) else 0,
        "median_community_size_all": (
            float(np.median(sizes_all)) if len(sizes_all) else 0.0
        ),
        "mean_community_size_all": (
            float(np.mean(sizes_all)) if len(sizes_all) else 0.0
        ),
        "median_community_size_ge_2": (
            float(np.median(sizes_kept)) if len(sizes_kept) else 0.0
        ),
        "mean_community_size_ge_2": (
            float(np.mean(sizes_kept)) if len(sizes_kept) else 0.0
        ),
        "largest_community_node_fraction": (
            int(sizes_all.max()) / graph_nodes
            if len(sizes_all) and graph_nodes
            else 0.0
        ),
    }


def community_size_distribution(
    gamma: float,
    selected_run: int,
    graph: nx.Graph,
    communities: Sequence[Set[Node]],
) -> List[Dict[str, float]]:
    split_communities = split_by_connectivity(graph, communities)
    size_counts = pd.Series(
        [len(community) for community in split_communities],
        dtype="int64",
    ).value_counts().sort_index()
    graph_nodes = graph.number_of_nodes()

    rows: List[Dict[str, float]] = []
    for community_size, community_count in size_counts.items():
        nodes_at_size = int(community_size) * int(community_count)
        rows.append(
            {
                "gamma": gamma,
                "selected_run": selected_run,
                "community_size": int(community_size),
                "community_count": int(community_count),
                "nodes_in_this_size": nodes_at_size,
                "fraction_of_graph_nodes": (
                    nodes_at_size / graph_nodes if graph_nodes else 0.0
                ),
            }
        )
    return rows


def run_gamma_sensitivity(
    graph: nx.Graph,
    graph_info: Dict[str, float],
    gammas: Sequence[float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_runs_path = output_dir / "gamma_all_runs.csv"
    selected_summary_path = output_dir / "gamma_selected_summary.csv"
    size_distribution_path = output_dir / "gamma_selected_size_distribution.csv"

    ig_graph, nodes = make_igraph(graph)
    all_run_rows: List[Dict[str, float]] = []
    selected_rows: List[Dict[str, float]] = []
    distribution_rows: List[Dict[str, float]] = []

    for gamma in gammas:
        print("\n" + "=" * 88)
        print(f"Running gamma={gamma:g}")
        print("=" * 88)

        gamma_candidates = []

        for run_index in range(COMMUNITY_RUNS):
            trial_seed = COMMUNITY_SEED + run_index

            # Set the Python and NumPy seeds used by the clustering workflow.
            random.seed(trial_seed)
            np.random.seed(trial_seed)

            clustering = call_weighted_cpm_leiden(ig_graph, gamma)
            communities = membership_to_communities(clustering.membership, nodes)
            modularity = float(
                nx_modularity(
                    graph,
                    communities,
                    weight="weight",
                    resolution=gamma,
                )
            )
            cpm_quality = safe_float(getattr(clustering, "quality", np.nan), np.nan)
            partition_stats = summarize_partition(graph, communities)

            run_row = {
                "gamma": gamma,
                "run": run_index + 1,
                "seed": trial_seed,
                "weighted_modularity": modularity,
                "cpm_quality": cpm_quality,
                **partition_stats,
                "graph_nodes": graph_info["graph_nodes"],
                "graph_edges": graph_info["graph_edges"],
                "graph_total_weight": graph_info["graph_total_weight"],
            }
            all_run_rows.append(run_row)
            gamma_candidates.append((modularity, run_index + 1, cpm_quality, communities, run_row))

            print(
                f"  run={run_index + 1}, seed={trial_seed}, "
                f"modularity={modularity:.10f}, "
                f"communities>=2={partition_stats['community_count_size_ge_2']:,}, "
                f"largest={partition_stats['largest_community_size']:,}"
            )

        # Select the candidate with maximum weighted modularity.
        best_modularity, selected_run, selected_cpm_quality, selected_partition, best_row = max(
            gamma_candidates,
            key=lambda item: item[0],
        )

        gamma_run_rows = [row for row in all_run_rows if row["gamma"] == gamma]
        selected_row = {
            "gamma": gamma,
            "selected_run": selected_run,
            "selected_seed": COMMUNITY_SEED + selected_run - 1,
            "selected_weighted_modularity": best_modularity,
            "selected_cpm_quality": selected_cpm_quality,
            "input_csv": graph_info["input_csv"],
            "score_column": SCORE_COLUMN,
            "bin_size": BIN_SIZE,
            "dbscan_eps": DBSCAN_EPS,
            "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
            "dbscan_metric": DBSCAN_METRIC,
            "leiden_objective": LEIDEN_OBJECTIVE,
            "leiden_beta": LEIDEN_BETA,
            "leiden_n_iterations": LEIDEN_N_ITERATIONS,
            "community_runs": COMMUNITY_RUNS,
            "community_seed": COMMUNITY_SEED,
            "selection_rule": "maximum_weighted_networkx_modularity",
            **{
                key: value
                for key, value in best_row.items()
                if key
                not in {
                    "gamma",
                    "run",
                    "seed",
                    "weighted_modularity",
                    "cpm_quality",
                }
            },
            "all_runs_community_ge2_min": min(
                row["community_count_size_ge_2"] for row in gamma_run_rows
            ),
            "all_runs_community_ge2_max": max(
                row["community_count_size_ge_2"] for row in gamma_run_rows
            ),
            "all_runs_largest_min": min(
                row["largest_community_size"] for row in gamma_run_rows
            ),
            "all_runs_largest_max": max(
                row["largest_community_size"] for row in gamma_run_rows
            ),
        }

        selected_rows.append(selected_row)
        distribution_rows.extend(
            community_size_distribution(
                gamma,
                selected_run,
                graph,
                selected_partition,
            )
        )

        # Save checkpoints after every gamma.
        pd.DataFrame(all_run_rows).to_csv(all_runs_path, index=False)
        pd.DataFrame(selected_rows).to_csv(selected_summary_path, index=False)
        pd.DataFrame(distribution_rows).to_csv(size_distribution_path, index=False)

        print(
            f"Selected run {selected_run}: modularity={best_modularity:.10f}, "
            f"communities>=2={selected_row['community_count_size_ge_2']:,}, "
            f"largest={selected_row['largest_community_size']:,}"
        )

    print("\n" + "=" * 88)
    print("Gamma sensitivity analysis finished")
    print(f"Selected summaries : {selected_summary_path}")
    print(f"All runs          : {all_runs_path}")
    print(f"Size distributions: {size_distribution_path}")
    print("No GraphML or graph-plot files were generated.")
    print("=" * 88)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run gamma sensitivity analysis on a ProDive fragment graph "
            "without generating GraphML files."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="ProDive subset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the three sensitivity-analysis CSV files.",
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=DEFAULT_GAMMAS,
        help="Gamma values to test.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="CSV chunk size used while constructing the graph.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1.")
    if not args.gammas:
        raise ValueError("At least one gamma value is required.")
    if any(gamma <= 0 for gamma in args.gammas):
        raise ValueError("All gamma values must be greater than 0.")
    if len(set(args.gammas)) != len(args.gammas):
        raise ValueError("Duplicate gamma values are not allowed.")

    gammas = sorted(args.gammas)

    print("=" * 88)
    print("ProDive Leiden gamma sensitivity analysis")
    print(f"Input CSV        : {args.input_csv}")
    print(f"Output directory : {args.output_dir}")
    print(f"Gamma values     : {', '.join(f'{gamma:g}' for gamma in gammas)}")
    print(f"Leiden objective : {LEIDEN_OBJECTIVE}")
    print(f"Runs per gamma   : {COMMUNITY_RUNS}")
    print(f"Base seed        : {COMMUNITY_SEED}")
    print(f"Beta             : {LEIDEN_BETA}")
    print(f"Iterations       : {LEIDEN_N_ITERATIONS}")
    print("Selection rule   : maximum weighted NetworkX modularity")
    print("=" * 88)

    graph, graph_info = build_fragment_graph(
        csv_path=args.input_csv,
        chunk_size=args.chunk_size,
        bin_size=BIN_SIZE,
        dbscan_eps=DBSCAN_EPS,
        dbscan_min_samples=DBSCAN_MIN_SAMPLES,
        dbscan_metric=DBSCAN_METRIC,
    )
    run_gamma_sensitivity(
        graph=graph,
        graph_info=graph_info,
        gammas=gammas,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
