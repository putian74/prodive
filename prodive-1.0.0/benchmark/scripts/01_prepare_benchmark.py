#!/usr/bin/env python3
"""Create deterministic, workload-stratified ProDive pair-list samples."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from benchmark_common import (
    collect_hardware_inventory,
    load_config,
    read_packed_index,
    resolved,
    write_json,
)


def unique_random_pair_codes(rng: np.random.Generator, n: int, target: int) -> np.ndarray:
    """Sample unique unordered non-self pair codes without enumerating N choose 2."""
    total = n * (n - 1) // 2
    target = min(int(target), int(total))
    if target <= 0:
        return np.empty(0, dtype=np.int64)

    collected = np.empty(0, dtype=np.int64)
    while collected.size < target:
        needed = target - collected.size
        batch_size = max(10000, int(needed * 1.25))
        left = rng.integers(0, n, size=batch_size, dtype=np.int64)
        right = rng.integers(0, n, size=batch_size, dtype=np.int64)
        mask = left != right
        lo = np.minimum(left[mask], right[mask])
        hi = np.maximum(left[mask], right[mask])
        codes = lo * np.int64(n) + hi
        collected = np.unique(np.concatenate((collected, codes)))
    if collected.size > target:
        keep = rng.choice(collected.size, size=target, replace=False)
        collected = collected[keep]
    return collected


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values.astype(np.float64), q)) if values.size else float("nan")


def select_cpu_validation_rows(
    rows: List[Dict[str, Any]], pair_count: int, max_window_cells_per_pair: int
) -> List[Dict[str, Any]]:
    """Select a deterministic, bounded-workload subset for the serial CPU run."""
    eligible = [
        dict(row)
        for row in rows
        if int(row["window_cells"]) <= int(max_window_cells_per_pair)
    ]
    if len(eligible) < pair_count:
        raise ValueError(
            f"Only {len(eligible)} sampled pairs have window_cells <= "
            f"{max_window_cells_per_pair:,}; cpu_gpu_validation.pair_count={pair_count}. "
            "Increase max_window_cells_per_pair, increase source_pair_count, or reduce pair_count."
        )

    # The master list is already interleaved by workload stratum.  Taking the
    # bounded eligible prefix preserves its deterministic ordering while
    # preventing one very large matrix from turning a validation run into an
    # accidental full CPU benchmark.
    selected = eligible[:pair_count]
    for rank, row in enumerate(selected, start=1):
        row["sample_rank"] = rank
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic ProDive benchmark pair lists.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Replace existing generated pair-list files.")
    args = parser.parse_args()

    config = load_config(args.config)
    packed_db = resolved(config, "packed_db")
    output_root = resolved(config, "output_root")
    design_dir = output_root / "design"
    pair_dir = design_dir / "pair_lists"
    pair_dir.mkdir(parents=True, exist_ok=True)

    sampling = config["sampling"]
    seed = int(sampling["seed"])
    sampling_pair_counts = sorted(
        {int(value) for value in sampling["pair_counts"]}
    )
    core_pair_counts = sorted(
        {
            int(value)
            for value in config.get("core", {}).get(
                "pair_counts", sampling_pair_counts
            )
        }
    )
    comparison = config.get("cpu_gpu_comparison", {})
    comparison_pair_counts = (
        sorted({int(value) for value in comparison.get("pair_counts", [])})
        if bool(comparison.get("enabled", True))
        else []
    )
    # One deterministic master list serves every mode.  Existing GPU pair
    # lists remain byte-identical because adding smaller nested prefixes does
    # not change the largest requested prefix or the random seed.
    pair_counts = sorted(
        set(sampling_pair_counts)
        | set(core_pair_counts)
        | set(comparison_pair_counts)
    )
    strata_count = int(sampling.get("workload_strata", 5))
    candidate_pool_size = int(sampling.get("candidate_pool_size", max(pair_counts) * 20))
    if not pair_counts or pair_counts[0] <= 0:
        raise ValueError("sampling.pair_counts must contain positive integers")
    if strata_count <= 0:
        raise ValueError("sampling.workload_strata must be positive")
    if not bool(sampling.get("exclude_self_pairs", True)):
        raise ValueError("This benchmark design currently requires exclude_self_pairs=true")

    records = read_packed_index(packed_db / "index.csv")
    records.sort(key=lambda row: int(row["family_id"]))
    family_ids = np.asarray([int(row["family_id"]) for row in records], dtype=np.int64)
    nfrags = np.asarray([int(row["nfrag"]) for row in records], dtype=np.int64)
    n = int(family_ids.size)
    total_pairs_excluding_self = n * (n - 1) // 2
    if pair_counts[-1] > total_pairs_excluding_self:
        raise ValueError(
            f"Requested {pair_counts[-1]:,} pairs but only {total_pairs_excluding_self:,} "
            "unordered non-self pairs are available"
        )
    candidate_pool_size = max(candidate_pool_size, pair_counts[-1] * strata_count)
    candidate_pool_size = min(candidate_pool_size, total_pairs_excluding_self)

    existing = [pair_dir / f"pairs_{count}.csv" for count in pair_counts]
    existing.extend(
        [
            pair_dir / "pairs_full_family_coverage.csv",
            design_dir / "sampling_manifest.json",
        ]
    )
    if bool(config.get("cpu_gpu_validation", {}).get("enabled", True)):
        existing.append(pair_dir / "pairs_cpu_gpu_validation.csv")
    if not args.force and all(path.is_file() for path in existing):
        print("All requested pair lists and manifests already exist; preparation was not repeated.")
        for path in existing:
            print(path)
        return

    rng = np.random.default_rng(seed)
    codes = unique_random_pair_codes(rng, n=n, target=candidate_pool_size)
    left_pos = codes // np.int64(n)
    right_pos = codes % np.int64(n)
    candidate_cells = nfrags[left_pos] * nfrags[right_pos]

    sorted_positions = np.argsort(candidate_cells, kind="mergesort")
    strata = [part.copy() for part in np.array_split(sorted_positions, strata_count) if part.size]
    for part in strata:
        rng.shuffle(part)

    master_size = pair_counts[-1]
    selected_positions: List[int] = []
    depth = 0
    while len(selected_positions) < master_size:
        made_progress = False
        for part in strata:
            if depth < part.size and len(selected_positions) < master_size:
                selected_positions.append(int(part[depth]))
                made_progress = True
        if not made_progress:
            break
        depth += 1
    if len(selected_positions) != master_size:
        raise RuntimeError(f"Could select only {len(selected_positions):,} of {master_size:,} requested pairs")

    stratum_lookup = np.full(codes.size, -1, dtype=np.int16)
    stratum_descriptions = []
    for stratum_index, part in enumerate(strata, start=1):
        stratum_lookup[part] = stratum_index
        cells = candidate_cells[part]
        stratum_descriptions.append(
            {
                "stratum": stratum_index,
                "label": f"Q{stratum_index}",
                "candidate_count": int(part.size),
                "min_window_cells": int(cells.min()),
                "max_window_cells": int(cells.max()),
                "median_window_cells": percentile(cells, 50),
            }
        )

    selected = np.asarray(selected_positions, dtype=np.int64)
    selected_left = left_pos[selected]
    selected_right = right_pos[selected]
    selected_rows: List[Dict[str, Any]] = []
    for rank, (lp, rp, candidate_position) in enumerate(
        zip(selected_left, selected_right, selected), start=1
    ):
        selected_rows.append(
            {
                "i": int(family_ids[lp]),
                "j": int(family_ids[rp]),
                "nfrag_i": int(nfrags[lp]),
                "nfrag_j": int(nfrags[rp]),
                "window_cells": int(nfrags[lp] * nfrags[rp]),
                "workload_stratum": f"Q{int(stratum_lookup[candidate_position])}",
                "sample_rank": int(rank),
            }
        )

    pair_summaries: List[Dict[str, Any]] = []
    fields = ["i", "j", "nfrag_i", "nfrag_j", "window_cells", "workload_stratum", "sample_rank"]
    for count in pair_counts:
        rows = selected_rows[:count]
        pair_path = pair_dir / f"pairs_{count}.csv"
        with pair_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        family_set = {int(row["i"]) for row in rows} | {int(row["j"]) for row in rows}
        cells = np.asarray([int(row["window_cells"]) for row in rows], dtype=np.int64)
        counts_by_stratum = {
            f"Q{idx}": sum(row["workload_stratum"] == f"Q{idx}" for row in rows)
            for idx in range(1, len(strata) + 1)
        }
        pair_summaries.append(
            {
                "pair_count": int(count),
                "unique_families": int(len(family_set)),
                "total_window_cells": int(cells.sum()),
                "estimated_float32_matrix_payload_bytes": int(cells.sum()) * 4,
                "estimated_npy_bytes_with_128_byte_header": int(cells.sum()) * 4 + int(count) * 128,
                "min_window_cells_per_pair": int(cells.min()),
                "median_window_cells_per_pair": percentile(cells, 50),
                "p95_window_cells_per_pair": percentile(cells, 95),
                "max_window_cells_per_pair": int(cells.max()),
                **{f"pairs_{key}": int(value) for key, value in counts_by_stratum.items()},
                "pair_list": str(pair_path),
            }
        )
        ids_path = pair_dir / f"family_ids_{count}.txt"
        ids_path.write_text("".join(f"{family_id}\n" for family_id in sorted(family_set)), encoding="utf-8")

    cpu_validation_summary: Dict[str, Any] = {"enabled": False}
    cpu_validation = config.get("cpu_gpu_validation", {})
    if bool(cpu_validation.get("enabled", True)):
        cpu_pair_count = int(cpu_validation.get("pair_count", 5))
        cpu_source_pair_count = int(
            cpu_validation.get("source_pair_count", pair_counts[0])
        )
        cpu_max_cells = int(
            cpu_validation.get("max_window_cells_per_pair", 10000)
        )
        if cpu_pair_count <= 0 or cpu_source_pair_count <= 0 or cpu_max_cells <= 0:
            raise ValueError(
                "cpu_gpu_validation pair_count, source_pair_count, and "
                "max_window_cells_per_pair must be positive"
            )
        if cpu_source_pair_count > len(selected_rows):
            raise ValueError(
                f"cpu_gpu_validation.source_pair_count={cpu_source_pair_count} exceeds "
                f"the largest sampled pair count {len(selected_rows)}"
            )
        cpu_rows = select_cpu_validation_rows(
            selected_rows[:cpu_source_pair_count],
            pair_count=cpu_pair_count,
            max_window_cells_per_pair=cpu_max_cells,
        )
        cpu_pair_path = pair_dir / "pairs_cpu_gpu_validation.csv"
        with cpu_pair_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(cpu_rows)
        cpu_family_set = {int(row["i"]) for row in cpu_rows} | {
            int(row["j"]) for row in cpu_rows
        }
        cpu_cells = np.asarray(
            [int(row["window_cells"]) for row in cpu_rows], dtype=np.int64
        )
        cpu_validation_summary = {
            "enabled": True,
            "purpose": "CPU/GPU correctness and serial-CPU resource comparison on raw symmetric KL only",
            "pair_list": str(cpu_pair_path),
            "pair_count": int(len(cpu_rows)),
            "source_pair_count": cpu_source_pair_count,
            "max_window_cells_per_pair_limit": cpu_max_cells,
            "unique_families": int(len(cpu_family_set)),
            "total_window_cells": int(cpu_cells.sum()),
            "min_window_cells_per_pair": int(cpu_cells.min()),
            "median_window_cells_per_pair": percentile(cpu_cells, 50),
            "max_window_cells_per_pair": int(cpu_cells.max()),
            "workload_strata_observed": sorted(
                {str(row["workload_stratum"]) for row in cpu_rows}
            ),
            "raw_kl_only": True,
            "normalization_or_background_filtering_included": False,
        }
        write_json(design_dir / "cpu_gpu_validation_sample_summary.json", cpu_validation_summary)

    # A random matching touches every family while requiring only ceil(N/2)
    # pair calculations.  This is a memory-stress sample for packed-data loading
    # and CPU matrix inversion; it is kept separate from the nested scaling sets.
    coverage_rng = np.random.default_rng(seed + 919)
    coverage_positions = np.arange(n, dtype=np.int64)
    coverage_rng.shuffle(coverage_positions)
    coverage_pairs = []
    for offset in range(0, n - 1, 2):
        coverage_pairs.append((int(coverage_positions[offset]), int(coverage_positions[offset + 1])))
    if n % 2 == 1:
        coverage_pairs.append((int(coverage_positions[-1]), int(coverage_positions[0])))
    coverage_rows: List[Dict[str, Any]] = []
    for rank, (left, right) in enumerate(coverage_pairs, start=1):
        if family_ids[left] > family_ids[right]:
            left, right = right, left
        coverage_rows.append(
            {
                "i": int(family_ids[left]),
                "j": int(family_ids[right]),
                "nfrag_i": int(nfrags[left]),
                "nfrag_j": int(nfrags[right]),
                "window_cells": int(nfrags[left] * nfrags[right]),
                "workload_stratum": "full_family_coverage",
                "sample_rank": int(rank),
            }
        )
    coverage_path = pair_dir / "pairs_full_family_coverage.csv"
    with coverage_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(coverage_rows)
    coverage_cells = np.asarray(
        [int(row["window_cells"]) for row in coverage_rows], dtype=np.int64
    )
    coverage_summary = {
        "sample_kind": "full_family_coverage",
        "pair_count": int(len(coverage_rows)),
        "unique_families": int(n),
        "total_window_cells": int(coverage_cells.sum()),
        "estimated_float32_matrix_payload_bytes": int(coverage_cells.sum()) * 4,
        "estimated_npy_bytes_with_128_byte_header": int(coverage_cells.sum()) * 4
        + int(len(coverage_rows)) * 128,
        "min_window_cells_per_pair": int(coverage_cells.min()),
        "median_window_cells_per_pair": percentile(coverage_cells, 50),
        "p95_window_cells_per_pair": percentile(coverage_cells, 95),
        "max_window_cells_per_pair": int(coverage_cells.max()),
        "pair_list": str(coverage_path),
        "purpose": (
            "Touch every packed family with ceil(N/2) pair calculations to measure "
            "full-family data-loading and CPU-inversion memory without all-pairs computation."
        ),
    }
    write_json(design_dir / "full_family_coverage_summary.json", coverage_summary)

    summary_csv = design_dir / "pair_sample_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(pair_summaries)

    sum_nfrag = int(nfrags.sum())
    sum_squared_nfrag = int(np.dot(nfrags, nfrags))
    full_cells_excluding_self = (sum_nfrag * sum_nfrag - sum_squared_nfrag) // 2
    full_cells_including_self = (sum_nfrag * sum_nfrag + sum_squared_nfrag) // 2
    workload = {
        "packed_database": str(packed_db),
        "family_count": n,
        "family_id_min": int(family_ids.min()),
        "family_id_max": int(family_ids.max()),
        "sum_nfrag": sum_nfrag,
        "sum_squared_nfrag": sum_squared_nfrag,
        "unordered_pairs_excluding_self": int(total_pairs_excluding_self),
        "upper_triangle_pairs_including_self": int(n * (n + 1) // 2),
        "ordered_pairs_including_self": int(n * n),
        "window_cells_unordered_excluding_self": int(full_cells_excluding_self),
        "window_cells_upper_triangle_including_self": int(full_cells_including_self),
        "window_cells_ordered_including_self": int(sum_nfrag * sum_nfrag),
        "nfrag_min": int(nfrags.min()),
        "nfrag_median": percentile(nfrags, 50),
        "nfrag_p95": percentile(nfrags, 95),
        "nfrag_max": int(nfrags.max()),
        "definition": "window_cells for pair (i,j) = nfrag_i * nfrag_j",
    }
    write_json(design_dir / "full_workload.json", workload)

    manifest = {
        "created_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": seed,
        "sampling_unit": "unordered non-self family pair (i < j)",
        "sampling_method": (
            "Uniform random candidate pairs, stratified into candidate-pool quantiles by "
            "nfrag_i*nfrag_j, then interleaved so every pair-list prefix is balanced across strata."
        ),
        "candidate_pool_size": int(codes.size),
        "workload_strata": len(strata),
        "stratum_descriptions": stratum_descriptions,
        "pair_counts": pair_counts,
        "sampling_pair_counts": sampling_pair_counts,
        "gpu_profile_pair_counts": core_pair_counts,
        "cpu_gpu_comparison_pair_counts": comparison_pair_counts,
        "pair_lists_are_nested_prefixes": True,
        "pair_summaries": pair_summaries,
        "cpu_gpu_validation_sample": cpu_validation_summary,
        "full_family_coverage_sample": coverage_summary,
        "full_workload_json": str(design_dir / "full_workload.json"),
        "index_csv": str(packed_db / "index.csv"),
    }
    write_json(design_dir / "sampling_manifest.json", manifest)
    write_json(design_dir / "resolved_config_snapshot.json", config)
    write_json(
        output_root / "hardware_inventory.json",
        collect_hardware_inventory(output_root, packed_db),
    )

    print(f"Prepared {len(pair_counts)} nested pair lists with fixed seed {seed}.")
    print(f"Sampling manifest: {design_dir / 'sampling_manifest.json'}")
    print(f"Pair summary:      {summary_csv}")
    print(f"Full workload:     {design_dir / 'full_workload.json'}")


if __name__ == "__main__":
    main()
