#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download Start2Fold XML records."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

DEFAULT_OUTPUT_DIR = Path("CHANGE_ME")
DEFAULT_BASE_URL = "https://www.bio2byte.be/start2fold/STF{:04d}.xml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Start2Fold XML records by STF numeric ID range.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for downloaded XML files.")
    p.add_argument("--start-id", type=int, default=1, help="First STF numeric ID to scan, inclusive.")
    p.add_argument("--end-id", type=int, default=100, help="Last STF numeric ID to scan, inclusive.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL template with one integer placeholder.")
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP request timeout in seconds.")
    p.add_argument("--sleep", type=float, default=0.2, help="Sleep interval between requests in seconds.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing XML files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading Start2Fold XML files to: {args.output_dir}")
    valid_count = 0
    skipped_existing = 0

    for i in range(args.start_id, args.end_id + 1):
        stf_id = f"STF{i:04d}"
        out_path = args.output_dir / f"{stf_id}.xml"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {stf_id} already exists")
            skipped_existing += 1
            continue

        url = args.base_url.format(i)
        try:
            response = requests.get(url, timeout=args.timeout)
            if response.status_code == 200:
                out_path.write_bytes(response.content)
                print(f"[OK] downloaded {stf_id}")
                valid_count += 1
            elif response.status_code == 404:
                print(f"[SKIP] {stf_id} not found")
            else:
                print(f"[WARN] {stf_id} HTTP status: {response.status_code}")
        except requests.RequestException as exc:
            print(f"[ERROR] request failed for {stf_id}: {exc}")

        time.sleep(args.sleep)

    print("=" * 50)
    print(f"[INFO] Download complete. downloaded={valid_count}, skipped_existing={skipped_existing}")
    print("=" * 50)


if __name__ == "__main__":
    main()
