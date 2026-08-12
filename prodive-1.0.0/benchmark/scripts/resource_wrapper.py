#!/usr/bin/env python3
"""Run one command and record per-invocation child resource usage.

This is a fallback for systems without GNU time.  Linux /proc process-tree
sampling remains the preferred source because it can capture concurrent summed
RSS across multiprocessing descendants.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("resource_wrapper.py requires a command after --")

    process = subprocess.Popen(command)
    returncode = process.wait()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    payload = {
        "max_rss_platform_units": float(usage.ru_maxrss),
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
        "minor_page_faults": int(usage.ru_minflt),
        "major_page_faults": int(usage.ru_majflt),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "platform": sys.platform,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
