#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash benchmark/run_benchmark.sh CONFIG.json [gpu|compare|all|validation|core|prepare|summary]" >&2
  exit 2
fi

CONFIG=$1
MODE=${2:-all}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "$MODE" in
  prepare)
    python3 "$SCRIPT_DIR/scripts/00_preflight.py" --config "$CONFIG" --mode core
    python3 "$SCRIPT_DIR/scripts/01_prepare_benchmark.py" --config "$CONFIG"
    ;;
  core)
    python3 "$SCRIPT_DIR/scripts/00_preflight.py" --config "$CONFIG" --mode core
    python3 "$SCRIPT_DIR/scripts/01_prepare_benchmark.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/02_run_core_benchmark.py" --config "$CONFIG" --profile gpu
    python3 "$SCRIPT_DIR/scripts/04_summarize_benchmark.py" --config "$CONFIG"
    ;;
  validation)
    python3 "$SCRIPT_DIR/scripts/00_preflight.py" --config "$CONFIG" --mode validation
    python3 "$SCRIPT_DIR/scripts/01_prepare_benchmark.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/05_run_cpu_gpu_validation.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/04_summarize_benchmark.py" --config "$CONFIG"
    ;;
  gpu)
    python3 "$SCRIPT_DIR/scripts/00_preflight.py" --config "$CONFIG" --mode gpu
    python3 "$SCRIPT_DIR/scripts/01_prepare_benchmark.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/02_run_core_benchmark.py" --config "$CONFIG" --profile gpu
    python3 "$SCRIPT_DIR/scripts/03_run_end_to_end_benchmark.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/04_summarize_benchmark.py" --config "$CONFIG"
    ;;
  compare)
    python3 "$SCRIPT_DIR/scripts/00_preflight.py" --config "$CONFIG" --mode compare
    python3 "$SCRIPT_DIR/scripts/01_prepare_benchmark.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/05_run_cpu_gpu_validation.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/02_run_core_benchmark.py" --config "$CONFIG" --profile compare
    python3 "$SCRIPT_DIR/scripts/06_run_cpu_gpu_comparison.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/04_summarize_benchmark.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/07_summarize_cpu_gpu_comparison.py" --config "$CONFIG"
    ;;
  all)
    python3 "$SCRIPT_DIR/scripts/00_preflight.py" --config "$CONFIG" --mode all
    python3 "$SCRIPT_DIR/scripts/01_prepare_benchmark.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/05_run_cpu_gpu_validation.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/02_run_core_benchmark.py" --config "$CONFIG" --profile gpu
    python3 "$SCRIPT_DIR/scripts/02_run_core_benchmark.py" --config "$CONFIG" --profile compare
    python3 "$SCRIPT_DIR/scripts/06_run_cpu_gpu_comparison.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/03_run_end_to_end_benchmark.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/04_summarize_benchmark.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/07_summarize_cpu_gpu_comparison.py" --config "$CONFIG"
    ;;
  summary)
    python3 "$SCRIPT_DIR/scripts/04_summarize_benchmark.py" --config "$CONFIG"
    python3 "$SCRIPT_DIR/scripts/07_summarize_cpu_gpu_comparison.py" --config "$CONFIG"
    ;;
  *)
    echo "Unknown mode: $MODE (expected gpu, compare, all, validation, core, prepare, or summary)" >&2
    exit 2
    ;;
esac
