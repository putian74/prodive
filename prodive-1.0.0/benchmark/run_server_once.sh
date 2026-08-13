#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

MODE=${1:-all}

case "$MODE" in
  gpu|compare|all|core|validation|summary) ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    echo "Use one of: gpu, compare, all, core, validation, summary" >&2
    exit 2
    ;;
esac

if [[ ! -f benchmark/config.json ]]; then
  echo "Missing benchmark/config.json." >&2
  echo "Create it with: cp benchmark/config.example.json benchmark/config.json" >&2
  echo "Then replace all /path/to placeholders and set the available GPU IDs." >&2
  exit 2
fi

if [[ "$MODE" != "summary" && ! -x src/cpp_cuda/kl_divergence ]]; then
  echo "C++/CUDA executable is missing; compiling it now..."
  make -C src/cpp_cuda -j "$(nproc)"
fi

bash benchmark/run_benchmark.sh benchmark/config.json "$MODE"
