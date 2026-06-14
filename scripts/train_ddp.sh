#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Convenience wrapper: launch EveryDream2 training with DDP.
#
# Usage examples:
#   # 2-GPU run
#   bash scripts/train_ddp.sh --nproc 2 --config train.json
#
#   # 3-GPU run with extra CLI overrides
#   bash scripts/train_ddp.sh --nproc 3 --config train-sdxl.json --max_epochs 50
#
# All arguments after --nproc N and --config <file> are forwarded verbatim
# to train.py.
# ---------------------------------------------------------------------------
set -euo pipefail

NPROC=2
CONFIG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc)
      NPROC="$2"; shift 2 ;;
    --config)
      CONFIG="$2"; shift 2 ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Usage: $0 --nproc <N> --config <config.json> [extra train.py args...]"
  exit 1
fi

echo "Launching DDP training: nproc_per_node=${NPROC}, config=${CONFIG}"
torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port=29500 \
  train.py \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"

