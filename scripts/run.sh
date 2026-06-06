#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-0_5b}"
CONFIG="configs/experiments/${PROFILE}.yaml"
LOG_DIR="outputs/logs"
LOG="${LOG_DIR}/${PROFILE}.log"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Usage: scripts/run.sh [0_5b|1_5b]"
  exit 2
fi

mkdir -p "${LOG_DIR}"

echo "[run] Verifying CUDA"
uv run python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"

echo "[run] Profile=${PROFILE} config=${CONFIG} log=${LOG}"
set -o pipefail
uv run mlu run -c "${CONFIG}" 2>&1 | tee "${LOG}"
