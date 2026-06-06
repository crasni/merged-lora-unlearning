#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  echo "Usage: scripts/run.sh <profile>"
  echo "Available profiles:"
  for config in configs/experiments/*.yaml; do
    basename "${config}" .yaml
  done
}

if [[ $# -ne 1 ]]; then
  show_usage
  exit 2
fi

PROFILE="$1"
CONFIG="configs/experiments/${PROFILE}.yaml"
LOG_DIR="outputs/logs"
LOG="${LOG_DIR}/${PROFILE}.log"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Unknown profile: ${PROFILE}"
  show_usage
  exit 2
fi

mkdir -p "${LOG_DIR}"

echo "[run] Verifying CUDA"
uv run python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"

echo "[run] Profile=${PROFILE} config=${CONFIG} log=${LOG}"
set -o pipefail
uv run mlu run -c "${CONFIG}" 2>&1 | tee "${LOG}"
