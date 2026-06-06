#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/experiments/local_0_5b.yaml"
RUN_NAME="local_0_5b"
OUTPUT_ROOT="outputs/runs"
ARTIFACT_ROOT="outputs/local"
LOG="${ARTIFACT_ROOT}/${RUN_NAME}.log"
ARCHIVE="${ARTIFACT_ROOT}/${RUN_NAME}-results.tar.gz"

mkdir -p "${ARTIFACT_ROOT}"

archive_results() {
  if [[ -d "${OUTPUT_ROOT}/${RUN_NAME}" ]]; then
    echo "[local] Archiving current results to ${ARCHIVE}"
    tar -czf "${ARCHIVE}" -C "${OUTPUT_ROOT}" "${RUN_NAME}"
  fi
}

trap archive_results EXIT

echo "[local] Verifying CUDA"
uv run python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"

echo "[local] Running ${RUN_NAME}; log=${LOG}"
set -o pipefail
uv run mlu run -c "${CONFIG}" 2>&1 | tee "${LOG}"

echo "[local] Complete. Results=${OUTPUT_ROOT}/${RUN_NAME}"
echo "[local] Log=${LOG}"
echo "[local] Archive=${ARCHIVE}"

