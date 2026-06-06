#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/experiments/colab_1_5b.yaml"
RUN_NAME="colab_1_5b"
OUTPUT_ROOT="/content/outputs/runs"
LOG="/content/${RUN_NAME}.log"
ARCHIVE="/content/${RUN_NAME}-results.tar.gz"

archive_results() {
  if [[ -d "${OUTPUT_ROOT}/${RUN_NAME}" ]]; then
    echo "[colab] Archiving current results to ${ARCHIVE}"
    tar -czf "${ARCHIVE}" -C "${OUTPUT_ROOT}" "${RUN_NAME}"
  fi
}

trap archive_results EXIT

echo "[colab] Verifying CUDA"
uv run python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"

echo "[colab] Running ${RUN_NAME}; log=${LOG}"
set -o pipefail
uv run mlu run -c "${CONFIG}" 2>&1 | tee "${LOG}"

echo "[colab] Complete. Download ${ARCHIVE} and ${LOG} before the runtime ends."
