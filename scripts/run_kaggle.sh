#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/kaggle.yaml}"
RUN_NAME="${2:-kaggle}"
OUTPUT_ROOT="/kaggle/working/outputs/runs"
ARCHIVE="/kaggle/working/${RUN_NAME}-results.tar.gz"

echo "[kaggle] Running experiment with ${CONFIG}"
uv run mlu run -c "${CONFIG}"

echo "[kaggle] Archiving results to ${ARCHIVE}"
tar -czf "${ARCHIVE}" -C "${OUTPUT_ROOT}" "${RUN_NAME}"
echo "[kaggle] Complete. Download ${ARCHIVE} from the Kaggle Files panel."

