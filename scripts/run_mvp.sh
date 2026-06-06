#!/usr/bin/env bash
set -euo pipefail

source scripts/workstation_env.sh
mlu run -c configs/experiments/qwen_mvp.yaml

