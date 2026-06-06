#!/usr/bin/env bash
set -euo pipefail

mlu data -c configs/experiments/tiny_local.yaml
mlu status -c configs/experiments/tiny_local.yaml

