#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  echo "Usage: scripts/run_paper.sh <baseline|main|scalability|all>"
}

if [[ $# -ne 1 ]]; then
  show_usage
  exit 2
fi

case "$1" in
  baseline)
    scripts/run.sh 1_5b_paper_base
    ;;
  main)
    scripts/run.sh 1_5b_paper
    ;;
  scalability)
    scripts/run.sh 1_5b_scale_05
    scripts/run.sh 1_5b_scale_10
    scripts/run.sh 1_5b_scale_40
    uv run python scripts/report_scalability.py
    ;;
  all)
    scripts/run.sh 1_5b_paper_base
    scripts/run.sh 1_5b_paper
    scripts/run.sh 1_5b_scale_05
    scripts/run.sh 1_5b_scale_10
    scripts/run.sh 1_5b_scale_40
    uv run python scripts/report_scalability.py
    ;;
  *)
    show_usage
    exit 2
    ;;
esac
