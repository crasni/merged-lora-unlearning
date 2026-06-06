#!/usr/bin/env bash
#SBATCH --job-name=merged-lora-unlearning
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

source ~/.bashrc
conda activate lora_unlearn
cd "${SLURM_SUBMIT_DIR:-$PWD}"
scripts/run_mvp.sh

