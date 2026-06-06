#!/usr/bin/env bash

if [[ -n "${SCRATCH:-}" ]]; then
  export HF_HOME="${HF_HOME:-$SCRATCH/$USER/hf_cache}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export TORCH_HOME="${TORCH_HOME:-$SCRATCH/$USER/torch_cache}"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

