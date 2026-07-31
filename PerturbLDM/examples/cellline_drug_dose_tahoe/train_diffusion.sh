#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID=${GPU_ID:-0}
: "${DATA_DIR:?Set DATA_DIR to a processed Tahoe split folder containing collection/ and processed/.}"
: "${LATENT_DIR:?Set LATENT_DIR to a trained Tahoe latent-model output directory.}"

python -u "${SCRIPT_DIR}/train_diffusionmlp_model.py" --gpu="${GPU_ID}" --data_root="${DATA_DIR}" --num_epochs=4 --dose_level_mapping_dim=2 \
  --latent_dir="${LATENT_DIR}" \
  >> output_train_diff_$(date +%F_%H-%M-%S).log 2>&1
