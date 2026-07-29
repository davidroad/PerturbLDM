#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID=${GPU_ID:-0}
: "${DATA_DIR:?Set DATA_DIR to a processed Tahoe split folder containing collection/ and processed/.}"

python -u "${SCRIPT_DIR}/train_latent_model.py" --gpu="${GPU_ID}" --data_root="${DATA_DIR}" \
  >> output_train_latent_$(date +%F_%H-%M-%S).log 2>&1
