#!/usr/bin/env bash
set -euo pipefail

: "${PSPT_MODEL_PACKAGE:?Set PSPT_MODEL_PACKAGE to the extracted BRACS model package}"
: "${BRACS_PATCH_DIR:?Set BRACS_PATCH_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

DEVICE="${DEVICE:-cuda:0}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHT_DIR="${PSPT_MODEL_PACKAGE}/weights/BRACS"
ASSET_DIR="${PSPT_MODEL_PACKAGE}/assets/BRACS"
CONCH_BACKBONE="${PSPT_MODEL_PACKAGE}/backbones/CONCH/pytorch_model.bin"

cd "${CODE_DIR}"
for encoder in UNI CONCH; do
  for model_id in 000 001 002 003 004; do
    extra=()
    if [[ "${encoder}" == "CONCH" ]]; then
      extra+=(--conch-backbone-checkpoint "${CONCH_BACKBONE}")
    fi
    python -m pspt.evaluate \
      --checkpoint "${WEIGHT_DIR}/${encoder}_M256_model${model_id}.pt" \
      --split-csv "${ASSET_DIR}/bright3_label.csv" \
      --scatter-png-dir "${BRACS_PATCH_DIR}" \
      --pcps-scores "${ASSET_DIR}/${encoder}_pcps_scores_fold0.pt" \
      --output-dir "${OUTPUT_DIR}/${encoder}_model${model_id}" \
      --device "${DEVICE}" \
      "${extra[@]}"
  done
done
