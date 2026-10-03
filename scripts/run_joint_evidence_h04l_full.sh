#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -u -m scripts.run_joint_evidence_full \
  --output "${OUTPUT_DIR:-outputs/joint_evidence_full_n2048_k6_v062}" \
  --documents "${DOCUMENTS:-2048}" \
  --evidence-top-k "${EVIDENCE_TOP_K:-6}" \
  --max-length "${F_MAX_LENGTH:-1024}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-8}" "$@"
