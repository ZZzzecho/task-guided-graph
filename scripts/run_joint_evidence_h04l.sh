#!/usr/bin/env bash
# Bounded new-representation trial; saves a separate graph-compatible cache.
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-outputs/joint_evidence_128_v060}"
SAMPLE_REPORT="${SAMPLE_REPORT-outputs/attention_design_128_v052/report.json}"
QWEN_MODEL="${QWEN_MODEL:-/laijizheng/models/Qwen3-Embedding-0.6B}"
PROJECTION_FILE="${PROJECTION_FILE:-data/patents_h04l/cache_main_4096_r64_pca/projection.npz}"
args=(
  --train data/patents_h04l/train.csv.gz
  --prototypes data/patents_h04l/concept_prototypes.npz
  --model "$QWEN_MODEL" --local-files-only --dtype bfloat16 --cache-dtype float32
  --max-length 512 --chunk-max-tokens 128 --evidence-top-k 3 --encode-batch-size 2
  --representation-dim 64 --projection-file "$PROJECTION_FILE"
  --max-samples 128 --seed 17 --review-documents 32
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$SAMPLE_REPORT" ]]; then
  args+=(--sample-manifest "$SAMPLE_REPORT")
fi
exec python -u scripts/diagnose_joint_evidence.py "${args[@]}" "$@"
