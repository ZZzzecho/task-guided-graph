#!/usr/bin/env bash
# All five diagnostic probes; does not build caches or train retrieval/graph models.
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-outputs/attention_design_128_v052}"
SAMPLE_REPORT="${SAMPLE_REPORT-outputs/collapse_real_128_v051.json}"
QWEN_MODEL="${QWEN_MODEL:-/laijizheng/models/Qwen3-Embedding-0.6B}"
PROJECTION_FILE="${PROJECTION_FILE:-data/patents_h04l/cache_main_4096_r64_pca/projection.npz}"
args=(
  --train data/patents_h04l/train.csv.gz
  --prototypes data/patents_h04l/concept_prototypes.npz
  --model "$QWEN_MODEL" --local-files-only --dtype bfloat16 --cache-dtype float16
  --max-length 512 --temperature 0.1 --temperatures 0.1 0.05 0.02
  --representation-dim 64 --projection-file "$PROJECTION_FILE"
  --max-samples 128 --batch-size 2 --seed 17 --review-documents 32
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$SAMPLE_REPORT" ]]; then
  args+=(--sample-manifest "$SAMPLE_REPORT")
fi
exec python -u scripts/diagnose_attention_design.py "${args[@]}" "$@"
