#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo_linear_trial_$(date +%Y%m%d_%H%M%S)}"
exec "$PYTHON" -u -m scripts.run_patent_graph_rl \
  --cache outputs/patent_longtext_graph_n2048_b4096_k6_v063_server_20261008_064102/cache \
  --initial-graph-fit outputs/patent_longtext_graph_fit_n2048_iter10000_20261009_101910 \
  --config configs/patient_h04l_main_v1.json --solver-max-iter 10000 \
  --policy grpo-linear --linear-prototypes data/patents_h04l/concept_prototypes.npz \
  --concepts data/patents_h04l/concepts.json --data-dir data/patents_h04l \
  --train-pools data/patents_h04l/retrieval_train_candidates.jsonl \
  --graph-pools data/patents_h04l/retrieval_graph_candidates.jsonl \
  --val-pools data/patents_h04l/retrieval_val_candidates.jsonl \
  --task-model /laijizheng/models/GLM-4.7-Flash --task-local-files-only --device-map cuda:0 \
  --max-mngm-documents 2048 --max-train-queries 1024 \
  --max-reward-queries 128 --max-val-queries 256 \
  --task-seed 11 --warmup-steps 50 --adapt-steps 0 \
  --phases 2 --policy-updates-per-phase 2 --num-candidates 4 \
  --task-batch-size 2 --candidate-batch-size 16 --task-max-length 384 \
  --output "$OUTPUT_DIR"
