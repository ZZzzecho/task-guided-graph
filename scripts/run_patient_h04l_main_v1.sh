#!/usr/bin/env bash
set -euo pipefail

python scripts/run_patent_graph_rl.py \
  --cache data/patents_h04l/cache_main_4096_r64 \
  --concepts data/patents_h04l/concepts.json \
  --data-dir data/patents_h04l \
  --train-pools data/patents_h04l/retrieval_train_candidates.jsonl \
  --graph-pools data/patents_h04l/retrieval_graph_candidates.jsonl \
  --val-pools data/patents_h04l/retrieval_val_candidates.jsonl \
  --test-pools data/patents_h04l/retrieval_test_candidates.jsonl \
  --config configs/patient_h04l_main_v1.json \
  --task-model /laijizheng/models/GLM-4.7-Flash \
  --task-local-files-only \
  --max-mngm-documents 2048 \
  --initial-lambda 0.8 \
  --max-train-queries 1024 \
  --max-reward-queries 128 \
  --max-val-queries 256 \
  --max-test-queries 1024 \
  --warmup-steps 50 \
  --adapt-steps 30 \
  --phases 8 \
  --policy-updates-per-phase 3 \
  --num-candidates 8 \
  --task-batch-size 2 \
  --candidate-batch-size 16 \
  --task-max-length 384 \
  --output outputs/patient_h04l_main_n2048_r64_v1