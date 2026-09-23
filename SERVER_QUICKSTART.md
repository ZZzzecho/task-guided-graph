# Server Quickstart — v0.4.0 H04L Mainline

> Current as of 2026-09-23. The earlier MIMIC/IMDb quickstart is obsolete.

## 1. Update and install

```bash
cd /laijizheng/task_guided_graph_mvp_v0.3.0_2026-09-22
git pull
python -m pip install -e ".[all]" --no-build-isolation
```

## 2. Current experiment data

Prepared H04L data should live under:

```text
data/patents_h04l/
```

with the 800-concept vocabulary, train/graph/val/test splits and fixed retrieval candidate pools.

See `docs/PATENTS_H04L_DATA.md` for preparation commands.

## 3. Patient-MNGM path

Build 64D patent concept matrices:

```bash
python scripts/build_patient_matrices.py \
  --train data/patents_h04l/train.csv.gz \
  --prototypes data/patents_h04l/concept_prototypes.npz \
  --model /laijizheng/models/Qwen3-Embedding-0.6B \
  --local-files-only \
  --output data/patents_h04l/cache_smoke_32_r64 \
  --dtype bfloat16 \
  --cache-dtype float16 \
  --representation-dim 64 \
  --batch-size 4 \
  --shard-size 16 \
  --max-patients 32 \
  --id-col patent_id
```

Run MNGM:

```bash
python scripts/run_patient_mngm.py \
  --cache data/patents_h04l/cache_smoke_32_r64 \
  --config configs/patent_h04l_glm_v0.4.json \
  --max-mngm-patients 32 \
  --output outputs/patent_h04l_mngm_smoke32_r64
```

End-to-end retrieval Graph-RL:

```bash
python scripts/run_patent_graph_rl.py \
  --cache data/patents_h04l/cache_smoke_32_r64 \
  --concepts data/patents_h04l/concepts.json \
  --data-dir data/patents_h04l \
  --train-pools data/patents_h04l/retrieval_train_candidates.jsonl \
  --graph-pools data/patents_h04l/retrieval_graph_candidates.jsonl \
  --val-pools data/patents_h04l/retrieval_val_candidates.jsonl \
  --config configs/patent_h04l_glm_v0.4.json \
  --task-model /laijizheng/models/GLM-4.7-Flash \
  --task-local-files-only \
  --max-mngm-documents 32 \
  --max-train-queries 8 \
  --max-reward-queries 4 \
  --max-val-queries 4 \
  --warmup-steps 2 \
  --adapt-steps 2 \
  --phases 1 \
  --num-candidates 2 \
  --policy-updates-per-phase 1 \
  --output outputs/patent_h04l_graph_rl_smoke
```

## 4. Bootstrap-MNGM path

Bootstrap v0.4 is epoch-based, not one-step-per-version.

Each version resets to the same initial embedding-LoRA parameters, trains independently for complete epochs, then snapshots one concept matrix.

Cost estimate:

```text
B * E * ceil(N_docs / batch_size) optimizer steps
```

Example development run:

```bash
python scripts/run_bootstrap_patent_graph_rl.py \
  --concepts data/patents_h04l/concepts.json \
  --data-dir data/patents_h04l \
  --train-pools data/patents_h04l/retrieval_train_candidates.jsonl \
  --graph-pools data/patents_h04l/retrieval_graph_candidates.jsonl \
  --val-pools data/patents_h04l/retrieval_val_candidates.jsonl \
  --config configs/bootstrap_patent_glm_v0.4.json \
  --bootstrap-model /laijizheng/models/GLM-4.7-Flash \
  --bootstrap-local-files-only \
  --bootstrap-versions 16 \
  --bootstrap-docs 8 \
  --bootstrap-epochs-per-version 3 \
  --bootstrap-batch-size 1 \
  --bootstrap-representation-dim 64 \
  --task-model /laijizheng/models/GLM-4.7-Flash \
  --task-local-files-only \
  --max-train-queries 8 \
  --max-reward-queries 4 \
  --max-val-queries 4 \
  --warmup-steps 2 \
  --adapt-steps 2 \
  --phases 1 \
  --num-candidates 2 \
  --policy-updates-per-phase 1 \
  --output outputs/bootstrap_patent_graph_rl_dev
```

Once a bundle exists, reuse it instead of retraining:

```bash
python scripts/run_bootstrap_patent_graph_rl.py \
  --bootstrap-bundle outputs/bootstrap_patent_graph_rl_dev/bootstrap_bundle.npz \
  ...
```

## 5. Current docs

```text
README.md
docs/CURRENT_RESEARCH_HANDOFF_2026-09-23.md
docs/IMPLEMENTATION_v0.4.md
docs/PATENTS_H04L_DATA.md
```

Old MIMIC-related documents are retained only as historical design records.