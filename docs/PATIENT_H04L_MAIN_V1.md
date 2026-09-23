# Patient H04L Main Pilot v1

This is the first non-smoke Patient-MNGM full-pipeline experiment.

## Budget

```text
Patient cache:           4096 patents, R=64, P=800
MNGM materialized:       2048 patents
task train queries:      1024
Graph-RL reward queries: 128
validation queries:      256
final-only test queries: 1024
Graph phases:            8
GRPO updates/phase:      3
candidates/update:       8
edge edits/candidate:    32
```

Maximum planned candidate graph evaluations:

```text
8 * 3 * 8 = 192
```

## Build cache

```bash
python scripts/build_patient_matrices.py \
  --train data/patents_h04l/train.csv.gz \
  --prototypes data/patents_h04l/concept_prototypes.npz \
  --model /laijizheng/models/Qwen3-Embedding-0.6B \
  --local-files-only \
  --output data/patents_h04l/cache_main_4096_r64 \
  --dtype bfloat16 \
  --cache-dtype float16 \
  --representation-dim 64 \
  --batch-size 16 \
  --shard-size 128 \
  --max-patients 4096 \
  --id-col patent_id
```

## Build final-test candidate pool

```bash
python scripts/build_patent_retrieval_candidates.py \
  --prepared-dir data/patents_h04l \
  --output data/patents_h04l/retrieval_test_candidates.jsonl \
  --split test \
  --negatives-per-query 63 \
  --max-queries 1024 \
  --seed 17
```

## Run

```bash
bash scripts/run_patient_h04l_main_v1.sh
```

## Outputs

```text
run_manifest.json
mngm/initial_summary.json
initial_graph.npz
final_graph.npz

rl/
  candidate_trajectory.jsonl
  phase_000.json
  ...
  phase_007.json

retrieval/
  initial_graph_queries.json
  initial_val_queries.json
  final_graph_queries.json
  final_val_queries.json
  final_test_queries.json

checkpoints/
  phase_000/
  ...

graph_analysis/
  edge_changes.csv
  top_changed_edges.csv
  degree_changes.csv

summary.json
```

Each candidate trajectory records reward decomposition, solver information, graph metrics, the 32 edited edges, separate selection/direction log probabilities and selected-edge state features such as task relevance and frontier score.

Each phase records parent validation, frozen proposal validation, post-adaptation validation, GRPO diagnostics, adaptation statistics and elapsed wall time.

The test candidate pool is evaluated only after graph search has completed; it is never passed to the policy, reward function or validation acceptance.