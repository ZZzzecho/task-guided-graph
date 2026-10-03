# v0.6.2: more document excerpts into F, then full training

For each concept c, rank chunks from the **same document** by prototype/chunk
cosine, take up to six deduplicated original chunks and pack them in source order.
This increases evidence inside F; it does not add neighboring sentences or alter
the formula `H_d[:,c] = W @ F(concept_c, evidence_dc)`.

F retains frozen Qwen last-token pooling followed by FP32 L2. W retains the shared
prototype PCA components with no centering, subtraction, gate, prototype addition
or projected-column L2. Chunk limit stays 128 tokens; the **joint F** input limit
increases to 1024 tokens. Retrieval GLM still uses 384 tokens. Old CLI defaults
remain intact; this experiment explicitly selects the new parameters.

Six is an upper bound: short documents use fewer chunks, without repetition.
Exact tokenizer budgets retain the full concept/template and may shorten/omit
evidence. Audit records track actual counts/truncation. Adding chunks may increase
shared or irrelevant evidence; this is an experiment, not a proven collapse fix.
The supplied training text may contain only a title/abstract, not the patent's
complete specification.

## Server: one background job

Use the existing Python environment and server checkout:

```bash
cd /laijizheng/task_guided_graph_mvp_v0.3.0_2026-09-22
git pull --ff-only
python -m pip install -e . --no-deps
mkdir -p logs
nohup bash scripts/run_joint_evidence_h04l_full.sh \
  > logs/joint_evidence_full_n2048_k6_v062.log 2>&1 < /dev/null &
echo $!
```

This launches cache build → F controls → full training in sequential subprocesses,
releasing Qwen before GLM loads. Cache or training failure stops the job; auxiliary
control failure is recorded and training continues. The server command performs
real model inference; local synthetic tests only verify plumbing.

The new cache uses the first **2048 training documents**, matching the existing
full runner's graph budget, R=64 and all concepts (normally 800). This requires
**1,638,400 joint document/concept inputs**, plus chunks and auxiliary controls.
Graph solving remains CPU-intensive. Existing full training budget:

| Parameter | Budget |
|---|---:|
| Train / reward / validation / final-only test queries | 1024 / 128 / 256 / 1024 |
| Query LoRA warmup / adaptation per accepted phase | 50 / 100 optimizer steps |
| Phases / policy updates per phase / candidates per update | 8 / 3 / 8 |
| Query batch / candidate batch | 2 / 16 |
| Task seed / initial lambda | 11 / 0.8 |

Actual pool sizes, presentations and optimizer steps remain logged. LoRA and
optimizer accumulate with continuous shuffled query coverage. All candidate
embeddings, including final-only test, use the initial frozen snapshot. GRPO,
MNGM, weighted Glasso and graph actions retain their existing definitions.

The output directory must be new. Overrides are explicit, for example:

```bash
OUTPUT_DIR=outputs/joint_evidence_full_n4096_k6_v062 DOCUMENTS=4096 \
ENCODE_BATCH_SIZE=4 nohup bash scripts/run_joint_evidence_h04l_full.sh \
  --qwen-model /laijizheng/models/Qwen3-Embedding-0.6B \
  --task-model /laijizheng/models/GLM-4.7-Flash \
  > logs/joint_evidence_full_n4096_k6_v062.log 2>&1 < /dev/null &
```

4096 increases both cache and graph input, with higher memory/cost. For Qwen OOM,
use a fresh output directory and reduce `ENCODE_BATCH_SIZE`. Failed partial caches
are not resumable. A completed cache can restart training in a fresh directory:

```bash
OUTPUT_DIR=outputs/joint_evidence_full_n2048_k6_v062_retry \
nohup bash scripts/run_joint_evidence_h04l_full.sh \
  --reuse-cache outputs/joint_evidence_full_n2048_k6_v062/cache --skip-controls \
  > logs/joint_evidence_full_n2048_k6_v062_retry.log 2>&1 < /dev/null &
```

Reuse validates matching input hashes, model path, batch size, dimensions/evidence
parameters and complete-cache content hashes. There is no interrupted graph-RL
phase resume. Old 128/top-3 caches are rejected. `--plan-only` with a fresh output
directory validates data and writes the command manifest without inference.

## Progress and analysis files

```bash
tail -f outputs/joint_evidence_full_n2048_k6_v062/build.log
# Once training starts:
tail -f outputs/joint_evidence_full_n2048_k6_v062/train.log
```

All paths below are relative to the experiment directory:

- `experiment_manifest.json`: parameters, hashes, commands and stage status;
  `training_complete_controls_failed` distinguishes completed training with
  failed auxiliary controls.
- `training/summary.json`: initial/final reward and validation retrieval, final
  test retrieval, accepted phases, candidate snapshot and query coverage.
- `training/run_manifest.json`, `training/rl/*.json`,
  `training/graph_analysis/*.csv`: provenance, decisions and edge changes.
- `training/mngm/initial_covariance_diagnostics.json` and
  `final_covariance_diagnostics.json`: actual effective concept covariance spectra,
  correlations, representation precision and solver status. Initial B=I; final B
  belongs to the accepted final graph and may still be I if none was accepted.
  Covariance includes the existing solver eigenvalue floor. No extra refit occurs.
- `controls/report.json`, `run_manifest.json`: evidence diversity across **all
  cache documents**, plus F controls on a seeded 32-document/64-concept training
  subset. This auxiliary subset does not limit full training.
- `cache/metadata.json`, `build_signature.json`, `build_integrity.json`: encoder,
  projection, representation, actual counts and integrity metadata.

Controls compare concept-only, top-1, matched top-6 and approximately length-matched
evidence from another training document for the same concept. Donors are selected
independently per concept; donor rows and actual lengths are recorded. Donor
evidence is **not a verified negative**, so relation labels remain blank. Cosine
changes show sensitivity, not semantic correctness. No diagnostic baseline is
subtracted from production H.

Compact `cache/evidence.jsonl` stores source text once per document and original
spans/selection hashes per concept. `recover_evidence` in the controls script
reconstructs exact evidence. This audit can still be large. Matrix shards, audit
and LoRA weights can stay on the server for follow-up; send the smaller reports
above for initial analysis.

Check final validation/test retrieval first, then relate it to evidence sharing
and solver covariance. One full trial does not establish causal attribution.
An isolated top-3 comparison needs the same **1024-token F budget**, training
prefix/config/seeds/pools and a separate full run; the old 128/512-token diagnostic
is not that control. Accepted phases may change realized LoRA training budgets.
Fixed-budget graph attribution and document-resampling graph stability remain
separate follow-up experiments; these diagnostics do not claim either.
