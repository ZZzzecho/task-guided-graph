# Server quickstart: Patient-MNGM + GLM-4.7-Flash

This file is intended for the first upload to the experiment server.

## 1. Install

```bash
cd task_guided_graph_mvp_v0.3.0_2026-09-22
python -m pip install -e ".[all]" --no-build-isolation
python -m pytest -q
```

The Patient encoder only needs `transformers>=4.51`; the GLM-4.7-Flash local task
path is pinned by this package to `transformers>=5.0.0` because GLM-4.7-Flash uses
the `glm4_moe_lite` Transformers architecture.

## 2. Prepare IMDb

Download and extract the Stanford Large Movie Review Dataset (`aclImdb`), then run:

```bash
prepare-imdb \
  --input /path/to/aclImdb \
  --output data/imdb_sentiment
```

This keeps the official 25k test set untouched. The official 25k training reviews
are split into train/graph/val. The prepared directory also contains `task.json`
with the task instruction and label strings `NEGATIVE/POSITIVE`.

## 3. Prepare concept vocabulary

Supported JSON:

```json
[
  {"concept_id": "c001", "concept_text": "heart failure"},
  {"concept_id": "c002", "concept_text": "chest pain"}
]
```

or a JSON mapping / CSV with columns `concept_id,concept_text`.

The same ordered vocabulary must be used by Bootstrap and Patient experiments.

## 4. Download Qwen3-Embedding-0.6B

```bash
python scripts/download_qwen3_embedding.py \
  --output /models/Qwen3-Embedding-0.6B
```

Then use `--model /models/Qwen3-Embedding-0.6B --local-files-only` below.

## 5. Build frozen concept prototypes

```bash
python scripts/build_concept_prototypes.py \
  --concepts data/concepts.json \
  --model /models/Qwen3-Embedding-0.6B \
  --local-files-only \
  --output data/patient_mngm/concept_prototypes.npz \
  --dtype bfloat16
```

## 6. Build a small Patient cache first

```bash
python scripts/build_patient_matrices.py \
  --train data/imdb_sentiment/train.csv.gz \
  --prototypes data/patient_mngm/concept_prototypes.npz \
  --model /models/Qwen3-Embedding-0.6B \
  --local-files-only \
  --output data/patient_mngm/imdb_cache_smoke \
  --dtype bfloat16 \
  --cache-dtype float16 \
  --batch-size 4 \
  --shard-size 16 \
  --max-patients 32
```

Inspect:

```bash
python scripts/inspect_patient_cache.py data/patient_mngm/imdb_cache_smoke
```

## 7. Smoke-test Patient-MNGM before loading the 30B task model

```bash
python scripts/run_patient_mngm.py \
  --cache data/patient_mngm/imdb_cache_smoke \
  --config configs/patient_mimic_glm_v0.3.json \
  --max-mngm-patients 32 \
  --output outputs/patient_mngm_smoke
```

Increase N gradually before deciding whether 1024-dimensional MNGM requires an
engineering change.

## 8. GLM-4.7-Flash

The default local task model ID is:

```text
zai-org/GLM-4.7-Flash
```

If you download it to `/models/GLM-4.7-Flash`, verify the local Transformers path:

```bash
python scripts/check_glm47_flash.py \
  --model /models/GLM-4.7-Flash \
  --local-files-only \
  --dtype bfloat16
```

### Important

The Graph-RL task path injects continuous graph tokens through `inputs_embeds`.
A standard OpenAI-compatible vLLM/SGLang HTTP chat endpoint does **not** expose that
interface. For the current method, keep the GLM weights locally accessible and let
`run_patient_graph_rl.py` load them with Transformers. vLLM can still be used for
unrelated inference checks, but not as a transparent replacement for this training
path.

## 9. First end-to-end smoke run

Use small subsets first:

```bash
python scripts/run_patient_graph_rl.py \
  --patient-cache data/patient_mngm/imdb_cache_smoke \
  --concepts data/concepts.json \
  --data-dir data/imdb_sentiment \
  --config configs/patient_mimic_glm_v0.3.json \
  --task-model /models/GLM-4.7-Flash \
  --task-local-files-only \
  --max-mngm-patients 32 \
  --max-train-examples 64 \
  --max-reward-examples 32 \
  --max-val-examples 32 \
  --warmup-steps 5 \
  --adapt-steps 5 \
  --policy-updates-per-phase 1 \
  --task-batch-size 1 \
  --output outputs/patient_glm_smoke
```

After the smoke path is stable, remove the `--max-*` limits and restore the planned
training budgets.

## 10. No test leakage

`run_patient_graph_rl.py` reads only:

```text
train.csv.gz
graph.csv.gz
val.csv.gz
```

It does not load `test.csv.gz`. Final test evaluation should be a separate explicit
post-search command/workflow.
