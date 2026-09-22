# v0.3 Patient-MNGM implementation map

Version: 2026-09-22

This release adds the frozen Patient-MNGM statistical front-end from
`PATIENT_MNGM_GRAPH_RL_HANDOFF_v0.1_2026-09-22.md` while retaining the v0.2
Bootstrap Graph-RL backend.

## 1. Statistical front-end

`graph_mvp/patient_repr.py` implements:

- `Qwen3EmbeddingEncoder`: frozen local `Qwen/Qwen3-Embedding-0.6B`;
- official Qwen3 last-token pooled concept embeddings;
- final-layer contextual patient token states;
- vectorized cosine + token-softmax attention;
- `H_d [1024,P]` patient-specific concept matrices;
- sharded `.npy` cache and lazy/memmap reader.

The representation formula is fixed as

$$
s_{dlj}=\cos(z_{dl},p_j),\qquad
\alpha_{dlj}=\operatorname{softmax}_l(s_{dlj}/\tau),
$$

$$
h_{dj}=\sum_l\alpha_{dlj}z_{dl},\qquad
H_d=[h_{d1},\ldots,h_{dP}]\in\mathbb R^{1024\times P}.
$$

Only `D_train` may be used to build this cache. The cache is frozen throughout
Graph-RL.

## 2. Two prototype spaces are intentional

Patient-MNGM and the downstream task model use two different representation spaces.
They must not be mixed.

### Patient-MNGM prototypes

`Qwen3-Embedding-0.6B` official pooled embeddings. These are used only to build
`H_patient` and therefore only affect statistical graph estimation.

### Task activation prototypes

`graph_mvp/task_prototypes.py` obtains mean sub-token vectors from the downstream
causal LM's own `get_input_embeddings()` module. For GLM-4.7-Flash these vectors
are in GLM's input-embedding space and are used only by the existing
`SoftGraphTokenizer` sample-activation path.

No Qwen-Embedding ↔ GLM alignment layer is required because the two spaces serve
different stages and are never compared directly.

## 3. GLM-4.7-Flash support

`graph_mvp/task_model.py` adds a local Transformers loader. Default model ID:

```text
zai-org/GLM-4.7-Flash
```

The graph-token path requires direct access to `inputs_embeds`; therefore a normal
OpenAI-compatible vLLM/SGLang HTTP endpoint is not a drop-in replacement for this
training/evaluation path. The model weights may be downloaded/managed by vLLM or
other tooling, but `run_patient_graph_rl.py` loads a local Transformers causal-LM
object.

For GLM-4.7-Flash, task LoRA defaults to the attention modules that exist in the
`glm4_moe_lite` architecture:

```text
q_a_proj
q_b_proj
kv_a_proj_with_mqa
kv_b_proj
o_proj
```

The helper checks that these modules actually exist before attaching PEFT LoRA.

The official model config identifies the architecture as `Glm4MoeLiteForCausalLM`
and uses hidden size 2048. The code does not hard-code 2048: SoftGraphTokenizer
output dimension is derived from `model.get_input_embeddings().weight.shape[-1]`.

## 4. Multi-GPU entry device

`GraphConditionedCausalLM` now places the small `SoftGraphTokenizer` on the task
LM input-embedding device. This avoids CPU/GPU device mismatch when the large task
LM is loaded with `device_map="auto"`.

## 5. Streaming task adaptation

v0.2 converted all task batches to a Python list before accepted-graph adaptation,
which would materialize the complete MIMIC training split on GPU. v0.3 iterates
batches lazily and restarts the iterator only when the configured optimizer-step
budget exceeds one data pass.

## 6. Current MNGM scale boundary

Representation generation is out-of-core, but the existing `MNGMEstimator` remains
an in-memory solver. `PatientMatrixDataset.materialize(max_samples=...)` makes this
boundary explicit.

Start with deterministic scale tests:

```text
N = 16/32/64 → 256 → 1k → ...
```

If `[N,1024,P]` or the `1024×1024` representation precision becomes the actual
bottleneck, dimension reduction/out-of-core MNGM is a later engineering change.
It is not silently enabled in v0.3.

## 7. Real pipeline entry points

```text
scripts/download_qwen3_embedding.py
scripts/build_concept_prototypes.py
scripts/build_patient_matrices.py
scripts/inspect_patient_cache.py
scripts/run_patient_mngm.py
scripts/check_glm47_flash.py
scripts/run_patient_graph_rl.py
```

The full Patient run uses the existing v0.2 components after MNGM:

```text
Patient cache
→ MNGMEstimator(patient_concept_matrix)
→ active + frontier pool
→ 32-edge GRPO action group
→ full MNGM re-solve
→ frozen graph-conditioned GLM reward
→ validation acceptance
→ accepted-only GLM LoRA + SoftGraphTokenizer adaptation
```

## 8. Validation

The package test suite currently reports:

```text
60 passed
```

Tests cover the old v0.2 behavior plus Patient matrix construction, attention
normalization/padding masks, prototype cache, sharded patient cache, Patient-MNGM
smoke solving, task-model-space prototype extraction, and GLM-4.7-Flash LoRA target
detection.
