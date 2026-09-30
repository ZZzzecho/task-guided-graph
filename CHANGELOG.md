# Changelog

## v0.5.0 — 2026-09-30

- Initialize each matrix-valued graph with `B_0 = I` and exactly one concept-axis
  weighted Graphical Lasso solve. Later penalty edits still run full alternating MNGM.
- Replace Patient-MNGM prefix truncation with an explicit 1024-to-R projection
  fitted by PCA on training concept prototypes. Save the projection with each
  patient cache and reject v0.4 prefix caches in the task runners.
- Preserve the option to use full-dimensional patient representations by omitting
  `--representation-dim`. Bootstrap-MNGM retains its existing fixed PCA basis.
- Add focused tests for the one-solve initialization and non-prefix projection.
- Repair a pre-existing syntax error in the patient runner's summary output.

## v0.3.0 — 2026-09-22

Patient-MNGM implementation based on `PATIENT_MNGM_GRAPH_RL_HANDOFF_v0.1_2026-09-22.md`.

### Added

- Frozen `Qwen3-Embedding-0.6B` Patient representation front-end.
- Official last-token pooled concept prototypes.
- Final-layer patient token-state extraction.
- Vectorized cosine + token-softmax construction of `H_d [1024,P]`.
- Sharded float16/float32 Patient matrix cache and lazy/memmap reader.
- Patient-MNGM smoke/scale runner with explicit `--max-mngm-patients`.
- Local GLM-4.7-Flash Transformers loader.
- GLM-4.7-Flash architecture-aware task LoRA target detection.
- Separate task-LLM-space concept prototypes for downstream sample activation.
- End-to-end `run_patient_graph_rl.py` entrypoint.
- Server deployment quickstart.

### Changed

- Package version bumped to 0.3.0.
- LLM extra now requires Transformers 5+ for GLM-4.7-Flash architecture support.
- `GraphConditionedCausalLM` moves `SoftGraphTokenizer` to the task input-embedding device for sharded models.
- Accepted-graph task adaptation streams batches instead of materializing all train tensors on GPU.

### Preserved

- v0.2 active+frontier candidate pool.
- 32-edge hierarchical batch GRPO.
- Full weighted MNGM re-solve after penalty edits.
- Independent validation acceptance.
- Bootstrap branch and existing estimator semantics.

### Verification

```text
60 passed
```
