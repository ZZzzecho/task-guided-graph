# Changelog

## v0.5.2 — 2026-10-01

- Add a separate bounded five-probe attention-design command: absolute logit
  levels vs token discrimination, temperature sweep, token common-direction
  geometry, fixed-attention raw/unit/centered value controls, and pooled chunk
  evidence on the same visible text range.
- Compare the production-native baseline with FP32 attention/aggregation on the
  same native states; trace every variant through pre-PCA, fixed PCA, L2, cache
  precision and production rank-Gaussian without changing defaults.
- Replay exact training rows from v0.5.1 reports, validate input/projection hashes,
  and export per-concept scores, evidence spans and blank human-label templates.
  Explicit labels can be evaluated later without reloading the model; unknown
  concepts are never fabricated as negatives and incomplete chunk comparisons
  are excluded from both methods' paired evaluation.
- Add a server wrapper and controlled analytic/CLI tests. Real Qwen experiments
  and human evidence labels are required before choosing a new formal design.

## v0.5.1 — 2026-10-01

- Keep a reproducibly shuffled query stream across warmup and accepted-graph
  adaptation, retaining partial batches and logging pool size, coverage,
  presentations, epoch/cursor, optimizer steps and seed.
- Preserve cumulative query LoRA/optimizer semantics and save initial/final
  trainable, optimizer, stream and RNG states for later equal-budget comparisons.
- Encode every candidate, including final-only unseen test candidates, with one
  initial encoder snapshot. Share frozen base weights without a second GLM,
  fingerprint actual state, and reject incompatible text/tokenization/cache reuse.
- Add bounded A-F diagnostics for prototypes, attention, full representations,
  PCA before L2, L2/cache precision, and production sample-axis rank-Gaussian.
  Report cosine/correlation, entropy, spectra and both effective-rank definitions.
- Record static-prototype PCA fit and per-concept L2 behavior explicitly, while
  preserving default projections and compatibility with existing caches.
- Add correctness/regression tests and a 128-sample synthetic diagnostic sanity
  check. Real-model GPU and real-patent collapse measurements remain unverified.
- Leave GRPO, MNGM/weighted Glasso, graph actions and large experiment configs
  unchanged.

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
