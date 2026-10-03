# v0.6.0: direct concept + evidence representation

### v0.6.1 BF16 fix and restarting a failed run

v0.6.0's legacy baseline could pass a BF16 tensor directly to NumPy, causing
`TypeError: Got unsupported ScalarType BFloat16`. v0.6.1 converts to FP32 before
NumPy, then applies the requested cache dtype. The representation formula and
native model/attention dtype are unchanged. The `torch_dtype` deprecation message
is a separate warning and did not cause this exception.

The diagnostic does not resume partial runs. Keep the failed directory for
inspection and rerun with a fresh output path:

```bash
git pull --ff-only origin main
python -m pip install -e . --no-deps --no-build-isolation
mkdir -p logs
OUTPUT_DIR=outputs/joint_evidence_128_v061 \
  nohup bash scripts/run_joint_evidence_h04l.sh \
  > logs/joint_evidence_128_v061.log 2>&1 < /dev/null &
echo $!
```

## Agreed computation

```text
complete supplied training document text
  -> sentence/newline chunks (long sentences split to token budget)
  -> independent concept/chunk cosine ranking
  -> at most 3 original evidence chunks, packed in source order
  -> frozen Qwen F(concept description, evidence text)
  -> one shared linear projection W
  -> H_d [R,P] -> existing rank-Gaussian -> MNGM -> GRPO
```

The new graph input is `H_d[:,c] = W @ F(c, evidence_dc)`, not delta.
There is no concept/empty/evidence baseline subtraction, no prototype addition,
no score gate, and no L2 normalization of projected columns. All concepts and
documents share the same frozen encoder, template, tokenizer and projection.
The retrieval query LoRA and frozen candidate snapshot remain independent.

## Functions and defaults

`graph_mvp/joint_evidence_repr.py` implements:

1. `split_document(text, tokenizer, max_chunk_tokens=128)` returns sentence/newline
   heuristic chunks with original `start/end/index`. Long sentences use fitting
   source prefixes. The token bound includes special tokens. Abbreviations can
   be split by this heuristic; it is not a linguistic sentence parser.
2. Chunk/prototype cosine uses independently pooled, normalized embeddings in
   the full encoder space. Each concept ranks its own chunks; concepts can share
   evidence and do not compete through a concept-axis softmax.
3. `select_and_pack_evidence(..., top_k=3, max_input_tokens=512)` removes duplicate
   whitespace-normalized texts, breaks score ties by source order, preserves the
   full concept, and checks the complete template's actual tokenizer length.
   Over-budget excerpts can be shortened; shortened fragments under 8 content
   tokens are omitted. Short complete excerpts remain allowed. Original spans,
   omitted-for-budget counts, scores and truncation flags are retained.
4. `JointEvidenceMatrixBuilder.encode_document(text)` encodes chunks and then
   each concept/excerpt pair in bounded batches. Qwen runs in eval/inference mode
   with frozen weights. Pooling retains the native model dtype and is followed
   by explicit FP32 L2 normalization. The output is a fixed D-dimensional vector.
5. `project_full_states()` applies only `W @ vector`. For prototype PCA the
   components are reused, but its fitted mean is not subtracted. A provided
   projection's mean is likewise ignored and its hash recorded; the saved
   effective projection has zero mean. The method rejects requested output L2.
6. `encode(texts)` returns `[B,R,P]` matrices and per-document evidence audits,
   compatible with `build_patient_cache_from_frame()`.

Template (no generated explanations):

```text
Concept:
{complete concept description}

Patent excerpts:
{selected original excerpts}
```

The builder fails on empty documents, overlong concept/template, nonfinite or
zero pooled embeddings, mismatched dimensions and any would-be silent truncation.
Matching scores are ranking signals, not probabilities or relevance labels.
Top-k can select unrelated excerpts for unsupported concepts; review this instead
of claiming that selection proves presence. Input length can also affect F.

## Server: first run the 128-document comparison

```bash
cd /laijizheng/task_guided_graph_mvp_v0.3.0_2026-09-22
git pull --ff-only origin main
python -m pip install -e . --no-deps --no-build-isolation
mkdir -p logs
nohup bash scripts/run_joint_evidence_h04l.sh \
  > logs/joint_evidence_128_v060.log 2>&1 < /dev/null &
echo $!
```

Default wrapper: seed 17, 128 documents, all prototype concepts, 512 input tokens,
128 tokens per chunk, top-k 3, Qwen batches of 2, R=64, float32 cache. It replays
`outputs/attention_design_128_v052/report.json` and reuses the old projection file.
Changed train/prototype/projection hashes fail closed. `SAMPLE_REPORT` can point
to another diagnostic report. To use a fresh seeded reservoir deliberately:

```bash
SAMPLE_REPORT="" OUTPUT_DIR=outputs/joint_evidence_128_v060_fresh \
  nohup bash scripts/run_joint_evidence_h04l.sh \
  > logs/joint_evidence_128_v060_fresh.log 2>&1 < /dev/null &
```

These runs encode 128 x 800 = 102,400 joint inputs, plus all document chunks.
They are substantially more expensive than encoding each document once. A
single Qwen instance is reused, batched; existing full caches are not rebuilt.
The wrapper does not train retrieval or run a large graph experiment.

Outputs under `outputs/joint_evidence_128_v060/`:

- `run_manifest.json`: running/failed/complete, version, Git/input hashes, args
  and SHA256 hashes for completed outputs.
- `report.json`: full joint embeddings, projected/cache vectors, post-rank
  cosine/correlation, spectra, lambda1/trace and effective ranks; input lengths,
  encoding counts and the old production attention comparison.
- `cache/`: standard sharded graph matrices, concept IDs, projection and explicit
  pipeline metadata. Incomplete new caches are rejected by `PatientMatrixDataset`.
- `evidence.jsonl`: every sampled document/concept's selected original chunks,
  scores, full packed input, token lengths, truncation and projection norm.
- `scores.jsonl`: compact per-pair selection/length statistics.
- `evidence_review.json`, `annotations_template.csv`: high/low-score review
  suggestions for up to 32 documents, with blank relations. Suggestions are not
  ground truth. This command does not calculate human-label AUC automatically.

Both methods use the same source rows and projection components, but their
evidence scope is explicitly different: new selection sees complete supplied
text; old production attention sees its max-length prefix. The baseline also
uses its original affine PCA + output L2. This is a pipeline comparison, not an
isolated pooling ablation. Post-rank statistics are essential; increased raw rank
does not establish useful semantics or the MNGM separable-covariance assumption.
The static-prototype basis may discard joint-semantic directions; compare full
and projected statistics before choosing any replacement projection.

## Feed the trial cache into the unchanged graph solver

After checking `run_manifest.json` is complete, a separate initial-graph check:

```bash
nohup python -u scripts/run_patient_mngm.py \
  --cache outputs/joint_evidence_128_v060/cache \
  --config configs/patient_h04l_main_v1.json \
  --max-mngm-patients 128 \
  --output outputs/joint_evidence_128_v060_mngm \
  > logs/joint_evidence_128_v060_mngm.log 2>&1 < /dev/null &
```

An 800-concept solve can still be expensive and convergence is not assured.
The existing `run_patent_graph_rl.py --cache <new-cache>` accepts these matrices
without changing MNGM, GRPO or retrieval semantics. Preserve matched sample sets,
query initialization and training budgets for later comparisons.

## Build a selected-size cache explicitly

The existing builder remains `token_attention` by default. Opt in:

```bash
python -u scripts/build_patient_matrices.py \
  --representation-mode joint_evidence \
  --train data/patents_h04l/train.csv.gz \
  --prototypes data/patents_h04l/concept_prototypes.npz \
  --model /laijizheng/models/Qwen3-Embedding-0.6B \
  --local-files-only --dtype bfloat16 --cache-dtype float32 \
  --max-length 512 --chunk-max-tokens 128 --evidence-top-k 3 \
  --encode-batch-size 2 --batch-size 1 --shard-size 16 \
  --representation-dim 64 \
  --projection-file data/patents_h04l/cache_main_4096_r64_pca/projection.npz \
  --max-patients 128 \
  --output data/patents_h04l/cache_joint_prefix128_v060
```

This command uses a deterministic **prefix**, unlike the diagnostic's replay or
seeded reservoir. Use the diagnostic cache for comparisons on the previous 128.
Every new cache includes an evidence audit (potentially large), effective
projection fingerprint, template/concept-text/prototype hashes and encoding
counts. Do not mix legacy and joint shards into one cache.

## Local instrumentation

```bash
python -m scripts.diagnose_joint_evidence --synthetic \
  --max-samples 128 --representation-dim 8 \
  --output-dir outputs/joint_synthetic_128_v060
python -m pytest -q
```

Synthetic hash vectors test plumbing and reproducibility, not patent semantics.
Real Qwen inference, human review and fair downstream comparisons are required
before claiming that collapse or retrieval quality has improved.
