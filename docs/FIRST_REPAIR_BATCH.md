# Retrieval correctness and representation audit (v0.5.1)

Baseline: `1915c5ffd411a902db0fa2f00bb194b94680ea5a` from
`ZZzzecho/task-guided-graph`. This batch adds no graph algorithm, UCB, action
redesign, or experiment-scale changes. Existing PCA/cache values are preserved.

## Query training semantics

`RetrievalQueryStream` owns a deterministic shuffled epoch, cursor, and per-query
presentation counts. The same instance is passed through warmup and every accepted
graph adaptation in both retrieval runners. The final partial epoch batch is kept.
Evaluation continues to use the original ordered context batches.

LoRA, SoftGraphTokenizer, and AdamW remain cumulative across accepted phases.
Changing a graph does **not** reset query weights or optimizer state. Therefore a
final gain alone cannot identify a graph effect separately from cumulative training.

Logs/phase results and final summaries report actual pool size after query limits,
unique queries seen, total presentations, successful optimizer steps, zero-based
epoch, cursor, seed, batch size, and training-pool fingerprint. At an epoch boundary,
cursor can equal pool size; the next batch starts the next shuffled epoch.

The patient runner supports `--task-seed` (default `runner.seed`); the bootstrap
retrieval stage uses `runner.seed`. They seed task initialization and shuffle.
No large experiment budgets/configuration files are changed.

`capture_retrieval_training_state` / `restore_retrieval_training_state` capture and
restore trainable query/graph-tokenizer parameters, model buffers, optimizer,
training stream, Python/NumPy/Torch/CUDA RNG state. The files are compatible with
`torch.load(..., weights_only=True)`. The patient runner writes
`checkpoints/retrieval_initial.pt` before warmup and `retrieval_final.pt`; the bootstrap
runner writes the same files at its output root. Initial trainable fingerprint is
logged. Restore with the same frozen candidate fingerprint, base model, tokenizer,
and context, e.g.:

```python
initial = torch.load("checkpoints/retrieval_initial.pt", weights_only=True)
restore_retrieval_training_state(
    model, optimizer, training_stream, initial,
    candidate_encoder_fingerprint=evaluator.candidate_metadata()["fingerprint"],
)
```

This prepares future fixed/learned-graph comparisons from one initialization with
identical query order, steps, presentations, and RNG. It is not a full Graph-RL
resume: graph environment, policy, and graph-search state remain separate.

## Fixed candidate encoder

Evaluator construction captures the **initial** candidate encoder. Frozen base
weights are shared, while initially trainable causal-LM parameters and buffers are
snapshotted on CPU. Candidate preparation temporarily activates these snapshots
in eval/no-grad mode, then restores current query parameters and every submodule's
mode, including on failure. No second full GLM backbone is allocated.

All train/reward/validation candidates and late unseen test candidates use the
same initial state. Test remains final-only: there is no eager test-pool loading.
Query-side LoRA and graph conditioning continue to train normally.

Candidate metadata contains a SHA256 of config and **all** initial parameter/buffer
bytes. Hashing occurs once at construction in bounded chunks, so large GLM models
incur one full weight scan. Cache entries carry that fingerprint; scoring rejects
mixed fingerprints. Tokenizer/vocabulary/settings/max-length signatures and text
hashes reject incompatible reuse of cached IDs. Frozen base mutation and parameter
replacement/trainability changes fail closed.

The time-shared snapshot requires sequential candidate/query execution with no
outstanding query autograd graph; both runners already follow this order. The
implementation requires materialized model parameters. Disk-offloaded/meta weights
and concurrent serving are outside this batch's verified execution path.

## Representation audit and diagnostics

The default remains token cosine attention in full Qwen hidden space, with softmax
over active tokens. PCA is fitted on the supplied static concept prototypes
(`P x D`, randomized SVD, default seed 17), **not document-conditioned states**.
The prototype mean is subtracted, projection maps `D` to `R`, and each concept
vector is L2-normalized on the representation axis before cache casting.

`project_full_states(normalize=False)` and optional `return_stages=True` expose
the intermediate representations without changing normal cache-building outputs.
Projection audit metadata/logs record fit source/count, dimensions, seed/solver,
centering, L2 axis, prototype and projection hashes, attention and rank axes.
New caches have additive `representation_pipeline` metadata; old caches stay readable.
The existing provided-matrix interface remains available. Training-representation
PCA is not implemented or selected by this batch.

Run from an installed checkout (`pip install -e ".[patient]"`), using training data
only and the same model/dtype/temperature/length as the original cache:

```bash
python scripts/diagnose_representation_collapse.py \
  --train data/patents_h04l/train.csv.gz \
  --prototypes data/patents_h04l/concept_prototypes.npz \
  --model /laijizheng/models/Qwen3-Embedding-0.6B \
  --local-files-only --dtype bfloat16 --cache-dtype float16 \
  --representation-dim 64 \
  --projection-file data/patents_h04l/cache/projection.npz \
  --max-samples 128 --batch-size 2 --seed 17 \
  --output outputs/collapse_128.json
```

Use an actual cache path for `--projection-file` to reuse its exact basis. Without
that flag, the same prototype-PCA procedure fits a diagnostic basis. Encoder ID
must match the prototype metadata. `--max-samples` is bounded to 512 (smaller
values are accepted for tests); seeded reservoir sampling records selected IDs,
source row indices and file hashes. Reports refuse to overwrite existing output.

The report includes:

| Stage | Measurement |
|---|---|
| A prototypes | Static prototype concept cosine and Gram spectrum |
| B attention | Entropy, entropy/log(active tokens), concept-pair cosine, Gram spectrum |
| C pre-PCA | Full document-conditioned H, within-document cosine and across-document covariance |
| D post-PCA/pre-L2 | Projected H before per-concept normalization |
| E post-L2 | The unquantized cache representation |
| E cache-quantized | Additional checkpoint after the configured FP16/FP32 cache cast |
| F rank-Gaussian | Production sample-axis rank transform of the quantized E checkpoint |

For each stage, the report provides an exact all-pair mean document cosine and
quantiles from up to 4096 fixed seeded concept pairs per document. Aggregate
Gram cosine/correlation statistics use all off-diagonal concept pairs. C-F also
report covariance after centering each `(representation, concept)` cell across
documents. A/B have no meaningful document-axis covariance (prototypes are static;
attention token positions vary). Zero-norm/constant vectors are reported as
undefined, not artificial perfect correlations.

The raw Gram is `sum_d H_d.T H_d / (N R)`, matching MNGM's `B=I` concept statistic
before ridge/regularization. Full spectra include lambda1/trace, entropy effective
rank `exp(-sum p log p)`, participation rank `1/sum p^2`, and numerical rank. The
two effective-rank definitions must not be conflated when comparing old results.

Full 1024D H and attention are streamed, retaining P-by-P sufficient statistics.
Only bounded post-L2 matrices are retained for the production rank transform over
the complete selected sample axis. No full 4096-document cache is rebuilt. The
extra quantization checkpoint distinguishes effects of cache precision from L2
and rank transform.

Instrumentation-only sanity command (does not measure real patent collapse):

```bash
python scripts/diagnose_representation_collapse.py --synthetic \
  --max-samples 128 --representation-dim 8 --output outputs/synthetic_128.json
```

## Scope and validation limits

Tests cover shuffled-tail coverage, cross-phase continuity, saved-state replay
including dropout and optimizer reset, late unseen candidates, query restoration
on failures, cache compatibility rejection, unchanged attention/PCA outputs,
streamed covariance vs a direct calculation, rank-one/zero-variance cases, exact
production rank-Gaussian use, and diagnostic CLI sampling. The patent-runner
integration test also runs real MNGM/GRPO with a tiny task backbone, verifies
on-disk coverage/checkpoints, and asserts that test pools are read only after
graph search and that unseen test vectors match the initial encoder.

Local verification: 85 tests passed; the 128-document synthetic A-F sanity check,
Python compilation, and Git whitespace checks passed. Tests use a CPU PyTorch
backbone surrogate for GLM; they do not imply full GLM/PEFT GPU validation.

Real GLM+PEFT GPU execution and 128-512 **real** training-document diagnostics
still require the corresponding model weights, prototype cache and corpus.
Synthetic statistics cannot localize the observed real-world rank-one collapse.
No final retrieval metric is recomputed in this batch, and prior mixed-state
test-index results are not retroactively repaired.
