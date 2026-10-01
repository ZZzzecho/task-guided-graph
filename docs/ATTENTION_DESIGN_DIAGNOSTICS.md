# Attention design diagnostics (v0.5.2)

The server v0.5.1 diagnostic found pre-PCA concept cosine 0.996212 and
Gram lambda1/trace 99.6113% on 128 training patents. This command investigates
why; it does not replace attention, PCA, cache values or graph/retrieval training.
Only training texts are loaded. No UCB or graph-action redesign is introduced.

## Run all five probes on the server

Use the environment that ran v0.5.1. From the repository root:

```bash
git pull --ff-only origin main
python -m pip install -e . --no-deps --no-build-isolation
python -c "import graph_mvp; print(graph_mvp.__version__)"
mkdir -p logs
nohup bash scripts/run_attention_design_h04l.sh \
  > logs/attention_design_128_v052.log 2>&1 < /dev/null &
echo $!
```

The wrapper uses the existing `/laijizheng/models/Qwen3-Embedding-0.6B`, training
CSV/prototypes, `cache_main_4096_r64_pca/projection.npz`, BF16 encoder, 512-token
limit, temperature .1, and FP16 cache simulation. It replays the exact 128 rows
from `outputs/collapse_real_128_v051.json`; source/prototype/projection hashes,
baseline temperature, max length, seed and native dtype must match that report.
It does not rebuild 4096 matrices or run GLM. Extra chunk encoding uses the same
frozen Qwen encoder. Do not launch concurrent large GPU jobs.

Watch `tail -f logs/attention_design_128_v052.log`. A completed run has
`run_manifest.json` status `complete`; partial files with `running`/`failed` are
not successful results. Every output directory must be new. For another run:

```bash
OUTPUT_DIR=outputs/attention_design_repeat_v052 \
  bash scripts/run_attention_design_h04l.sh
```

Set `SAMPLE_REPORT=''` to use seeded reservoir sampling instead of exact replay.
Override `QWEN_MODEL` and `PROJECTION_FILE` only with corresponding compatible
inputs. For generic datasets call `scripts/diagnose_attention_design.py --help`.
The library requires torch; real execution also needs the existing Qwen dependencies
and a fast tokenizer with source offsets. Data/schema problems fail rather than
silently change the visible evidence. Suggested sample count is 128, capped at 512.

## Controlled measurements

| Hypothesis | Probe | Evidence |
|---|---|---|
| Scores low vs uninformative | Per-concept token cosine mean/std/max/median; centered logit patterns | Absolute level and max-minus-median are separate; softmax ignores additive shifts |
| Temperature too high | .1/.05/.02 on the exact same FP32 logits | Entropy, maximum weight, top-5 mass, attention similarity and output spectrum |
| Shared token direction | Raw token geometry, uniform pooling, centered value control | Mean-vector energy fraction, sampled token spectrum, distance to uniform, residual/raw norm |
| Raw value norms dominate | Same FP32 alpha, original vs unit-norm values | Output differences, token norm distributions, alpha*norm contribution mass and top contributors |
| Token/document grain mismatch; absent concepts | Independently pooled sentence/chunk vectors plus manual relevant/hard-negative/unrelated labels | Matching evidence spans and within-document score AUC, not rank alone |

Eight variants are generated with default temperatures:

- `production_native`: unchanged production attention/PCA/L2 on native encoder states.
- `fp32_raw_values`: scoring and aggregation in FP32 on those same native states.
- `unit_values`: fixed FP32 baseline attention; normalize each value token.
- `centered_raw_values`: fixed FP32 baseline attention; subtract the document's mean token vector.
- `uniform_raw_values`: uniform weights over active tokens, repeated for all concepts.
- Two `temperature_*_raw_values` variants: change temperature only.
- `chunk_pooled`: normalize independently encoded last-token pooled chunks, then match/aggregate.

Each variant reports pre-PCA, post-PCA/pre-L2, post-L2, cache cast and the existing
sample-axis rank-Gaussian statistics: concept cosine/correlation, full spectrum,
lambda1/trace, entropy and participation effective rank, zero vectors/variances
and pre-PCA vector norms. Full high-dimensional states are streamed. P-by-P FP64
sufficient statistics are retained, with matrix products on the encoder device.
Only bounded post-L2 matrices are kept per variant for rank-Gaussian. CPU spectral
summaries and eight low-dimensional variants take more time/RAM than the A-F tool;
this is an instrumented small experiment rather than full cache generation.

Token-pair mean cosine uses all active tokens. Token spectra use at most 128 seeded
tokens per document (configurable up to 256); they are not the full token spectrum
when sampling occurs. The mean-vector energy fraction is
`||mean_l z_l||^2 / mean_l ||z_l||^2`. Contribution alpha*norm is a magnitude proxy,
not an additive attribution of vector energy when token directions cancel.

## Text coverage and semantic comparisons

Token offsets are checked against the actual encoder mask, excluding padding.
Chunks are split only from the token-visible prefix, retaining original character
spans; long sentences split at bounded character/space windows. This heuristic
is not a patent-aware linguistic parser. At most 64 chunks/document are encoded
by default. Unit limits and any encoder truncation are reported. Incomplete chunk
coverage is excluded from both methods' paired human-label score metrics.

Chunk number, context and value normalization differ from the token baseline.
Compare semantic evidence before interpreting spectrum changes. The FP32 control
does not re-encode the model in FP32 or recover precision already lost during
BF16 encoding. Centering can amplify tiny noise; check residual/raw amplitude.
The same prototype-PCA mean is applied even to centered-value controls, so later
projection may reintroduce a shared constant. Pre-PCA results remain the primary
diagnostic. Higher effective rank alone is not a success criterion.

## Outputs and human evidence

Under `outputs/attention_design_128_v052/`:

- `report.json`: all five probe statistics, variant stages, coverage and provenance.
- `scores.jsonl`: all document/concept matching scores, entropy and distance to uniform.
- `evidence_review.json`: up to 32 review documents, concept descriptions, visible
  texts, top cosine tokens/chunks and norm-weighted token contributors.
- `annotations_template.csv`: candidate review pairs with blank `relation` fields.
- `run_manifest.json`: completion state, arguments and hashes of result files.

Default review pairs are high/low score suggestions, not positive/negative labels.
Use the evidence and text to fill `relation` with `relevant`, `hard_negative`,
`unrelated`, or `unknown` (blank also means unknown). Unlisted/unlabeled concepts
are not negatives. CPC labels may help select pairs but lack of a CPC label is not
proof of irrelevance. Add manually selected hard concepts in the same CSV schema;
the scores file contains every concept for the selected documents.

Evaluate completed human labels without another Qwen run:

```bash
python scripts/diagnose_attention_design.py \
  --evaluate-only outputs/attention_design_128_v052/scores.jsonl \
  --annotations outputs/attention_design_128_v052/annotations_template.csv \
  --output-dir outputs/attention_labels_v052
```

This emits `annotation_metrics.json`: explicit-label counts, exclusions, positive/
negative score distributions, pooled ROC-AUC, within-document AUC and hard-negative
AUC. Per-relation diagnostics also compare token entropy, raw output norm, distance
to uniform and residual/raw amplitude for relevant vs absent concepts. Both methods
use identical complete labeled pairs. Ties receive half credit;
AUC is undefined without both label classes. Pooled AUC can be affected by document
score calibration; prefer paired within-document comparisons. Blank labels leave
the status `pending_human_labels`; automatic geometry does not verify semantics.

Alternatively pass `--annotations labels.csv` to the full run to export evidence
for chosen pairs and evaluate labels at the same time. Review is bounded to 32
documents and 256 explicit pairs. Record source judgments in `notes`.

For analysis, send this entire output directory and the nohup log, without model
weights/cache shards. Both automatic and human evaluation outputs should be kept.

## Minimal validation

```bash
python -m pytest tests/test_attention_design_diagnostics.py \
  tests/test_representation_diagnostics.py tests/test_patient_repr.py -q
python scripts/diagnose_attention_design.py --synthetic --max-samples 128 \
  --representation-dim 8 --output-dir outputs/attention_synthetic_128_v052
```

Synthetic vectors test instrumentation only and provide no evidence about Qwen
or patent concepts. Tests check analytic score-shift invariance, streamed FP64
statistics, common-direction and value-norm controls, source spans, masking,
exact sample replay, explicit-label semantics and model-free saved-score evaluation.

Local v0.5.2 validation: 96 tests passed, including a real-input-path CLI integration
with a deterministic tiny encoder and left padding. The 128-document synthetic
eight-variant sanity check, Python compilation and Git whitespace checks passed.
Real Qwen execution of these new probes and human judgments remain unverified.
