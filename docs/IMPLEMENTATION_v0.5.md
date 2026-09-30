# v0.5 graph initialization and representation projection

## Initial graph

For matrix observations `H_s` with shape `[R,P]`, the initial representation
precision is fixed to `B_0 = I_R`. The estimator forms the concept covariance
`S_A(B_0)` using the configured nonparanormal transform, then calls the existing
weighted Graphical Lasso solver once to obtain `A_0`. This is the graph used for
task warmup and the first Graph Phase. No representation-axis solve is run during
initialization. A changed edge penalty still triggers a full alternating MNGM
solve, warm-started from `A_0` and `B_0`.

The first task comparison therefore includes both the penalty edit and the
transition from fixed `B_0` to an estimated representation precision. Interpret
the first phase accordingly when attributing gains to the policy.

## Patient-MNGM projection

Concept-conditioned attention remains in the full Qwen3-Embedding hidden space.
When `--representation-dim R` is set, PCA is fitted once on the full-dimensional
concept prototypes from the training vocabulary. Let `W` be its `[R,D]`
component matrix and `mu` its `[D]` mean. The patient concept vectors are

```text
H_d^projected = column_L2_normalize(W (H_d^full - mu))
```

No prefix coordinates are selected. The patient cache stores `projection.npz`
with `mean` and `matrix`, and metadata records a digest of both arrays. The
projection is fixed while candidate graphs are evaluated, so all candidates
in a Graph Phase use the same statistical observations. Omitting
`--representation-dim` retains all `D` coordinates. The main H04L run still
uses `R=64` for MNGM memory and solver cost.

Existing v0.4 caches built with `qwen3_mrl_prefix_l2` must be regenerated;
the task runners reject them. The v0.4 smoke result does not verify the new
projection's downstream performance. The Bootstrap-MNGM branch already uses a
separate fixed PCA projection and is unchanged here.

For the current H04L budget, rebuild the cache using the existing v1 data files:

```bash
python scripts/build_patient_matrices.py \
  --train data/patents_h04l/train.csv.gz \
  --prototypes data/patents_h04l/concept_prototypes.npz \
  --model /laijizheng/models/Qwen3-Embedding-0.6B \
  --local-files-only \
  --output data/patents_h04l/cache_main_4096_r64_pca \
  --dtype bfloat16 --cache-dtype float16 \
  --representation-dim 64 --batch-size 16 --shard-size 128 \
  --max-patients 4096 --id-col patent_id
bash scripts/run_patient_h04l_main_v2.sh
```

An explicit projection matrix can be supplied to
`PatientConceptMatrixBuilder`, allowing later research on task-guided updates.
This version does not learn `W` from task loss: the MNGM solver is not an
end-to-end differentiable layer, and changing `W` within a candidate group
would confound graph comparisons. Any future update must rebuild the projected
statistical state and establish a new fixed baseline before evaluating actions.
