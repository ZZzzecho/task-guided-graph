# Current Research Handoff — 2026-09-23

## 0. Current project state

The project now has one common downstream task and two alternative statistical graph front-ends:

```text
                         ┌─ Patient-MNGM
H04L patent corpus ──────┤
                         └─ Bootstrap-MNGM
                                ↓
                             concept graph
                                ↓
                         task-guided Graph-RL
                                ↓
                         citation retrieval
```

The common downstream task is H04L examiner-citation retrieval. The old MIMIC HOME/ADMITTED design is no longer the current first experiment.

## 1. Unified downstream task

Corpus: PatentsView H04L granted utility patents, 2005–2024, using title + abstract.

Concept universe:

```text
P = 800
high-frequency H04L CPC groups/subgroups
```

CPC assignments are used to define/filter the concept universe and to build fixed hard-negative retrieval pools. They are not provided as sample activation, and CPC hierarchy edges are not used as learned graph edges.

For one query patent:

```text
1 examiner-cited earlier H04L patent = positive
63 fixed earlier H04L hard negatives
```

The downstream problem is citation retrieval / link prediction, not CPC classification.

## 2. Shared Graph-RL backend

Both front-ends eventually produce a concept precision matrix and partial-correlation graph. Graph-RL never edits adjacency directly; it edits edge-specific sparsity penalties Lambda_ij. Each candidate is obtained only through a full MNGM re-solve.

Shared downstream path:

```text
global graph W
+ query patent text
-> GLM concept activation a(x)
-> W_x = D(a) W D(a)
-> SoftGraphTokenizer
-> graph-conditioned GLM query embedding
-> compare against fixed candidate-patent index
-> retrieval NLL reward
```

Candidate patent embeddings are frozen during Graph-RL.

## 3. Patient-MNGM line

One real patent is one MNGM replicate:

```text
[N patents, 64 representation dimensions, 800 concepts]
```

Representation pipeline:

```text
frozen Qwen3-Embedding-0.6B
-> full 1024D token states
+ 1024D pooled concept prototypes
-> cosine + token-softmax
-> h_dj [1024]
-> first 64 dims + L2 normalization
-> H_d [64,800]
```

The attention is still computed in the full 1024D space.

Current smoke result:

```text
[32,64,800]
-> Patient-MNGM converged
-> 800-node graph
-> fixed-pool retrieval
-> one Graph-RL phase
```

This proves the end-to-end software path, not statistical performance.

## 4. Bootstrap-MNGM line

### 4.1 Why it is now considered expensive

The earlier simplified smoke used one optimizer step per perturbation version. That is no longer considered a faithful version of the historical representation procedure.

The current interpretation is that each version receives multiple complete ordinary-LM epochs.

If B is the number of versions, E is epochs/version, N is corpus documents, and M is batch size, representation generation alone requires approximately:

```text
B * E * ceil(N / M)
```

optimizer steps. With B=50, this can dominate total experiment time.

### 4.2 Version semantics

Each fixed sentence-order perturbation version is trained independently:

```text
same base GLM
same initial embedding-LoRA weights
fresh optimizer
different fixed perturbed corpus
train E full epochs
snapshot H^(b)
```

Versions do not share a continuous LoRA training trajectory.

### 4.3 Why this change matters statistically

Using snapshots along one continuous training trajectory makes the version axis partly encode training time. Since MNGM estimates dependence across the replicate axis, that can create strong common rank patterns and unstable covariance estimates.

Independent resets give cleaner replicate semantics:

```text
same initialization / same training budget / different corpus perturbation
```

### 4.4 Current projection

The task model hidden size is independent from MNGM representation width. The current bootstrap patent runner uses one fixed PCA basis, fitted before version-specific training, to produce:

```text
H^(b) [64,800]
```

The same projection is reused for every version.

### 4.5 Current bootstrap status

Representation generation itself works and bundles can be saved. The current blocker is whether independently trained perturbation versions provide enough variation for stable 800-node MNGM estimation at practical training cost.

The previous B=4 and B=16 smoke bundles were trained with only one step/version and both failed concept-axis MNGM convergence. These are no longer treated as evidence against the bootstrap idea because their training budget was intentionally too weak.

## 5. Bootstrap cost-control rule

Do not regenerate bootstrap representations while debugging MNGM or Graph-RL. Once generated, save:

```text
bootstrap_bundle.npz
```

Then reuse it with `--bootstrap-bundle ...`. This separates expensive representation generation from MNGM/RL development.

## 6. Current code behavior in v0.4.0

`run_bootstrap_patent_graph_rl.py` now uses:

```text
--bootstrap-epochs-per-version
```

instead of the old smoke-oriented `--bootstrap-steps-per-version`.

For every version it:

1. restores identical initial embedding-LoRA weights;
2. initializes a fresh optimizer;
3. shuffles document order deterministically within each epoch;
4. consumes the complete version corpus each epoch;
5. snapshots the concept matrix only after all configured epochs.

Before training it prints the total estimated optimizer-step budget.

## 7. MNGM solver state

Both Patient and Bootstrap use the same `MNGMEstimator`.

Concept axis: custom weighted ADMM graphical lasso, because Graph-RL later requires edge-specific Lambda_ij.

Representation axis: sklearn `graphical_lasso(emp_cov=..., scalar alpha)` because only one scalar representation penalty is needed.

For Patient-MNGM this hybrid solver already converges on the current smoke.

## 8. Current repository version

```text
v0.4.0
```

Current primary docs:

```text
README.md
docs/CURRENT_RESEARCH_HANDOFF_2026-09-23.md
docs/IMPLEMENTATION_v0.4.md
docs/PATENTS_H04L_DATA.md
```

Old MIMIC-related handoffs remain historical references only.

## 9. Immediate next experiments

Patient line: scale beyond 32 patent matrices and then evaluate task performance/baselines.

Bootstrap line: do not immediately jump to full B=50 on the full corpus. First measure cost and representation variation under a realistic epoch budget, for example smaller fixed corpus with B=8 or 16 and multiple full epochs.

If representation variation remains too weak despite realistic training, Bootstrap-MNGM may be computationally unattractive relative to Patient-MNGM.

## 10. Main research comparison

```text
Patient-MNGM: real documents as statistical replicates
vs
Bootstrap-MNGM: independently trained perturbation versions as statistical replicates
```

with identical concept vocabulary, MNGM backend, Graph-RL policy, task model, graph-token injection, citation-retrieval reward, and validation protocol.