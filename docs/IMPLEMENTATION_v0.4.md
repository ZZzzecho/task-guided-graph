# Implementation v0.4

Version: 2026-09-23

## 1. Current unified experiment

Both statistical front-ends now use the same H04L patent citation-retrieval task:

```text
PatentsView H04L corpus
concept axis: 800 CPC-derived concepts
downstream: fixed-pool examiner-citation retrieval
task model: GLM-4.7-Flash
```

This replaces the earlier MIMIC-first implementation plan as the current engineering mainline.

## 2. Patient-MNGM front-end

Current patent representation pipeline:

```text
patent title + abstract
-> frozen Qwen3-Embedding-0.6B token states [L,1024]
-> concept-conditioned attention in full 1024D
-> h_dj [1024]
-> MRL-style prefix truncation to 64D + L2 normalization
-> H_d [64,800]
```

The first successful smoke used `[32,64,800]`.

MNGM now uses a hybrid solver:

- concept axis: custom weighted ADMM GLASSO, because Graph-RL needs edge-specific penalties;
- representation axis: sklearn covariance-level scalar graphical lasso with coordinate descent.

The Patient-MNGM H04L smoke converged and completed one citation-retrieval Graph-RL phase.

## 3. Bootstrap-MNGM front-end

### 3.1 Fixed perturbation versions

Each corpus version is produced by document-local sentence-order perturbation. Sentences are only shuffled inside their original document.

### 3.2 Independent version training

v0.4 removes the old smoke-only interpretation in which one version could correspond to one optimizer step.

Current definition:

```text
for bootstrap version b:
    restore same initial embedding-LoRA state
    create fresh optimizer
    train the complete perturbed corpus for E full epochs
    snapshot one concept matrix H^(b)
```

All versions share the same initialization and training budget but differ in fixed perturbed corpus ordering.

### 3.3 Epoch-level cost

For B versions, E epochs/version, N documents and batch size M, the optimizer-step budget is approximately:

```text
B * E * ceil(N / M)
```

With historically discussed B=50 and multiple epochs/version, representation generation can become the dominant computation. Bootstrap-MNGM is therefore not treated as a lightweight front-end.

### 3.4 Representation snapshot

Concept embeddings are extracted through the active embedding-module forward rather than raw weight lookup.

The current patent runner applies one fixed PCA basis, fitted before version-specific training, to obtain:

```text
H^(b) [64,800]
```

The same projection is reused for every version. After generation, all matrices are saved to `bootstrap_bundle.npz` and frozen throughout Graph-RL.

### 3.5 Bundle reuse

`run_bootstrap_patent_graph_rl.py` accepts:

```bash
--bootstrap-bundle path/to/bootstrap_bundle.npz
```

to skip expensive bootstrap representation generation during MNGM/Graph-RL debugging.

## 4. Shared MNGM + Graph-RL backend

Both front-ends feed `MNGMEstimator([replicates,R,P])`, but replicate semantics differ:

```text
Patient-MNGM:
one replicate = one real patent document

Bootstrap-MNGM:
one replicate = one independently trained perturbed-corpus version
```

GRPO only edits concept-axis penalty entries Lambda_ij. Every candidate performs a complete MNGM re-solve before downstream evaluation.

## 5. Shared downstream graph injection

The task model has its own concept prototypes in GLM input-embedding space:

```text
query patent token embeddings
-> soft concept activation a(x)
-> W_x = D(a) W D(a)
-> one weighted graph message-passing layer
-> query-attention pooling
-> K soft graph tokens
-> GLM-4.7-Flash
```

Statistical representation coordinates from Qwen3-Embedding/PCA are not injected directly into GLM.

## 6. Retrieval evaluator

For each query patent:

```text
1 examiner-cited earlier patent
63 fixed hard negatives
```

Candidate patent embeddings are encoded once and kept fixed. Graph changes only affect the query-side graph-conditioned embedding.

Primary Graph-RL task utility: negative mean 64-way retrieval NLL.

Secondary metrics: MRR, Recall@1, Recall@10, Recall@50, NDCG@10, mean rank.

## 7. Graph Phase

```text
freeze task LoRA + SoftGraphTokenizer
-> sample multiple edge-penalty action groups
-> full MNGM re-solve per candidate
-> score candidates on fixed D_graph retrieval panel
-> GRPO update
-> select best proposal
-> compare proposal vs current graph on independent D_val
-> if accepted: promote graph
-> then briefly adapt task LoRA + SoftGraphTokenizer on D_train
```

The statistical replicate tensor remains frozen.

## 8. Current engineering status

Completed:

```text
H04L data preparation
800-node concept vocabulary
fixed retrieval candidate pools
Patient-MNGM R=64 cache
hybrid MNGM solve
GLM graph-token retrieval evaluator
one-phase Patient-MNGM Graph-RL smoke
bootstrap perturbation generation
independent embedding-LoRA reset per version
epoch-based bootstrap training
bootstrap bundle persistence/reuse
```

Still open:

```text
Bootstrap replicate variation sufficient for stable 800-node MNGM
production-scale B / epochs / corpus-size cost
large-sample Patient-MNGM stability
formal baseline/ablation runs
```

## 9. Current entry points

```text
scripts/build_patient_matrices.py
scripts/run_patient_mngm.py
scripts/run_patent_graph_rl.py
scripts/run_bootstrap_patent_graph_rl.py
scripts/build_patent_retrieval_candidates.py
```

## 10. Historical files

The MIMIC-IV-ED plans remain useful design history but are not the current experiment entry point. See `docs/CURRENT_RESEARCH_HANDOFF_2026-09-23.md` for the current state.