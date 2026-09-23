# PatentsView H04L data pipeline

This is the first non-medical domain selected for the Graph-RL experiment.

## Why H04L

The domain is restricted to CPC subclass H04L (transmission of digital information).
The graph nodes are shared CPC subgroup concepts. The downstream target is patent
citation / prior-art retrieval, not CPC prediction.

The key task-fit rule is preserved:

- CPC assignments define/filter the concept universe and are retained for analysis.
- Per-patent CPC assignments are not injected into the model as node activations.
- The sample activation is produced from patent text against the shared concept vocabulary.
- The retrieval target is a cited patent ID, which is distinct from concept identity.
- During graph attribution experiments, text and activation stay fixed and only graph
  edge structure changes.

## Raw files

The first version uses only these PatentsView granted-patent bulk files:

- g_patent.tsv.zip
- g_patent_abstract.tsv.zip
- g_cpc_current.tsv.zip
- g_cpc_title.tsv.zip
- g_us_patent_citation.tsv.zip

Download:

```bash
bash scripts/download_patentsview_h04l.sh /laijizheng/datasets/patentsview_h04l_raw
```

The script uses resume-safe wget downloads.

## Prepare the H04L corpus

```bash
prepare-patents-h04l \
  --data-dir /laijizheng/datasets/patentsview_h04l_raw \
  --output data/patents_h04l \
  --min-year 2005 \
  --max-year 2024 \
  --target-concepts 800 \
  --min-concept-patents 50
```

The processor is chunked and does not load the multi-gigabyte PatentsView tables
into memory all at once.

## Outputs

```text
data/patents_h04l/
├── corpus.csv.gz
├── train.csv.gz
├── graph.csv.gz
├── val.csv.gz
├── test.csv.gz
├── retrieval_train.csv.gz
├── retrieval_graph.csv.gz
├── retrieval_val.csv.gz
├── retrieval_test.csv.gz
├── concepts.json
├── concept_assignments.csv.gz
├── citations.csv.gz
├── task.json
└── summary.json
```

The four main corpus splits are chronological. Concept frequency selection is based
only on the train split. The official CPC hierarchy edges are not used during graph
training; they remain available for later structural evaluation.

## First smoke after preprocessing

Inspect:

```bash
cat data/patents_h04l/summary.json
```

Then build the first 32-sample concept matrix cache from the training texts:

```bash
python scripts/build_concept_prototypes.py \
  --concepts data/patents_h04l/concepts.json \
  --model /laijizheng/models/Qwen3-Embedding-0.6B \
  --local-files-only \
  --output data/patents_h04l/concept_prototypes.npz \
  --dtype bfloat16

python scripts/build_patient_matrices.py \
  --train data/patents_h04l/train.csv.gz \
  --prototypes data/patents_h04l/concept_prototypes.npz \
  --model /laijizheng/models/Qwen3-Embedding-0.6B \
  --local-files-only \
  --output data/patents_h04l/cache_smoke \
  --dtype bfloat16 \
  --cache-dtype float16 \
  --representation-dim 64 \
  --batch-size 4 \
  --shard-size 16 \
  --max-patients 32
```

The current generic cache builder still uses historical patient naming, but its
input contract is only a text column, so it is valid for patents.

The citation-retrieval Graph-RL evaluator is a separate downstream adapter and is
not the old binary HOME/ADMITTED evaluator.


## Representation dimension for Patent-MNGM

The patent mainline uses `R=64` for MNGM, matching the scale of the original
GRIAN/MNGM experiments.

Important: Qwen3-Embedding still computes token states and concept-conditioned
attention in its full hidden space. Only the final concept-specific vector
`h_dj` is reduced before caching:

```text
Qwen token states [L,1024]
        +
concept prototype [1024]
        ↓
full-space cosine + token softmax
        ↓
h_dj [1024]
        ↓
Qwen3 MRL-style first 64 dims + L2 normalize
        ↓
h_tilde_dj [64]
        ↓
H_d [64,800]
```

Thus the reduction changes only the statistical MNGM representation axis; it does
not change downstream GLM hidden size or graph-token output dimension.


## Citation-retrieval Graph-RL

Build deterministic fixed candidate pools for the three non-test roles:

```bash
python scripts/build_patent_retrieval_candidates.py \
  --prepared-dir data/patents_h04l \
  --output data/patents_h04l/retrieval_train_candidates.jsonl \
  --split train \
  --negatives-per-query 63 \
  --max-queries 512 \
  --seed 17

python scripts/build_patent_retrieval_candidates.py \
  --prepared-dir data/patents_h04l \
  --output data/patents_h04l/retrieval_graph_candidates.jsonl \
  --split graph \
  --negatives-per-query 63 \
  --max-queries 512 \
  --seed 17

python scripts/build_patent_retrieval_candidates.py \
  --prepared-dir data/patents_h04l \
  --output data/patents_h04l/retrieval_val_candidates.jsonl \
  --split val \
  --negatives-per-query 63 \
  --max-queries 512 \
  --seed 17
```

The downstream task is fixed-pool examiner-citation retrieval. For each query patent,
the pool contains one examiner-cited earlier patent and 63 fixed hard negatives.
Graph-RL uses mean 64-way retrieval NLL as the task utility
(`task_metric = -mean_nll`). MRR, Recall@K, NDCG@10 and mean rank are reported as
secondary metrics.

Candidate patent embeddings are encoded once into a frozen retrieval index. The
query patent alone receives sample-conditioned graph tokens, so changing a graph
candidate changes the query representation while the retrieval pool and candidate
representations stay fixed within and across graph phases.

Minimal end-to-end smoke:

```bash
python scripts/run_patent_graph_rl.py \
  --cache data/patents_h04l/cache_smoke_32_r64 \
  --concepts data/patents_h04l/concepts.json \
  --data-dir data/patents_h04l \
  --train-pools data/patents_h04l/retrieval_train_candidates.jsonl \
  --graph-pools data/patents_h04l/retrieval_graph_candidates.jsonl \
  --val-pools data/patents_h04l/retrieval_val_candidates.jsonl \
  --config configs/patent_h04l_glm_v0.4.json \
  --task-model /laijizheng/models/GLM-4.7-Flash \
  --task-local-files-only \
  --initial-lambda 0.4 \
  --max-mngm-documents 32 \
  --max-train-queries 8 \
  --max-reward-queries 4 \
  --max-val-queries 4 \
  --warmup-steps 2 \
  --adapt-steps 2 \
  --phases 1 \
  --num-candidates 2 \
  --policy-updates-per-phase 1 \
  --task-batch-size 1 \
  --candidate-batch-size 4 \
  --task-max-length 256 \
  --output outputs/patent_h04l_graph_rl_smoke
```

The test split is not loaded by this runner.
