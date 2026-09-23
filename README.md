# Task-Guided Graph RL v0.4.0

当前代码包维护两条共享 Graph-RL 后端、但统计样本定义不同的 MNGM 路线：

```text
Patient-MNGM
  patient documents
  -> frozen Qwen3-Embedding concept-conditioned matrices H_d
  -> fixed MNGM evidence
  -> task-guided Graph-RL

Bootstrap-MNGM
  fixed sentence-order perturbation versions
  -> independent embedding-LoRA ordinary-LM training
  -> one concept matrix H^(b) per trained version
  -> fixed MNGM evidence
  -> task-guided Graph-RL
```

两条路线从 `MNGMEstimator` 之后共用：

```text
edge-wise Lambda
-> full MNGM re-solve
-> global concept graph
-> sample-conditioned SoftGraphTokenizer
-> GLM-4.7-Flash
-> downstream reward
-> GRPO
-> independent validation acceptance
```

## 当前主实验

当前第一套统一下游实验已经切换到：

```text
PatentsView H04L granted utility patents
+ 800 CPC-derived concept nodes
+ fixed-pool patent citation retrieval
```

下游目标不是 CPC 分类，而是 examiner-cited prior patent retrieval。这样可以把 concept vocabulary 与 downstream supervision 分开。

数据与任务说明：

```text
docs/PATENTS_H04L_DATA.md
docs/CURRENT_RESEARCH_HANDOFF_2026-09-23.md
```

## 当前状态

截至 v0.4.0：

- Patient-MNGM H04L smoke 已跑通：`[N,64,800] -> MNGM -> citation-retrieval Graph-RL`。
- Bootstrap representation generation 已跑通：`sentence perturbation -> independent embedding-LoRA training -> H^(b)`。
- Bootstrap-MNGM 仍在验证其 replicate variation / solver stability。
- Bootstrap 路线不再被描述为轻量方案：正式实现需要每个 version 训练多个完整 epoch，representation generation 很可能成为主要计算成本。
- Graph-RL 后端支持复用固定 `bootstrap_bundle.npz`，避免调 MNGM/RL 时重复训练 bootstrap 前端。

Bootstrap 前端 optimizer-step 预算近似为：

```text
B * E * ceil(N_docs / batch_size)
```

其中 B 是 perturbation versions，E 是 epochs/version。

## Bootstrap v0.4.0 的训练语义

每个 sentence-order perturbation version 是一个独立 representation replicate：

```text
same base model
same initial embedding-LoRA weights
same optimizer hyperparameters
different fixed perturbed corpus version
        ↓
train E complete ordinary-LM epochs
        ↓
snapshot concept embedding matrix H^(b)
```

版本之间会恢复到相同初始 adapter 参数、重建 optimizer，并独立训练完整 epoch；不再沿一个连续 LoRA trajectory 累积训练。

runner 会在训练前打印总预算，例如：

```text
B=50 versions × E=3 epochs/version × 1000 steps/epoch = 150000 optimizer steps
```

因此 Bootstrap-MNGM 当前被视为高成本统计前端，而不是轻量替代。

## Bootstrap runner

入口：

```bash
python scripts/run_bootstrap_patent_graph_rl.py ...
```

正式参数现在使用 epoch：

```bash
--bootstrap-versions 50
--bootstrap-epochs-per-version 3
--bootstrap-docs <planned corpus size>
--bootstrap-batch-size <batch size>
```

如果 `bootstrap_bundle.npz` 已经生成，可直接跳过 representation generation：

```bash
python scripts/run_bootstrap_patent_graph_rl.py \
  --bootstrap-bundle /path/to/bootstrap_bundle.npz \
  ...
```

## Patient-MNGM 当前实现

统计 encoder 为 `Qwen3-Embedding-0.6B`。对每篇 patent：

```text
token contextual states [L,1024]
+ pooled concept prototypes [800,1024]
-> full-1024D concept-token attention
-> concept-conditioned vectors
-> MRL-style first 64 dims + L2 normalization
-> H_d [64,800]
```

当前 H04L smoke 已验证：

```text
[32,64,800]
-> MNGM converged
-> GLM-4.7-Flash citation retrieval
-> Graph-RL phase
```

## Citation retrieval Graph-RL

固定 candidate pool：

```text
1 examiner-cited earlier patent
+ 63 fixed hard negatives
```

Candidate patent embeddings 构成固定 retrieval index。Graph-RL candidate 之间只重新计算 graph-conditioned query representation。

主要 reward 是负 retrieval NLL；同时记录 MRR、Recall@1/10/50、NDCG@10 和 mean rank。测试集不进入 graph search 或 validation acceptance。

## 当前推荐阅读顺序

```text
README.md
docs/CURRENT_RESEARCH_HANDOFF_2026-09-23.md
docs/PATENTS_H04L_DATA.md
docs/IMPLEMENTATION_v0.4.md
```

以下文件保留为历史设计记录，不再代表当前主实验入口：

```text
docs/BOOTSTRAP_MNGM_GRAPH_RL_HANDOFF_v0.2_2026-09-22.md
docs/PATIENT_MNGM_GRAPH_RL_HANDOFF_v0.1_2026-09-22.md
docs/IMPLEMENTATION_v0.3.md
SERVER_QUICKSTART.md
```

## 安装

```bash
python -m pip install -e ".[all]" --no-build-isolation
python -m pytest -q
```

GLM graph-token path 需要本地 Transformers 模型对象，因为 soft graph tokens 通过 `inputs_embeds` 注入。标准 OpenAI-compatible vLLM/SGLang chat API 不能直接替代。

## 版本

`v0.4.0`：H04L retrieval 成为统一 downstream；Patient-MNGM 已跑通 smoke；Bootstrap 改为独立版本 + 完整 epoch 训练，并明确其高计算成本。