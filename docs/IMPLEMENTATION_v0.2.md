# Bootstrap-MNGM + Task-Guided Graph RL v0.2 实现说明

本文件对应 `BOOTSTRAP_MNGM_GRAPH_RL_HANDOFF_v0.2_2026-09-22.md`，记录本轮代码落地的边界。核心原则是：**固定 Bootstrap 统计证据；policy 只改 concept-edge penalty；每个 candidate 必须完整重求统计图；Graph Phase 冻结 task model；validation acceptance 独立于 GRPO reward。**

## 1. 已实现组件

### 1.1 固定 Bootstrap 证据

`graph_mvp/bootstrap.py`：

- 原样抽取 `legacy_reference/NMF_PK3rd.py` 的句子切分/文档内打乱规则；
- `extract_concept_embedding_matrix()` 通过 embedding module 的真实 forward 读取 concept embedding，而不是直接假定 `.weight` 已包含 PEFT 增量；
- `embedding_forward_weight_gap()` 用于检查 PEFT `embed_tokens` forward 与 raw `.weight` 是否一致；
- `save_bootstrap_bundle()` / `load_bootstrap_bundle()` 固定保存 `[B,R,P]`、`concept_ids`、bootstrap version IDs 和模型/adapter 元数据。

**没有擅自决定的部分：**连续 Bootstrap LoRA 训练中“每个 H^(b) 的具体 snapshot 时点”在交接文档中仍被列为 sanity check，因此本包提供快照接口，不把一个未经确认的训练时序写死。

### 1.2 MNGM 与完整 re-solve

原 `MNGMEstimator` / `WeightedGraphicalLasso` 保留。`GraphEnvironment.step()` 现在同时支持：

- 单边 `ActionRecord`（兼容旧 greedy/random baseline）；
- 多边 `ActionGroup`（Bootstrap v0.2 主线）。

对 `ActionGroup`：先一次性修改所有指定的 `Lambda_ij`，再调用一次完整 GGM/MNGM solve。不会逐边求图，也不会直接修改 `Theta` 或 adjacency。

### 1.3 Active + KKT frontier candidate pool

`CandidateBuilder` 不再默认遍历全部 `P(P-1)/2` 边，而使用：

```text
active precision edges + near-active zero edges
```

当前 bundled weighted graphical-lasso 的目标为：

$$
-\log\det\Theta + \operatorname{tr}(S\Theta)
+\sum_{i<j}\Lambda_{ij}|\Theta_{ij}|.
$$

由于对称 full-matrix gradient 对非对角元素计数两次，零边 KKT 条件为：

$$
\left|(S-\Theta^{-1})_{ij}\right|\le \Lambda_{ij}/2.
$$

因此实现的 frontier score 为：

$$
\mathrm{frontier}_{ij}
=\min\left(1,\frac{2|(S-\Theta^{-1})_{ij}|}{\Lambda_{ij}}\right),
\qquad \Theta_{ij}=0.
$$

越接近 1 表示越靠近当前 solver 的 active-set 边界。该公式是按本项目真实 objective / `lam/2` proximal convention 推导的，不是直接套 sklearn GraphicalLasso。

### 1.4 Batch-edge GRPO

`GRPOPolicy` 已从旧的 `2E+1` 单动作策略升级为：

```text
shared edge encoder
  ├─ selection head
  └─ direction head (increase/decrease)
```

一个 candidate 默认包含 `m=32` 条不同 edge edits：

1. selection logits 上按 Plackett-Luce 式 sequential sampling without replacement 依次选边；
2. 每条选中边采样 `increase/decrease`；
3. 未选边即 keep；
4. 32 条 edit 的 selection+direction log-prob 相加得到 trajectory joint log-prob；
5. 同一 state 下 K 个 candidate graph 的 reward 做组内标准化；
6. 每个 candidate 的 32 个 edit 共用一个 trajectory advantage；
7. 使用 PPO/GRPO clipped ratio；
8. KL/entropy 在已采样 trajectory 的每个局部 selection/direction distribution 上计算。

### 1.5 Task-aware edge feature

`EdgeFeatures` 增加：

```text
frontier_score
node degree i / j
error-weighted task relevance
```

`graph_mvp/graph_tokens.py::error_weighted_edge_relevance()` 实现：

$$
q_{ij}^{err}
=\frac1N\sum_n \ell_n a_i(x_n)a_j(x_n).
$$

`ErrorWeightedTaskRelevanceProvider` 可在冻结 task model 的 Graph Phase 内计算并缓存该矩阵，再交给 `CandidateBuilder`。它只决定 policy 的搜索预算，不直接改图。

### 1.6 Graph Phase / validation acceptance

`GraphPhaseRunner` 实现块式交替：

```text
accepted graph state fixed
    ↓
several GRPO candidate groups + policy updates
    ↓
best reward-panel proposal
    ↓
independent validation context
    ↓
accept / reject
    ↓
accepted only → task_adapter callback
```

一个 phase 内 `Lambda/Theta/G/task model` 的 accepted state 不被 candidate rollout 连续覆盖。只有 validation 通过时才 `promote_candidate_to_state()`；只有这时才调用 `task_adapter`，用于 task LoRA + SoftGraphTokenizer 的 accepted-graph adaptation。

### 1.7 Sample-conditioned SoftGraphTokenizer

`graph_mvp/graph_tokens.py` 实现：

1. 固定 prototype：
   $$c_i=B^{-1}\sum_b H_i^{(b)}.$$
2. concept-token cosine similarity；
3. smooth max / log-sum-exp activation；默认 `sigmoid`，也保留 `softmax/none` 供消融；
4. 病例图：
   $$W_x=D_xWD_x.$$
5. 一层 signed weighted message passing；
6. learnable-query attention pooling；
7. `K` 个 continuous soft graph tokens 投影到 Qwen hidden size。

候选评价使用 `snapshot.Rho`，不额外做 downstream hard threshold，因此 penalty 的小变化不会再被一个新阈值人为截断。

### 1.8 Qwen/causal-LM 接口

`graph_mvp/llm_task.py` 不硬编码 transformers 类，而接收任意支持：

```python
model.get_input_embeddings()
model(inputs_embeds=..., attention_mask=..., labels=...)
```

的 causal LM，因此可以直接传 PEFT-wrapped Qwen。

关键实现：

- graph activation 只读取 **patient triage text token**，固定 instruction、`Disposition:` suffix 和 gold answer 均被 activation mask 排除；
- graph soft tokens 在 embedding level 注入；
- candidate dense reward 对每个病例同时计算 `HOME` 与 `ADMITTED` 两个 label-string sequence score，再二分类 log-softmax，使用：
  $$R=\frac1N\sum_n\log p(y_n\mid x_n,G).$$
- 同时报告 AUROC / AUPRC / Accuracy / F1；
- `FrozenGraphCausalEvaluator` 使用 `eval()+no_grad()`，candidate 之间模型参数不更新；
- `adapt_task_model()` 只应在 accepted graph 后调用，实际哪些参数 `requires_grad=True` 由调用方控制（预期仅 task LoRA + SoftGraphTokenizer）。

### 1.9 MIMIC-IV-ED preprocessing

`graph_mvp/mimic_ed.py`：

- 只读 `triage` + `edstays`；
- 只保留 `HOME` / `ADMITTED`；
- 输入仅包含 chief complaint、vitals、pain、acuity；
- 不接受 diagnosis/discharge summary；
- 固定 triage 文本模板；
- 按 `subject_id` 做 `train/graph/val/test` 四划分并验证无患者交叉；
- 支持 `include_acuity=False` 消融；
- 输出 cohort/split CSV.GZ 与缺失率、类别比例、encounter 数统计。

CLI：

```bash
prepare-mimic-ed \
  --triage /path/to/triage.csv.gz \
  --edstays /path/to/edstays.csv.gz \
  --output data/mimic_ed_home_admitted
```

## 2. 默认 Bootstrap v0.2 参数

`configs/bootstrap_mimic_v0.2.json`：

```text
rho penalty multiplier = 1.1
eta = log(1.1)
m = 32 edits / candidate
K = 8 candidates / group
frontier budget <= 256
minimum frontier reserve = 32
```

这些是 handoff 中的 MVP 初始工程值，均为可配置超参数，不是方法定义。

## 3. 没有伪造/没有写死的外部条件

本包没有 MIMIC-IV-ED 原始数据，也没有用户实际的 Qwen checkpoint / task LoRA / 800-node concept vocabulary。因此本轮没有宣称跑出任何真实医疗效果数字。

在真实实验前仍需由实际模型/数据完成：

1. 50 套 `H^(b)` 的最终 snapshot 时点确认与导出；
2. 对真实 PEFT Qwen 运行 `embedding_forward_weight_gap()`；
3. 用真实 concept vocabulary 建立固定 `concept_ids`；
4. 下载/授权 MIMIC-IV-ED 并运行 preprocessing；
5. 用 initial MNGM graph 做 task LoRA + tokenizer warm-up；
6. 冻结模型后进入 `GraphPhaseRunner`；
7. validation acceptance 的主指标在实验启动前固定。本实现的 causal-LM evaluator 默认以 normalized binary NLL（等价于 mean gold log-prob）作为 task utility，同时记录 AUROC/AUPRC 等指标。

## 4. 验证

当前源码测试：

```text
53 passed
```

额外完成：

- vector-GGM + batch GRPO CLI smoke；
- bootstrap-MNGM + batch GRPO CLI smoke；
- ActionGroup 多 penalty → 单次 full solve；
- active + KKT frontier；
- SoftGraphTokenizer；
- MIMIC subject-level split；
- graph-conditioned causal-LM dummy evaluator；
- Graph Phase 固定 state + independent validation acceptance。

这些 smoke tests 是工程正确性证据，不是方法效果证据。
