# Task-Guided Bootstrap/Patient-MNGM Graph RL v0.3.0

本版本在 v0.2.0 Bootstrap-MNGM 基线上加入已经敲定的 **Patient-MNGM 支线**。两条线从 `MNGMEstimator` 之后完全共用同一套 Graph-RL；区别仅在统计 replicates：Bootstrap 使用固定扰动版本，Patient 使用 `D_train` 中冻结的真实病例矩阵。

Patient 主链：

```text
MIMIC-IV-ED D_train
  ↓
Frozen Qwen3-Embedding-0.6B
  ↓
patient final-layer token states + pooled concept prototypes
  ↓
cosine + token-softmax
  ↓
H_d [1024,P]
  ↓
fixed Patient-MNGM evidence
  ↓
existing active+frontier + batch-GRPO + full MNGM re-solve
  ↓
GLM-4.7-Flash task model + SoftGraphTokenizer
  ↓
validation acceptance
```

## 先看这里

准备上传服务器时优先阅读：

```text
SERVER_QUICKSTART.md
docs/PATIENT_MNGM_GRAPH_RL_HANDOFF_v0.1_2026-09-22.md
docs/IMPLEMENTATION_v0.3.md
```

安装：

```bash
python -m pip install -e ".[all]" --no-build-isolation
python -m pytest -q
```

当前验证：

```text
60 passed
```

### GLM-4.7-Flash 的重要限制

下游 graph token 是通过 `inputs_embeds` 直接注入模型，因此当前训练/Graph-RL 路径需要 **本地 Transformers causal-LM 对象**。只启动一个 OpenAI-compatible vLLM/SGLang HTTP 服务不能透明替代这一步，因为标准 chat API 不接受任意连续 soft embeddings。代码默认支持本地 `zai-org/GLM-4.7-Flash`，并提供 `scripts/check_glm47_flash.py`。

### 两套 prototype 不要混用

- `Patient-MNGM prototype`：Qwen3-Embedding 官方 pooled embedding，只用于生成 `H_d` 和学图；
- `Task activation prototype`：GLM 自己的 `get_input_embeddings()` 子 token 均值，只用于下游 sample-conditioned SoftGraphTokenizer。

两套空间无需对齐，也绝不能在代码里互换。

### Patient cache 不一次性塞内存

`H_d` 第一版保持 1024 维不降维。representation generation 按 shard 落盘，`PatientMatrixDataset` 支持 memmap/lazy read；当前 MNGM solver 仍是 in-memory，所以先用 `--max-mngm-patients` 做 16/32/64 → 256 → 1k 的规模测试，再决定是否真的需要降维或 out-of-core MNGM。

---


## Current first non-medical experiment: H04L patents

The first domain-adaptation experiment is now **PatentsView H04L + CPC concepts +
patent citation retrieval**.

Research constraint: the downstream target is deliberately **not** CPC concept
prediction. CPC assignments define/filter the shared concept universe, while the
actual task target is a cited prior patent. This keeps node activation distinct
from downstream supervision and lets graph-edge quality be tested separately.

Data preparation:

```bash
bash scripts/download_patentsview_h04l.sh /laijizheng/datasets/patentsview_h04l_raw

prepare-patents-h04l \
  --data-dir /laijizheng/datasets/patentsview_h04l_raw \
  --output data/patents_h04l \
  --min-year 2005 \
  --max-year 2024 \
  --target-concepts 800 \
  --min-concept-patents 50
```

The processor streams the large PatentsView tables, restricts the corpus to H04L
utility patents with usable title/abstract text, selects the concept vocabulary
using train-period CPC frequency only, keeps chronologically later graph/val/test
splits, and builds internal H04L citation positives.

See:

```text
docs/DATASET_TASK_FIT.md
docs/PATENTS_H04L_DATA.md
```

## v0.2 Bootstrap 基线说明（仍然有效）

本版本按照 `docs/BOOTSTRAP_MNGM_GRAPH_RL_HANDOFF_v0.2_2026-09-22.md` 对原 `task_guided_graph_mvp_v0.1.2` 做增量升级。统计后端继续使用已经验证的 `WeightedGraphicalLasso` / `MNGMEstimator`，主要新增 **active+frontier 搜索、32-edge batch GRPO、Graph Phase 独立 validation acceptance、MIMIC-IV-ED preprocessing、sample-conditioned SoftGraphTokenizer 与 Qwen causal-LM 接口**。

核心链路：

```text
fixed H^(1:50)
  ↓
MNGM + edge-wise Lambda
  ↓
global partial-correlation graph W
  ↓
patient activation a(x)
  ↓
W_x = D_x W D_x
  ↓
1-layer weighted message passing
  ↓
query pooling → K soft graph tokens
  ↓
frozen Qwen/task-LoRA during Graph Phase
  ↓
dense task reward
  ↓
batch-edge GRPO: select edges + increase/decrease Lambda
  ↓
full MNGM re-solve
  ↓
independent validation acceptance
  ↓
accepted only: task LoRA + SoftGraphTokenizer adaptation
```

## 1. 安装与验证

Python 3.10+：

```bash
python -m pip install -e . --no-build-isolation
python -m pip install -e ".[test,rl,medical]" --no-build-isolation
python -m pytest -q
```

当前验证：

```text
53 passed
```

若要直接接 Hugging Face Qwen + PEFT：

```bash
python -m pip install -e ".[llm,medical]" --no-build-isolation
```

## 2. 统计后端保持三种样本语义

```text
patient_activation_vector              [N,P]
patient_concept_matrix                 [N,R,P]
bootstrap_concept_embedding_matrix     [B,R,P]
```

后两者共用 `MNGMEstimator`，同时估计 concept precision 与 representation precision。GRPO 始终只控制 concept-axis `Lambda`。

任何 penalty 改动都遵循：

```text
Lambda proposal
→ full weighted GGM/MNGM solve
→ Theta
→ partial correlation Rho
```

不存在 policy 直接编辑 `Theta/A` 的路径。

## 3. v0.2 的 candidate pool

默认 `CandidateBuilder` 使用：

```text
active Theta edges
+
zero edges closest to the current weighted-GLASSO KKT boundary
```

frontier score 的推导和实际公式见 `docs/IMPLEMENTATION_v0.2.md`。它基于本仓库 `WeightedGraphicalLasso` 的真实 `Lambda/2` 对称 proximal convention。

## 4. v0.2 的 GRPO action

`GRPOPolicy` 不再是旧版 `2E+1` 单边动作。默认：

```text
one candidate graph = 32 unique edge edits
one GRPO group      = 8 candidate graphs
```

每个 edge 共享一个 MLP encoder，并有两个 head：

```text
selection head
increase/decrease direction head
```

选边按 sequential sampling without replacement；未选边即 keep。整张 candidate graph 只有一个 reward，32 个 edit 共用一个 trajectory-level advantage。

## 5. Bootstrap 表示工具

`graph_mvp/bootstrap.py`：

```python
from graph_mvp.bootstrap import (
    make_shuffled_versions_csv,
    extract_concept_embedding_matrix,
    embedding_forward_weight_gap,
    save_bootstrap_bundle,
    load_bootstrap_bundle,
)
```

句序扰动规则与 `legacy_reference/NMF_PK3rd.py` 保持一致。概念 embedding 通过 `model.get_input_embeddings()(ids)` 的 forward 获取，便于检查 `embed_tokens` LoRA 是否真实生效。

## 6. MIMIC-IV-ED HOME vs ADMITTED

准备数据：

```bash
prepare-mimic-ed \
  --triage /path/to/mimic-iv-ed/triage.csv.gz \
  --edstays /path/to/mimic-iv-ed/edstays.csv.gz \
  --output data/mimic_ed_home_admitted
```

生成：

```text
cohort.csv.gz
train.csv.gz
graph.csv.gz
val.csv.gz
test.csv.gz
summary.json
```

划分按 `subject_id` 分组，代码会显式检查四个 split 之间没有患者交叉。MVP 只保留 HOME / ADMITTED；病例输入只来自 triage 字段。`--without-acuity` 可直接生成 acuity 消融版本。

## 7. SoftGraphTokenizer

```python
from graph_mvp.graph_tokens import bootstrap_concept_prototypes, SoftGraphTokenizer

prototypes = bootstrap_concept_prototypes(H_fixed)  # H_fixed: [B,R,P] -> [P,R]
tokenizer = SoftGraphTokenizer(
    prototypes,
    output_dim=qwen_hidden_size,
    num_tokens=8,
    graph_hidden_dim=128,
    activation_normalization="sigmoid",
)
```

内部实现：

```text
patient token embeddings
→ cosine(concept prototype, token)
→ tau * logsumexp
→ activation a(x)
→ W_x = D_x W D_x
→ one signed weighted message-passing layer
→ learnable-query attention pooling
→ K graph tokens
```

## 8. 接 Qwen / PEFT

`graph_mvp.llm_task.GraphConditionedCausalLM` 接受任意支持 Hugging Face causal-LM 常见接口的模型，因此可以把已经挂好 task LoRA 的 Qwen 直接传入：

```python
from graph_mvp.llm_task import GraphConditionedCausalLM, FrozenGraphCausalEvaluator

model = GraphConditionedCausalLM(peft_qwen, soft_graph_tokenizer)
evaluator = FrozenGraphCausalEvaluator(model)
```

candidate reward 对每个病例同时计算 HOME / ADMITTED 两个答案的 sequence score，再做二分类归一化，task utility 为 mean gold log-prob（即负 binary NLL）。同时记录 AUROC、AUPRC、Accuracy、F1。

病例 concept activation 不读取 gold answer：activation mask 只保留 triage patient text tokens，固定 instruction、`Disposition:` suffix 和答案 token 均被排除。

## 9. Block-wise Graph Phase

```python
from graph_mvp.runner import GraphPhaseRunner
from graph_mvp.llm_task import ErrorWeightedTaskRelevanceProvider

runner = GraphPhaseRunner(
    env,
    grpo_policy,
    reward_evaluator,
    reward_fn,
    candidate_builder=builder,
    validation_evaluator=validation_evaluator,
    feature_provider=ErrorWeightedTaskRelevanceProvider(reward_evaluator),
    task_adapter=accepted_graph_task_adapter,
)

result = runner.run(
    initial_state,
    reward_context=D_graph_context,
    validation_context=D_val_context,
    phases=4,
    policy_updates_per_phase=4,
    num_candidates=8,
)
```

同一 Graph Phase 内 accepted state 固定；candidate rollout 不串联。phase 结束后才用独立 validation context 决定 proposal 是否进入下一 state；只有 accepted graph 才调用 `task_adapter`。

## 10. 工程 smoke CLI

原 QSAR/synthetic CLI 仍保留，可用于检查统计图闭环。GRPO 已自动使用 batch action：

```bash
python run_mvp.py \
  --config configs/mvp.json \
  --policy grpo \
  --rounds 1 \
  --num-candidates 3 \
  --output outputs/smoke_grpo
```

Bootstrap-MNGM 小型工程 fixture：

```bash
PYTHONPATH=. python examples/make_three_mode_smoke_data.py
python run_mvp.py \
  --data examples/generated/downstream_task.npz \
  --graph-data examples/generated/bootstrap_concept_embedding_matrix.npz \
  --config configs/mngm_patient_example.json \
  --policy grpo \
  --rounds 1 \
  --num-candidates 2 \
  --output outputs/smoke_bootstrap_mngm
```

## 11. 真实实验前仍需外部输入

本代码包不包含 MIMIC-IV-ED 原始数据、实际 Qwen 权重、用户的真实 concept vocabulary，也没有擅自决定交接文档中尚未锁死的 Bootstrap snapshot 时点。因此 `53 passed` 与两个 CLI smoke 只证明工程链路能运行，不代表真实医疗任务效果。

完整实现映射、数学细节与剩余接口见：

```text
docs/IMPLEMENTATION_v0.2.md
docs/BOOTSTRAP_MNGM_GRAPH_RL_HANDOFF_v0.2_2026-09-22.md
```
