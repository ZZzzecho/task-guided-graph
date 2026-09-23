> **Historical document.** This file records the 2026-09-22 MIMIC-first design stage and is not the current experiment entry point. For the current H04L + Patient/Bootstrap-MNGM mainline, read `README.md` and `docs/CURRENT_RESEARCH_HANDOFF_2026-09-23.md`.

# Patient-MNGM + Task-Guided Graph RL：方案交接与代码实现说明

> 日期：2026-09-22  
> 文档版本：v0.1  
> 对应主线基线：`task_guided_graph_mvp_v0.2.0` / `BOOTSTRAP_MNGM_GRAPH_RL_HANDOFF_v0.2_2026-09-22`  
> 目标：锁定 Patient-level MNGM 支线的方法定义、数据边界和代码接口，作为下一步实现的唯一交接入口。  
> 当前结论：**方法定义层面已经闭合，可以进入代码实现。** 后续不再扩展 attention、额外对齐模块或动态 patient encoder；未锁死的内容只保留为工程超参数和规模适配开关。

---

# 0. 一句话定义

Patient 支线与 Bootstrap 支线的核心区别只有 **MNGM 的统计 replicates 来源不同**。

Bootstrap 支线：

$$
H^{(1)},\ldots,H^{(50)}
\quad\text{来自固定的 sentence-order perturbation / bootstrap versions}
$$

Patient 支线：

$$
H_1,\ldots,H_N
\quad\text{来自固定的真实患者病例}
$$

其中每个病例对应一个 patient-specific concept matrix：

$$
H_d\in\mathbb R^{R\times P}.
$$

从 `MNGMEstimator` 之后，**Patient 支线原则上完全复用 Bootstrap v0.2 的 Graph-RL、candidate evaluation、validation acceptance、sample conditioning、SoftGraphTokenizer 和 task Qwen 模块，不再另起一套下游链路。**

完整链路：

```text
MIMIC-IV-ED D_train patient triage text
        ↓
Frozen Qwen3-Embedding-0.6B
        ↓
patient final-layer contextual token states
        ↓
fixed concept pooled embeddings
        ↓
concept-conditioned cosine attention
        ↓
patient-specific concept vectors h_dj
        ↓
H_d ∈ R^(1024 × P)
        ↓
all allowed patient replicates H_patient
        ↓
【从这里开始 H_patient 永久冻结】
        ↓
nonparanormal Patient-MNGM
        ↓
concept precision Θ + representation precision Ω
        ↓
global concept graph G
        ↓
edge-wise penalty matrix Λ
        ↓
Batch-edge GRPO
        ↓
修改 Λ
        ↓
完整重新求 Patient-MNGM
        ↓
candidate graph
        ↓
Bootstrap v0.2 downstream stack
        ↓
sample-conditioned graph
        ↓
SoftGraphTokenizer
        ↓
Qwen task LoRA
        ↓
HOME vs ADMITTED reward
        ↓
validation acceptance
```

---

# 1. 与 Bootstrap 主线的关系

## 1.1 共用部分

以下内容直接继承 `task_guided_graph_mvp_v0.2.0`，Patient 支线不重新设计：

- `GraphSnapshot`
- `GraphState`
- `EdgeFeatures`
- `PolicyInput`
- `WeightedGraphicalLasso`
- `MNGMEstimator`
- active + frontier candidate pool
- batch-edge `ActionGroup`
- selection policy + direction policy
- without-replacement edge sampling
- trajectory-level GRPO reward / advantage
- candidate invalid handling
- full MNGM re-solve
- graph acceptance / rejection
- MIMIC-IV-ED HOME vs ADMITTED 下游任务
- `D_train / D_graph / D_val / D_test` 数据职责
- sample-conditioned graph
- `W_x = D_x W D_x`
- 1-layer weighted message passing
- query-attention `SoftGraphTokenizer`
- Qwen task LoRA
- accepted graph 后的 task adaptation

因此，Patient 支线不是第二套完整算法，而是 **Bootstrap v0.2 的另一种统计前端**。

## 1.2 唯一需要新增的核心模块

需要新增：

```text
PatientRepresentationEncoder
        ↓
PatientConceptMatrixBuilder
        ↓
patient_concept_matrix dataset/cache
        ↓
existing MNGMEstimator
```

数学上：

$$
x_d
\longrightarrow
H_d\in\mathbb R^{1024\times P}.
$$

现有 `MNGMEstimator` 已经支持：

```text
mode = patient_concept_matrix
shape = [N, R, P]
```

所以统计后端原则上不重写，只需要补齐真实 patient representation 的生成、缓存和规模适配。

---

# 2. 数据集与任务

Patient 支线继续使用与 Bootstrap 主线相同的第一套真实医疗实验：

$$
\boxed{
\text{MIMIC-IV-ED triage}
\rightarrow
\text{HOME vs ADMITTED}
}
$$

病例输入只包含 triage 时已经可知的信息：

```text
chiefcomplaint
temperature
heartrate
resprate
o2sat
sbp
dbp
pain
acuity
```

不允许把以下后验信息输入 Patient representation encoder：

- disposition；
- ED discharge diagnosis；
- hospital discharge summary；
- ED stay 后续才产生的结果文本。

病例文本模板与 Bootstrap v0.2 共用，避免两条线出现输入差异。

---

# 3. 数据划分：已经锁死

仍按 `subject_id` 做 patient-grouped split：

$$
D_{\mathrm{train}},
D_{\mathrm{graph}},
D_{\mathrm{val}},
D_{\mathrm{test}}.
$$

职责：

| Split | Patient 支线用途 |
|---|---|
| `D_train` | 生成 Patient-MNGM 的统计 replicates；训练 accepted graph 下的 task LoRA / tokenizer |
| `D_graph` | Graph Phase 中 candidate task reward |
| `D_val` | graph proposal 独立 acceptance |
| `D_test` | 最终一次性评测 |

Patient-MNGM 的正式统计样本定义为：

$$
\boxed{
\mathcal H_{\mathrm{MNGM}}
=
\{H_d:d\in D_{\mathrm{train}}\}
}
$$

MNGM **不读取 HOME/ADMITTED 标签**。

因此初始 Patient-MNGM 图仍然是 unsupervised patient-derived statistical graph；任务标签第一次影响图，是后续 Graph-RL 通过 `D_graph` reward 调整 $\Lambda$。

严禁：

- `D_graph` 参与初始 Patient-MNGM 统计估计；
- `D_val` 参与 MNGM 或 policy reward；
- `D_test` 参与任意训练、图估计、policy 或 acceptance。

---

# 4. Concept vocabulary：直接继承 Bootstrap 版本

Patient 支线使用与 Bootstrap 支线完全相同的 concept vocabulary：

$$
V=\{c_1,\ldots,c_P\}.
$$

要求：

- concept IDs 完全相同；
- concept 顺序完全相同；
- graph node semantics 完全相同；
- 不为 Patient 支线额外创建一套 vocabulary；
- 不因 patient representation 改变 concept 轴。

因此两条线最终可以直接比较：

$$
G_{\mathrm{Bootstrap}}
\quad\text{vs}\quad
G_{\mathrm{Patient}}.
$$

代码中所有缓存必须显式保存：

```text
concept_ids
concept_order
vocabulary_version
encoder_version
representation_shape
```

---

# 5. Patient representation encoder：已经锁死

## 5.1 模型

第一版使用：

$$
\boxed{
E_{\mathrm{pat}}
=
\text{Qwen3-Embedding-0.6B}
}
$$

部署方式：

- 服务器本地下载；
- Hugging Face / Transformers 本地加载；
- 支持指定本地模型目录；
- 支持离线运行；
- 模型权重在 Patient-MNGM / Graph-RL 整个实验中 **永久冻结**。

代码包需提供下载脚本，并提供：

```text
--model-name
--model-path
--local-files-only
```

等配置。

不在第一版中训练 embedding model LoRA。

## 5.2 为什么不做 Qwen LLM ↔ Embedding model 对齐

Patient representation 的统计空间是独立的图估计空间。

它只负责：

$$
\text{patient}
\rightarrow
H_d
\rightarrow
\text{MNGM}
\rightarrow
G.
$$

真正传给下游 task Qwen 的是：

- concept identity；
- graph edge；
- partial-correlation weight。

不会直接把 Patient-MNGM 的 1024 维坐标送入 task Qwen。

因此只要求：

$$
\boxed{
\text{concept prototype 和 patient token representation 来自同一个 embedding encoder}
}
$$

不要求：

$$
\text{Qwen3-Embedding space}
=
\text{task Qwen hidden space}.
$$

第一版不增加 Procrustes、CCA、linear alignment 等跨模型映射。

---

# 6. Concept prototype：方案 A 已锁死

对于 concept $j$，直接使用 Qwen3-Embedding 官方 pooling 后的 sequence embedding：

$$
\boxed{
p_j
=
E_{\mathrm{pat}}(\text{concept text}_j)
}
$$

其中：

$$
p_j\in\mathbb R^{1024}.
$$

不采用：

- `embed_tokens.weight` 的 token 均值；
- 手工 token pooling；
- task Qwen hidden state；
- learnable concept query；
- 跨模型 alignment。

所有 concept prototypes：

$$
P_{\mathrm{concept}}
=
[p_1,\ldots,p_P]
$$

在实验开始前一次性预计算并永久冻结。

建议保存：

```text
concept_prototypes.pt
concept_ids.json
encoder_metadata.json
```

---

# 7. Patient token representation

对于 patient $d$，先使用与 Bootstrap 下游相同的固定 triage text template 得到：

$$
x_d.
$$

Qwen3-Embedding 自己的 tokenizer 完成 subword tokenization：

$$
x_d
\rightarrow
t_{d1},\ldots,t_{dL_d}.
$$

然后读取 embedding encoder 最后一层 contextual hidden states：

$$
Z_d
=
[z_{d1},\ldots,z_{dL_d}],
$$

其中：

$$
z_{dl}\in\mathbb R^{1024}.
$$

第一版固定：

$$
\boxed{
\text{patient token states = final hidden layer}
}
$$

不做：

- 多层融合；
- learnable layer weighting；
- span extractor；
- 额外 clinical chunking；
- 人工“语义单元”切分。

---

# 8. Patient-specific concept representation：公式已经锁死

对于病例 $d$、token $l$、concept $j$，定义 cosine similarity：

$$
s_{dlj}
=
\cos(z_{dl},p_j).
$$

在 patient token 维度上做 softmax：

$$
\alpha_{dlj}
=
\frac{
\exp(s_{dlj}/\tau)
}{
\sum_{m=1}^{L_d}\exp(s_{dmj}/\tau)
}.
$$

得到 concept-conditioned patient vector：

$$
\boxed{
h_{dj}
=
\sum_{l=1}^{L_d}
\alpha_{dlj}z_{dl}
}
$$

其中：

$$
h_{dj}\in\mathbb R^{1024}.
$$

所有 concept 拼成：

$$
\boxed{
H_d
=
[h_{d1},\ldots,h_{dP}]
\in
\mathbb R^{1024\times P}
}
$$

所有允许用于 MNGM 的训练病例形成：

$$
\mathcal H_{\mathrm{patient}}
\in
\mathbb R^{N_{\mathrm{train}}\times 1024\times P}.
$$

---

# 9. 明确不继续优化 attention

第一版不处理以下问题：

> 如果病例中与某个 concept 基本无关，softmax 仍然会产生一个 $h_{dj}$。

这是当前定义的一部分，不增加：

- null token；
- null concept；
- activation gate；
- sparse attention；
- sigmoid attention；
- hard threshold；
- concept presence classifier；
- second-stage attention optimization。

因此：

$$
\boxed{
\text{cosine + token-softmax + weighted sum}
}
$$

作为固定 PatientConceptMatrixBuilder，不再扩展。

---

# 10. Representation dimension：第一版不降维

Qwen3-Embedding-0.6B 的 Patient representation 第一版保留原始：

$$
\boxed{
R=1024
}
$$

即：

$$
H_d\in\mathbb R^{1024\times P}.
$$

第一版不采用：

- PCA；
- random projection；
- MRL truncation；
- learnable projector；
- autoencoder。

只有实际运行发现：

- GPU / CPU memory 不够；
- MNGM solver wall-clock 无法接受；
- representation precision 数值不稳定；

才考虑后续工程降维。

任何后续降维都必须作为显式配置和消融，不允许静默改变 representation semantics。

---

# 11. Patient representations 在 Graph-RL 中永久冻结

这是 Patient 支线最重要的设计原则之一。

生成：

$$
\mathcal H_{\mathrm{patient}}
$$

以后：

$$
\boxed{
\mathcal H_{\mathrm{patient}}\ \text{在整个 Graph-RL 阶段永久固定}
}
$$

因此：

- embedding encoder 不更新；
- concept prototype 不更新；
- patient token states 不刷新；
- patient-specific concept matrices 不刷新；
- nonparanormal 基础统计证据不因 task LoRA 改变。

Graph-RL 中图的变化只来自：

$$
\Lambda_t
\rightarrow
\Theta_t
\rightarrow
G_t.
$$

accepted graph 后虽然会更新 task Qwen LoRA 和 SoftGraphTokenizer，但不会反向刷新 Patient-MNGM 的 $H_d$。

第一版不做慢时间尺度 patient encoder refresh。

---

# 12. Patient-MNGM

固定：

$$
\mathcal H_{\mathrm{patient}}
=
\{H_d\}_{d\in D_{\mathrm{train}}}.
$$

调用现有：

```text
MNGMEstimator(mode="patient_concept_matrix")
```

输入 tensor semantics：

```text
[N, R, P]
N = patient replicates
R = 1024 representation dimensions
P = concepts
```

估计：

- concept-axis precision：

$$
\Theta_C\in\mathbb R^{P\times P};
$$

- representation-axis precision：

$$
\Theta_R\in\mathbb R^{1024\times1024}.
$$

concept partial correlation：

$$
W_{ij}
=
-\frac{
(\Theta_C)_{ij}
}{
\sqrt{
(\Theta_C)_{ii}(\Theta_C)_{jj}
}
}.
$$

得到 global concept graph：

$$
G=(V,W).
$$

继续维护 edge-wise concept penalty matrix：

$$
\Lambda\in\mathbb R^{P\times P}.
$$

GRPO 只控制 concept-axis 的 $\Lambda_{ij}$，representation precision 继续由 MNGM alternating solver 自行估计。

---

# 13. Graph-RL：全部继承 Bootstrap v0.2

Patient 支线不重新定义 Graph-RL。

## 13.1 Candidate pool

继续使用：

$$
E_t^{\mathrm{pool}}
=
E_t^{\mathrm{active}}
\cup
E_t^{\mathrm{frontier}}.
$$

## 13.2 Batch-edge policy

默认：

```text
m = 32 edits / candidate graph
K = 8 candidates / GRPO group
```

policy：

```text
shared edge MLP
  ├── selection head
  └── direction head
```

selection 使用 stochastic sampling without replacement。

## 13.3 Candidate graph transition

一个 candidate 中的 32 条 penalty edits 先一起写入：

$$
\Lambda_t^{(k)}.
$$

之后只执行一次完整联合求解：

$$
\boxed{
\Lambda_t^{(k)}
\xrightarrow{\text{full Patient-MNGM re-solve}}
\Theta_t^{(k)}
\rightarrow
G_t^{(k)}
}
$$

不能逐边求图，不能直接修改 adjacency。

## 13.4 Reward 与 acceptance

Graph Phase 中：

- task Qwen frozen；
- task LoRA frozen；
- SoftGraphTokenizer frozen；
- Patient representations frozen；
- current accepted graph state frozen。

所有 candidate 在同一个 `D_graph` reward batch 上比较。

proposal 最终在独立：

$$
D_{\mathrm{val}}
$$

上与 current graph 比较。

只有通过固定 validation criterion，才 jointly accept：

$$
(\Lambda,\Theta,G).
$$

accepted graph 后才允许在：

$$
D_{\mathrm{train}}
$$

更新 task LoRA + SoftGraphTokenizer。

无论 task model 如何更新：

$$
\mathcal H_{\mathrm{patient}}
$$

都不刷新。

---

# 14. Downstream：完全复用 Bootstrap，不重复设计

Patient 支线得到的是另一张 global concept graph。

之后直接调用 Bootstrap v0.2：

```text
global graph G
        ↓
existing sample activation
        ↓
W_x = D_x W D_x
        ↓
existing 1-layer weighted message passing
        ↓
existing query-attention pooling
        ↓
K soft graph tokens
        ↓
existing Qwen task model
        ↓
HOME vs ADMITTED
```

因此 Patient-MNGM 前端生成的 $h_{dj}$：

- 只用于 MNGM statistical graph estimation；
- 不直接作为下游 patient activation；
- 不直接送进 task Qwen；
- 不创建第二套 SoftGraphTokenizer。

---

# 15. 关键比较：Patient vs Bootstrap

两条线的对比必须控制其他变量。

Bootstrap：

$$
\mathcal H_{\mathrm{boot}}
\in
\mathbb R^{50\times R_{\mathrm{boot}}\times P}
$$

replicate = fixed sentence-order perturbation version。

Patient：

$$
\mathcal H_{\mathrm{patient}}
\in
\mathbb R^{N_{\mathrm{train}}\times1024\times P}
$$

replicate = real patient encounter。

从 `MNGMEstimator` 之后，尽可能保持：

- vocabulary；
- initial concept penalty rule；
- candidate builder；
- policy architecture；
- edit count；
- reward；
- acceptance；
- downstream task model；
- graph tokenization；
- evaluation metrics；

完全一致。

核心研究比较：

$$
\boxed{
\text{perturbation-derived statistical replicates}
\quad\text{vs}\quad
\text{real patient-derived statistical replicates}
}
$$

---

# 16. Patient scalar GGM 的定位

旧方案中的：

```text
patient_activation_vector
[N, P]
```

不再作为第三条主方法。

它保留为：

$$
\boxed{
\text{baseline / ablation}
}
$$

主方法仍然是：

```text
patient_concept_matrix
[N, 1024, P]
```

---

# 17. 规模问题：方法不改，但代码必须支持分片

假设：

$$
P\approx800,\quad R=1024.
$$

单个 patient matrix：

$$
1024\times800=819200
$$

个数值。

若使用 `float32`，约为：

$$
3.125\ \text{MiB / patient}.
$$

因此完整训练集的 Patient matrix cache 可能非常大。

正式数学定义仍然保持：

$$
R=1024.
$$

但代码实现不能默认把完整：

```text
[N, 1024, P]
```

一次性放进 GPU / RAM。

## 17.1 Representation generation 必须分 shard

建议：

```text
patient_repr/
  metadata.json
  concept_ids.json
  shard_00000.*
  shard_00001.*
  ...
```

每个 shard 保存：

```text
subject_id
stay_id
H_d
```

生成逻辑：

```text
patient mini-batch
→ frozen embedding forward
→ batched concept attention
→ H_batch
→ write shard
→ release GPU tensors
```

## 17.2 dtype

缓存允许：

```text
float16 / bfloat16
```

进入 nonparanormal / MNGM 数值计算时转换为 solver 需要的 precision。

dtype 是工程配置，不改变方法定义。

## 17.3 Pilot subset 开关

正式方法定义仍然是全部 `D_train`。

但 smoke test 必须支持：

```text
--max-mngm-patients N
```

从 `D_train` 中按固定 seed / deterministic ordering 取固定子集，用于测：

- 显存；
- 磁盘；
- solver wall-clock；
- convergence。

如果正式实验最终由于算力限制使用子集，必须显式报告，不能伪装成完整 `D_train`。

---

# 18. 下一版建议新增代码

```text
src/
  patient_repr/
    qwen3_embedding_encoder.py
    concept_prototypes.py
    patient_concept_builder.py
    patient_matrix_dataset.py
    cache_io.py

scripts/
  download_qwen3_embedding.py
  build_concept_prototypes.py
  build_patient_matrices.py
  inspect_patient_cache.py
  run_patient_mngm.py
  run_patient_graph_rl.py
```

## 18.1 `Qwen3EmbeddingEncoder`

职责：

```text
load tokenizer
load frozen encoder
return official pooled sequence embeddings
return final-layer token hidden states
```

必须：

```text
model.eval()
requires_grad_(False)
inference_mode()
```

## 18.2 `ConceptPrototypeStore`

输出：

```text
[P, 1024] pooled prototypes
```

绑定：

```text
encoder revision
tokenizer revision
concept vocabulary version
normalization rule
```

## 18.3 `PatientConceptMatrixBuilder`

设：

```text
Z: [B, L, 1024]
C: [P, 1024]
```

归一化后：

$$
S=\bar Z\bar C^\top
$$

shape：

```text
[B, L, P]
```

padding mask 后：

$$
A=\operatorname{softmax}(S/\tau,\text{dim}=L).
$$

然后：

$$
H=Z^\top A
$$

输出：

```text
[B, 1024, P]
```

实现必须 batch-vectorized，不允许 Python 三重循环逐 patient/token/concept 计算。

---

# 19. 必须有的 sanity checks

## Shape

```text
patient token states: [B, L, 1024]
concept prototypes:    [P, 1024]
similarities:          [B, L, P]
attention:             [B, L, P]
H_batch:               [B, 1024, P]
```

## Attention normalization

$$
\sum_l\alpha_{dlj}\approx1.
$$

## Concept order

缓存后：

```text
H[..., j]
```

始终对应同一个 `concept_id[j]`。

## Determinism

frozen model + fixed template 下，同一病例重跑得到相同 $H_d$。

## Freeze

Graph-RL 前后：

```text
patient cache hash
concept prototype hash
encoder metadata
```

应保持不变。

## No-label

`PatientConceptMatrixBuilder` 不允许接收：

```text
label
disposition
diagnosis
```

只接收输入文本与 identifiers。

## Small MNGM smoke

先用：

```text
N = 16 / 32 / 64
```

确认：

```text
patient matrices
→ Patient-MNGM
→ Theta_C
→ Theta_R
→ graph
```

完整通过。

## End-to-end Graph-RL smoke

```text
Patient representation
→ Patient-MNGM
→ active/frontier
→ batch-edge candidate
→ Lambda update
→ full MNGM re-solve
→ existing downstream evaluator
→ reward
→ GRPO
→ acceptance
```

---

# 20. 第一版明确排除的内容

为防止开发阶段再次扩链，以下内容全部排除：

```text
embedding model fine-tuning
embedding-model LoRA
slow representation refresh
task Qwen ↔ embedding space alignment
PCA / learned projection
learnable concept prototype
null concept / null token
attention gate
attention sparsification
hard concept-presence threshold
multi-layer hidden-state fusion
medical span extraction
LLM-based concept matching
patient-specific precision graph
第二套 downstream graph tokenizer
```

---

# 21. 当前只剩工程超参数，不剩方法级 blocker

建议配置：

```yaml
patient_encoder:
  model_name: Qwen/Qwen3-Embedding-0.6B
  hidden_layer: final
  hidden_size: 1024
  frozen: true

patient_concept:
  similarity: cosine
  aggregation: token_softmax_weighted_sum
  temperature: configurable
  dimensionality_reduction: none

mngm_data:
  source_split: train
  max_mngm_patients: null
  cache_dtype: configurable
  shard_size: configurable

graph_rl:
  reuse_bootstrap_v0_2: true
```

`temperature`、batch size、shard size、cache dtype 等通过 smoke experiment 选择，不属于方法定义。

---

# 22. 推荐开发顺序

1. **Embedding model deployment**  
   下载、本地加载、offline load、pooled embedding、token hidden states。

2. **Concept prototypes**  
   固定 vocabulary，生成 `[P,1024]` prototype matrix。

3. **Patient matrix generation**  
   `D_train → fixed triage text → hidden states → cosine → token softmax → H_d → shard cache`。

4. **PatientMatrixDataset**  
   实现 lazy / sharded 读取，并兼容当前 MNGM 输入。

5. **Patient-MNGM smoke**  
   小 subset 测 RAM、GPU memory、wall-clock、solver convergence。

6. **规模测试**  
   `N = 64 → 256 → 1k → 5k → ...`，先找到真实瓶颈再决定是否需要降维。

7. **复用 Graph-RL**  
   接入 `task_guided_graph_mvp_v0.2.0` 的 candidate pool、batch action、GRPO、full solve、reward、acceptance 和 downstream graph injection。

---

# 23. 最终锁定的方法定义

$$
\boxed{
x_d
\xrightarrow{
\text{Frozen Qwen3-Embedding}
}
Z_d
}
$$

$$
\boxed{
p_j
=
\text{official pooled concept embedding}
}
$$

$$
s_{dlj}
=
\cos(z_{dl},p_j),
$$

$$
\alpha_{dlj}
=
\operatorname{softmax}_l(s_{dlj}/\tau),
$$

$$
\boxed{
h_{dj}
=
\sum_l\alpha_{dlj}z_{dl}
}
$$

$$
\boxed{
H_d
=
[h_{d1},\ldots,h_{dP}]
\in\mathbb R^{1024\times P}
}
$$

$$
\boxed{
\mathcal H_{\mathrm{patient}}
=
\{H_d:d\in D_{\mathrm{train}}\}
\quad\text{fixed throughout Graph-RL}
}
$$

$$
\mathcal H_{\mathrm{patient}}
\rightarrow
\mathrm{Patient\text{-}MNGM}
\rightarrow
G_0
\rightarrow
\Lambda\text{-GRPO}
\rightarrow
G_\star.
$$

从 $G$ 开始：

$$
\boxed{
\text{直接复用 Bootstrap v0.2 downstream stack}
}
$$

最终两条主线只有一处本质区别：

$$
\boxed{
\text{Bootstrap-MNGM: perturbation matrices as replicates}
}
$$

vs.

$$
\boxed{
\text{Patient-MNGM: real patient matrices as replicates}
}
$$

这就是下一步代码实现需要遵守的边界。
