# Graph-RL 研究交接文档：Bootstrap-MNGM 与 Patient-level 两条主线

> 日期：2026-09-22  
> 目的：作为下一轮新对话窗口的交接文档。本文只记录目前已经讨论、已经实现或明确标记为待讨论的内容；对尚未敲定的地方会明确写成“未定”，不默认补全。  
> 当前代码基线：`task_guided_graph_mvp_v0.1.2`。

---

## 0. 一页式总览

目前研究框架已经收敛为一个共同的强化学习骨架，加两条不同的“统计学图”主线。

共同骨架是：

```text
LLM / LoRA 产生或影响 concept 表示
        ↓
统计学图估计器
        ↓
全局 concept graph
        ↓
候选 edge-wise penalty 调整
        ↓
完整重新求图
        ↓
下游任务评价
        ↓
reward
        ↓
GRPO 更新 penalty policy
        ↓
独立 acceptance 决定是否接受新图
```

最重要的不变量是：

> **强化学习不直接加边、删边或改 precision matrix；它只调整 concept 边的稀疏惩罚强度。每次惩罚改变后，都必须由统计求解器完整重新求图。**

当前保留两条研究线：

### 主线 A：Bootstrap-MNGM + GRPO

这条线最接近最早的 GRIAN/MNGM 思路，目前已经敲定较多。一个 MNGM replicate 对应一次“文档内部句序扰动”所关联的 concept embedding matrix。50 个扰动版本对应 50 套 concept embedding matrices，全部作为 MNGM 的矩阵观测。

### 主线 B：Patient-level + GRPO

这条线以真实病例作为统计样本，目前还没有完全敲定。内部暂时保留两个统计后端：

1. 一个患者对应一个 concept scalar activation vector，使用普通 nonparanormal weighted GGM；
2. 一个患者对应一个 `representation × concept` 矩阵，每个 concept 是患者特异性的向量表示，使用 patient-level MNGM。

其中第二种 patient-level MNGM 对当前研究最有吸引力，但它的 LLM 表示提取模块还未实现。已经标记的第一版方案是：**用 concept-conditioned attention 对患者 token hidden states 加权，生成 patient-specific concept vector。**

---

# 1. 当前统一任务定义

设固定的医学 concept vocabulary 中有 `P` 个 concept。这里 `P` 表示概念节点数，例如 40、50 或 60。

无论使用哪一种统计学图方式，强化学习最终操作的对象都是同一个 concept graph。当前代码将 concept graph 的 precision matrix 记为 concept precision，并通过 partial correlation 得到图结构。

强化学习维护一个 `P × P` 的 edge-wise penalty matrix。下面把它记作：

\[
\Lambda = (\lambda_{ij}).
\]

其中 `\lambda_{ij}` 表示 concept `i` 与 concept `j` 这条潜在边的稀疏惩罚强度。

GRPO 的动作不是“加边/删边”，而是对某一条边执行：

```text
increase penalty
keep
or
decrease penalty
```

当前环境使用乘法更新：

\[
\lambda'_{ij}=\lambda_{ij}\exp(\pm \eta),
\]

其中 `η` 是 penalty 调整步长。

随后，统计求解器必须重新求解新的图。这个原则在 vector GGM 和 MNGM 两种后端中都不变。

---

# 2. 当前代码已经实现什么

当前 `v0.1.2` 已经把学图层正式抽象为三种 sample semantics。

| 代码模式 | 一个统计样本 | 输入 shape | 求解器 |
|---|---|---|---|
| `patient_activation_vector` | 一个患者的 concept scalar activation vector | `[N, P]` | nonparanormal weighted GGM |
| `patient_concept_matrix` | 一个患者的 concept vector matrix | `[N, R, P]` | nonparanormal MNGM |
| `bootstrap_concept_embedding_matrix` | 一个 bootstrap/扰动版本的 concept embedding matrix | `[B, R, P]` | nonparanormal MNGM |

符号含义：

- `N`：患者/病例样本数；
- `B`：bootstrap/句序扰动版本数；
- `R`：representation 维度；
- `P`：concept 数。

后两种矩阵模式共用同一个 `MNGMEstimator`，所以 patient-level MNGM 和 bootstrap-MNGM 的求解算法本身一致，只是第一维的统计含义不同。

### 当前 MNGM 求解器

对一个矩阵样本，记为：

\[
H_s\in\mathbb R^{R\times P}.
\]

这里 `s` 表示第几个矩阵样本；在 bootstrap 线里它是第几个扰动版本，在 patient 线里它是第几个患者。

MNGM 同时估两个 precision matrix：

- concept precision：`P × P`，这是我们真正对外暴露给 GRPO 和下游任务的图；
- representation precision：`R × R`，用于建模 representation 轴的依赖结构。

固定 representation precision 后，代码计算 concept 方向的有效 covariance，再调用已经验证的 `WeightedGraphicalLasso` 更新 concept precision；固定 concept precision 后，再更新 representation precision。两个方向交替到收敛。

GRPO **只控制 concept 轴的 edge-wise penalty matrix**。representation 轴目前只使用统一的 scalar penalty。

### 当前 MNGM 非参数变换

代码支持两种选项：

- `rank_gaussian`：当前默认，更易扩展；
- `spearman_sine`：兼容早期 `bigraph_phy.py` 中的 Gaussian-copula / Spearman-sine 思路。

当前已经明确：**后续直接使用现在规范化后的 v0.1.2 求解器，不再回到早期 `bigraph_phy.py` 的具体 GraphicalLasso 实现细节。** 旧代码只作为历史设计参考。

### 当前测试状态

在 2026-09-22 再次验证：

```text
45 passed
```

已经跑通的工程路径包括：

```text
patient_concept_matrix
bootstrap_concept_embedding_matrix
patient_concept_matrix + GRPO
```

这些 MNGM 测试是工程正确性 smoke test，不是方法效果证据。

---

# 3. 主线 A：Bootstrap-MNGM + GRPO

## 3.1 这条线目前的定位

这条线保留最早的 MNGM 思路：concept 不是患者级 scalar，而是模型中的 embedding vector；多次受控的语料扰动与 LoRA 训练产生多套 concept embedding matrices，再由 MNGM 同时估计 concept 轴和 representation 轴的条件依赖。

这条线当前已经敲定的部分明显多于 patient-level 线。

---

## 3.2 Bootstrap/扰动版本如何生成：已敲定

不是对 concept occurrence 单独 bootstrap，也不是把整个 corpus 的所有句子混在一起全局打乱。

当前保留旧代码的做法：

> **对每一篇医疗文本内部先切句，再只在该文档内部随机打乱句子顺序。不同文档之间的句子不会混合。**

假设一篇摘要原来是：

```text
S1 S2 S3 S4
```

一个扰动版本可能变成：

```text
S3 S1 S4 S2
```

另一篇文档单独进行自己的重排。

当前保留：

```text
version_1
version_2
...
version_50
```

共 50 个固定 shuffled versions。

这些 version 的随机种子应固定并在后续 RL / LoRA 外层迭代中重复使用，以避免把“扰动集合变化”误认为“模型表示变化”。

严格统计术语上，这更接近 **sentence-order permutation / contextual perturbation**，不是 textbook iid bootstrap。项目内部可以继续沿用 “bootstrap version” 的历史命名，但论文写作时需要精确定义。

---

## 3.3 Concept representation 如何取：已敲定

当前明确不切换到 contextual hidden state，而是保留旧代码的方式：

> **直接使用当前模型 input embedding 中 concept 对应 token 的 embedding；如果一个 concept 被 tokenizer 切成多个 token，则取 token embedding 均值。**

对第 `j` 个 concept，假设它包含 `K_j` 个 token，则 concept embedding 为：

\[
e_j=\frac{1}{K_j}\sum_{k=1}^{K_j}E[\mathrm{token}_{jk}],
\]

其中：

- `E`：当前模型的有效 input embedding；
- `K_j`：第 `j` 个 concept 的 token 数；
- `e_j`：该 concept 的向量表示。

当前明确：

- 不使用 contextual hidden states；
- 不额外做 PCA；
- 不人为规定必须降到 64 维；
- embedding 实际是多少维，MNGM 就接收多少维。

旧 `bigraph_phy.py` 文件名里的 `64_version_*_embeddings.csv` 只能说明当时输入文件恰好是 64 维，不能推出 MNGM 必须做 64 维降维。当前新框架不加入额外降维步骤。

---

## 3.4 为什么这个阶段必须训练 LLM/LoRA：已敲定

因为当前 concept 表示直接来自 input embedding，而不是 contextual hidden state。

如果 Qwen+LoRA 完全冻结，仅仅改变句子顺序，直接读取同一个 embedding table 时 concept embedding 不会变化。因此句序扰动必须通过 **训练路径** 间接产生不同的 concept embedding。

当前路径是：

```text
sentence-order perturbation
        ↓
ordinary causal LM training
        ↓
LoRA / effective embedding parameters change
        ↓
concept embeddings change
        ↓
形成多套 concept embedding matrices
```

所以 bootstrap-MNGM 的 representation-generation 阶段本身就包含 LLM/LoRA 训练。

---

## 3.5 这个阶段使用什么 loss：已敲定

只使用标准 causal LM loss。

不再保留最早 `NMF_PK3rd.py` 中把 LM loss、MNGM-like loss 和 Laplacian loss 混在一起的 `CustomLoss`。

标准 causal LM loss 可以写成：

\[
\mathcal L_{\mathrm{LM}}
=-\sum_t \log p_\theta(x_t\mid x_{<t}),
\]

其中：

- `x_t`：当前位置真实 token；
- `x_<t`：它前面的 token；
- `θ`：当前 Qwen+LoRA 的参数。

新的职责分离是：

```text
普通 LM loss → 负责产生模型/embedding 的变化
MNGM solver → 负责从保存下来的 embedding matrices 学图
GRPO → 负责通过下游 task reward 调 concept edge penalty
```

不再在一个 loss 中同时反向传播学习图和模型。

---

## 3.6 LoRA 旧参数：暂时按原代码默认设置保留

目前讨论结果是：如果旧代码已有默认参数，第一版先照抄，不急于调参。

当前参考设置：

```text
n_versions = 50
seed = 123
max_length = 512
per_device_train_batch_size = 2
grad_accum = 3
learning_rate = 1e-4
epochs = 1
optimizer = adamw_torch
```

LoRA：

```text
rank r = 4
lora_alpha = 16
lora_dropout = 0.05
bias = none
task_type = CAUSAL_LM

target_modules:
- q_proj
- k_proj
- v_proj
- embed_tokens
```

注意：旧脚本中的函数默认值与 CLI 默认值并不完全一致。上面列的是旧脚本通过 `main()` 实际默认传入训练函数的核心配置。

这里的 `embed_tokens` 很关键，因为当前 Bootstrap-MNGM 的 concept representation 就来自 embedding 层。

---

## 3.7 50 个 MNGM 矩阵如何组织：已敲定

旧 `bigraph_phy.py` 明确读取：

```text
version_1_embeddings
version_2_embeddings
...
version_50_embeddings
```

所有 50 套都存下来，然后一次性组成 MNGM 数据。

如果每套 embedding matrix 有 `P` 个 concepts、每个 concept 是 `R` 维向量，那么新框架统一存成：

```text
[B, R, P]
```

在当前默认计划中：

```text
B = 50
```

每一个 `H^(b)` 都是：

```text
representation dimension × concept
```

然后：

```text
H^(1)
H^(2)
...
H^(50)
        ↓
当前 v0.1.2 MNGMEstimator
        ↓
concept precision + representation precision
```

当前不再使用早期 `bigraph_phy.py` 自己的 GPU Spearman / sklearn GraphicalLasso 实现；它的核心统计思想已经被规范化进当前 estimator。

---

## 3.8 仍有一个上游细节尚未完全闭合

目前已选择“原代码式”，即 **不为 50 个 shuffled versions 分叉出 50 个相互独立的 LoRA 模型**，而是保留一条连续的训练轨迹。

同时已明确：最终 50 套 version-specific embedding matrices 都要保存，并作为 MNGM 的 50 个矩阵观测。

但当前提供的两份旧代码之间仍缺少一段明确的“embedding snapshot 生成脚本”：

- `NMF_PK3rd.py` 会在一个 epoch 中抽若干 version 拼接后连续训练；
- `bigraph_phy.py` 假定磁盘上已经存在 50 个 `version_i_embeddings.csv`。

因此还需要在正式接 LLM 时明确一件实现细节：

> **在连续 LoRA 训练轨迹中，每个 version 对应的 embedding matrix 具体在哪个训练时点读取和保存。**

当前研究方向不需要重新讨论是否保存 50 套——这个已经确定；只需要把 snapshot 的工程时序补齐。

---

## 3.9 一个必须验证的 PEFT 实现细节

旧代码直接读取：

```python
model.get_input_embeddings().weight
```

但在 PEFT/LoRA 中，如果 `embed_tokens` 的适配是通过额外低秩支路作用在 forward 上，那么直接读取 `.weight` 未必等于“包含 LoRA 增量后的有效 embedding”。

因此正式实现时必须做一个小验证：

```text
LoRA training before / after
        ↓
同一个 concept token
        ↓
直接读 .weight 与实际 embedding forward 输出比较
```

如果 `.weight` 只暴露 base weight，需要改成：

- 通过 embedding module 的 forward 得到有效 embedding；或
- 临时 merge adapter 后读取；
- 但不能悄悄读取一个没有发生变化的 frozen base table。

这是 Bootstrap 线当前最重要的工程 sanity check 之一。

---

# 4. Bootstrap 线：图出来之后的 RL / LLM 闭环

这一部分昨天已经开始讨论，但还没有全部拍板。

## 4.1 MNGM → candidate graph：代码已实现

MNGM 当前 graph state 中包含：

- concept covariance；
- concept penalty matrix；
- concept precision；
- partial correlation；
- adjacency；
- 对 MNGM 额外保存 representation precision / covariance 等 auxiliary state。

GRPO 改变一条 concept edge 的 penalty 后，MNGM 必须完整重新交替求解 concept / representation precision。

这一部分当前代码已经支持，不需要再重写求解器。

---

## 4.2 Candidate graph 如何进入真实 LLM：未定

目前讨论过三种 graph injection 方式：

1. **文本序列化**：把局部图的边/路径写成文本拼到病例 prompt 中；
2. **soft graph tokens**：把局部图编码成若干连续向量作为 soft tokens；
3. **Graph encoder / GAT → K/V memory**：最接近早期 GRIAN 的结构注入。

当前倾向曾经是：第一版为了先跑通闭环，可以优先测试文本序列化；但这一点还没有最终锁死。

一个重要共同原则已经明确：

> MNGM 学到的是全局 concept graph；对一个具体患者下游任务时，应该先根据患者相关 concepts 从全局图中抽取 local subgraph，再注入 LLM，而不是把整张全局图无差别塞进去。

---

## 4.3 一轮 GRPO 中 candidate graph 怎样通过 LLM 得到 reward：未定

目前保留三种方案。

### 方案 A：Frozen-LLM reward

当前 LLM 参数完全固定。不同 candidate graph 在相同病例 reward panel 上前向，比较下游任务分数。

优点：

- 最省算力；
- candidate 之间唯一主要变化是 graph，归因最干净。

它回答的问题是：

> “哪张 graph 对当前模型立即最有用？”

### 方案 B：Short-LoRA reward

这是最初 RL 框架文档里的方案：每个 candidate 从同一个 LLM 起点出发，做相同预算的短 LoRA，再在 reward batch 上评价。

它回答的问题是：

> “哪张 graph 经过短暂适配以后最有潜力？”

缺点是成本很高，并且 candidate 差异和短 LoRA 优化轨迹会混在一起。

### 方案 C：Two-stage evaluation

先用 Frozen-LLM 对较多 candidate 做便宜初筛，再只对 Top-M candidate 做 short-LoRA 复核。

这是目前一个很有吸引力的折中，但尚未最终选择。

---

## 4.4 Accepted graph 之后如何 LoRA：概念已明确，具体方案未定

“用图做 LoRA”并不是把 graph 本身当监督标签。

准确含义是：

```text
accepted global graph
        ↓
对每个病例取 local graph
        ↓
patient input + graph-conditioned input
        ↓
Qwen
        ↓
真实下游任务 loss
        ↓
LoRA update
```

例如下游任务可以是 `HOME vs ADMITTED`，那么监督标签还是患者真实结局，graph 只是额外 conditioning information。

因此这一步是：

> **graph-conditioned LoRA**，不是 graph-supervised LoRA。

尚未敲定：

- graph injection 具体采用哪一种；
- downstream LoRA 用多少 step / epoch；
- target modules；
- bootstrap representation-generation 的 LM adapter 与 downstream graph-conditioned adapter 是同一个 adapter 连续更新，还是分开维护。

最后一点非常重要，目前还没有决定。

---

## 4.5 LoRA 更新后 representation 是否刷新：原则已定，频率未定

如果下游 graph-conditioned LoRA 更新了模型，那么 bootstrap concept embeddings 也会随模型改变。

因此闭环原则上需要：

```text
accepted graph
        ↓
downstream LoRA update
        ↓
模型状态改变
        ↓
重新使用固定的 50 个 shuffled versions
        ↓
重新生成 50 套 concept embedding matrices
        ↓
重新 MNGM
```

这里固定的 50 个 shuffled versions 应保持不变，以便将表示变化尽量归因于模型更新。

尚未决定：

- 每接受一次新图就完整 refresh；
- 还是累计若干 graph update 后再 refresh，以节省算力。

---

# 5. 主线 B：Patient-level + GRPO

这条线还没有 Bootstrap 线那么成熟。当前应该把它看成“一个病例作为真正统计样本”的大方向，内部保留两个后端。

---

## 5.1 Patient scalar activation GGM

对第 `d` 个患者，构造一个长度为 `P` 的 concept activation vector：

\[
x_d=(x_{d1},\ldots,x_{dP}).
\]

其中：

- `d`：第几个患者；
- `j`：第几个 concept；
- `x_dj`：患者 `d` 对 concept `j` 的 scalar activation 强度。

所有患者组成：

```text
[N patients, P concepts]
```

然后使用当前代码已经实现的：

```text
rank-Gaussian / nonparanormal
        ↓
concept covariance
        ↓
edge-weighted graphical lasso
        ↓
concept precision
```

这条线的统计解释很自然：

> 一个患者是一条 `P` 维随机向量观测；图描述不同 concept activation 在患者群体中的条件依赖。

当前代码已经能完成这类学图，但真实 LLM-based activation extractor 尚未实现。

---

## 5.2 Patient-level MNGM

这是目前个人上更有吸引力的 patient 路线。

对第 `d` 个患者，不把 concept 压成 scalar，而是为每个 concept 生成一个 `R` 维 patient-specific vector。

把一个患者的所有 concept vectors 拼起来得到：

\[
H_d\in\mathbb R^{R\times P}.
\]

其中：

- 每一列对应一个 concept；
- 每一列是该患者条件下的 concept vector；
- 一个患者就是一个真正的 matrix replicate。

所有患者形成：

```text
[N patients, R representation dims, P concepts]
```

然后直接使用和 Bootstrap-MNGM 完全相同的 `MNGMEstimator`。

因此两条 MNGM 路线最核心的区别是：

```text
Bootstrap-MNGM：一个 matrix replicate = 一次模型/语料扰动下的 concept embedding matrix
Patient-MNGM：  一个 matrix replicate = 一个患者的 concept representation matrix
```

这使得 patient-MNGM 在“统计样本是什么”这个问题上更自然，但它的 representation extractor 还没落地。

---

## 5.3 已经标记的 patient-specific concept vector 方案

这个组件已经决定先按照 attention 方案实施，但暂缓开发，等 LLM 在全流程中的角色、LoRA 与 graph injection 先讨论清楚。

第一版设想是：

1. 每个患者文本只做一次当前 LLM forward；
2. 得到患者 token-level contextual hidden states；
3. 为固定 concept vocabulary 中的每个 concept 准备 concept representation/prototype；
4. 用 concept representation 与患者 token hidden states 的相似性产生 attention 权重；
5. 对患者 token hidden states 加权求和，得到该患者关于这个 concept 的向量。

如果患者 `d` 的第 `l` 个 token hidden state 记为 `h_dl`，concept `j` 的 prototype 记为 `e_j`，那么可以先定义 attention：

\[
a_{dlj}=\operatorname{softmax}_l\left(\frac{\operatorname{sim}(h_{dl},e_j)}{\tau}\right),
\]

其中：

- `τ`：温度参数；
- `sim`：相似度，例如 cosine similarity；
- `a_dlj`：concept `j` 对患者第 `l` 个 token 的注意力权重。

再得到 patient-specific concept vector：

\[
h_{dj}=\sum_l a_{dlj}h_{dl}.
\]

所有 `P` 个 concept vectors 组成该患者的 `R × P` matrix，送入 patient-MNGM。

这套向量也可以进一步压成 scalar activation，从而为 patient scalar GGM 提供输入。因此未来两个 patient 后端可能共用同一次 LLM forward。

这一组件目前是 **明确 mark、尚未实现**。

---

## 5.4 Patient 线的统计优势与需要谨慎的地方

### 优势

一个患者/一次 encounter 作为一条统计观测，比“embedding coordinate 或训练 snapshot 当独立样本”更自然。

尤其 patient-MNGM 可以同时保留：

- concept axis precision；
- representation axis precision；

但 matrix replicate 由真实患者构成。

### 需要谨慎

图的含义不是“真实生理因果图”，而是：

> **当前 LLM 测量出来的 patient-level concept states / representations 的条件依赖图。**

它是 representation-dependent graph。

如果同一患者有多次 encounter，不能简单把它们都当完全独立；以后真实医疗实验应考虑按 `subject_id` 划分，或者第一版限制一名患者一个 encounter。

如果 scalar activation 大量饱和在 0/1 或出现大量 ties，rank-Gaussian 会变得不稳定，因此 patient extractor 更适合产生连续 soft score，而不是硬阈值标签。

---

## 5.5 Patient 线目前还没敲定什么

主要包括：

- 固定 concept vocabulary 如何得到；
- scalar activation 的最终定义；
- attention-based patient concept vectors 的具体 hidden layer / prototype 定义；
- graph injection；
- candidate reward 采用 Frozen、short-LoRA 还是 two-stage；
- accepted graph 后 LoRA 的具体训练规则；
- graph / LLM representation 多久 refresh 一次。

所以 patient 线目前应被视为：

> **统计后端已经能跑，LLM 数据生成与下游闭环尚待设计。**

---

# 6. 两条主线的对照

| 维度 | Bootstrap-MNGM | Patient scalar GGM | Patient-level MNGM |
|---|---|---|---|
| 一个统计样本 | 一个扰动 version 的 concept matrix | 一个患者的 scalar activation vector | 一个患者的 concept vector matrix |
| 输入 shape | `[B,R,P]` | `[N,P]` | `[N,R,P]` |
| concept 是什么 | 当前模型 input embedding 中的 concept vector | 患者对 concept 的 scalar activation | 患者特异的 concept vector |
| 是否双轴建模 | 是 | 否 | 是 |
| concept precision | 有 | 有 | 有 |
| representation precision | 有 | 无 | 有 |
| LLM 表示模块 | 已基本确定 | 未实现 | 已 mark attention 方案 |
| 统计样本自然性 | 较弱，replicate 来自扰动/训练轨迹 | 强 | 强 |
| 与最早 GRIAN/MNGM 连续性 | 最高 | 较低 | 高 |
| 当前成熟度 | 最高 | 中等 | 中等偏低，但研究吸引力高 |

---

# 7. 当前共同 RL 框架的实现状态

## 7.1 已实现

- edge-wise penalty matrix；
- `increase / decrease / keep` 动作；
- 每次 penalty 改动后完整重新求图；
- vector weighted GGM；
- MNGM alternating solver；
- Random / Greedy / GRPO policy；
- GRPO 组内 reward 标准化；
- PPO-like clipping；
- reference KL；
- entropy regularization；
- policy gradient clipping；
- reward 与 acceptance 分离；
- 独立 final test 入口；
- 三种 graph sample semantics；
- `--graph-data` 与 downstream task data 解耦。

## 7.2 当前 downstream 只是工程代理，不是最终 LLM

当前代码中的 `DownstreamEvaluator` 仍然是 graph-gated logistic regression：

- concept scalar main effects；
- 所有 pairwise interaction slot；
- graph support 决定哪条 interaction 被打开；
- reward utility 当前主要是 reward split 上的 `-LogLoss`。

它的作用是验证：

```text
penalty action
→ statistical graph re-solve
→ graph changes downstream behavior
→ reward
→ policy update
```

这条软件闭环能够工作。

它 **不是** 最终研究方法中的 Qwen evaluator。

---

# 8. 已经做过的 QSAR 验证及其含义

QSAR Biodegradation 数据：

```text
N = 1055
P = 41
RB = 356
NRB = 699
```

单次真实数据实验中：

### 初始图

```text
initial lambda = 0.8
edges = 49
reward LogLoss ≈ 0.37916
test LogLoss ≈ 0.31336
test AUROC ≈ 0.93400
```

### Greedy

```text
reward LogLoss ≈ 0.35931
reward improvement ≈ 0.01985

test LogLoss ≈ 0.31244
test improvement ≈ 0.00092

edges 49 → 51
accepted 6 / 20
```

### GRPO

```text
reward LogLoss ≈ 0.36682
reward improvement ≈ 0.01234

test LogLoss ≈ 0.30545
test improvement ≈ 0.00791
test AUROC ≈ 0.93753

edges 49 → 53
accepted 5 / 20
```

这些结果只能说明：

> 真实数据上 reward 不是完全 flat，当前 RL→penalty→graph→task 的工程闭环有信号。

不能据此宣称 GRPO 优于 Greedy，因为：

- 只有单 seed；
- Greedy 与 GRPO 的 candidate coverage 并不完全公平；
- 当前 candidate builder 太宽；
- reward 很稀疏；
- 当前 downstream 只是 support-gated logistic proxy。

当时 GRPO 320 个 candidate 中约 306 个 reward 接近 0，主要原因是很多 penalty 改动并没有改变 graph support，导致 downstream features 完全一样。

因此后续 LLM 版仍应考虑更精细的 candidate pool：

```text
current edges
+ near-threshold nonedges
+ unstable edges
+ historically useful edges
```

而不是每轮在所有 `P(P-1)/2` 条边上等价搜索。

---

# 9. 当前代码包结构

核心文件：

```text
graph_mvp/
├── estimators.py          # VectorGGMEstimator + MNGMEstimator
├── graph_data.py          # 三种 graph-data contract
├── weighted_glasso.py     # edge-weighted GLASSO 内层求解器
├── environment.py         # penalty action → full re-solve
├── policy.py              # Random / Greedy / GRPO
├── reward.py              # task + graph/statistical reward
├── downstream.py          # 当前 logistic proxy evaluator
├── runner.py              # round/candidate/policy/acceptance orchestration
├── types.py               # immutable graph state/snapshot/candidate types
├── config.py
└── cli.py
```

重要文档：

```text
README.md
VERIFICATION.md
docs/THREE_GRAPH_ESTIMATORS.md
docs/RESEARCH_HANDOFF_2026-09-22.md
```

历史参考脚本已放入 handoff 包的：

```text
legacy_reference/
├── NMF_PK3rd.py
└── bigraph_phy.py
```

历史脚本不应直接替换当前 solver；它们用于追踪 Bootstrap 表示生成和早期 MNGM 的设计来源。

---

# 10. 当前明确“不要再回退”的决定

以下内容已经讨论清楚，下一轮对话默认保持，除非有新的实验或明确理由修改。

### Bootstrap 线

1. 一次扰动不是全 corpus 全局打乱，而是**每篇医疗文本内部句子独立重排**。
2. 固定保留 50 个 shuffled versions。
3. Bootstrap 表示生成阶段需要 LLM/LoRA 训练。
4. 该阶段使用 **ordinary causal LM loss**。
5. concept representation 保留旧方案：**input embedding token 均值**。
6. 暂不改为 contextual hidden state。
7. 不额外加 PCA，不强制 64 维。
8. 50 套 concept embedding matrices 全部保存并进入 MNGM。
9. MNGM 使用当前 v0.1.2 规范化求解器。
10. GRPO 只控制 concept edge penalties，不控制 representation-axis edge penalties。

### Patient 线

1. 同时保留 patient scalar GGM 和 patient-level MNGM。
2. patient-MNGM 与 bootstrap-MNGM 共用同一个 MNGM solver。
3. attention-based patient-specific concept vector extractor 已经 mark，后续第一版就按这个思路实现。
4. 该组件暂缓，先把整个 LLM / graph injection / LoRA 时序讨论清楚。

---

# 11. 当前最需要继续讨论的问题

新对话建议不要一次性展开所有问题，继续逐个敲定。

## Bootstrap 线优先级最高

### 第一优先级：graph 如何进入 Qwen

必须在以下方案中确定第一版：

```text
text serialization
soft graph tokens
GAT / graph encoder → K/V memory
```

同时要定义 patient-specific local subgraph 的抽取规则。

### 第二优先级：candidate graph 的 LLM reward 机制

正式选择：

```text
Frozen-LLM
Short-LoRA
Two-stage: frozen screening + short-LoRA re-ranking
```

这个决定直接控制一轮 RL 的训练次数和总成本。

### 第三优先级：accepted graph 后的 LoRA

需要明确：

- 用什么 downstream task loss；
- 训练多少；
- LoRA target modules；
- bootstrap LM adapter 与 downstream task adapter 是同一个还是两个；
- graph-conditioned input 的具体格式。

### 第四优先级：representation refresh 时序

需要决定：

```text
每接受一次图 → 重新生成 50 个 bootstrap matrices
```

还是：

```text
累计若干 accepted updates → 批量 refresh
```

### 第五优先级：补齐 version-specific embedding snapshot 生成器

当前已经确定 50 套都要存，但还需把连续 LoRA 轨迹中“何时保存第 b 套 matrix”的工程逻辑写死，并验证 PEFT embedding 的有效值读取方式。

---

# 12. Patient 线下一阶段再讨论的问题

等 Bootstrap 线的 LLM 闭环完全定型后，再处理：

1. 固定 concept vocabulary；
2. patient scalar activation 的定义；
3. attention-based patient-specific concept vector extractor；
4. patient-MNGM 的真实医疗数据输入；
5. patient-level graph 的下游 graph injection；
6. 是否沿用 Bootstrap 线完全相同的 candidate reward / accepted-LoRA 时序。

理想状态是：

> 三种统计学图模式只在“如何形成 graph estimation samples”这一段不同，从 global concept graph 开始，后面的 RL、local graph、LLM、reward、acceptance 和 LoRA 尽量使用同一套协议。

这样论文才能把主要差异归因于：

> **统计样本定义 + graph estimator**

而不是不同的 LLM 调用方式造成混杂。

---

# 13. 新对话可以直接从哪里继续

建议下一轮直接用下面这句话开始：

> “我们已经把 Bootstrap-MNGM 的前半段定为：文档内部句序扰动 → 50 fixed versions → ordinary LM LoRA → input-embedding concept mean → 保存 50 套 matrices → v0.1.2 MNGM。求解器不再讨论。现在从 ‘global concept graph 怎么进入 Qwen’ 开始逐项敲定。”

如果希望先补工程缺口，也可以从：

> “先把 continuous LoRA 训练里 50 个 version-specific embedding snapshots 的保存时序和 PEFT effective embedding 读取方式写死。”

开始。

---

# 14. 最终状态标签

当前项目可以简化为：

```text
统计求解层：        已基本完成
GRPO 工程闭环：     已完成 MVP
Bootstrap 表示定义：已基本敲定
Bootstrap LLM 时序：部分未定
Patient scalar 图： 求解器已完成，LLM extractor 未实现
Patient MNGM：      求解器已完成，attention extractor 已 mark 未实现
真实 LLM downstream：未接入
Graph injection：   未定
最终医疗实验：      尚未开始
```

这就是截至 2026-09-22 的交接状态。
