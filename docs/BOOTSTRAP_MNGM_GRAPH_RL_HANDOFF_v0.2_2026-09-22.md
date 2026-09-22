# Bootstrap-MNGM + Task-Guided Graph RL：当前完整方案与下一步实验交接

> 日期：2026-09-22  
> 文档版本：v0.2  
> 当前代码基线：`task_guided_graph_mvp_v0.1.2`  
> 目标：作为下一轮对话/开发窗口的唯一交接入口，记录当前已经敲定的 Bootstrap-MNGM 主线、尚未实现的接口升级、第一套真实医疗实验（MIMIC-IV-ED）以及明确不再采用的旧设计。

---

## 0. 当前结论：Bootstrap 主线已经基本闭合

当前研究主线已经从“统计图 + RL + LLM”这一宽泛框架，收敛成一条相对清晰、可实现、可消融的链路：

```text
医疗语料
  ↓
固定 50 个文档内句序扰动版本
  ↓
Bootstrap representation-generation LoRA
  ↓
50 套 concept embedding matrices H^(1:50)
  ↓
【从这里开始永久冻结 H^(1:50)】
  ↓
MNGM + edge-wise penalty matrix Λ
  ↓
全局 concept graph G
  ↓
对具体病例 x 计算 concept activation a(x)
  ↓
sample-conditioned activated graph G_x
  ↓
1-layer weighted message passing
  ↓
query-attention pooling
  ↓
K 个 soft graph tokens
  ↓
Qwen + task LoRA
  ↓
病例级下游任务
  ↓
task reward
  ↓
GRPO 只更新 edge-wise penalty policy
  ↓
修改 Λ → 完整重新求 MNGM → 新候选图
  ↓
独立 validation acceptance
  ↓
若接受：固定新图，短暂更新 task LoRA + SoftGraphTokenizer
  ↓
再次冻结模型，进入下一轮 Graph Phase
```

这条线当前最重要的设计原则是：

1. **Bootstrap 统计证据固定。** RL 阶段不再刷新 50 套 representation。
2. **RL 不直接编辑邻接矩阵或 precision matrix。** Policy 只能改边级稀疏惩罚 `Lambda_ij`。
3. **每次 penalty 改动都必须完整重求 MNGM。** 图变化是统计求解器的结果，不是 policy 直接赋值。
4. **候选图比较时冻结 Qwen task LoRA 和 graph tokenizer。** Candidate 之间唯一系统性变化应是 graph。
5. **只有 accepted graph 才触发 task model adaptation。** 如果图被拒绝，task LoRA 和 tokenizer 不更新。
6. **global graph 不等于 global graph token。** 图是全局的，但 graph tokens 必须随病例的 concept activation 改变。

---

# 1. 当前代码基线与已经实现的统计后端

当前代码包：

```text
task_guided_graph_mvp_v0.1.2
```

已验证测试：

```text
45 passed
```

当前已经统一支持三种统计样本语义：

| 模式 | 一个统计样本 | 输入 shape | 求解器 |
|---|---|---:|---|
| `patient_activation_vector` | 一个患者的 concept scalar activation vector | `[N, P]` | nonparanormal weighted GGM |
| `patient_concept_matrix` | 一个患者的 concept vector matrix | `[N, R, P]` | nonparanormal MNGM |
| `bootstrap_concept_embedding_matrix` | 一个 bootstrap/扰动版本的 concept embedding matrix | `[B, R, P]` | nonparanormal MNGM |

其中：

- `N`：患者/病例数；
- `B`：bootstrap/扰动版本数，当前固定为 50；
- `R`：representation dimension；
- `P`：concept 数；当前真实医疗版本预期约 800 个节点。

Bootstrap 线与 patient-level MNGM 共用同一个 `MNGMEstimator`。MNGM 同时估计：

- concept precision：`P × P`；
- representation precision：`R × R`。

**GRPO 只控制 concept 轴的 edge-wise penalty matrix。** representation precision 仍由当前 MNGM alternating solver 自行估计。

---

# 2. Bootstrap representation generation：已经锁死的部分

## 2.1 50 个扰动版本

当前采用旧项目中已经存在的文档内句序扰动逻辑：

- 每篇文档内部切句；
- 只在同一篇文档内部打乱句子顺序；
- 不把不同文档的句子混在一起；
- 固定随机种子，生成 50 个固定版本；
- 这 50 个版本在整个研究中保持不变。

严格统计术语上，它更接近 sentence-order permutation / contextual perturbation，而不是 textbook iid bootstrap。项目内部仍可沿用“bootstrap version”这一历史命名，但论文中需要精确定义。

---

## 2.2 Concept representation

当前 Bootstrap-MNGM concept 表示仍沿用原代码思想：从模型的有效 input embedding 空间获取 concept embedding。

若 concept `j` 被 tokenizer 切成 `K_j` 个 token，则：

$$
e_j=\frac{1}{K_j}\sum_{k=1}^{K_j}E[\mathrm{token}_{jk}].
$$

当前不再额外加入 PCA，也不强制降到 64 维；embedding 维度是多少，MNGM 就接收多少维。

---

## 2.3 Bootstrap LoRA 的作用

如果只读取完全冻结的 embedding table，句序扰动不会改变 concept embedding。因此 Bootstrap 阶段需要 ordinary causal LM training，使不同版本沿连续训练轨迹产生不同的有效 concept embeddings：

```text
sentence-order perturbation
      ↓
ordinary causal LM loss
      ↓
Bootstrap LoRA / effective embedding changes
      ↓
version-specific concept embedding matrix
```

Bootstrap 阶段只使用普通 causal LM loss：

$$
\mathcal L_{\mathrm{LM}}
=-\sum_t \log p_\theta(x_t\mid x_{<t}).
$$

不再把 MNGM loss、Laplacian loss 与 LM loss 混在一起反向传播。

---

## 2.4 50 套矩阵如何进入 MNGM

每个版本保存一套 concept embedding matrix：

$$
H^{(b)}\in\mathbb R^{R\times P},\qquad b=1,\ldots,50.
$$

组成：

$$
\mathcal H\in\mathbb R^{50\times R\times P}.
$$

这 50 个 matrix replicates 一次性进入当前 `MNGMEstimator`。

**从 RL 阶段开始：**

$$
\boxed{H^{(1)},\ldots,H^{(50)}\ \text{永久固定}}
$$

因此后续统计观测、nonparanormal 变换得到的基础统计量也固定。RL 只通过 `Lambda` 改变稀疏约束。

---

## 2.5 Bootstrap adapter 与 task adapter 分离

Bootstrap representation-generation adapter 记为：

$$
\theta_{\mathrm{boot}}.
$$

下游图条件任务 adapter 记为：

$$
\theta_{\mathrm{task}}.
$$

MVP 明确采用：

$$
\boxed{\theta_{\mathrm{boot}}\neq\theta_{\mathrm{task}}}
$$

在 task-guided RL 阶段：

- `theta_boot` 冻结；
- `H^(1:50)` 冻结；
- task LoRA 可以在 accepted graph 之后更新；
- task LoRA 的变化不能回流到 Bootstrap representation。

这保证不同 graph round 仍在同一套统计证据上比较。

---

# 3. 初始 MNGM 图与 penalty state

设固定统计输入得到当前 covariance/statistical state `S`。对于 concept 轴维护：

$$
\Lambda_t=(\lambda_{ij}^{(t)}),
$$

并由当前 MNGM solver 完整求解 concept precision：

$$
\Theta_t.
$$

concept partial correlation 定义为：

$$
W_{ij}^{(t)}
=
-\frac{\Theta_{ij}^{(t)}}{
\sqrt{\Theta_{ii}^{(t)}\Theta_{jj}^{(t)}}
},\qquad i\neq j.
$$

得到全局 concept graph：

$$
G_t=(V,W_t).
$$

当前图状态必须持久保存：

$$
(S,\Lambda_t,\Theta_t,W_t,G_t).
$$

不能只保存邻接矩阵。

---

# 4. 第一套真实医疗实验：MIMIC-IV-ED

## 4.1 为什么不优先使用 OHSUMED

第一套真正的医疗实验优先定为 **MIMIC-IV-ED v2.2**，而不是 OHSUMED。

原因是 MIMIC-IV-ED 可以在同一个病例级任务里同时提供：

1. **真实临床状态**：triage 时点的 chief complaint、生命体征、pain、acuity；
2. **明确时间顺序**：输入发生在 ED triage，结果发生在 ED 离开/处置时；
3. **病例级下游标签**：ED disposition；
4. **后续诊断信息**：ED discharge ICD diagnosis，可用于后续分析，但不进入第一个任务的输入。

PhysioNet 官方 v2.2 页面说明：MIMIC-IV-ED 包含约 42.5 万次 ED stays，数据来自 2011–2019 年；`triage` 表提供 `chiefcomplaint`、`temperature`、`heartrate`、`resprate`、`o2sat`、`sbp`、`dbp`、`pain`、`acuity`；`edstays` 提供最终 `disposition`；`diagnosis` 表提供 ED discharge ICD-9/10 diagnosis。

---

## 4.2 第一个真实任务：triage → HOME vs ADMITTED

第一套任务锁定为：

$$
\boxed{
\text{ED triage information}
\rightarrow
\text{HOME vs ADMITTED}
}
$$

标签：

$$
y_d=
\begin{cases}
0,&\text{HOME},\\
1,&\text{ADMITTED}.
\end{cases}
$$

MIMIC-IV-ED `disposition` 官方可取值包括：

```text
ADMITTED
ELOPED
EXPIRED
HOME
LEFT AGAINST MEDICAL ADVICE
LEFT WITHOUT BEING SEEN
TRANSFER
OTHER
```

MVP 第一版仅保留：

```text
HOME
ADMITTED
```

其余 disposition 全部排除，不把它们粗暴并到负类或正类。

---

## 4.3 下游输入严格限制为 triage 时已经可知的信息

主输入字段：

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

禁止把以下后验信息作为模型输入：

- ED discharge diagnosis；
- hospital discharge summary；
- disposition 本身；
- ED stay 后续才产生的明确结局文本。

这是为了保证时间方向：

$$
t_{\mathrm{input}}=\text{triage}
<
t_{\mathrm{label}}=\text{ED disposition}.
$$

这样可以避免“用 discharge summary 预测 discharge diagnosis”一类明显的标签泄漏。

### Acuity 的处理

`acuity` 在 triage 时已经存在，因此主实验可以合法使用；但它本身是强临床严重度信号，可能主导 HOME/ADMITTED 预测。

因此建议至少保留一个明确消融：

```text
Full triage input
vs
Triage input without acuity
```

用于判断 graph gain 是否只是在复述 clinician-assigned acuity。

---

## 4.4 病例文本化方式

第一版可以把 triage 结构化字段转换为固定模板文本，例如：

```text
Chief complaint: ...
Temperature: ...
Heart rate: ...
Respiratory rate: ...
Oxygen saturation: ...
Blood pressure: ...
Pain: ...
Acuity: ...
```

这里的原则是：

- 输入字段顺序固定；
- 缺失值使用统一占位方式；
- 不让不同 candidate graph 改变病例文本；
- candidate 之间只改变 graph-conditioned soft tokens。

具体 prompt wording 属于实现细节，不作为方法创新点。

---

## 4.5 数据划分：必须按 patient 分组

MIMIC-IV-ED 中同一个 `subject_id` 可能有多次 ED encounter。为避免同一患者跨 train/test 泄漏，建议按 `subject_id` group split，而不是按 `stay_id` 随机逐行切分。

至少维护：

$$
D_{\mathrm{train}},\quad
D_{\mathrm{graph}},\quad
D_{\mathrm{val}},\quad
D_{\mathrm{test}}.
$$

职责严格区分：

| split | 用途 |
|---|---|
| `D_train` | 训练 task LoRA + SoftGraphTokenizer |
| `D_graph` | Graph Phase 中计算 candidate dense reward |
| `D_val` | accepted / rejected graph 的独立复核 |
| `D_test` | 最终一次性报告，不进入 policy 或 acceptance |

同一个 `subject_id` 不能跨这些 split。

---

# 5. 从 global MNGM graph 到 sample-conditioned graph

全局 graph 是所有病例共享的统计结构，但不同病例不应该得到同一组 graph tokens。

核心定义：

$$
\boxed{
\text{global graph }G
+
\text{sample activation }a(x)
\rightarrow
\text{sample-conditioned graph }G_x
}
$$

---

## 5.1 Concept prototype

对 concept `i`，使用固定 Bootstrap representation 构造 frozen concept prototype。MVP 可取 50 个版本上的平均：

$$
c_i
=
\frac1{B}\sum_{b=1}^{B}H_i^{(b)}.
$$

因为 Bootstrap representations 在 RL 阶段固定，所以 `c_i` 也固定。

---

## 5.2 病例级 concept activation

给定病例文本 `x`，从 Qwen 的 input embedding 空间得到 token embeddings：

$$
E(x)=[e_1,\ldots,e_L].
$$

计算 concept-token cosine similarity：

$$
s_{ij}=\cos(c_i,e_j).
$$

对病例所有 token 做 smooth max / log-sum-exp 聚合：

$$
\tilde a_i(x)
=
\tau\log
\sum_{j=1}^{L}
\exp\left(\frac{s_{ij}}{\tau}\right).
$$

这一步已经作为 MVP 主方向接受。

**尚未完全锁死的细节：** `tilde a_i` 之后采用 sigmoid independent activation 还是 concept-wise softmax。当前更倾向 sigmoid，因为一个病例可以同时高激活多个 concept，不需要强制所有 activation 总和为 1。

---

## 5.3 Sample-conditioned edge weight

设：

$$
D_x=\operatorname{diag}(a_1(x),\ldots,a_P(x)).
$$

当前 global partial-correlation graph 为 `W`，则病例特异图定义为：

$$
\boxed{
W_x=D_xWD_x
}
$$

即：

$$
(W_x)_{ij}
=a_i(x)W_{ij}a_j(x).
$$

第一版不在这里做 hard threshold，以免 penalty 的小变化经过 threshold 后完全无法传到 downstream reward。

---

# 6. SoftGraphTokenizer：MVP 已确定的 architecture

SoftGraphTokenizer 采用尽量短的结构：

$$
\boxed{
\text{1-layer weighted message passing}
+
\text{query attention pooling}
+
\text{linear projection}
}
$$

不在 MVP 中加入多层 GNN、GAT 或复杂 graph transformer。

---

## 6.1 一层 weighted message passing

对 concept node `i`：

$$
m_i(x)
=
\sum_{j\neq i}
(W_x)_{ij}U c_j.
$$

再构造节点状态：

$$
z_i(x)
=
\operatorname{MLP}
\left[
 a_i(x)c_i
\Vert
 m_i(x)
\Vert
 a_i(x)
\right].
$$

因此节点状态同时保留：

- 当前病例对 concept 本身的激活；
- 当前病例激活后的 graph neighborhood message；
- 显式 activation scalar。

---

## 6.2 Query attention pooling

使用 `K` 个 learnable query：

$$
q_1,\ldots,q_K.
$$

每个 soft graph token：

$$
t_k(x)
=
\operatorname{Attention}
(q_k,Z_x,Z_x),
$$

得到：

$$
T_G(x)
=
[t_1(x),\ldots,t_K(x)].
$$

最后投影到 Qwen hidden size，并在 embedding level 作为连续 soft tokens 注入。

MVP 初始建议：

```text
K = 8
```

但 `K=8/16` 属于超参数，不属于方法定义。

---

## 6.3 最终 Qwen 输入

概念上：

```text
[instruction embeddings]
[soft graph token 1]
...
[soft graph token K]
[triage patient text embeddings]
```

因此 graph-conditioned Qwen 计算：

$$
p_{\theta,\phi}(y\mid x,G),
$$

其中：

- `theta`：task LoRA 参数；
- `phi`：SoftGraphTokenizer 参数。

---

# 7. Task LoRA 与 Graph RL：采用块式交替优化

整个 Bootstrap-MVP 不采用“每个 candidate 单独训练一次 LoRA”这一旧方案。

采用：

$$
\boxed{
\text{Graph Search Block}
\leftrightarrow
\text{LLM Adaptation Block}
}
$$

两个 block 内严格冻结对方。

---

## 7.1 Stage 0：初始图上的 graph-conditioned warm-up

先用固定 `G_0` 在 `D_train` 上训练：

$$
\theta_{\mathrm{task}},\phi.
$$

目标：

$$
\mathcal L_{\mathrm{task}}
=
-\sum_{(x,y)\in D_{\mathrm{train}}}
\log p_{\theta,\phi}(y\mid x,G_0).
$$

目的不是优化图，而是先让 Qwen 和 tokenizer 学会利用 soft graph tokens。

warm-up 结束后：

$$
\boxed{\theta_0,\phi_0\ \text{freeze}}
$$

进入 Graph Phase。

---

## 7.2 Graph Phase：冻结 task model，只搜索 graph

在第 `t` 个 Graph Phase：

$$
\theta_t,\phi_t
$$

保持完全冻结。

Policy 基于当前 accepted state 生成多个 edge-penalty action groups：

$$
A_t^{(1)},\ldots,A_t^{(K)}.
$$

每个 candidate 改变当前 `Lambda_t` 的若干边，得到：

$$
\Lambda_t^{(k)}.
$$

然后必须完整重新求 MNGM：

$$
\Lambda_t^{(k)}
\xrightarrow{\text{full MNGM re-solve}}
\Theta_t^{(k)}
\rightarrow
G_t^{(k)}.
$$

在同一 `D_graph` reward panel 上、同一冻结模型下比较候选图。

---

## 7.3 Candidate reward：使用连续任务信号

MVP 不优先使用二值 accuracy 作为 inner-loop GRPO reward，因为它过于离散。

推荐：

$$
R_k
=
\frac1{|B|}
\sum_{(x_n,y_n)\in B}
\log p_{\theta_t,\phi_t}
(y_n\mid x_n,G_t^{(k)}),
$$

其中：

$$
B\subset D_{\mathrm{graph}}.
$$

同一个 GRPO group 的所有 candidate 必须使用同一个 reward batch。

MVP reward 尽量保持纯粹：

```text
task log-likelihood
+ hard invalid-candidate penalty
```

不在第一版里一开始就叠加过多 graph drift / likelihood degradation / sparsity regularization 项。必要时后续再逐项加入。

---

## 7.4 Graph acceptance

一个 Graph Phase 不直接逐步覆盖当前 accepted state，而是在当前 `G_t` 周围做局部搜索。

Phase 结束后得到 proposal：

$$
G_t^\star.
$$

再在完全独立的 `D_val` 上，与当前图 `G_t` 比较。

只有 proposal 在预先规定的 validation criterion 上优于当前图，才：

$$
G_{t+1}=G_t^\star.
$$

否则：

$$
G_{t+1}=G_t.
$$

**若图被拒绝：**

$$
\theta_{t+1}=\theta_t,\qquad
\phi_{t+1}=\phi_t.
$$

即不额外训练 task model。

Acceptance 最终使用 validation log-likelihood、AUROC 或固定主指标中的哪一个，尚需在真实数据实现时最后锁死；但不能在训练中途根据结果临时更换。

---

## 7.5 Accepted graph 后的 task adaptation

如果：

$$
G_{t+1}\neq G_t,
$$

则固定 `G_{t+1}`，只在 `D_train` 上更新：

$$
\theta_{\mathrm{task}},\phi.
$$

不重新训练 Bootstrap adapter，不刷新 50 套 embeddings。

MVP 可以从：

```text
200 optimizer steps
```

作为工程初始值；这一数值是超参数，不是方法定义。

更新完以后重新冻结 task LoRA 和 tokenizer，再进入下一 Graph Phase。

---

# 8. Candidate edge pool：从当前 MNGM 解出发

Policy 不在 32 万条潜在边上盲搜，也不另外调用 LLM 生成候选关系。

对于约 `P=800` 个节点：

$$
\binom{800}{2}=319600
$$

条潜在无向边。

当前决定：candidate pool 完全从当前 MNGM 状态产生：

$$
\boxed{
E_t^{\mathrm{pool}}
=
E_t^{\mathrm{active}}
\cup
E_t^{\mathrm{frontier}}
}
$$

---

## 8.1 Active edges

当前 MNGM graph 中已有的非零边：

$$
E_t^{\mathrm{active}}
=
\{(i,j):\Theta_{t,ij}\neq 0\}.
$$

Policy 可以通过：

- increase `lambda_ij`：增强收缩；
- decrease `lambda_ij`：减弱收缩；

让该关系更容易减弱/消失或被保留/增强。

---

## 8.2 Frontier edges

只允许 active edges 会导致一个问题：被当前 MNGM 稀疏化为 0 的边永远无法重新进入图。

因此需要从 **当前 MNGM 优化状态本身**定义 frontier：

> 已经为零、但位于 active-set 边界附近，稍微降低 penalty 就可能进入图的边。

可优先考虑基于 KKT residual / KKT margin 的 frontier score。

对于标准 weighted graphical lasso 形式，零边处可以考察未惩罚梯度与 `lambda_ij` 的距离；但当前项目使用的是规范化 MNGM alternating solver，因此 **不能直接把标准 graphical lasso 公式硬套入实现**。

实现前必须针对当前 `MNGMEstimator + WeightedGraphicalLasso` 的实际 objective 推导并验证 frontier score。

这是当前仍需落实的一处数学/代码细节。

---

# 9. Policy：从单边 GRPO 升级为 batch-edge GRPO

## 9.1 当前 v0.1.2 已有接口

现有代码已经规范了：

```text
GraphState
  ↓
CandidateBuilder
  ↓
PolicyInput
  ↓
Policy
```

当前 `PolicyInput`：

```python
PolicyInput(
    state_id,
    candidate_edges: tuple[EdgeFeatures, ...],
    global_features,
)
```

当前 `EdgeFeatures` 已包含：

```text
partial_corr
abs_partial_corr
penalty
edge_exists
uncertainty
task_relevance
previous_reward
```

因此 policy 输入接口无需推翻。

---

## 9.2 当前代码需要升级的地方

v0.1.2 仍然是：

```text
one candidate = one edge edit
```

而最新 MVP 决定：

$$
\boxed{
\text{one candidate graph}
=
32\ \text{edge penalty edits}
}
$$

第一版建议固定：

```text
m = 32 edits / candidate
K = 8 candidate action groups / GRPO group
```

后续可做：

```text
m ∈ {8, 16, 32, 64}
```

的 sensitivity analysis。

---

## 9.3 分层 policy：选哪条边 + 往哪个方向改

不直接对 candidate pool 中所有边做三分类，而把 policy 拆成：

1. **selection policy**：选择本轮真正修改的 32 条边；
2. **direction policy**：对每条被选中的边决定 increase / decrease penalty。

未被选中的边自动等价于 keep。

设 candidate pool 为：

$$
E_t^{\mathrm{pool}}
=
\{e_1,\ldots,e_M\}.
$$

Policy 先选择：

$$
S_t=\{e_{i_1},\ldots,e_{i_{32}}\}.
$$

然后为每条选中边输出：

$$
d_e\in\{\text{increase},\text{decrease}\}.
$$

完整 joint action：

$$
A_t=\{(e,d_e):e\in S_t\}.
$$

联合概率：

$$
\pi_\psi(A_t\mid s_t)
=
\pi_\psi^{\mathrm{sel}}(S_t\mid s_t)
\prod_{e\in S_t}
\pi_\psi^{\mathrm{dir}}(d_e\mid e,s_t).
$$

---

## 9.4 Shared edge scorer

Policy 不直接记忆 edge ID，而对所有候选边共享一个 edge network：

$$
h_{ij}=\operatorname{MLP}_\psi(f_{ij}).
$$

两个 head：

$$
u_{ij}=w_{\mathrm{sel}}^\top h_{ij}
$$

用于 selection；以及：

$$
[z_{ij}^{+},z_{ij}^{-}]
=W_{\mathrm{dir}}h_{ij}
$$

用于 increase / decrease direction。

MVP 不需要用另一个 LLM 或复杂 GNN 做 policy。

---

## 9.5 Task-aware edge feature

当前最有价值的 task feature 是病例级 activation 与当前模型错误的结合。

普通共同激活：

$$
q_{ij}^{\mathrm{act}}
=
\frac1N
\sum_n a_i(x_n)a_j(x_n).
$$

更推荐 error-weighted activation：

$$
\boxed{
q_{ij}^{\mathrm{err}}
=
\frac1N
\sum_n
\ell_n\,a_i(x_n)a_j(x_n)
}
$$

其中：

$$
\ell_n=-\log p(y_n\mid x_n,G_t).
$$

它表达：

> concept `i,j` 是否经常在当前模型做得较差的病例中共同被激活。

这个 feature 只帮助 policy 分配“搜索预算”，不直接决定图结构；最终边是否存在仍由 MNGM 完整重求决定。

建议新版 edge feature 至少包含：

```text
partial correlation
abs(partial correlation)
current lambda_ij
active/frontier indicator
KKT/frontier score
node degrees
uncertainty (if available)
task relevance q_ij^err
historical action/reward summary
```

---

## 9.6 32 条边如何采样

MVP 使用 stochastic sampling without replacement，而不是 deterministic Top-32。

根据 selection logits：

$$
p(e)
=
\operatorname{softmax}(u_e/\tau_\pi).
$$

顺序抽取 32 条，每抽一条后 mask 掉，再从剩余 pool 中采下一条。

可视为 Plackett-Luce 式 sequential sampling。

对于被选中边，再根据 direction head 采样 increase / decrease。

---

## 9.7 Penalty 更新

继续保留乘法更新以保证 penalty 为正：

$$
\lambda'_{ij}
=
\begin{cases}
\rho\lambda_{ij}, & \text{increase},\\
\lambda_{ij}/\rho, & \text{decrease}.
\end{cases}
$$

MVP 初始建议：

```text
rho = 1.1
```

这只是初始工程值。

因为一次 candidate 同时改约 32 条 penalty，且每次都完整重新求 precision，其他未直接操作的 graph edges 也可能随联合最优化发生变化。

---

# 10. GRPO：整图 reward，trajectory-level advantage

一个 action group 中有 32 个 edge edits，但最终只有一个 candidate graph reward。

MVP 不做复杂 edge-level credit redistribution。

对于同一 state 采样 `K` 个 candidate graphs：

$$
A^{(1)},\ldots,A^{(K)}
$$

得到：

$$
R_1,\ldots,R_K.
$$

组内标准化：

$$
\hat A_k
=
\frac{R_k-\bar R}{\sigma_R+\epsilon}.
$$

第 `k` 条 trajectory 中的 32 个 edit decisions 共用同一个 advantage `A_k`。

这和语言模型 GRPO 中“一整个 response 一个 reward，但每个 token 的 log-prob 都参与更新”类似。

因此 MVP 可以先采用 trajectory/group-level reward，不必先解决精细的单边信用归因。

---

# 11. Graph Phase 内的 state 处理

一个 Graph Phase 内，当前 accepted graph state 固定：

$$
(S,\Lambda_t,\Theta_t,G_t,\theta_t,\phi_t)
$$

都不变；只有 policy 参数 `psi` 在 GRPO update 中变化。

即每个 rollout 都从同一个：

$$
\Lambda_t
$$

出发生成局部 candidate，而不是：

```text
candidate 1 被接受到环境
→ 再从 candidate 1 接着改
→ candidate 2
→ ...
```

Graph Phase 是在一个固定 accepted state 周围进行局部搜索。

Phase 结束后才产生一个 proposal 去 `D_val` 做 acceptance。

---

# 12. 第一版真实医疗实验的主要对比与消融

## 12.1 下游任务基线

至少需要：

```text
Qwen / task LoRA without graph
Initial MNGM graph + graph tokens
Task-guided learned graph + graph tokens
```

用于回答：初始统计图本身是否有增益，以及 task-guided graph adaptation 是否进一步提升。

---

## 12.2 防止 soft graph token 退化成普通 soft prompt

这是核心消融。

需要比较：

```text
No Graph
Free Soft Prompt (same K tokens, no graph)
Random / shuffled graph
Initial MNGM graph
Task-guided learned graph
```

如果 learned graph tokens 只相当于多了 `K` 个可训练 continuous vectors，则方法无法证明 graph structure 本身有价值。

---

## 12.3 Sample conditioning 消融

至少比较：

```text
Static global graph tokens
vs
Sample-conditioned activation graph tokens
```

以及：

```text
activation only, no message passing
vs
activation + weighted graph message passing
```

用于证明收益来自病例特异的 graph-conditioned structure，而不是固定 domain prompt。

---

## 12.4 MIMIC 输入消融

建议至少：

```text
Full triage input
vs
Triage without acuity
```

必要时进一步报告只用 chief complaint / 只用 structured vitals 的简单基线。

---

## 12.5 Policy / search 消融

后续可测试：

```text
m = 8 / 16 / 32 / 64 edits per candidate
```

以及：

```text
Random edge edits
Greedy coordinate edits
Single-edge GRPO
Batch-edge GRPO
```

这些属于第二阶段实验，不影响 MVP 首轮实现。

---

# 13. 评价指标

对于 HOME vs ADMITTED 二分类，最终 `D_test` 至少报告：

```text
AUROC
AUPRC
Accuracy
F1
NLL / log loss
```

考虑到可能存在类别不平衡，不建议只看 Accuracy。

GRPO inner reward 优先用连续 log-likelihood / NLL；最终 paper 表格则报告上述完整任务指标。

---

# 14. 与旧《面向下游任务的强化学习图迭代框架》的关系

旧框架的核心仍保留：

1. policy 不直接改 adjacency；
2. policy 改 edge-wise penalty；
3. 每次动作后完整重求 penalized likelihood / MNGM；
4. policy update 与 environment acceptance 分离；
5. graph state 持久保存 `Lambda + Theta + graph`；
6. reward 不允许读取 test set；
7. 求解失败、非正定、NaN 等必须成为 invalid candidate。

但以下旧设计在 Bootstrap-MVP 中明确 **不再采用**：

### 旧设计 A：每个 candidate 都执行 short-LoRA 再比较

现改为：

```text
candidate evaluation = frozen task LoRA + frozen tokenizer
```

Candidate 之间唯一系统性变化是 graph。

### 旧设计 B：accepted graph 后重新刷新 Bootstrap representations

现改为：

```text
H^(1:50) fixed forever during task-guided RL
```

Task LoRA 与 Bootstrap LoRA 分离。

### 旧设计 C：graph injection 只写成抽象 local subgraph/gate

现已具体化为：

```text
sample activation
→ W_x = D_x W D_x
→ one-layer weighted message passing
→ query attention pooling
→ soft graph tokens
```

---

# 15. 当前代码接口：哪些保留，哪些要升级

## 15.1 保留

当前 v0.1.2 已有：

```text
GraphSnapshot
GraphState
EdgeFeatures
PolicyInput
PolicyExperience
WeightedGraphicalLasso
MNGMEstimator
GRPOPolicy
acceptance / runner 骨架
```

`PolicyInput` 的结构化接口继续保留：policy 不直接访问 solver、raw patient dataset 或 test set。

---

## 15.2 需要升级

### CandidateBuilder

当前实现遍历全部上三角边：

$$
P(P-1)/2.
$$

需要改成：

```text
active MNGM edges
+
KKT/frontier edges
```

### ActionRecord

当前：

```text
one ActionRecord = one edge edit
```

需要保留单边 action 作为最小单元，并增加 `ActionGroup` / trajectory：

```text
one candidate = 32 EdgeAction records
```

### PolicyExperience

当前是 single-edge experience；需要升级为一条 candidate trajectory 包含 32 个 actions/log-probs，但共享一个 graph reward。

### GRPOPolicy.sample

当前 action space 为 `2E+1` 的单边 proposal；需要升级为：

```text
sequential edge selection without replacement
+
direction sampling
```

### Downstream evaluator

当前 QSAR logistic regression evaluator 只是工程代理；需要接入：

```text
MIMIC-IV-ED preprocessing
Qwen task model
sample activation
SoftGraphTokenizer
dense task reward
```

---

# 16. 正式跑真实实验前必须完成的工程 sanity checks

## 16.1 Bootstrap snapshot 时点

当前仍需明确连续 LoRA training 中 50 个 version-specific embedding matrices 的读取/保存时点。

这不是方法问题，但必须保证 50 套 `H^(b)` 的语义一致且可复现。

---

## 16.2 PEFT embed_tokens 的有效 embedding

需要验证：

```python
model.get_input_embeddings().weight
```

是否真正包含 `embed_tokens` LoRA 的有效增量。

若 LoRA 增量只在 forward 支路生效，则不能直接把 frozen base `.weight` 当成训练后的 concept embedding。

需要比较：

```text
LoRA before / after
same concept token
direct .weight
vs
actual embedding-module forward output
```

必要时使用 module forward 或临时 merge adapter 后读取。

---

## 16.3 MNGM frontier score

必须基于当前真实 solver 推导并实现 active-set frontier / KKT score，而不是直接复制标准 Graphical Lasso 的公式。

---

## 16.4 Activation normalization

`logsumexp` 聚合已经定；后续 sigmoid vs softmax 需要用小规模 MIMIC smoke experiment 选定，当前更倾向 sigmoid。

---

## 16.5 Acceptance 主指标

需要在实验开始前锁死，例如 validation NLL 或 AUROC，不能训练中途根据结果改目标。

---

# 17. 推荐的实际开发顺序

当前不再继续扩展方法链，按以下顺序落地：

### Step 1：MIMIC-IV-ED preprocessing

实现：

```text
triage + edstays
→ HOME/ADMITTED cohort
→ subject-level train/graph/val/test split
→ fixed triage text template
```

先统计：

- 样本数；
- HOME/ADMITTED 比例；
- chief complaint 缺失率；
- vitals/pain/acuity 缺失率；
- 每个 subject 的 encounter 数。

### Step 2：实现 sample activation + SoftGraphTokenizer

先固定初始 `G_0`，不接 RL，验证：

```text
x + G_0
→ a(x)
→ W_x
→ graph tokens
→ Qwen
```

能够端到端训练 HOME/ADMITTED。

### Step 3：完成 initial graph warm-up

训练 task LoRA + tokenizer，获得冻结的：

$$
\theta_0,\phi_0.
$$

### Step 4：升级 CandidateBuilder 与 batch-edge ActionGroup

实现：

```text
active + frontier pool
→ 32-edge action group
```

### Step 5：升级 GRPO policy

实现：

```text
selection head
+
direction head
+
without-replacement sampling
+
trajectory-level advantage
```

### Step 6：接 frozen candidate evaluator

```text
candidate Λ
→ full MNGM re-solve
→ graph
→ sample-conditioned soft graph tokens
→ frozen Qwen
→ dense reward
```

### Step 7：接 validation acceptance + task adaptation

形成完整 outer loop。

---

# 18. 当前仍保留但暂缓的第二条线：Patient-level MNGM

Patient-level MNGM 仍然保留为后续研究线：

$$
H_d\in\mathbb R^{R\times P}
$$

一个真实病例作为一个 matrix replicate，统计解释比 Bootstrap replicate 更自然。

已经讨论过的 patient-specific concept representation 是：

$$
h_{dj}
=
\sum_l
\operatorname{softmax}_l
\left(
\frac{\operatorname{sim}(h_{dl},e_j)}{\tau}
\right)
 h_{dl}.
$$

但当前优先级明确：

> **先把 Bootstrap-MNGM + MIMIC-IV-ED task-guided graph RL 完整跑通，再回头发展 patient-level MNGM。**

这样避免同时维护两条尚未闭环的 LLM pipeline。

---

# 19. 一句话方法定义

当前 Bootstrap 主线可以概括为：

> 在固定的 Bootstrap-MNGM 统计证据上，用病例级下游任务 reward 通过 GRPO 学习 edge-wise sparsity-penalty policy；每个 policy action 只改变候选 concept edge 的稀疏惩罚，新的 graph 必须由 MNGM 完整重求；全局 graph 再结合当前病例的 concept activation 生成 sample-conditioned soft graph tokens，并注入冻结或交替适配的 Qwen。

核心算法链：

$$
\boxed{
H^{(1:50)}_{\mathrm{fixed}}
\rightarrow
\Lambda
\rightarrow
\mathrm{MNGM}
\rightarrow
G
\rightarrow
G_x
\rightarrow
\mathrm{SoftGraphTokens}
\rightarrow
\mathrm{Qwen}
\rightarrow
R_{\mathrm{task}}
\rightarrow
\mathrm{GRPO}
\rightarrow
\Lambda'
}
$$

其中 task model 与 graph policy 按 block coordinate / alternating schedule 更新，而 Bootstrap statistical evidence 始终不更新。

---

# 20. 下一窗口直接从哪里继续

下一步不需要再讨论 Bootstrap 主框架。

优先进入实现阶段：

1. **读取/准备 MIMIC-IV-ED 数据，确定 HOME/ADMITTED cohort 与 subject-level split；**
2. **实现 sample activation + SoftGraphTokenizer；**
3. **在固定初始 MNGM graph 上完成 Qwen task warm-up；**
4. **然后升级现有 single-edge GRPO policy 为 active+frontier 的 32-edge batch policy。**

如果实现过程中需要继续讨论，优先解决的剩余问题是：

```text
A. activation 的最终 normalization
B. MNGM-specific KKT/frontier score
C. acceptance 主指标
D. 真实 MIMIC 输入缺失值与 prompt 模板
E. Bootstrap embedding snapshot / PEFT embed_tokens sanity check
```

除此之外，当前不建议再给方法链增加新的模块。

---

# 21. 数据集参考

1. Johnson A, Bulgarelli L, Pollard T, Celi LA, Mark R, Horng S. **MIMIC-IV-ED (version 2.2)**. PhysioNet, 2023. DOI: `10.13026/5ntk-km72`.
2. MIMIC-IV-ED v2.2 官方说明：约 425,000 次 ED stays；`triage` 表提供 chief complaint、生命体征、pain、acuity；`edstays` 提供 disposition；`diagnosis` 提供 ED discharge ICD diagnoses。
3. 内部设计依据：`面向下游任务的强化学习图迭代框架(1).docx`，其中保留 penalty-policy / full re-solve / acceptance 分离等原则；本文已经明确记录相对旧框架的修改。

