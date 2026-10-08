# v0.6.3 长文本候选池与大规模学图实验

本入口基于已完成的 v0.6.2 实验。默认使用与该运行完全相同的 2048 篇训练专利、800 个概念、冻结 Qwen3-Embedding-0.6B 和 64 维静态投影。它只构建 H 并拟合 MNGM 图，不加载 GLM，不运行任务训练、LoRA、GRPO、奖励/验证/测试检索评估。

## 服务器启动

从 GitHub 拉取分支。激活已有 `graph_rl` 环境后执行：

```bash
cd /laijizheng/task_guided_graph_mvp_v0.3.0_2026-09-22
git fetch origin
git switch --track origin/feat/patent-longtext-graph-only
bash scripts/run_patent_longtext_graph_only.sh
```

如果本地已有该分支，使用 `git switch feat/patent-longtext-graph-only`，再执行 `git pull --ff-only`。入口记录实际 Git commit 和关键代码校验和，并验证已完成 v0.6.2 的输入与缓存校验和，发现数据不匹配就停止。首次运行自动下载所需授权年份的三类长文本文件到独立原始数据目录；下载校验和来自 Zenodo 文件元数据，支持中断下载续传。GPU 建缓存或拟图中断不支持部分缓存/阶段续跑，需要另选新的输出目录；完整下载的原始 ZIP 可以复用。

准备阶段下载失败后，保留原运行日志和原始数据目录。短的 `.part` 文件从已下载字节续传；完整但校验和不匹配或超长的 `.part` 文件改名为 `.part.invalid.<时间戳>` 保留后重新下载。完整 ZIP 必须通过原始元数据的大小和校验和检查才会使用；已通过检查的 ZIP 不会重复下载。HTTP 206 必须匹配续传位置和元数据总大小，成功的分段响应可以连续续传，传输/校验失败最多重试三次。错误日志包含实际字节数和预期大小/校验和。

更新分支后，使用新的实验输出目录和日志名重启（旧启动脚本的固定日志/PID 名会阻止重复启动）：

```bash
git switch feat/patent-longtext-graph-only
git pull --ff-only
mkdir -p outputs
run_name="patent_longtext_graph_n2048_b4096_k6_v063_retry_$(date +%Y%m%d_%H%M%S)"
nohup python -u -m scripts.run_patent_longtext_graph --output "outputs/$run_name" > "outputs/$run_name.launch.log" 2>&1 < /dev/null &
tail -f "outputs/$run_name.launch.log"
```

这会从准备阶段重新运行，复用已下载的原始 ZIP/部分下载；不会恢复旧的模型缓存或拟图阶段。报告文件名相应变为 `outputs/${run_name}_reports.tar.gz`。

日志：

```bash
tail -f outputs/patent_longtext_graph_v063.launch.log
tail -f outputs/patent_longtext_graph_n2048_b4096_k6_v063/prepare.log
tail -f outputs/patent_longtext_graph_n2048_b4096_k6_v063/build.log
tail -f outputs/patent_longtext_graph_n2048_b4096_k6_v063/graph.log
```

运行结束，把 `outputs/patent_longtext_graph_n2048_b4096_k6_v063_reports.tar.gz` 发回分析。无需传模型、原始全文 ZIP 或 H 缓存。流水线失败也会打包已产生的诊断和失败日志，检查 `experiment_manifest.json` 的状态。

## 长度与数据处理

每篇候选文本最多 4096 个 Qwen tokens（不计 tokenizer 特殊 token；包含原有标题、摘要、章节标题与间隔）。原有标题和摘要全部保留。剩余预算按概要 20%、权利要求 25%、详细说明 55% 分配，短或缺失章节的份额分给其他章节。

超过预算的章节按原文句子/128-token 片段，在开头、结尾及中间不同位置均匀保留，最终恢复原文顺序，不仅截取开头，也不引入另一篇专利。该策略是预算下的覆盖采样，未宣称语义相关性、正确性或最优性。权利要求按 `claim_sequence` 排序，未额外假定独立权利要求标签。全文与训练样本按授权专利号和授权年份精确匹配。每篇完整来源 token 数、选入 token 数、缺失字段及原文位置均有记录。

来源均为 2024-12-31 同快照：

- 发明概要：https://zenodo.org/records/15062198
- 权利要求：https://zenodo.org/records/15062183
- 详细说明：https://zenodo.org/records/15062212

附图说明此次不纳入，因为同快照具体归档下载地址尚未核实。

候选文档再按现有 128-token 分片切分，由原有 prototype 余弦排名按概念独立选择 top-6，联合输入最多 1024 tokens，编码 batch 为 8。实际长度增长仍可能增加显存；运行记录 Qwen 构建阶段 CUDA 峰值。不改变数学定义 `H_d[:,c] = W @ F(concept_c, evidence_dc)`，不减 prototype，不加 residual、强度门控、输出列 L2 或消除相似度的损失。

## 执行与诊断

初始 Glasso 达到迭代上限后，可以用 `scripts.diagnose_patent_glasso_convergence`
对完整 H 缓存单独检查收敛趋势。它校验缓存和原配置，仅增加数值迭代预算，
保留 λ=0.8、B=I、rho、残差和 KKT 标准；不重新做 GPU 编码，也不运行交替
MNGM 或第二阶段。使用新的输出目录，例如：

```bash
python -u -m scripts.diagnose_patent_glasso_convergence \
  --cache outputs/<原实验>/cache \
  --reference-experiment outputs/joint_evidence_full_n2048_k6_v062 \
  --config configs/patient_h04l_main_v1.json \
  --output outputs/<新诊断目录> --max-iter 6000 --diagnostic-every 250
```

`loss_trace.jsonl` 记录正定 ADMM 变量 X 的原始目标、稀疏变量 Z 在正定时的
原始目标、最小特征值、KKT、primal/dual 及其阈值。目标含无向边的 L1 惩罚，
不是下游检索 loss。稀疏变量不正定时其目标/KKT 为 null。诊断为可选观察，
不改变 ADMM 迭代或停止规则；测试验证开关诊断后求解结果逐位相同。
ADMM 目标不要求逐步单调，最终仍须通过残差、正定性和 KKT 检查。
只有实际收敛时才保存 `initial_graph_diagnostic.npz`；这不代表交替 MNGM 已完成。

1. 读取已完成 v0.6.2 cache 的真实样本 ID 和顺序；准备对应 2048 篇的有限长文本。
2. 为全部文档/800 个概念构建 H，记录全部证据多样性，以及原始 Qwen 空间和投影空间中对 prototype、空证据同模板表示的余弦。旧缓存的投影空间指标按同一基线重算，无需重复旧 Qwen 编码。旧原始 Qwen 全空间向量未保存，不能直接补算。
3. 保持旧 λ=0.8 和 MNGM/weighted Glasso 配置，先用 B=I 拟合 G0（与旧 initial_graph 可比），再运行一次固定 λ 的交替 MNGM 拟合。CPU 求解算法和线程设置不改。
4. 打包全部报告与小型图矩阵。没有 16—32 篇服务器预跑；本地合成测试仅验证代码和数据边界。

重点观察概念使用完全相同证据的文档比例、概念对证据重合率、每篇独立证据集数、表示对 prototype/空证据表示的余弦、真实 rank-Gaussian 协方差的相关性与谱、Glasso 收敛和图结构变化。相似度变低本身不能证明表示更正确；本轮不评估第二阶段任务收益。
