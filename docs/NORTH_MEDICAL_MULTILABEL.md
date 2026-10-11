# 北医三院多标签诊断探索实验

沿用服务器`QwenGAT/ch_data`既有description与8维diagnosis。依据原始
`data_processing.py::prepare_full_ch_data`，训练合并train_data.npy（1164）与
data0.npy（3399），原777条val_data保持验证用途。不读取test_data.npy。
标签索引7为髌股关节软骨损伤；其余索引保持原顺序。description原样沿用，
原数据包含病史、查体和核磁描述，因此这轮任务不是仅用病史字段。

用户明确选择全部4563条合并训练样本用于任务训练，同时全部用于每个候选的
GRPO奖励，不额外留出或抽取128条。原验证与合并训练文本不得重叠。
源文件无患者ID，不能据文本去重宣称患者级独立。3条重复记录仍保留原标签与
训练样本权重。使用现有具体关键词.csv第二列，去空、'-'、'0'并稳定去重，
实际876概念；不使用其他1245词或专利CPC词表。

全部4563条任务训练样本构建冻结Qwen3-Embedding H：[4563,64,876]。64-token片段
按概念选top2，joint最大512tokens，batch8。H使用静态prototype PCA线性投影；
训练标签不进入H或prototype。沿用原MNGM统计配置：lambda=.8、表示轴=.4、
原容差与10000迭代上限，先初始B=I求解，再固定lambda交替拟合初始图。

GLM-4.7-Flash BF16全GPU，LoRA r16/alpha32/dropout.05。查询侧8个图token
加description，最长512tokens，最后有效token隐藏态接8维线性分类头。
每个输出独立sigmoid，训练BCEWithLogits；不使用softmax/argmax，不将多个
诊断压成一个标签。LoRA、图token模块与分类头在全部任务训练池预热1轮，
batch4、lr1e-4；之后全部冻结。阈值固定0.5，报告micro/macro-F1、逐类F1与AP。

linear-GRPO沿用8共享系数、log lambda有界更新、完整MNGM重解与验证接受。
2阶段×2组×4候选=16候选，奖励用全部4563条训练样本，独立验证用全部777条。
主奖励为负BCE的改善，加原图密度/统计拟合惩罚；F1额外报告。接受后不继续任务
模型训练，属于冻结任务模型下的改图探索，不是完整交替训练。

同一个冻结模型分别评估原图、零图、节点随机置换图，在搜索前后保存逐样本
概率与损失以诊断图敏感性。置换只改变概念-图对应关系，不重新训练模型。
这些是推理干预，不能替代同预算随机搜索或独立训练无图基线。

```bash
source /laijizheng/miniconda3/etc/profile.d/conda.sh
conda activate graph_rl
cd /laijizheng/task_guided_graph_mvp_v0.3.0_2026-09-22
run_name="north_multilabel_linear_$(date -u +%Y%m%d_%H%M%S)"
nohup python -u -m scripts.run_north_medical_graph_rl --output "outputs/$run_name" > "outputs/$run_name.launch.log" 2>&1 < /dev/null &
echo $! > "outputs/$run_name.pid"
```

数据、病史、模型权重和逐样本结果只写服务器新outputs目录；GitHub只提交代码、
合成测试与说明。manifest记录阶段、Git SHA、输入校验和及失败状态；阶段日志
分别为prototypes.log、build_h.log，整体为launch.log。试跑只有单种子，尚无
优于原诊断方法或策略基线的结论。
