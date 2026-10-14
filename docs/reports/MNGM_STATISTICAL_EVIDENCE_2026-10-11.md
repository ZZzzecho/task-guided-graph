> 公开存档：仅研究说明及汇总结果；私有服务器路径及产物指纹已省略。历史运行状态按文中日期理解，当前结论以 [2026-10-14 最终报告](GRAPH_UTILITY_2026-10-14.md) 为准。

# Patient MNGM 的统计证据：文献与方法讨论

日期：2026-10-11。本文讨论证据定义与可用统计研究，不修改正在运行的北医实验。

## 结论

MNGM 的惩罚似然求解器通常输出精度矩阵点估计，而不是逐边置信度。可以另加统计推断层，得到边检验统计量、p 值、多重检验调整值，或者去偏估计与置信区间。另一条容易接入现有求解器的路线是重采样选择频率与符号稳定性。这些量分别衡量显著性、估计不确定性和可复现性，不能互相替代。

最直接相关的研究是 Xia 与 Li（2017）以及 Chen 与 Liu（2019）。两者都针对矩阵正态图模型，而非简单把矩阵展开后假定所有行独立。下文的工程建议是基于文献与当前代码作出的判断，不代表论文已经证明我们的 embedding、rank-Gaussian 和强化学习组合满足其条件。

## 当前模型的对应关系

每条记录形成 H_s ∈ R^(r×p)，当前 n=4563、r=64、p=876。A 对应概念轴精度矩阵，B 对应表示轴精度矩阵；当前代码将 A 命名为 Theta，将 B 保存为 representation_precision。

在矩阵正态模型中，Cov(vec(H_s)) = A^(-1) ⊗ B^(-1)，vec 采用列堆叠约定。概念边对应 A_ij≠0，常用标准化效应为 rho_ij=-A_ij/sqrt(A_ii A_jj)。它是模型中的条件关联，不是医学因果关系。

当前概念协方差计算为 S(B)=sum_s H_s^T B H_s/(nr)。先固定 B、求 A，再固定 A、求 B。相关代码：task-guided-graph-github/graph_mvp/estimators.py 的 _concept_covariance、solve_initial、solve。

独立抽样单位应是独立患者；如果同一患者有多条记录，应按患者整体重采样。现有数据尚未证明患者级独立性。r=64 个表示维度不能直接当成64份独立患者数据。B0=I 是初始化选择，不意味着真实 B=I。

## 文献

| 文献 | 核心结果 | 与本项目的关系 |
|---|---|---|
| Yin, J. & Li, H. (2012). Model selection and estimation in the matrix normal graphical model. Journal of Multivariate Analysis 107, 119–140. DOI: 10.1016/j.jmva.2012.01.005. | 矩阵正态模型的 l1 惩罚似然估计、算法及相应渐近性质。 | 估计层的背景；不能仅因求解收敛就宣称得到逐边置信度。 |
| Xia, Y. & Li, L. (2017). Hypothesis Testing of Matrix Graph Model with Application to Brain Connectivity Analysis. Biometrics 73(3), 780–791. DOI: 10.1111/biom.12633. | 分离矩阵两轴依赖，进行全局及逐边检验；在相应条件下控制 FDR。 | 最适合优先研读的概念轴检验路线；另一轴是需要处理的干扰结构。 |
| Chen, X. & Liu, W. (2019). Graph Estimation for Matrix-variate Gaussian Data. Statistica Sinica 29, 479–504. DOI: 10.5705/ss.202017.0076. | 用回归残差构造边统计量，对另一轴相关引起的方差膨胀作修正，结合 BH 实现渐近 FDR 控制。 | 提供不用把相关表示维度强当独立样本的另一条推断路线。 |
| Janková, J. & van de Geer, S. (2015). Confidence intervals for high-dimensional inverse covariance estimation. Electronic Journal of Statistics 9(1), 1205–1229. DOI: 10.1214/15-EJS1031. | 对 Graphical Lasso 作去偏/去稀疏化，构造精度矩阵元素的渐近置信区间。 | 支持“先消除惩罚偏差，再谈区间”的原则；普通向量数据结论不能直接套用 MNGM 的相关矩阵行。 |
| Meinshausen, N. & Bühlmann, P. (2010). Stability selection. JRSS B 72(4), 417–473. DOI: 10.1111/j.1467-9868.2010.00740.x. | 子采样与选择频率；在相应条件下控制预期错误选择数。 | 可给 MNGM 边定义稳定性，但选择频率不是边为真的后验概率。 |
| Shah, R. D. & Samworth, R. J. (2013). Variable selection with error control: another look at stability selection. JRSS B 75(1), 55–80. DOI: 10.1111/j.1467-9868.2011.01034.x. | 互补半样本稳定性选择（CPSS），对低选择概率项被选入等事件给出界。 | 可考虑作为配对重采样设计；“低选择概率项”不自动等于真实零边。 |
| Liu, H., Roeder, K. & Wasserman, L. (2010). Stability Approach to Regularization Selection (StARS) for High Dimensional Graphical Models. NeurIPS 2010. | 使用重采样图的不稳定程度选择正则化强度。 | 适合全图稳定性约束；原论文不是对逐边任意 lambda 矩阵的 RL 策略给出保证。 |
| Wang, H. & West, M. (2009). Bayesian analysis of matrix normal graphical models. Biometrika 96(4), 821–834. DOI: 10.1093/biomet/asp049. | 图结构先验、MCMC 后验计算与图不确定性分析。 | 可以得到真正具有模型条件概率解释的后验边包含概率；需要改用贝叶斯推断，不是当前 Glasso 顺带输出的量。 |

原文或作者版本链接：

- Yin & Li：https://www.sciencedirect.com/science/article/pii/S0047259X12000061
- Xia & Li：https://academic.oup.com/biometrics/article/73/3/780/7537729；作者预印本 https://arxiv.org/abs/1511.00718
- Chen & Liu：https://www3.stat.sinica.edu.tw/statistica/oldpdf/A29n124.pdf
- Janková & van de Geer：https://arxiv.org/abs/1403.6752
- Meinshausen & Bühlmann：https://arxiv.org/abs/0809.2932
- Shah & Samworth：https://arxiv.org/abs/1105.5578
- StARS：https://proceedings.neurips.cc/paper_files/paper/2010/file/301ad0e3bd5cb1627a2044908a42fdc2-Paper.pdf
- Wang & West：https://www2.stat.duke.edu/~mw/MWextrapubs/Wang2009.pdf；作者软件 https://www2.stat.duke.edu/~mw/mwsoftware/MATGRAPH/index.html

## 可选择的证据

### 1. 边效应与检验统计量

检验 H0,ij: A_ij=0 对 H1,ij: A_ij≠0。对概念轴，每对概念都构造经过相应偏差与方差处理的统计量。当前共有383250个假设，不能每条边独立用未经校正的 p<0.05 筛选。

检验统计量、p 值、BH 调整 p 值和原始效应量应分别保存。p 值衡量数据与零假设的相容程度，不是“边不存在的概率”；FDR 控制针对发现集合的预期错误比例，也不是每条边的错误概率。显著性还应配合实际效应大小。未显著不等于证明无边。

Xia–Li 路线：转置当前 H 后可把概念轴对应到论文的空间轴，把表示轴对应到时间轴；需要合适的表示协方差估计和回归残差检验，不能直接把当前惩罚 B 当作满足论文速率条件的估计量。

Chen–Liu 路线：使用相关行样本，但修正其检验方差。论文方差修正包含另一轴协方差的迹与 Frobenius 范数，说明相关维度提供的信息量不同于同等数量的独立维度。

### 2. 置信区间

置信区间可以表达某条边估计的精度。Janková–van de Geer 对普通向量 GGM 的去偏估计提供了直接依据。若迁移到 MNGM，需要处理另一轴估计和相关性；不能照抄独立样本的标准误。

原始 Glasso 有惩罚偏差和精确零点。对它直接 bootstrap，再用百分位区间是否跨0判断显著性，不自动获得正确覆盖率。用于描述重采样变异可以，但正式区间需要另行论证或校准。

### 3. 重采样稳定性

固定表示和明确的正则化配置，按独立样本单位子采样，多次拟合完整 A、B。对每条边计算 pi_ij = 平均[该次图中边非零]，并记录非零出现时的符号一致率和效应分布。

pi_ij=0.9 表示在指定子采样和拟合规则下约90%被选入，不表示90%概率是真边。该值依赖 lambda、样本量、非零阈值和求解规则。可以使用多个预先确定的 lambda 形成稳定性路径，避免只考察一个很稀疏的图就把所有未出现边认作无证据。

StARS 的单边不稳定度可写为2*pi_ij*(1-pi_ij)，全图取边对平均。但是 pi=0 和 pi=1 都稳定，空图也稳定，不能只追求最小不稳定度。稳定性选择、CPSS 与 StARS 的目标和理论保证各不相同。

完整 MNGM 的重采样成本是多次图求解；可以缓存 H、复用统计量和 warm start，但只做 B=I 的轻量筛选并不等同于完整 MNGM 的稳定性估计。

### 4. 贝叶斯后验边包含概率

在带有图结构变量/零约束的贝叶斯模型下，P((i,j)∈E | H) 可解释为给定模型和先验的后验概率。普通连续精度先验下“恰好非零”的概率并不能直接用于稀疏图选择，因此需要真正的图结构或适当的点质量设计。

Wang–West 已对矩阵正态图模型作了相关研究。但改用这条路线需要后验计算与收敛诊断；对本项目的876节点不应未经基准测试就承诺计算成本可接受。

### 5. KKT 分数的正确位置

当前 KKT frontier 衡量边与当前惩罚最优性边界的接近程度。它依赖当前 lambda，可用于“哪些边容易随惩罚改变”的搜索信息，但它没有自动校准的零假设分布或覆盖率，因此不是边置信度。可作为策略输入的优化敏感性特征，不适合作为唯一统计证据。

## 对当前实现的适用性检查

1. 当前对每个固定表示维度×概念单元，沿记录轴做 rank-Gaussian。这使边际分布近似正态，但不保证联合正态，也不保证协方差具有 Kronecker 可分离结构。向量 Gaussian copula 研究可作为变换背景，不能代替矩阵模型的推断条件。背景原文：https://arxiv.org/abs/1202.2169。
2. 概念表示共享 prototype 和编码器，可能含共同成分。统计结论应限定于所定义表示中的条件关联，不能称为临床因果连接。
3. 固定表示下的证据可先校准；投影矩阵一旦随任务更新，统计对象和数据适应性也改变。固定证据不能永久沿用。可考虑阶段内冻结表示，并使用交叉拟合或独立证据评估设计；这只是待讨论方案，本轮不改变全部训练数据奖励的设置。
4. 重复、同患者记录和数据来源异质性会影响独立同分布假设。需要患者标识或相应分组信息才能更严格地做患者级重采样。
5. 原论文的一次性 FDR 控制不能直接延伸为多轮 RL 后最终图的 FDR 控制。即使候选集经检验筛选，后续按任务收益自适应选子集，也不自动保留同样的 FDR 保证。
6. 用模拟数据先检查零边的统计量分布、实际 FDR、区间覆盖率和识别能力；同时考察非可分离、重尾、共享成分等模型偏离。不能仅用任务 F1 提升来验证证据校准。

## 待讨论的优先方案

正式方法路线：以矩阵图检验的统计量与多重检验结果为主要证据，以部分相关效应大小和重采样选择/符号稳定性为补充。先读透 Xia–Li（2017）和 Chen–Liu（2019）并比较所需估计条件，再选实现，暂不把多个量人为乘成一个“置信度”。

工程基线路线：先用完整 MNGM 的子采样选择频率定义可复现性证据，明确称为稳定性支持，不宣称逐边后验概率或正式显著性。

建议保留逐边证据向量：效应估计、标准误/区间（若有效）、检验统计量、调整 p 值、选择频率、非零条件下符号一致率。KKT 另列为优化敏感性特征。证据可以在初始图为零的边上计算，否则无法区分“被惩罚压成零”与“缺乏统计支持”。

下一次讨论再决定如何将强支持、弱支持和不确定的概念对映射为动作范围及方向；本次不预先锁定筛选阈值，也不重新设计策略网络。
