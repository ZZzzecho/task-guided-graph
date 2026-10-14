> 公开存档：仅研究说明及汇总结果；私有服务器路径及产物指纹已省略。历史运行状态按文中日期理解，当前结论以 [2026-10-14 最终报告](GRAPH_UTILITY_2026-10-14.md) 为准。

# GRPO 线性批量策略试跑记录

## 最新状态：2026-10-11 北京时间19:05，试跑完成

服务器19:05核验：manifest=complete，PID3498529已正常结束，日志最后更新2026-10-11 05:33:01。总runtime=69925.65秒，约19小时25分；10-10 10:07:34启动。16/16候选全部valid并完成评估，4/4组GRPO均updated=true；两阶段均完成。

phase1未接受：奖励最高候选在独立validation上任务改善-0.000846568，未过阈值。phase2接受grpo-linear-v1-000015：validation NLL改善0.000903868，accepted_phases=1。最终图16986→16387边，净减少599条（并非已统计599次边编辑；尚未核对增删集合）。Theta/B均保持正定。

独立validation（256 queries、64 candidates）：NLL 4.072348764166236→4.0714448960497975（下降0.000903868，约0.0222%）；MRR 0.15467389785175645→0.15514538447602677；Recall@10 0.27734375不变；Recall@50 0.8671875→0.859375。reward panel（128 queries）NLL3.9670261777937412→3.966468332335353，但MRR0.19174877294896137→0.18911771562491675、Recall@10 0.40625→0.390625。收益很小且排序指标混合，不能宣称RL策略优于随机搜索或全流程已验证；单种子、4次策略更新、预热后冻结任务模型，无匹配基线，test_split_touched=false。

四组reward_mean依次0.0003930292、-0.0002314791、-0.0007452529、-0.0000343453，未持续上升；policy loss -0.2088816、-0.1924264、-0.0734514、-0.1482619，不能当作任务loss收敛趋势。KL对初始参考0.02677→0.19390→0.52018→0.83297。当前试跑已完成，无新的实验进程由此次状态检查启动。

### 完成后的流程核验与图差异分析

本轮是固定长文本H/初始拟合图、检索器50步预热、冻结任务模型后的linear-GRPO图搜索；warmup优化LoRA与SoftGraphTokenizer，之后无task_adapt。下游读取data/patents_h04l/corpus.csv.gz的text（Title+Abstract），GLM输入上限384tokens；新增长文本4096-token候选池仅用于Qwen/MNGM学图，未同步用于下游检索输入。图经查询侧概念激活、样本条件化消息传递、learnable pooling压缩为8个图token，前置到GLM query输入；候选编码快照在预热前固定且无图。Qwen与GLM使用同一概念ID但各自空间的prototype。

每个phase固定父图，两组各4候选；组间只更新policy，phase结束从全部8候选按reward选一个，再独立validation检查NLL改善>1e-8。phase1拒绝，所以phase2仍从同一初始图出发；直到试跑末尾才第一次接受图，并未形成连续图-任务模型交替训练。summary的training_semantics字符串是通用描述，实际只有50optimizer steps、100个不同train queries、adapted=false。

按实际adjacency_threshold=1e-5比较partial_corr：新增2条、删除601条、保留16385条，净减少599；rho Frobenius变化0.21465877，相对原矩阵约2.978%。最终候选15 coefficients=[-1,0,-1,-1,-1,1,0,-1]，delta范围[-0.9013988293,-0.3453516513]，所有319600对lambda都下降，约3.2%–8.2%，没有同一候选内一部分升一部分降。由于完整MNGM同步重估B/有效S，不能据lambda方向推断最终边数。这个候选也提示需以相同预算的标量lambda搜索、随机8系数搜索为对照，区分GRPO策略学习收益与整体惩罚调参收益。

## 历史状态：2026-10-10 北京时间15:27

15:27:03服务器核验：PID3498529处于R/running，日志最后更新15:26:59，manifest仍running，无异常退出。首组4/4候选均完整求解且valid=true，均完成奖励评估；4/16候选完成，当前phase1/2、update2/2、candidate1/4，即总第5个候选grpo-linear-v1-000005。当前outer1/12的concept GLASSO到2425/10000，primal从2000步0.005697降至2425步0.003015，仍在收敛。

首组任务改善/奖励：c1 delta_task=0.000476880、reward=0.0000248454；c2两者0.000871753；c3 delta_task=-0.000409769、reward=-0.000421408；c4两者0.0010969266。3个候选奖励为正，1个为负；收益幅度较小，尚无独立validation接受结果，不能据此宣称有效性已验证。

首个GRPO策略更新updated=true、num_valid=4、reward_mean=0.0003930292、reward_std=0.0006171540、loss=-0.2088816、kl_to_reference=0.02676646、clip_fraction=1.0；证明已发生实际策略学习，但一次loss不能证明持续下降。首组耗时17505.15秒（约4小时52分）。GPU0%、66645/143771 MiB，当前CPU上的MNGM求解。阶段输出尚未生成，下一步观察第二组完成和phase1独立验证。

## 历史状态：2026-10-10 北京时间12:57

12:57:46服务器核验：PID3498529仍在运行，最新日志12:57:41，无Traceback，manifest=running/error=null。50/50步任务warmup完成，末步loss=4.065868，最后10步均值4.004186；实际训练query池512（max-train-queries=1024只是上限），已见100个不同query。

当前phase1/2、GRPO组1/2、组内候选3/4（grpo-linear-v1-000003）；已完成2/16个候选完整MNGM求解，两者valid=true。第3个候选concept GLASSO到4225/10000，primal residual从3050步9.219e-4降到4200步1.387e-4，再到4225步1.331e-4，数值收敛正常；10000是求解上限，不是整体进度分母。

第2个候选全部319600对惩罚发生改变，图边数由16986变为16892（减少94），证明批量动作能实际改变图。当前candidate_done=0、grpo_update_done=0：实现先完成组内4个候选求解，再评估reward并更新策略；尚无奖励或RL有效性结论。此刻GPU计算0%，显存66645/143771 MiB，属于CPU上的MNGM求解阶段。

下一次优先检查第3/4候选完成、首次reward/validation和首次GRPO更新。无定时监控。

## 历史状态：2026-10-10 北京时间10:19

首个输出`outputs/grpo_linear_trial_20261009_064600`在10-09 14:53:52因`NotImplementedError: Cannot copy out of meta tensor; no data!`退出。错误发生于模型自动CPU offload后的FrozenCandidateSnapshot构造，尚未warmup或GRPO，0/16候选。10-10检查发现PID已不存在；已依据原日志将该run_manifest从running修正为failed，保留原始输出。

已修复、测试、通过GitHub发布并重启。新提交`b31368f6d33dff4a72ce433bc5c9e8af8fdc154f`，本地/GitHub/服务器一致，本地工作树干净；仍为草稿[PR#2](https://github.com/ZZzzecho/task-guided-graph/pull/2)。新试跑明确`--device-map cuda:0`，全部BF16模型留在GPU；新增meta参数/缓冲检查和启动异常失败状态记录，不跳过候选快照、冻结encoder fingerprint或统计求解检查。52项相关测试在独立验证目录与实际graph_rl项目环境各通过一次。

新nohup PID：3498529。输出`outputs/grpo_linear_trial_20261010_020734_gpu`；日志同名加`.launch.log`，PID文件同名加`.pid`。实际启动北京时间10:07:34，目录后缀为服务器UTC，即使设置TZ=Asia/Shanghai，服务器date仍输出UTC；用户时间以显式Python timezone(+08:00)核验。

预算/模型/数据/策略不变：2048篇完整H，800概念，R64，原fitted graph/B；50步任务warmup后冻结，adapt=0；2 phases×2 updates×4 candidates=16，reward128/val256，test未传。完整cache与图/B复用检查再次通过，重算KKT=1.999184151868505e-5。

10:19:15实际检查：进程Rl、运行11分41秒，模型751/751加载完成；`candidate-index train`报告encoded/cached=8530、cache_consistent=True。冻结候选encoder fingerprint=<OMITTED_ARTIFACT_FINGERPRINT>。GPU显存59975/143771 MiB（约58.6/140.4 GiB），计算利用率98%。这证实已越过原meta失败位置、实际GPU编码正常；目前继续graph/val候选索引准备，尚未出现warmup或GRPO进度，因此仍0/16候选，不能宣称策略效果或训练完成。run_manifest=running，无error；candidate_encoder字段待所有索引准备完写入，字段暂缺不代表快照失败。

下一次优先查看这个新PID和新日志，确认三个候选索引、50步warmup、GRPO首次候选求解。无定时监控。以下原10-09文字只保留为历史启动快照。

---

2026-10-09北京时间14:46启动，最后14:50左右检查仍正常运行。新增策略已实现和发布，探索实验尚未完成，尚无有效性结论。

GitHub：[草稿PR#2](https://github.com/ZZzzecho/task-guided-graph/pull/2)，基于PR#1分支。分支`feat/grpo-linear-penalty`，提交`8fc2a62f6062f2d3ee8dc6bd9060337d8052b09b`；本地和服务器代码一致，本地工作树干净。生产代码由服务器从GitHub拉取。临时验证目录只用于发布前测试。

策略：8个共享系数，每个取{-1,0,+1}，根据固定prototype cosine、有效协方差关联、partial correlation、active indicator、KKT frontier、节点度和训练reward-panel任务相关性等8项特征，线性展开到全部319600对概念惩罚。特征按行L1归一化，每对log Lambda变化不超过原eta=log(1.1)。对称、对角为零、[0.01,4]边界保持原设定。动作概率和精确KL只在8个系数上计算；不按展开边数重复计数，不使用edge reward EMA。原32对策略仍为默认对照。

完整MNGM仍逐候选求解，warm Theta/B；严格数值检查和独立validation接受不变。可用`--policy grpo-linear`选择，定义与启动脚本见仓库`docs/LINEAR_PENALTY_GRPO.md`和`scripts/run_linear_penalty_trial.sh`。

服务器：laijizheng；环境graph_rl；项目`<PRIVATE_SERVER_PATH>`。

输出：`outputs/grpo_linear_trial_20261009_064600`（目录后缀为服务器UTC时间）。日志同名加`.launch.log`，PID文件同名加`.pid`。nohup PID：3431446。

复用cache：`outputs/patent_longtext_graph_n2048_b4096_k6_v063_server_20261008_064102/cache`。完整2048篇、R64、800 concepts，不重编码H。复用图目录：`outputs/patent_longtext_graph_fit_n2048_iter10000_20261009_101910`。输入缓存全部校验和、配置、概念顺序、完整样本fingerprint、B正定、从H/B重新计算S和原KKT检查均通过。图文件SHA256=<OMITTED_ARTIFACT_FINGERPRINT>，KKT=1.999184151868505e-5；初始图16986边，density=0.053147684605757195。

探索预算：50步任务LoRA预热，训练query pool1024；之后任务模型冻结，不做接受后的LoRA adaptation（adapt_steps=0）。reward128 queries、validation256 queries；2 phases ×2 groups ×4 candidates=16候选完整求解。不传test pools，不查看测试集。本轮验证即时改图收益和策略是否可训练，不是完整多种子因果归因对照。

已通过46项服务器相关测试，且GitHub拉取后在实际项目graph_rl环境再次通过46项；脚本语法和模块编译通过。覆盖概率重算、真实参数更新、invalid/flat组处理、约束/keep/真实求解与奖励/验证流程、图复用校验/损坏拒绝、旧策略与检索/MNGM接口。

最后观察：模型加载到480/751（64%），日志无Traceback；GPU显存32491/143771 MiB，计算利用率0%，此时属于加载阶段。尚未到GRPO候选求解，不能用这个时点推算整体训练时间。已安装distribution metadata显示0.6.2，实际源码__version__为0.6.3，实验代码以记录的Git SHA为准。

下一次检查先读PID和启动日志，确认模型/候选索引/50步warmup，然后看progress.jsonl及rl/phase_*.json：实际修改惩罚对数、Rho变化、候选有效率、奖励标准差、GRPO更新/KL/entropy和validation接受。若失败，保留现有输出并记录失败原因，重新启动使用新目录。不重建长文本输入，不放宽数值阈值。没有创建定时监控，服务器实验会继续在后台运行。
