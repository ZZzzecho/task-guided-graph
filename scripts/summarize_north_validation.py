"""Aggregate completed experiments without reading clinical records or test data."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['implicit','aligned','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    implicit=json.loads((args.implicit/'summary.json').read_text())
    aligned=json.loads((args.aligned/'summary.json').read_text())
    if implicit['test_split_touched'] or aligned['protocol']['test_split_touched']:raise ValueError('test-isolation violation')
    seeds=aligned['protocol']['seeds'];arms=aligned['protocol']['arms']
    if len(aligned['runs'])!=len(seeds)*len(arms):raise ValueError('incomplete aligned experiment')
    index={(r['seed'],r['arm']):r for r in aligned['runs']}
    if len(index)!=len(seeds)*len(arms):raise ValueError('duplicate or missing aligned run')
    def val(row):
        return {'macro_AP':row['val']['macro_average_precision'],'macro_F1':row['val']['macro_f1']}
    arm_means={a:{k:float(np.mean([val(index[s,a])[k] for s in seeds])) for k in ['macro_AP','macro_F1']} for a in arms}
    contrasts={a:{'mean_AP_delta':arm_means['graph']['macro_AP']-arm_means[a]['macro_AP'],
        'mean_F1_delta':arm_means['graph']['macro_F1']-arm_means[a]['macro_F1'],
        'seed_AP_deltas':[val(index[s,'graph'])['macro_AP']-val(index[s,a])['macro_AP'] for s in seeds]} for a in arms if a!='graph'}
    reader_gate=all(c['mean_AP_delta']>=.005 and c['mean_F1_delta']>=-.002 and all(d>0 for d in c['seed_AP_deltas']) for c in contrasts.values())
    rows={r['name']:r for r in implicit['runs'] if 'val' in r}
    final={family:{'selected':selection['selected'],'accepted':selection['accepted'],**val(rows[selection['selected']])}
        for family,selection in implicit['final_train_only'].items()}
    comparator=max([val(rows['initial'])]+[final[f] for f in ['random','global']],key=lambda r:r['macro_AP'])
    directed=final['directed'];ap_delta=directed['macro_AP']-comparator['macro_AP'];f1_delta=directed['macro_F1']-comparator['macro_F1']
    result={'test_split_touched':False,'aligned_reader_means':arm_means,'aligned_contrasts':contrasts,
        'aligned_exploration_gate_met':reader_gate,'implicit_final':final,
        'implicit_directed_vs_strongest_AP_delta':ap_delta,'implicit_directed_vs_strongest_F1_delta':f1_delta,
        'implicit_single_seed_signal':ap_delta>=.005 and f1_delta>=-.002,
        'confirmed_task_guided_efficacy':False,
        'evidence_limits':['one implicit task-model seed','reader seeds share one frozen backbone',
            'exploration validation reused; test closed','no GRPO comparison on the new receiver',
            'no formal statistical-evidence action mask or initial-graph transfer test'],
        'decision':'存在值得独立确认的有限信号' if reader_gate or (ap_delta>=.005 and f1_delta>=-.002) else '本轮有限实验未达到 task-guided 收益目标',
        'interpretation':'This cannot prove all future task-guided algorithms impossible. Readout gains and certified solver speedups are separate from graph-policy gains.'}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'aggregate_decision.json').write_text(json.dumps(result,indent=2))
    lines=['# 北医 task-guided 验证：已完成实验汇总','',
        '以下均为复用验证集上的探索性结果，测试集未打开。','',
        '| 匹配表示的读出 | 平均 macro-AP | 平均 macro-F1 |','|---|---:|---:|']
    for a,m in arm_means.items():lines.append(f"| {a} | {m['macro_AP']:.6f} | {m['macro_F1']:.6f} |")
    lines+=['','种子 11/17/23 仅改变接收器，使用同一个冻结语言模型。','',
        '| 图优化家族（训练集选择及完整 MNGM 检查后） | 选用 | macro-AP | macro-F1 |','|---|---|---:|---:|']
    for family,m in final.items():lines.append(f"| {family} | {m['selected']} | {m['macro_AP']:.6f} | {m['macro_F1']:.6f} |")
    lines+=['',f'定向优化相对最强固定图/随机/统一搜索对照：AP 差 {ap_delta:+.6f}，F1 差 {f1_delta:+.6f}。',
        '',f"有限实验判定：{result['decision']}。",
        '', '没有得到独立确认的 task-guided 或 GRPO 有效性结论。单种子、复用验证集、固定骨干模型和未实现的统计证据方案，限制了结论范围。',
        '这些实验不能证明所有后续 task-guided 算法都不可行。分类头的改善和统计求解提速需要与图策略的收益分别表述。']
    (args.output/'completed_experiments.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
