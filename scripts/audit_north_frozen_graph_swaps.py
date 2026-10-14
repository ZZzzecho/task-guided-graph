"""Frozen mention-reader graph interventions, with matched permuted teachers."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from graph_mvp.multilabel_task import multilabel_metrics
from scripts.probe_north_cached_graph_reader import MentionGraphResidual


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','features','heads','reader','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    rho=torch.tensor(np.load(args.source/'initial_graph.npz',allow_pickle=False)['partial_corr'],dtype=torch.float32)
    terms=[r['concept_text'] for r in json.loads((args.source/'prepared/concepts.json').read_text())]
    train=np.load(args.features/'train_private_features.npz',allow_pickle=False)['mean']
    v=np.load(args.features/'val_private_features.npz',allow_pickle=False)['mean'];v=torch.tensor((v-train.mean(0))/np.maximum(train.std(0),1e-4))
    rows=[json.loads(l) for l in (args.source/'prepared/val.jsonl').read_text().splitlines()]
    y=np.asarray([r['labels'] for r in rows]);a=torch.tensor([[t in r['text'] for t in terms] for r in rows],dtype=torch.float32)
    scale=torch.tensor(np.load(args.reader/'private_message_normalization.npz',allow_pickle=False)['scale'])
    graphs={'true':rho,'zero':torch.zeros_like(rho)}
    for seed in range(1001,1006):
        order=np.random.default_rng(seed).permutation(len(rho));graphs[f'permutation_{seed}']=rho[order][:,order]
    report={'protocol':{'frozen_teachers':['graph','permuted'],'seeds':[11,17,23],
        'interventions':['true','zero']+[f'permutation_{s}' for s in range(1001,1006)],
        'no_retraining':True,'test_split_touched':False,
        'interpretation':'graph swap sensitivity shows receiver dependence, not standalone graph efficacy; permuted-trained teachers test generic distribution-shift sensitivity'},'runs':[]}
    with torch.no_grad():
        for seed in [11,17,23]:
            head=nn.Linear(v.shape[1],8);head.load_state_dict(torch.load(args.heads/f'seed_{seed}_uniform_private_head.pt',weights_only=True));base=head(v)
            for teacher in ['graph','permuted']:
                reader=MentionGraphResidual(len(terms));reader.load_state_dict(torch.load(args.reader/f'seed_{seed}_{teacher}_private_reader.pt',weights_only=True))
                native='true' if teacher=='graph' else 'permutation_1001'
                original=base+reader(a,(a@graphs[native])*a/scale)
                saved=np.load(args.reader/f'seed_{seed}_{teacher}_val_private_scores.npz',allow_pickle=False)['probabilities']
                np.testing.assert_allclose(original.sigmoid().numpy(),saved,rtol=1e-5,atol=1e-6)
                for name,g in graphs.items():
                    z=base+reader(a,(a@g)*a/scale);m,loss,prob=multilabel_metrics(y,z.numpy())
                    report['runs'].append({'seed':seed,'teacher':teacher,'intervention':name,'native':name==native,
                        'validation':dict(m.extra_metrics),'mean_absolute_probability_change':float((z.sigmoid()-original.sigmoid()).abs().mean())})
                    np.savez_compressed(args.output/f'seed_{seed}_{teacher}_{name}_private_scores.npz',labels=y,probabilities=prob)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    means={}
    for teacher in ['graph','permuted']:
        means[teacher]={name:{'AP':float(np.mean([r['validation']['macro_average_precision'] for r in report['runs'] if r['teacher']==teacher and r['intervention']==name])),
            'F1':float(np.mean([r['validation']['macro_f1'] for r in report['runs'] if r['teacher']==teacher and r['intervention']==name]))} for name in graphs}
    print(json.dumps({'teacher_intervention_means':means}),flush=True)


if __name__=='__main__':main()
