"""Full-population frozen-feature readout checks; never access the test split."""
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from graph_mvp.multilabel_task import multilabel_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[11,17,23])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    report={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol':{'seeds':args.seeds,'steps':300,'learning_rate':.01,'weight_penalty':.001,
            'readout':'mean','normalization':'training mean and std, minimum scale 1e-4',
            'weightings':['uniform','sqrt_positive'],'primary_weighting':'uniform',
            'population':4563,'validation':777,'selection':'none; report every seed and weighting',
            'test_split_touched':False,'source_groups':'first 1164 original training; remaining 3399 data0',
            'limitation':'common frozen backbone may have been trained with graph; not an independent text-only training baseline'},
        'runs':[]}
    labels={};features={}
    for split,n in [('train',4563),('val',777)]:
        rows=[json.loads(line) for line in (args.source/'prepared'/f'{split}.jsonl').read_text().splitlines()]
        labels[split]=np.asarray([r['labels'] for r in rows],dtype=np.float32)
        features[split]=np.load(args.features/f'{split}_private_features.npz',allow_pickle=False)['mean']
        if labels[split].shape!=(n,8) or len(features[split])!=n:raise ValueError('population mismatch')
        if not np.isfinite(features[split]).all():raise ValueError('nonfinite features')
    center=features['train'].mean(0);scale=np.maximum(features['train'].std(0),1e-4)
    x={s:torch.from_numpy((f-center)/scale) for s,f in features.items()}
    y=torch.from_numpy(labels['train']);positives=y.sum(0)
    for seed in args.seeds:
        for weighting in report['protocol']['weightings']:
            torch.manual_seed(seed);head=nn.Linear(x['train'].shape[1],8).float()
            opt=torch.optim.Adam(head.parameters(),lr=.01);start=time.monotonic()
            weights=torch.clamp(torch.sqrt((len(y)-positives)/positives.clamp_min(1)),1,10) if weighting!='uniform' else None
            for step in range(300):
                loss=F.binary_cross_entropy_with_logits(head(x['train']),y,pos_weight=weights)+.001*head.weight.square().mean()
                if not torch.isfinite(loss):raise RuntimeError('nonfinite readout loss')
                opt.zero_grad();loss.backward();opt.step()
            row={'seed':seed,'weighting':weighting,'seconds':time.monotonic()-start,'final_training_objective':float(loss.detach())}
            for split in ['train','val']:
                with torch.no_grad():logits=head(x[split]).numpy()
                metrics,losses,prob=multilabel_metrics(labels[split],logits)
                row[split]=dict(metrics.extra_metrics)
                np.savez_compressed(args.output/f'seed_{seed}_{weighting}_{split}_private_scores.npz',
                    labels=labels[split],probabilities=prob,per_example_loss=losses)
                if split=='train':
                    row['training_sources']={name:dict(multilabel_metrics(labels[split][sl],logits[sl])[0].extra_metrics)
                        for name,sl in [('original',slice(0,1164)),('data0',slice(1164,None))]}
            torch.save(head.state_dict(),args.output/f'seed_{seed}_{weighting}_private_head.pt')
            report['runs'].append(row)
            (args.output/'summary.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'stage':'head_complete',**row}),flush=True)


if __name__=='__main__':main()
