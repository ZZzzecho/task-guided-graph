"""Cheap matched graph mechanism probe using private cached text features.

Same frozen backbone and text head in all arms. Literal mentions are an
exploratory concept gate, not annotated medical entities or statistical evidence.
"""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from graph_mvp.multilabel_task import multilabel_metrics


class MentionGraphResidual(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.self_scale=nn.Parameter(torch.ones(p))
        self.graph_scale=nn.Parameter(torch.ones(p))
        self.bias=nn.Parameter(torch.zeros(p))
        self.readout=nn.Linear(p,8,bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self,a,message):
        h=torch.tanh(a*self.self_scale+message*self.graph_scale+self.bias)
        return self.readout(h)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--heads',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    graph=np.load(args.source/'initial_graph.npz',allow_pickle=False)
    concepts=json.loads((args.source/'prepared/concepts.json').read_text())
    if tuple(str(i) for i in graph['concept_ids'])!=tuple(c['concept_id'] for c in concepts):raise ValueError('concept order mismatch')
    terms=[c['concept_text'] for c in concepts];labels={};features={};mentions={}
    for split,n in [('train',4563),('val',777)]:
        rows=[json.loads(l) for l in (args.source/'prepared'/f'{split}.jsonl').read_text().splitlines()]
        labels[split]=np.asarray([r['labels'] for r in rows],dtype=np.float32)
        features[split]=np.load(args.features/f'{split}_private_features.npz',allow_pickle=False)['mean']
        mentions[split]=torch.tensor([[t in r['text'] for t in terms] for r in rows],dtype=torch.float32)
        if labels[split].shape!=(n,8) or len(features[split])!=n:raise ValueError('population mismatch')
    center=features['train'].mean(0);scale=np.maximum(features['train'].std(0),1e-4)
    x={s:torch.from_numpy((f-center)/scale) for s,f in features.items()}
    y=torch.from_numpy(labels['train']);rho=torch.tensor(graph['partial_corr'],dtype=torch.float32)
    initial_message=(mentions['train']@rho)*mentions['train']
    message_scale=initial_message.std(0,correction=0).clamp_min(.1)
    permutation=np.random.default_rng(1001).permutation(len(terms))
    permuted=rho[permutation][:,permutation]
    report={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol':{'seeds':[11,17,23],'arms':['text','concept','graph','permuted'],
            'text_head':'frozen corresponding uniform-BCE cached head',
            'concept_gate':'literal substring binary mention, no label-derived gating',
            'message':'a_i sum_j rho_ji a_j / fixed initial-training per-node std (minimum .1)',
            'residual':'per-node tanh with learned self/message scales and bias; zero-initialized 8-label readout',
            'train':4563,'val':777,'steps':300,'optimizer':'Adam .01, .001 mean-squared readout penalty',
            'selection':'none; all seeds and arms reported','test_split_touched':False,
            'limitations':'exploratory new graph reader on one frozen backbone; not full-pipeline text baseline, not GRPO or statistical confidence'},
        'runs':[]}
    (args.output/'protocol.json').write_text(json.dumps(report['protocol'],indent=2))
    for seed in report['protocol']['seeds']:
        head=nn.Linear(x['train'].shape[1],8).float()
        head.load_state_dict(torch.load(args.heads/f'seed_{seed}_uniform_private_head.pt',weights_only=True))
        with torch.no_grad():base={s:head(v).detach() for s,v in x.items()}
        for arm in report['protocol']['arms']:
            torch.manual_seed(seed);reader=MentionGraphResidual(len(terms));start=time.monotonic()
            messages={s:((a@(rho if arm=='graph' else permuted))*a/message_scale
                if arm in ('graph','permuted') else torch.zeros_like(a)) for s,a in mentions.items()}
            if arm!='text':
                optimizer=torch.optim.Adam(reader.parameters(),lr=.01)
                for step in range(300):
                    logits=base['train']+reader(mentions['train'],messages['train'])
                    loss=F.binary_cross_entropy_with_logits(logits,y)+.001*reader.readout.weight.square().mean()
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite residual loss')
                    optimizer.zero_grad();loss.backward();optimizer.step()
            row={'seed':seed,'arm':arm,'seconds':time.monotonic()-start}
            for split in ['train','val']:
                with torch.no_grad():z=(base[split]+(reader(mentions[split],messages[split]) if arm!='text' else 0)).numpy()
                metrics,losses,prob=multilabel_metrics(labels[split],z);row[split]=dict(metrics.extra_metrics)
                np.savez_compressed(args.output/f'seed_{seed}_{arm}_{split}_private_scores.npz',labels=labels[split],probabilities=prob,per_example_loss=losses)
            torch.save(reader.state_dict(),args.output/f'seed_{seed}_{arm}_private_reader.pt')
            report['runs'].append(row);(args.output/'summary.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'stage':'reader_complete',**row}),flush=True)
    np.savez_compressed(args.output/'private_message_normalization.npz',scale=message_scale.numpy())


if __name__=='__main__':main()
