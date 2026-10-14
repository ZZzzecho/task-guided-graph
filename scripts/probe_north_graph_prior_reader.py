"""Use a statistical graph as a classifier-weight prior, not injected features."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from graph_mvp.multilabel_task import multilabel_metrics
from graph_mvp.implicit_glasso import macro_pairwise_logit_gradient


def graph_prior_penalty(weight,q,representation_dim,strength=.01):
    """strength/(2L) sum_label,row w Q w; Q has a constant unit diagonal."""
    l=weight.shape[0];w=weight.reshape(l*representation_dim,-1)
    qw=torch.sparse.mm(q,w.T).T
    return strength*(w*qw).sum()/(2*l)


def sparse_prior(theta,mix=.7,device='cuda:0'):
    d=np.sqrt(np.diag(theta));c=theta/np.outer(d,d)
    q=(1-mix)*np.eye(len(theta))+mix*c
    np.linalg.cholesky(q);i,j=np.nonzero(q)
    return torch.sparse_coo_tensor(np.stack([i,j]),q[i,j],q.shape,dtype=torch.float32,device=device,check_invariants=True).coalesce()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','aligned','features','heads','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    report={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol':{'seeds':[11,17,23],'arms':['diagonal','true_graph']+[f'permutation_{s}' for s in range(1001,1006)],
            'train':4563,'validation':777,'test_split_touched':False,'selection':'none; report every graph/seed',
            'features':'same training-ECDF rank H, centered on training mean, divided by sqrt(RP)',
            'reader':'zero-initialized linear logit correction, no bias, same capacity and frozen text logits',
            'prior':'Q=.3 I+.7 normalized concept precision; diagonal=1 in every arm',
            'random':'five node permutations of normalized precision, same spectrum and diagonal as true prior',
            'objective':'macro pair ranking temperature1 +.1 BCE +.1 mean-square correction +.01/(2L) sum wQw',
            'optimizer':'full-population Adam .003, 500 steps, gradient clip1',
            'limits':'exploration validation, one frozen backbone, fixed graph, not GRPO; convex readout objective but finite-step optimizer not claimed exactly converged'},'runs':[]}
    (args.output/'protocol.json').write_text(json.dumps(report['protocol'],indent=2))
    train=np.load(args.aligned/'train_private_rank_matrices.npy',mmap_mode='r',allow_pickle=False)
    val=np.load(args.aligned/'val_private_rank_matrices.npy',mmap_mode='r',allow_pickle=False)
    if train.shape!=(4563,64,876) or val.shape!=(777,64,876):raise ValueError('population mismatch')
    center=train.mean(0,dtype=np.float64).astype(np.float32);normalizer=np.sqrt(64*876)
    h={s:torch.tensor((np.asarray(v,dtype=np.float32)-center).reshape(len(v),-1)/normalizer,device='cuda:0') for s,v in [('train',train),('val',val)]}
    labels={s:np.asarray([json.loads(l)['labels'] for l in (args.source/'prepared'/f'{s}.jsonl').read_text().splitlines()]) for s in ['train','val']}
    y=torch.tensor(labels['train'],dtype=torch.float64,device='cuda:0')
    text={s:np.load(args.features/f'{s}_private_features.npz',allow_pickle=False)['mean'] for s in ['train','val']}
    mu=text['train'].mean(0);sd=np.maximum(text['train'].std(0),1e-4)
    x={s:torch.tensor((v-mu)/sd,device='cuda:0') for s,v in text.items()}
    bases={}
    for seed in report['protocol']['seeds']:
        head=nn.Linear(x['train'].shape[1],8).cuda();head.load_state_dict(torch.load(args.heads/f'seed_{seed}_uniform_private_head.pt',map_location='cuda:0',weights_only=True))
        with torch.no_grad():bases[seed]={s:head(v).detach() for s,v in x.items()}
    theta=np.load(args.source/'initial_graph.npz',allow_pickle=False)['Theta']
    d=np.sqrt(np.diag(theta));c=theta/np.outer(d,d);graphs={'diagonal':np.eye(len(theta)),'true_graph':c}
    for seed in range(1001,1006):
        order=np.random.default_rng(seed).permutation(len(theta));graphs[f'permutation_{seed}']=c[np.ix_(order,order)]
    for name,a in graphs.items():
        q=sparse_prior(a)
        for seed in report['protocol']['seeds']:
            model=nn.Linear(h['train'].shape[1],8,bias=False).cuda();nn.init.zeros_(model.weight)
            opt=torch.optim.Adam(model.parameters(),lr=.003);start=time.monotonic()
            for step in range(500):
                correction=model(h['train']);z=bases[seed]['train']+correction
                surrogate,coefficient=macro_pairwise_logit_gradient(z,y)
                prior=graph_prior_penalty(model.weight,q,64)
                loss=(z.double()*coefficient).sum()+.1*correction.square().mean()+prior
                opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                if step in [0,99,499]:print(json.dumps({'stage':'prior_training','graph':name,'seed':seed,'step':step,'ranking_surrogate':surrogate,'prior':float(prior.detach()),'correction_std':float(correction.detach().std()),'gradient_norm':float(grad)}),flush=True)
            row={'graph':name,'seed':seed,'seconds':time.monotonic()-start}
            with torch.no_grad():
                for split in ['train','val']:
                    z=(bases[seed][split]+model(h[split])).cpu().numpy();metrics,losses,prob=multilabel_metrics(labels[split],z)
                    row[split]=dict(metrics.extra_metrics)
                    np.savez_compressed(args.output/f'{name}_seed_{seed}_{split}_private_scores.npz',labels=labels[split],probabilities=prob,per_example_loss=losses)
            torch.save(model.state_dict(),args.output/f'{name}_seed_{seed}_private_reader.pt')
            report['runs'].append(row);(args.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({'stage':'prior_complete',**row}),flush=True)


if __name__=='__main__':main()
