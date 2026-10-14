"""Task-label anchors read their H and the graph's conditional prediction of H."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from graph_mvp.patient_repr import Qwen3EmbeddingEncoder,load_concept_prototypes
from graph_mvp.north_medical import LABEL_NAMES
from graph_mvp.multilabel_task import multilabel_metrics
from graph_mvp.implicit_glasso import macro_pairwise_logit_gradient


def select_anchors(prototypes,queries,k=8):
    p=np.asarray(prototypes,dtype=np.float64);q=np.asarray(queries,dtype=np.float64)
    if p.ndim!=2 or q.ndim!=2 or p.shape[1]!=q.shape[1] or not 1<=k<=len(p):raise ValueError('invalid anchor dimensions')
    pn=np.linalg.norm(p,axis=1);qn=np.linalg.norm(q,axis=1)
    if np.any(pn==0) or np.any(qn==0) or not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('invalid anchor embeddings')
    similarity=(q/qn[:,None])@(p/pn[:,None]).T
    return np.argsort(-similarity,axis=1,kind='stable')[:,:k]


def anchor_features(x,theta,anchors):
    """Graph prediction excludes each target concept's own value."""
    a=torch.as_tensor(anchors,device=x.device);n,r,p=x.shape;l,k=a.shape
    own=x[:,:,a].mean(-1).transpose(1,2)
    flat=np.asarray(anchors).reshape(-1)
    beta=-theta[flat,:]/np.diag(theta)[flat,None]
    beta[np.arange(len(flat)),flat]=0.
    predicted=(x.reshape(n*r,p)@torch.tensor(beta.T,dtype=x.dtype,device=x.device)).reshape(n,r,l,k).mean(-1).transpose(1,2)
    return torch.cat([own,predicted],dim=-1)


class AnchorResidual(nn.Module):
    def __init__(self,features,hidden=32):
        super().__init__()
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(features,hidden),nn.GELU(),nn.Linear(hidden,1)) for _ in range(8)])
        for head in self.heads:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def forward(self,x):return torch.cat([head(x[:,i,:]) for i,head in enumerate(self.heads)],dim=-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','aligned','features','heads','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    manifest=json.loads((args.source/'run_manifest.json').read_text())
    encoder=Qwen3EmbeddingEncoder(manifest['args']['embedding_model'],device='cuda:0',dtype='bfloat16',local_files_only=True)
    queries=encoder.encode_pooled(LABEL_NAMES).float().cpu().numpy()
    prototypes,meta=load_concept_prototypes(args.source/'prepared/concept_prototypes.npz')
    anchors=select_anchors(prototypes,queries);del encoder;torch.cuda.empty_cache()
    initial=np.load(args.source/'initial_graph.npz',allow_pickle=False)
    if tuple(meta['concept_ids'])!=tuple(str(c) for c in initial['concept_ids']):raise ValueError('concept order mismatch')
    (args.output/'private_anchor_map.json').write_text(json.dumps({'label_indices':list(range(8)),'concept_ids':[[meta['concept_ids'][i] for i in row] for row in anchors]},indent=2))
    protocol={'seeds':[11,17,23],'arms':['no_edge','true_graph']+[f'permutation_{s}' for s in range(1001,1006)],
        'anchors':'top8 frozen prototype cosine matches for each original formal diagnosis label; no example labels or validation selection',
        'inputs':'own anchor H mean and conditional predicted anchor H mean from all other concepts',
        'reader':'eight identical-size 128-to-32 GELU label MLPs, zero output initialization; common frozen text logits',
        'normalization':'each arm training mean/std only, floor1e-4','train':4563,'validation':777,
        'steps':300,'learning_rate':.01,'objective':'full-population macro pairwise ranking, temperature1, plus .1 BCE and .1 mean-squared logit correction',
        'clip_gradient':1.,'random':'permute normalized precision, retain original precision diagonal and SPD',
        'test_split_touched':False,'selection':'none, every graph and head seed reported',
        'limitations':'exploratory changed task reader on one frozen backbone; not GRPO or joint statistical projection learning'}
    report={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'protocol':protocol,'runs':[]}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    train=np.load(args.aligned/'train_private_rank_matrices.npy',mmap_mode='r',allow_pickle=False)
    val=np.load(args.aligned/'val_private_rank_matrices.npy',mmap_mode='r',allow_pickle=False)
    if train.shape!=(4563,64,876) or val.shape!=(777,64,876):raise ValueError('population mismatch')
    center=train.mean(0,dtype=np.float64)
    h={s:torch.tensor(np.asarray(v,dtype=np.float32)-center.astype(np.float32),device='cuda:0') for s,v in [('train',train),('val',val)]}
    labels={s:np.asarray([json.loads(l)['labels'] for l in (args.source/'prepared'/f'{s}.jsonl').read_text().splitlines()]) for s in ['train','val']}
    y=torch.tensor(labels['train'],dtype=torch.float64,device='cuda:0')
    text={s:np.load(args.features/f'{s}_private_features.npz',allow_pickle=False)['mean'] for s in ['train','val']}
    mean=text['train'].mean(0);scale=np.maximum(text['train'].std(0),1e-4)
    x={s:torch.tensor((v-mean)/scale,device='cuda:0') for s,v in text.items()}
    bases={}
    for seed in protocol['seeds']:
        head=nn.Linear(x['train'].shape[1],8).cuda()
        head.load_state_dict(torch.load(args.heads/f'seed_{seed}_uniform_private_head.pt',map_location='cuda:0',weights_only=True))
        with torch.no_grad():bases[seed]={s:head(v).detach() for s,v in x.items()}
    theta=initial['Theta'];d=np.sqrt(np.diag(theta));normalized=theta/np.outer(d,d)
    graphs={'no_edge':np.diag(np.diag(theta)),'true_graph':theta}
    for seed in range(1001,1006):
        order=np.random.default_rng(seed).permutation(len(theta));graphs[f'permutation_{seed}']=normalized[np.ix_(order,order)]*np.outer(d,d)
    for name,a in graphs.items():
        features={s:anchor_features(v,a,anchors) for s,v in h.items()}
        mu=features['train'].mean(0);sd=features['train'].std(0,correction=0).clamp_min(1e-4)
        features={s:(v-mu)/sd for s,v in features.items()}
        np.savez_compressed(args.output/f'{name}_private_feature_normalization.npz',mean=mu.cpu().numpy(),scale=sd.cpu().numpy())
        for seed in protocol['seeds']:
            torch.manual_seed(seed);model=AnchorResidual(features['train'].shape[-1]).cuda();opt=torch.optim.Adam(model.parameters(),lr=.01);start=time.monotonic()
            for step in range(300):
                correction=model(features['train']);z=bases[seed]['train']+correction
                rank_loss,coefficient=macro_pairwise_logit_gradient(z,y)
                loss=(z.double()*coefficient).sum()+.1*correction.square().mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                if step in [0,99,299]:print(json.dumps({'stage':'anchor_training','graph':name,'seed':seed,'step':step,'ranking_surrogate':rank_loss,'correction_std':float(correction.detach().std())}),flush=True)
            row={'seed':seed,'graph':name,'seconds':time.monotonic()-start}
            with torch.no_grad():
                for split in ['train','val']:
                    z=(bases[seed][split]+model(features[split])).cpu().numpy();metrics,losses,prob=multilabel_metrics(labels[split],z)
                    row[split]=dict(metrics.extra_metrics)
                    np.savez_compressed(args.output/f'{name}_seed_{seed}_{split}_private_scores.npz',labels=labels[split],probabilities=prob,per_example_loss=losses)
            torch.save(model.state_dict(),args.output/f'{name}_seed_{seed}_private_reader.pt')
            report['runs'].append(row);(args.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({'stage':'anchor_complete',**row}),flush=True)


if __name__=='__main__':main()
