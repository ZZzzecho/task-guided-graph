"""Read the same patient-concept matrices used by MNGM; no test access."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from graph_mvp.patient_repr import PatientMatrixDataset
from graph_mvp.data import RankGaussianTransformer
from graph_mvp.multilabel_task import multilabel_metrics


class AlignedGraphResidual(nn.Module):
    def __init__(self,p,r,hidden=8):
        super().__init__()
        self.projection=nn.Linear(r,hidden)
        self.identity=nn.Parameter(torch.randn(p,hidden)*.01)
        self.self_layer=nn.Linear(hidden,hidden)
        self.message_layer=nn.Linear(hidden,hidden,bias=False)
        self.queries=nn.Parameter(torch.randn(8,hidden)/hidden**.5)
        self.label_weights=nn.Parameter(torch.zeros(8,hidden))
        self.label_bias=nn.Parameter(torch.zeros(8))

    def forward(self,h,graph):
        nodes=self.projection(h)+self.identity
        message=torch.sparse.mm(graph,nodes.permute(1,0,2).reshape(len(self.identity),-1))
        message=message.reshape(len(self.identity),len(h),-1).permute(1,0,2)
        nodes=torch.tanh(self.self_layer(nodes)+self.message_layer(message))
        attention=torch.softmax(torch.einsum('bph,lh->blp',nodes,self.queries)/nodes.shape[-1]**.5,dim=-1)
        pooled=torch.einsum('blp,bph->blh',attention,nodes)
        return (pooled*self.label_weights).sum(-1)+self.label_bias


def sparse_normalized_graph(rho,device):
    degree=np.abs(rho).sum(1);scale=np.sqrt(np.maximum(degree,1e-8))
    norm=rho/scale[:,None]/scale[None,:];i,j=np.nonzero(norm)
    return torch.sparse_coo_tensor(np.stack([i,j]),norm[i,j],rho.shape,dtype=torch.float32,device=device,check_invariants=True).coalesce()


def rank_cache(source,val_cache,output):
    train_ds=PatientMatrixDataset(source/'cache');val_ds=PatientMatrixDataset(val_cache)
    if len(train_ds)!=4563 or len(val_ds)!=777:raise ValueError('population mismatch')
    if train_ds.concept_ids!=val_ds.concept_ids or train_ds.metadata['projection_sha256']!=val_ds.metadata['projection_sha256']:
        raise ValueError('representation identity mismatch')
    if val_ds.metadata['source_split']!='validation':raise ValueError('validation cache must be explicitly identified')
    expected=[str(json.loads(line)['id'])+f'-{i}' for i,line in enumerate((source/'prepared/train.jsonl').read_text().splitlines())]
    actual=[rec['subject_id'] for _,records in train_ds.iter_shards() for rec in records]
    if actual!=expected:raise ValueError('training record order mismatch')
    graph_ids=tuple(str(c) for c in np.load(source/'initial_graph.npz',allow_pickle=False)['concept_ids'])
    if graph_ids!=train_ds.concept_ids:raise ValueError('statistical graph order mismatch')
    # Record ordering must match the validation JSONL used to build this cache.
    val_records=[rec['subject_id'] for _,records in val_ds.iter_shards() for rec in records]
    if val_records!=list(range(777)):raise ValueError('validation record order mismatch')
    matrices={'train':train_ds.materialize(),'val':val_ds.materialize()};paths={}
    flat={s:h.reshape(len(h),-1) for s,h in matrices.items()}
    transformed={s:np.lib.format.open_memmap(output/f'{s}_private_rank_matrices.npy',mode='w+',dtype='float32',shape=h.shape) for s,h in matrices.items()}
    out={s:a.reshape(len(a),-1) for s,a in transformed.items()}
    for offset in range(0,flat['train'].shape[1],256):
        sl=slice(offset,offset+256);transformer=RankGaussianTransformer().fit(flat['train'][:,sl])
        for s in ['train','val']:out[s][:,sl]=transformer.transform(flat[s][:,sl])
        if offset%8192==0:print(json.dumps({'stage':'training_ecdf','coordinates':offset}),flush=True)
    for s,a in transformed.items():a.flush();paths[s]=output/f'{s}_private_rank_matrices.npy'
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','val-cache','features','heads','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    protocol={'seeds':[11,17,23],'arms':['text','concept','graph','permuted'],'train':4563,'val':777,
        'representation':'same cached joint H and exact training projection; training-ECDF Gaussianization per matrix cell',
        'receiver':'shared trainable 64-to-8 task projection, concept identities, signed normalized graph message, label attention',
        'projection_limitation':'reader projection only; statistical basis stays fixed, not jointly optimized MNGM embedding',
        'frozen_text_head':'same seed uniform-BCE cached mean-feature head','epochs':5,'batch_size':128,
        'learning_rate':.003,'residual_logit_penalty':1.,'weight_penalty':.001,
        'selection':'none; report every seed/arm','test_split_touched':False,
        'limitation':'one frozen backbone; classifier seeds only, exploration validation, not GRPO'}
    report={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'protocol':protocol,'runs':[]}
    (args.output/'protocol.json').write_text(json.dumps(report,indent=2))
    paths=rank_cache(args.source,args.val_cache,args.output)
    device='cuda:0';h={s:torch.tensor(np.load(path,allow_pickle=False).transpose(0,2,1).copy(),device=device) for s,path in paths.items()}
    features={s:np.load(args.features/f'{s}_private_features.npz',allow_pickle=False)['mean'] for s in ['train','val']}
    center=features['train'].mean(0);scale=np.maximum(features['train'].std(0),1e-4)
    x={s:torch.tensor((f-center)/scale,device=device) for s,f in features.items()}
    labels={s:np.asarray([json.loads(l)['labels'] for l in (args.source/'prepared'/f'{s}.jsonl').read_text().splitlines()]) for s in ['train','val']}
    y=torch.tensor(labels['train'],dtype=torch.float32,device=device)
    rho=np.load(args.source/'initial_graph.npz',allow_pickle=False)['partial_corr'];permutation=np.random.default_rng(1001).permutation(len(rho))
    graphs={'concept':sparse_normalized_graph(np.zeros_like(rho),device),'graph':sparse_normalized_graph(rho,device),
            'permuted':sparse_normalized_graph(rho[np.ix_(permutation,permutation)],device)}
    for seed in protocol['seeds']:
        head=nn.Linear(x['train'].shape[1],8).to(device)
        head.load_state_dict(torch.load(args.heads/f'seed_{seed}_uniform_private_head.pt',map_location=device,weights_only=True))
        with torch.no_grad():base={s:head(v).detach() for s,v in x.items()}
        for arm in protocol['arms']:
            torch.manual_seed(seed);reader=AlignedGraphResidual(h['train'].shape[1],h['train'].shape[2]).to(device);start=time.monotonic()
            if arm!='text':
                opt=torch.optim.Adam(reader.parameters(),lr=protocol['learning_rate']);rng=np.random.default_rng(seed)
                for epoch in range(protocol['epochs']):
                    order=rng.permutation(4563)
                    for begin in range(0,4563,128):
                        index=torch.tensor(order[begin:begin+128],device=device)
                        residual=reader(h['train'][index],graphs[arm]);z=base['train'][index]+residual
                        loss=F.binary_cross_entropy_with_logits(z,y[index])+residual.square().mean()+.001*reader.label_weights.square().mean()
                        if not torch.isfinite(loss):raise RuntimeError('nonfinite aligned reader loss')
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(reader.parameters(),1.);opt.step()
            row={'seed':seed,'arm':arm,'seconds':time.monotonic()-start}
            for split in ['train','val']:
                zs=[]
                with torch.no_grad():
                    for begin in range(0,len(h[split]),128):
                        sl=slice(begin,begin+128);z=base[split][sl]
                        if arm!='text':z=z+reader(h[split][sl],graphs[arm])
                        zs.append(z.cpu().numpy())
                metrics,losses,prob=multilabel_metrics(labels[split],np.concatenate(zs));row[split]=dict(metrics.extra_metrics)
                np.savez_compressed(args.output/f'seed_{seed}_{arm}_{split}_private_scores.npz',labels=labels[split],probabilities=prob,per_example_loss=losses)
            torch.save(reader.state_dict(),args.output/f'seed_{seed}_{arm}_private_reader.pt')
            report['runs'].append(row);(args.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({'stage':'aligned_reader_complete',**row}),flush=True)


if __name__=='__main__':main()
