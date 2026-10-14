"""Paired record uncertainty for every completed graph-prior arm, no test data."""
import argparse,json
from pathlib import Path
import numpy as np


def weighted_macro_metrics(y,probability,weights):
    """Exact weighted macro AP (ties grouped) and threshold-.5 macro F1."""
    y=np.asarray(y,dtype=bool);p=np.asarray(probability);w=np.asarray(weights,dtype=np.float64)
    if y.shape!=p.shape or w.ndim!=2 or w.shape[1]!=len(y):raise ValueError('score dimensions differ')
    aps=[];present=w@y.astype(float)>0
    for j in range(y.shape[1]):
        order=np.argsort(-p[:,j],kind='stable');value=p[order,j]
        endpoint=np.r_[np.flatnonzero(value[:-1]!=value[1:]),len(value)-1]
        tp=np.cumsum(w[:,order]*y[order,j],axis=1)[:,endpoint]
        total=np.cumsum(w[:,order],axis=1)[:,endpoint]
        precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
        increments=np.diff(np.c_[np.zeros(len(w)),tp],axis=1)
        ap=np.divide((increments*precision).sum(1),tp[:,-1],out=np.zeros(len(w)),where=tp[:,-1]>0)
        aps.append(ap)
    ap=(np.stack(aps,axis=1)*present).sum(1)/present.sum(1)
    prediction=p>=.5;tp=w@(y&prediction).astype(float);fp=w@(~y&prediction).astype(float);fn=w@(y&~prediction).astype(float)
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=2*tp+fp+fn>0).mean(1)
    return np.stack([ap,f1],axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--heads',type=Path,required=True)
    args=p.parse_args();s=json.loads((args.source/'summary.json').read_text());seeds=s['protocol']['seeds'];arms=s['protocol']['arms']
    if len(s['runs'])!=len(seeds)*len(arms) or s['protocol']['test_split_touched']:raise ValueError('incomplete or contaminated experiment')
    weights=np.zeros((1001,777),dtype=np.float64);weights[0]=1;rng=np.random.default_rng(41)
    for b in range(1,1001):weights[b]=np.bincount(rng.integers(777,size=777),minlength=777)
    metrics={};reference_labels=None
    for graph in arms+['frozen_text']:
        per_seed=[]
        for seed in seeds:
            path=args.heads/f'seed_{seed}_uniform_val_private_scores.npz' if graph=='frozen_text' else args.source/f'{graph}_seed_{seed}_val_private_scores.npz'
            a=np.load(path,allow_pickle=False)
            if reference_labels is None:reference_labels=a['labels'].copy()
            elif not np.array_equal(reference_labels,a['labels']):raise ValueError('record labels or ordering differ across arms')
            per_seed.append(weighted_macro_metrics(a['labels'],a['probabilities'],weights))
        metrics[graph]=np.mean(per_seed,axis=0)
    truth=metrics['true_graph'];nulls=[g for g in arms if g.startswith('permutation_')]
    controls={'diagonal':metrics['diagonal'],'random_mean':np.mean([metrics[g] for g in nulls],axis=0),'frozen_text':metrics['frozen_text']}
    choices=['diagonal','frozen_text']+nulls;all_controls=np.stack([metrics[g] for g in choices],axis=1)
    best=np.argmax(all_controls[:,:,0],axis=1);controls['strongest_AP_control']=all_controls[np.arange(len(weights)),best]
    report={'test_split_touched':False,'means':{g:dict(zip(['macro_AP','macro_F1'],m[0].tolist())) for g,m in metrics.items()},
        'comparisons':{},'uncertainty':'1000 paired RECORD bootstrap draws, average metrics over three fixed heads, not independent backbone or test uncertainty',
        'warning':'exploration validation reused; patient IDs unavailable; no positive efficacy claim from a small nominal mean'}
    for name,control in controls.items():
        delta=truth-control
        report['comparisons'][name]={metric:{'delta':float(delta[0,j]),'descriptive_95_interval':np.quantile(delta[1:,j],[.025,.975]).tolist()} for j,metric in enumerate(['macro_AP','macro_F1'])}
    (args.source/'paired_record_uncertainty.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
