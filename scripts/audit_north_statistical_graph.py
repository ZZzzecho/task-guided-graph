"""Held-out graph density and leave-one-concept-out reconstruction, not diagnosis."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch


def precision_graph_scores(x,theta,b):
    """Per-record MSE, joint NLL and conditional pseudo-NLL per scalar cell.

    Each concept is predicted from all OTHER concepts in that record. Rows follow
    the matrix-normal model; no label or held-out value is used in its own mean.
    """
    theta=torch.as_tensor(theta,dtype=x.dtype,device=x.device);b=torch.as_tensor(b,dtype=x.dtype,device=x.device)
    n,r,p=x.shape
    if theta.shape!=(p,p) or b.shape!=(r,r):raise ValueError('precision dimensions differ')
    chol=torch.linalg.cholesky(theta);bchol=torch.linalg.cholesky(b)
    loga=2*chol.diag().log().sum();logb=2*bchol.diag().log().sum()
    indices=theta.nonzero().T;values=theta[indices[0],indices[1]]
    sparse=torch.sparse_coo_tensor(indices,values,theta.shape,check_invariants=True).coalesce()
    ax=torch.sparse.mm(sparse,x.reshape(n*r,p).T).T.reshape(n,r,p)
    residual=ax/theta.diag()[None,None,:]
    bx=torch.einsum('ab,nbp->nap',b,x)
    br=torch.einsum('ab,nbp->nap',b,residual)
    normalizer=r*p*np.log(2*np.pi)
    joint=((bx*ax).sum((1,2))-r*loga-p*logb+normalizer)/(2*r*p)
    conditional=((br*residual).sum(1).mul(theta.diag()).sum(1)-r*theta.diag().log().sum()-p*logb+normalizer)/(2*r*p)
    return {'reconstruction_mse':residual.square().mean((1,2)),
        'joint_nll_per_cell':joint,'conditional_pseudo_nll_per_cell':conditional}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','aligned','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    initial=np.load(args.source/'initial_graph.npz',allow_pickle=False)
    train=np.load(args.aligned/'train_private_rank_matrices.npy',mmap_mode='r',allow_pickle=False)
    val=np.load(args.aligned/'val_private_rank_matrices.npy',mmap_mode='r',allow_pickle=False)
    if train.shape!=(4563,64,876) or val.shape!=(777,64,876):raise ValueError('population or representation mismatch')
    center=train.mean(0,dtype=np.float64);b=initial['representation_precision'];theta=initial['Theta']
    # Conditional no-edge fit under the SAME fixed row precision, using training
    # covariance only. Keep the source fit's numerical covariance ridge.
    centered_variance=np.diag(initial['S'])-np.einsum('ri,rs,si->i',center,b,center)/64
    diagonal=np.diag(1/np.maximum(centered_variance,1e-4))
    d=np.sqrt(np.diag(theta));normalized=theta/np.outer(d,d)
    graphs={'initial_MNGM':theta,'diagonal_train_fit':diagonal}
    for seed in range(1001,1006):
        order=np.random.default_rng(seed).permutation(len(theta))
        graphs[f'permutation_{seed}']=normalized[np.ix_(order,order)]*np.outer(d,d)
    report={'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol':{'train':4563,'validation':777,'test_split_touched':False,
            'inputs':'exact training-projection H and training-only per-cell Gaussian ECDF',
            'mean':'per-cell mean fitted on training rank matrices only, common to all models',
            'row_precision':'initial MNGM B fixed in every model',
            'no_edge':'diagonal concept precision fitted on training covariance at that fixed B',
            'random':'five permutations of normalized concept precision; preserve original precision diagonal, partial edge weights and SPD',
            'reconstruction':'leave each concept column out, predict from all remaining concept columns',
            'metrics':['reconstruction MSE','joint Gaussian NLL per cell','conditional Gaussian pseudo-NLL per cell'],
            'bootstrap':'1000 paired record resamples; descriptive, patient IDs unavailable',
            'warning':'statistical representation usefulness, not diagnosis or GRPO efficacy'},'runs':[]}
    (args.output/'protocol.json').write_text(json.dumps(report['protocol'],indent=2))
    x=torch.tensor(np.asarray(val,dtype=np.float64)-center,device='cuda:0');scores={};start=time.monotonic()
    with torch.no_grad():
        for name,a in graphs.items():
            values={k:v.cpu().numpy() for k,v in precision_graph_scores(x,a,b).items()}
            scores[name]=values;np.savez_compressed(args.output/f'{name}_private_record_scores.npz',**values)
            row={'name':name,'metrics':{k:float(v.mean()) for k,v in values.items()},'seconds':time.monotonic()-start}
            report['runs'].append(row);print(json.dumps(row),flush=True)
    contrasts={};rng=np.random.default_rng(31);indices=rng.integers(777,size=(1000,777))
    for comparator in ['diagonal_train_fit','permutation_mean']:
        other=scores[comparator] if comparator!='permutation_mean' else {k:np.mean([scores[f'permutation_{s}'][k] for s in range(1001,1006)],axis=0) for k in scores['initial_MNGM']}
        contrasts[comparator]={}
        for metric,values in scores['initial_MNGM'].items():
            delta=values-other[metric];bootstrap=delta[indices].mean(1)
            contrasts[comparator][metric]={'true_minus_control':float(delta.mean()),
                'descriptive_95_interval':np.quantile(bootstrap,[.025,.975]).tolist()}
    report['paired_contrasts']=contrasts
    report['reconstruction_relative_gain_vs_no_edge']=1-scores['initial_MNGM']['reconstruction_mse'].mean()/scores['diagonal_train_fit']['reconstruction_mse'].mean()
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'contrasts':contrasts,'relative_mse_reduction_vs_no_edge':report['reconstruction_relative_gain_vs_no_edge']}),flush=True)


if __name__=='__main__':main()
