"""Full-training directed versus matched-random conditional penalty search.

Freeze a stabilized task model; final selected penalties receive a full alternating
MNGM refit. Never select with validation or read the test split.
"""
import argparse,json,random,subprocess,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from graph_mvp.config import Config,SolverConfig
from graph_mvp.task_model import load_local_causal_lm,attach_task_lora
from graph_mvp.task_prototypes import build_task_soft_graph_tokenizer
from graph_mvp.north_validation import MechanismClassifier,contexts_from_prepared,restore_trainables
from graph_mvp.multilabel_task import MultilabelEvaluator
from graph_mvp.implicit_glasso import implicit_penalty_gradient,macro_pairwise_logit_gradient
from graph_mvp.weighted_glasso import WeightedGraphicalLasso,statistical_loss
from graph_mvp.patient_repr import PatientMatrixDataset
from graph_mvp.estimators import MNGMEstimator,PATIENT_MATRIX_MODE


def correlation(theta):
    scale=np.sqrt(np.diag(theta));rho=-theta/np.outer(scale,scale);np.fill_diagonal(rho,0.)
    return rho


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    initial=np.load(args.source/'initial_graph.npz',allow_pickle=False)
    report=json.loads((args.controls/f'seed_{args.seed}_graph.json').read_text())
    protocol=json.loads((args.controls/'protocol.json').read_text())
    if not protocol['fp32_heads'] or protocol['readout']!='mean':raise ValueError('requires stabilized mean/FP32 checkpoint')
    if report['validation']['macro_average_precision']<.5:raise ValueError('failed task-training health check')
    policy={'seed':args.seed,'steps':[.1,.3,.7],'action_support':'initial |partial correlation| >= .001',
        'direction':'negative macro pairwise logistic ranking gradient through the active-set Glasso adjoint',
        'ranking_temperature':1.,'bce_weight':.1,'gradient_scale':'95th percentile, clipped to [-1,1]',
        'random_control':'one permutation of directed edge values, identical support and step magnitudes',
        'global_penalties':[.2,.4,1.2],'conditional_nll_budget_per_concept':.01,'edge_limit_factor':2,
        'selection':'all 4563 training macro-AP; baseline included in each family',
        'final':'full alternating MNGM refit of training-selected penalty; recheck full NLL/complexity/train AP',
        'test_split_touched':False,'warning':'gradient optimizer, not GRPO; support is empirical fit, not confidence',
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    (args.output/'protocol.json').write_text(json.dumps(policy,indent=2))
    manifest=json.loads((args.source/'run_manifest.json').read_text())
    concepts=json.loads((args.source/'prepared/concepts.json').read_text());ids=tuple(str(c) for c in initial['concept_ids'])
    if ids!=tuple(c['concept_id'] for c in concepts):raise ValueError('concept order mismatch')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    lm,tok=load_local_causal_lm(manifest['args']['task_model'],device_map='cuda:0',dtype='bfloat16',local_files_only=True)
    lm=attach_task_lora(lm,r=16,alpha=32,dropout=.05)
    model=MechanismClassifier(lm,build_task_soft_graph_tokenizer(lm,tok,[c['concept_text'] for c in concepts]),fp32_heads=True,readout='mean')
    restore_trainables(model,torch.load(args.controls/f'seed_{args.seed}_graph_trainables.pt',map_location='cpu',weights_only=True))
    for par in model.parameters():par.requires_grad_(False)
    model.eval();contexts=contexts_from_prepared(args.source/'prepared',ids,tok)
    if len(contexts['train'].texts)!=4563 or len(contexts['val'].texts)!=777:raise ValueError('population mismatch')
    evaluator=MultilabelEvaluator(model);rows=[];penalties={'initial':initial['Lambda']}
    upper=np.triu_indices(len(ids),1);baseline_edges=int((np.abs(initial['partial_corr'][upper])>1e-5).sum())
    baseline_stat=statistical_loss(initial['S'],initial['Theta']);start=time.monotonic()
    def log(row):print(json.dumps({**row,'seconds':time.monotonic()-start}),flush=True)
    def score(name,rho,extra):
        row={'name':name,**extra}
        for split in ['train','val']:
            metrics,_,loss,prob=evaluator.score_graph(rho,contexts[split],True)
            row[split]=dict(metrics.extra_metrics)
            np.savez_compressed(args.output/f'{name}_{split}_private_scores.npz',probabilities=prob,labels=contexts[split].labels,per_example_loss=loss)
        rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2));log({'stage':'candidate_complete',**row})
        return row
    baseline=score('initial',initial['partial_corr'],{'family':'baseline','admissible':True,'valid':True})
    if abs(baseline['val']['macro_average_precision']-report['validation']['macro_average_precision'])>1e-5:
        raise RuntimeError('checkpoint reproduction differs from the matched control')
    # First pass on the entire training population obtains logit coefficients.
    zs=[]
    with torch.no_grad():
        for batch,_ in contexts['train'].batches(model.device):zs.append(model(**batch,global_graph=initial['partial_corr']).double())
    logits=torch.cat(zs);labels=torch.tensor(contexts['train'].labels.copy(),device=model.device)
    surrogate,coefficients=macro_pairwise_logit_gradient(logits,labels)
    theta=torch.tensor(initial['Theta'],dtype=torch.float64,device=model.device,requires_grad=True)
    offset=0
    for step,(batch,_) in enumerate(contexts['train'].batches(model.device)):
        scale=torch.diag(theta).sqrt();rho=-theta/(scale[:,None]*scale[None,:]);rho=rho-torch.diag(torch.diag(rho))
        z=model(**batch,global_graph=rho);n=len(z)
        (z.double()*coefficients[offset:offset+n]).sum().backward();offset+=n
        if step%100==0:log({'stage':'full_train_precision_gradient','step':step})
    if offset!=4563:raise RuntimeError('precision gradient did not cover the full training population')
    np.savez_compressed(args.output/'private_precision_task_gradient.npz',Theta=initial['Theta'],
                        task_gradient=theta.grad.cpu().numpy(),surrogate=surrogate)
    adjoint=implicit_penalty_gradient(theta.detach(),theta.grad,device=model.device,tolerance=1e-6,max_iter=1000)
    log({'stage':'adjoint_complete','converged':adjoint.converged,'iterations':adjoint.iterations,'relative_residual':adjoint.relative_residual,'surrogate':surrogate})
    if not adjoint.converged:raise RuntimeError('implicit adjoint did not converge; no uncertified direction used')
    mask=np.abs(initial['partial_corr'][upper])>=.001
    gradient=adjoint.gradient.cpu().numpy()[upper]*initial['Lambda'][upper]
    normalization=float(np.quantile(np.abs(gradient[mask]),.95))
    if normalization<=1e-16:raise RuntimeError('task gradient is too small to define a direction')
    direction=np.zeros(len(gradient));direction[mask]=np.clip(-gradient[mask]/normalization,-1.,1.)
    shuffled=direction.copy();shuffled[mask]=np.random.default_rng(args.seed+1001).permutation(direction[mask])
    np.savez_compressed(args.output/'private_gradient_direction.npz',direction=direction,random_direction=shuffled,eligible=mask)
    del logits,labels,coefficients,theta,adjoint;torch.cuda.empty_cache()
    proposals=[]
    for family,vector in [('directed',direction),('random',shuffled)]:
        for step in policy['steps']:
            lam=initial['Lambda'].copy();values=np.clip(lam[upper]*np.exp(step*vector),.05,3.)
            lam[upper]=values;lam[(upper[1],upper[0])]=values
            proposals.append((f'{family}_{step:g}',family,lam))
    for value in policy['global_penalties']:
        lam=np.full_like(initial['Lambda'],value);np.fill_diagonal(lam,0)
        proposals.append((f'global_{value:g}','global',lam))
    for name,family,lam in proposals:
        def progress(row):
            if row.get('iteration',0)%250==0:log({**row,'candidate':name})
        with threadpool_limits(limits=32,user_api='blas'):
            result=WeightedGraphicalLasso(SolverConfig(max_iter=10000,rho=10.)).solve(initial['S'],lam,initial_theta=initial['Theta'],progress_callback=progress)
            if not result.converged:
                result=WeightedGraphicalLasso(SolverConfig(max_iter=10000,rho=1.)).solve(initial['S'],lam,initial_theta=initial['Theta'],adaptive_rho=True,progress_callback=progress)
        if not result.converged:
            row={'name':name,'family':family,'valid':False,'admissible':False,'solver':result.info()};rows.append(row);log({'stage':'candidate_failed',**row});continue
        rho=correlation(result.Theta);edges=int((np.abs(rho[upper])>1e-5).sum())
        delta=(statistical_loss(initial['S'],result.Theta)-baseline_stat)/len(ids)
        penalties[name]=lam
        np.savez_compressed(args.output/f'{name}_conditional_graph.npz',Theta=result.Theta,Lambda=lam,partial_corr=rho)
        score(name,rho,{'family':family,'valid':True,'admissible':delta<=.01 and edges<=2*baseline_edges,
            'conditional_nll_delta_per_concept':delta,'num_edges':edges,'solver':result.info()})
    chosen={family:max([baseline]+[r for r in rows if r.get('family')==family and r['admissible']],key=lambda r:r['train']['macro_average_precision'])['name'] for family in ['directed','random','global']}
    log({'stage':'conditional_selection_train_only','chosen':chosen})
    # Refine each family's selected penalty, never its validation winner.
    cfg=Config.load(manifest['args']['config']);ds=PatientMatrixDataset(args.source/'cache')
    with threadpool_limits(limits=32,user_api='blas'):
        estimator=MNGMEstimator(ds.materialize(dtype=np.float32),PATIENT_MATRIX_MODE,cfg.mngm,replace(cfg.solver,max_iter=10000,rho=10.),covariance_device='cuda:0')
    full_baseline=estimator._statistical_loss(initial['Theta'],initial['representation_precision']);final={}
    for family,name in chosen.items():
        if name=='initial':final[family]={'selected':'initial','accepted':False,'reason':'baseline selected by training AP'};continue
        with threadpool_limits(limits=32,user_api='blas'):
            result=estimator.solve(penalties[name],initial_theta=initial['Theta'],initial_representation_precision=initial['representation_precision'],progress_callback=lambda r:log({**r,'family':family}))
        if not result.converged:
            final[family]={'selected':'initial','accepted':False,'reason':'full MNGM refinement failed','estimator':result.info()};continue
        rho=correlation(result.Theta);delta=(result.stat_objective-full_baseline)/len(ids)
        edges=int((np.abs(rho[upper])>1e-5).sum())
        row=score(f'{family}_refined',rho,{'family':family,'mode':'full_MNGM_refinement','valid':True,
            'admissible':delta<=.01 and edges<=2*baseline_edges,'full_nll_delta_per_concept':delta,'num_edges':edges})
        accepted=row['admissible'] and row['train']['macro_average_precision']>baseline['train']['macro_average_precision']+1e-8
        final[family]={'selected':row['name'] if accepted else 'initial','accepted':accepted,'conditional_selected':name}
        np.savez_compressed(args.output/f'{family}_refined_graph.npz',Theta=result.Theta,Lambda=penalties[name],S=result.S,partial_corr=rho,
            representation_precision=result.auxiliary['representation_precision'])
    (args.output/'summary.json').write_text(json.dumps({'protocol':policy,'runs':rows,'conditional_selected':chosen,
        'final_train_only':final,'test_split_touched':False,'warning':'validation exploration, no final claim'},indent=2))


if __name__=='__main__':main()
