"""Private-server pipeline: clinical H -> MNGM -> multilabel warmup -> linear GRPO."""
from __future__ import annotations
import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace, asdict
from pathlib import Path
import json, time, subprocess, sys, random
import numpy as np


def clean(o):
    if is_dataclass(o): return {f.name:clean(getattr(o,f.name)) for f in fields(o)}
    if isinstance(o,Mapping): return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [clean(v) for v in o]
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,np.generic): return clean(o.item())
    if isinstance(o,float) and not np.isfinite(o): return None
    if isinstance(o,Path): return str(o)
    return o


def write(path,o):
    Path(path).write_text(json.dumps(clean(o),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def compact_phase(record):
    compact={'phase':record.phase,'accepted':record.accepted,'adapted':record.adapted,
             'parent':record.parent_state_id,'next':record.next_state_id,'proposal':record.proposal_id,
             'acceptance_reason':record.acceptance_reason,'validation_baseline':record.validation_baseline,
             'validation_proposal':record.validation_proposal,'updates':[]}
    for u in record.updates:
        compact['updates'].append({'update':u.update,'policy_update':u.policy_update,
            'rewards':u.rewards,'candidates':[{'id':c.candidate_id,'valid':c.valid,'error':c.error,
            'coefficients':c.action.coefficients,'graph_metrics':c.graph_metrics} for c in u.candidates]})
    return clean(compact)


def run(args,manifest,progress):
    from graph_mvp.north_medical import prepare_north_data, LABEL_NAMES
    from graph_mvp.config import Config
    from graph_mvp.patient_repr import PatientMatrixDataset,load_concept_prototypes
    from graph_mvp.estimators import MNGMEstimator,PATIENT_MATRIX_MODE
    from graph_mvp.environment import GraphEnvironment,graph_metrics
    from graph_mvp.types import GraphState
    from graph_mvp.linear_policy import LinearCandidateBuilder,LinearGRPOPolicy
    from graph_mvp.reward import RewardFunction
    from graph_mvp.runner import GraphPhaseRunner
    from graph_mvp.task_model import load_local_causal_lm,attach_task_lora
    from graph_mvp.task_prototypes import build_task_soft_graph_tokenizer
    from graph_mvp.multilabel_task import GraphMultilabelClassifier,MultilabelContext,MultilabelEvaluator,adapt_multilabel
    import torch
    root=Path(__file__).resolve().parents[1]
    prepared=args.output/'prepared'
    manifest['stage']='prepare';write(args.output/'run_manifest.json',manifest)
    audit=prepare_north_data(args.raw_dir,args.concepts_csv,prepared,seed=args.seed)
    progress({'stage':'data_prepared',**{k:audit[k] for k in ['merged_train','task_train','reward','validation','concepts']}})
    prototypes=prepared/'concept_prototypes.npz';cache=args.output/'cache'
    commands=[('prototypes',[sys.executable,'-u','-m','scripts.build_concept_prototypes',
               '--concepts',str(prepared/'concepts.json'),'--model',args.embedding_model,
               '--local-files-only','--dtype','bfloat16','--output',str(prototypes)]),
              ('build_h',[sys.executable,'-u','-m','scripts.build_patient_matrices',
               '--train',str(prepared/'graph_train.csv.gz'),'--id-col','sample_id',
               '--prototypes',str(prototypes),'--model',args.embedding_model,'--local-files-only',
               '--dtype','bfloat16','--cache-dtype','float32','--representation-dim','64',
               '--representation-mode','joint_evidence','--evidence-top-k','2','--chunk-max-tokens','64',
               '--max-length','512','--encode-batch-size','8','--batch-size','1','--shard-size','16',
               '--compact-evidence-audit','--output',str(cache)])]
    for stage,command in commands:
        manifest['stage']=stage;write(args.output/'run_manifest.json',manifest)
        progress({'stage':stage+'_start'})
        with (args.output/(stage+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        progress({'stage':stage+'_complete'})
    ds=PatientMatrixDataset(cache)
    cfg=Config.load(args.config)
    cfg=replace(cfg,solver=replace(cfg.solver,max_iter=10000))
    manifest.update(stage='initial_graph',effective_solver=asdict(cfg.solver),data_audit=audit,
                    cache_fingerprint=ds.fingerprint(),shape=[len(ds),ds.hidden_size,ds.num_concepts])
    write(args.output/'run_manifest.json',manifest)
    estimator=MNGMEstimator(ds.materialize(dtype=np.float32),PATIENT_MATRIX_MODE,cfg.mngm,cfg.solver)
    env=GraphEnvironment(config=cfg.environment,estimator=estimator)
    initial=env.initialize_from_estimator(.8,ds.concept_ids,progress_callback=progress)
    fitted=estimator.solve(initial.snapshot.Lambda,initial_theta=initial.snapshot.Theta,
                          initial_representation_precision=initial.snapshot.auxiliary['representation_precision'],
                          progress_callback=progress)
    if not fitted.converged: raise RuntimeError('initial alternating MNGM did not converge: '+fitted.message)
    snapshot=env._snapshot_from_estimator_result(ds.concept_ids,initial.snapshot.Lambda,fitted)
    state0=GraphState('north-fitted-state-0',0,snapshot,fitted.info())
    def save_graph(path,state):
        s=state.snapshot
        np.savez_compressed(path,concept_ids=s.concept_ids,S=s.S,Lambda=s.Lambda,Theta=s.Theta,
                            partial_corr=s.Rho,representation_precision=s.auxiliary['representation_precision'])
    save_graph(args.output/'initial_graph.npz',state0)
    manifest['stage']='load_task_model';write(args.output/'run_manifest.json',manifest)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    lm,tok=load_local_causal_lm(args.task_model,device_map='cuda:0',dtype='bfloat16',local_files_only=True)
    lm=attach_task_lora(lm,r=16,alpha=32,dropout=.05)
    graph_tok=build_task_soft_graph_tokenizer(lm,tok,[c['concept_text'] for c in json.loads((prepared/'concepts.json').read_text())],num_tokens=8)
    model=GraphMultilabelClassifier(lm,graph_tok)
    def context(name,cap=None):
        rows=[json.loads(l) for l in (prepared/(name+'.jsonl')).read_text().splitlines()]
        # Random subset, fixed once before policy search; never chosen by validation performance.
        if cap and len(rows)>cap:
            indices=np.random.default_rng(args.seed).permutation(len(rows))[:cap]
            rows=[rows[i] for i in indices]
        return MultilabelContext(ds.concept_ids,tuple(r['text'] for r in rows),np.asarray([r['labels'] for r in rows]),tok,batch_size=4,max_length=512)
    train=context('train');reward=context('reward');val=context('val')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    manifest['stage']='task_warmup';write(args.output/'run_manifest.json',manifest)
    warmup=adapt_multilabel(model,snapshot.Rho,train,optimizer,epochs=args.warmup_epochs,seed=args.seed,progress=progress)
    del optimizer
    lm.save_pretrained(args.output/'task_lora')
    torch.save({'graph_tokenizer':graph_tok.state_dict(),'head':model.head.state_dict()},args.output/'task_modules.pt')
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    evaluator=MultilabelEvaluator(model)
    initial_reward=evaluator.evaluate_snapshot(snapshot,reward)
    initial_val=evaluator.evaluate_snapshot(snapshot,val)
    perm=np.random.default_rng(args.seed).permutation(ds.num_concepts)
    def controls(state,label):
        rho=state.snapshot.Rho
        out={}
        for name,g in [('original',rho),('zero_graph',np.zeros_like(rho)),('permuted_graph',rho[np.ix_(perm,perm)])]:
            m,a,loss,prob=evaluator.score_graph(g,val,True)
            out[name]=dict(m.extra_metrics)
            np.savez_compressed(args.output/f'{label}_{name}_validation_scores.npz',probabilities=prob,labels=val.labels,per_example_loss=loss)
        write(args.output/f'{label}_graph_controls.json',out)
        return out
    initial_controls=controls(state0,'initial')
    protos,meta=load_concept_prototypes(prototypes)
    if tuple(meta['concept_ids'])!=ds.concept_ids: raise ValueError('prototype axis mismatch')
    policy=LinearGRPOPolicy(cfg.grpo,seed=args.seed)
    runner=GraphPhaseRunner(env,policy,evaluator,RewardFunction(cfg.reward),
        candidate_builder=LinearCandidateBuilder(protos,cfg.candidate),validation_evaluator=evaluator,
        min_validation_improvement=cfg.acceptance.min_task_improvement,task_adapter=None,
        feature_provider=lambda state,ctx,baseline:{'task_relevance':evaluator.task_relevance(state.snapshot,ctx)})
    manifest.update(stage='grpo',warmup=warmup,task_frozen=True,test_split_touched=False)
    write(args.output/'run_manifest.json',manifest)
    def report(record):
        # Persist compact candidate actions, not 383k expanded delta entries per candidate.
        write(args.output/f'phase_{record.phase:03d}.json',compact_phase(record))
        policy.save(args.output/'graph_policy.pt')
        progress({'stage':'phase_complete','phase':record.phase+1,'accepted':record.accepted,'reason':record.acceptance_reason})
    result=runner.run(state0,reward,val,phases=2,policy_updates_per_phase=2,num_candidates=4,
                      on_phase=report,progress_callback=progress)
    save_graph(args.output/'final_graph.npz',result.final_state)
    final_controls=controls(result.final_state,'final')
    write(args.output/'summary.json',{'dataset':'North Medical','label_names':LABEL_NAMES,'warmup':warmup,
        'task_frozen_during_graph_search':True,'task_adapt_steps':0,'accepted_phases':sum(p.accepted for p in result.phases),
        'initial_graph':graph_metrics(snapshot),'final_graph':graph_metrics(result.final_state.snapshot),
        'initial_reward':initial_reward,'final_reward':evaluator.evaluate_snapshot(result.final_state.snapshot,reward),
        'initial_validation':initial_val,'final_validation':evaluator.evaluate_snapshot(result.final_state.snapshot,val),
        'initial_controls':initial_controls,'final_controls':final_controls,'test_split_touched':False,
        'warning':'single-seed exploratory run; validation selects graph, no matched random-policy control'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,default=Path('/laijizheng/QwenGAT/ch_data'))
    p.add_argument('--concepts-csv',type=Path,default=Path('/laijizheng/QwenGAT/具体关键词.csv'))
    p.add_argument('--embedding-model',default='/laijizheng/models/Qwen3-Embedding-0.6B')
    p.add_argument('--task-model',default='/laijizheng/models/GLM-4.7-Flash')
    p.add_argument('--config',type=Path,default=Path('configs/patient_h04l_main_v1.json'))
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--warmup-epochs',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.warmup_epochs<1: p.error('warmup epochs must be positive')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    manifest={'status':'running','stage':'starting','git_sha':revision,'args':vars(args),'test_split_touched':False}
    def progress(info):
        row=clean({**info,'seconds':time.monotonic()-start})
        with (args.output/'progress.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        if 'iteration' not in row or row['iteration']==1 or row['iteration']%250==0:
            print(json.dumps(row,ensure_ascii=False),flush=True)
    write(args.output/'run_manifest.json',manifest)
    try:
        run(args,manifest,progress);manifest.update(status='complete',stage='complete')
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['runtime_seconds']=time.monotonic()-start;write(args.output/'run_manifest.json',manifest)


if __name__=='__main__': main()
