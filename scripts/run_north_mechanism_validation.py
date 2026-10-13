"""Reuse a fixed graph and run paired full-data architecture controls."""
from __future__ import annotations
import argparse, hashlib, json, random, subprocess, time
from pathlib import Path
import numpy as np
import torch
from graph_mvp.task_model import load_local_causal_lm, attach_task_lora
from graph_mvp.task_prototypes import build_task_soft_graph_tokenizer
from graph_mvp.multilabel_task import MultilabelEvaluator, adapt_multilabel
from graph_mvp.north_validation import (MechanismClassifier, contexts_from_prepared,
    capture_trainables, restore_trainables, paired_bootstrap)
from scripts.run_north_medical_graph_rl import write


def run(args, progress):
    source=Path(args.source)
    graph=np.load(source/'initial_graph.npz',allow_pickle=False)
    concept_ids=tuple(str(c) for c in graph['concept_ids'])
    rho=graph['partial_corr']
    concepts=json.loads((source/'prepared/concepts.json').read_text())
    if tuple(c['concept_id'] for c in concepts)!=concept_ids: raise ValueError('concept order mismatch')
    model_manifest=json.loads((source/'run_manifest.json').read_text())
    lm_path=model_manifest['args']['task_model']
    protocol={'seeds':args.seeds,'arms':['text','concept','graph'],'epochs':args.epochs,
        'primary':'macro_average_precision','secondary':['macro_f1','micro_f1','bce'],
        'train':4563,'validation':777,'reward_population':'all training if later graph search',
        'graph':'same pre-existing initial MNGM, fixed for this experiment',
        'source_sha':model_manifest['git_sha'],'learning_rate':1e-4,'batch_size':4,
        'max_length':512,'lora':{'r':16,'alpha':32,'dropout':.05},
        'activation':'unchanged legacy, to isolate architecture before altering activation',
        'fp32_heads':args.fp32_heads,'readout':args.readout,
        'training_rng':args.training_rng,
        'test_split_touched':False,'threshold':.5,
        'interpretation':'mechanism controls; not task-guided graph learning evidence'}
    write(args.output/'protocol.json',protocol)
    completed=[]
    # One checkpoint per completed arm; the untouched base model is never copied.
    for seed in args.seeds:
        if all((args.output/f'seed_{seed}_{a}.json').exists() for a in protocol['arms']):
            progress({'stage':'seed_cached','seed':seed}); continue
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        progress({'stage':'load_model','seed':seed})
        lm,tok=load_local_causal_lm(lm_path,device_map='cuda:0',dtype='bfloat16',local_files_only=True)
        lm=attach_task_lora(lm,r=16,alpha=32,dropout=.05)
        graph_tok=build_task_soft_graph_tokenizer(lm,tok,[c['concept_text'] for c in concepts],num_tokens=8)
        model=MechanismClassifier(lm,graph_tok,fp32_heads=args.fp32_heads,readout=args.readout)
        initial=capture_trainables(model)
        post_init_rng=torch.get_rng_state().clone()
        post_init_cuda_rng=torch.cuda.get_rng_state_all()
        contexts=contexts_from_prepared(source/'prepared',concept_ids,tok)
        if len(contexts['train'].texts)!=4563 or len(contexts['val'].texts)!=777:
            raise ValueError('unexpected full-data populations')
        evaluator=MultilabelEvaluator(model)
        for arm in protocol['arms']:
            target=args.output/f'seed_{seed}_{arm}.json'
            if target.exists(): progress({'stage':'arm_cached','seed':seed,'arm':arm}); continue
            restore_trainables(model,initial);model.arm=arm
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if args.training_rng=='post-init':
                torch.set_rng_state(post_init_rng)
                torch.cuda.set_rng_state_all(post_init_cuda_rng)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
            started=time.monotonic()
            progress({'stage':'arm_train_start','seed':seed,'arm':arm})
            warm=adapt_multilabel(model,rho,contexts['train'],optimizer,epochs=args.epochs,seed=seed,
                progress=lambda x:progress({**x,'seed':seed,'arm':arm}))
            del optimizer
            metrics,a,loss,prob=evaluator.score_graph(rho,contexts['val'],True)
            np.savez_compressed(args.output/f'seed_{seed}_{arm}_validation_scores.npz',
                                probabilities=prob,labels=contexts['val'].labels,per_example_loss=loss)
            # Save small trainable checkpoints for later frozen-policy comparisons.
            torch.save(capture_trainables(model),args.output/f'seed_{seed}_{arm}_trainables.pt')
            row={'seed':seed,'arm':arm,'warmup':warm,'validation':dict(metrics.extra_metrics),
                 'runtime_seconds':time.monotonic()-started}
            if arm!='text':
                row['activations']={'mean':float(a.mean()),'std':float(a.std()),
                    'fraction_above_05':float((a>.5).mean()),
                    'mean_patient_activation_std':float(a.std(axis=1).mean())}
            write(target,row);completed.append(row)
            progress({'stage':'arm_complete',**row})
        del model,lm,tok,graph_tok,evaluator,initial,contexts
        torch.cuda.empty_cache()
    rows=[json.loads((args.output/f'seed_{s}_{a}.json').read_text())
          for s in args.seeds for a in protocol['arms']]
    comparisons=[]
    for seed in args.seeds:
        scores={a:np.load(args.output/f'seed_{seed}_{a}_validation_scores.npz') for a in protocol['arms']}
        for left,right in [('concept','text'),('graph','concept'),('graph','text')]:
            for metric in ['macro_ap','macro_f1']:
                comparisons.append({'seed':seed,'left':left,'right':right,'metric':metric,
                    **paired_bootstrap(scores[left]['labels'],scores[left]['probabilities'],
                                       scores[right]['probabilities'],metric,repetitions=1000)})
    write(args.output/'summary.json',{'protocol':protocol,'runs':rows,'paired_comparisons':comparisons,
        'test_split_touched':False,'warning':'validation exploration; no final method claim'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,17,23])
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--fp32-heads',action='store_true')
    parser.add_argument('--readout',choices=['last','mean'],default='last')
    parser.add_argument('--training-rng',choices=['seed','post-init'],default='post-init')
    args=parser.parse_args()
    if args.epochs<1 or len(set(args.seeds))!=len(args.seeds): parser.error('invalid epochs/seeds')
    args.output.mkdir(parents=True,exist_ok=True)
    revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    inputs={name:hashlib.sha256((args.source/name).read_bytes()).hexdigest() for name in
            ['initial_graph.npz','prepared/train.jsonl','prepared/val.jsonl','prepared/concepts.json']}
    fingerprint=hashlib.sha256(json.dumps({'source':str(args.source.resolve()),'seeds':args.seeds,
        'epochs':args.epochs,'fp32_heads':args.fp32_heads,'readout':args.readout,
        'training_rng':args.training_rng,
        'git_sha':revision,'inputs':inputs},sort_keys=True).encode()).hexdigest()
    manifest_path=args.output/'run_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['fingerprint']!=fingerprint:
        raise ValueError('resume configuration mismatch')
    start=time.monotonic();manifest={'status':'running','fingerprint':fingerprint,
        'git_sha':revision,'source_fingerprints':inputs,
        'test_split_touched':False}
    def progress(row):
        row={**row,'seconds':time.monotonic()-start}
        with (args.output/'progress.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    write(manifest_path,manifest)
    try: run(args,progress);manifest['status']='complete'
    except Exception as exc:manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:manifest['runtime_seconds']=time.monotonic()-start;write(manifest_path,manifest)


if __name__=='__main__': main()
