"""Frozen-checkpoint conditional graph search using the entire training reward set."""
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from graph_mvp.task_model import load_local_causal_lm
from graph_mvp.task_prototypes import build_task_soft_graph_tokenizer
from graph_mvp.north_validation import MechanismClassifier, contexts_from_prepared
from graph_mvp.multilabel_task import MultilabelEvaluator
from graph_mvp.weighted_glasso import statistical_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--grid',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    initial=np.load(args.source/'initial_graph.npz',allow_pickle=False)
    grid=json.loads((args.grid/'summary.json').read_text())
    concepts=json.loads((args.source/'prepared/concepts.json').read_text())
    ids=tuple(str(c) for c in initial['concept_ids'])
    if ids!=tuple(c['concept_id'] for c in concepts):raise ValueError('concept order mismatch')
    baseline_stat=statistical_loss(initial['S'],initial['Theta'])
    upper=np.triu_indices(len(ids),1)
    baseline_edges=int((np.abs(initial['partial_corr'][upper])>1e-5).sum())
    protocol={'selection':'highest macro-AP on all 4563 training records among admissible fitted graphs',
        'statistical_gate':'fixed-B unpenalized NLL increase per concept <= .01',
        'complexity_gate':'edges at |partial correlation|>1e-5 <= 2 * initial edges',
        'mode':'fixed_B_conditional_glasso','task_checkpoint':'original frozen task modules and LoRA',
        'threshold':.5,'test_split_touched':False,
        'warning':'validation exploration; no GRPO, full alternating refit or transfer claim',
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    manifest=json.loads((args.source/'run_manifest.json').read_text())
    torch.manual_seed(11)
    lm,tok=load_local_causal_lm(manifest['args']['task_model'],device_map='cuda:0',dtype='bfloat16',local_files_only=True)
    lm=PeftModel.from_pretrained(lm,args.source/'task_lora',is_trainable=False)
    model=MechanismClassifier(lm,build_task_soft_graph_tokenizer(lm,tok,[c['concept_text'] for c in concepts]))
    state=torch.load(args.source/'task_modules.pt',map_location='cpu',weights_only=True)
    model.graph_tokenizer.load_state_dict(state['graph_tokenizer']);model.head.load_state_dict(state['head'])
    for parameter in model.parameters():parameter.requires_grad_(False)
    contexts=contexts_from_prepared(args.source/'prepared',ids,tok)
    if len(contexts['train'].texts)!=4563 or len(contexts['val'].texts)!=777:raise ValueError('population mismatch')
    evaluator=MultilabelEvaluator(model);rows=[];baseline_prob={};start=time.monotonic()
    graphs=[('initial',initial,True)]+[(f"lambda_{r['penalty']:g}",np.load(args.grid/r['file'],allow_pickle=False),True) for r in grid['graphs']]
    permutation=np.random.default_rng(101).permutation(len(ids))
    controls=[('zero',np.zeros_like(initial['partial_corr'])),('permuted',initial['partial_corr'][np.ix_(permutation,permutation)])]
    for name,data,fitted in graphs+[(name,{'partial_corr':rho},False) for name,rho in controls]:
        rho=data['partial_corr'];row={'name':name,'fitted':fitted,'num_edges':int((np.abs(rho[upper])>1e-5).sum())}
        if fitted:
            if tuple(str(c) for c in data['concept_ids'])!=ids:raise ValueError('grid concept order mismatch')
            if not np.allclose(data['S'],initial['S']):raise ValueError('conditional covariance changed')
            row['statistical_nll_delta_per_concept']=(statistical_loss(initial['S'],data['Theta'])-baseline_stat)/len(ids)
            row['admissible']=row['statistical_nll_delta_per_concept']<=.01 and row['num_edges']<=2*baseline_edges
        else:row['admissible']=False
        for split in ['train','val']:
            metrics,_,loss,prob=evaluator.score_graph(rho,contexts[split],True)
            row[split]=dict(metrics.extra_metrics)
            if name=='initial':baseline_prob[split]=prob
            difference=np.abs(prob-baseline_prob[split])
            row[split]['probability_change_mean']=float(difference.mean())
            row[split]['probability_change_max']=float(difference.max())
            np.savez_compressed(args.output/f'{name}_{split}_private_scores.npz',probabilities=prob,labels=contexts[split].labels,per_example_loss=loss)
        row['elapsed_seconds']=time.monotonic()-start;rows.append(row)
        (args.output/'progress.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps({'stage':'graph_complete',**row}),flush=True)
    selected=max((row for row in rows if row['admissible']),key=lambda row:row['train']['macro_average_precision'])
    (args.output/'summary.json').write_text(json.dumps({'protocol':protocol,'runs':rows,
        'selected_using_train_only':selected['name'],'test_split_touched':False},indent=2))


if __name__=='__main__':main()
