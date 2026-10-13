"""Probe text representations without publishing private examples or features."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from graph_mvp.task_model import load_local_causal_lm,attach_task_lora
from graph_mvp.task_prototypes import build_task_soft_graph_tokenizer
from graph_mvp.north_validation import (MechanismClassifier,contexts_from_prepared,restore_trainables,masked_mean_pool)
from graph_mvp.retrieval_task import _last_active_pool
from graph_mvp.multilabel_task import multilabel_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--mechanism',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',choices=['pilot','base','original'],default='pilot')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.source/'run_manifest.json').read_text())
    concepts=json.loads((args.source/'prepared/concepts.json').read_text());ids=tuple(c['concept_id'] for c in concepts)
    torch.manual_seed(11)
    lm,tok=load_local_causal_lm(manifest['args']['task_model'],device_map='cuda:0',dtype='bfloat16',local_files_only=True)
    if args.checkpoint=='original':
        from peft import PeftModel
        lm=PeftModel.from_pretrained(lm,args.source/'task_lora',is_trainable=False)
    else:lm=attach_task_lora(lm,r=16,alpha=32,dropout=.05)
    model=MechanismClassifier(lm,build_task_soft_graph_tokenizer(lm,tok,[c['concept_text'] for c in concepts]))
    if args.checkpoint=='pilot':
        restore_trainables(model,torch.load(args.mechanism/'seed_11_text_trainables.pt',map_location='cpu',weights_only=True))
    elif args.checkpoint=='original':
        state=torch.load(args.source/'task_modules.pt',map_location='cpu',weights_only=True)
        model.head.load_state_dict(state['head']);model.graph_tokenizer.load_state_dict(state['graph_tokenizer'])
    for par in model.parameters():par.requires_grad_(False)
    model.eval();contexts=contexts_from_prepared(args.source/'prepared',ids,tok)
    features={};report={'test_split_touched':False,'probes':[],'representation_audit':{},
        'checkpoint':args.checkpoint,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    if args.checkpoint=='original':
        from graph_mvp.multilabel_task import MultilabelEvaluator
        graph=np.load(args.source/'initial_graph.npz',allow_pickle=False)['partial_corr']
        report['original_checkpoint_reproduction']=dict(MultilabelEvaluator(model).score_graph(graph,contexts['val']).extra_metrics)
        print(json.dumps({'stage':'original_checkpoint_reproduction',**report['original_checkpoint_reproduction']}),flush=True)
    with torch.no_grad():
        for split,ctx in contexts.items():
            rows={'last':[],'mean':[]};last_ids=[]
            for step,(batch,_) in enumerate(ctx.batches(model.device)):
                emb=lm.get_input_embeddings()(batch['input_ids'])
                hidden=model.backbone(inputs_embeds=emb,attention_mask=batch['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
                rows['last'].append(_last_active_pool(hidden,batch['attention_mask']).float().cpu().numpy())
                rows['mean'].append(masked_mean_pool(hidden,batch['attention_mask']).cpu().numpy())
                positions=torch.arange(batch['input_ids'].shape[1],device=model.device)[None,:].expand_as(batch['input_ids'])
                index=positions.masked_fill(~batch['attention_mask'].bool(),-1).max(1).values
                last_ids.extend(batch['input_ids'][torch.arange(len(index),device=model.device),index].cpu().tolist())
                if step%100==0:print(json.dumps({'stage':'extract','split':split,'step':step}),flush=True)
            features[split]={name:np.concatenate(value) for name,value in rows.items()}
            report['representation_audit'][split]={'last_token_unique_ids':len(set(last_ids)),
                'features':{name:{'mean_column_std':float(value.std(0).mean()),'mean_row_norm':float(np.linalg.norm(value,axis=1).mean())} for name,value in features[split].items()}}
            np.savez_compressed(args.output/f'{split}_private_features.npz',**features[split])
    for readout in ['last','mean']:
        train=features['train'][readout];val=features['val'][readout]
        center=train.mean(0);scale=np.maximum(train.std(0),1e-4)
        x=torch.as_tensor((train-center)/scale,device='cuda',dtype=torch.float32)
        v=torch.as_tensor((val-center)/scale,device='cuda',dtype=torch.float32)
        y=torch.as_tensor(contexts['train'].labels.copy(),device='cuda',dtype=torch.float32)
        for weighting in ['uniform','sqrt_positive']:
            torch.manual_seed(101);head=nn.Linear(x.shape[1],8).cuda().float();opt=torch.optim.Adam(head.parameters(),lr=.01)
            positives=y.sum(0);weights=torch.clamp(torch.sqrt((len(y)-positives)/positives.clamp_min(1)),1,10) if weighting!='uniform' else None
            for _ in range(300):
                loss=F.binary_cross_entropy_with_logits(head(x),y,pos_weight=weights)+.001*head.weight.square().mean()
                opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():logits=head(v).cpu().numpy()
            metrics,losses,prob=multilabel_metrics(contexts['val'].labels,logits)
            row={'readout':readout,'weighting':weighting,'validation':dict(metrics.extra_metrics)}
            report['probes'].append(row)
            np.savez_compressed(args.output/f'{readout}_{weighting}_scores.npz',probabilities=prob,labels=contexts['val'].labels)
            print(json.dumps({'stage':'probe_complete',**row}),flush=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
