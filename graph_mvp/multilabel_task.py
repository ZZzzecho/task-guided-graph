"""Graph-conditioned multilabel classifier and paired frozen-model evaluation."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score, average_precision_score
from .retrieval_task import _unwrap_backbone, _last_active_pool, encode_text_batch
from .downstream import TaskMetrics, EvaluationResult
from .graph_tokens import error_weighted_edge_relevance


@dataclass(frozen=True)
class MultilabelContext:
    concept_ids: tuple
    texts: tuple
    labels: np.ndarray
    tokenizer: object
    batch_size: int = 4
    max_length: int = 512

    def __post_init__(self):
        y = np.array(self.labels, dtype=float, copy=True)
        if len(set(self.concept_ids)) != len(self.concept_ids) or len(self.concept_ids)<2:
            raise ValueError('invalid concept IDs')
        if not self.texts or y.shape != (len(self.texts),8) or not np.isin(y,[0,1]).all():
            raise ValueError('labels must be [N,8] multi-hot')
        if self.batch_size<1 or self.max_length<8:
            raise ValueError('invalid batch/length')
        y.setflags(write=False)
        object.__setattr__(self,'labels',y)

    def batches(self, device):
        for i in range(0,len(self.texts),self.batch_size):
            yield encode_text_batch(self.tokenizer,self.texts[i:i+self.batch_size],
                                    max_length=self.max_length,device=device), torch.as_tensor(
                self.labels[i:i+self.batch_size].copy(),dtype=torch.float32,device=device)


class GraphMultilabelClassifier(nn.Module):
    def __init__(self, causal_lm, graph_tokenizer):
        super().__init__()
        self.causal_lm=causal_lm
        self.graph_tokenizer=graph_tokenizer
        self.backbone=_unwrap_backbone(causal_lm)
        emb=causal_lm.get_input_embeddings().weight
        self.head=nn.Linear(emb.shape[-1],8).to(device=emb.device,dtype=emb.dtype)

    @property
    def device(self):
        return self.causal_lm.get_input_embeddings().weight.device

    def forward(self, input_ids, attention_mask, global_graph, return_aux=False):
        emb=self.causal_lm.get_input_embeddings()(input_ids)
        tokens,aux=self.graph_tokenizer(emb,global_graph,token_mask=attention_mask.bool(),return_aux=True)
        graph_mask=torch.ones(tokens.shape[:2],dtype=attention_mask.dtype,device=self.device)
        mask=torch.cat((graph_mask,attention_mask),dim=1)
        out=self.backbone(inputs_embeds=torch.cat((tokens,emb),dim=1),attention_mask=mask,
                          use_cache=False,return_dict=True)
        logits=self.head(_last_active_pool(out.last_hidden_state,mask)).float()
        return (logits,aux) if return_aux else logits


def multilabel_metrics(labels, logits):
    y=np.asarray(labels); z=np.asarray(logits,dtype=np.float64)
    if y.shape != z.shape or y.ndim!=2 or y.shape[1]!=8 or not np.isfinite(z).all():
        raise ValueError('invalid multilabel scores')
    losses=(np.logaddexp(0,z)-y*z).mean(axis=1)
    prob=1/(1+np.exp(-np.clip(z,-700,700)))
    pred=prob>=.5
    ap=[float(average_precision_score(y[:,j],prob[:,j])) if y[:,j].any() else None for j in range(8)]
    return TaskMetrics(float(losses.mean()),-float(losses.mean()),{
        'bce':float(losses.mean()),'micro_f1':float(f1_score(y,pred,average='micro',zero_division=0)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'per_label_f1':f1_score(y,pred,average=None,zero_division=0).tolist(),
        'macro_average_precision':float(np.mean([v for v in ap if v is not None])) if any(v is not None for v in ap) else None,
        'per_label_average_precision':ap,'threshold':.5,'n_samples':len(y),
    }),losses,prob


class MultilabelEvaluator:
    def __init__(self,model): self.model=model

    def score_graph(self,graph,ctx,return_details=False):
        self.model.eval(); scores=[]; activations=[]
        with torch.no_grad():
            for batch,_ in ctx.batches(self.model.device):
                logits,aux=self.model(**batch,global_graph=graph,return_aux=True)
                scores.append(logits.cpu().numpy());activations.append(aux['activations'].float().cpu().numpy())
        metrics,losses,prob=multilabel_metrics(ctx.labels,np.concatenate(scores))
        return (metrics,np.concatenate(activations),losses,prob) if return_details else metrics

    def evaluate_snapshot(self,snapshot,ctx):
        if tuple(snapshot.concept_ids)!=tuple(ctx.concept_ids): raise ValueError('concept axis mismatch')
        return self.score_graph(snapshot.Rho,ctx)

    def evaluate(self,candidate,ctx,baseline):
        if not candidate.valid: raise ValueError('invalid candidate')
        m=self.evaluate_snapshot(candidate.snapshot,ctx)
        return EvaluationResult(candidate.candidate_id,m.task_loss,m.task_metric,
                                baseline.task_loss,baseline.task_metric,m.task_metric-baseline.task_metric,m.extra_metrics)

    def task_relevance(self,snapshot,ctx):
        _,a,losses,_=self.score_graph(snapshot.Rho,ctx,True)
        return error_weighted_edge_relevance(a,losses)


def adapt_multilabel(model,graph,ctx,optimizer,*,epochs=1,seed=11,progress=None):
    rng=np.random.default_rng(seed); model.train(); step=0; losses=[]
    for epoch in range(epochs):
        order=rng.permutation(len(ctx.texts))
        for i in range(0,len(order),ctx.batch_size):
            indices=order[i:i+ctx.batch_size]
            batch=encode_text_batch(ctx.tokenizer,[ctx.texts[j] for j in indices],max_length=ctx.max_length,device=model.device)
            labels=torch.as_tensor(ctx.labels[indices].copy(),dtype=torch.float32,device=model.device)
            logits=model(**batch,global_graph=graph)
            loss=F.binary_cross_entropy_with_logits(logits,labels)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite multilabel loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step();step+=1;losses.append(float(loss.detach().cpu()))
            if progress and (step==1 or step%25==0 or i+len(indices)==len(order)):
                progress({'stage':'task_warmup','epoch':epoch+1,'epochs':epochs,'step':step,
                          'loss':losses[-1],'moving_avg_25':float(np.mean(losses[-25:]))})
    model.eval()
    return {'epochs':epochs,'optimizer_steps':step,'presentations':len(ctx.texts)*epochs,
            'mean_loss':float(np.mean(losses)),'last_loss':losses[-1]}
