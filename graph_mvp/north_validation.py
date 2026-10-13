"""Matched text/concept/graph experiments; no test access or graph fitting."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .multilabel_task import GraphMultilabelClassifier, MultilabelContext
from .retrieval_task import _last_active_pool


class MechanismClassifier(GraphMultilabelClassifier):
    def __init__(self, causal_lm, graph_tokenizer):
        super().__init__(causal_lm, graph_tokenizer)
        self.arm = 'graph'

    def forward(self, input_ids, attention_mask, global_graph, return_aux=False):
        if self.arm != 'text':
            graph = np.zeros_like(global_graph) if self.arm == 'concept' else global_graph
            return super().forward(input_ids, attention_mask, graph, return_aux)
        emb = self.causal_lm.get_input_embeddings()(input_ids)
        out = self.backbone(inputs_embeds=emb, attention_mask=attention_mask,
                            use_cache=False, return_dict=True)
        logits = self.head(_last_active_pool(out.last_hidden_state, attention_mask)).float()
        aux = {'activations': torch.zeros((len(input_ids), self.graph_tokenizer.num_concepts),
                                         device=input_ids.device)}
        return (logits, aux) if return_aux else logits


def contexts_from_prepared(prepared, concept_ids, tokenizer):
    """Only explicitly named train/val paths; never discover a test file."""
    contexts = {}
    for name in ('train', 'val'):
        rows = [json.loads(line) for line in (Path(prepared)/(name+'.jsonl')).read_text().splitlines()]
        contexts[name] = MultilabelContext(tuple(concept_ids),
            tuple(r['text'] for r in rows), np.asarray([r['labels'] for r in rows]), tokenizer)
    return contexts


def capture_trainables(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore_trainables(model, saved):
    parameters = dict(model.named_parameters())
    if set(saved) != {name for name,p in parameters.items() if p.requires_grad}:
        raise ValueError('trainable parameter identities changed between matched arms')
    with torch.no_grad():
        for name,value in saved.items(): parameters[name].copy_(value.to(parameters[name]))


def paired_bootstrap(labels, left, right, metric='macro_ap', repetitions=1000, seed=101):
    from sklearn.metrics import average_precision_score, f1_score
    labels=np.asarray(labels);left=np.asarray(left);right=np.asarray(right)
    if labels.shape != left.shape or labels.shape != right.shape or labels.ndim != 2:
        raise ValueError('paired score shapes differ')
    def score(y,p):
        if metric == 'macro_ap':
            return np.mean([average_precision_score(y[:,j],p[:,j])
                            for j in range(y.shape[1]) if y[:,j].any()])
        if metric == 'macro_f1': return f1_score(y,p>=.5,average='macro',zero_division=0)
        raise ValueError('unknown metric')
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repetitions):
        idx=rng.integers(len(labels),size=len(labels))
        values.append(score(labels[idx],left[idx])-score(labels[idx],right[idx]))
    return {'delta':float(score(labels,left)-score(labels,right)),
            'descriptive_95_interval':np.quantile(values,[.025,.975]).tolist(),
            'repetitions':repetitions,'unit':'record; patient IDs unavailable',
            'warning':'exploration validation interval; not independent test or multi-seed uncertainty'}
