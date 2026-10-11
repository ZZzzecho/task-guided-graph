from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from torch import nn
from graph_mvp.north_medical import normalize_diagnosis,split_training,prepare_north_data,LABEL_NAMES,record_id
from graph_mvp.multilabel_task import MultilabelContext,GraphMultilabelClassifier,MultilabelEvaluator,adapt_multilabel,multilabel_metrics
from graph_mvp.graph_tokens import SoftGraphTokenizer


def rows(n,prefix='train'):
    return [{'description':f'{prefix} example {i}', 'diagnosis':[1]*8} for i in range(n)]


def test_preserve_multihot_and_negative_cases():
    y=[1,0,1,1,0,0,0,1]
    assert normalize_diagnosis(y)==y
    assert normalize_diagnosis(str(y))==y
    assert normalize_diagnosis([0]*8)==[0]*8
    assert LABEL_NAMES[7]=='髌股关节软骨损伤'


@pytest.mark.parametrize('invalid',[[0]*7,[0]*9,[0,0,0,2,0,0,0,0],[0,0,0,np.nan,0,0,0,0]])
def test_invalid_labels_rejected(invalid):
    with pytest.raises(ValueError): normalize_diagnosis(invalid)


def test_duplicate_group_split_is_disjoint_and_repeatable():
    data=rows(40)+[rows(40)[3]]
    a,b=split_training(data,seed=11)
    assert not {record_id(r['description']) for r in a}&{record_id(r['description']) for r in b}
    assert len(a)+len(b)==len(data)
    assert (a,b)==split_training(data,seed=11)


def test_data0_is_training_and_test_file_not_opened(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    for name,data in [('train_data',rows(24)),('data0',rows(16,'extra')),('val_data',rows(4,'val'))]:
        np.save(raw/(name+'.npy'),data,allow_pickle=True)
    (raw/'test_data.npy').write_bytes(b'NOT AN NPY; must never be loaded')
    terms=tmp_path/'terms.csv';terms.write_text(',0\n0,疼痛\n1,-\n2,肿胀\n3,疼痛\n',encoding='utf-8')
    audit=prepare_north_data(raw,terms,tmp_path/'prepared')
    assert audit['merged_train']==40 and audit['data0_training']==16
    assert audit['task_train']==40 and audit['reward']==40
    assert audit['reward_mode']=='full_training'
    assert audit['concepts']==2 and audit['test_split_touched'] is False
    ids=[]
    for name in ['train','reward']:
        ids += [json.loads(l)['id'] for l in (tmp_path/'prepared'/f'{name}.jsonl').read_text().splitlines()]
    assert set(ids)=={record_id(r['description']) for r in rows(24)+rows(16,'extra')}
    assert 'test_data.npy' not in audit['source_sha256']


def test_optional_holdout_keeps_group_isolation(tmp_path):
    for name,data in [('train_data',rows(24)),('data0',rows(16,'extra')),('val_data',rows(4,'val'))]:
        np.save(tmp_path/(name+'.npy'),data,allow_pickle=True)
    terms=tmp_path/'terms.csv';terms.write_text(',0\n0,pain\n1,swelling\n')
    audit=prepare_north_data(tmp_path,terms,tmp_path/'prepared',reward_fraction=.1)
    assert audit['task_train']+audit['reward']==40
    assert audit['reward_mode']=='grouped_training_holdout'


def test_train_validation_overlap_rejected(tmp_path):
    for name,data in [('train_data',rows(24)),('data0',rows(16,'extra')),('val_data',rows(1))]:
        np.save(tmp_path/(name+'.npy'),data,allow_pickle=True)
    with pytest.raises(ValueError,match='overlap'):
        prepare_north_data(tmp_path,tmp_path/'missing.csv',tmp_path/'out')


def test_independent_labels_metrics_and_continuous_utility():
    y=np.eye(8,dtype=int);y[0,1]=1
    m,loss,p=multilabel_metrics(y,np.where(y,6.,-6.))
    assert m.extra_metrics['micro_f1']==1 and m.extra_metrics['macro_f1']==1
    assert m.task_metric==-m.task_loss
    assert (p[0]>=.5).sum()==2
    m2,_,_=multilabel_metrics(y,np.where(y,5.,-5.))
    assert m.task_loss<m2.task_loss


class Tokenizer:
    pad_token_id=0
    def __call__(self,texts,**kwargs):
        ids=torch.tensor([[1,2,3],[3,1,0]][:len(texts)])
        return {'input_ids':ids,'attention_mask':(ids!=0).long()}


class Body(nn.Module):
    def __init__(self): super().__init__();self.mix=nn.Linear(12,12)
    def forward(self,inputs_embeds,attention_mask,**kwargs):
        # A deterministic causal operation lets prefix graph tokens affect the pooled state.
        x=inputs_embeds*attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=self.mix(x.cumsum(1)))


class LM(nn.Module):
    def __init__(self): super().__init__();self.emb=nn.Embedding(4,12);self.model=Body()
    def get_input_embeddings(self):return self.emb


def test_classifier_trains_all_labels_and_graph_changes_scores():
    torch.manual_seed(11)
    graph=np.ones((4,4))-np.eye(4)
    lm=LM();tok=SoftGraphTokenizer(np.random.default_rng(11).normal(size=(4,12)),12,num_tokens=2,graph_hidden_dim=6)
    model=GraphMultilabelClassifier(lm,tok)
    y=np.array([[1,0,1,0,0,0,0,1],[0]*8])
    ctx=MultilabelContext(tuple('abcd'),('one','two'),y,Tokenizer(),batch_size=2)
    evaluator=MultilabelEvaluator(model)
    _,_,_,before=evaluator.score_graph(graph,ctx,True)
    _,_,_,zero=evaluator.score_graph(np.zeros_like(graph),ctx,True)
    assert not np.allclose(before,zero)
    head_before=model.head.weight.detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    info=adapt_multilabel(model,graph,ctx,optimizer,epochs=2)
    assert info['optimizer_steps']==2 and info['presentations']==4
    assert not torch.equal(head_before,model.head.weight)
    assert np.array_equal(ctx.labels,y)
    # Frozen paired comparisons must not mutate parameters or leave dropout active.
    for p in model.parameters():p.requires_grad_(False)
    saved=[p.detach().clone() for p in model.parameters()]
    a=evaluator.score_graph(graph,ctx);b=evaluator.score_graph(graph,ctx)
    assert a.task_loss==b.task_loss
    assert all(torch.equal(p,old) for p,old in zip(model.parameters(),saved))


def test_context_rejects_single_label_targets():
    with pytest.raises(ValueError,match='multi-hot'):
        MultilabelContext(tuple('abcd'),('one','two'),np.array([0,1]),Tokenizer())


def test_real_graph_solve_reward_grpo_validation_and_compact_report():
    from graph_mvp.environment import GraphEnvironment
    from graph_mvp.linear_policy import LinearGRPOPolicy,LinearCandidateBuilder
    from graph_mvp.runner import GraphPhaseRunner
    from graph_mvp.reward import RewardFunction
    from scripts.run_north_medical_graph_rl import compact_phase
    torch.manual_seed(11)
    proto=np.random.default_rng(11).normal(size=(4,12))
    model=GraphMultilabelClassifier(LM(),SoftGraphTokenizer(proto,12,num_tokens=2,graph_hidden_dim=6))
    for p in model.parameters():p.requires_grad_(False)
    ctx=MultilabelContext(tuple('abcd'),('one','two'),np.array([[1,0,1,0,0,0,0,1],[0]*8]),Tokenizer(),batch_size=2)
    evaluator=MultilabelEvaluator(model)
    env=GraphEnvironment()
    state=env.initialize(np.eye(4)*.7+np.ones((4,4))*.3,.1,tuple('abcd'))
    runner=GraphPhaseRunner(env,LinearGRPOPolicy(seed=11),evaluator,RewardFunction(),
                           candidate_builder=LinearCandidateBuilder(proto),task_adapter=None)
    reports=[]
    result=runner.run(state,ctx,ctx,phases=1,policy_updates_per_phase=2,num_candidates=4,
                      on_phase=lambda r:reports.append(compact_phase(r)))
    assert len(result.phases)==1 and len(reports[0]['updates'])==2
    assert all(len(u['candidates'])==4 for u in reports[0]['updates'])
    assert not result.phases[0].adapted
    assert any(u.policy_update['updated'] for u in result.phases[0].updates)
    json.dumps(reports,allow_nan=False)
