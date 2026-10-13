import json
import numpy as np
import pytest
import torch
from graph_mvp.graph_tokens import SoftGraphTokenizer
from graph_mvp.north_validation import (MechanismClassifier,contexts_from_prepared,
    capture_trainables,restore_trainables,paired_bootstrap)
from test_north_multilabel import LM,Tokenizer
from graph_mvp.weighted_glasso import WeightedGraphicalLasso
from graph_mvp.north_validation import masked_mean_pool


def test_mean_pool_ignores_padding_and_fp32_control_gradients():
    hidden=torch.tensor([[[100.,100.],[1.,3.],[3.,1.]],[[1.,3.],[3.,1.],[100.,100.]]])
    mask=torch.tensor([[0,1,1],[1,1,0]])
    torch.testing.assert_close(masked_mean_pool(hidden,mask),torch.ones((2,2))*2)
    model=MechanismClassifier(LM(),SoftGraphTokenizer(np.random.default_rng(3).normal(size=(4,12)),12,2,6),fp32_heads=True,readout='mean')
    for arm in ['text','concept','graph']:
        model.arm=arm;model.zero_grad()
        logits=model(torch.tensor([[1,2,3]]),torch.ones((1,3),dtype=torch.long),np.ones((4,4))-np.eye(4))
        logits.sum().backward()
        assert model.head.weight.dtype==torch.float32 and model.head.weight.grad.abs().sum()>0


def test_paired_restore_and_text_concept_graph_controls():
    torch.manual_seed(11)
    model=MechanismClassifier(LM(),SoftGraphTokenizer(np.random.default_rng(11).normal(size=(4,12)),12,2,6))
    ids=torch.tensor([[1,2,3]]);mask=torch.ones_like(ids);g=np.ones((4,4))-np.eye(4)
    saved=capture_trainables(model)
    outputs={}
    for arm in ['text','concept','graph']:
        model.arm=arm;outputs[arm]=model(ids,mask,g)
    assert not torch.equal(outputs['text'],outputs['concept'])
    assert not torch.equal(outputs['concept'],outputs['graph'])
    outputs['graph'].sum().backward()
    with torch.no_grad():model.head.weight.add_(1)
    restore_trainables(model,saved)
    for name,p in model.named_parameters():assert torch.equal(p.cpu(),saved[name])


def test_context_loader_never_reads_test(tmp_path):
    rows=[{'text':'synthetic case','labels':[0]*8}]
    for name in ['train','val']:(tmp_path/f'{name}.jsonl').write_text(json.dumps(rows[0])+'\n')
    (tmp_path/'test.jsonl').write_text('INVALID SHOULD NOT BE OPENED')
    assert set(contexts_from_prepared(tmp_path,tuple('abcd'),Tokenizer()))=={'train','val'}


def test_paired_bootstrap_identity_and_direction():
    y=np.tile(np.eye(8),(4,1))
    right=np.ones_like(y)*.5;left=y*.8+.1
    assert paired_bootstrap(y,left,left,repetitions=20)['delta']==0
    result=paired_bootstrap(y,left,right,repetitions=20)
    assert result['delta']>0 and result['descriptive_95_interval'][0]>0


def test_dual_reconstruction_reuses_certified_solution():
    s=np.array([[1.,.6],[.6,2.]])
    solver=WeightedGraphicalLasso();first=solver.solve(s,.4)
    second=solver.solve(s,.4,initial_theta=first.Theta)
    assert first.converged and second.converged and second.iterations==0
    np.testing.assert_array_equal(first.Theta,second.Theta)


def test_dual_reconstruction_matches_cold_and_legacy_warm_for_changed_problem():
    s=np.array([[1.,.5,.25],[.5,1.4,.6],[.25,.6,1.2]])
    solver=WeightedGraphicalLasso();old=solver.solve(s,.4)
    cold=solver.solve(s,.5)
    legacy=solver.solve(s,.5,initial_theta=old.Theta,reconstruct_dual=False)
    warm=solver.solve(s,.5,initial_theta=old.Theta)
    assert all(r.converged for r in [cold,legacy,warm])
    np.testing.assert_allclose(warm.Theta,cold.Theta,atol=5e-5)
    assert warm.kkt_residual<=solver.config.kkt_tol
