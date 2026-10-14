import numpy as np
import torch
from scripts.probe_north_conditional_anchor_reader import select_anchors,anchor_features,AnchorResidual


def test_anchor_selection_uses_only_fixed_embedding_similarity():
    p=np.array([[1.,0.],[0.,1.],[-1.,0.]])
    np.testing.assert_array_equal(select_anchors(p,np.array([[1.,0.],[0.,1.]]),1),[[0],[1]])


def test_conditional_anchor_features_exclude_own_value_and_pool_only_selected_nodes():
    theta=np.array([[2.,-.4,0.],[-.4,1.2,-.3],[0.,-.3,1.]])
    x=torch.tensor([[[.2,.7,1.],[-.4,.3,-.5]]],dtype=torch.float64)
    anchors=np.array([[0],[1]])
    f=anchor_features(x,theta,anchors)
    torch.testing.assert_close(f[0,0,:2],x[0,:,0])
    torch.testing.assert_close(f[0,0,2:],.2*x[0,:,1])
    torch.testing.assert_close(f[0,1,2:],(.4*x[0,:,0]+.3*x[0,:,2])/1.2)


def test_anchor_residual_initialization_preserves_text_logits():
    model=AnchorResidual(128)
    assert torch.count_nonzero(model(torch.randn(3,8,128)))==0
