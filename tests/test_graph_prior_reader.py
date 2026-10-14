import torch
import numpy as np
from scripts.probe_north_graph_prior_reader import graph_prior_penalty,sparse_prior


def test_prior_penalty_and_gradient_match_dense_quadratic_oracle():
    a=np.array([[2.,-.4,0.],[-.4,1.2,-.3],[0.,-.3,1.]])
    q=sparse_prior(a,device='cpu').double();w=torch.randn(2,6,dtype=torch.float64,requires_grad=True)
    value=graph_prior_penalty(w,q,2,.2);gradient=torch.autograd.grad(value,w)[0]
    flat=w.detach().reshape(4,3);expected=.2*(flat*(flat@q.to_dense())).sum()/4
    torch.testing.assert_close(value,expected)
    torch.testing.assert_close(gradient,(.2/2*(flat@q.to_dense())).reshape(2,6))
    assert value>0


def test_prior_controls_have_same_diagonal_and_permuted_spectrum():
    a=np.array([[1.,-.2,.1],[-.2,1.,-.3],[.1,-.3,1.]])
    q=sparse_prior(a,device='cpu').to_dense();order=np.array([2,0,1])
    perm=sparse_prior(a[np.ix_(order,order)],device='cpu').to_dense()
    torch.testing.assert_close(q.diag(),torch.ones(3))
    torch.testing.assert_close(torch.linalg.eigvalsh(q),torch.linalg.eigvalsh(perm))
