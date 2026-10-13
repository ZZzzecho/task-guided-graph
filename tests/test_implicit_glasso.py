import numpy as np
import torch
from graph_mvp.config import SolverConfig
from graph_mvp.weighted_glasso import WeightedGraphicalLasso,penalty_matrix
from graph_mvp.implicit_glasso import implicit_penalty_gradient


def test_implicit_edge_gradient_matches_certified_finite_difference():
    s=np.array([[1.,.5,.25],[.5,1.4,.6],[.25,.6,1.2]])
    solver=WeightedGraphicalLasso(SolverConfig(max_iter=10000,abs_tol=1e-10,rel_tol=1e-10,kkt_tol=1e-9))
    lam=penalty_matrix(3,.2);base=solver.solve(s,lam)
    assert base.converged
    g=np.array([[.2,.4,-.1],[.4,-.3,.2],[-.1,.2,.6]])
    result=implicit_penalty_gradient(base.Theta.copy(),g,tolerance=1e-10)
    assert result.converged and result.relative_residual<=1e-10
    upper=np.triu_indices(3,1);k=np.argmax(abs(base.Theta[upper]));i,j=upper[0][k],upper[1][k]
    eps=1e-3;plus=lam.copy();minus=lam.copy();plus[i,j]+=eps;plus[j,i]+=eps;minus[i,j]-=eps;minus[j,i]-=eps
    left=solver.solve(s,plus);right=solver.solve(s,minus)
    assert left.converged and right.converged
    expected=(np.sum(g*left.Theta)-np.sum(g*right.Theta))/(2*eps)
    np.testing.assert_allclose(float(result.gradient[i,j]),expected,rtol=1e-4,atol=1e-5)
    assert torch.equal(result.gradient,result.gradient.T) and not result.gradient.diag().any()


def test_zero_task_gradient_cannot_create_an_update():
    result=implicit_penalty_gradient(np.eye(3),np.zeros((3,3)))
    assert result.converged and result.iterations==0 and not result.gradient.any()
