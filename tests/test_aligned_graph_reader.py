import numpy as np
import torch
from graph_mvp.data import RankGaussianTransformer
from graph_mvp.estimators import _rank_gaussian_tensor
from scripts.probe_north_aligned_graph_reader import AlignedGraphResidual,sparse_normalized_graph


def test_training_ecdf_matches_the_statistical_tensor_transform_with_ties():
    x=np.array([[[1.,2.],[0.,3.]],[[1.,4.],[1.,2.]],[[2.,4.],[3.,2.]],[[3.,5.],[2.,1.]]])
    transform=RankGaussianTransformer().fit(x.reshape(4,-1))
    expected=_rank_gaussian_tensor(x)
    np.testing.assert_allclose(transform.transform(x.reshape(4,-1)).reshape(x.shape),expected,atol=1e-14)
    saved=transform.sorted_.copy()
    transform.transform(np.full((2,4),100.))
    np.testing.assert_array_equal(transform.sorted_,saved)


def test_initial_aligned_residual_is_zero_with_and_without_graph():
    torch.manual_seed(4);reader=AlignedGraphResidual(3,4)
    h=torch.randn(2,3,4)
    rho=np.array([[0.,.2,0.],[.2,0.,-.3],[0.,-.3,0.]])
    for graph in [rho,np.zeros_like(rho)]:
        assert torch.count_nonzero(reader(h,sparse_normalized_graph(graph,'cpu')))==0


def test_signed_graph_normalization_commutes_with_node_permutation():
    rho=np.array([[0.,.2,.4],[.2,0.,-.3],[.4,-.3,0.]])
    order=np.array([2,0,1])
    original=sparse_normalized_graph(rho,'cpu').to_dense()
    permuted=sparse_normalized_graph(rho[np.ix_(order,order)],'cpu').to_dense()
    torch.testing.assert_close(permuted,original[order][:,order])
