import numpy as np
import torch
from scripts.audit_north_statistical_graph import precision_graph_scores


def test_joint_density_matches_full_kronecker_gaussian_oracle():
    a=torch.tensor([[2.,-.4],[-.4,1.2]],dtype=torch.float64)
    b=torch.tensor([[1.3,.2],[.2,.8]],dtype=torch.float64)
    x=torch.tensor([[[.2,.7],[-.4,.3]],[[1.,-.1],[.3,.9]]],dtype=torch.float64)
    actual=precision_graph_scores(x,a,b)['joint_nll_per_cell']
    distribution=torch.distributions.MultivariateNormal(torch.zeros(4,dtype=torch.float64),precision_matrix=torch.kron(b,a))
    expected=-distribution.log_prob(x.reshape(2,4))/4
    torch.testing.assert_close(actual,expected)


def test_conditional_means_exclude_the_concept_being_predicted():
    a=torch.tensor([[2.,-.4],[-.4,1.2]],dtype=torch.float64);b=torch.eye(2,dtype=torch.float64)
    x=torch.tensor([[[.2,.7],[-.4,.3]]],dtype=torch.float64)
    means=torch.stack([.2*x[:,:,1],(.4/1.2)*x[:,:,0]],dim=-1)
    expected=(x-means).square().mean((1,2))
    torch.testing.assert_close(precision_graph_scores(x,a,b)['reconstruction_mse'],expected)
    conditional=torch.zeros(1,dtype=torch.float64)
    for i in range(2):
        distribution=torch.distributions.MultivariateNormal(means[:,:,i],precision_matrix=b*a[i,i])
        conditional-=distribution.log_prob(x[:,:,i])/4
    torch.testing.assert_close(precision_graph_scores(x,a,b)['conditional_pseudo_nll_per_cell'],conditional)
