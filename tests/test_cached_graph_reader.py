import torch
from scripts.probe_north_cached_graph_reader import MentionGraphResidual


def test_zero_initial_residual_preserves_the_frozen_text_predictions():
    reader=MentionGraphResidual(3).double()
    a=torch.tensor([[1.,0.,1.],[0.,1.,1.]],dtype=torch.float64)
    for message in [torch.zeros_like(a),torch.tensor([[.2,0.,-.3],[0.,-.4,.2]],dtype=torch.float64)]:
        assert torch.count_nonzero(reader(a,message))==0


def test_graph_gradient_matches_finite_difference_and_excludes_unmentioned_nodes():
    reader=MentionGraphResidual(3).double()
    with torch.no_grad():reader.readout.weight.fill_(.2)
    a=torch.tensor([[1.,1.,0.]],dtype=torch.float64)
    rho=torch.tensor([[0.,.3,.4],[.3,0.,.5],[.4,.5,0.]],dtype=torch.float64,requires_grad=True)
    def objective(w):return reader(a,(a@w)*a).square().sum()
    gradient=torch.autograd.grad(objective(rho),rho)[0]
    step=torch.zeros_like(rho);step[0,1]=1e-6
    numerical=(objective(rho.detach()+step)-objective(rho.detach()-step))/(2e-6)
    torch.testing.assert_close(gradient[0,1],numerical,rtol=1e-6,atol=1e-8)
    assert abs(gradient[0,1])>0
    assert torch.count_nonzero(gradient[2,:])==0
    assert torch.count_nonzero(gradient[:,2])==0
