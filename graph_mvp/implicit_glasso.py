"""Active-set implicit derivative of the project's weighted Glasso objective.

This is a local derivative at a fixed, strictly SPD solution; it is not a
confidence interval or an uncertainty estimator. Support-changing steps require
a new certified statistical solve and task evaluation.
"""
from dataclasses import dataclass
import torch


@dataclass
class ImplicitPenaltyGradient:
    gradient: torch.Tensor
    converged: bool
    iterations: int
    relative_residual: float


def implicit_penalty_gradient(theta, task_gradient, *, device='cpu',
                              tolerance=1e-6, max_iter=1000):
    """Return dL/dLambda as a symmetric matrix with a zero diagonal.

    The objective penalizes each upper-triangle edge once. Solve the symmetric
    active-set Hessian adjoint with preconditioned conjugate gradients, retaining
    free diagonal coordinates. An unconverged direction must not be used.
    """
    if not 0<tolerance<1 or max_iter<1:raise ValueError('invalid CG settings')
    theta=torch.as_tensor(theta,dtype=torch.float64,device=device).detach()
    g=torch.as_tensor(task_gradient,dtype=torch.float64,device=device).detach()
    if theta.ndim!=2 or theta.shape[0]!=theta.shape[1] or g.shape!=theta.shape:
        raise ValueError('precision and task gradient shapes differ')
    if not torch.isfinite(theta).all() or not torch.isfinite(g).all():raise ValueError('nonfinite inputs')
    if not torch.allclose(theta,theta.T,rtol=1e-10,atol=1e-10):raise ValueError('precision must be symmetric')
    torch.linalg.cholesky(theta)
    w=torch.linalg.inv(theta);w=(w+w.T)*.5;active=theta!=0
    g=(g+g.T)*.5*active
    norm=torch.linalg.vector_norm(g)
    if float(norm)==0:return ImplicitPenaltyGradient(torch.zeros_like(theta),True,0,0.)
    preconditioner=w.diag().outer(w.diag())+w.square()
    preconditioner.diagonal().copy_(w.diag().square())
    preconditioner=preconditioner.clamp_min(1e-30)
    def hessian(value):
        result=w@value@w
        return (result+result.T)*.5*active
    x=torch.zeros_like(g);residual=g.clone();z=residual/preconditioner
    direction=z.clone();rz=(residual*z).sum();relative=1.
    for iteration in range(1,max_iter+1):
        hd=hessian(direction);denominator=(direction*hd).sum()
        if float(denominator)<=0:break
        alpha=rz/denominator;x=x+alpha*direction;residual=residual-alpha*hd
        relative=float(torch.linalg.vector_norm(residual)/norm)
        if relative<=tolerance:break
        z=residual/preconditioner;next_rz=(residual*z).sum()
        direction=z+(next_rz/rz)*direction;rz=next_rz
    # Recompute the true residual; recursive CG residual can drift.
    relative=float(torch.linalg.vector_norm(g-hessian(x))/norm)
    gradient=-x*torch.sign(theta);gradient=(gradient+gradient.T)*.5;gradient.fill_diagonal_(0)
    return ImplicitPenaltyGradient(gradient,relative<=tolerance,iteration,relative)


def macro_pairwise_logit_gradient(logits, labels, *, temperature=1., bce_weight=.1):
    """Full-population macro logistic-ranking surrogate plus a small BCE term.

    Return the loss and its logit gradient without retaining a task-model graph.
    A ranking surrogate is not average precision; AP selects evaluated candidates.
    """
    z=torch.as_tensor(logits,dtype=torch.float64).detach()
    y=torch.as_tensor(labels,dtype=torch.float64,device=z.device).detach()
    if z.ndim!=2 or z.shape!=y.shape or not len(z) or temperature<=0 or bce_weight<0:
        raise ValueError('invalid ranking inputs')
    if not torch.isfinite(z).all() or not torch.all((y==0)|(y==1)):raise ValueError('invalid ranking values')
    gradient=torch.zeros_like(z);loss=torch.zeros((),dtype=z.dtype,device=z.device)
    for j in range(z.shape[1]):
        positive=torch.where(y[:,j]==1)[0];negative=torch.where(y[:,j]==0)[0]
        if not len(positive) or not len(negative):continue
        gaps=(z[negative,j][None,:]-z[positive,j][:,None])/temperature
        denominator=len(positive)*len(negative)*z.shape[1]
        probabilities=torch.sigmoid(gaps)
        loss=loss+torch.nn.functional.softplus(gaps).sum()/denominator
        gradient[positive,j]=-probabilities.sum(1)/(denominator*temperature)
        gradient[negative,j]=probabilities.sum(0)/(denominator*temperature)
    if bce_weight:
        loss=loss+bce_weight*torch.nn.functional.binary_cross_entropy_with_logits(z,y)
        gradient=gradient+bce_weight*(torch.sigmoid(z)-y)/z.numel()
    return float(loss),gradient
