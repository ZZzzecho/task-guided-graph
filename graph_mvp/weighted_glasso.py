"""Dense ADMM for -logdet(T) + tr(S T) + sum_{i<j} Lambda_ij |T_ij|.

Lambda/2 is the off-diagonal full-matrix proximal weight: the Frobenius
quadratic counts each undirected edge twice. The diagonal is never penalized.
The sparse ADMM variable is returned only when SPD and KKT checks pass.
"""
from dataclasses import dataclass
import numpy as np
from .config import SolverConfig
from .types import symmetric_matrix, readonly


def penalty_matrix(p, penalty):
    if np.ndim(penalty) == 0:
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("scalar penalty must be finite and positive")
        lam = np.full((p, p), float(penalty))
        np.fill_diagonal(lam, 0.0)
    else:
        lam = symmetric_matrix(penalty, "Lambda", p)
    if np.any(lam[np.triu_indices(p, 1)] <= 0) or np.any(np.diag(lam) != 0):
        raise ValueError("Lambda must be positive off diagonal and zero on diagonal")
    return lam


def statistical_loss(s, theta):
    np.linalg.cholesky(theta)
    return float(-np.linalg.slogdet(theta)[1] + np.sum(s * theta))


def objective(s, lam, theta):
    return statistical_loss(s, theta) + float(np.sum(np.triu(lam * np.abs(theta), 1)))


def kkt_residual(s, lam, theta):
    """Infinity norm of the minimum full-matrix subgradient (zero entries exact)."""
    g = s - np.linalg.inv(theta)
    active = theta != 0
    residual = np.maximum(np.abs(g) - lam / 2, 0)
    residual[active] = np.abs(g[active] + (lam / 2)[active] * np.sign(theta[active]))
    return float(np.max(residual))


@dataclass(frozen=True)
class SolverResult:
    Theta: np.ndarray | None
    converged: bool
    iterations: int
    primal_residual: float
    dual_residual: float
    kkt_residual: float | None
    objective: float | None
    min_eigenvalue: float | None
    message: str

    def info(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "Theta"}


class WeightedGraphicalLasso:
    def __init__(self, config=SolverConfig()):
        self.config = config
        self.solve_calls = 0

    def solve(self, S, penalty, initial_theta=None):
        s = symmetric_matrix(S, "S")
        p = s.shape[0]
        if np.linalg.eigvalsh(s)[0] < -1e-10 or np.any(np.diag(s) <= 0):
            raise ValueError("S must be PSD with positive diagonal; preprocessing owns ridge")
        lam = penalty_matrix(p, penalty)
        if initial_theta is None:
            z = np.diag(1 / np.diag(s))
        else:
            z = symmetric_matrix(initial_theta, "initial_theta", p).copy()
            np.linalg.cholesky(z)
        self.solve_calls += 1
        u = np.zeros_like(z)
        c = self.config
        kkt = mineig = obj = None
        for iteration in range(1, c.max_iter + 1):
            eigen, q = np.linalg.eigh(c.rho * (z - u) - s)
            root = np.sqrt(eigen * eigen + 4 * c.rho)
            # Stable quadratic root for large negative eigenvalues.
            values = np.empty_like(eigen)
            positive = eigen >= 0
            values[positive] = (eigen[positive] + root[positive]) / (2 * c.rho)
            values[~positive] = 2 / (root[~positive] - eigen[~positive])
            x = (q * values) @ q.T
            x = (x + x.T) / 2
            previous_z = z
            v = x + u
            z = np.sign(v) * np.maximum(np.abs(v) - lam / (2 * c.rho), 0)
            z = (z + z.T) / 2
            u += x - z
            primal = float(np.linalg.norm(x - z, "fro"))
            dual = float(c.rho * np.linalg.norm(z - previous_z, "fro"))
            eps_primal = p * c.abs_tol + c.rel_tol * max(np.linalg.norm(x), np.linalg.norm(z))
            eps_dual = p * c.abs_tol + c.rel_tol * np.linalg.norm(c.rho * u)
            if primal <= eps_primal and dual <= eps_dual:
                mineig = float(np.linalg.eigvalsh(z)[0])
                if mineig > 0:
                    kkt = kkt_residual(s, lam, z)
                    if kkt <= c.kkt_tol:
                        obj = objective(s, lam, z)
                        return SolverResult(readonly(z), True, iteration, primal, dual,
                                            kkt, obj, mineig, "converged")
        # Never publish an unconverged matrix as an optimized graph.
        return SolverResult(None, False, iteration, primal, dual, kkt, obj, mineig,
                            "max_iter reached before residual/SPD/KKT checks passed")

