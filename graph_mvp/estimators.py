"""Graph-estimator backends for three statistical sample semantics.

The framework supports:
1) patient_activation_vector: one patient -> one p-dimensional scalar concept vector;
2) patient_concept_matrix: one patient -> one r x p concept-representation matrix;
3) bootstrap_concept_embedding_matrix: one bootstrap replicate -> one r x p matrix.

The two matrix-valued modes share the same nonparanormal matrix-normal graphical
model (MNGM) solver.  GRPO always controls only the concept-axis penalty matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Mapping, Any
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
import warnings
from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning

from .config import DataConfig, MNGMConfig, SolverConfig
from .data import RankGaussianTransformer, training_covariance
from .types import readonly, freeze, symmetric_matrix
from .weighted_glasso import WeightedGraphicalLasso, penalty_matrix

VECTOR_MODE = "patient_activation_vector"
PATIENT_MATRIX_MODE = "patient_concept_matrix"
BOOTSTRAP_MATRIX_MODE = "bootstrap_concept_embedding_matrix"
MATRIX_MODES = (PATIENT_MATRIX_MODE, BOOTSTRAP_MATRIX_MODE)
GRAPH_MODES = (VECTOR_MODE,) + MATRIX_MODES


def _fingerprint_array(x: np.ndarray, mode: str) -> str:
    a = np.asarray(x)
    h = sha256()
    h.update(mode.encode())
    h.update(str((a.shape, a.dtype)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _symmetrize(a):
    a = np.asarray(a, dtype=float)
    return (a + a.T) / 2


def _psd_ridge(a, ridge: float, min_eig: float):
    """Return a symmetric positive-definite covariance without min-max rescaling."""
    a = _symmetrize(a)
    eig, vec = np.linalg.eigh(a)
    floor = max(float(ridge), float(min_eig))
    eig = np.maximum(eig, floor)
    out = (vec * eig) @ vec.T
    return _symmetrize(out)


def _rank_gaussian_tensor(samples: np.ndarray) -> np.ndarray:
    """Rank-Gaussian transform each matrix cell across the true sample axis."""
    x = np.asarray(samples, dtype=float)
    n, r, p = x.shape
    flat = x.reshape(n, r * p)
    ranks = rankdata(flat, axis=0, method="average")
    probs = np.clip((ranks - 0.5) / n, 0.5 / n, 1 - 0.5 / n)
    z = ndtri(probs)
    return z.reshape(n, r, p)


def _spearman_sine_correlation(samples: np.ndarray, min_eig: float) -> np.ndarray:
    """Legacy-compatible Gaussian-copula correlation used by the original MNGM code.

    Matrix cells are flattened in explicit representation-major order:
    index = representation_coordinate * p + concept_index.  This fixes the
    ambiguous flatten/block indexing in the original experimental script.
    """
    x = np.asarray(samples, dtype=float)
    n, r, p = x.shape
    flat = x.reshape(n, r * p)
    ranks = rankdata(flat, axis=0, method="average")
    ranks -= ranks.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(ranks * ranks, axis=0))
    denom = np.where(denom > 0, denom, 1.0)
    spearman = (ranks.T @ ranks) / np.outer(denom, denom)
    spearman = np.clip(_symmetrize(spearman), -1.0, 1.0)
    latent = 2.0 * np.sin(np.pi * spearman / 6.0)
    np.fill_diagonal(latent, 1.0)
    # Pairwise copula correlations can be slightly indefinite; project once.
    eig, vec = np.linalg.eigh(_symmetrize(latent))
    eig = np.maximum(eig, min_eig)
    latent = (vec * eig) @ vec.T
    d = np.sqrt(np.diag(latent))
    latent = latent / np.outer(d, d)
    np.fill_diagonal(latent, 1.0)
    return _symmetrize(latent)


@dataclass(frozen=True)
class EstimatorResult:
    concept_covariance: np.ndarray | None
    concept_precision: np.ndarray | None
    converged: bool
    iterations: int
    message: str
    estimator_kind: str
    auxiliary: Mapping[str, Any] = field(default_factory=dict)
    stat_objective: float | None = None
    penalized_objective: float | None = None

    @property
    def S(self):
        return self.concept_covariance

    @property
    def Theta(self):
        return self.concept_precision

    def info(self):
        out = {
            "converged": self.converged,
            "iterations": self.iterations,
            "message": self.message,
            "estimator_kind": self.estimator_kind,
            "stat_objective": self.stat_objective,
            "penalized_objective": self.penalized_objective,
        }
        for k, v in self.auxiliary.items():
            if isinstance(v, np.ndarray):
                continue
            out[k] = v
        return out


class VectorGGMEstimator:
    """One patient = one scalar concept vector; current v0.1.1 estimator."""
    kind = VECTOR_MODE

    def __init__(self, samples, data_config=DataConfig(), solver_config=SolverConfig()):
        x = np.asarray(samples, dtype=float)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2 or not np.isfinite(x).all():
            raise ValueError("VectorGGM samples must be finite [n_samples, n_concepts]")
        self.raw_samples = readonly(x)
        self.data_config = data_config
        self.transformer = RankGaussianTransformer().fit(x) if data_config.nonparanormal else None
        z = self.transformer.transform(x) if self.transformer is not None else x
        self.S = readonly(training_covariance(z, data_config.covariance_ridge))
        self.data_fingerprint = _fingerprint_array(x, self.kind)
        self.inner_solver = WeightedGraphicalLasso(solver_config)

    @property
    def solve_calls(self):
        return self.inner_solver.solve_calls

    def solve(self, concept_penalty, initial_theta=None, initial_representation_precision=None):
        result = self.inner_solver.solve(self.S, concept_penalty, initial_theta=initial_theta)
        aux = {
            "data_fingerprint": self.data_fingerprint,
            "sample_semantics": "patient",
            "n_samples": self.raw_samples.shape[0],
            "representation_dim": 1,
            "inner_solver_calls": 1,
        }
        return EstimatorResult(
            self.S if result.converged else None,
            result.Theta,
            result.converged,
            result.iterations,
            result.message,
            self.kind,
            freeze(aux),
            None if not result.converged else float(-np.linalg.slogdet(result.Theta)[1] + np.sum(self.S * result.Theta)),
            result.objective,
        )


class MNGMEstimator:
    """Nonparanormal matrix-normal graphical model with alternating precisions.

    Input is [n_matrix_samples, representation_dim, n_concepts].  The sample axis
    may represent patients or bootstrap model replicates; estimation is identical.

    Compared with the original `bigraph_phy.py` prototype, this implementation:
    - preserves its Gaussian-copula / alternating A-B idea;
    - fixes matrix-cell flattening so block semantics are explicit;
    - feeds effective covariance matrices directly into graphical-lasso objectives
      instead of calling GraphicalLasso.fit(covariance_as_observations);
    - uses the custom weighted solver only on the concept axis, where GRPO needs
      edge-specific penalties; the representation axis uses sklearn graphical_lasso
      with a scalar penalty and coordinate descent;
    - supports an edge-specific concept penalty matrix for GRPO;
    - avoids the original min-max normalization of covariance entries;
    - has no mandatory CUDA/numba dependency.
    """

    def __init__(self, samples, mode: str, config=MNGMConfig(), solver_config=SolverConfig()):
        if mode not in MATRIX_MODES:
            raise ValueError(f"MNGM mode must be one of {MATRIX_MODES}")
        x = np.asarray(samples, dtype=float)
        if x.ndim != 3 or min(x.shape) < 2 or not np.isfinite(x).all():
            raise ValueError("MNGM samples must be finite [n_samples, representation_dim, n_concepts]")
        self.samples = readonly(x)
        self.mode = mode
        self.kind = mode
        self.config = config
        self.n_samples, self.r, self.p = x.shape
        self.sample_semantics = "patient" if mode == PATIENT_MATRIX_MODE else "bootstrap"
        self.data_fingerprint = _fingerprint_array(x, mode)
        self.concept_solver = WeightedGraphicalLasso(solver_config)
        self.solver_config = solver_config
        self.representation_solve_calls = 0
        self.solve_calls = 0
        if config.transform == "rank_gaussian":
            self.transformed = readonly(_rank_gaussian_tensor(x))
            self.latent_correlation = None
        elif config.transform == "spearman_sine":
            self.transformed = None
            self.latent_correlation = readonly(_spearman_sine_correlation(x, config.min_eig))
        else:
            raise ValueError("Unknown MNGM transform")

    def _concept_covariance(self, representation_precision):
        b = symmetric_matrix(representation_precision, "representation_precision", self.r)
        if self.transformed is not None:
            # (1/(N r)) sum_s H_s^T B H_s
            s = np.einsum("ab,sai,sbj->ij", b, self.transformed, self.transformed,
                          optimize=True) / (self.n_samples * self.r)
        else:
            r4 = self.latent_correlation.reshape(self.r, self.p, self.r, self.p)
            s = np.einsum("ab,aibj->ij", b, r4, optimize=True) / self.r
        return _psd_ridge(s, self.config.covariance_ridge, self.config.min_eig)

    def _representation_covariance(self, concept_precision):
        a = symmetric_matrix(concept_precision, "concept_precision", self.p)
        if self.transformed is not None:
            # (1/(N p)) sum_s H_s A H_s^T
            s = np.einsum("ij,sai,sbj->ab", a, self.transformed, self.transformed,
                          optimize=True) / (self.n_samples * self.p)
        else:
            r4 = self.latent_correlation.reshape(self.r, self.p, self.r, self.p)
            s = np.einsum("ij,aibj->ab", a, r4, optimize=True) / self.p
        return _psd_ridge(s, self.config.covariance_ridge, self.config.min_eig)

    def _solve_representation_scalar(self, covariance, progress_callback=None):
        """Solve the scalar-penalty representation graph with sklearn GLasso.

        sklearn's objective penalizes the sum over all off-diagonal entries, so a
        symmetric edge is counted twice.  Our project convention stores lambda once
        per undirected edge (upper triangle), hence alpha=lambda/2 matches the same
        objective.
        """
        s = symmetric_matrix(covariance, "representation_covariance", self.r)
        alpha = float(self.config.representation_penalty) / 2.0
        self.representation_solve_calls += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                cov_est, precision, costs, n_iter = graphical_lasso(
                    emp_cov=s,
                    alpha=alpha,
                    mode="cd",
                    tol=max(float(self.solver_config.kkt_tol), 1e-6),
                    enet_tol=min(1e-4, max(float(self.solver_config.kkt_tol) / 10.0, 1e-8)),
                    max_iter=min(int(self.solver_config.max_iter), 500),
                    verbose=False,
                    return_costs=True,
                    return_n_iter=True,
                )
        except (ConvergenceWarning, FloatingPointError, np.linalg.LinAlgError) as exc:
            return None, {
                "converged": False,
                "iterations": None,
                "message": f"sklearn graphical_lasso failed: {exc}",
                "alpha": alpha,
                "solver": "sklearn_graphical_lasso_cd",
            }
        precision = _symmetrize(np.asarray(precision, dtype=float))
        try:
            mineig = float(np.linalg.eigvalsh(precision)[0])
        except np.linalg.LinAlgError:
            mineig = float("nan")
        if not np.isfinite(precision).all() or not np.isfinite(mineig) or mineig <= 0:
            return None, {
                "converged": False,
                "iterations": int(n_iter),
                "message": "sklearn graphical_lasso returned non-SPD precision",
                "alpha": alpha,
                "solver": "sklearn_graphical_lasso_cd",
                "min_eigenvalue": mineig,
            }
        final_cost = None
        final_dual_gap = None
        if costs:
            try:
                final_cost = float(costs[-1][0])
                final_dual_gap = float(costs[-1][1])
            except (TypeError, ValueError, IndexError):
                pass
        if progress_callback is not None:
            progress_callback({
                "stage": "representation_glasso",
                "iteration": int(n_iter),
                "max_iter": min(int(self.solver_config.max_iter), 500),
                "dimension": int(self.r),
                "primal": float("nan"),
                "dual": 0.0 if final_dual_gap is None else abs(final_dual_gap),
            })
        return precision, {
            "converged": True,
            "iterations": int(n_iter),
            "message": "converged",
            "alpha": alpha,
            "solver": "sklearn_graphical_lasso_cd",
            "final_cost": final_cost,
            "final_dual_gap": final_dual_gap,
            "min_eigenvalue": mineig,
        }

    def _normalize_representation_precision(self, b):
        if self.config.scale_constraint == "trace":
            scale = float(np.trace(b) / self.r)
            if not np.isfinite(scale) or scale <= 0:
                raise np.linalg.LinAlgError("invalid representation precision scale")
            return _symmetrize(b / scale), scale
        return _symmetrize(b), 1.0

    def _statistical_loss(self, a, b):
        sign_a, logdet_a = np.linalg.slogdet(a)
        sign_b, logdet_b = np.linalg.slogdet(b)
        if sign_a <= 0 or sign_b <= 0:
            raise np.linalg.LinAlgError("MNGM precisions must be SPD")
        if self.transformed is not None:
            trace = np.einsum("ab,sai,ij,sbj->", b, self.transformed, a,
                              self.transformed, optimize=True) / self.n_samples
        else:
            full_precision = np.kron(b, a)  # representation-major flattening
            trace = float(np.sum(self.latent_correlation * full_precision))
        # Divide by representation dimension so RewardFunction's later /p gives
        # approximately per-cell degradation, independent of representation width.
        return float((-self.p * logdet_b - self.r * logdet_a + trace) / self.r)

    def solve(self, concept_penalty, initial_theta=None, initial_representation_precision=None,
              progress_callback=None):
        lam_a = penalty_matrix(self.p, concept_penalty)
        lam_b = penalty_matrix(self.r, self.config.representation_penalty)
        a = np.eye(self.p) if initial_theta is None else symmetric_matrix(initial_theta, "initial_theta", self.p).copy()
        b = (np.eye(self.r) if initial_representation_precision is None else
             symmetric_matrix(initial_representation_precision, "initial_representation_precision", self.r).copy())
        np.linalg.cholesky(a)
        np.linalg.cholesky(b)
        b, _ = self._normalize_representation_precision(b)
        self.solve_calls += 1
        start_inner_calls = self.concept_solver.solve_calls + self.representation_solve_calls
        last_sc = last_sr = None
        concept_info = representation_info = None
        for iteration in range(1, self.config.max_iter + 1):
            if progress_callback is not None:
                progress_callback({
                    "stage": "mngm_outer",
                    "iteration": int(iteration),
                    "max_iter": int(self.config.max_iter),
                    "representation_dim": int(self.r),
                    "concept_dim": int(self.p),
                })
            last_sc = self._concept_covariance(b)
            ra = self.concept_solver.solve(
                last_sc, lam_a, initial_theta=a,
                progress_callback=progress_callback, label="concept_glasso"
            )
            concept_info = ra.info()
            if not ra.converged:
                return EstimatorResult(None, None, False, iteration,
                    f"concept-axis solve failed: {ra.message}", self.kind,
                    {"data_fingerprint": self.data_fingerprint,
                     "sample_semantics": self.sample_semantics,
                     "representation_dim": self.r,
                     "n_samples": self.n_samples,
                     "inner_solver_calls": self.concept_solver.solve_calls + self.representation_solve_calls - start_inner_calls})
            a_new = np.asarray(ra.Theta)
            last_sr = self._representation_covariance(a_new)
            b_raw, representation_info = self._solve_representation_scalar(
                last_sr, progress_callback=progress_callback
            )
            if b_raw is None:
                return EstimatorResult(None, None, False, iteration,
                    f"representation-axis solve failed: {representation_info['message']}", self.kind,
                    {"data_fingerprint": self.data_fingerprint,
                     "sample_semantics": self.sample_semantics,
                     "representation_dim": self.r,
                     "n_samples": self.n_samples,
                     "representation_solver_info": representation_info,
                     "inner_solver_calls": self.concept_solver.solve_calls + self.representation_solve_calls - start_inner_calls})
            b_new, scale = self._normalize_representation_precision(np.asarray(b_raw))
            diff_a = np.linalg.norm(a_new - a, "fro") / max(1.0, np.linalg.norm(a, "fro"))
            diff_b = np.linalg.norm(b_new - b, "fro") / max(1.0, np.linalg.norm(b, "fro"))
            diff = float(diff_a + diff_b)
            a, b = a_new, b_new
            if diff <= self.config.tol:
                # Recompute concept covariance under the final normalized B.
                last_sc = self._concept_covariance(b)
                final_a = self.concept_solver.solve(
                    last_sc, lam_a, initial_theta=a,
                    progress_callback=progress_callback, label="concept_glasso_final"
                )
                concept_info = final_a.info()
                if final_a.converged:
                    a = np.asarray(final_a.Theta)
                last_sr = self._representation_covariance(a)
                stat = self._statistical_loss(a, b)
                penalty_a = float(np.sum(np.triu(lam_a * np.abs(a), 1)))
                penalty_b = float(np.sum(np.triu(lam_b * np.abs(b), 1))) * self.p / self.r
                aux = {
                    "data_fingerprint": self.data_fingerprint,
                    "sample_semantics": self.sample_semantics,
                    "n_samples": self.n_samples,
                    "representation_dim": self.r,
                    "representation_precision": readonly(b),
                    "representation_covariance": readonly(last_sr),
                    "mngm_transform": self.config.transform,
                    "scale_constraint": self.config.scale_constraint,
                    "last_scale_projection": scale,
                    "alternating_diff": diff,
                    "concept_solver_info": concept_info,
                    "representation_solver_info": representation_info,
                    "inner_solver_calls": self.concept_solver.solve_calls + self.representation_solve_calls - start_inner_calls,
                    "stat_objective": stat,
                    "penalized_objective": stat + penalty_a + penalty_b,
                }
                return EstimatorResult(readonly(last_sc), readonly(a), True, iteration,
                                       "converged", self.kind, freeze(aux), stat,
                                       stat + penalty_a + penalty_b)
        return EstimatorResult(None, None, False, self.config.max_iter,
                               "MNGM alternating updates did not converge", self.kind,
                               {"data_fingerprint": self.data_fingerprint,
                                "sample_semantics": self.sample_semantics,
                                "representation_dim": self.r,
                                "n_samples": self.n_samples,
                                "inner_solver_calls": self.concept_solver.solve_calls + self.representation_solve_calls - start_inner_calls})
