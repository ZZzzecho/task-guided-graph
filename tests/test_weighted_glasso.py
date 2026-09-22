import numpy as np
import pytest
from sklearn.covariance import graphical_lasso
from graph_mvp.config import SolverConfig
from graph_mvp.weighted_glasso import WeightedGraphicalLasso, penalty_matrix, kkt_residual, objective


def test_diagonal_analytic_solution():
    s = np.diag([.5, 1., 2.])
    r = WeightedGraphicalLasso().solve(s, .7)
    assert r.converged
    np.testing.assert_allclose(r.Theta, np.diag([2., 1., .5]), atol=3e-5)


@pytest.mark.parametrize("lam", [.1, .6, 1.4])
def test_two_dimensional_analytic_solution_and_factor_two(lam):
    s = np.array([[1., .6], [.6, 2.]])
    off = max(.6 - lam / 2, 0.)
    expected = np.linalg.inv([[1., off], [off, 2.]])
    result = WeightedGraphicalLasso().solve(s, lam)
    assert result.converged
    np.testing.assert_allclose(result.Theta, expected, atol=3e-5)


def test_scalar_agrees_with_independent_sklearn_solver():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(120, 7)) @ rng.normal(size=(7, 7))
    s = np.cov(x, rowvar=False) + np.eye(7) * .3
    lam = .4
    ours = WeightedGraphicalLasso().solve(s, lam)
    _, expected = graphical_lasso(s, alpha=lam / 2, tol=1e-9, enet_tol=1e-12, max_iter=1000)
    assert ours.converged
    np.testing.assert_allclose(ours.Theta, expected, atol=5e-5, rtol=1e-4)


def test_heterogeneous_penalties_kkt_and_global_optimization():
    s = np.array([[1., .5, .25], [.5, 1.4, .6], [.25, .6, 1.2]])
    lam = np.array([[0., .15, .9], [.15, 0., .4], [.9, .4, 0.]])
    solver = WeightedGraphicalLasso()
    r = solver.solve(s, lam)
    assert r.converged and r.kkt_residual <= solver.config.kkt_tol
    assert np.linalg.eigvalsh(r.Theta)[0] > 0
    np.testing.assert_allclose(r.Theta, r.Theta.T, atol=1e-14)
    # Convex objective + SPD + KKT gives a global optimality certificate.
    assert kkt_residual(s, lam, r.Theta) < 2e-5
    assert objective(s, lam, r.Theta) <= objective(s, lam, np.linalg.inv(s))
    changed = lam.copy()
    changed[0, 1] = changed[1, 0] = .8
    cold = solver.solve(s, changed)
    warm = solver.solve(s, changed, initial_theta=r.Theta)
    assert cold.converged and warm.converged
    np.testing.assert_allclose(cold.Theta, warm.Theta, atol=3e-5)
    assert not np.allclose(r.Theta, warm.Theta)


def test_matrix_penalty_matches_scalar():
    s = np.array([[1., .5], [.5, 1.]])
    solver = WeightedGraphicalLasso()
    np.testing.assert_allclose(solver.solve(s, .3).Theta, solver.solve(s, penalty_matrix(2, .3)).Theta)


def test_failed_solve_publishes_no_precision():
    r = WeightedGraphicalLasso(SolverConfig(max_iter=1)).solve([[1., .7], [.7, 1.]], .3)
    assert not r.converged and r.Theta is None


@pytest.mark.parametrize("s,penalty", [([[1., 2.], [2., 1.]], .2),
    ([[1., .2], [.3, 1.]], .2), ([[1., 0.], [0., 1.]], -.1),
    ([[1., 0.], [0., 1.]], [[0., .1], [.3, 0.]]),
    ([[1., float('nan')], [float('nan'), 1.]], .2)])
def test_invalid_inputs(s, penalty):
    with pytest.raises(ValueError):
        WeightedGraphicalLasso().solve(s, penalty)
