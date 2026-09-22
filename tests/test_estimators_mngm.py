import numpy as np

from graph_mvp.config import DataConfig, MNGMConfig, SolverConfig, EnvironmentConfig
from graph_mvp.estimators import (
    VectorGGMEstimator, MNGMEstimator,
    VECTOR_MODE, PATIENT_MATRIX_MODE, BOOTSTRAP_MATRIX_MODE,
)
from graph_mvp.environment import GraphEnvironment
from graph_mvp.graph_data import GraphSampleSet, save_graph_samples, load_graph_samples
from graph_mvp.types import ActionRecord


def matrix_samples(seed=1, n=120, r=3, p=4):
    rng = np.random.default_rng(seed)
    row_cov = np.array([[1., .25, 0.], [.25, 1., .12], [0., .12, 1.]])[:r, :r]
    col_cov = np.eye(p)
    for j in range(p - 1):
        col_cov[j, j + 1] = col_cov[j + 1, j] = .25
    lr = np.linalg.cholesky(row_cov)
    lc = np.linalg.cholesky(col_cov)
    return np.stack([lr @ rng.normal(size=(r, p)) @ lc.T for _ in range(n)])


def mcfg(transform="rank_gaussian"):
    return MNGMConfig(max_iter=12, tol=3e-3, representation_penalty=.4,
                      covariance_ridge=1e-3, transform=transform)


def scfg():
    return SolverConfig(max_iter=1200, kkt_tol=1e-4)


def test_vector_estimator_is_current_patient_scalar_mode():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 5))
    est = VectorGGMEstimator(x, DataConfig(p=5, n_train=10, n_reward=10, n_test=10), scfg())
    result = est.solve(.4)
    assert result.converged and result.estimator_kind == VECTOR_MODE
    assert result.S.shape == (5, 5) and result.Theta.shape == (5, 5)
    assert result.auxiliary["representation_dim"] == 1


def test_patient_and_bootstrap_modes_share_same_mngm_solver_semantics():
    x = matrix_samples()
    a = MNGMEstimator(x, PATIENT_MATRIX_MODE, mcfg(), scfg()).solve(.25)
    b = MNGMEstimator(x, BOOTSTRAP_MATRIX_MODE, mcfg(), scfg()).solve(.25)
    assert a.converged and b.converged
    np.testing.assert_allclose(a.S, b.S, atol=1e-8)
    np.testing.assert_allclose(a.Theta, b.Theta, atol=1e-8)
    assert a.auxiliary["sample_semantics"] == "patient"
    assert b.auxiliary["sample_semantics"] == "bootstrap"
    assert a.auxiliary["representation_precision"].shape == (3, 3)


def test_mngm_supports_legacy_spearman_sine_transform():
    x = matrix_samples(n=80, r=2, p=3)
    result = MNGMEstimator(x, BOOTSTRAP_MATRIX_MODE, mcfg("spearman_sine"), scfg()).solve(.3)
    assert result.converged
    assert result.auxiliary["mngm_transform"] == "spearman_sine"
    assert np.linalg.eigvalsh(result.Theta)[0] > 0
    assert np.linalg.eigvalsh(result.auxiliary["representation_precision"])[0] > 0


def test_edge_specific_penalty_and_environment_reoptimize_both_axes():
    x = matrix_samples(n=140, r=3, p=4)
    est = MNGMEstimator(x, PATIENT_MATRIX_MODE, mcfg(), scfg())
    env = GraphEnvironment(config=EnvironmentConfig(eta=.5), estimator=est)
    ids = tuple(f"c{i}" for i in range(4))
    state = env.initialize_from_estimator(.2, ids)
    old_b = np.asarray(state.snapshot.auxiliary["representation_precision"])
    action = ActionRecord("cand", state.state_id, (0, 1), "increase", -1., "test")
    cand = env.step(state, action)
    assert cand.valid and cand.snapshot is not None
    assert cand.snapshot.estimator_kind == PATIENT_MATRIX_MODE
    assert cand.snapshot.Lambda[0, 1] > state.snapshot.Lambda[0, 1]
    assert cand.snapshot.auxiliary["representation_precision"].shape == old_b.shape
    assert cand.solver_info["inner_solver_calls"] >= 2
    # The concept-axis penalty is the only policy action; the row-axis precision is re-estimated.
    assert est.solve_calls == 2


def test_graph_sample_npz_roundtrip(tmp_path):
    x = matrix_samples(n=10, r=2, p=3)
    sample_set = GraphSampleSet(("a", "b", "c"), PATIENT_MATRIX_MODE, x)
    path = tmp_path / "graph_data.npz"
    save_graph_samples(path, sample_set)
    loaded = load_graph_samples(path)
    assert loaded.mode == PATIENT_MATRIX_MODE and loaded.concept_ids == sample_set.concept_ids
    np.testing.assert_array_equal(loaded.samples, x)
