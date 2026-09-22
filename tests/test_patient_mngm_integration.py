import numpy as np

from graph_mvp.estimators import MNGMEstimator, PATIENT_MATRIX_MODE
from graph_mvp.config import MNGMConfig, SolverConfig
from graph_mvp.weighted_glasso import penalty_matrix


def test_small_patient_matrix_mngm_smoke():
    rng = np.random.default_rng(123)
    # Small stand-in for [N,1024,P]; semantics, not production size, is tested here.
    x = rng.normal(size=(10, 5, 4))
    x[:, :, 1] += 0.5 * x[:, :, 0]
    est = MNGMEstimator(
        x, PATIENT_MATRIX_MODE,
        MNGMConfig(max_iter=12, tol=1e-3, representation_penalty=0.5),
        SolverConfig(max_iter=1000),
    )
    result = est.solve(penalty_matrix(4, 0.4))
    assert result.converged
    assert result.Theta.shape == (4, 4)
    assert result.auxiliary["representation_precision"].shape == (5, 5)
    assert result.auxiliary["sample_semantics"] == "patient"
