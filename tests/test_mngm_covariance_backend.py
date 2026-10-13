import numpy as np
import pytest
pytest.importorskip('torch')
from graph_mvp.config import MNGMConfig
from graph_mvp.estimators import MNGMEstimator, PATIENT_MATRIX_MODE


def test_double_tensor_covariances_and_likelihood_match_numpy():
    rng=np.random.default_rng(51);samples=rng.normal(size=(17,3,5))
    left=MNGMEstimator(samples,PATIENT_MATRIX_MODE)
    right=MNGMEstimator(samples,PATIENT_MATRIX_MODE,covariance_device='cpu')
    a=rng.normal(size=(5,5));a=a.T@a+np.eye(5)
    b=rng.normal(size=(3,3));b=b.T@b+np.eye(3)
    np.testing.assert_allclose(left._concept_covariance(b),right._concept_covariance(b),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(left._representation_covariance(a),right._representation_covariance(a),rtol=1e-12,atol=1e-12)
    assert left._statistical_loss(a,b)==pytest.approx(right._statistical_loss(a,b),rel=1e-12)


def test_tensor_backend_does_not_silently_change_copula_transform():
    samples=np.random.default_rng(2).normal(size=(17,3,5))
    with pytest.raises(ValueError,match='rank_gaussian'):
        MNGMEstimator(samples,PATIENT_MATRIX_MODE,MNGMConfig(transform='spearman_sine'),covariance_device='cpu')
