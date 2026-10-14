import numpy as np
from sklearn.metrics import average_precision_score,f1_score
from scripts.summarize_north_graph_prior import weighted_macro_metrics


def test_weighted_bootstrap_metrics_match_sklearn_with_ties_and_duplicate_counts():
    y=np.array([[1,0],[0,1],[1,1],[0,0],[1,0]])
    probability=np.array([[.8,.1],[.8,.8],[.5,.8],[.2,.5],[.5,.2]])
    weights=np.array([[1,1,1,1,1],[0,3,2,0,1],[4,0,1,2,0]])
    actual=weighted_macro_metrics(y,probability,weights)
    expected=np.array([[np.mean([average_precision_score(y[:,j],probability[:,j],sample_weight=w) for j in range(2)]),
        f1_score(y,probability>=.5,average='macro',sample_weight=w,zero_division=0)] for w in weights])
    np.testing.assert_allclose(actual,expected,atol=1e-14)
