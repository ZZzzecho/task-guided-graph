"""Real-data interaction-headroom diagnostics used before expensive graph search."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from .data import RankGaussianTransformer


def _standardize_fit(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    return mean, np.where(scale > 1e-12, scale, 1.0)


def _features(X, pairs, mean, scale):
    z = (X - mean) / scale
    if not pairs:
        return z
    inter = np.column_stack([z[:, i] * z[:, j] for i, j in pairs])
    return np.column_stack((z, inter))


def _fit_tuned(X_train, y_train, X_reward, y_reward, X_test, y_test, pairs,
               c_grid=(0.1, 0.3, 1.0, 3.0, 10.0), max_iter=2000):
    mean, scale = _standardize_fit(X_train)
    tr = _features(X_train, pairs, mean, scale)
    rw = _features(X_reward, pairs, mean, scale)
    te = _features(X_test, pairs, mean, scale)
    best = None
    for C in c_grid:
        model = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, random_state=0)
        model.fit(tr, y_train)
        pr = model.predict_proba(rw)[:, 1]
        loss = float(log_loss(y_reward, pr, labels=[0, 1]))
        if best is None or loss < best[0]:
            best = (loss, C, model)
    reward_loss, C, model = best
    pt = model.predict_proba(te)[:, 1]
    pred = (pt >= 0.5).astype(int)
    return {"num_interactions": len(pairs), "num_features": tr.shape[1], "C": C,
            "reward_log_loss": reward_loss,
            "test_log_loss": float(log_loss(y_test, pt, labels=[0, 1])),
            "test_auroc": float(roc_auc_score(y_test, pt)),
            "test_accuracy": float(accuracy_score(y_test, pred))}


def _residual_ranked_pairs(X_train, y_train, main_model, mean, scale):
    z = (X_train - mean) / scale
    residual = y_train - main_model.predict_proba(z)[:, 1]
    scored = []
    for i in range(z.shape[1]):
        for j in range(i + 1, z.shape[1]):
            v = z[:, i] * z[:, j]
            if np.std(v) <= 1e-12:
                score = 0.0
            else:
                score = abs(float(np.corrcoef(v, residual)[0, 1]))
                if not np.isfinite(score):
                    score = 0.0
            scored.append(((i, j), score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def interaction_headroom(dataset, nonparanormal=True, top_ks=(0, 1, 2, 4, 8, 16, 32),
                         c_grid=(0.1, 0.3, 1.0, 3.0, 10.0)):
    if dataset.X_test is None:
        raise ValueError("Headroom diagnostic requires a held-out test split")
    if nonparanormal:
        transform = RankGaussianTransformer().fit(dataset.X_train)
        Xtr = transform.transform(dataset.X_train)
        Xrw = transform.transform(dataset.X_reward)
        Xte = transform.transform(dataset.X_test)
    else:
        Xtr, Xrw, Xte = dataset.X_train, dataset.X_reward, dataset.X_test
    mean, scale = _standardize_fit(Xtr)
    main_tr, main_rw = _features(Xtr, (), mean, scale), _features(Xrw, (), mean, scale)
    best = None
    for C in c_grid:
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=0).fit(main_tr, dataset.y_train)
        ll = float(log_loss(dataset.y_reward, m.predict_proba(main_rw)[:, 1], labels=[0, 1]))
        if best is None or ll < best[0]:
            best = (ll, C, m)
    ranked = _residual_ranked_pairs(Xtr, dataset.y_train, best[2], mean, scale)
    results = []
    for k in top_ks:
        pairs = [edge for edge, _ in ranked[:int(k)]]
        results.append(_fit_tuned(Xtr, dataset.y_train, Xrw, dataset.y_reward,
                                  Xte, dataset.y_test, pairs, c_grid))
    all_pairs = [(i, j) for i in range(Xtr.shape[1]) for j in range(i + 1, Xtr.shape[1])]
    results.append(_fit_tuned(Xtr, dataset.y_train, Xrw, dataset.y_reward,
                              Xte, dataset.y_test, all_pairs, c_grid))
    return {"top_ranked_pairs": [{"edge": list(edge), "concepts": [dataset.concept_ids[edge[0]], dataset.concept_ids[edge[1]]],
                                  "score": score} for edge, score in ranked[:10]],
            "results": results}
