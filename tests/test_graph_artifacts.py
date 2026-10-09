from dataclasses import asdict, replace
import json
import numpy as np
import pytest
from graph_mvp.config import Config
from graph_mvp.estimators import MNGMEstimator, PATIENT_MATRIX_MODE
from graph_mvp.graph_artifacts import load_fitted_state, file_digest


@pytest.fixture
def completed_fit(tmp_path):
    cfg = Config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(cfg)), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    x = np.random.default_rng(2).normal(size=(20, 2, 3)).astype(np.float32)
    np.save(cache / "matrices.npy", x)
    (cache / "build_integrity.json").write_text(json.dumps({
        "matrices.npy": file_digest(cache / "matrices.npy")}), encoding="utf-8")
    estimator = MNGMEstimator(x, PATIENT_MATRIX_MODE, cfg.mngm, cfg.solver)
    result = estimator.solve_initial(.8)
    assert result.converged
    fit = tmp_path / "fit"
    fit.mkdir()
    manifest = {"status": "complete", "config_sha256": file_digest(config_path),
        "effective_solver": asdict(cfg.solver), "mngm_config": asdict(cfg.mngm),
        "cache_integrity_sha256": file_digest(cache / "build_integrity.json"), "git_sha": "test"}
    (fit / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (fit / "fitted_summary.json").write_text(json.dumps(result.info()), encoding="utf-8")
    lam = np.full((3, 3), .8)
    np.fill_diagonal(lam, 0.)
    np.savez(fit / "fitted_graph.npz", concept_ids=np.array(["a", "b", "c"]),
        S=result.S, Theta=result.Theta, Lambda=lam, B=np.eye(2))
    return fit, cache, estimator, ("a", "b", "c"), cfg, config_path


def test_verified_fit_reuse_without_solver_calls(completed_fit):
    fit, cache, est, ids, cfg, config = completed_fit
    before = est.concept_solver.solve_calls
    state, provenance = load_fitted_state(fit, cache, est, ids, cfg, config)
    assert est.concept_solver.solve_calls == before
    assert state.snapshot.auxiliary["data_fingerprint"] == est.data_fingerprint
    assert provenance["recomputed_kkt"] <= cfg.solver.kkt_tol


@pytest.mark.parametrize("corruption", ["cache", "axis", "config", "fingerprint", "theta", "B"])
def test_fit_reuse_rejects_mismatches(completed_fit, corruption):
    fit, cache, est, ids, cfg, config = completed_fit
    if corruption == "cache":
        (cache / "matrices.npy").write_bytes(b"changed")
    elif corruption == "axis":
        ids = tuple(reversed(ids))
    elif corruption == "config":
        cfg = replace(cfg, solver=replace(cfg.solver, max_iter=10000))
    elif corruption == "fingerprint":
        est.data_fingerprint = "changed"
    else:
        path = fit / "fitted_graph.npz"
        with np.load(path) as r:
            values = {k: r[k].copy() for k in r.files}
        values["Theta" if corruption == "theta" else "B"] *= 2
        np.savez(path, **values)
    with pytest.raises(ValueError):
        load_fitted_state(fit, cache, est, ids, cfg, config)
