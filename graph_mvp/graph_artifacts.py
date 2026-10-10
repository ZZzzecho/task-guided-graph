"""Verified reuse of a completed full-scale fixed-lambda MNGM fit."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .types import GraphSnapshot, GraphState, symmetric_matrix
from .weighted_glasso import kkt_residual


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_fitted_state(directory, cache_path, estimator, concept_ids, cfg, config_path):
    directory, cache_path = Path(directory), Path(cache_path)
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    info = json.loads((directory / "fitted_summary.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or not info["converged"]:
        raise ValueError("Graph fit is incomplete or unconverged")
    if (manifest["config_sha256"] != file_digest(config_path)
            or manifest["effective_solver"] != asdict(cfg.solver)
            or manifest["mngm_config"] != asdict(cfg.mngm)):
        raise ValueError("Graph fit configuration differs from this experiment")
    integrity_path = cache_path / "build_integrity.json"
    if manifest["cache_integrity_sha256"] != file_digest(integrity_path):
        raise ValueError("Graph fit cache integrity manifest differs")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    for name, expected in integrity.items():
        if file_digest(cache_path / name) != expected:
            raise ValueError(f"Cache checksum mismatch: {name}")
    if info["data_fingerprint"] != estimator.data_fingerprint:
        raise ValueError("Graph fit sample fingerprint differs")
    path = directory / "fitted_graph.npz"
    with np.load(path, allow_pickle=False) as raw:
        if tuple(raw["concept_ids"].astype(str)) != tuple(concept_ids):
            raise ValueError("Graph fit concept IDs/order differs")
        b = symmetric_matrix(raw["B"], "B", estimator.r)
        np.linalg.cholesky(b)
        s, theta, lam = (raw[k].copy() for k in ("S", "Theta", "Lambda"))
    recomputed_s = estimator._concept_covariance(b)
    if not np.allclose(s, recomputed_s, atol=1e-10, rtol=1e-8):
        raise ValueError("Graph fit effective covariance differs from cache/B")
    residual = kkt_residual(s, lam, theta)
    if residual > cfg.solver.kkt_tol:
        raise ValueError("Saved graph fails original KKT tolerance")
    aux = dict(info, representation_precision=b)
    snapshot = GraphSnapshot(tuple(concept_ids), s, lam, theta,
        cfg.environment.adjacency_threshold, estimator.kind, aux)
    state = GraphState("patent-fitted-state-0", 0, snapshot, info)
    return state, {"path": str(path), "sha256": file_digest(path),
        "fit_git_sha": manifest["git_sha"], "recomputed_kkt": float(residual),
        "cache_integrity_sha256": manifest["cache_integrity_sha256"]}
