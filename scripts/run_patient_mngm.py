#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from graph_mvp.config import Config
from graph_mvp.estimators import MNGMEstimator, PATIENT_MATRIX_MODE
from graph_mvp.patient_repr import PatientMatrixDataset
from graph_mvp.weighted_glasso import penalty_matrix


def partial_correlation(theta):
    d = np.sqrt(np.clip(np.diag(theta), 1e-30, None))
    rho = -theta / np.outer(d, d)
    np.fill_diagonal(rho, 0.0)
    return (rho + rho.T) / 2


def main():
    p = argparse.ArgumentParser(description="Smoke/scale-test Patient-MNGM from sharded cache")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-mngm-patients", type=int, default=None)
    p.add_argument("--initial-lambda", type=float, default=None)
    args = p.parse_args()
    cfg = Config.load(args.config)
    ds = PatientMatrixDataset(args.cache)
    x = ds.materialize(max_samples=args.max_mngm_patients, dtype=np.float32)
    est = MNGMEstimator(x, PATIENT_MATRIX_MODE, cfg.mngm, cfg.solver)
    lam0 = cfg.runner.initial_lambda if args.initial_lambda is None else args.initial_lambda
    lam = penalty_matrix(ds.num_concepts, lam0)
    def progress(info):
        stage = info.get("stage", "solver")
        if stage == "mngm_outer":
            print(
                f"[MNGM] outer {info['iteration']}/{info['max_iter']} "
                f"R={info['representation_dim']} P={info['concept_dim']}",
                flush=True,
            )
        else:
            print(
                f"[{stage}] iter {info['iteration']}/{info['max_iter']} "
                f"dim={info['dimension']} primal={info['primal']:.3e} "
                f"dual={info['dual']:.3e}",
                flush=True,
            )

    print(
        f"Starting MNGM with shape={list(x.shape)} lambda={lam0} "
        f"(dense NumPy solver; CPU-bound)",
        flush=True,
    )
    result = est.solve(lam, progress_callback=progress)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "cache": str(args.cache),
        "cache_fingerprint": ds.fingerprint(),
        "n_materialized": int(len(x)),
        "shape": list(x.shape),
        "initial_lambda": float(lam0),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "message": result.message,
        "estimator": result.info(),
    }
    if result.converged:
        rho = partial_correlation(result.Theta)
        np.savez_compressed(
            args.output / "initial_patient_mngm.npz",
            concept_ids=np.asarray(ds.concept_ids),
            Lambda=lam,
            Theta=result.Theta,
            partial_corr=rho,
        )
        summary["n_nonzero_edges"] = int(np.count_nonzero(np.triu(np.abs(result.Theta) > 1e-10, 1)))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not result.converged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
