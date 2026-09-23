#!/usr/bin/env python
"""End-to-end Patent-MNGM + citation-retrieval Graph-RL runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import time
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping
import numpy as np

from graph_mvp.config import Config
from graph_mvp.environment import GraphEnvironment, CandidateBuilder, graph_metrics
from graph_mvp.estimators import MNGMEstimator, PATIENT_MATRIX_MODE
from graph_mvp.patient_repr import PatientMatrixDataset, load_concept_vocabulary
from graph_mvp.policy import GRPOPolicy
from graph_mvp.reward import RewardFunction
from graph_mvp.runner import GraphPhaseRunner
from graph_mvp.task_model import attach_task_lora, load_local_causal_lm, GLM47_FLASH_MODEL_ID
from graph_mvp.task_prototypes import build_task_soft_graph_tokenizer
from graph_mvp.retrieval_task import (
    GraphConditionedRetriever,
    FrozenGraphRetrievalEvaluator,
    RetrievalTaskRelevanceProvider,
    adapt_retrieval_model,
    load_retrieval_context,
)



def _jsonable(value):
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _append_jsonl(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")


def _file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _graph_diff_rows(initial, final):
    ids = initial.concept_ids
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dl = float(final.Lambda[i, j] - initial.Lambda[i, j])
            dr = float(final.Rho[i, j] - initial.Rho[i, j])
            before = bool(initial.A[i, j] != 0)
            after = bool(final.A[i, j] != 0)
            if dl != 0.0 or dr != 0.0 or before != after:
                rows.append({
                    "i": i,
                    "j": j,
                    "concept_i": ids[i],
                    "concept_j": ids[j],
                    "initial_lambda": float(initial.Lambda[i, j]),
                    "final_lambda": float(final.Lambda[i, j]),
                    "delta_lambda": dl,
                    "initial_rho": float(initial.Rho[i, j]),
                    "final_rho": float(final.Rho[i, j]),
                    "delta_rho": dr,
                    "edge_initial": before,
                    "edge_final": after,
                })
    rows.sort(key=lambda x: abs(x["delta_rho"]), reverse=True)
    return rows


def _write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_graph(path, state):
    s = state.snapshot
    np.savez_compressed(
        path,
        concept_ids=np.asarray(s.concept_ids),
        S=s.S,
        Lambda=s.Lambda,
        Theta=s.Theta,
        partial_corr=s.Rho,
        adjacency=s.A,
    )


def main():
    p = argparse.ArgumentParser(description="Patent H04L citation-retrieval Graph-RL")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--concepts", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--train-pools", type=Path, required=True)
    p.add_argument("--graph-pools", type=Path, required=True)
    p.add_argument("--val-pools", type=Path, required=True)
    p.add_argument("--test-pools", type=Path, default=None,
                   help="Optional final-only test candidate pools; never used during graph search.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-mngm-documents", type=int, default=None)
    p.add_argument("--initial-lambda", type=float, default=None)
    p.add_argument("--max-train-queries", type=int, default=32)
    p.add_argument("--max-reward-queries", type=int, default=8)
    p.add_argument("--max-val-queries", type=int, default=8)
    p.add_argument("--max-test-queries", type=int, default=1024)
    p.add_argument("--task-model", default=GLM47_FLASH_MODEL_ID)
    p.add_argument("--task-local-files-only", action="store_true")
    p.add_argument("--task-dtype", default="bfloat16",
                   choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--task-lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--adapt-steps", type=int, default=10)
    p.add_argument("--task-batch-size", type=int, default=1)
    p.add_argument("--candidate-batch-size", type=int, default=8)
    p.add_argument("--task-max-length", type=int, default=384)
    p.add_argument("--score-temperature", type=float, default=0.07)
    p.add_argument("--graph-tokens", type=int, default=8)
    p.add_argument("--graph-hidden-dim", type=int, default=128)
    p.add_argument("--activation-tau", type=float, default=0.1)
    p.add_argument("--phases", type=int, default=None)
    p.add_argument("--policy-updates-per-phase", type=int, default=3)
    p.add_argument("--num-candidates", type=int, default=None)
    args = p.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory already exists and is nonempty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    for sub in ("mngm", "retrieval", "rl", "checkpoints", "graph_analysis"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()
    cfg = Config.load(args.config)
    vocab = load_concept_vocabulary(args.concepts)
    cache = PatientMatrixDataset(args.cache)
    if tuple(cache.concept_ids) != tuple(vocab.concept_ids):
        raise SystemExit("cache concept axis does not match vocabulary")

    manifest = {
        "git_sha": _git_sha(),
        "graph_mvp_version": _pkg_version("task-guided-graph-mvp"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "peft": _pkg_version("peft"),
        "scikit_learn": _pkg_version("scikit-learn"),
        "config_path": str(args.config),
        "config_sha256": _file_sha256(args.config),
        "concepts_path": str(args.concepts),
        "concepts_sha256": _file_sha256(args.concepts),
        "cache": str(args.cache),
        "cache_fingerprint": cache.fingerprint(),
        "train_pools": str(args.train_pools),
        "graph_pools": str(args.graph_pools),
        "val_pools": str(args.val_pools),
        "test_pools": None if args.test_pools is None else str(args.test_pools),
        "candidate_pool_hashes": {
            "train": _file_sha256(args.train_pools),
            "graph": _file_sha256(args.graph_pools),
            "val": _file_sha256(args.val_pools),
            "test": None if args.test_pools is None else _file_sha256(args.test_pools),
        },
        "args": vars(args),
    }
    _write_json(args.output / "run_manifest.json", manifest)

    matrices = cache.materialize(args.max_mngm_documents, dtype=np.float32)
    estimator = MNGMEstimator(matrices, PATIENT_MATRIX_MODE, cfg.mngm, cfg.solver)
    env = GraphEnvironment(config=cfg.environment, estimator=estimator)
    lam0 = cfg.runner.initial_lambda if args.initial_lambda is None else float(args.initial_lambda)
    print(f"Initializing MNGM graph from shape={list(matrices.shape)} lambda={lam0}", flush=True)
    state0 = env.initialize_from_estimator(lam0, cache.concept_ids, state_id="patent-state-0")
    print(f"Initial graph: {graph_metrics(state0.snapshot)}", flush=True)
    _save_graph(args.output / "initial_graph.npz", state0)
    _write_json(args.output / "mngm" / "initial_summary.json", {
        "shape": list(matrices.shape),
        "initial_lambda": lam0,
        "graph_metrics": graph_metrics(state0.snapshot),
        "solver_info": state0.solver_info,
    })

    task_lm, tokenizer = load_local_causal_lm(
        args.task_model,
        dtype=args.task_dtype,
        device_map=args.device_map,
        local_files_only=args.task_local_files_only,
    )
    task_lm = attach_task_lora(
        task_lm, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    graph_tokenizer = build_task_soft_graph_tokenizer(
        task_lm,
        tokenizer,
        vocab.concept_texts,
        num_tokens=args.graph_tokens,
        graph_hidden_dim=args.graph_hidden_dim,
        activation_tau=args.activation_tau,
        activation_normalization="sigmoid",
    )
    model = GraphConditionedRetriever(task_lm, graph_tokenizer)

    common = dict(
        prepared_dir=args.data_dir,
        concept_ids=cache.concept_ids,
        tokenizer=tokenizer,
        batch_size=args.task_batch_size,
        max_length=args.task_max_length,
        score_temperature=args.score_temperature,
    )
    train_ctx = load_retrieval_context(
        pool_path=args.train_pools, max_queries=args.max_train_queries, **common
    )
    graph_ctx = load_retrieval_context(
        pool_path=args.graph_pools, max_queries=args.max_reward_queries, **common
    )
    val_ctx = load_retrieval_context(
        pool_path=args.val_pools, max_queries=args.max_val_queries, **common
    )
    test_ctx = None
    if args.test_pools is not None:
        test_ctx = load_retrieval_context(
            pool_path=args.test_pools, max_queries=args.max_test_queries, **common
        )

    evaluator = FrozenGraphRetrievalEvaluator(
        model, candidate_batch_size=args.candidate_batch_size
    )
    for name, ctx in (("train", train_ctx), ("graph", graph_ctx), ("val", val_ctx)):
        prep = evaluator.prepare(ctx)
        print(f"candidate-index {name}: {prep}", flush=True)
    if test_ctx is not None:
        prep = evaluator.prepare(test_ctx)
        print(f"candidate-index test(final-only): {prep}", flush=True)

    import torch
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.task_lr)

    warmup = None
    if args.warmup_steps > 0:
        warmup = adapt_retrieval_model(
            model, evaluator, state0.snapshot, train_ctx, optimizer,
            steps=args.warmup_steps, max_grad_norm=cfg.grpo.max_grad_norm,
        )
        print(f"retrieval warmup: {warmup}", flush=True)

    initial_graph = evaluator.evaluate_snapshot(state0.snapshot, graph_ctx)
    initial_val = evaluator.evaluate_snapshot(state0.snapshot, val_ctx)
    print(f"initial graph retrieval={dict(initial_graph.extra_metrics)}", flush=True)
    print(f"initial val retrieval={dict(initial_val.extra_metrics)}", flush=True)
    _write_json(args.output / "retrieval" / "initial_graph_queries.json",
                evaluator.score_details(state0.snapshot, graph_ctx, top_k=10))
    _write_json(args.output / "retrieval" / "initial_val_queries.json",
                evaluator.score_details(state0.snapshot, val_ctx, top_k=10))

    policy = GRPOPolicy(cfg.grpo, seed=cfg.runner.seed)
    reward_fn = RewardFunction(cfg.reward)
    builder = CandidateBuilder(cfg.candidate)
    feature_provider = RetrievalTaskRelevanceProvider(evaluator)

    def task_adapter(accepted_state):
        return adapt_retrieval_model(
            model, evaluator, accepted_state.snapshot, train_ctx, optimizer,
            steps=args.adapt_steps, max_grad_norm=cfg.grpo.max_grad_norm,
        )

    runner = GraphPhaseRunner(
        env, policy, evaluator, reward_fn,
        candidate_builder=builder,
        validation_evaluator=evaluator,
        min_validation_improvement=cfg.acceptance.min_task_improvement,
        task_adapter=task_adapter,
        feature_provider=feature_provider,
    )

    def report(record):
        print(
            f"phase={record.phase + 1} accepted={record.accepted} "
            f"{record.parent_state_id}->{record.next_state_id} "
            f"elapsed={record.elapsed_seconds:.1f}s "
            f"reason={record.acceptance_reason}",
            flush=True,
        )
        phase_payload = {
            "phase": record.phase,
            "parent_state_id": record.parent_state_id,
            "next_state_id": record.next_state_id,
            "proposal_id": record.proposal_id,
            "accepted": record.accepted,
            "acceptance_reason": record.acceptance_reason,
            "adapted": record.adapted,
            "elapsed_seconds": record.elapsed_seconds,
            "validation_baseline": record.validation_baseline,
            "validation_proposal": record.validation_proposal,
            "post_adaptation_validation": record.post_adaptation_validation,
            "adaptation_result": record.adaptation_result,
            "updates": [],
        }
        for update in record.updates:
            candidate_rows = []
            feature_by_candidate = {
                x["candidate_id"]: x["edges"] for x in update.selected_edge_features
            }
            for candidate, evaluation, reward in zip(
                update.candidates, update.evaluations, update.rewards
            ):
                action_rows = []
                actions = candidate.action.actions if hasattr(candidate.action, "actions") else (candidate.action,)
                for action in actions:
                    i, j = action.edge
                    row = {
                        "edge": [i, j],
                        "concept_i": state0.snapshot.concept_ids[i],
                        "concept_j": state0.snapshot.concept_ids[j],
                        "action": action.action,
                        "joint_component_log_prob": action.log_prob,
                        "selection_log_prob": action.selection_log_prob,
                        "direction_log_prob": action.direction_log_prob,
                        "old_lambda": float(state0.snapshot.Lambda[i, j]) if candidate.parent_state_id == state0.state_id else None,
                    }
                    action_rows.append(row)
                candidate_rows.append({
                    "candidate_id": candidate.candidate_id,
                    "valid": candidate.valid,
                    "error": candidate.error,
                    "solver_info": candidate.solver_info,
                    "graph_metrics": candidate.graph_metrics,
                    "evaluation": evaluation,
                    "reward": reward,
                    "selected_edge_features": feature_by_candidate.get(candidate.candidate_id, []),
                    "actions": action_rows,
                })
                _append_jsonl(args.output / "rl" / "candidate_trajectory.jsonl", {
                    "phase": record.phase,
                    "update": update.update,
                    **candidate_rows[-1],
                })
            phase_payload["updates"].append({
                "update": update.update,
                "elapsed_seconds": update.elapsed_seconds,
                "solver_calls": update.solver_calls,
                "policy_update": update.policy_update,
                "candidates": candidate_rows,
            })
        _write_json(args.output / "rl" / f"phase_{record.phase:03d}.json", phase_payload)

        if record.proposal_id is not None:
            proposal = None
            for update in record.updates:
                for candidate in update.candidates:
                    if candidate.candidate_id == record.proposal_id:
                        proposal = candidate
                        break
            if proposal is not None and proposal.snapshot is not None:
                ckpt = args.output / "checkpoints" / f"phase_{record.phase:03d}"
                ckpt.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    ckpt / "proposal_graph.npz",
                    concept_ids=np.asarray(proposal.snapshot.concept_ids),
                    Lambda=proposal.snapshot.Lambda,
                    Theta=proposal.snapshot.Theta,
                    partial_corr=proposal.snapshot.Rho,
                )
                policy.save(ckpt / "graph_policy.pt")
                torch.save(graph_tokenizer.state_dict(), ckpt / "soft_graph_tokenizer.pt")

    result = runner.run(
        state0,
        graph_ctx,
        val_ctx,
        phases=cfg.runner.rounds if args.phases is None else args.phases,
        policy_updates_per_phase=args.policy_updates_per_phase,
        num_candidates=cfg.runner.num_candidates if args.num_candidates is None else args.num_candidates,
        on_phase=report,
    )

    _save_graph(args.output / "final_graph.npz", result.final_state)
    policy.save(args.output / "graph_policy.pt")
    torch.save(graph_tokenizer.state_dict(), args.output / "soft_graph_tokenizer.pt")
    if hasattr(task_lm, "save_pretrained"):
        task_lm.save_pretrained(args.output / "task_lora")

    final_graph = evaluator.evaluate_snapshot(result.final_state.snapshot, graph_ctx)
    final_val = evaluator.evaluate_snapshot(result.final_state.snapshot, val_ctx)
    _write_json(args.output / "retrieval" / "final_graph_queries.json",
                evaluator.score_details(result.final_state.snapshot, graph_ctx, top_k=10))
    _write_json(args.output / "retrieval" / "final_val_queries.json",
                evaluator.score_details(result.final_state.snapshot, val_ctx, top_k=10))

    test_metrics = None
    if test_ctx is not None:
        test_metrics = evaluator.evaluate_snapshot(result.final_state.snapshot, test_ctx)
        _write_json(args.output / "retrieval" / "final_test_queries.json",
                    evaluator.score_details(result.final_state.snapshot, test_ctx, top_k=10))

    diff_rows = _graph_diff_rows(state0.snapshot, result.final_state.snapshot)
    _write_csv(args.output / "graph_analysis" / "edge_changes.csv", diff_rows)
    _write_csv(args.output / "graph_analysis" / "top_changed_edges.csv", diff_rows[:500])
    degree_rows = []
    d0 = np.count_nonzero(state0.snapshot.A, axis=1)
    d1 = np.count_nonzero(result.final_state.snapshot.A, axis=1)
    for i, cid in enumerate(state0.snapshot.concept_ids):
        degree_rows.append({
            "concept_index": i, "concept_id": cid,
            "initial_degree": int(d0[i]), "final_degree": int(d1[i]),
            "delta_degree": int(d1[i] - d0[i]),
        })
    degree_rows.sort(key=lambda x: abs(x["delta_degree"]), reverse=True)
    _write_csv(args.output / "graph_analysis" / "degree_changes.csv", degree_rows)

    summary = {
        "cache": str(args.cache),
        "cache_fingerprint": cache.fingerprint(),
        "mngm_shape": list(matrices.shape),
        "initial_lambda": lam0,
        "task": "fixed-pool examiner-citation retrieval",
        "candidate_index_frozen": True,
        "pool_size": graph_ctx.pool_size,
        "task_model": args.task_model,
        "warmup": warmup,
        "phases": len(result.phases),
        "accepted_phases": int(sum(x.accepted for x in result.phases)),
        "initial_graph_metrics": graph_metrics(state0.snapshot),
        "final_graph_metrics": graph_metrics(result.final_state.snapshot),
        "initial_reward_retrieval": dict(initial_graph.extra_metrics),
        "initial_validation_retrieval": dict(initial_val.extra_metrics),
        "final_reward_retrieval": dict(final_graph.extra_metrics),
        "final_validation_retrieval": dict(final_val.extra_metrics),
        "policy_updates_per_phase": args.policy_updates_per_phase,
        "num_candidates_per_update": (
            cfg.runner.num_candidates if args.num_candidates is None else args.num_candidates
        ),
        "total_candidate_evaluations_planned": int(
            len(result.phases) * args.policy_updates_per_phase
            * (cfg.runner.num_candidates if args.num_candidates is None else args.num_candidates)
        ),
        "test_split_touched": test_ctx is not None,
        "final_test_retrieval": None if test_metrics is None else dict(test_metrics.extra_metrics),
        "runtime_seconds": float(time.perf_counter() - run_start),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
