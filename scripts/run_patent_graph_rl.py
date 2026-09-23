#!/usr/bin/env python
"""End-to-end Patent-MNGM + citation-retrieval Graph-RL runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-mngm-documents", type=int, default=None)
    p.add_argument("--initial-lambda", type=float, default=None)
    p.add_argument("--max-train-queries", type=int, default=32)
    p.add_argument("--max-reward-queries", type=int, default=8)
    p.add_argument("--max-val-queries", type=int, default=8)
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
    p.add_argument("--policy-updates-per-phase", type=int, default=1)
    p.add_argument("--num-candidates", type=int, default=None)
    args = p.parse_args()

    cfg = Config.load(args.config)
    vocab = load_concept_vocabulary(args.concepts)
    cache = PatientMatrixDataset(args.cache)
    if tuple(cache.concept_ids) != tuple(vocab.concept_ids):
        raise SystemExit("cache concept axis does not match vocabulary")

    matrices = cache.materialize(args.max_mngm_documents, dtype=np.float32)
    estimator = MNGMEstimator(matrices, PATIENT_MATRIX_MODE, cfg.mngm, cfg.solver)
    env = GraphEnvironment(config=cfg.environment, estimator=estimator)
    lam0 = cfg.runner.initial_lambda if args.initial_lambda is None else float(args.initial_lambda)
    print(f"Initializing MNGM graph from shape={list(matrices.shape)} lambda={lam0}", flush=True)
    state0 = env.initialize_from_estimator(lam0, cache.concept_ids, state_id="patent-state-0")
    print(f"Initial graph: {graph_metrics(state0.snapshot)}", flush=True)

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

    evaluator = FrozenGraphRetrievalEvaluator(
        model, candidate_batch_size=args.candidate_batch_size
    )
    for name, ctx in (("train", train_ctx), ("graph", graph_ctx), ("val", val_ctx)):
        prep = evaluator.prepare(ctx)
        print(f"candidate-index {name}: {prep}", flush=True)

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
            f"reason={record.acceptance_reason}",
            flush=True,
        )

    result = runner.run(
        state0,
        graph_ctx,
        val_ctx,
        phases=cfg.runner.rounds if args.phases is None else args.phases,
        policy_updates_per_phase=args.policy_updates_per_phase,
        num_candidates=cfg.runner.num_candidates if args.num_candidates is None else args.num_candidates,
        on_phase=report,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    _save_graph(args.output / "initial_graph.npz", state0)
    _save_graph(args.output / "final_graph.npz", result.final_state)
    policy.save(args.output / "graph_policy.pt")
    torch.save(graph_tokenizer.state_dict(), args.output / "soft_graph_tokenizer.pt")
    if hasattr(task_lm, "save_pretrained"):
        task_lm.save_pretrained(args.output / "task_lora")

    final_graph = evaluator.evaluate_snapshot(result.final_state.snapshot, graph_ctx)
    final_val = evaluator.evaluate_snapshot(result.final_state.snapshot, val_ctx)
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
        "test_split_touched": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
