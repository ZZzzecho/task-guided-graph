#!/usr/bin/env python
"""Bootstrap-MNGM + patent citation-retrieval Graph-RL end-to-end runner."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from graph_mvp.bootstrap import (
    attach_bootstrap_embedding_lora,
    extract_concept_embedding_matrix,
    fixed_pca_projection,
    make_shuffled_versions_frame,
    project_concept_matrix,
    save_bootstrap_bundle,
    train_bootstrap_version,
)
from graph_mvp.config import Config
from graph_mvp.environment import GraphEnvironment, CandidateBuilder, graph_metrics
from graph_mvp.estimators import MNGMEstimator, BOOTSTRAP_MATRIX_MODE
from graph_mvp.patient_repr import load_concept_vocabulary
from graph_mvp.policy import GRPOPolicy
from graph_mvp.reward import RewardFunction
from graph_mvp.runner import GraphPhaseRunner
from graph_mvp.task_model import (
    GLM47_FLASH_MODEL_ID,
    attach_task_lora,
    load_local_causal_lm,
)
from graph_mvp.task_prototypes import build_task_soft_graph_tokenizer
from graph_mvp.retrieval_task import (
    FrozenGraphRetrievalEvaluator,
    GraphConditionedRetriever,
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


def _load_bootstrap_frame(data_dir, max_docs):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Bootstrap patent runner requires pandas") from exc
    frame = pd.read_csv(Path(data_dir) / "train.csv.gz", usecols=["patent_id", "text"])
    if max_docs is not None:
        frame = frame.iloc[: int(max_docs)].copy()
    if len(frame) < 1:
        raise ValueError("bootstrap corpus is empty")
    return frame


def main():
    p = argparse.ArgumentParser(
        description="Bootstrap-MNGM + H04L citation-retrieval Graph-RL"
    )
    p.add_argument("--concepts", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--train-pools", type=Path, required=True)
    p.add_argument("--graph-pools", type=Path, required=True)
    p.add_argument("--val-pools", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p.add_argument("--bootstrap-model", default=GLM47_FLASH_MODEL_ID)
    p.add_argument("--bootstrap-local-files-only", action="store_true")
    p.add_argument("--bootstrap-dtype", default="bfloat16",
                   choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--bootstrap-device-map", default="auto")
    p.add_argument("--bootstrap-versions", type=int, default=4)
    p.add_argument("--bootstrap-docs", type=int, default=8)
    p.add_argument("--bootstrap-steps-per-version", type=int, default=1)
    p.add_argument("--bootstrap-batch-size", type=int, default=1)
    p.add_argument("--bootstrap-max-length", type=int, default=256)
    p.add_argument("--bootstrap-lr", type=float, default=1e-4)
    p.add_argument("--bootstrap-lora-r", type=int, default=8)
    p.add_argument("--bootstrap-lora-alpha", type=int, default=16)
    p.add_argument("--bootstrap-lora-dropout", type=float, default=0.0)
    p.add_argument("--bootstrap-representation-dim", type=int, default=64)
    p.add_argument("--bootstrap-seed", type=int, default=123)

    p.add_argument("--initial-lambda", type=float, default=None)
    p.add_argument("--task-model", default=GLM47_FLASH_MODEL_ID)
    p.add_argument("--task-local-files-only", action="store_true")
    p.add_argument("--task-dtype", default="bfloat16",
                   choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--task-lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--adapt-steps", type=int, default=2)
    p.add_argument("--task-batch-size", type=int, default=1)
    p.add_argument("--candidate-batch-size", type=int, default=4)
    p.add_argument("--task-max-length", type=int, default=256)
    p.add_argument("--score-temperature", type=float, default=0.07)
    p.add_argument("--graph-tokens", type=int, default=8)
    p.add_argument("--graph-hidden-dim", type=int, default=128)
    p.add_argument("--activation-tau", type=float, default=0.1)
    p.add_argument("--max-train-queries", type=int, default=8)
    p.add_argument("--max-reward-queries", type=int, default=4)
    p.add_argument("--max-val-queries", type=int, default=4)
    p.add_argument("--phases", type=int, default=1)
    p.add_argument("--num-candidates", type=int, default=2)
    p.add_argument("--policy-updates-per-phase", type=int, default=1)
    args = p.parse_args()

    if args.bootstrap_versions < 2:
        raise SystemExit("--bootstrap-versions must be >=2")
    if args.bootstrap_docs < 1 or args.bootstrap_steps_per_version < 1:
        raise SystemExit("invalid bootstrap smoke budget")
    if args.bootstrap_lr <= 0:
        raise SystemExit("--bootstrap-lr must be positive")

    cfg = Config.load(args.config)
    vocab = load_concept_vocabulary(args.concepts)
    args.output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage A: bootstrap representation generation.
    # ------------------------------------------------------------------
    print("Loading bootstrap representation model...", flush=True)
    bootstrap_lm, bootstrap_tokenizer = load_local_causal_lm(
        args.bootstrap_model,
        dtype=args.bootstrap_dtype,
        device_map=args.bootstrap_device_map,
        local_files_only=args.bootstrap_local_files_only,
    )
    bootstrap_lm, embed_target = attach_bootstrap_embedding_lora(
        bootstrap_lm,
        r=args.bootstrap_lora_r,
        alpha=args.bootstrap_lora_alpha,
        dropout=args.bootstrap_lora_dropout,
    )

    import torch
    trainable = [p for p in bootstrap_lm.parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit("bootstrap embedding LoRA exposed no trainable parameters")
    bootstrap_optimizer = torch.optim.AdamW(trainable, lr=args.bootstrap_lr)

    frame = _load_bootstrap_frame(args.data_dir, args.bootstrap_docs)
    versions = make_shuffled_versions_frame(
        frame,
        text_col="text",
        n_versions=args.bootstrap_versions,
        seed=args.bootstrap_seed,
    )

    # Fit the dimension-reduction basis once, before any bootstrap training.
    reference_h = extract_concept_embedding_matrix(
        bootstrap_lm, bootstrap_tokenizer, vocab.concept_texts
    )
    pca_mean, pca_components = fixed_pca_projection(
        reference_h,
        output_dim=args.bootstrap_representation_dim,
        seed=args.bootstrap_seed,
    )
    np.savez_compressed(
        args.output / "bootstrap_projection.npz",
        mean=pca_mean,
        components=pca_components,
    )

    snapshots = []
    bootstrap_training = []
    for b, texts in enumerate(versions):
        stats = train_bootstrap_version(
            bootstrap_lm,
            bootstrap_tokenizer,
            texts,
            bootstrap_optimizer,
            steps=args.bootstrap_steps_per_version,
            batch_size=args.bootstrap_batch_size,
            max_length=args.bootstrap_max_length,
            max_grad_norm=cfg.grpo.max_grad_norm,
        )
        h_full = extract_concept_embedding_matrix(
            bootstrap_lm, bootstrap_tokenizer, vocab.concept_texts
        )
        h = project_concept_matrix(
            h_full, pca_mean, pca_components, normalize=True
        )
        snapshots.append(h)
        bootstrap_training.append(stats)
        print(
            f"bootstrap version {b + 1}/{len(versions)} "
            f"loss={stats['final_loss']:.6f} H={list(h.shape)}",
            flush=True,
        )

    H = np.stack(snapshots, axis=0).astype(np.float32)
    bundle_path = args.output / "bootstrap_bundle.npz"
    save_bootstrap_bundle(
        bundle_path,
        H,
        vocab.concept_ids,
        model_id=args.bootstrap_model,
        adapter_id=f"embedding-lora:{embed_target}",
        version_ids=tuple(f"version-{i + 1:02d}" for i in range(len(H))),
        metadata={
            "sentence_order_perturbation": True,
            "continuous_lora_trajectory": True,
            "representation_projection": "fixed_pretraining_PCA",
            "representation_dim": int(args.bootstrap_representation_dim),
            "bootstrap_docs": int(len(frame)),
            "steps_per_version": int(args.bootstrap_steps_per_version),
        },
    )
    print(f"Saved fixed bootstrap bundle {list(H.shape)} -> {bundle_path}", flush=True)

    # Bootstrap evidence is now permanently frozen. Free this model before loading
    # the independent downstream task adapter/model.
    del bootstrap_optimizer, bootstrap_lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage B: Bootstrap-MNGM initial graph.
    # ------------------------------------------------------------------
    estimator = MNGMEstimator(
        H, BOOTSTRAP_MATRIX_MODE, cfg.mngm, cfg.solver
    )
    env = GraphEnvironment(config=cfg.environment, estimator=estimator)
    lam0 = cfg.runner.initial_lambda if args.initial_lambda is None else float(args.initial_lambda)
    print(
        f"Initializing Bootstrap-MNGM from shape={list(H.shape)} lambda={lam0}",
        flush=True,
    )
    state0 = env.initialize_from_estimator(
        lam0, vocab.concept_ids, state_id="bootstrap-state-0"
    )
    print(f"Initial bootstrap graph: {graph_metrics(state0.snapshot)}", flush=True)

    # ------------------------------------------------------------------
    # Stage C: independent task model + retrieval Graph-RL.
    # ------------------------------------------------------------------
    print("Loading independent retrieval task model...", flush=True)
    task_lm, tokenizer = load_local_causal_lm(
        args.task_model,
        dtype=args.task_dtype,
        device_map=args.device_map,
        local_files_only=args.task_local_files_only,
    )
    task_lm = attach_task_lora(
        task_lm,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
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
        concept_ids=vocab.concept_ids,
        tokenizer=tokenizer,
        batch_size=args.task_batch_size,
        max_length=args.task_max_length,
        score_temperature=args.score_temperature,
    )
    train_ctx = load_retrieval_context(
        pool_path=args.train_pools,
        max_queries=args.max_train_queries,
        **common,
    )
    graph_ctx = load_retrieval_context(
        pool_path=args.graph_pools,
        max_queries=args.max_reward_queries,
        **common,
    )
    val_ctx = load_retrieval_context(
        pool_path=args.val_pools,
        max_queries=args.max_val_queries,
        **common,
    )

    evaluator = FrozenGraphRetrievalEvaluator(
        model, candidate_batch_size=args.candidate_batch_size
    )
    for name, ctx in (("train", train_ctx), ("graph", graph_ctx), ("val", val_ctx)):
        print(f"candidate-index {name}: {evaluator.prepare(ctx)}", flush=True)

    task_trainable = [p for p in model.parameters() if p.requires_grad]
    if not task_trainable:
        raise SystemExit("task model exposed no trainable LoRA/tokenizer parameters")
    optimizer = torch.optim.AdamW(task_trainable, lr=args.task_lr)

    warmup = None
    if args.warmup_steps > 0:
        warmup = adapt_retrieval_model(
            model,
            evaluator,
            state0.snapshot,
            train_ctx,
            optimizer,
            steps=args.warmup_steps,
            max_grad_norm=cfg.grpo.max_grad_norm,
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
            model,
            evaluator,
            accepted_state.snapshot,
            train_ctx,
            optimizer,
            steps=args.adapt_steps,
            max_grad_norm=cfg.grpo.max_grad_norm,
        )

    runner = GraphPhaseRunner(
        env,
        policy,
        evaluator,
        reward_fn,
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
        phases=args.phases,
        policy_updates_per_phase=args.policy_updates_per_phase,
        num_candidates=args.num_candidates,
        on_phase=report,
    )

    _save_graph(args.output / "initial_graph.npz", state0)
    _save_graph(args.output / "final_graph.npz", result.final_state)
    policy.save(args.output / "graph_policy.pt")
    torch.save(graph_tokenizer.state_dict(), args.output / "soft_graph_tokenizer.pt")
    if hasattr(task_lm, "save_pretrained"):
        task_lm.save_pretrained(args.output / "task_lora")

    final_graph = evaluator.evaluate_snapshot(result.final_state.snapshot, graph_ctx)
    final_val = evaluator.evaluate_snapshot(result.final_state.snapshot, val_ctx)
    summary = {
        "graph_estimator": "bootstrap_concept_embedding_matrix",
        "bootstrap_bundle": str(bundle_path),
        "bootstrap_shape": list(H.shape),
        "bootstrap_model": args.bootstrap_model,
        "bootstrap_adapter_target": embed_target,
        "bootstrap_training": bootstrap_training,
        "bootstrap_evidence_frozen_before_graph_rl": True,
        "bootstrap_adapter_separate_from_task_adapter": True,
        "initial_lambda": lam0,
        "task": "fixed-pool examiner-citation retrieval",
        "task_model": args.task_model,
        "candidate_index_frozen": True,
        "pool_size": graph_ctx.pool_size,
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
