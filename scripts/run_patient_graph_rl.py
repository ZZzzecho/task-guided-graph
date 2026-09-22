#!/usr/bin/env python
"""End-to-end Patient-MNGM + task-guided graph RL entrypoint.

Requires:
- prepared split directory with train/graph/val/test CSV.GZ files;
- frozen Patient-MNGM cache built only from train.csv.gz;
- concept vocabulary used by that cache;
- a local Transformers causal LM (default GLM-4.7-Flash).

The patient statistical encoder is never reloaded here: H_patient is fixed on disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from graph_mvp.config import Config
from graph_mvp.environment import GraphEnvironment, CandidateBuilder, graph_metrics
from graph_mvp.estimators import MNGMEstimator, PATIENT_MATRIX_MODE
from graph_mvp.llm_task import (
    CausalTaskContext, ErrorWeightedTaskRelevanceProvider,
    FrozenGraphCausalEvaluator, GraphConditionedCausalLM, adapt_task_model,
)
from graph_mvp.patient_repr import PatientMatrixDataset, load_concept_vocabulary
from graph_mvp.policy import GRPOPolicy
from graph_mvp.reward import RewardFunction
from graph_mvp.runner import GraphPhaseRunner
from graph_mvp.task_model import attach_task_lora, load_local_causal_lm, GLM47_FLASH_MODEL_ID
from graph_mvp.task_prototypes import build_task_soft_graph_tokenizer


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Install pandas: pip install -e '.[all]'") from exc
    return pd


def _context(frame, concept_ids, tokenizer, batch_size, max_length, task):
    return CausalTaskContext(
        concept_ids=tuple(concept_ids),
        texts=tuple(frame["text"].astype(str).tolist()),
        labels=tuple(frame["label"].astype(int).tolist()),
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        instruction=task["instruction"],
        label_texts=tuple(task["label_texts"]),
        answer_prefix=task["answer_prefix"],
    )


def _limit(frame, n):
    return frame if n is None else frame.iloc[:n].copy()


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
    p = argparse.ArgumentParser(description="Patient-MNGM + GLM task-guided Graph RL")
    p.add_argument("--patient-cache", type=Path, required=True)
    p.add_argument("--concepts", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=False, help="Prepared split directory (train/graph/val/test.csv.gz)")
    p.add_argument("--mimic-dir", type=Path, required=False, help="Deprecated alias for --data-dir")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-mngm-patients", type=int, default=None)
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-reward-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--task-model", default=GLM47_FLASH_MODEL_ID)
    p.add_argument("--task-local-files-only", action="store_true")
    p.add_argument("--task-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--task-lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--adapt-steps", type=int, default=200)
    p.add_argument("--task-batch-size", type=int, default=2)
    p.add_argument("--task-max-length", type=int, default=512)
    p.add_argument("--graph-tokens", type=int, default=8)
    p.add_argument("--graph-hidden-dim", type=int, default=128)
    p.add_argument("--activation-tau", type=float, default=0.1)
    p.add_argument("--policy-updates-per-phase", type=int, default=4)
    args = p.parse_args()

    if args.task_lr <= 0 or args.warmup_steps < 0 or args.adapt_steps < 1:
        raise SystemExit("Invalid task training hyperparameters")
    data_dir = args.data_dir or args.mimic_dir
    if data_dir is None:
        raise SystemExit("Provide --data-dir (preferred) or legacy --mimic-dir")
    task_path = data_dir / "task.json"
    if task_path.exists():
        task = json.loads(task_path.read_text(encoding="utf-8"))
    else:
        task = {
            "instruction": (
                "Predict the emergency-department disposition using only the triage information. "
                "Answer exactly HOME or ADMITTED.\n\n"
            ),
            "label_texts": ["HOME", "ADMITTED"],
            "answer_prefix": "\n\nDisposition:",
        }
    cfg = Config.load(args.config)
    vocab = load_concept_vocabulary(args.concepts)
    cache = PatientMatrixDataset(args.patient_cache)
    if tuple(cache.concept_ids) != tuple(vocab.concept_ids):
        raise SystemExit("Patient cache concept axis does not match --concepts vocabulary")

    # Current MNGM implementation is in-memory. The sharded cache keeps representation
    # generation out-of-core; this boundary is intentionally explicit for scale tests.
    patient_matrices = cache.materialize(args.max_mngm_patients, dtype=np.float32)
    estimator = MNGMEstimator(patient_matrices, PATIENT_MATRIX_MODE, cfg.mngm, cfg.solver)
    env = GraphEnvironment(config=cfg.environment, estimator=estimator)
    state0 = env.initialize_from_estimator(
        cfg.runner.initial_lambda, cache.concept_ids, state_id="patient-state-0")

    task_lm, task_tokenizer = load_local_causal_lm(
        args.task_model,
        dtype=args.task_dtype,
        device_map=args.device_map,
        local_files_only=args.task_local_files_only,
    )
    task_lm = attach_task_lora(
        task_lm, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    graph_tokenizer = build_task_soft_graph_tokenizer(
        task_lm, task_tokenizer, vocab.concept_texts,
        num_tokens=args.graph_tokens,
        graph_hidden_dim=args.graph_hidden_dim,
        activation_tau=args.activation_tau,
        activation_normalization="sigmoid",
    )
    model = GraphConditionedCausalLM(task_lm, graph_tokenizer)

    pd = _pd()
    train_df = _limit(pd.read_csv(data_dir / "train.csv.gz"), args.max_train_examples)
    graph_df = _limit(pd.read_csv(data_dir / "graph.csv.gz"), args.max_reward_examples)
    val_df = _limit(pd.read_csv(data_dir / "val.csv.gz"), args.max_val_examples)
    train_ctx = _context(train_df, cache.concept_ids, task_tokenizer,
                         args.task_batch_size, args.task_max_length, task)
    graph_ctx = _context(graph_df, cache.concept_ids, task_tokenizer,
                         args.task_batch_size, args.task_max_length, task)
    val_ctx = _context(val_df, cache.concept_ids, task_tokenizer,
                       args.task_batch_size, args.task_max_length, task)

    import torch
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit("No trainable task LoRA / graph-tokenizer parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.task_lr)

    warmup = None
    if args.warmup_steps > 0:
        warmup = adapt_task_model(
            model, state0.snapshot, train_ctx, optimizer,
            steps=args.warmup_steps, max_grad_norm=cfg.grpo.max_grad_norm)

    evaluator = FrozenGraphCausalEvaluator(model)
    policy = GRPOPolicy(cfg.grpo, seed=cfg.runner.seed)
    reward_fn = RewardFunction(cfg.reward)
    builder = CandidateBuilder(cfg.candidate)
    feature_provider = ErrorWeightedTaskRelevanceProvider(evaluator)

    def task_adapter(accepted_state):
        return adapt_task_model(
            model, accepted_state.snapshot, train_ctx, optimizer,
            steps=args.adapt_steps, max_grad_norm=cfg.grpo.max_grad_norm)

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
            f"state={record.parent_state_id}->{record.next_state_id} reason={record.acceptance_reason}",
            flush=True,
        )

    result = runner.run(
        state0,
        graph_ctx,
        val_ctx,
        phases=cfg.runner.rounds,
        policy_updates_per_phase=args.policy_updates_per_phase,
        num_candidates=cfg.runner.num_candidates,
        on_phase=report,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    _save_graph(args.output / "initial_graph.npz", state0)
    _save_graph(args.output / "final_graph.npz", result.final_state)
    policy.save(args.output / "graph_policy.pt")
    torch.save(graph_tokenizer.state_dict(), args.output / "soft_graph_tokenizer.pt")
    if hasattr(task_lm, "save_pretrained"):
        task_lm.save_pretrained(args.output / "task_lora")

    final_val = evaluator.evaluate_snapshot(result.final_state.snapshot, val_ctx)
    summary = {
        "patient_cache": str(args.patient_cache),\n        "data_dir": str(data_dir),
        "patient_cache_fingerprint": cache.fingerprint(),
        "mngm_shape": list(patient_matrices.shape),
        "task_model": args.task_model,
        "task_definition": task,
        "task_activation_prototype_space": "task_lm_input_embeddings",
        "patient_mngm_prototype_space": "Qwen3-Embedding pooled",
        "warmup": warmup,
        "phases": len(result.phases),
        "accepted_phases": int(sum(r.accepted for r in result.phases)),
        "initial_graph_metrics": graph_metrics(state0.snapshot),
        "final_graph_metrics": graph_metrics(result.final_state.snapshot),
        "final_validation": {
            "task_loss": final_val.task_loss,
            "task_metric": final_val.task_metric,
            **dict(final_val.extra_metrics),
        },
        "test_split_touched": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
