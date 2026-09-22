from dataclasses import asdict, replace, fields, is_dataclass
from collections.abc import Mapping
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import scipy
import sklearn
from . import __version__
from .config import Config
from .data import synthetic_dataset, load_dataset, RankGaussianTransformer, training_covariance
from .qsar import qsar_dataset
from .downstream import TaskContext, DownstreamEvaluator
from .environment import GraphEnvironment, CandidateBuilder, graph_metrics
from .weighted_glasso import WeightedGraphicalLasso
from .estimators import VectorGGMEstimator, MNGMEstimator, VECTOR_MODE, MATRIX_MODES
from .graph_data import load_graph_samples
from .policy import RandomPolicy, GreedyPolicy, GRPOPolicy
from .reward import RewardFunction, Acceptor
from .runner import TrainingRunner, select_global_baseline, select_global_baseline_estimator


def jsonable(value):
    if is_dataclass(value):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def save_state(path, state):
    s = state.snapshot
    payload = {"concept_ids": np.asarray(s.concept_ids), "S": s.S,
               "Lambda": s.Lambda, "Theta": s.Theta, "Rho": s.Rho, "A": s.A,
               "adjacency_threshold": s.adjacency_threshold, "state_id": state.state_id,
               "iteration": state.iteration, "estimator_kind": np.asarray(s.estimator_kind)}
    for key in ("representation_precision", "representation_covariance"):
        if key in s.auxiliary:
            payload[key] = np.asarray(s.auxiliary[key])
    for key in ("sample_semantics", "representation_dim", "n_samples", "mngm_transform",
                "data_fingerprint", "stat_objective", "penalized_objective"):
        if key in s.auxiliary and isinstance(s.auxiliary[key], (str, int, float, np.integer, np.floating)):
            payload[key] = np.asarray(s.auxiliary[key])
    np.savez_compressed(path, **payload)


def round_log(record):
    candidates = []
    for c, e, r in zip(record.candidates, record.evaluations, record.rewards):
        if c.snapshot is None:
            new_penalties = None
        elif hasattr(c.action, "actions"):
            new_penalties = [{"edge": a.edge, "penalty": float(c.snapshot.Lambda[a.edge])}
                             for a in c.action.actions]
        else:
            new_penalties = [{"edge": c.action.edge, "penalty": float(c.snapshot.Lambda[c.action.edge])}]
        candidates.append({"candidate_id": c.candidate_id, "parent_state_id": c.parent_state_id,
                           "action": c.action, "valid": c.valid, "error": c.error,
                           "solver_info": c.solver_info, "graph_metrics": c.graph_metrics,
                           "evaluation": e, "reward": r,
                           "new_edge_penalties": new_penalties})
    return {"round": record.round, "parent_state_id": record.parent_state_id,
            "next_state_id": record.next_state_id, "candidates": candidates,
            "acceptance": record.acceptance, "policy_update": record.policy_update,
            "solver_calls": record.solver_calls}


def fingerprint(context):
    digest = hashlib.sha256()
    digest.update(json.dumps(context.concept_ids).encode())
    for a in (context.X_train, context.y_train, context.X_reward, context.y_reward):
        digest.update(str((a.shape, a.dtype)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Task-guided statistical graph learning MVP")
    parser.add_argument("--config", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, help="Pre-split NPZ; omit for synthetic smoke experiment")
    source.add_argument("--qsar-csv", type=Path, help="Official UCI biodeg.csv; split deterministically in-process")
    parser.add_argument("--graph-data", type=Path, help="Optional graph-estimation NPZ: graph_mode, concept_ids, graph_samples")
    parser.add_argument("--policy", choices=("random", "greedy", "grpo"))
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--output", type=Path, required=True, help="New output directory (no overwriting)")
    parser.add_argument("--evaluate-test", action="store_true", help="Final-only holdout evaluation")
    args = parser.parse_args(argv)
    config = Config.load(args.config) if args.config else Config()
    changes = {k: v for k, v in vars(args).items() if k in ("policy", "rounds", "num_candidates") and v is not None}
    config = replace(config, runner=replace(config.runner, **changes))
    if args.qsar_csv:
        dataset = qsar_dataset(args.qsar_csv, seed=config.data.seed)
    else:
        dataset = load_dataset(args.data) if args.data else synthetic_dataset(config.data)
    if args.evaluate_test and dataset.X_test is None:
        parser.error("--evaluate-test requires X_test/y_test in the dataset")
    if config.data.nonparanormal:
        transformer = RankGaussianTransformer().fit(dataset.X_train)
        x_train, x_reward = transformer.transform(dataset.X_train), transformer.transform(dataset.X_reward)
    else:
        transformer = None
        x_train, x_reward = dataset.X_train, dataset.X_reward
    context = TaskContext(dataset.concept_ids, x_train, dataset.y_train, x_reward, dataset.y_reward,
                          asdict(config.downstream), config.runner.seed)
    graph_samples = load_graph_samples(args.graph_data) if args.graph_data else None
    if graph_samples is not None and graph_samples.concept_ids != context.concept_ids:
        parser.error("--graph-data concept_ids must exactly match downstream task concept_ids")
    if graph_samples is None:
        # Backward-compatible v0.1.1 path: downstream training matrix is also graph data.
        S = training_covariance(context.X_train, config.data.covariance_ridge)
        solver = WeightedGraphicalLasso(config.solver)
        estimator = None
        env = GraphEnvironment(solver, config.environment)
        graph_mode = VECTOR_MODE
    else:
        S = None
        if graph_samples.mode == VECTOR_MODE:
            estimator = VectorGGMEstimator(graph_samples.samples, config.data, config.solver)
        elif graph_samples.mode in MATRIX_MODES:
            estimator = MNGMEstimator(graph_samples.samples, graph_samples.mode, config.mngm, config.solver)
        else:
            raise ValueError(f"Unsupported graph mode: {graph_samples.mode}")
        solver = estimator
        env = GraphEnvironment(config=config.environment, estimator=estimator)
        graph_mode = graph_samples.mode
    evaluator = DownstreamEvaluator()
    # Validate and reserve output directory before running any expensive search.
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", config)
    if transformer is not None:
        np.savez_compressed(args.output / "preprocessing.npz", sorted_train=transformer.sorted_)
    grid = config.runner.scalar_grid or (config.runner.initial_lambda,)
    if estimator is None:
        initial, trials = select_global_baseline(env, evaluator, S, context.concept_ids, context, grid)
    else:
        initial, trials = select_global_baseline_estimator(env, evaluator, context.concept_ids, context, grid)
    write_json(args.output / "global_baseline.json", trials)
    save_state(args.output / "initial_state.npz", initial)
    if config.runner.policy == "random":
        policy = RandomPolicy(config.runner.seed)
    elif config.runner.policy == "greedy":
        policy = GreedyPolicy()
    else:
        policy = GRPOPolicy(config.grpo, config.runner.seed)
    runner = TrainingRunner(env, policy, evaluator, RewardFunction(config.reward), Acceptor(config.acceptance),
                            CandidateBuilder(config.candidate))
    with (args.output / "trajectory.jsonl").open("w", encoding="utf-8") as log:
        def on_round(record):
            log.write(json.dumps(jsonable(round_log(record)), ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            best = max((r.reward for r in record.rewards if r.valid), default=None)
            print(f"round={record.round + 1} accepted={record.acceptance.accepted} best_reward={best}", flush=True)
        result = runner.run(initial, context, config.runner.rounds, config.runner.num_candidates, on_round)
    save_state(args.output / "final_state.npz", result.final_state)
    policy_checkpoint = None
    if hasattr(policy, "save"):
        ckpt = args.output / "policy.pt"
        if policy.save(ckpt):
            policy_checkpoint = ckpt.name
    if args.qsar_csv:
        data_source = f"UCI QSAR biodegradation: {args.qsar_csv.resolve()}"
    elif args.data:
        data_source = str(args.data.resolve())
    else:
        data_source = "synthetic smoke fixture"
    summary = {"data_source": data_source,
               "graph_data_source": None if args.graph_data is None else str(args.graph_data.resolve()),
               "graph_mode": graph_mode,
               "search_data_sha256": fingerprint(context), "policy": config.runner.policy,
               "policy_checkpoint": policy_checkpoint,
               "initial_state_id": initial.state_id, "final_state_id": result.final_state.state_id,
               "rounds": len(result.rounds), "accepted_rounds": sum(r.acceptance.accepted for r in result.rounds),
               "initial_reward_metrics": result.initial_metrics, "final_reward_metrics": result.final_metrics,
               "initial_graph_metrics": graph_metrics(initial.snapshot),
               "final_graph_metrics": graph_metrics(result.final_state.snapshot),
               "solver_calls": solver.solve_calls,
               "versions": {"package": __version__, "python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "sklearn": sklearn.__version__},
               "test_evaluated": args.evaluate_test}
    if args.evaluate_test:
        # Only after search has ended; never forwarded to runner or policy.
        x_test = transformer.transform(dataset.X_test) if transformer is not None else dataset.X_test
        summary["initial_test_metrics"] = evaluator.evaluate_holdout(initial.snapshot, context, x_test, dataset.y_test)
        summary["final_test_metrics"] = evaluator.evaluate_holdout(result.final_state.snapshot, context, x_test, dataset.y_test)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

