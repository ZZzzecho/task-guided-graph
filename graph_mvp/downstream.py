"""Paired, deterministic graph-gated logistic classification on fixed splits."""
from dataclasses import dataclass, field
from typing import Mapping, Protocol
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score
from .config import DownstreamConfig
from .types import GraphSnapshot, GraphCandidate, readonly, freeze


@dataclass(frozen=True, eq=False)
class TaskContext:
    concept_ids: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_reward: np.ndarray
    y_reward: np.ndarray
    base_model_config: Mapping = field(default_factory=dict)
    random_seed: int = 0

    def __post_init__(self):
        ids = tuple(self.concept_ids)
        if len(ids) < 2 or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("concept_ids must be unique nonempty strings")
        for split in ("train", "reward"):
            x, y = np.asarray(getattr(self, f"X_{split}")), np.asarray(getattr(self, f"y_{split}"))
            if x.ndim != 2 or x.shape[1] != len(ids) or x.shape[0] < 2 or not np.isfinite(x).all():
                raise ValueError(f"Invalid {split} feature matrix")
            if y.shape != (len(x),) or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
                raise ValueError("MVP supports binary labels {0, 1}")
            if split == "train" and len(np.unique(y)) != 2:
                raise ValueError("Training split must contain both classes")
            object.__setattr__(self, f"X_{split}", readonly(x))
            object.__setattr__(self, f"y_{split}", readonly(y, np.int64))
        DownstreamConfig(**self.base_model_config)
        object.__setattr__(self, "concept_ids", ids)
        object.__setattr__(self, "base_model_config", freeze(self.base_model_config))


@dataclass(frozen=True)
class TaskMetrics:
    task_loss: float
    task_metric: float
    extra_metrics: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "extra_metrics", freeze(self.extra_metrics))


@dataclass(frozen=True)
class EvaluationResult:
    candidate_id: str
    task_loss: float
    task_metric: float
    baseline_loss: float
    baseline_metric: float
    delta_task: float
    extra_metrics: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "extra_metrics", freeze(self.extra_metrics))


class Evaluator(Protocol):
    def evaluate_snapshot(self, snapshot: GraphSnapshot, task_context: TaskContext) -> TaskMetrics: ...
    def evaluate(self, candidate: GraphCandidate, task_context: TaskContext,
                 baseline: TaskMetrics) -> EvaluationResult: ...


class EvaluationError(RuntimeError):
    pass


class DownstreamEvaluator:
    """Utility = negative reward-set log loss. A controls available interactions.

    Every graph uses p+p(p-1)/2 fixed feature slots; absent edges have zero slots.
    Only training data determines scaling and fitted coefficients. Each fit starts
    from scratch with the identical solver, seed, tolerance and maximum budget.
    """
    @staticmethod
    def features(X, A, mean, scale):
        x = (np.asarray(X) - mean) / scale
        i, j = np.triu_indices(x.shape[1], 1)
        interaction = x[:, i] * x[:, j] * (A[i, j] != 0)
        return np.column_stack((x, interaction))

    def _fit_score(self, snapshot, context, X_score, y_score):
        if snapshot.concept_ids != context.concept_ids:
            raise ValueError("Graph/data concept axis mismatch")
        config = DownstreamConfig(**context.base_model_config)
        mean = context.X_train.mean(axis=0)
        scale = context.X_train.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        train = self.features(context.X_train, snapshot.A, mean, scale)
        score = self.features(X_score, snapshot.A, mean, scale)
        model = LogisticRegression(C=config.C, solver="lbfgs", max_iter=config.max_iter,
                                   tol=config.tol, random_state=context.random_seed)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train, context.y_train)
        except ConvergenceWarning as exc:
            raise EvaluationError("Downstream model did not converge within configured budget") from exc
        prob = model.predict_proba(score)[:, 1]
        if not np.isfinite(prob).all():
            raise EvaluationError("Nonfinite downstream predictions")
        loss = float(log_loss(y_score, prob, labels=[0, 1]))
        pred = (prob >= 0.5).astype(int)
        auc = float(roc_auc_score(y_score, prob)) if len(np.unique(y_score)) == 2 else None
        return TaskMetrics(loss, -loss, {"auroc": auc,
                           "f1": float(f1_score(y_score, pred, zero_division=0)),
                           "accuracy": float(accuracy_score(y_score, pred)),
                           "fit_iterations": int(model.n_iter_[0]),
                           "num_feature_slots": train.shape[1]})

    def evaluate_snapshot(self, snapshot, task_context):
        return self._fit_score(snapshot, task_context, task_context.X_reward, task_context.y_reward)

    def evaluate(self, candidate, task_context, baseline):
        if not candidate.valid or candidate.snapshot is None:
            raise ValueError("Cannot evaluate an invalid candidate")
        metrics = self.evaluate_snapshot(candidate.snapshot, task_context)
        return EvaluationResult(candidate.candidate_id, metrics.task_loss, metrics.task_metric,
                                baseline.task_loss, baseline.task_metric,
                                metrics.task_metric - baseline.task_metric, metrics.extra_metrics)

    def evaluate_holdout(self, snapshot, task_context, X_test, y_test):
        """Explicit final-only entry point; runner/policy never receive these arrays."""
        x, y = np.asarray(X_test), np.asarray(y_test)
        if x.ndim != 2 or x.shape[1] != len(task_context.concept_ids) or not np.isfinite(x).all():
            raise ValueError("Invalid holdout features")
        if len(x) < 2 or y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
            raise ValueError("Invalid holdout labels")
        return self._fit_score(snapshot, task_context, x, y)

