"""Shared immutable contracts for graph search and block-wise RL."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Any
import numpy as np


def readonly(value, dtype=float):
    a = np.asarray(value, dtype=dtype)
    # A bytes-backed buffer cannot be made writable with setflags(write=True).
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return readonly(value, value.dtype)
    return value


def thaw(value):
    """Recursively convert immutable/internal values into plain Python containers.

    Useful for JSON/log serialization while keeping runtime contracts immutable.
    Arrays are converted to lists and NumPy scalars to native Python scalars.
    """
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def symmetric_matrix(value, name, p=None):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError(f"{name} must be square with at least two concepts")
    if p is not None and a.shape != (p, p):
        raise ValueError(f"{name} shape mismatch")
    if not np.isfinite(a).all() or not np.allclose(a, a.T, atol=1e-10, rtol=0):
        raise ValueError(f"{name} must be finite and symmetric")
    return (a + a.T) / 2


@dataclass(frozen=True, eq=False)
class GraphSnapshot:
    concept_ids: tuple[str, ...]
    S: np.ndarray
    Lambda: np.ndarray
    Theta: np.ndarray
    adjacency_threshold: float = 1e-5
    estimator_kind: str = "patient_activation_vector"
    auxiliary: Mapping[str, Any] = field(default_factory=dict)
    Rho: np.ndarray = field(init=False)
    A: np.ndarray = field(init=False)

    def __post_init__(self):
        ids = tuple(self.concept_ids)
        p = len(ids)
        if p < 2 or len(set(ids)) != p or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("concept_ids must contain unique nonempty strings")
        if not 0 <= self.adjacency_threshold < 1:
            raise ValueError("adjacency_threshold must be in [0, 1)")
        if not isinstance(self.estimator_kind, str) or not self.estimator_kind:
            raise ValueError("estimator_kind must be a nonempty string")
        s, lam, theta = (symmetric_matrix(getattr(self, k), k, p)
                         for k in ("S", "Lambda", "Theta"))
        if np.linalg.eigvalsh(s)[0] < -1e-10:
            raise ValueError("S must be positive semidefinite")
        if np.any(np.diag(s) <= 0):
            raise ValueError("S diagonal must be positive")
        if np.any(lam[np.triu_indices(p, 1)] <= 0) or np.any(np.diag(lam) != 0):
            raise ValueError("Lambda must be positive off diagonal and zero on diagonal")
        np.linalg.cholesky(theta)
        d = np.sqrt(np.diag(theta))
        rho = -theta / np.outer(d, d)
        np.fill_diagonal(rho, 0.0)
        a = np.where(np.abs(rho) > self.adjacency_threshold, rho, 0.0)
        object.__setattr__(self, "concept_ids", ids)
        object.__setattr__(self, "auxiliary", freeze(self.auxiliary))
        for k, v in (("S", s), ("Lambda", lam), ("Theta", theta), ("Rho", rho), ("A", a)):
            object.__setattr__(self, k, readonly(v))


@dataclass(frozen=True)
class ActionRecord:
    """One edge-wise penalty edit.

    `candidate_id` identifies the candidate graph. Multiple records may therefore
    share one candidate_id when they belong to the same ActionGroup.
    """
    candidate_id: str
    state_id: str
    edge: tuple[int, int]
    action: str
    log_prob: float
    policy_version: str
    selection_log_prob: float | None = None
    direction_log_prob: float | None = None

    def __post_init__(self):
        edge = tuple(self.edge)
        if len(edge) != 2 or any(not isinstance(x, (int, np.integer)) for x in edge):
            raise ValueError("edge must contain two integer indices")
        if not 0 <= edge[0] < edge[1]:
            raise ValueError("edge must satisfy 0 <= i < j; diagonal actions forbidden")
        if self.action not in ("increase", "keep", "decrease"):
            raise ValueError("unknown penalty action")
        if not np.isfinite(self.log_prob) or self.log_prob > 1e-12:
            raise ValueError("log_prob must be finite and <= 0")
        for name in ("selection_log_prob", "direction_log_prob"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value > 1e-12):
                raise ValueError(f"{name} must be finite and <= 0 when provided")
        object.__setattr__(self, "edge", edge)


@dataclass(frozen=True)
class ActionGroup:
    """Joint candidate trajectory: several edge-penalty edits, one graph reward."""
    candidate_id: str
    state_id: str
    actions: tuple[ActionRecord, ...]
    log_prob: float
    policy_version: str

    def __post_init__(self):
        actions = tuple(self.actions)
        if not actions:
            raise ValueError("ActionGroup must contain at least one edge edit")
        if not np.isfinite(self.log_prob) or self.log_prob > 1e-12:
            raise ValueError("ActionGroup.log_prob must be finite and <= 0")
        edges = []
        for action in actions:
            if action.candidate_id != self.candidate_id or action.state_id != self.state_id:
                raise ValueError("ActionGroup/action identity mismatch")
            if action.policy_version != self.policy_version:
                raise ValueError("ActionGroup mixes policy versions")
            if action.action == "keep":
                raise ValueError("Batch action groups encode keep by not selecting an edge")
            edges.append(action.edge)
        if len(set(edges)) != len(edges):
            raise ValueError("ActionGroup edges must be sampled without replacement")
        object.__setattr__(self, "actions", actions)

    @property
    def edges(self):
        return tuple(action.edge for action in self.actions)


@dataclass(frozen=True, eq=False)
class GraphState:
    state_id: str
    iteration: int
    snapshot: GraphSnapshot
    solver_info: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "solver_info", freeze(self.solver_info))
        object.__setattr__(self, "metadata", freeze(self.metadata))


@dataclass(frozen=True, eq=False)
class GraphCandidate:
    candidate_id: str
    parent_state_id: str
    snapshot: GraphSnapshot | None
    action: ActionRecord | ActionGroup
    solver_info: Mapping[str, Any]
    graph_metrics: Mapping[str, float]
    valid: bool
    error: str | None = None

    def __post_init__(self):
        if self.candidate_id != self.action.candidate_id or self.parent_state_id != self.action.state_id:
            raise ValueError("candidate/action identity mismatch")
        if self.valid and (self.snapshot is None or self.error is not None):
            raise ValueError("valid candidates require a snapshot and no error")
        object.__setattr__(self, "solver_info", freeze(self.solver_info))
        object.__setattr__(self, "graph_metrics", freeze(self.graph_metrics))


@dataclass(frozen=True)
class EdgeFeatures:
    edge: tuple[int, int]
    partial_corr: float
    abs_partial_corr: float
    penalty: float
    edge_exists: bool
    uncertainty: float | None = None
    task_relevance: float | None = None
    previous_reward: float | None = None
    frontier_score: float = 0.0
    degree_i: float = 0.0
    degree_j: float = 0.0

    def vector(self):
        return readonly([
            self.partial_corr,
            self.abs_partial_corr,
            self.penalty,
            float(self.edge_exists),
            self.uncertainty or 0.,
            self.task_relevance or 0.,
            self.previous_reward or 0.,
            self.frontier_score,
            self.degree_i,
            self.degree_j,
        ])


@dataclass(frozen=True)
class PolicyInput:
    state_id: str
    candidate_edges: tuple[EdgeFeatures, ...]
    global_features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "candidate_edges", tuple(self.candidate_edges))
        object.__setattr__(self, "global_features", freeze(self.global_features))


@dataclass(frozen=True, eq=False)
class PolicyExperience:
    """Legacy single-edge experience, retained for baseline compatibility."""
    state_id: str
    candidate_id: str
    edge_features: np.ndarray
    action: int
    old_log_prob: float
    reward: float
    valid_mask: bool

    def __post_init__(self):
        object.__setattr__(self, "edge_features", readonly(self.edge_features))


@dataclass(frozen=True)
class GroupPolicyExperience:
    """One candidate graph / trajectory experience for batch-edge GRPO."""
    state_id: str
    candidate_id: str
    old_log_prob: float
    reward: float
    valid_mask: bool
    num_edits: int

    def __post_init__(self):
        if not np.isfinite(self.old_log_prob) or self.old_log_prob > 1e-12:
            raise ValueError("old_log_prob must be finite and <= 0")
        if not np.isfinite(self.reward):
            raise ValueError("reward must be finite")
        if not isinstance(self.num_edits, int) or self.num_edits < 1:
            raise ValueError("num_edits must be a positive integer")
