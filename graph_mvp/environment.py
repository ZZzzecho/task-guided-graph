from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .config import EnvironmentConfig, CandidateConfig
from .types import (GraphSnapshot, GraphState, GraphCandidate, ActionRecord,
                    ActionGroup, EdgeFeatures, PolicyInput)
from .weighted_glasso import WeightedGraphicalLasso, penalty_matrix, statistical_loss, objective


def graph_metrics(snapshot):
    p = len(snapshot.concept_ids)
    edges = int(np.count_nonzero(np.triu(snapshot.A, 1)))
    aux = snapshot.auxiliary
    stat = aux.get("stat_objective") if "stat_objective" in aux else statistical_loss(snapshot.S, snapshot.Theta)
    pen = aux.get("penalized_objective") if "penalized_objective" in aux else objective(snapshot.S, snapshot.Lambda, snapshot.Theta)
    metrics = {
        "num_edges": edges,
        "density": edges / (p * (p - 1) / 2),
        "stat_objective": float(stat),
        "penalized_objective": float(pen),
        "min_eigenvalue": float(np.linalg.eigvalsh(snapshot.Theta)[0]),
        "estimator_kind": snapshot.estimator_kind,
    }
    if "representation_precision" in aux:
        metrics["representation_min_eigenvalue"] = float(np.linalg.eigvalsh(aux["representation_precision"])[0])
        metrics["representation_dim"] = int(aux.get("representation_dim", len(aux["representation_precision"])))
    return metrics


def concept_axis_kkt_frontier(snapshot: GraphSnapshot, zero_tolerance=1e-10):
    """Return solver-specific active mask and zero-edge frontier scores.

    The bundled weighted graphical-lasso objective is

        -logdet(Theta) + tr(S Theta) + sum_{i<j} Lambda_ij |Theta_ij|.

    Because the symmetric full-matrix gradient counts an undirected off-diagonal
    twice, a zero edge satisfies |(S - Theta^{-1})_ij| <= Lambda_ij / 2.
    For a zero edge we therefore define a boundary score

        score_ij = min(1, 2 |g_ij| / Lambda_ij),

    so values near one are closest to entering the active set if its penalty is
    decreased. This is derived from the actual solver objective rather than copied
    from an unrelated sklearn graphical-lasso convention.
    """
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be nonnegative")
    theta = np.asarray(snapshot.Theta)
    lam = np.asarray(snapshot.Lambda)
    grad = np.asarray(snapshot.S) - np.linalg.inv(theta)
    active = np.abs(theta) > zero_tolerance
    np.fill_diagonal(active, False)
    denom = np.maximum(lam / 2.0, np.finfo(float).tiny)
    score = np.minimum(1.0, np.abs(grad) / denom)
    score[active] = 0.0
    np.fill_diagonal(score, 0.0)
    return active, score, grad


class GraphEnvironment:
    """Task-guided graph environment with full re-solve after penalty edits.

    A single ActionRecord remains supported for legacy coordinate baselines.  The
    Bootstrap-MNGM MVP uses ActionGroup: all selected penalties are changed first,
    then the complete weighted GGM/MNGM problem is re-solved exactly once.
    """
    def __init__(self, solver=None, config=EnvironmentConfig(), estimator=None):
        if solver is not None and estimator is not None:
            raise ValueError("Provide either legacy solver or bound estimator, not both")
        self.estimator = estimator
        self.solver = estimator if estimator is not None else (solver if solver is not None else WeightedGraphicalLasso())
        self.config = config

    def initialize(self, S, penalty, concept_ids, state_id="state-0", progress_callback=None):
        if self.estimator is not None:
            raise ValueError("Bound-estimator environment must use initialize_from_estimator")
        lam = penalty_matrix(len(concept_ids), penalty)
        self._check_bounds(lam)
        result = self.solver.solve(S, lam, progress_callback=progress_callback)
        if not result.converged:
            raise RuntimeError(f"Initial graph solve failed: {result.message}")
        snapshot = GraphSnapshot(tuple(concept_ids), S, lam, result.Theta,
                                 self.config.adjacency_threshold,
                                 "patient_activation_vector",
                                 {"stat_objective": statistical_loss(S, result.Theta),
                                  "penalized_objective": objective(S, lam, result.Theta)})
        if graph_metrics(snapshot)["density"] > self.config.max_density:
            raise ValueError("Initial graph exceeds max_density")
        return GraphState(state_id, 0, snapshot, result.info())

    def initialize_from_estimator(self, penalty, concept_ids, state_id="state-0", progress_callback=None):
        if self.estimator is None:
            raise ValueError("No bound graph estimator")
        lam = penalty_matrix(len(concept_ids), penalty)
        self._check_bounds(lam)
        result = self.estimator.solve(lam, progress_callback=progress_callback)
        if not result.converged:
            raise RuntimeError(f"Initial graph solve failed: {result.message}")
        snapshot = self._snapshot_from_estimator_result(tuple(concept_ids), lam, result)
        if graph_metrics(snapshot)["density"] > self.config.max_density:
            raise ValueError("Initial graph exceeds max_density")
        return GraphState(state_id, 0, snapshot, result.info())

    def _snapshot_from_estimator_result(self, concept_ids, lam, result):
        aux = dict(result.auxiliary)
        if result.stat_objective is not None:
            aux["stat_objective"] = result.stat_objective
        if result.penalized_objective is not None:
            aux["penalized_objective"] = result.penalized_objective
        return GraphSnapshot(concept_ids, result.S, lam, result.Theta,
                             self.config.adjacency_threshold,
                             result.estimator_kind, aux)

    def _check_bounds(self, lam):
        off = lam[np.triu_indices(len(lam), 1)]
        if np.any(off < self.config.lambda_min) or np.any(off > self.config.lambda_max):
            raise ValueError("Lambda outside configured bounds")

    @staticmethod
    def _records(action):
        if isinstance(action, ActionGroup):
            return action.actions
        if isinstance(action, ActionRecord):
            return (action,)
        raise TypeError("action must be ActionRecord or ActionGroup")

    def _edited_lambda(self, state: GraphState, action: ActionRecord | ActionGroup):
        if action.state_id != state.state_id:
            raise ValueError("Action references a stale or different state")
        lam = state.snapshot.Lambda.copy()
        p = len(lam)
        for record in self._records(action):
            i, j = record.edge
            if j >= p:
                raise ValueError("edge outside concept axis")
            direction = {"increase": 1, "keep": 0, "decrease": -1}[record.action]
            if direction:
                value = np.exp(np.clip(np.log(lam[i, j]) + direction * self.config.eta,
                                       np.log(self.config.lambda_min), np.log(self.config.lambda_max)))
                lam[i, j] = lam[j, i] = np.clip(value, self.config.lambda_min, self.config.lambda_max)
        return lam

    def step(self, state: GraphState, action: ActionRecord | ActionGroup, progress_callback=None):
        parent = state.snapshot
        if parent.adjacency_threshold != self.config.adjacency_threshold:
            raise ValueError("Adjacency threshold cannot change during a search")
        self._check_bounds(parent.Lambda)
        lam = self._edited_lambda(state, action)
        if np.array_equal(lam, parent.Lambda):
            info = dict(state.solver_info, reused_unchanged=True)
            snapshot = parent
        else:
            try:
                if self.estimator is None:
                    result = self.solver.solve(
                        parent.S, lam, initial_theta=parent.Theta,
                        progress_callback=progress_callback
                    )
                    info = result.info()
                    if not result.converged:
                        return GraphCandidate(action.candidate_id, state.state_id, None, action,
                                              info, {}, False, result.message)
                    snapshot = GraphSnapshot(parent.concept_ids, parent.S, lam, result.Theta,
                                             self.config.adjacency_threshold,
                                             parent.estimator_kind,
                                             {"stat_objective": statistical_loss(parent.S, result.Theta),
                                              "penalized_objective": objective(parent.S, lam, result.Theta)})
                else:
                    if parent.estimator_kind != self.estimator.kind:
                        raise ValueError("State estimator kind does not match bound estimator")
                    repr_precision = parent.auxiliary.get("representation_precision")
                    result = self.estimator.solve(
                        lam,
                        initial_theta=parent.Theta,
                        initial_representation_precision=repr_precision,
                        progress_callback=progress_callback,
                    )
                    info = result.info()
                    if not result.converged:
                        return GraphCandidate(action.candidate_id, state.state_id, None, action,
                                              info, {}, False, result.message)
                    snapshot = self._snapshot_from_estimator_result(parent.concept_ids, lam, result)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
                return GraphCandidate(action.candidate_id, state.state_id, None, action,
                                      {"converged": False}, {}, False, str(exc))
        current, metrics = graph_metrics(parent), graph_metrics(snapshot)
        metrics["delta_edges"] = metrics["num_edges"] - current["num_edges"]
        metrics["density_delta"] = metrics["density"] - current["density"]
        metrics["stat_objective_delta"] = metrics["stat_objective"] - current["stat_objective"]
        metrics["num_direct_edits"] = len(self._records(action))
        metrics["lambda_l1_change"] = float(np.sum(np.triu(np.abs(lam - parent.Lambda), 1)))
        valid = metrics["density"] <= self.config.max_density
        return GraphCandidate(action.candidate_id, state.state_id, snapshot, action, info, metrics,
                              valid, None if valid else "Candidate exceeds max_density")


class CandidateBuilder:
    """Build active + KKT-frontier concept edges from the current solved state."""
    def __init__(self, config=CandidateConfig()):
        self.config = config

    @staticmethod
    def _signal_value(signal, edge, p):
        if signal is None:
            return None
        if isinstance(signal, Mapping):
            value = signal.get(edge, signal.get((edge[1], edge[0]), None))
            return None if value is None else float(value)
        a = np.asarray(signal, dtype=float)
        if a.shape != (p, p):
            raise ValueError("edge signal matrices must have shape [P, P]")
        return float(a[edge])

    def build(self, state, task_relevance=None, uncertainty=None, previous_reward=None):
        g = state.snapshot
        p = len(g.concept_ids)
        active_mask, frontier, _ = concept_axis_kkt_frontier(g, self.config.zero_tolerance)
        degrees = np.sum(active_mask, axis=1).astype(float)
        active_edges = [(i, j) for i in range(p) for j in range(i + 1, p) if active_mask[i, j]]
        zero_edges = [(i, j) for i in range(p) for j in range(i + 1, p) if not active_mask[i, j]]
        zero_edges.sort(key=lambda e: (-float(frontier[e]), e[0], e[1]))
        high = [e for e in zero_edges if frontier[e] >= self.config.frontier_min_score]
        floor = zero_edges[:min(self.config.min_frontier_edges, len(zero_edges))]
        selected_frontier = []
        seen = set()
        if self.config.max_frontier_edges > 0:
            for e in high + floor:
                if e not in seen:
                    selected_frontier.append(e)
                    seen.add(e)
                if len(selected_frontier) >= self.config.max_frontier_edges:
                    break
            # If the high-score set is larger than the floor, retain as many as budget allows.
            if len(selected_frontier) < self.config.max_frontier_edges:
                for e in zero_edges:
                    if frontier[e] < self.config.frontier_min_score:
                        break
                    if e not in seen:
                        selected_frontier.append(e)
                        seen.add(e)
                    if len(selected_frontier) >= self.config.max_frontier_edges:
                        break

        pool = active_edges + selected_frontier
        features = []
        for i, j in pool:
            edge = (i, j)
            is_active = bool(active_mask[edge])
            features.append(EdgeFeatures(
                edge=edge,
                partial_corr=float(g.Rho[edge]),
                abs_partial_corr=abs(float(g.Rho[edge])),
                penalty=float(g.Lambda[edge]),
                edge_exists=is_active,
                uncertainty=self._signal_value(uncertainty, edge, p),
                task_relevance=self._signal_value(task_relevance, edge, p),
                previous_reward=self._signal_value(previous_reward, edge, p),
                frontier_score=0.0 if is_active else float(frontier[edge]),
                degree_i=float(degrees[i]),
                degree_j=float(degrees[j]),
            ))
        return PolicyInput(state.state_id, tuple(features), {
            "representation_dim": float(g.auxiliary.get("representation_dim", 1)),
            "num_active_edges": float(len(active_edges)),
            "num_frontier_edges": float(len(selected_frontier)),
            "candidate_pool_size": float(len(features)),
        })


def promote_candidate_to_state(state, candidate):
    if not candidate.valid or candidate.snapshot is None or candidate.parent_state_id != state.state_id:
        raise ValueError("Cannot promote invalid or stale candidate")
    if candidate.snapshot.concept_ids != state.snapshot.concept_ids:
        raise ValueError("Candidate changed concept axis")
    if candidate.snapshot.estimator_kind != state.snapshot.estimator_kind:
        raise ValueError("Candidate changed graph estimator")
    old_fp = state.snapshot.auxiliary.get("data_fingerprint")
    new_fp = candidate.snapshot.auxiliary.get("data_fingerprint")
    if old_fp is not None or new_fp is not None:
        if old_fp != new_fp:
            raise ValueError("Candidate changed statistical sample set")
    elif not np.array_equal(candidate.snapshot.S, state.snapshot.S):
        raise ValueError("Candidate changed the statistical environment")
    return GraphState(f"state-{candidate.candidate_id}", state.iteration + 1, candidate.snapshot,
                      candidate.solver_info, {"parent_state_id": state.state_id,
                                              "accepted_candidate_id": candidate.candidate_id})
