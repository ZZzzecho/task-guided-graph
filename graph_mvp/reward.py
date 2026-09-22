"""Reward calculation and acceptance are deliberately independent."""
from dataclasses import dataclass
from typing import Mapping
import math
from .config import RewardConfig, AcceptanceConfig
from .environment import graph_metrics


@dataclass(frozen=True)
class RewardRecord:
    candidate_id: str
    delta_task: float
    density_penalty: float
    stat_penalty: float
    reward: float
    valid: bool
    error: str | None = None


class RewardFunction:
    def __init__(self, config=RewardConfig()):
        self.config = config

    def invalid(self, candidate_id, reason):
        return RewardRecord(candidate_id, 0., 0., 0., self.config.invalid_reward, False, reason)

    def __call__(self, candidate, evaluation, current_state_metrics):
        if not candidate.valid:
            return self.invalid(candidate.candidate_id, candidate.error or "Invalid graph")
        if evaluation is None:
            return self.invalid(candidate.candidate_id, "Missing downstream evaluation")
        if evaluation.candidate_id != candidate.candidate_id:
            raise ValueError("Evaluation/candidate identity mismatch")
        values = (evaluation.task_loss, evaluation.task_metric, evaluation.baseline_loss,
                  evaluation.baseline_metric, evaluation.delta_task)
        if not all(math.isfinite(x) for x in values):
            return self.invalid(candidate.candidate_id, "Nonfinite downstream evaluation")
        delta = evaluation.task_metric - evaluation.baseline_metric
        if not math.isclose(delta, evaluation.delta_task, abs_tol=1e-10):
            raise ValueError("Inconsistent task utility delta")
        current = current_state_metrics if isinstance(current_state_metrics, Mapping) else graph_metrics(current_state_metrics.snapshot)
        metrics = graph_metrics(candidate.snapshot)
        density_delta = metrics["density"] - current["density"]
        density = max(0., density_delta) if self.config.density_mode == "increase_only" else density_delta
        # Comparable across Lambda choices: use unpenalized NLL, normalized per concept.
        stat = max(0., metrics["stat_objective"] - current["stat_objective"]) / len(candidate.snapshot.concept_ids)
        reward = delta - self.config.alpha * density - self.config.beta * stat
        if not math.isfinite(reward):
            return self.invalid(candidate.candidate_id, "Nonfinite reward")
        return RewardRecord(candidate.candidate_id, delta, density, stat, reward, True)


@dataclass(frozen=True)
class AcceptanceResult:
    accepted: bool
    candidate_id: str | None
    reason: str


class Acceptor:
    def __init__(self, config=AcceptanceConfig()):
        self.config = config

    def select(self, current_state, candidates, rewards):
        by_id = {r.candidate_id: r for r in rewards}
        ids = [c.candidate_id for c in candidates]
        if len(by_id) != len(rewards) or len(set(ids)) != len(ids) or set(ids) != set(by_id):
            raise ValueError("Candidates and rewards must have a one-to-one ID mapping")
        eligible = []
        for candidate in candidates:
            if candidate.parent_state_id != current_state.state_id:
                raise ValueError("Cannot accept a candidate from another state")
            r = by_id[candidate.candidate_id]
            if (candidate.valid and r.valid and math.isfinite(r.reward) and math.isfinite(r.delta_task)
                    and r.reward > self.config.reward_threshold
                    and r.delta_task > self.config.min_task_improvement):
                eligible.append(r)
        if not eligible:
            return AcceptanceResult(False, None, "No valid candidate passes absolute reward and task-improvement thresholds")
        # Stable tie break: order of candidates. Relative group ranking alone never suffices.
        best = max(eligible, key=lambda r: r.reward)
        return AcceptanceResult(True, best.candidate_id, "Best reward among candidates passing independent acceptance gates")

