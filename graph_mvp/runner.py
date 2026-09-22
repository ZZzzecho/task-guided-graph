from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .types import (GraphState, GraphCandidate, PolicyExperience,
                    GroupPolicyExperience, ActionGroup)
from .environment import CandidateBuilder, graph_metrics, promote_candidate_to_state
from .downstream import TaskMetrics, EvaluationError
from .reward import RewardRecord, AcceptanceResult
from .policy import ACTIONS


@dataclass(frozen=True)
class RoundRecord:
    round: int
    parent_state_id: str
    next_state_id: str
    candidates: tuple[GraphCandidate, ...]
    evaluations: tuple
    rewards: tuple[RewardRecord, ...]
    acceptance: AcceptanceResult
    policy_update: dict
    solver_calls: int


@dataclass(frozen=True)
class RunResult:
    initial_state: GraphState
    final_state: GraphState
    initial_metrics: TaskMetrics
    final_metrics: TaskMetrics
    rounds: tuple[RoundRecord, ...]


class TrainingRunner:
    """Backward-compatible per-round runner.

    It now understands both single ActionRecord proposals and batch ActionGroup
    proposals.  New Bootstrap-MNGM experiments should prefer GraphPhaseRunner
    below because it keeps the accepted state fixed during a whole GRPO search
    block and performs a separate validation acceptance check.
    """
    def __init__(self, graph_env, policy, evaluator, reward_fn, acceptor, candidate_builder=None):
        self.graph_env = graph_env
        self.policy = policy
        self.evaluator = evaluator
        self.reward_fn = reward_fn
        self.acceptor = acceptor
        self.builder = candidate_builder or CandidateBuilder()

    @staticmethod
    def _experience(state_id, candidate, action, reward, features):
        if isinstance(action, ActionGroup):
            return GroupPolicyExperience(state_id, candidate.candidate_id, action.log_prob,
                                         reward.reward, reward.valid, len(action.actions))
        return PolicyExperience(state_id, candidate.candidate_id,
                                features[action.edge], ACTIONS.index(action.action),
                                action.log_prob, reward.reward, reward.valid)

    def run(self, initial_state, task_context, rounds=8, num_candidates=12,
            on_round: Callable | None = None):
        if rounds < 0 or num_candidates < 1:
            raise ValueError("Invalid run budget")
        if initial_state.snapshot.concept_ids != task_context.concept_ids:
            raise ValueError("Initial graph/data concept axis mismatch")
        state = initial_state
        current = self.evaluator.evaluate_snapshot(state.snapshot, task_context)
        initial_metrics = current
        records = []
        for iteration in range(rounds):
            calls = self.graph_env.solver.solve_calls
            inp = self.builder.build(state)
            actions = self.policy.sample(inp, num_candidates=num_candidates)
            ids = [a.candidate_id for a in actions]
            if len(set(ids)) != len(ids):
                raise ValueError("Policy produced duplicate candidate IDs")
            candidates = [self.graph_env.step(state, action) for action in actions]
            evaluations, rewards, experiences = [], [], []
            features = {f.edge: f.vector() for f in inp.candidate_edges}
            for candidate, action in zip(candidates, actions):
                evaluation = None
                if not candidate.valid:
                    reward = self.reward_fn.invalid(candidate.candidate_id, candidate.error)
                else:
                    try:
                        evaluation = self.evaluator.evaluate(candidate, task_context, current)
                        reward = self.reward_fn(candidate, evaluation, state)
                    except (EvaluationError, np.linalg.LinAlgError, FloatingPointError) as exc:
                        reward = self.reward_fn.invalid(candidate.candidate_id, str(exc))
                evaluations.append(evaluation)
                rewards.append(reward)
                experiences.append(self._experience(state.state_id, candidate, action, reward, features))
            update = self.policy.update(experiences)
            acceptance = self.acceptor.select(state, candidates, rewards)
            parent = state
            if acceptance.accepted:
                index = ids.index(acceptance.candidate_id)
                state = promote_candidate_to_state(state, candidates[index])
                evaluation = evaluations[index]
                current = TaskMetrics(evaluation.task_loss, evaluation.task_metric, evaluation.extra_metrics)
            record = RoundRecord(iteration, parent.state_id, state.state_id, tuple(candidates),
                                 tuple(evaluations), tuple(rewards), acceptance, update,
                                 self.graph_env.solver.solve_calls - calls)
            records.append(record)
            if on_round:
                on_round(record)
        return RunResult(initial_state, state, initial_metrics, current, tuple(records))


@dataclass(frozen=True)
class PhaseUpdateRecord:
    update: int
    candidates: tuple[GraphCandidate, ...]
    evaluations: tuple
    rewards: tuple[RewardRecord, ...]
    policy_update: dict
    solver_calls: int


@dataclass(frozen=True)
class GraphPhaseRecord:
    phase: int
    parent_state_id: str
    next_state_id: str
    updates: tuple[PhaseUpdateRecord, ...]
    proposal_id: str | None
    validation_baseline: TaskMetrics
    validation_proposal: object | None
    accepted: bool
    acceptance_reason: str
    adapted: bool


@dataclass(frozen=True)
class GraphPhaseRunResult:
    initial_state: GraphState
    final_state: GraphState
    phases: tuple[GraphPhaseRecord, ...]


class GraphPhaseRunner:
    """Block-wise graph search matching BOOTSTRAP_MNGM_GRAPH_RL_HANDOFF v0.2.

    Within one Graph Phase the accepted graph state is immutable.  Multiple GRPO
    candidate groups are sampled/evaluated around that same Lambda/Theta state.
    The best reward-panel proposal is then checked on an independent validation
    context.  Only a validation-accepted graph is promoted.  An optional
    `task_adapter` callback is invoked only after acceptance, allowing the caller
    to update task LoRA + SoftGraphTokenizer before they are frozen again.

    Evaluators are expected to be frozen during each Graph Phase.  This runner does
    not itself train or clone an LLM; that boundary is explicit by design.
    """
    def __init__(self, graph_env, policy, reward_evaluator, reward_fn,
                 candidate_builder=None, validation_evaluator=None,
                 min_validation_improvement=0.0, task_adapter=None,
                 feature_provider=None):
        if min_validation_improvement < 0:
            raise ValueError("min_validation_improvement must be nonnegative")
        self.graph_env = graph_env
        self.policy = policy
        self.reward_evaluator = reward_evaluator
        self.validation_evaluator = validation_evaluator or reward_evaluator
        self.reward_fn = reward_fn
        self.builder = candidate_builder or CandidateBuilder()
        self.min_validation_improvement = float(min_validation_improvement)
        self.task_adapter = task_adapter
        self.feature_provider = feature_provider

    def _build_input(self, state, reward_context, baseline):
        signals = {}
        if self.feature_provider is not None:
            supplied = self.feature_provider(state, reward_context, baseline)
            if supplied:
                signals = dict(supplied)
        allowed = {"task_relevance", "uncertainty", "previous_reward"}
        if set(signals) - allowed:
            raise ValueError(f"Unknown candidate feature signals: {set(signals) - allowed}")
        return self.builder.build(state, **signals)

    def run(self, initial_state, reward_context, validation_context,
            phases=4, policy_updates_per_phase=4, num_candidates=8,
            on_phase: Callable | None = None):
        if phases < 0 or policy_updates_per_phase < 1 or num_candidates < 1:
            raise ValueError("Invalid graph-phase budget")
        state = initial_state
        records = []
        for phase in range(phases):
            parent = state
            reward_baseline = self.reward_evaluator.evaluate_snapshot(state.snapshot, reward_context)
            validation_baseline = self.validation_evaluator.evaluate_snapshot(state.snapshot, validation_context)
            update_records = []
            best = None  # (reward, candidate)
            for update_index in range(policy_updates_per_phase):
                calls = self.graph_env.solver.solve_calls
                inp = self._build_input(state, reward_context, reward_baseline)
                actions = self.policy.sample(inp, num_candidates)
                ids = [a.candidate_id for a in actions]
                if len(ids) != len(set(ids)):
                    raise ValueError("Policy produced duplicate candidate IDs")
                candidates = [self.graph_env.step(state, a) for a in actions]
                evaluations, rewards, experiences = [], [], []
                features = {f.edge: f.vector() for f in inp.candidate_edges}
                for candidate, action in zip(candidates, actions):
                    evaluation = None
                    if not candidate.valid:
                        reward = self.reward_fn.invalid(candidate.candidate_id, candidate.error)
                    else:
                        try:
                            evaluation = self.reward_evaluator.evaluate(candidate, reward_context, reward_baseline)
                            reward = self.reward_fn(candidate, evaluation, state)
                        except (EvaluationError, np.linalg.LinAlgError, FloatingPointError) as exc:
                            reward = self.reward_fn.invalid(candidate.candidate_id, str(exc))
                    evaluations.append(evaluation)
                    rewards.append(reward)
                    experiences.append(TrainingRunner._experience(
                        state.state_id, candidate, action, reward, features))
                    if candidate.valid and reward.valid and (best is None or reward.reward > best[0]):
                        best = (reward.reward, candidate)
                policy_update = self.policy.update(experiences)
                update_records.append(PhaseUpdateRecord(
                    update_index, tuple(candidates), tuple(evaluations), tuple(rewards),
                    policy_update, self.graph_env.solver.solve_calls - calls))

            proposal = None if best is None else best[1]
            validation_proposal = None
            accepted = False
            adapted = False
            if proposal is None:
                reason = "No valid reward-panel proposal"
            else:
                try:
                    validation_proposal = self.validation_evaluator.evaluate(
                        proposal, validation_context, validation_baseline)
                    improvement = float(validation_proposal.delta_task)
                    accepted = improvement > self.min_validation_improvement
                    reason = (f"validation task utility improved by {improvement:.6g}"
                              if accepted else
                              f"validation task utility improvement {improvement:.6g} did not pass threshold")
                except (EvaluationError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    reason = f"Validation evaluation failed: {exc}"
            if accepted:
                state = promote_candidate_to_state(state, proposal)
                if self.task_adapter is not None:
                    self.task_adapter(state)
                    adapted = True
            record = GraphPhaseRecord(
                phase, parent.state_id, state.state_id, tuple(update_records),
                None if proposal is None else proposal.candidate_id,
                validation_baseline, validation_proposal, accepted, reason, adapted)
            records.append(record)
            if on_phase:
                on_phase(record)
        return GraphPhaseRunResult(initial_state, state, tuple(records))


def select_global_baseline(env, evaluator, S, concept_ids, task_context, scalar_grid):
    """Choose scalar lambda using reward split only; no test data in this interface."""
    trials, eligible = [], []
    for index, penalty in enumerate(scalar_grid):
        try:
            state = env.initialize(S, penalty, concept_ids, state_id=f"global-{index}")
            metrics = evaluator.evaluate_snapshot(state.snapshot, task_context)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            trials.append({"lambda": penalty, "valid": False, "error": str(exc)})
            continue
        trials.append({"lambda": penalty, "valid": True, "task_loss": metrics.task_loss,
                       **graph_metrics(state.snapshot)})
        eligible.append((metrics.task_metric, state))
    if not eligible:
        raise RuntimeError(f"All scalar baseline trials failed: {trials}")
    return max(eligible, key=lambda item: item[0])[1], trials


def select_global_baseline_estimator(env, evaluator, concept_ids, task_context, scalar_grid):
    """Choose scalar concept-axis penalty for a bound estimator using reward data only."""
    trials, eligible = [], []
    for index, penalty in enumerate(scalar_grid):
        try:
            state = env.initialize_from_estimator(penalty, concept_ids, state_id=f"global-{index}")
            metrics = evaluator.evaluate_snapshot(state.snapshot, task_context)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            trials.append({"lambda": penalty, "valid": False, "error": str(exc)})
            continue
        trials.append({"lambda": penalty, "valid": True, "task_loss": metrics.task_loss,
                       **graph_metrics(state.snapshot)})
        eligible.append((metrics.task_metric, state))
    if not eligible:
        raise RuntimeError(f"All scalar baseline trials failed: {trials}")
    return max(eligible, key=lambda item: item[0])[1], trials
