from dataclasses import replace
import numpy as np
from graph_mvp.policy import GreedyPolicy, RandomPolicy
from graph_mvp.environment import CandidateBuilder, GraphEnvironment
from graph_mvp.downstream import DownstreamEvaluator, TaskMetrics, EvaluationResult, EvaluationError
from graph_mvp.reward import RewardFunction, Acceptor
from graph_mvp.runner import TrainingRunner
from graph_mvp.config import SolverConfig, AcceptanceConfig
from graph_mvp.weighted_glasso import WeightedGraphicalLasso


def test_policy_contract_and_greedy_coverage(state):
    inp = CandidateBuilder().build(state)
    greedy = GreedyPolicy()
    records = greedy.sample(inp, 4) + greedy.sample(inp, 3)
    assert len({(a.edge, a.action) for a in records}) == 7
    assert {a.edge for a in records} == {(0, 1), (0, 2), (1, 2)}
    random1, random2 = RandomPolicy(9), RandomPolicy(9)
    assert random1.sample(inp, 10) == random2.sample(inp, 10)
    assert all(np.isclose(a.log_prob, -np.log(9)) for a in random1.sample(inp, 10))


class RecordingPolicy(GreedyPolicy):
    def update(self, batch):
        self.last_batch = batch
        return super().update(batch)


def test_end_to_end_reproducibility_and_no_acceptance_identity(env, state, context):
    def run():
        return TrainingRunner(env, RandomPolicy(5), DownstreamEvaluator(), RewardFunction(),
                              Acceptor(AcceptanceConfig(reward_threshold=100.))).run(state, context, 2, 5)
    a, b = run(), run()
    assert a.final_state is state and b.final_state is state
    assert [r.rewards for r in a.rounds] == [r.rewards for r in b.rounds]


def test_failed_solves_are_logged_masked_and_do_not_corrupt_state(state, context):
    env = GraphEnvironment(WeightedGraphicalLasso(SolverConfig(max_iter=1)))
    policy = RecordingPolicy()
    result = TrainingRunner(env, policy, DownstreamEvaluator(), RewardFunction(), Acceptor()).run(state, context, 1, 2)
    assert result.final_state is state
    assert all(not e.valid_mask for e in policy.last_batch)
    assert len(result.rounds[0].candidates) == 2
    assert all(c.error for c in result.rounds[0].candidates)


def test_negative_rewards_still_reach_policy(env, state, context):
    class WorseEvaluator:
        def evaluate_snapshot(self, snapshot, context):
            return TaskMetrics(.5, -.5)
        def evaluate(self, candidate, context, baseline):
            return EvaluationResult(candidate.candidate_id, .6, -.6, .5, -.5, -.1)
    policy = RecordingPolicy()
    result = TrainingRunner(env, policy, WorseEvaluator(), RewardFunction(), Acceptor()).run(state, context, 1, 3)
    assert result.final_state is state
    assert all(e.valid_mask and e.reward < 0 for e in policy.last_batch)


def test_successful_acceptance_uses_new_baseline(env, state, context):
    class BetterEvaluator:
        def evaluate_snapshot(self, snapshot, context):
            return TaskMetrics(.8, -.8)
        def evaluate(self, candidate, context, baseline):
            loss = baseline.task_loss - .1
            return EvaluationResult(candidate.candidate_id, loss, -loss,
                                    baseline.task_loss, baseline.task_metric, .1)
    result = TrainingRunner(env, GreedyPolicy(), BetterEvaluator(), RewardFunction(), Acceptor()).run(state, context, 2, 2)
    assert all(r.acceptance.accepted for r in result.rounds)
    assert result.final_state.iteration == 2
    assert result.rounds[1].evaluations[0].baseline_loss == result.rounds[0].evaluations[0].task_loss
    assert state.iteration == 0


def test_evaluator_failure_not_accepted(env, state, context):
    class FailingEvaluator(DownstreamEvaluator):
        def evaluate(self, *args):
            raise EvaluationError("budget exhausted")
    result = TrainingRunner(env, GreedyPolicy(), FailingEvaluator(), RewardFunction(), Acceptor()).run(state, context, 1, 2)
    assert result.final_state is state
    assert all(not r.valid and r.error == "budget exhausted" for r in result.rounds[0].rewards)


def test_actual_penalty_solve_graph_task_reward_acceptance_chain(env, context):
    # A controlled interaction fixture, using real solver and real classifier.
    s = np.array([[1., .4, 0.], [.4, 1., 0.], [0., 0., 1.]])
    initial = env.initialize(s, 1., context.concept_ids)
    result = TrainingRunner(env, GreedyPolicy(), DownstreamEvaluator(), RewardFunction(), Acceptor()).run(
        initial, context, rounds=2, num_candidates=7)
    assert result.rounds[0].acceptance.accepted
    assert initial.snapshot.A[0, 1] == 0
    assert result.final_state.snapshot.A[0, 1] != 0
    assert result.final_state.snapshot.Lambda[0, 1] < initial.snapshot.Lambda[0, 1]
    assert result.final_metrics.task_loss < result.initial_metrics.task_loss - .1
