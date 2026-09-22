from graph_mvp.downstream import TaskMetrics, EvaluationResult
from graph_mvp.environment import CandidateBuilder
from graph_mvp.policy import GreedyPolicy
from graph_mvp.reward import RewardFunction
from graph_mvp.runner import GraphPhaseRunner


class AlwaysBetterEvaluator:
    def evaluate_snapshot(self, snapshot, context):
        return TaskMetrics(1.0, -1.0)

    def evaluate(self, candidate, context, baseline):
        return EvaluationResult(candidate.candidate_id, .9, -.9,
                                baseline.task_loss, baseline.task_metric, .1)


class RecordingGreedy(GreedyPolicy):
    def __init__(self):
        super().__init__()
        self.states = []

    def sample(self, policy_input, num_candidates):
        self.states.append(policy_input.state_id)
        return super().sample(policy_input, num_candidates)


def test_graph_phase_keeps_state_fixed_then_validates_once(env, state):
    policy = RecordingGreedy()
    adapted = []
    runner = GraphPhaseRunner(env, policy, AlwaysBetterEvaluator(), RewardFunction(),
                              CandidateBuilder(), min_validation_improvement=0.01,
                              task_adapter=lambda accepted_state: adapted.append(accepted_state.state_id))
    result = runner.run(state, object(), object(), phases=1,
                        policy_updates_per_phase=2, num_candidates=2)
    assert policy.states == [state.state_id, state.state_id]
    phase = result.phases[0]
    assert phase.accepted
    assert result.final_state.state_id != state.state_id
    assert phase.adapted and adapted == [result.final_state.state_id]
