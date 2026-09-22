from dataclasses import replace
import pytest
from graph_mvp.downstream import EvaluationResult
from graph_mvp.environment import graph_metrics
from graph_mvp.reward import RewardFunction, RewardRecord, Acceptor
from graph_mvp.config import RewardConfig
from graph_mvp.types import ActionRecord


def candidate(env, state):
    return env.step(state, ActionRecord("c", state.state_id, (0, 1), "increase", 0., "test"))


def evaluation(c, delta=.1):
    return EvaluationResult(c.candidate_id, .6 - delta, -.6 + delta, .6, -.6, delta)


def test_reward_components_and_statistical_reference(env, state):
    c = candidate(env, state)
    fn = RewardFunction(RewardConfig(alpha=.2, beta=.3))
    r = fn(c, evaluation(c), state)
    current = graph_metrics(state.snapshot)
    expected_stat = max(0, c.graph_metrics["stat_objective"] - current["stat_objective"]) / 3
    assert r.stat_penalty == pytest.approx(expected_stat)
    assert r.reward == pytest.approx(.1 - .2 * r.density_penalty - .3 * expected_stat)
    assert fn(c, evaluation(c, .2), state).reward > r.reward


def test_acceptance_never_accepts_relative_best_negative(env, state):
    c = candidate(env, state)
    r = RewardRecord(c.candidate_id, .01, 0., 0., -.05, True)
    assert not Acceptor().select(state, [c], [r]).accepted
    assert not Acceptor().select(state, [c], [replace(r, reward=.1, delta_task=0.)]).accepted
    assert Acceptor().select(state, [c], [replace(r, reward=.1)]).accepted
    assert not Acceptor().select(state, [c], [replace(r, reward=float("nan"))]).accepted


def test_invalid_and_identity_checks(env, state):
    c = candidate(env, state)
    invalid = replace(c, valid=False, error="failed")
    r = RewardFunction()(invalid, None, state)
    assert not r.valid and r.reward < 0
    assert not Acceptor().select(state, [invalid], [replace(r, valid=True, reward=1., delta_task=1.)]).accepted
    with pytest.raises(ValueError):
        RewardFunction()(c, replace(evaluation(c), candidate_id="wrong"), state)
    with pytest.raises(ValueError):
        Acceptor().select(state, [c], [])
    with pytest.raises(ValueError):
        Acceptor().select(state, [c], [r, r])

