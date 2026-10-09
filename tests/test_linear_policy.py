from dataclasses import replace
import numpy as np
import pytest
from graph_mvp.config import GRPOConfig
from graph_mvp.linear_policy import LinearCandidateBuilder, LinearGRPOPolicy
from graph_mvp.types import LinearPenaltyAction, GroupPolicyExperience
from graph_mvp.runner import GraphPhaseRunner
from graph_mvp.downstream import DownstreamEvaluator
from graph_mvp.reward import RewardFunction


def test_global_linear_mapping_bounds_symmetry_and_single_resolve(env, state):
    inp = LinearCandidateBuilder(np.eye(3)).build(state, task_relevance=np.ones((3, 3)))
    phi = inp.linear_features
    assert phi.shape == (3, 8)
    assert np.all(np.abs(phi).sum(axis=1) <= 1 + 1e-7)
    coefficients = (-1.,) * 8
    action = LinearPenaltyAction("linear-1", state.state_id, coefficients,
        np.clip(phi.astype(float) @ np.asarray(coefficients), -1., 1.),
        3, -8 * np.log(3), "linear-test")
    old = state.snapshot.Lambda.copy()
    before = env.solver.solve_calls
    result = env.step(state, action)
    assert result.valid
    assert env.solver.solve_calls == before + 1
    upper = np.triu_indices(3, 1)
    np.testing.assert_allclose(result.snapshot.Lambda[upper],
        old[upper] * np.exp(env.config.eta * action.delta))
    np.testing.assert_array_equal(state.snapshot.Lambda, old)
    np.testing.assert_array_equal(result.snapshot.Lambda, result.snapshot.Lambda.T)
    assert np.all(np.diag(result.snapshot.Lambda) == 0)
    assert result.graph_metrics["num_changed_penalties"] == 3
    with pytest.raises(ValueError, match="stale"):
        env.step(state, replace(action, state_id="wrong"))
    with pytest.raises(ValueError, match="axis"):
        env.step(state, replace(action, num_concepts=2, delta=np.zeros(1)))


def test_global_keep_and_clipping(env, state):
    zero = LinearPenaltyAction("zero", state.state_id, (0.,) * 8, np.zeros(3), 3, 0., "test")
    calls = env.solver.solve_calls
    candidate = env.step(state, zero)
    assert candidate.snapshot is state.snapshot
    assert env.solver.solve_calls == calls
    assert candidate.graph_metrics["num_changed_penalties"] == 0
    env.config = replace(env.config, eta=100.)
    inc = replace(zero, delta=np.ones(3))
    lam = env._edited_lambda(state, inc)
    np.testing.assert_array_equal(lam[np.triu_indices(3, 1)], env.config.lambda_max)


def test_linear_grpo_probability_counts_coefficients_and_learns(state):
    torch = pytest.importorskip("torch")
    inp = LinearCandidateBuilder(np.eye(3)).build(state)
    cfg = GRPOConfig(hidden_dim=8, update_epochs=2, entropy_coef=0.)
    p, other = LinearGRPOPolicy(cfg, seed=17), LinearGRPOPolicy(cfg, seed=17)
    actions, repeated = p.sample(inp, 8), other.sample(inp, 8)
    assert [a.coefficients for a in actions] == [a.coefficients for a in repeated]
    for a in actions:
        np.testing.assert_allclose(a.delta, inp.linear_features.astype(float) @ a.coefficients)
        assert a.log_prob == pytest.approx(-8 * np.log(3), abs=1e-6)
        item = p._cache[a.candidate_id]
        lp, kl, ent = p._trajectory_stats(item["features"], (), item["directions"])
        assert float(lp.detach()) == pytest.approx(a.log_prob, abs=1e-6)
        assert float(kl.detach()) == pytest.approx(0., abs=1e-6)
        assert float(ent.detach()) == pytest.approx(8 * np.log(3), abs=1e-6)
    before = {k: v.clone() for k, v in p._model.state_dict().items()}
    batch = [GroupPolicyExperience(state.state_id, a.candidate_id, a.log_prob,
        float(i), i != 0, len(a.delta)) for i, a in enumerate(actions)]
    update = p.update(batch)
    assert update["updated"] and update["num_valid"] == 7
    assert np.isfinite(update["loss"])
    assert any(not torch.equal(v, p._model.state_dict()[k]) for k, v in before.items())
    assert not p._cache and not p._edge_reward_ema


def test_linear_grpo_flat_rewards_do_not_update(state):
    torch = pytest.importorskip("torch")
    p = LinearGRPOPolicy(GRPOConfig(hidden_dim=8), seed=4)
    actions = p.sample(LinearCandidateBuilder().build(state), 4)
    before = {k: v.clone() for k, v in p._model.state_dict().items()}
    result = p.update([GroupPolicyExperience(state.state_id, a.candidate_id,
        a.log_prob, 0., True, len(a.delta)) for a in actions])
    assert not result["updated"]
    assert all(torch.equal(v, p._model.state_dict()[k]) for k, v in before.items())


def test_linear_full_reward_validation_pipeline(env, state, context):
    pytest.importorskip("torch")
    policy = LinearGRPOPolicy(GRPOConfig(hidden_dim=8, update_epochs=1), seed=11)
    result = GraphPhaseRunner(env, policy, DownstreamEvaluator(), RewardFunction(),
        candidate_builder=LinearCandidateBuilder(np.eye(3))).run(
            state, context, context, phases=1, policy_updates_per_phase=2, num_candidates=4)
    assert len(result.phases[0].updates) == 2
    assert all(c.valid for u in result.phases[0].updates for c in u.candidates)
    assert result.phases[0].updates[0].selected_edge_features[0]["num_pairs"] == 3
    assert not result.phases[0].adapted
