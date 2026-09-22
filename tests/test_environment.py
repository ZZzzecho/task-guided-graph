from dataclasses import replace
import numpy as np
import pytest
from graph_mvp.types import ActionRecord, GraphSnapshot
from graph_mvp.config import SolverConfig, EnvironmentConfig
from graph_mvp.environment import GraphEnvironment, promote_candidate_to_state
from graph_mvp.weighted_glasso import WeightedGraphicalLasso


def action(state, name="decrease", edge=(0, 1), id="candidate"):
    return ActionRecord(id, state.state_id, edge, name, 0., "test")


def test_candidates_are_independent_and_parent_is_immutable(env, state):
    before = {k: getattr(state.snapshot, k).copy() for k in ("S", "Lambda", "Theta", "Rho", "A")}
    a = env.step(state, action(state))
    b = env.step(state, action(state, "increase", (1, 2), "second"))
    assert a.valid and b.valid
    assert env.solver.solve_calls == 3  # initial + two changed Lambda solves
    for k, value in before.items():
        np.testing.assert_array_equal(getattr(state.snapshot, k), value)
        with pytest.raises(ValueError):
            getattr(a.snapshot, k)[0, 0] = 3
        with pytest.raises(ValueError):
            getattr(state.snapshot, k).setflags(write=True)
    assert b.snapshot.Lambda[0, 1] == state.snapshot.Lambda[0, 1]
    assert a.snapshot.Lambda[1, 2] == state.snapshot.Lambda[1, 2]
    np.testing.assert_array_equal(a.snapshot.S, b.snapshot.S)
    np.testing.assert_allclose(a.snapshot.Lambda[0, 1], .3 * np.exp(-env.config.eta))
    assert np.count_nonzero(a.snapshot.Lambda != state.snapshot.Lambda) == 2


def test_derived_graph_and_promotion(env, state):
    c = env.step(state, action(state))
    d = np.sqrt(np.diag(c.snapshot.Theta))
    rho = -c.snapshot.Theta / np.outer(d, d)
    np.fill_diagonal(rho, 0)
    np.testing.assert_allclose(c.snapshot.Rho, rho)
    np.testing.assert_array_equal(c.snapshot.A, np.where(abs(rho) > env.config.adjacency_threshold, rho, 0))
    promoted = promote_candidate_to_state(state, c)
    assert promoted is not state and promoted.snapshot is c.snapshot
    assert state.iteration == 0 and promoted.iteration == 1
    with pytest.raises(ValueError):
        promote_candidate_to_state(promoted, c)


def test_keep_reuses_exact_snapshot(env, state):
    calls = env.solver.solve_calls
    c = env.step(state, action(state, "keep"))
    assert c.snapshot is state.snapshot and c.valid
    assert env.solver.solve_calls == calls


def test_bounds_and_numerical_failure(state):
    env = GraphEnvironment(config=EnvironmentConfig(lambda_min=.3, lambda_max=.3))
    c = env.step(state, action(state))
    assert c.snapshot is state.snapshot
    failed = GraphEnvironment(WeightedGraphicalLasso(SolverConfig(max_iter=1)))
    c = failed.step(state, action(state))
    assert not c.valid and c.snapshot is None and c.error
    with pytest.raises(ValueError):
        promote_candidate_to_state(state, c)


def test_axis_and_action_validation(env, state):
    with pytest.raises(ValueError):
        env.step(state, replace(action(state), state_id="stale"))
    with pytest.raises(ValueError):
        action(state, edge=(0, 0))
    with pytest.raises(ValueError):
        env.step(state, action(state, edge=(0, 8)))
    with pytest.raises(ValueError):
        GraphSnapshot(("a", "a", "c"), state.snapshot.S, state.snapshot.Lambda, state.snapshot.Theta)


def test_metadata_immutable(env, state):
    with pytest.raises(TypeError):
        state.solver_info["converged"] = False

