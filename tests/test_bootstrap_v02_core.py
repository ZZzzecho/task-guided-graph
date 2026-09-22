import numpy as np
import pytest

from graph_mvp.config import CandidateConfig
from graph_mvp.environment import CandidateBuilder, concept_axis_kkt_frontier
from graph_mvp.types import ActionRecord, ActionGroup
from graph_mvp.graph_tokens import (bootstrap_concept_prototypes, error_weighted_edge_relevance,
                                    SoftGraphTokenizer)


def test_action_group_edits_multiple_penalties_with_one_full_resolve(env, state):
    before = env.solver.solve_calls
    cid = "batch-1"
    records = (
        ActionRecord(cid, state.state_id, (0, 1), "decrease", -0.2, "test"),
        ActionRecord(cid, state.state_id, (1, 2), "increase", -0.3, "test"),
    )
    group = ActionGroup(cid, state.state_id, records, -0.5, "test")
    candidate = env.step(state, group)
    assert candidate.valid
    assert env.solver.solve_calls == before + 1
    assert candidate.graph_metrics["num_direct_edits"] == 2
    assert candidate.snapshot.Lambda[0, 1] < state.snapshot.Lambda[0, 1]
    assert candidate.snapshot.Lambda[1, 2] > state.snapshot.Lambda[1, 2]
    np.testing.assert_array_equal(candidate.snapshot.Lambda[[0, 2], [2, 0]],
                                  state.snapshot.Lambda[[0, 2], [2, 0]])


def test_candidate_builder_is_active_plus_solver_specific_frontier(state):
    active, frontier, grad = concept_axis_kkt_frontier(state.snapshot)
    cfg = CandidateConfig(max_frontier_edges=1, min_frontier_edges=1,
                          frontier_min_score=1.0)
    inp = CandidateBuilder(cfg).build(state)
    active_edges = {(i, j) for i in range(3) for j in range(i + 1, 3) if active[i, j]}
    pool = {f.edge for f in inp.candidate_edges}
    assert active_edges.issubset(pool)
    assert len(pool - active_edges) <= 1
    assert inp.global_features["num_active_edges"] == len(active_edges)
    for f in inp.candidate_edges:
        if not f.edge_exists:
            assert 0 <= f.frontier_score <= 1


def test_bootstrap_prototypes_and_error_weighted_relevance():
    h = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    c = bootstrap_concept_prototypes(h)
    assert c.shape == (4, 3)
    np.testing.assert_allclose(c, h.mean(axis=0).T)
    a = np.array([[1., 0., .5], [.5, 1., .5]])
    loss = np.array([2., 1.])
    q = error_weighted_edge_relevance(a, loss, normalize=False)
    expected = np.einsum("n,ni,nj->ij", loss, a, a) / 2
    np.fill_diagonal(expected, 0.)
    np.testing.assert_allclose(q, expected)


def test_soft_graph_tokenizer_shapes_and_sample_conditioning():
    torch = pytest.importorskip("torch")
    torch.manual_seed(1)
    prototypes = torch.randn(4, 6)
    module = SoftGraphTokenizer(prototypes, output_dim=10, num_tokens=3,
                                graph_hidden_dim=8, activation_tau=.2)
    token_embeddings = torch.randn(2, 5, 6)
    token_embeddings[1] = token_embeddings[1] + 2.0
    w = torch.tensor([[0., .5, 0., 0.], [.5, 0., -.2, 0.],
                      [0., -.2, 0., .3], [0., 0., .3, 0.]])
    mask = torch.ones(2, 5, dtype=torch.bool)
    tokens, aux = module(token_embeddings, w, return_aux=True, token_mask=mask)
    assert tokens.shape == (2, 3, 10)
    assert aux["activations"].shape == (2, 4)
    assert aux["sample_graph"].shape == (2, 4, 4)
    expected = w.unsqueeze(0) * aux["activations"].unsqueeze(2) * aux["activations"].unsqueeze(1)
    torch.testing.assert_close(aux["sample_graph"], expected)
    assert not torch.allclose(aux["activations"][0], aux["activations"][1])
