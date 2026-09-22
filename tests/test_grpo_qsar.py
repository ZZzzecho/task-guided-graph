import numpy as np
import pytest

from graph_mvp.config import GRPOConfig
from graph_mvp.environment import CandidateBuilder
from graph_mvp.policy import GRPOPolicy
from graph_mvp.types import GroupPolicyExperience, ActionGroup
from graph_mvp.qsar import load_qsar_uci_csv, qsar_dataset, QSAR_CONCEPT_IDS


def test_qsar_parser_and_stratified_split(tmp_path):
    path = tmp_path / "biodeg.csv"
    lines = []
    for k in range(20):
        values = [str(k + j / 100) for j in range(41)]
        lines.append(";".join(values + ["RB" if k % 2 == 0 else "NRB"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    X, y = load_qsar_uci_csv(path)
    assert X.shape == (20, 41) and y.sum() == 10
    ds = qsar_dataset(path, seed=9)
    assert ds.concept_ids == QSAR_CONCEPT_IDS
    assert (len(ds.X_train), len(ds.X_reward), len(ds.X_test)) == (12, 4, 4)
    assert set(np.unique(ds.y_train)) == {0, 1}


def test_qsar_parser_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("1;2;RB\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_qsar_uci_csv(path)


def test_batch_grpo_sampling_is_reproducible_without_replacement_and_update_runs(state):
    pytest.importorskip("torch")
    inp = CandidateBuilder().build(state)
    cfg = GRPOConfig(hidden_dim=16, learning_rate=1e-3, update_epochs=2,
                     kl_coef=0.01, entropy_coef=0.0, edits_per_candidate=2)
    a, b = GRPOPolicy(cfg, seed=123), GRPOPolicy(cfg, seed=123)
    ra, rb = a.sample(inp, 6), b.sample(inp, 6)
    assert all(isinstance(x, ActionGroup) for x in ra)
    sig_a = [[(r.edge, r.action) for r in g.actions] for g in ra]
    sig_b = [[(r.edge, r.action) for r in g.actions] for g in rb]
    assert sig_a == sig_b
    assert all(len(g.actions) == 2 and len(set(g.edges)) == 2 for g in ra)
    assert all(np.isfinite(g.log_prob) and g.log_prob <= 0 for g in ra)
    rewards = np.linspace(-0.2, 0.3, len(ra))
    batch = [GroupPolicyExperience(state.state_id, g.candidate_id, g.log_prob,
                                   float(reward), True, len(g.actions))
             for g, reward in zip(ra, rewards)]
    update = a.update(batch)
    assert update["updated"] is True
    assert update["num_valid"] == len(batch)
    assert np.isfinite(update["loss"])
    assert np.isfinite(update["kl_to_reference"])
    assert update["mean_edits_per_candidate"] == 2


def test_batch_grpo_masks_invalid_and_handles_flat_groups(state):
    pytest.importorskip("torch")
    inp = CandidateBuilder().build(state)
    policy = GRPOPolicy(GRPOConfig(hidden_dim=8, update_epochs=1, edits_per_candidate=2), seed=2)
    groups = policy.sample(inp, 3)
    invalid = [GroupPolicyExperience(state.state_id, g.candidate_id, g.log_prob,
                                     -1.0, False, len(g.actions)) for g in groups]
    out = policy.update(invalid)
    assert out["updated"] is False and out["num_valid"] == 0

    groups = policy.sample(inp, 3)
    flat = [GroupPolicyExperience(state.state_id, g.candidate_id, g.log_prob,
                                  0.1, True, len(g.actions)) for g in groups]
    out = policy.update(flat)
    assert out["updated"] is False
    assert "variation" in out["reason"]
