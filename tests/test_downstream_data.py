from dataclasses import replace
import numpy as np
import pytest
from graph_mvp.data import RankGaussianTransformer, training_covariance, load_dataset
from graph_mvp.downstream import DownstreamEvaluator, TaskContext
from graph_mvp.types import ActionRecord


def test_train_only_rank_transform_and_ties():
    x = np.array([[1., 5.], [2., 5.], [2., 5.], [4., 5.]])
    transform = RankGaussianTransformer().fit(x)
    z = transform.transform(x)
    assert np.isfinite(z).all()
    np.testing.assert_array_equal(z[:, 1], np.zeros(4))
    assert z[1, 0] == z[2, 0]
    transform.transform(np.array([[1e10, -1e10]]))
    np.testing.assert_array_equal(z, transform.transform(x))
    assert np.linalg.eigvalsh(training_covariance(z))[0] > 0


def test_deterministic_paired_evaluation_and_fixed_feature_slots(env, state, context):
    evaluator = DownstreamEvaluator()
    first = evaluator.evaluate_snapshot(state.snapshot, context)
    second = evaluator.evaluate_snapshot(state.snapshot, context)
    assert first == second
    c = env.step(state, ActionRecord("keep", state.state_id, (0, 1), "keep", 0., "test"))
    r = evaluator.evaluate(c, context, first)
    assert r.delta_task == 0
    assert r.extra_metrics["num_feature_slots"] == 6
    empty = evaluator.features(context.X_train, np.zeros((3, 3)), np.zeros(3), np.ones(3))
    assert empty.shape[1] == 6 and np.count_nonzero(empty[:, 3:]) == 0


def test_graph_has_real_downstream_effect(env, context):
    s = np.array([[1., .4, 0.], [.4, 1., 0.], [0., 0., 1.]])
    absent = env.initialize(s, 1., context.concept_ids)
    present = env.initialize(s, .1, context.concept_ids)
    evaluator = DownstreamEvaluator()
    a = evaluator.evaluate_snapshot(absent.snapshot, context)
    b = evaluator.evaluate_snapshot(present.snapshot, context)
    assert b.task_loss < a.task_loss - .1


def test_alignment_and_binary_validation(state, context):
    with pytest.raises(ValueError):
        DownstreamEvaluator().evaluate_snapshot(state.snapshot, replace(context, concept_ids=("c", "b", "a")))
    with pytest.raises(ValueError):
        replace(context, y_train=np.full(len(context.y_train), 2))
    with pytest.raises(ValueError):
        context.X_train[0, 0] = 10
    assert not hasattr(context, "X_test")


def test_npz_roundtrip_and_explicit_axis(tmp_path, context):
    path = tmp_path / "data.npz"
    np.savez(path, concept_ids=np.array(context.concept_ids), X_train=context.X_train,
             y_train=context.y_train, X_reward=context.X_reward, y_reward=context.y_reward)
    data = load_dataset(path)
    assert data.concept_ids == context.concept_ids and data.X_test is None
    np.testing.assert_array_equal(data.X_reward, context.X_reward)
    np.savez(path, X_train=context.X_train)
    with pytest.raises(ValueError):
        load_dataset(path)

