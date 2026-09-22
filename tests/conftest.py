import numpy as np
import pytest
from graph_mvp.environment import GraphEnvironment
from graph_mvp.downstream import TaskContext


@pytest.fixture
def env():
    return GraphEnvironment()


@pytest.fixture
def state(env):
    return env.initialize(np.array([[1., .45, .18], [.45, 1., .28], [.18, .28, 1.]]),
                          .3, ("a", "b", "c"))


@pytest.fixture
def context():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(180, 3))
    y = (x[:, 0] * x[:, 1] + .2 * rng.normal(size=len(x)) > 0).astype(int)
    return TaskContext(("a", "b", "c"), x[:120], y[:120], x[120:], y[120:])

