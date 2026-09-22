"""Offline synthetic fixture and explicit pre-split NPZ data adapter."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.special import ndtri, expit
from .config import DataConfig


class RankGaussianTransformer:
    """Training ECDF only; ties have midranks, endpoints are clipped at 0.5/n."""
    def fit(self, X):
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("Rank transform requires a finite 2D training matrix")
        self.sorted_ = np.sort(x, axis=0)
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=float)
        if not hasattr(self, "sorted_"):
            raise ValueError("Transformer has not been fitted")
        if x.ndim != 2 or x.shape[1] != self.sorted_.shape[1] or not np.isfinite(x).all():
            raise ValueError("Invalid transform input")
        n = len(self.sorted_)
        out = np.empty_like(x)
        for j in range(x.shape[1]):
            lo = np.searchsorted(self.sorted_[:, j], x[:, j], side="left")
            hi = np.searchsorted(self.sorted_[:, j], x[:, j], side="right")
            out[:, j] = ndtri(np.clip((lo + hi) / (2 * n), 0.5 / n, 1 - 0.5 / n))
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def training_covariance(X_train, ridge=1e-4):
    x = np.asarray(X_train, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Invalid covariance input or ridge")
    centered = x - x.mean(axis=0)
    return centered.T @ centered / len(x) + ridge * np.eye(x.shape[1])


@dataclass
class Dataset:
    concept_ids: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_reward: np.ndarray
    y_reward: np.ndarray
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None


def synthetic_dataset(config=DataConfig()):
    """Small controlled binary interaction task, not evidence of real-data efficacy."""
    rng = np.random.default_rng(config.seed)
    p = config.p
    precision = np.zeros((p, p))
    for i in range(p - 1):
        precision[i, i + 1] = precision[i + 1, i] = -0.3
    task_edges = [(0, p - 1)]
    if p >= 6:
        task_edges.append((1, p - 2))
    for i, j in task_edges:
        precision[i, j] = precision[j, i] = -0.1
    np.fill_diagonal(precision, 0.6 + np.abs(precision).sum(axis=1))
    n = config.n_train + config.n_reward + config.n_test
    z = rng.multivariate_normal(np.zeros(p), np.linalg.inv(precision), size=n)
    logits = 0.2 * z[:, 0] + sum(2 * z[:, i] * z[:, j] for i, j in task_edges)
    y = (rng.random(n) < expit(logits)).astype(int)
    x = z.copy()
    x[:, 1::3] = z[:, 1::3] ** 3
    x[:, 2::3] = np.sinh(z[:, 2::3])
    a, b = config.n_train, config.n_train + config.n_reward
    return Dataset(tuple(f"concept_{j:03d}" for j in range(p)),
                   x[:a], y[:a], x[a:b], y[a:b], x[b:], y[b:])


def load_dataset(path: str | Path):
    """Requires explicit concept order and splits. Never silently constructs a split."""
    with np.load(path, allow_pickle=False) as raw:
        required = {"concept_ids", "X_train", "y_train", "X_reward", "y_reward"}
        if required - set(raw.files):
            raise ValueError(f"Missing NPZ keys: {required - set(raw.files)}")
        ids = raw["concept_ids"]
        if ids.ndim != 1 or ids.dtype.kind != "U":
            raise ValueError("concept_ids must be a 1D Unicode string array")
        has_test = "X_test" in raw.files
        if has_test != ("y_test" in raw.files):
            raise ValueError("X_test and y_test must be supplied together")
        return Dataset(tuple(ids.tolist()), *(raw[k].copy() for k in
                       ("X_train", "y_train", "X_reward", "y_reward")),
                       raw["X_test"].copy() if has_test else None,
                       raw["y_test"].copy() if has_test else None)
