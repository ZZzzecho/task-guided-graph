"""All non-blocking implementation choices live in typed configuration."""
from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path


def positive(name, value, allow_zero=False):
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


@dataclass(frozen=True)
class SolverConfig:
    max_iter: int = 3000
    rho: float = 1.0
    abs_tol: float = 1e-7
    rel_tol: float = 1e-6
    kkt_tol: float = 2e-5

    def __post_init__(self):
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        for name in ("rho", "abs_tol", "rel_tol", "kkt_tol"):
            positive(name, getattr(self, name))




@dataclass(frozen=True)
class MNGMConfig:
    """Alternating nonparanormal matrix-normal graph estimator."""
    max_iter: int = 12
    tol: float = 1e-4
    representation_penalty: float = 0.4
    covariance_ridge: float = 1e-4
    min_eig: float = 1e-6
    transform: str = "rank_gaussian"
    scale_constraint: str = "trace"

    def __post_init__(self):
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("MNGM max_iter must be a positive integer")
        for name in ("tol", "representation_penalty", "min_eig"):
            positive(name, getattr(self, name))
        positive("covariance_ridge", self.covariance_ridge, True)
        if self.transform not in ("rank_gaussian", "spearman_sine"):
            raise ValueError("MNGM transform must be rank_gaussian or spearman_sine")
        if self.scale_constraint not in ("trace", "none"):
            raise ValueError("MNGM scale_constraint must be trace or none")


@dataclass(frozen=True)
class EnvironmentConfig:
    # Log-penalty step. Legacy default retained; Bootstrap v0.2 config uses log(1.1).
    eta: float = 0.5
    lambda_min: float = 0.01
    lambda_max: float = 4.0
    adjacency_threshold: float = 1e-5
    max_density: float = 1.0

    def __post_init__(self):
        for name in ("eta", "lambda_min", "lambda_max"):
            positive(name, getattr(self, name))
        if self.lambda_min > self.lambda_max:
            raise ValueError("lambda_min must not exceed lambda_max")
        if not 0 <= self.adjacency_threshold < 1 or not 0 <= self.max_density <= 1:
            raise ValueError("invalid adjacency threshold or max_density")


@dataclass(frozen=True)
class CandidateConfig:
    """Active-set + solver-KKT frontier candidate pool."""
    max_frontier_edges: int = 256
    min_frontier_edges: int = 32
    frontier_min_score: float = 0.80
    zero_tolerance: float = 1e-10

    def __post_init__(self):
        if not isinstance(self.max_frontier_edges, int) or self.max_frontier_edges < 0:
            raise ValueError("max_frontier_edges must be a nonnegative integer")
        if not isinstance(self.min_frontier_edges, int) or self.min_frontier_edges < 0:
            raise ValueError("min_frontier_edges must be a nonnegative integer")
        if self.min_frontier_edges > self.max_frontier_edges:
            raise ValueError("min_frontier_edges must not exceed max_frontier_edges")
        if not 0 <= self.frontier_min_score <= 1:
            raise ValueError("frontier_min_score must be in [0, 1]")
        positive("zero_tolerance", self.zero_tolerance, True)


@dataclass(frozen=True)
class DownstreamConfig:
    C: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-8

    def __post_init__(self):
        positive("C", self.C)
        positive("tol", self.tol)
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("downstream max_iter must be a positive integer")


@dataclass(frozen=True)
class RewardConfig:
    alpha: float = 0.02
    beta: float = 0.01
    density_mode: str = "increase_only"
    invalid_reward: float = -1.0

    def __post_init__(self):
        positive("alpha", self.alpha, True)
        positive("beta", self.beta, True)
        if self.density_mode not in ("increase_only", "signed_delta"):
            raise ValueError("density_mode must be increase_only or signed_delta")
        if not math.isfinite(self.invalid_reward) or self.invalid_reward >= 0:
            raise ValueError("invalid_reward must be finite and negative")


@dataclass(frozen=True)
class AcceptanceConfig:
    reward_threshold: float = 0.0
    min_task_improvement: float = 1e-8

    def __post_init__(self):
        positive("reward_threshold", self.reward_threshold, True)
        positive("min_task_improvement", self.min_task_improvement, True)


@dataclass(frozen=True)
class DataConfig:
    seed: int = 7
    n_train: int = 600
    n_reward: int = 300
    n_test: int = 300
    p: int = 10
    nonparanormal: bool = True
    covariance_ridge: float = 1e-4

    def __post_init__(self):
        if any(not isinstance(v, int) or v < 2 for v in
               (self.p, self.n_train, self.n_reward, self.n_test)):
            raise ValueError("data dimensions must be integers >= 2")
        positive("covariance_ridge", self.covariance_ridge, True)


@dataclass(frozen=True)
class GRPOConfig:
    """Hierarchical batch-edge GRPO over edge selection and penalty direction.

    Unselected edges are implicit keep actions. The policy sees structured edge-state
    features only; a fixed reference network anchors KL regularization.
    """
    hidden_dim: int = 32
    learning_rate: float = 3e-3
    update_epochs: int = 4
    clip_epsilon: float = 0.2
    kl_coef: float = 0.01
    entropy_coef: float = 0.002
    max_grad_norm: float = 1.0
    min_group_std: float = 1e-8
    reward_memory_decay: float = 0.8
    edits_per_candidate: int = 32
    selection_temperature: float = 1.0
    direction_temperature: float = 1.0
    device: str = "cpu"

    def __post_init__(self):
        if not isinstance(self.hidden_dim, int) or self.hidden_dim < 2:
            raise ValueError("hidden_dim must be an integer >= 2")
        if not isinstance(self.update_epochs, int) or self.update_epochs < 1:
            raise ValueError("update_epochs must be a positive integer")
        if not isinstance(self.edits_per_candidate, int) or self.edits_per_candidate < 1:
            raise ValueError("edits_per_candidate must be a positive integer")
        for name in ("learning_rate", "clip_epsilon", "max_grad_norm", "min_group_std",
                     "selection_temperature", "direction_temperature"):
            positive(name, getattr(self, name))
        for name in ("kl_coef", "entropy_coef"):
            positive(name, getattr(self, name), True)
        if not 0 <= self.reward_memory_decay < 1:
            raise ValueError("reward_memory_decay must be in [0, 1)")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")


@dataclass(frozen=True)
class RunnerConfig:
    rounds: int = 8
    num_candidates: int = 12
    seed: int = 11
    policy: str = "greedy"
    initial_lambda: float = 0.4
    scalar_grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2)

    def __post_init__(self):
        object.__setattr__(self, "scalar_grid", tuple(self.scalar_grid))
        if not isinstance(self.rounds, int) or self.rounds < 0:
            raise ValueError("rounds must be a nonnegative integer")
        if not isinstance(self.num_candidates, int) or self.num_candidates < 1:
            raise ValueError("num_candidates must be a positive integer")
        if self.policy not in ("random", "greedy", "grpo"):
            raise ValueError("policy must be random, greedy, or grpo")
        positive("initial_lambda", self.initial_lambda)
        for x in self.scalar_grid:
            positive("scalar_grid entry", x)


@dataclass(frozen=True)
class Config:
    solver: SolverConfig = field(default_factory=SolverConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    mngm: MNGMConfig = field(default_factory=MNGMConfig)
    downstream: DownstreamConfig = field(default_factory=DownstreamConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    @classmethod
    def load(cls, path: str | Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        defaults = cls()
        constructors = {f.name: type(getattr(defaults, f.name)) for f in fields(cls)}
        if set(raw) - set(constructors):
            raise ValueError(f"Unknown config sections: {set(raw) - set(constructors)}")
        return cls(**{k: constructors[k](**v) for k, v in raw.items()})
