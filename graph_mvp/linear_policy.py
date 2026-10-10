"""Low-dimensional GRPO actions expanded linearly over every concept pair."""
from copy import deepcopy
from collections.abc import Mapping
import numpy as np

from .config import GRPOConfig, CandidateConfig
from .environment import concept_axis_kkt_frontier
from .policy import GRPOPolicy
from .types import PolicyInput, LinearPenaltyAction, readonly

LINEAR_FEATURES = (
    "intercept", "prototype_cosine", "absolute_covariance_correlation",
    "absolute_partial_correlation", "active_edge", "kkt_frontier",
    "mean_normalized_degree", "task_relevance",
)
COEFFICIENT_VALUES = (-1., 0., 1.)


class LinearCandidateBuilder:
    """Structured state features for all pairs; no raw document or test access.

    Row L1 normalization bounds |phi @ a| <= 1 for coefficients in [-1,1].
    Semantic prototypes are fixed and must follow the graph concept order.
    """
    def __init__(self, prototypes=None, config=CandidateConfig()):
        self.config = config
        self.semantic = None
        if prototypes is not None:
            x = np.asarray(prototypes, dtype=float)
            if x.ndim != 2 or not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) == 0):
                raise ValueError("Prototypes must be finite nonzero rows")
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            self.semantic = readonly(np.clip(x @ x.T, -1., 1.))

    def build(self, state, task_relevance=None, uncertainty=None, previous_reward=None):
        g = state.snapshot
        p = len(g.concept_ids)
        upper = np.triu_indices(p, 1)
        active, frontier, _ = concept_axis_kkt_frontier(g, self.config.zero_tolerance)
        degree = active.sum(axis=1) / max(p - 1, 1)
        d = np.sqrt(np.diag(g.S))
        cov = np.clip(g.S / np.outer(d, d), -1., 1.)
        semantic = np.zeros(len(upper[0]))
        if self.semantic is not None:
            if self.semantic.shape != (p, p):
                raise ValueError("Prototype concept axis mismatch")
            semantic = self.semantic[upper]
        relevance = np.zeros(len(upper[0]))
        if task_relevance is not None:
            if isinstance(task_relevance, Mapping):
                relevance = np.array([task_relevance.get((int(i), int(j)),
                    task_relevance.get((int(j), int(i)), 0.)) or 0.
                    for i, j in zip(*upper)], dtype=float)
            else:
                signal = np.asarray(task_relevance, dtype=float)
                if signal.shape != (p, p) or not np.allclose(signal, signal.T):
                    raise ValueError("Task relevance must be symmetric [P,P]")
                relevance = signal[upper]
            if not np.isfinite(relevance).all():
                raise ValueError("Task relevance must be finite")
            relevance = relevance / max(float(np.max(np.abs(relevance))), 1e-12)
        phi = np.column_stack((np.ones(len(upper[0])), semantic, np.abs(cov[upper]),
            np.abs(g.Rho[upper]), active[upper], frontier[upper],
            (degree[upper[0]] + degree[upper[1]]) / 2., relevance))
        phi /= np.maximum(np.sum(np.abs(phi), axis=1, keepdims=True), 1.)
        return PolicyInput(state.state_id, (), {
            "candidate_pool_size": len(phi), "num_active_edges": int(active[upper].sum()),
            "representation_dim": float(g.auxiliary.get("representation_dim", 1)),
        }, phi.astype(np.float32), p)


class LinearGRPOPolicy(GRPOPolicy):
    """Eight categorical coefficient choices, one graph-level GRPO reward.

    The existing clipped GRPO optimizer is reused. Expanded pair adjustments
    have no independent probabilities. No edge reward EMA is maintained.
    """
    version = "grpo-linear-v1"

    def _ensure_model(self, input_dim):
        if self._model is not None:
            if self._model.input_dim != input_dim:
                raise ValueError("Linear policy feature dimension changed")
            return
        torch, nn = self._import_torch()
        device = torch.device(self.config.device)

        class CoefficientNet(nn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.input_dim = d
                self.body = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.Tanh(),
                    nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 24))
                # Initially uniform coefficient choices, without a directional bias.
                nn.init.zeros_(self.body[-1].weight)
                nn.init.zeros_(self.body[-1].bias)

            def forward(self, x):
                return self.body(x).reshape(-1, 8, 3)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self._model = CoefficientNet(input_dim, self.config.hidden_dim).to(device)
        self._reference = deepcopy(self._model).to(device).eval()
        for param in self._reference.parameters():
            param.requires_grad_(False)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config.learning_rate)
        self._generator = torch.Generator(device=device).manual_seed(self.seed + 17)

    def sample(self, policy_input, num_candidates):
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        phi = policy_input.linear_features
        if phi is None:
            raise ValueError("Linear GRPO requires LinearCandidateBuilder")
        features = np.concatenate((phi.mean(axis=0), phi.std(axis=0)))[None, :]
        x = self._tensor_features(features)
        torch, _ = self._import_torch()
        with torch.no_grad():
            lp = torch.log_softmax(self._model(x)[0] / self.config.direction_temperature, dim=-1)
        groups = []
        for _ in range(num_candidates):
            choices = torch.multinomial(lp.exp(), 1, generator=self._generator).squeeze(-1)
            indices = tuple(int(v) for v in choices.cpu())
            coefficients = tuple(COEFFICIENT_VALUES[k] for k in indices)
            delta = np.clip(phi.astype(float) @ np.asarray(coefficients), -1., 1.)
            log_prob = float(lp[torch.arange(8, device=x.device), choices].sum().cpu())
            self._serial += 1
            cid = f"{self.version}-{self._serial:06d}"
            groups.append(LinearPenaltyAction(cid, policy_input.state_id, coefficients, delta,
                policy_input.num_concepts, log_prob, self.version))
            self._cache[cid] = {"state_id": policy_input.state_id, "features": features.copy(),
                "selected": (), "directions": indices, "edges": ()}
        return groups

    def _trajectory_stats(self, features, selected, directions):
        torch, _ = self._import_torch()
        x = self._tensor_features(features)
        lp = torch.log_softmax(self._model(x)[0] / self.config.direction_temperature, dim=-1)
        with torch.no_grad():
            ref_lp = torch.log_softmax(self._reference(x)[0] / self.config.direction_temperature, dim=-1)
        choices = torch.as_tensor(directions, device=x.device)
        joint = lp[torch.arange(8, device=x.device), choices].sum()
        # Sum matches the joint coefficient distribution, not expanded pair count.
        kl = (lp.exp() * (lp - ref_lp)).sum()
        entropy = -(lp.exp() * lp).sum()
        return joint, kl, entropy

    def _update_reward_memory(self, experiences):
        pass
