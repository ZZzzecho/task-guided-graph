"""Search/RL policies consume structured graph-state features only, never raw task data."""
from __future__ import annotations

from copy import deepcopy
from typing import Protocol, Sequence
import numpy as np

from .config import GRPOConfig
from .types import (ActionRecord, ActionGroup, PolicyInput, PolicyExperience,
                    GroupPolicyExperience)

ACTIONS = ("increase", "keep", "decrease")
BATCH_DIRECTIONS = ("increase", "decrease")


class Policy(Protocol):
    def sample(self, policy_input: PolicyInput, num_candidates: int): ...
    def update(self, policy_batch: Sequence): ...


class BaselinePolicy:
    def __init__(self):
        self._serial = 0

    def _record(self, inp, edge, action, log_prob):
        self._serial += 1
        return ActionRecord(f"{self.version}-{self._serial:06d}", inp.state_id, edge,
                            action, float(log_prob), self.version)

    def update(self, policy_batch):
        return {"updated": False, "num_experiences": len(policy_batch),
                "num_valid": sum(bool(e.valid_mask) for e in policy_batch)}

    def save(self, path):
        return False


class RandomPolicy(BaselinePolicy):
    """Legacy single-edge random-search baseline."""
    version = "random-v1"

    def __init__(self, seed=0):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def sample(self, policy_input, num_candidates):
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        edges = policy_input.candidate_edges
        if not edges:
            return []
        return [self._record(policy_input, edges[int(self.rng.integers(len(edges)))].edge,
                             ACTIONS[int(self.rng.integers(3))], -np.log(3 * len(edges)))
                for _ in range(num_candidates)]


class GreedyPolicy(BaselinePolicy):
    """Legacy single-edge coordinate-search baseline."""
    version = "greedy-v1"

    def __init__(self):
        super().__init__()
        self.cursor = 0

    def sample(self, policy_input, num_candidates):
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        edges = policy_input.candidate_edges
        if not edges:
            return []
        choices = [(f.edge, a) for f in edges for a in ("increase", "decrease")]
        choices.append((edges[0].edge, "keep"))
        count = min(num_candidates, len(choices))
        batch = [choices[(self.cursor + k) % len(choices)] for k in range(count)]
        self.cursor = (self.cursor + count) % len(choices)
        return [self._record(policy_input, edge, action, 0.) for edge, action in batch]


class GRPOPolicy(BaselinePolicy):
    """Hierarchical batch-edge GRPO policy from the Bootstrap-MNGM handoff.

    One candidate graph is a trajectory of `m` unique edge edits.  The policy has
    a shared edge encoder with two heads:
      1) selection logits used for sequential sampling without replacement;
      2) increase/decrease logits for every selected edge.

    Unselected edges are the keep action.  Each trajectory receives one downstream
    graph reward; within-group normalized reward is used as the trajectory-level
    GRPO advantage.  The graph environment, not this policy, decides which edges
    ultimately enter or leave the precision graph after the full MNGM re-solve.
    """
    version = "grpo-batch-v2"

    def __init__(self, config=GRPOConfig(), seed=0):
        super().__init__()
        self.config = config
        self.seed = int(seed)
        self._model = None
        self._reference = None
        self._optimizer = None
        self._torch = None
        self._nn = None
        self._generator = None
        self._cache: dict[str, dict] = {}
        self._edge_reward_ema: dict[tuple[int, int], float] = {}

    def _import_torch(self):
        if self._torch is None:
            try:
                import torch
                import torch.nn as nn
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "GRPO requires PyTorch. Install with `pip install -e '.[rl]'` or install torch separately."
                ) from exc
            self._torch, self._nn = torch, nn
        return self._torch, self._nn

    def _encode(self, policy_input: PolicyInput):
        edges = policy_input.candidate_edges
        if not edges:
            return np.empty((0, 12), dtype=np.float32)
        max_index = max(max(f.edge) for f in edges)
        denom = float(max(max_index, 1))
        rows = []
        for f in edges:
            base = np.asarray(f.vector(), dtype=np.float32).copy()
            # EdgeFeatures vector index 6 is previous_reward. Replace it with policy
            # reward memory so historical reward remains a state-only feature.
            base[6] = np.float32(self._edge_reward_ema.get(f.edge, base[6]))
            i, j = f.edge
            rows.append(np.concatenate((base, np.asarray([i / denom, j / denom], dtype=np.float32))))
        return np.asarray(rows, dtype=np.float32)

    def _ensure_model(self, input_dim: int):
        if self._model is not None:
            if self._model.input_dim != input_dim:
                raise ValueError("GRPO policy feature dimension changed during a run")
            return
        torch, nn = self._import_torch()
        device = torch.device(self.config.device)

        class EdgePolicyNet(nn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.input_dim = d
                self.body = nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, h), nn.Tanh(),
                    nn.Linear(h, h), nn.Tanh(),
                )
                self.selection = nn.Linear(h, 1)
                self.direction = nn.Linear(h, 2)

            def forward(self, x):
                h = self.body(x)
                return self.selection(h).squeeze(-1), self.direction(h)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self._model = EdgePolicyNet(input_dim, self.config.hidden_dim).to(device)
        self._reference = deepcopy(self._model).to(device).eval()
        for p in self._reference.parameters():
            p.requires_grad_(False)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config.learning_rate)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(self.seed + 17)

    def _tensor_features(self, feature_matrix):
        torch, _ = self._import_torch()
        self._ensure_model(feature_matrix.shape[1])
        return torch.as_tensor(feature_matrix, dtype=torch.float32,
                               device=torch.device(self.config.device))

    def sample(self, policy_input, num_candidates):
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        edges = policy_input.candidate_edges
        if not edges:
            return []
        features = self._encode(policy_input)
        x = self._tensor_features(features)
        torch, _ = self._import_torch()
        with torch.no_grad():
            selection_logits, direction_logits = self._model(x)
        m = min(self.config.edits_per_candidate, len(edges))
        groups = []
        for _ in range(int(num_candidates)):
            self._serial += 1
            candidate_id = f"{self.version}-{self._serial:06d}"
            available = torch.ones(len(edges), dtype=torch.bool, device=x.device)
            selected, directions, records = [], [], []
            joint_log_prob = 0.0
            for _step in range(m):
                logits = selection_logits / self.config.selection_temperature
                masked = logits.masked_fill(~available, float("-inf"))
                sel_log_probs = torch.log_softmax(masked, dim=0)
                index = int(torch.multinomial(torch.exp(sel_log_probs), 1,
                                              replacement=False,
                                              generator=self._generator).item())
                dir_log_probs = torch.log_softmax(
                    direction_logits[index] / self.config.direction_temperature, dim=0)
                direction = int(torch.multinomial(torch.exp(dir_log_probs), 1,
                                                  replacement=True,
                                                  generator=self._generator).item())
                selection_lp = float(sel_log_probs[index].cpu())
                direction_lp = float(dir_log_probs[direction].cpu())
                component = selection_lp + direction_lp
                action_name = BATCH_DIRECTIONS[direction]
                records.append(ActionRecord(
                    candidate_id,
                    policy_input.state_id,
                    edges[index].edge,
                    action_name,
                    component,
                    self.version,
                    selection_log_prob=selection_lp,
                    direction_log_prob=direction_lp,
                ))
                selected.append(index)
                directions.append(direction)
                joint_log_prob += component
                available[index] = False
            group = ActionGroup(candidate_id, policy_input.state_id, tuple(records),
                                float(joint_log_prob), self.version)
            self._cache[candidate_id] = {
                "state_id": policy_input.state_id,
                "features": features.copy(),
                "selected": tuple(selected),
                "directions": tuple(directions),
                "edges": tuple(edges[i].edge for i in selected),
            }
            groups.append(group)
        return groups

    def _trajectory_stats(self, features, selected, directions):
        """Differentiable joint log-p plus exact local KL/entropy along a path."""
        torch, _ = self._import_torch()
        x = self._tensor_features(features)
        sel, direc = self._model(x)
        with torch.no_grad():
            ref_sel, ref_dir = self._reference(x)
        available = torch.ones(len(features), dtype=torch.bool, device=x.device)
        joint = torch.zeros((), device=x.device)
        kl_terms, entropy_terms = [], []
        for index, direction in zip(selected, directions):
            cur_sel_lp = torch.log_softmax(
                (sel / self.config.selection_temperature).masked_fill(~available, float("-inf")), dim=0)
            ref_sel_lp = torch.log_softmax(
                (ref_sel / self.config.selection_temperature).masked_fill(~available, float("-inf")), dim=0)
            available_idx = torch.where(available)[0]
            psel = torch.exp(cur_sel_lp[available_idx])
            kl_sel = torch.sum(psel * (cur_sel_lp[available_idx] - ref_sel_lp[available_idx]))
            ent_sel = -torch.sum(psel * cur_sel_lp[available_idx])

            cur_dir_lp = torch.log_softmax(direc[index] / self.config.direction_temperature, dim=0)
            ref_dir_lp = torch.log_softmax(ref_dir[index] / self.config.direction_temperature, dim=0)
            pdir = torch.exp(cur_dir_lp)
            kl_dir = torch.sum(pdir * (cur_dir_lp - ref_dir_lp))
            ent_dir = -torch.sum(pdir * cur_dir_lp)

            joint = joint + cur_sel_lp[index] + cur_dir_lp[direction]
            kl_terms.append(kl_sel + kl_dir)
            entropy_terms.append(ent_sel + ent_dir)
            available[index] = False
        return joint, torch.stack(kl_terms).mean(), torch.stack(entropy_terms).mean()

    def _update_reward_memory(self, experiences):
        decay = self.config.reward_memory_decay
        for exp in experiences:
            item = self._cache.get(exp.candidate_id)
            if item is None or not exp.valid_mask:
                continue
            for edge in item["edges"]:
                old = self._edge_reward_ema.get(edge, 0.0)
                self._edge_reward_ema[edge] = decay * old + (1 - decay) * float(exp.reward)

    def update(self, policy_batch):
        batch = list(policy_batch)
        if batch and isinstance(batch[0], PolicyExperience):
            raise TypeError("Batch GRPO expects GroupPolicyExperience from ActionGroup candidates")
        valid = [e for e in batch if isinstance(e, GroupPolicyExperience)
                 and e.valid_mask and e.candidate_id in self._cache]
        self._update_reward_memory(batch)
        if not valid:
            for e in batch:
                self._cache.pop(e.candidate_id, None)
            return {"updated": False, "reason": "no valid on-policy experiences",
                    "num_experiences": len(batch), "num_valid": 0}
        state_ids = {e.state_id for e in valid}
        if len(state_ids) != 1:
            raise ValueError("GRPO update expects one same-state candidate group")
        rewards = np.asarray([e.reward for e in valid], dtype=float)
        mean, std = float(rewards.mean()), float(rewards.std())
        if len(valid) < 2 or std < self.config.min_group_std:
            for e in batch:
                self._cache.pop(e.candidate_id, None)
            return {"updated": False, "reason": "insufficient within-group reward variation",
                    "num_experiences": len(batch), "num_valid": len(valid),
                    "reward_mean": mean, "reward_std": std}

        advantages = (rewards - mean) / (std + 1e-12)
        torch, _ = self._import_torch()
        device = torch.device(self.config.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        old = torch.as_tensor([e.old_log_prob for e in valid], dtype=torch.float32, device=device)

        last = {}
        for _ in range(self.config.update_epochs):
            joint_logs, kls, entropies = [], [], []
            for exp in valid:
                item = self._cache[exp.candidate_id]
                joint, kl, entropy = self._trajectory_stats(
                    item["features"], item["selected"], item["directions"])
                joint_logs.append(joint)
                kls.append(kl)
                entropies.append(entropy)
            selected = torch.stack(joint_logs)
            kl = torch.stack(kls).mean()
            entropy = torch.stack(entropies).mean()
            ratio = torch.exp(selected - old)
            clipped = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
            surrogate = torch.minimum(ratio * adv, clipped * adv)
            loss = -torch.mean(surrogate) + self.config.kl_coef * kl - self.config.entropy_coef * entropy
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.config.max_grad_norm)
            self._optimizer.step()
            with torch.no_grad():
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.config.clip_epsilon).float())
            last = {
                "loss": float(loss.detach().cpu()),
                "kl_to_reference": float(kl.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "clip_fraction": float(clip_fraction.detach().cpu()),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            }

        with torch.no_grad():
            final = []
            for exp in valid:
                item = self._cache[exp.candidate_id]
                joint, _, _ = self._trajectory_stats(item["features"], item["selected"], item["directions"])
                final.append(joint)
            final_lp = torch.stack(final)
            approx_kl_old = float(torch.mean(old - final_lp).cpu())
        for e in batch:
            self._cache.pop(e.candidate_id, None)
        return {
            "updated": True,
            "num_experiences": len(batch),
            "num_valid": len(valid),
            "reward_mean": mean,
            "reward_std": std,
            "advantage_min": float(advantages.min()),
            "advantage_max": float(advantages.max()),
            "mean_old_minus_new_trajectory_logprob": approx_kl_old,
            "mean_edits_per_candidate": float(np.mean([e.num_edits for e in valid])),
            **last,
        }

    def save(self, path):
        if self._model is None:
            return False
        torch, _ = self._import_torch()
        payload = {
            "version": self.version,
            "seed": self.seed,
            "config": self.config.__dict__,
            "model_state_dict": self._model.state_dict(),
            "reference_state_dict": self._reference.state_dict(),
            "edge_reward_ema": {f"{i},{j}": v for (i, j), v in self._edge_reward_ema.items()},
        }
        torch.save(payload, path)
        return True
