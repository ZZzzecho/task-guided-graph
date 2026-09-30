"""Reproducible query streams and fixed retrieval encoder/training states."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
import random

import numpy as np


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _hash_tensor(hasher, name, tensor):
    """Hash actual weights once, including BF16, with bounded CPU transfers."""
    import torch
    if tensor.device.type == "meta":
        raise ValueError("Fixed candidate snapshot requires materialized weights")
    hasher.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), 1 << 20):
        block = flat[start:start + (1 << 20)].cpu().contiguous()
        hasher.update(block.view(torch.uint8).numpy().tobytes())


class FrozenCandidateSnapshot:
    """Time-share frozen base weights and an immutable initial adapter snapshot.

    Candidate forwards temporarily use the captured trainable weights/buffers in
    eval/no-grad mode; query weights and every module's mode are restored even on
    error. This avoids a second GLM backbone. Calls must be sequential, with no
    outstanding query autograd graph. Changes to frozen base weights fail closed.
    """

    def __init__(self, causal_lm):
        self.model = causal_lm
        self._parameters = dict(causal_lm.named_parameters())
        self._buffers = dict(causal_lm.named_buffers())
        self._mutable = {
            name: p.detach().cpu().clone()
            for name, p in self._parameters.items() if p.requires_grad
        }
        self._buffer_state = {
            name: b.detach().cpu().clone() for name, b in self._buffers.items()
        }
        self._frozen_versions = {
            name: p._version for name, p in self._parameters.items()
            if name not in self._mutable
        }
        hasher = sha256()
        config = getattr(causal_lm, "config", None)
        config = config.to_dict() if hasattr(config, "to_dict") else str(config)
        hasher.update(json.dumps(config, sort_keys=True, default=str).encode())
        for name, tensor in sorted(self._parameters.items()):
            _hash_tensor(hasher, name, tensor)
        for name, tensor in sorted(self._buffers.items()):
            _hash_tensor(hasher, "buffer:" + name, tensor)
        self.fingerprint = hasher.hexdigest()

    def assert_stable(self):
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        if params.keys() != self._parameters.keys() or buffers.keys() != self._buffers.keys():
            raise ValueError("Candidate encoder parameter/buffer topology changed")
        for name, p in params.items():
            if p is not self._parameters[name] or p.requires_grad != (name in self._mutable):
                raise ValueError(f"Candidate encoder parameter identity/trainability changed: {name}")
            if name in self._frozen_versions and p._version != self._frozen_versions[name]:
                raise ValueError(f"Frozen candidate base weight changed: {name}")
        for name, buffer in buffers.items():
            if buffer is not self._buffers[name]:
                raise ValueError(f"Candidate encoder buffer identity changed: {name}")

    @contextmanager
    def activate(self):
        import torch
        self.assert_stable()
        current = {n: self._parameters[n].detach().cpu().clone() for n in self._mutable}
        current_buffers = {n: b.detach().cpu().clone() for n, b in self._buffers.items()}
        modes = [(m, m.training) for m in self.model.modules()]
        with torch.no_grad():
            try:
                for n, value in self._mutable.items():
                    self._parameters[n].copy_(value)
                for n, value in self._buffer_state.items():
                    self._buffers[n].copy_(value)
                self.model.eval()
                yield
            finally:
                for n, value in current.items():
                    self._parameters[n].copy_(value)
                for n, value in current_buffers.items():
                    self._buffers[n].copy_(value)
                for module, mode in modes:
                    module.training = mode

    def metadata(self):
        return {
            "state": "initial_encoder_snapshot",
            "strategy": "frozen_base_with_initial_parameter_snapshot",
            "fingerprint": self.fingerprint,
            "fingerprint_scope": "config_and_all_parameter_and_buffer_bytes",
            "snapshot_parameter_names": sorted(self._mutable),
        }


class RetrievalQueryStream:
    """Seeded shuffled epochs; retain cursor across warmup/accepted phases.

    Epoch is zero-based. The final partial batch is retained; cursor can equal
    pool size until the next batch starts a new epoch. Only successful optimizer
    steps advance counts. Evaluation keeps using the context's ordered batches.
    """

    def __init__(self, context, *, seed=17):
        self.seed = int(seed)
        self.query_ids = tuple(context.query_ids)
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("Training query IDs must be unique")
        self.pool_fingerprint = self.context_fingerprint(context)
        self.batch_size = int(context.batch_size)
        self.epoch = self.cursor = self.optimizer_steps = 0
        self.presentations = np.zeros(len(self.query_ids), dtype=np.int64)
        self.order = self._order()

    @staticmethod
    def context_fingerprint(context):
        return _digest([context.query_ids, context.query_texts,
                        context.candidate_ids, context.positive_indices,
                        context.batch_size, context.max_length, context.score_temperature])

    def _order(self):
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).permutation(len(self.query_ids))

    def validate_context(self, context):
        if self.context_fingerprint(context) != self.pool_fingerprint:
            raise ValueError("Training stream/context mismatch")

    def batch(self, context):
        self.validate_context(context)
        if self.cursor == len(self.order):
            self.epoch += 1
            self.cursor = 0
            self.order = self._order()
        indices = self.order[self.cursor:self.cursor + self.batch_size]
        return {key: tuple(getattr(context, key)[int(i)] for i in indices)
                for key in ("query_ids", "query_texts", "candidate_ids", "positive_indices")}

    def advance(self):
        if self.cursor >= len(self.order):
            raise ValueError("Call batch before advancing a completed epoch")
        indices = self.order[self.cursor:self.cursor + self.batch_size]
        self.presentations[indices] += 1
        self.cursor += len(indices)
        self.optimizer_steps += 1

    def summary(self):
        return {
            "actual_train_pool_size": len(self.query_ids),
            "unique_queries_seen": int(np.count_nonzero(self.presentations)),
            "total_query_presentations": int(self.presentations.sum()),
            "optimizer_steps": self.optimizer_steps,
            "epoch": self.epoch, "cursor": self.cursor, "seed": self.seed,
            "batch_size": self.batch_size, "pool_fingerprint": self.pool_fingerprint,
            "semantics": "continuous_shuffled_epochs_keep_partial_batch",
        }

    def state_dict(self):
        return {**self.summary(), "schema_version": 1,
                "query_presentations": self.presentations.tolist(),
                "order": self.order.tolist()}

    def load_state_dict(self, state):
        if (state.get("schema_version") != 1 or state["pool_fingerprint"] != self.pool_fingerprint
                or state["seed"] != self.seed or state["batch_size"] != self.batch_size):
            raise ValueError("Incompatible training stream state")
        n = len(self.query_ids)
        order = np.asarray(state["order"], dtype=int)
        counts = np.asarray(state["query_presentations"], dtype=np.int64)
        if (order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n))
                or counts.shape != (n,) or np.any(counts < 0)
                or not 0 <= state["cursor"] <= n or state["epoch"] < 0
                or state["optimizer_steps"] < 0):
            raise ValueError("Invalid training stream state")
        self.order, self.presentations = order.copy(), counts.copy()
        self.epoch, self.cursor = int(state["epoch"]), int(state["cursor"])
        self.optimizer_steps = int(state["optimizer_steps"])


def capture_retrieval_training_state(model, optimizer, stream, *, candidate_encoder_fingerprint=None):
    """Reusable same-initialization/budget state, including dropout RNG.

    Frozen base/candidate weights are managed by the evaluator. Restore only into
    the same base model and candidate snapshot; this is not a graph-RL resume.
    """
    import torch
    tensors = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    fingerprint = sha256()
    for n, value in sorted(tensors.items()):
        _hash_tensor(fingerprint, n, value)
    numpy_rng = np.random.get_state()
    return {
        "schema_version": 1, "trainable_parameters": tensors,
        "candidate_encoder_fingerprint": candidate_encoder_fingerprint,
        "trainable_fingerprint": fingerprint.hexdigest(),
        "buffers": {n: b.detach().cpu().clone() for n, b in model.named_buffers()},
        "optimizer": deepcopy(optimizer.state_dict()), "stream": stream.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_retrieval_training_state(model, optimizer, stream, state, *, candidate_encoder_fingerprint=None):
    import torch
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())
    if (state.get("schema_version") != 1
            or state.get("candidate_encoder_fingerprint") != candidate_encoder_fingerprint
            or params.keys() != state["trainable_parameters"].keys()
            or buffers.keys() != state["buffers"].keys()):
        raise ValueError("Incompatible retrieval training state")
    for dest, source in ((params, state["trainable_parameters"]), (buffers, state["buffers"])):
        if any(dest[n].shape != source[n].shape for n in dest):
            raise ValueError("Training state tensor shape mismatch")
    stream.load_state_dict(state["stream"])
    with torch.no_grad():
        for n, value in state["trainable_parameters"].items():
            params[n].copy_(value)
        for n, value in state["buffers"].items():
            buffers[n].copy_(value)
    optimizer.load_state_dict(deepcopy(state["optimizer"]))
    random.setstate(state["python_rng"])
    numpy_rng = state["numpy_rng"]
    np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
