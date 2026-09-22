"""Sample-conditioned graph activation and the MVP SoftGraphTokenizer."""
from __future__ import annotations

import math
import numpy as np


def bootstrap_concept_prototypes(bootstrap_embeddings):
    """Average fixed H^(b) matrices into [P, R] concept prototypes.

    Input follows the handoff convention [B, R, P].  No PCA is applied.
    """
    h = np.asarray(bootstrap_embeddings, dtype=float)
    if h.ndim != 3 or min(h.shape) < 1 or not np.isfinite(h).all():
        raise ValueError("bootstrap_embeddings must be finite [B, R, P]")
    return np.asarray(h.mean(axis=0).T, dtype=np.float32)


def error_weighted_edge_relevance(activations, losses, normalize=True):
    """q_ij^err = mean_n loss_n a_ni a_nj, with zero diagonal."""
    a = np.asarray(activations, dtype=float)
    ell = np.asarray(losses, dtype=float)
    if a.ndim != 2 or ell.shape != (len(a),) or not np.isfinite(a).all() or not np.isfinite(ell).all():
        raise ValueError("activations must be [N,P] and losses [N]")
    q = np.einsum("n,ni,nj->ij", ell, a, a, optimize=True) / max(len(a), 1)
    q = (q + q.T) / 2
    np.fill_diagonal(q, 0.0)
    if normalize:
        off = np.abs(q[np.triu_indices(a.shape[1], 1)])
        scale = float(off.max()) if len(off) else 0.0
        if scale > 0:
            q = q / scale
    return q


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SoftGraphTokenizer requires PyTorch") from exc
    return torch, nn


def smooth_concept_activation(concept_prototypes, token_embeddings, tau=0.1,
                              normalization="sigmoid", token_mask=None):
    """Compute sample-specific concept activation from token embeddings.

    concept_prototypes: [P,R]
    token_embeddings: [B,L,R]
    returns: [B,P]
    """
    torch, _ = _torch()
    if tau <= 0:
        raise ValueError("tau must be positive")
    c = concept_prototypes
    e = token_embeddings
    if c.ndim != 2 or e.ndim != 3 or c.shape[-1] != e.shape[-1]:
        raise ValueError("Expected concept prototypes [P,R] and token embeddings [B,L,R]")
    c = torch.nn.functional.normalize(c, p=2, dim=-1)
    e = torch.nn.functional.normalize(e, p=2, dim=-1)
    sim = torch.einsum("blr,pr->blp", e, c)
    if token_mask is not None:
        mask = torch.as_tensor(token_mask, dtype=torch.bool, device=sim.device)
        if mask.shape != sim.shape[:2] or not torch.all(mask.any(dim=1)):
            raise ValueError("token_mask must be [B,L] with at least one active token per sample")
        sim = sim.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    pooled = tau * torch.logsumexp(sim / tau, dim=1)
    if normalization == "sigmoid":
        return torch.sigmoid(pooled)
    if normalization == "softmax":
        return torch.softmax(pooled, dim=-1)
    if normalization == "none":
        return pooled
    raise ValueError("normalization must be sigmoid, softmax, or none")


def sample_conditioned_graph(global_graph, activations):
    """W_x = D_x W D_x for a batch of activations."""
    torch, _ = _torch()
    w = global_graph
    a = activations
    if w.ndim != 2 or w.shape[0] != w.shape[1] or a.ndim != 2 or a.shape[1] != w.shape[0]:
        raise ValueError("global_graph must be [P,P] and activations [B,P]")
    return w.unsqueeze(0) * a.unsqueeze(2) * a.unsqueeze(1)


class SoftGraphTokenizerBase:
    pass


def _build_soft_graph_tokenizer_class():
    torch, nn = _torch()

    class SoftGraphTokenizer(nn.Module):
        """One weighted message-passing layer + learnable query pooling.

        Concept prototypes are registered as a frozen buffer.  The trainable
        parameters are the message projection, node MLP, query vectors and final
        projection into the Qwen hidden dimension.
        """
        def __init__(self, concept_prototypes, output_dim, num_tokens=8,
                     graph_hidden_dim=128, activation_tau=0.1,
                     activation_normalization="sigmoid"):
            super().__init__()
            c = torch.as_tensor(concept_prototypes, dtype=torch.float32)
            if c.ndim != 2 or c.shape[0] < 2 or c.shape[1] < 1:
                raise ValueError("concept_prototypes must be [P,R]")
            if not isinstance(num_tokens, int) or num_tokens < 1:
                raise ValueError("num_tokens must be positive")
            if not isinstance(output_dim, int) or output_dim < 1:
                raise ValueError("output_dim must be positive")
            if not isinstance(graph_hidden_dim, int) or graph_hidden_dim < 1:
                raise ValueError("graph_hidden_dim must be positive")
            self.register_buffer("concept_prototypes", c, persistent=True)
            self.num_concepts, self.representation_dim = c.shape
            self.output_dim = int(output_dim)
            self.num_tokens = int(num_tokens)
            self.activation_tau = float(activation_tau)
            self.activation_normalization = activation_normalization
            self.message_projection = nn.Linear(self.representation_dim, graph_hidden_dim, bias=False)
            self.node_mlp = nn.Sequential(
                nn.Linear(self.representation_dim + graph_hidden_dim + 1, graph_hidden_dim),
                nn.GELU(),
                nn.Linear(graph_hidden_dim, graph_hidden_dim),
            )
            self.queries = nn.Parameter(torch.empty(num_tokens, graph_hidden_dim))
            nn.init.normal_(self.queries, mean=0.0, std=1 / math.sqrt(graph_hidden_dim))
            self.output_projection = nn.Linear(graph_hidden_dim, output_dim)

        def activations(self, token_embeddings, token_mask=None):
            c = self.concept_prototypes.to(dtype=token_embeddings.dtype,
                                           device=token_embeddings.device)
            return smooth_concept_activation(c, token_embeddings,
                                             self.activation_tau,
                                             self.activation_normalization,
                                             token_mask=token_mask)

        def forward(self, token_embeddings, global_graph, return_aux=False, token_mask=None):
            if token_embeddings.ndim != 3 or token_embeddings.shape[-1] != self.representation_dim:
                raise ValueError("token_embeddings must be [B,L,R] matching concept prototype dimension")
            graph_value = (np.array(global_graph, copy=True)
                           if isinstance(global_graph, np.ndarray) else global_graph)
            w = torch.as_tensor(graph_value, dtype=token_embeddings.dtype,
                                device=token_embeddings.device)
            if w.shape != (self.num_concepts, self.num_concepts):
                raise ValueError("global_graph shape does not match concept axis")
            w = (w + w.T) / 2
            w = w.clone()
            w.fill_diagonal_(0)
            a = self.activations(token_embeddings, token_mask=token_mask)
            wx = sample_conditioned_graph(w, a)
            c = self.concept_prototypes.to(dtype=token_embeddings.dtype,
                                           device=token_embeddings.device)
            projected = self.message_projection(c)  # [P,H]
            message = torch.einsum("bij,jh->bih", wx, projected)
            activated_c = a.unsqueeze(-1) * c.unsqueeze(0)
            node_input = torch.cat((activated_c, message, a.unsqueeze(-1)), dim=-1)
            z = self.node_mlp(node_input)  # [B,P,H]
            q = self.queries.to(dtype=z.dtype)
            scores = torch.einsum("kh,bph->bkp", q, z) / math.sqrt(z.shape[-1])
            weights = torch.softmax(scores, dim=-1)
            pooled = torch.einsum("bkp,bph->bkh", weights, z)
            tokens = self.output_projection(pooled)
            if return_aux:
                return tokens, {"activations": a, "sample_graph": wx,
                                "attention_weights": weights, "node_states": z}
            return tokens

    return SoftGraphTokenizer


# Create the nn.Module subclass lazily enough to keep import errors explicit but
# still expose a normal class to users when torch is installed.
try:  # pragma: no cover - import branch depends on installation
    SoftGraphTokenizer = _build_soft_graph_tokenizer_class()
except RuntimeError:  # pragma: no cover
    SoftGraphTokenizer = SoftGraphTokenizerBase
