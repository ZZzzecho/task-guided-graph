"""Concept prototypes in the downstream task-LM input-embedding space."""
from __future__ import annotations
import numpy as np


def task_lm_concept_prototypes(model, tokenizer, concept_texts, device=None):
    """Mean sub-token input embeddings [P,D] for sample activation.

    These prototypes are intentionally separate from Patient-MNGM's Qwen3-Embedding
    pooled prototypes. They must share a space with the downstream task model's input
    token embeddings (e.g. GLM-4.7-Flash).
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Task concept prototypes require PyTorch") from exc
    concepts = tuple(str(x) for x in concept_texts)
    if len(concepts) < 2 or any(not x.strip() for x in concepts):
        raise ValueError("concept_texts must contain >=2 nonempty strings")
    embedding = model.get_input_embeddings()
    if embedding is None:
        raise ValueError("task model has no input embedding module")
    if device is None:
        try:
            device = embedding.weight.device
        except AttributeError:
            device = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        for text in concepts:
            encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            ids = encoded["input_ids"].to(device)
            if ids.numel() == 0:
                raise ValueError(f"Concept tokenized to zero tokens: {text!r}")
            rows.append(embedding(ids).mean(dim=1).squeeze(0).detach().float().cpu().numpy())
    return np.stack(rows, axis=0).astype(np.float32)


def build_task_soft_graph_tokenizer(model, tokenizer, concept_texts, *, num_tokens=8,
                                    graph_hidden_dim=128, activation_tau=0.1,
                                    activation_normalization="sigmoid"):
    """Construct SoftGraphTokenizer in the task model's own embedding space."""
    from .graph_tokens import SoftGraphTokenizer
    prototypes = task_lm_concept_prototypes(model, tokenizer, concept_texts)
    output_dim = int(model.get_input_embeddings().weight.shape[-1])
    return SoftGraphTokenizer(
        prototypes,
        output_dim=output_dim,
        num_tokens=num_tokens,
        graph_hidden_dim=graph_hidden_dim,
        activation_tau=activation_tau,
        activation_normalization=activation_normalization,
    )
