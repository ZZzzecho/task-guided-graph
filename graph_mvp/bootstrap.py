"""Utilities for fixed Bootstrap-MNGM representation snapshots."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
import numpy as np



def split_sentences_regex(text: str):
    """Exact sentence splitter preserved from legacy_reference/NMF_PK3rd.py."""
    return re.split(r"(?<=[.!?，。;])\s+", str(text).strip())


def shuffle_sentences(sentences, rng: random.Random):
    values = list(sentences)
    rng.shuffle(values)
    return values


def make_shuffled_versions_csv(train_csv, out_csv, text_col="medical_abstract",
                               n_versions=50, seed=123):
    """Legacy-compatible document-local sentence-order perturbations.

    Version 0 is the unshuffled text; versions 1..n_versions use independent
    seeds `seed+i+1`, exactly matching the preserved prototype.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CSV perturbation generation requires pandas") from exc
    if n_versions < 1:
        raise ValueError("n_versions must be positive")
    df = pd.read_csv(train_csv)
    if text_col not in df.columns:
        raise ValueError(f"Missing text column {text_col!r}")
    splittext = df[text_col].astype(str).apply(split_sentences_regex)
    out = {"version_0": splittext.apply(lambda xs: " ".join(x for x in xs if x))}
    for i in range(n_versions):
        rng = random.Random(seed + i + 1)
        shuffled = splittext.apply(lambda xs: shuffle_sentences(xs, rng))
        out[f"version_{i + 1}"] = shuffled.apply(lambda xs: " ".join(x for x in xs if x))
    result = pd.DataFrame(out)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    return result


def extract_concept_embedding_matrix(model, tokenizer, concepts, device=None):
    """Extract effective concept embeddings through the embedding module forward.

    This intentionally does *not* read `.weight` directly.  If PEFT/LoRA wraps the
    input embedding module, calling the module forward is the safer way to include
    the active adapter contribution.  Each concept is tokenized independently and
    its sub-token vectors are averaged.  Returns H with shape [R,P].
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Embedding extraction requires PyTorch") from exc
    concepts = tuple(concepts)
    if not concepts or any(not isinstance(c, str) or not c for c in concepts):
        raise ValueError("concepts must be nonempty strings")
    embedding = model.get_input_embeddings()
    if embedding is None:
        raise ValueError("model.get_input_embeddings() returned None")
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    columns = []
    with torch.no_grad():
        for concept in concepts:
            encoded = tokenizer(concept, add_special_tokens=False, return_tensors="pt")
            ids = encoded["input_ids"].to(device)
            if ids.numel() == 0:
                raise ValueError(f"Concept tokenized to zero tokens: {concept!r}")
            vec = embedding(ids).mean(dim=1).squeeze(0)
            columns.append(vec.detach().float().cpu().numpy())
    return np.stack(columns, axis=1)


def embedding_forward_weight_gap(model, token_ids):
    """Sanity diagnostic: compare module-forward output with raw `.weight` lookup.

    A nonzero gap is expected when an embedding adapter affects forward without
    being merged into the underlying weight.  Returns max and mean absolute gaps.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Embedding sanity check requires PyTorch") from exc
    embedding = model.get_input_embeddings()
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    ids = ids.to(device)
    with torch.no_grad():
        forward = embedding(ids)
        raw = embedding.weight[ids]
        gap = (forward - raw).abs()
    return {"max_abs_gap": float(gap.max().cpu()),
            "mean_abs_gap": float(gap.mean().cpu())}


def save_bootstrap_bundle(path, embeddings, concept_ids, *, model_id=None,
                          adapter_id=None, version_ids=None, metadata=None):
    """Persist fixed H^(1:B) together with concept-axis/version metadata."""
    h = np.asarray(embeddings, dtype=np.float32)
    ids = tuple(concept_ids)
    if h.ndim != 3 or h.shape[2] != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("embeddings must be [B,R,P] aligned with unique concept_ids")
    if not np.isfinite(h).all():
        raise ValueError("embeddings contain nonfinite values")
    if version_ids is None:
        version_ids = tuple(f"bootstrap-{i:02d}" for i in range(h.shape[0]))
    version_ids = tuple(version_ids)
    if len(version_ids) != h.shape[0] or len(set(version_ids)) != len(version_ids):
        raise ValueError("version_ids must uniquely identify every bootstrap matrix")
    meta = {
        "shape_semantics": ["bootstrap_version", "representation_dim", "concept"],
        "concept_ids": ids,
        "version_ids": version_ids,
        "model_id": model_id,
        "adapter_id": adapter_id,
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, H=h, metadata_json=json.dumps(meta, ensure_ascii=False))
    return path


def load_bootstrap_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        h = np.asarray(data["H"], dtype=np.float32)
        meta = json.loads(str(data["metadata_json"]))
    if h.ndim != 3 or h.shape[2] != len(meta["concept_ids"]):
        raise ValueError("Corrupt bootstrap bundle")
    return h, meta
