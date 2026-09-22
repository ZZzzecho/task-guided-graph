"""Frozen patient-specific concept matrices for Patient-MNGM.

The statistical encoder is intentionally independent from the downstream task LLM.
Qwen3-Embedding-0.6B supplies:
- official last-token pooled concept prototypes [P,1024];
- final-layer contextual patient token states [B,L,1024].

Concept-conditioned token attention produces H_d [1024,P].  H_d is cached once and
never refreshed during Graph-RL.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_PATIENT_ENCODER = "Qwen/Qwen3-Embedding-0.6B"
CACHE_SCHEMA_VERSION = 1


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Patient representation code requires PyTorch") from exc
    return torch, F


def _transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Qwen3-Embedding requires transformers>=4.51; install `pip install -e '.[patient]'`"
        ) from exc
    return AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states, attention_mask):
    """Official Qwen3-Embedding pooling rule.

    Works for left or right padding and returns one vector per sequence.
    """
    torch, _ = _torch()
    if last_hidden_states.ndim != 3 or attention_mask.shape != last_hidden_states.shape[:2]:
        raise ValueError("last_hidden_states must be [B,L,D] with attention_mask [B,L]")
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    if torch.any(sequence_lengths < 0):
        raise ValueError("Every sequence must contain at least one non-padding token")
    batch = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch, sequence_lengths]


def _resolve_dtype(torch, name):
    if name in (None, "auto"):
        return "auto"
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError("dtype must be auto, float32, float16, or bfloat16")
    return table[name]


class Qwen3EmbeddingEncoder:
    """Frozen local Qwen3 embedding encoder with pooled + token-state APIs."""

    def __init__(self, model_name_or_path=DEFAULT_PATIENT_ENCODER, *, device=None,
                 dtype="auto", local_files_only=False, max_length=512,
                 attn_implementation=None):
        torch, F = _torch()
        AutoModel, AutoTokenizer = _transformers()
        if not isinstance(max_length, int) or max_length < 8:
            raise ValueError("max_length must be an integer >= 8")
        self.model_name_or_path = str(model_name_or_path)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side="left",
            local_files_only=bool(local_files_only),
        )
        kwargs = {"local_files_only": bool(local_files_only)}
        resolved = _resolve_dtype(torch, dtype)
        if resolved != "auto":
            kwargs["torch_dtype"] = resolved
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if device is None:
            kwargs["device_map"] = "auto"
        self.model = AutoModel.from_pretrained(self.model_name_or_path, **kwargs)
        if device is not None:
            self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self._F = F

    @property
    def device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover
            return _torch()[0].device("cpu")

    @property
    def hidden_size(self):
        value = getattr(self.model.config, "hidden_size", None)
        if value is None:
            raise ValueError("Embedding model config has no hidden_size")
        return int(value)

    def _batch(self, texts):
        texts = [str(x) for x in texts]
        if not texts:
            raise ValueError("texts must be nonempty")
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    def encode_token_states(self, texts):
        """Return final contextual states [B,L,D] and attention mask [B,L]."""
        torch, _ = _torch()
        batch = self._batch(texts)
        with torch.inference_mode():
            output = self.model(**batch, return_dict=True)
        hidden = output.last_hidden_state
        mask = batch["attention_mask"]
        return hidden, mask

    def encode_pooled(self, texts, normalize=True):
        """Return official Qwen3 last-token pooled sequence embeddings [B,D]."""
        torch, F = _torch()
        hidden, mask = self.encode_token_states(texts)
        pooled = last_token_pool(hidden, mask)
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled


@dataclass(frozen=True)
class ConceptVocabulary:
    concept_ids: tuple[str, ...]
    concept_texts: tuple[str, ...]

    def __post_init__(self):
        ids = tuple(str(x) for x in self.concept_ids)
        texts = tuple(str(x) for x in self.concept_texts)
        if len(ids) < 2 or len(ids) != len(texts) or len(set(ids)) != len(ids):
            raise ValueError("Concept vocabulary needs >=2 unique IDs aligned to texts")
        if any(not x.strip() for x in texts):
            raise ValueError("concept_texts must be nonempty")
        object.__setattr__(self, "concept_ids", ids)
        object.__setattr__(self, "concept_texts", texts)


def load_concept_vocabulary(path):
    """Load JSON or CSV concept vocabulary.

    Supported JSON:
      [{"concept_id": "c1", "concept_text": "heart failure"}, ...]
      {"c1": "heart failure", ...}
    Supported CSV columns: concept_id, concept_text.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            items = [(str(k), str(v)) for k, v in raw.items()]
        elif isinstance(raw, list):
            items = [(str(x["concept_id"]), str(x["concept_text"])) for x in raw]
        else:
            raise ValueError("Unsupported concept JSON format")
    elif path.suffix.lower() == ".csv":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("CSV vocabulary loading requires pandas") from exc
        df = pd.read_csv(path)
        required = {"concept_id", "concept_text"}
        if not required.issubset(df.columns):
            raise ValueError("Concept CSV must contain concept_id, concept_text")
        items = list(zip(df["concept_id"].astype(str), df["concept_text"].astype(str)))
    else:
        raise ValueError("Concept vocabulary must be .json or .csv")
    return ConceptVocabulary(tuple(x[0] for x in items), tuple(x[1] for x in items))


def save_concept_prototypes(path, prototypes, vocabulary: ConceptVocabulary, *, encoder_id,
                            metadata=None):
    p = np.asarray(prototypes, dtype=np.float32)
    if p.ndim != 2 or p.shape[0] != len(vocabulary.concept_ids) or not np.isfinite(p).all():
        raise ValueError("prototypes must be finite [P,D] aligned with vocabulary")
    meta = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "concept_ids": list(vocabulary.concept_ids),
        "concept_texts": list(vocabulary.concept_texts),
        "encoder_id": str(encoder_id),
        "hidden_size": int(p.shape[1]),
        "pooling": "qwen3_official_last_token_normalized",
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prototypes=p, metadata_json=json.dumps(meta, ensure_ascii=False))
    return path


def load_concept_prototypes(path):
    with np.load(path, allow_pickle=False) as raw:
        p = np.asarray(raw["prototypes"], dtype=np.float32)
        meta = json.loads(str(raw["metadata_json"]))
    if p.ndim != 2 or p.shape[0] != len(meta["concept_ids"]):
        raise ValueError("Corrupt concept prototype cache")
    return p, meta


def build_concept_prototypes(encoder: Qwen3EmbeddingEncoder, vocabulary: ConceptVocabulary,
                             batch_size=64):
    torch, _ = _torch()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = []
    for start in range(0, len(vocabulary.concept_texts), batch_size):
        pooled = encoder.encode_pooled(vocabulary.concept_texts[start:start + batch_size])
        values.append(pooled.detach().float().cpu())
    return torch.cat(values, dim=0).numpy()


class PatientConceptMatrixBuilder:
    """Vectorized cosine + token-softmax builder for H_d [D,P]."""

    def __init__(self, encoder, concept_prototypes, *, temperature=0.1):
        torch, F = _torch()
        p = torch.as_tensor(concept_prototypes, dtype=torch.float32)
        if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 2 or not torch.isfinite(p).all():
            raise ValueError("concept_prototypes must be finite [P,D]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if hasattr(encoder, "hidden_size") and int(encoder.hidden_size) != int(p.shape[1]):
            raise ValueError("Prototype dimension does not match encoder hidden size")
        self.encoder = encoder
        self.prototypes = p
        self.temperature = float(temperature)
        self._F = F

    @property
    def num_concepts(self):
        return int(self.prototypes.shape[0])

    @property
    def hidden_size(self):
        return int(self.prototypes.shape[1])

    def from_hidden_states(self, token_states, attention_mask):
        """Build [B,D,P] matrices from final token states [B,L,D]."""
        torch, F = _torch()
        z = token_states
        mask = torch.as_tensor(attention_mask, dtype=torch.bool, device=z.device)
        if z.ndim != 3 or z.shape[-1] != self.hidden_size or mask.shape != z.shape[:2]:
            raise ValueError("token_states [B,L,D] / attention_mask [B,L] shape mismatch")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Each patient must contain at least one active token")
        proto = self.prototypes.to(device=z.device, dtype=z.dtype)
        z_norm = F.normalize(z, p=2, dim=-1)
        p_norm = F.normalize(proto, p=2, dim=-1)
        sim = torch.einsum("bld,pd->blp", z_norm, p_norm)
        sim = sim.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(sim / self.temperature, dim=1)
        # Weighted sum of original contextual states; output [B,D,P].
        h = torch.einsum("blp,bld->bdp", alpha, z)
        return h, alpha

    def encode(self, texts):
        torch, _ = _torch()
        with torch.inference_mode():
            z, mask = self.encoder.encode_token_states(texts)
            h, alpha = self.from_hidden_states(z, mask)
        return h, alpha, mask


def _jsonable_id(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value.item() if hasattr(value, "item") else value


class PatientMatrixCacheWriter:
    """Append-only sharded .npy cache for large [N,D,P] patient matrices."""

    def __init__(self, output_dir, concept_ids: Sequence[str], hidden_size: int, *,
                 dtype="float16", encoder_id=DEFAULT_PATIENT_ENCODER, temperature=0.1,
                 overwrite=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.output_dir / "metadata.json"
        if self.meta_path.exists() and not overwrite:
            raise FileExistsError(f"Cache already exists: {self.output_dir}")
        if dtype not in ("float16", "float32"):
            raise ValueError("cache dtype must be float16 or float32")
        self.dtype = np.dtype(dtype)
        self.concept_ids = tuple(str(x) for x in concept_ids)
        self.hidden_size = int(hidden_size)
        self.encoder_id = str(encoder_id)
        self.temperature = float(temperature)
        self.shards = []
        self.n_samples = 0

    def write_shard(self, matrices, subject_ids, stay_ids):
        x = np.asarray(matrices)
        if x.ndim != 3 or x.shape[1:] != (self.hidden_size, len(self.concept_ids)):
            raise ValueError("matrices must be [B,D,P] matching cache metadata")
        if len(subject_ids) != len(x) or len(stay_ids) != len(x):
            raise ValueError("ID arrays must match matrices batch")
        if not np.isfinite(x).all():
            raise ValueError("Patient matrices contain nonfinite values")
        idx = len(self.shards)
        matrix_name = f"shard_{idx:05d}.npy"
        records_name = f"shard_{idx:05d}.records.json"
        np.save(self.output_dir / matrix_name, x.astype(self.dtype, copy=False), allow_pickle=False)
        records = [
            {"subject_id": _jsonable_id(s), "stay_id": _jsonable_id(t)}
            for s, t in zip(subject_ids, stay_ids)
        ]
        (self.output_dir / records_name).write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8")
        self.shards.append({"matrix": matrix_name, "records": records_name, "n_samples": len(x)})
        self.n_samples += len(x)
        self._flush_metadata()

    def _flush_metadata(self):
        meta = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "mode": "patient_concept_matrix",
            "shape_semantics": ["patient", "representation_dim", "concept"],
            "n_samples": self.n_samples,
            "hidden_size": self.hidden_size,
            "num_concepts": len(self.concept_ids),
            "concept_ids": list(self.concept_ids),
            "dtype": self.dtype.name,
            "encoder_id": self.encoder_id,
            "temperature": self.temperature,
            "frozen_during_graph_rl": True,
            "source_split": "train",
            "shards": self.shards,
        }
        self.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def close(self):
        self._flush_metadata()
        return self.meta_path


class PatientMatrixDataset:
    """Lazy/memmap reader for sharded Patient-MNGM matrices."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.metadata = json.loads((self.cache_dir / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("mode") != "patient_concept_matrix":
            raise ValueError("Not a patient_concept_matrix cache")
        self.concept_ids = tuple(self.metadata["concept_ids"])
        self.hidden_size = int(self.metadata["hidden_size"])
        self.num_concepts = int(self.metadata["num_concepts"])
        self.n_samples = int(self.metadata["n_samples"])
        self.shards = tuple(self.metadata["shards"])
        if self.num_concepts != len(self.concept_ids) or self.n_samples < 1:
            raise ValueError("Corrupt patient cache metadata")

    def __len__(self):
        return self.n_samples

    def iter_shards(self, mmap_mode="r"):
        for item in self.shards:
            arr = np.load(self.cache_dir / item["matrix"], mmap_mode=mmap_mode, allow_pickle=False)
            expected = (int(item["n_samples"]), self.hidden_size, self.num_concepts)
            if arr.shape != expected:
                raise ValueError(f"Shard shape mismatch: {item['matrix']}")
            records = json.loads((self.cache_dir / item["records"]).read_text(encoding="utf-8"))
            if len(records) != len(arr):
                raise ValueError(f"Shard ID count mismatch: {item['records']}")
            yield arr, records

    def materialize(self, max_samples=None, dtype=np.float32):
        """Materialize a deterministic prefix for the current in-memory MNGM solver."""
        if max_samples is not None and (not isinstance(max_samples, int) or max_samples < 2):
            raise ValueError("max_samples must be None or an integer >=2")
        target = self.n_samples if max_samples is None else min(self.n_samples, max_samples)
        chunks = []
        total = 0
        for arr, _ in self.iter_shards("r"):
            take = min(len(arr), target - total)
            if take > 0:
                chunks.append(np.asarray(arr[:take], dtype=dtype))
                total += take
            if total >= target:
                break
        if total < 2:
            raise ValueError("Need at least two patient matrices")
        return np.concatenate(chunks, axis=0)

    def fingerprint(self):
        h = sha256()
        meta = dict(self.metadata)
        h.update(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        for item in self.shards:
            path = self.cache_dir / item["matrix"]
            h.update(str((path.name, path.stat().st_size)).encode())
        return h.hexdigest()


def build_patient_cache_from_frame(frame, builder: PatientConceptMatrixBuilder, output_dir,
                                   concept_ids, *, batch_size=8, shard_size=32,
                                   cache_dtype="float16", text_col="text",
                                   subject_col="subject_id", stay_col="stay_id",
                                   encoder_id=DEFAULT_PATIENT_ENCODER, overwrite=False):
    """Build a sharded cache from a train DataFrame without ever storing all H in RAM."""
    required = {text_col, subject_col, stay_col}
    if not required.issubset(frame.columns):
        raise ValueError(f"Patient frame is missing columns: {sorted(required - set(frame.columns))}")
    if batch_size < 1 or shard_size < 1:
        raise ValueError("batch_size and shard_size must be positive")
    writer = PatientMatrixCacheWriter(
        output_dir, concept_ids, builder.hidden_size, dtype=cache_dtype,
        encoder_id=encoder_id, temperature=builder.temperature, overwrite=overwrite)
    matrix_buffer, subject_buffer, stay_buffer = [], [], []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start:start + batch_size]
        h, _, _ = builder.encode(batch[text_col].astype(str).tolist())
        h = h.detach().float().cpu().numpy()
        for i in range(len(batch)):
            matrix_buffer.append(h[i])
            subject_buffer.append(batch.iloc[i][subject_col])
            stay_buffer.append(batch.iloc[i][stay_col])
            if len(matrix_buffer) >= shard_size:
                writer.write_shard(np.stack(matrix_buffer), subject_buffer, stay_buffer)
                matrix_buffer, subject_buffer, stay_buffer = [], [], []
    if matrix_buffer:
        writer.write_shard(np.stack(matrix_buffer), subject_buffer, stay_buffer)
    writer.close()
    return PatientMatrixDataset(output_dir)
