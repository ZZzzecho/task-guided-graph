"""Direct document-conditioned concepts: H_d[:, c] = W F(c, evidence_dc).

No residual subtraction, score gating, prototype addition, or output L2.
The frozen encoder and projection are shared across all documents/concepts.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re

import numpy as np

from .patient_repr import PatientConceptMatrixBuilder


JOINT_TEMPLATE = "Concept:\n{concept}\n\nPatent excerpts:\n{evidence}"


def token_count(tokenizer, text, *, special=True):
    return len(tokenizer.encode(text, add_special_tokens=special))


def format_joint_input(concept, evidence):
    return JOINT_TEMPLATE.format(concept=concept, evidence=evidence)


def _fitting_prefix(text, fits):
    """Find a fitting original-text prefix; verify rather than assume BPE counts."""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(text[:mid]):
            lo = mid
        else:
            hi = mid - 1
    prefix = text[:lo].rstrip()
    if not prefix or not fits(prefix):
        return ""
    return prefix


def split_document(text, tokenizer, max_chunk_tokens=128):
    """Sentence/newline heuristic, token-bounded long sentences, original spans.

    This is not a linguistic sentence parser. No chunk-count cap drops a suffix.
    Token budgets include special tokens so independent chunk encoding is safe.
    """
    if max_chunk_tokens < 8 or not isinstance(text, str) or not text.strip():
        raise ValueError("Use nonempty document text and >=8 chunk tokens")
    units = []
    for match in re.finditer(r"[^.!?。！？\n]+[.!?。！？]*|[.!?。！？]+", text):
        start, end = match.span()
        while start < end:
            while start < end and text[start].isspace():
                start += 1
            if start == end:
                break
            tail = text[start:end].rstrip()
            prefix = (tail if token_count(tokenizer, tail) <= max_chunk_tokens else
                      _fitting_prefix(tail, lambda x: token_count(tokenizer, x) <= max_chunk_tokens))
            if not prefix:
                raise ValueError("Chunk budget cannot fit a nonempty source prefix")
            stop = start + len(prefix)
            units.append({"index": len(units), "text": prefix, "start": start, "end": stop,
                          "tokens": token_count(tokenizer, prefix)})
            start = stop
    if not units:
        raise ValueError("No nonempty evidence chunks")
    return units


def select_and_pack_evidence(concept, chunks, scores, tokenizer, *, top_k=3, max_input_tokens=512):
    """Independent stable top-k, exact template budget, source-order packing.

    Scores are ranking signals, not labels/probabilities. No uncalibrated gate.
    All concept text is retained. A too-long concept fails explicitly.
    """
    scores = np.asarray(scores, dtype=float)
    if not chunks or scores.shape != (len(chunks),) or not np.isfinite(scores).all() or top_k < 1:
        raise ValueError("Finite scores must align with chunks; top_k must be positive")
    if not str(concept).strip() or token_count(tokenizer, format_joint_input(concept, "")) >= max_input_tokens:
        raise ValueError("Full concept/template must leave room for evidence")
    chosen, seen, budget_omitted = [], set(), 0
    ranked = sorted(range(len(chunks)), key=lambda j: (-scores[j], chunks[j]["index"]))
    for j in ranked:
        unit = chunks[j]
        key = " ".join(unit["text"].split())
        if key in seen:
            continue
        seen.add(key)
        def fits(prefix):
            candidate = {**unit, "text": prefix}
            parts = sorted([*chosen, candidate], key=lambda x: x["index"])
            text = "\n\n".join(x["text"] for x in parts)
            return token_count(tokenizer, format_joint_input(concept, text)) <= max_input_tokens
        prefix = unit["text"] if fits(unit["text"]) else _fitting_prefix(unit["text"], fits)
        if not prefix or (prefix != unit["text"] and token_count(tokenizer, prefix, special=False) < 8):
            budget_omitted += 1
            continue
        chosen.append({**unit, "text": prefix, "end": unit["start"] + len(prefix),
                       "score": float(scores[j]), "truncated": prefix != unit["text"]})
        if len(chosen) == top_k:
            break
    if not chosen:
        raise ValueError("Input budget cannot fit evidence alongside the full concept")
    chosen.sort(key=lambda x: x["index"])
    evidence = "\n\n".join(x["text"] for x in chosen)
    packed = format_joint_input(concept, evidence)
    return {"evidence_text": evidence, "input_text": packed, "chunks": chosen,
            "input_tokens": token_count(tokenizer, packed), "candidate_chunks": len(chunks),
            "max_score": float(scores.max()), "mean_score": float(scores.mean()),
            "truncated": any(x["truncated"] for x in chosen),
            "budget_omitted_chunks": budget_omitted,
            "selected_evidence_tokens": token_count(tokenizer, evidence, special=False),
            "selection_is_not_relevance_label": True}


class JointEvidenceMatrixBuilder(PatientConceptMatrixBuilder):
    """Drop-in cache builder for direct F(concept, evidence) matrices."""
    representation_mode = "joint_evidence"

    def __init__(self, encoder, concept_prototypes, concept_texts, *, top_k=3,
                 chunk_max_tokens=128, encode_batch_size=8, **projection_kwargs):
        super().__init__(encoder, concept_prototypes, **projection_kwargs)
        self.temperature = None  # There is no token softmax in this representation.
        self.concept_texts = tuple(str(x) for x in concept_texts)
        if len(self.concept_texts) != self.num_concepts or any(not x.strip() for x in self.concept_texts):
            raise ValueError("Concept texts must be nonempty and aligned with prototypes")
        if not hasattr(encoder, "tokenizer") or not hasattr(encoder, "max_length"):
            raise ValueError("Joint evidence requires encoder tokenizer and max_length")
        if top_k < 1 or encode_batch_size < 1 or not 8 <= chunk_max_tokens <= encoder.max_length:
            raise ValueError("Invalid top_k, encode batch size, or chunk token budget")
        import torch
        if torch.any(torch.linalg.vector_norm(self.prototypes, dim=1) == 0):
            raise ValueError("Concept prototypes must be nonzero for cosine ranking")
        self.top_k, self.chunk_max_tokens = int(top_k), int(chunk_max_tokens)
        self.encode_batch_size = int(encode_batch_size)
        self.source_projection_mean_sha256 = None
        if self.use_projection:
            self.source_projection_mean_sha256 = sha256(self.projection_mean.numpy().tobytes()).hexdigest()
            # Only the shared basis W is used. Persist the effective zero mean.
            self.projection_mean = torch.zeros(self.encoder_hidden_size, dtype=torch.float32)
        self.stats = {"documents": 0, "chunk_inputs": 0, "joint_inputs": 0,
                      "packed_truncated_pairs": 0, "budget_limited_pairs": 0, "max_joint_input_tokens": 0}
        self.native_dtypes = set()
        for concept in self.concept_texts:
            if token_count(encoder.tokenizer, format_joint_input(concept, "")) >= encoder.max_length:
                raise ValueError("Full concept/template must leave room for evidence")
        model = getattr(encoder, "model", None)
        if model is not None and any(p.requires_grad for p in model.parameters()):
            raise ValueError("Joint evidence encoder must be frozen")

    def _pooled(self, texts):
        import torch
        import torch.nn.functional as F
        values = []
        for start in range(0, len(texts), self.encode_batch_size):
            batch = texts[start:start + self.encode_batch_size]
            if any(token_count(self.encoder.tokenizer, x) > self.encoder.max_length for x in batch):
                raise ValueError("Encoding would silently truncate a packed input")
            model = getattr(self.encoder, "model", None)
            if model is not None:
                model.eval()
            with torch.inference_mode():
                # Explicit FP32 normalization on native encoder output.
                pooled = self.encoder.encode_pooled(batch, normalize=False).detach()
                self.native_dtypes.add(str(pooled.dtype))
                raw = pooled.float()
                if raw.shape != (len(batch), self.encoder_hidden_size) or not torch.isfinite(raw).all():
                    raise ValueError("Encoder must return finite [B,D] pooled vectors")
                if torch.any(torch.linalg.vector_norm(raw, dim=1) <= 1e-12):
                    raise ValueError("Encoder returned an undefined zero pooled vector")
                values.append(F.normalize(raw, dim=1))
        return torch.cat(values)

    def project_full_states(self, h_full, *, normalize=False):
        import torch
        if normalize:
            raise ValueError("Direct joint evidence has no output per-concept L2")
        if not self.use_projection:
            return h_full
        return torch.einsum("rd,bdp->brp", self.projection_matrix.to(h_full), h_full)

    def encode_document(self, text):
        import torch.nn.functional as F
        chunks = split_document(text, self.encoder.tokenizer, self.chunk_max_tokens)
        chunk_vectors = self._pooled([x["text"] for x in chunks])
        p = F.normalize(self.prototypes.to(chunk_vectors), dim=1)
        scores = (chunk_vectors @ p.T).detach().cpu().numpy()
        packed = [select_and_pack_evidence(c, chunks, scores[:, i], self.encoder.tokenizer,
                    top_k=self.top_k, max_input_tokens=self.encoder.max_length)
                  for i, c in enumerate(self.concept_texts)]
        full = self._pooled([x["input_text"] for x in packed]).T[None]
        h = self.project_full_states(full)
        self.stats["documents"] += 1
        self.stats["chunk_inputs"] += len(chunks)
        self.stats["joint_inputs"] += self.num_concepts
        self.stats["packed_truncated_pairs"] += sum(x["truncated"] for x in packed)
        self.stats["budget_limited_pairs"] += sum(x["truncated"] or x["budget_omitted_chunks"] > 0 for x in packed)
        self.stats["max_joint_input_tokens"] = max(self.stats["max_joint_input_tokens"],
                                                   max(x["input_tokens"] for x in packed))
        for i, row in enumerate(packed):
            row.update({"concept_index": i, "concept_text": self.concept_texts[i],
                        "projection_norm": float(h[0, :, i].norm())})
        return h, full, packed

    def encode(self, texts):
        import torch
        if not texts:
            raise ValueError("texts must be nonempty")
        matrices, audits = [], []
        for text in texts:
            h, _, trace = self.encode_document(text)
            matrices.append(h)
            audits.append(trace)
        return torch.cat(matrices), audits, None

    def projection_metadata(self):
        meta = super().projection_metadata()
        meta.update({"representation_mode": self.representation_mode,
            "formula": "H_d[:,c] = W @ F(concept_c, selected_evidence_dc)",
            "center_before_projection": False, "per_concept_l2_after_projection": False,
            "l2_axis": None, "joint_pooling": "qwen3_last_token_then_fp32_l2",
            "prototype_addition": False, "residual_subtraction": False, "score_gating": False,
            "attention_space": None, "attention_softmax_axis": None, "temperature": None,
            "template": JOINT_TEMPLATE, "template_sha256": sha256(JOINT_TEMPLATE.encode()).hexdigest(),
            "concept_texts_sha256": sha256(json.dumps(self.concept_texts, ensure_ascii=False).encode()).hexdigest(),
            "encoder_id": str(getattr(self.encoder, "model_name_or_path", "test_encoder")),
            "encoder_frozen": True, "max_input_tokens": self.encoder.max_length,
            "top_k": self.top_k, "chunk_max_tokens": self.chunk_max_tokens,
            "chunk_splitter": "sentence_newline_heuristic_then_token_bounded_source_prefixes",
            "selection": "independent_cosine_topk_deduplicated_stable_source_order",
            "evidence_source": "complete_supplied_document_text",
            "selection_is_not_relevance_label": True, "encode_batch_size": self.encode_batch_size,
            "source_projection_mean_sha256": self.source_projection_mean_sha256,
            "encoder_native_dtypes": sorted(self.native_dtypes),
            "pooling_normalization_dtype": "float32",
            "encoding_counts": dict(self.stats)})
        return meta
