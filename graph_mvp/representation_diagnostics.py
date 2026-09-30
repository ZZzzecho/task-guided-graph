"""Bounded A-F collapse diagnostics using the production attention/projection."""
from __future__ import annotations

import numpy as np

from .estimators import _rank_gaussian_tensor


def distribution(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    finite = x[np.isfinite(x)]
    return {"count": int(len(finite)), "undefined_count": int(len(x) - len(finite)),
            "mean": float(finite.mean()) if len(finite) else None,
            "quantiles": dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                                  np.quantile(finite, [0, .05, .25, .5, .75, .95, 1]).tolist()))
                         if len(finite) else None}


def spectrum(matrix):
    eig = np.linalg.eigvalsh((matrix + matrix.T) / 2)[::-1]
    positive = np.maximum(eig, 0)
    trace = float(positive.sum())
    weights = positive[positive > 0] / trace if trace > 0 else np.array([])
    return {"eigenvalues_descending": eig.tolist(), "trace": trace,
            "lambda1_over_trace": float(positive[0] / trace) if trace > 0 else None,
            "effective_rank_entropy": float(np.exp(-np.sum(weights * np.log(weights)))) if trace > 0 else 0.,
            "effective_rank_participation": float(1 / np.sum(weights ** 2)) if trace > 0 else 0.,
            "numerical_rank": int(np.count_nonzero(positive > (positive[0] * 1e-10 if trace > 0 else 0)))}


def correlation(matrix):
    diagonal = np.maximum(np.diag(matrix), 0)
    denom = np.sqrt(np.outer(diagonal, diagonal))
    out = np.full_like(matrix, np.nan)
    np.divide(matrix, denom, out=out, where=denom > 1e-15)
    return np.clip(out, -1, 1)


class ConceptStageAccumulator:
    """Retain P-by-P sufficient statistics, not N-by-1024-by-P states.

    Raw Gram: sum_d H_d.T H_d / (N R), matching MNGM B=I before
    regularization. Sample covariance centers each (r,c) over documents.
    Per-document cosines use a fixed seeded sample of concept pairs for
    quantiles, and an exact all-pair mean computed in O(R P).
    """

    def __init__(self, concepts, *, pairs=4096, seed=17, sample_centering=True):
        self.p = int(concepts)
        if self.p < 2 or pairs < 1:
            raise ValueError("Need >=2 concepts and >=1 diagnostic pairs")
        first, second = np.triu_indices(self.p, 1)
        take = np.random.default_rng(seed).choice(len(first), min(pairs, len(first)), replace=False)
        self.first, self.second = first[take], second[take]
        self.cross = np.zeros((self.p, self.p), dtype=np.float64)
        self.sum_x = None
        self.sample_centering = sample_centering
        self.samples = self.rows = 0
        self.cosines = []
        self.exact_means = []
        self.zero_norm_concept_vectors = 0

    def add(self, samples):
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim != 3 or x.shape[2] != self.p or min(x.shape[:2]) < 1 or not np.isfinite(x).all():
            raise ValueError("Stage must be finite nonempty [N,R,P]")
        flat = x.reshape(-1, self.p)
        self.cross += flat.T @ flat
        if self.sample_centering:
            if self.sum_x is None:
                self.sum_x = np.zeros(x.shape[1:])
            if self.sum_x.shape != x.shape[1:]:
                raise ValueError("Representation axis changed during accumulation")
            self.sum_x += x.sum(axis=0)
        for row in x:
            norms = np.linalg.norm(row, axis=0)
            active = norms > 1e-15
            self.zero_norm_concept_vectors += int(np.count_nonzero(~active))
            unit = row / np.maximum(norms, 1e-15)
            count = int(active.sum())
            summed = unit[:, active].sum(axis=1)
            self.exact_means.append((float(summed @ summed) - count) / (count * (count - 1))
                                    if count > 1 else np.nan)
            vals = np.empty(len(self.first))
            for start in range(0, len(vals), 256):
                a, b = self.first[start:start+256], self.second[start:start+256]
                vals[start:start+256] = np.einsum("ra,ra->a", unit[:, a], unit[:, b])
            vals[~(active[self.first] & active[self.second])] = np.nan
            self.cosines.append(vals)
        self.samples += len(x)
        self.rows += len(flat)

    def report(self):
        if not self.samples:
            raise ValueError("No diagnostic samples")
        gram = self.cross / self.rows
        upper = np.triu_indices(self.p, 1)
        out = {
            "n_samples": self.samples, "num_concepts": self.p,
            "sampled_pairs_per_document": len(self.first),
            "pairwise_document_cosine": distribution(np.concatenate(self.cosines)),
            "all_pair_document_cosine_mean": distribution(self.exact_means)["mean"],
            "zero_norm_concept_vectors": self.zero_norm_concept_vectors,
            "concept_gram": spectrum(gram),
            "pairwise_aggregate_cosine": distribution(correlation(gram)[upper]),
        }
        if self.sum_x is not None and self.samples > 1:
            cov = (self.cross - self.sum_x.T @ self.sum_x / self.samples) / self.rows
            out["sample_axis_centered_covariance"] = spectrum(cov)
            out["pairwise_sample_axis_correlation"] = distribution(correlation(cov)[upper])
        return out


def diagnose_representations(builder, text_batches, *, seed=17, pairs=4096,
                             cache_dtype="float16", max_samples=512):
    """Stream encoder states; keep only bounded post-L2 matrices for rank transform.

    text_batches yields batches of training texts. Encoder outputs retain their
    native dtype for the same arithmetic as cache building. Summaries use FP64.
    Rank-Gaussian is the production transform, after the specified cache cast.
    """
    import torch
    if cache_dtype not in ("float16", "float32") or not 2 <= max_samples <= 512:
        raise ValueError("Use float16/float32 cache dtype and 2..512 samples")
    if not builder.use_projection:
        raise ValueError("A-F diagnostics require a projection to expose pre/post-L2 stages")
    p = builder.num_concepts
    proto = ConceptStageAccumulator(p, seed=seed, pairs=pairs, sample_centering=False)
    proto.add(builder.prototypes.detach().cpu().numpy().T[None])
    attention = ConceptStageAccumulator(p, seed=seed, pairs=pairs, sample_centering=False)
    stages = {name: ConceptStageAccumulator(p, seed=seed, pairs=pairs)
              for name in ("C_pre_pca", "D_post_pca_pre_l2", "E_post_l2", "E_cache_quantized", "F_rank_gaussian")}
    entropies, normalized_entropies, post_l2_chunks = [], [], []
    n = 0
    native_dtypes = set()
    for texts in text_batches:
        texts = list(texts)
        if not texts:
            continue
        if n + len(texts) > max_samples:
            raise ValueError("Diagnostic input exceeds max_samples; subset before encoding")
        with torch.inference_mode():
            z, mask = builder.encoder.encode_token_states(texts)
            h, alpha, intermediate = builder.from_hidden_states(z, mask, return_stages=True)
        native_dtypes.add(str(z.dtype))
        for i in range(len(texts)):
            a = alpha[i, mask[i].bool()].detach().float().cpu().numpy()
            attention.add(a[None])
            entropy = -(a * np.log(np.maximum(a, 1e-30))).sum(axis=0)
            entropies.append(entropy)
            normalized_entropies.append(entropy / np.log(len(a)) if len(a) > 1 else np.zeros_like(entropy))
        for name, key in (("C_pre_pca", "h_full"), ("D_post_pca_pre_l2", "post_projection_pre_l2"),
                          ("E_post_l2", "post_l2")):
            stages[name].add(intermediate[key].detach().float().cpu().numpy())
        quantized = h.detach().float().cpu().numpy().astype(cache_dtype).astype(np.float32)
        stages["E_cache_quantized"].add(quantized)
        post_l2_chunks.append(quantized)
        n += len(texts)
    if n < 2:
        raise ValueError("Need at least two documents for sample-axis diagnostics")
    ranked = _rank_gaussian_tensor(np.concatenate(post_l2_chunks, axis=0))
    # Chunk the summary pass to avoid duplicate large tensor workspaces.
    for start in range(0, n, 4):
        stages["F_rank_gaussian"].add(ranked[start:start+4])
    attention_report = attention.report()
    attention_report.update({"entropy_nats": distribution(np.concatenate(entropies)),
                             "entropy_divided_by_log_active_tokens": distribution(np.concatenate(normalized_entropies)),
                             "sample_axis_covariance": "not_defined_for_variable_token_positions"})
    return {
        "schema_version": 1, "n_documents": n, "seed": seed, "pairs_requested": pairs,
        "encoder_native_dtypes": sorted(native_dtypes), "rank_gaussian_input_cache_dtype": cache_dtype,
        "rank_gaussian_axis": "documents_for_each_fixed_representation_concept_cell",
        "covariance_definition": "sample-axis-centered sum_d H_d.T H_d / (N R); B=I; no ridge",
        "effective_rank_definitions": {"entropy": "exp(-sum p_i log p_i)", "participation": "1/sum p_i^2"},
        "projection": builder.projection_metadata(),
        "stages": {"A_prototypes": proto.report(), "B_attention": attention_report,
                   **{name: stage.report() for name, stage in stages.items()}},
    }
