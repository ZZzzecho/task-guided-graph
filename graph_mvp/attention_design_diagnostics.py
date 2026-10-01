"""Controlled attention-design probes; never used by production cache builders."""
from __future__ import annotations

import math
import re

import numpy as np

from .estimators import _rank_gaussian_tensor
from .representation_diagnostics import ConceptStageAccumulator, distribution, spectrum


def split_evidence(text, *, max_chars=500, max_units=64):
    """Sentence/newline units, split long sentences at spaces; retain source offsets.

    This deterministic heuristic is not a linguistic sentence parser. Coverage is
    reported explicitly if the bounded number of units drops a suffix.
    """
    if max_chars < 8 or max_units < 1:
        raise ValueError("Evidence limits must be >=8 chars and >=1 unit")
    units = []
    for match in re.finditer(r"[^.!?。！？\n]+[.!?。！？]*|[.!?。！？]+", text):
        start, end = match.span()
        while start < end:
            while start < end and text[start].isspace():
                start += 1
            if start == end:
                break
            stop = min(start + max_chars, end)
            if stop < end:
                space = text.rfind(" ", start, stop)
                if space > start:
                    stop = space
            while stop > start and text[stop - 1].isspace():
                stop -= 1
            if stop == start:
                break
            units.append({"text": text[start:stop], "start": start, "end": stop})
            start = stop
    if not units:
        raise ValueError("No nonempty evidence units")
    selected = units[:max_units]
    return selected, {"total_units": len(units), "encoded_units": len(selected),
                      "dropped_units": max(0, len(units) - max_units),
                      "visible_chars": len(text), "last_encoded_end": selected[-1]["end"],
                      "complete": len(selected) == len(units)}


def cosine_logits(states, prototypes):
    """FP32 scoring on the SAME cached native encoder states, not an FP32 encoder."""
    import torch.nn.functional as F
    return F.normalize(states.float(), dim=-1) @ F.normalize(prototypes.float(), dim=-1).T


def logit_profiles(logits):
    """Per-concept statistics across active evidence units [L,P]."""
    import torch
    x = logits.float()
    if x.ndim != 2 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError("Expected finite nonempty evidence logits")
    median = torch.quantile(x, .5, dim=0)
    return {"mean": x.mean(0), "std": x.std(0, unbiased=False),
            "max": x.max(0).values, "min": x.min(0).values, "median": median,
            "max_minus_median": x.max(0).values - median}


def attention_profiles(alpha):
    import torch
    a = alpha.float()
    entropy = -(a * a.clamp_min(1e-30).log()).sum(0)
    return {"entropy_nats": entropy,
            "normalized_entropy": entropy / math.log(len(a)) if len(a) > 1 else torch.zeros_like(entropy),
            "max_weight": a.max(0).values,
            "top5_mass": a.topk(min(5, len(a)), dim=0).values.sum(0)}


def relative_distance(first, second):
    """Concept-wise distance relative to the reference; zero references undefined."""
    import torch
    numerator = torch.linalg.vector_norm(first - second, dim=0)
    denom = torch.linalg.vector_norm(second, dim=0)
    return torch.where(denom > 1e-12, numerator / denom, torch.full_like(denom, float("nan")))


class TensorStageAccumulator(ConceptStageAccumulator):
    """Same statistics as A-F diagnostics, with matrix products on the input device.

    FP64 sufficient statistics and small sampled-pair workspaces avoid retaining
    all N*D*P full states, and avoid expensive full-dimensional CPU GEMMs on GPU runs.
    """
    def add_tensor(self, samples):
        import torch
        x = samples.detach().double()
        if x.ndim != 3 or x.shape[2] != self.p or min(x.shape[:2]) < 1 or not torch.isfinite(x).all():
            raise ValueError("Stage must be finite nonempty [N,R,P]")
        flat = x.reshape(-1, self.p)
        self.cross += (flat.T @ flat).cpu().numpy()
        if self.sample_centering:
            sum_x = x.sum(0).cpu().numpy()
            if self.sum_x is None:
                self.sum_x = np.zeros_like(sum_x)
            if self.sum_x.shape != sum_x.shape:
                raise ValueError("Representation axis changed")
            self.sum_x += sum_x
        first = torch.as_tensor(self.first, device=x.device)
        second = torch.as_tensor(self.second, device=x.device)
        for row in x:
            norms = torch.linalg.vector_norm(row, dim=0)
            active = norms > 1e-15
            self.zero_norm_concept_vectors += int((~active).sum())
            unit = row / norms.clamp_min(1e-15)
            count = int(active.sum())
            summed = unit[:, active].sum(1)
            self.exact_means.append(float((summed @ summed - count) / (count * (count - 1)))
                                    if count > 1 else np.nan)
            vals = []
            for start in range(0, len(first), 256):
                a, b = first[start:start + 256], second[start:start + 256]
                value = (unit[:, a] * unit[:, b]).sum(0)
                value[~(active[a] & active[b])] = float("nan")
                vals.append(value.cpu().numpy())
            self.cosines.append(np.concatenate(vals))
        self.samples += len(x)
        self.rows += len(flat)


class VariantAccumulator:
    def __init__(self, builder, *, pairs, seed, cache_dtype):
        self.builder = builder
        self.cache_dtype = cache_dtype
        self.stages = {name: TensorStageAccumulator(builder.num_concepts, pairs=pairs, seed=seed)
                       for name in ("pre_pca", "post_pca_pre_l2", "post_l2", "cache_quantized", "rank_gaussian")}
        self.post_l2 = []
        self.norms = []

    def add(self, full, *, native_stages=None):
        import torch.nn.functional as F
        if native_stages is None:
            projected = self.builder.project_full_states(full, normalize=False)
            post_l2 = F.normalize(projected, dim=1)
        else:
            projected = native_stages["post_projection_pre_l2"]
            post_l2 = native_stages["post_l2"]
        self.stages["pre_pca"].add_tensor(full)
        self.stages["post_pca_pre_l2"].add_tensor(projected)
        self.stages["post_l2"].add_tensor(post_l2)
        self.norms.append(full.detach().float().norm(dim=1).cpu().numpy())
        quantized = post_l2.detach().float().cpu().numpy().astype(self.cache_dtype).astype(np.float32)
        self.stages["cache_quantized"].add(quantized)
        self.post_l2.append(quantized)

    def report(self):
        ranked = _rank_gaussian_tensor(np.concatenate(self.post_l2))
        for start in range(0, len(ranked), 4):
            self.stages["rank_gaussian"].add(ranked[start:start + 4])
        self.post_l2.clear()
        return {"pre_pca_vector_norm": distribution(np.concatenate(self.norms)),
                "stages": {name: stage.report() for name, stage in self.stages.items()}}


def binary_auc(positive, negative):
    """Pairwise ROC-AUC with half credit for ties; no negative labels are invented."""
    a, b = np.asarray(positive), np.asarray(negative)
    if not len(a) or not len(b):
        return None
    delta = a[:, None] - b[None]
    return float(np.mean((delta > 0) + .5 * (delta == 0)))


def annotation_metrics(records):
    labeled = [r for r in records if r.get("relation") in ("relevant", "hard_negative", "unrelated")]
    comparable = [r for r in labeled if r.get("chunk_max_cosine") is not None
                  and r.get("token_max_cosine") is not None and r.get("chunk_coverage_complete", True)]
    output = {"status": "evaluated" if labeled else "pending_human_labels",
              "n_labeled_pairs": len(labeled), "n_comparable_pairs": len(comparable),
              "n_excluded_incomplete_pairs": len(labeled) - len(comparable),
              "label_source": "explicit_human_annotation_only",
              "methods": {}}
    output["diagnostics_by_relation"] = {}
    for relation in ("relevant", "hard_negative", "unrelated"):
        rows = [r for r in labeled if r["relation"] == relation]
        output["diagnostics_by_relation"][relation] = {
            name: distribution([r.get(name) for r in rows]) for name in
            ("token_max_cosine", "token_std_cosine", "token_normalized_entropy", "raw_vector_norm",
             "raw_to_uniform_relative_distance", "residual_to_raw_norm_ratio", "cosine_to_uniform")}
    for method in ("token_max_cosine", "chunk_max_cosine"):
        known = comparable
        positive = [r[method] for r in known if r["relation"] == "relevant"]
        negative = [r[method] for r in known if r["relation"] != "relevant"]
        per_document = []
        for doc in sorted({r["document_id"] for r in known}):
            rows = [r for r in known if r["document_id"] == doc]
            auc = binary_auc([r[method] for r in rows if r["relation"] == "relevant"],
                             [r[method] for r in rows if r["relation"] != "relevant"])
            if auc is not None:
                per_document.append(auc)
        output["methods"][method] = {"positive_scores": distribution(positive),
                                      "negative_scores": distribution(negative),
                                      "pooled_auc": binary_auc(positive, negative),
                                      "within_document_auc": distribution(per_document),
                                      "hard_negative_auc": binary_auc(positive, [r[method] for r in known
                                                                                 if r["relation"] == "hard_negative"])}
    return output


class AttentionDesignDiagnostics:
    """Analyze each document once; keep only bounded low-dimensional rank inputs."""
    def __init__(self, builder, concept_ids, *, seed=17, pairs=4096,
                 temperatures=(.1, .05, .02), cache_dtype="float16",
                 token_spectrum_samples=128, top_k=5):
        if not builder.use_projection:
            raise ValueError("Design diagnostics require a fixed projection")
        if len(concept_ids) != builder.num_concepts or len(set(concept_ids)) != len(concept_ids):
            raise ValueError("Unique concept IDs must align with prototypes")
        temps = tuple(dict.fromkeys((builder.temperature, *map(float, temperatures))))
        if any(not np.isfinite(t) or t <= 0 for t in temps):
            raise ValueError("Temperatures must be finite and positive")
        if cache_dtype not in ("float16", "float32") or token_spectrum_samples < 2 or top_k < 1:
            raise ValueError("Invalid cache/diagnostic limits")
        self.builder, self.concept_ids = builder, tuple(concept_ids)
        self.seed, self.temperatures = seed, temps
        self.token_spectrum_samples, self.top_k = token_spectrum_samples, top_k
        names = ["production_native", "fp32_raw_values", "unit_values", "centered_raw_values", "uniform_raw_values",
                 "chunk_pooled"] + [self.temperature_name(t) for t in temps if t != builder.temperature]
        self.variants = {name: VariantAccumulator(builder, pairs=pairs, seed=seed, cache_dtype=cache_dtype)
                         for name in names}
        self.attention = {str(t): TensorStageAccumulator(builder.num_concepts, pairs=pairs,
                                                        seed=seed, sample_centering=False) for t in temps}
        self.score_patterns = TensorStageAccumulator(builder.num_concepts, pairs=pairs, seed=seed,
                                                     sample_centering=False)
        self.values = {}
        self.token_spectra, self.documents = [], []
        self.native_dtypes = set()

    @staticmethod
    def temperature_name(t):
        return f"temperature_{t:g}_raw_values"

    def collect(self, name, tensor):
        value = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        self.values.setdefault(name, []).append(value.reshape(-1))

    def add_document(self, document, states, native_alpha, native_stages, *,
                     token_units, chunk_vectors, chunk_units, coverage, evidence_concepts=()):
        import torch
        import torch.nn.functional as F
        # Inputs contain active tokens only. No padded positions enter any probe.
        if len(self.documents) >= 512:
            raise ValueError("Design diagnostics are bounded to 512 documents")
        if states.ndim != 2 or len(token_units) != len(states) or not torch.isfinite(states).all():
            raise ValueError("Token states and evidence metadata must align and be finite")
        if chunk_vectors.ndim != 2 or len(chunk_vectors) != len(chunk_units) or not len(chunk_units):
            raise ValueError("Chunk states and units must align")
        self.native_dtypes.add(str(states.dtype))
        z = states.float()
        proto = self.builder.prototypes.to(device=z.device)
        logits = cosine_logits(z, proto)
        profiles = logit_profiles(logits)
        for name, value in profiles.items():
            self.collect("token_logits/" + name, value)
        self.score_patterns.add_tensor((logits - logits.mean(0))[None])
        norms = z.norm(dim=1)
        mean = z.mean(0)
        uniform = mean[:, None].expand(-1, self.builder.num_concepts)
        unit = F.normalize(z, dim=1)
        n = len(z)
        self.collect("token_norm", norms)
        norm_mass = norms / norms.sum().clamp_min(1e-12)
        self.collect("largest_10_percent_token_norm_mass", norm_mass.topk(max(1, math.ceil(.1 * n))).values.sum())
        self.collect("token_mean_energy_fraction", mean.square().sum() / z.square().sum(1).mean().clamp_min(1e-12))
        summed = unit.double().sum(0)
        count = int((norms > 1e-12).sum())
        self.collect("token_pair_cosine", float((summed @ summed - count) / (count * (count - 1)))
                     if count > 1 else np.nan)
        take = np.random.default_rng(np.random.SeedSequence([self.seed, document["row_index"]])).choice(
            n, min(n, self.token_spectrum_samples), replace=False)
        selected = z[torch.as_tensor(take, device=z.device)].double()
        gram = selected @ selected.T
        spec = spectrum(gram.cpu().numpy())
        self.token_spectra.append({"document_id": document["id"], "active_tokens": n,
                                   "sampled_tokens": len(take), "spectrum": spec})
        fp32_alpha = None
        for t in self.temperatures:
            alpha = torch.softmax(logits / t, dim=0)
            self.attention[str(t)].add_tensor(alpha[None])
            for name, value in attention_profiles(alpha).items():
                self.collect(f"temperature/{t}/{name}", value)
            self.collect(f"temperature/{t}/max_minus_median_scaled", profiles["max_minus_median"] / t)
            if t == self.builder.temperature:
                fp32_alpha = alpha
            else:
                self.variants[self.temperature_name(t)].add((z.T @ alpha)[None])
        raw = z.T @ fp32_alpha
        unit_output = unit.T @ fp32_alpha
        residual = (z - mean).T @ fp32_alpha
        self.variants["production_native"].add(native_stages["h_full"][None],
                                             native_stages={k: v[None] for k, v in native_stages.items()})
        for name, full in (("fp32_raw_values", raw), ("unit_values", unit_output),
                           ("centered_raw_values", residual), ("uniform_raw_values", uniform)):
            self.variants[name].add(full[None])
        self.collect("raw_vs_uniform_relative_distance", relative_distance(raw, uniform))
        self.collect("raw_vs_uniform_cosine", F.cosine_similarity(raw, uniform, dim=0))
        self.collect("residual_to_raw_norm_ratio", residual.norm(dim=0) / raw.norm(dim=0).clamp_min(1e-12))
        self.collect("native_vs_fp32_full_relative_distance", relative_distance(native_stages["h_full"].float(), raw))
        self.collect("native_vs_fp32_attention_abs_difference", (native_alpha.float() - fp32_alpha).abs())
        contribution = fp32_alpha * norms[:, None]
        contribution = contribution / contribution.sum(0, keepdim=True).clamp_min(1e-12)
        self.collect("contribution_top5_mass", contribution.topk(min(5, n), dim=0).values.sum(0))
        # Pooled chunk embeddings are separately encoded and normalized in FP32.
        chunks = F.normalize(chunk_vectors.to(z.device).float(), dim=1)
        chunk_logits = cosine_logits(chunks, proto)
        chunk_alpha = torch.softmax(chunk_logits / self.builder.temperature, dim=0)
        self.variants["chunk_pooled"].add((chunks.T @ chunk_alpha)[None])
        chunk_profile = logit_profiles(chunk_logits)
        for name, value in chunk_profile.items():
            self.collect("chunk_logits/" + name, value)
        for name, value in attention_profiles(chunk_alpha).items():
            self.collect("chunk_attention/" + name, value)
        complete = bool(coverage["complete"])
        self.documents.append({"document_id": document["id"], "row_index": document["row_index"],
                               "active_tokens": n, "evidence_coverage": coverage})
        cpu_profiles = {k: v.detach().cpu().numpy() for k, v in profiles.items()}
        entropy = attention_profiles(fp32_alpha)["normalized_entropy"].cpu().numpy()
        distances = relative_distance(raw, uniform).cpu().numpy()
        chunk_max = chunk_profile["max"].cpu().numpy()
        raw_norms = raw.norm(dim=0).cpu().numpy()
        residual_ratios = (residual.norm(dim=0) / raw.norm(dim=0).clamp_min(1e-12)).cpu().numpy()
        uniform_cosines = F.cosine_similarity(raw, uniform, dim=0).cpu().numpy()
        scores = [{"document_id": document["id"], "row_index": document["row_index"], "concept_id": cid,
                   "token_mean_cosine": float(cpu_profiles["mean"][c]),
                   "token_std_cosine": float(cpu_profiles["std"][c]),
                   "token_max_cosine": float(cpu_profiles["max"][c]),
                   "token_max_minus_median": float(cpu_profiles["max_minus_median"][c]),
                   "token_normalized_entropy": float(entropy[c]),
                   "chunk_max_cosine": float(chunk_max[c]) if complete else None,
                   "chunk_coverage_complete": complete,
                   "raw_vector_norm": float(raw_norms[c]),
                   "residual_to_raw_norm_ratio": float(residual_ratios[c]),
                   "cosine_to_uniform": float(uniform_cosines[c]),
                   "raw_to_uniform_relative_distance": float(distances[c]) if np.isfinite(distances[c]) else None}
                  for c, cid in enumerate(self.concept_ids)]
        evidence = []
        requested = set(evidence_concepts)
        if requested == {"__suggest__"}:
            # Suggestions are score extremes, never inferred positive/negative labels.
            order = torch.argsort(profiles["max"]).cpu().tolist()
            requested = {self.concept_ids[c] for c in order[:3] + order[-3:]}
        for c, cid in enumerate(self.concept_ids):
            if cid not in requested:
                continue
            def best(values, units):
                ids = values.topk(min(self.top_k, len(units))).indices.cpu().tolist()
                return [{**units[i], "unit_index": i, "score": float(values[i])} for i in ids]
            evidence.append({**scores[c], "relation": "", "suggestion_is_not_label": True,
                             "token_evidence": best(logits[:, c], token_units),
                             "norm_weighted_contributors": best(contribution[:, c], token_units),
                             "chunk_evidence": best(chunk_logits[:, c], chunk_units)})
        return scores, evidence

    def report(self):
        if len(self.documents) < 2:
            raise ValueError("Need at least two documents")
        variants = {name: acc.report() for name, acc in self.variants.items()}
        return {"schema_version": 1, "n_documents": len(self.documents), "seed": self.seed,
                "mode": "diagnostic_only_production_defaults_unchanged",
                "encoder_native_dtypes": sorted(self.native_dtypes),
                "fp32_control": "recompute_attention_and_aggregation_on_same_native_states_not_fp32_encoder",
                "value_control_attention": "same_fp32_alpha_at_baseline_temperature",
                "rank_gaussian_input_cache_dtype": self.variants["production_native"].cache_dtype,
                "temperatures": list(self.temperatures), "projection": self.builder.projection_metadata(),
                "statistics": {name: distribution(np.concatenate(parts)) for name, parts in self.values.items()},
                "token_geometry": self.token_spectra,
                "centered_logit_patterns": self.score_patterns.report(),
                "attention_temperature_comparison": {t: acc.report() for t, acc in self.attention.items()},
                "variants": variants, "sample_rows": self.documents,
                "interpretation_limits": [
                    "Higher effective rank alone is not evidence of better concept semantics.",
                    "Residuals can amplify tiny noise; inspect residual/raw norms and human evidence.",
                    "The same prototype-PCA mean is applied to all variants, including centered values.",
                    "Chunk evidence uses independently encoded normalized pooled vectors; count/context differ from tokens.",
                    "Incomplete chunk coverage is excluded from human score comparisons.",
                    "The FP32 control does not rerun the encoder or recover information lost during BF16 encoding.",
                    "Cross-document Gram spectra do not establish every document's mathematical rank.",
                ]}
