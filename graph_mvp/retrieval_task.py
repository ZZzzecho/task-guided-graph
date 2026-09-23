"""Graph-conditioned patent citation retrieval for task-guided Graph-RL."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .downstream import TaskMetrics, EvaluationResult, EvaluationError
from .graph_tokens import error_weighted_edge_relevance
from .patent_retrieval import load_candidate_pools


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Patent retrieval requires PyTorch") from exc
    return torch, F


def _pad_token_id(tokenizer):
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(pad, int):
        raise ValueError("tokenizer must define a scalar pad_token_id or eos_token_id")
    return pad


def encode_text_batch(tokenizer, texts, *, max_length, device):
    """Tokenize a batch for encoder-style pooling through a causal-LM backbone."""
    if not texts:
        raise ValueError("texts must be nonempty")
    _pad_token_id(tokenizer)
    batch = tokenizer(
        list(map(str, texts)),
        padding=True,
        truncation=True,
        max_length=int(max_length),
        add_special_tokens=True,
        return_tensors="pt",
    )
    if "input_ids" not in batch or "attention_mask" not in batch:
        raise ValueError("tokenizer batch must contain input_ids and attention_mask")
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }


def _last_active_pool(hidden, attention_mask):
    """Pool the last non-padding token for left- or right-padded batches."""
    torch, _ = _torch()
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("hidden/attention_mask shape mismatch")
    positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    last = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    if torch.any(last < 0):
        raise ValueError("every sequence must contain at least one active token")
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, last]


def _unwrap_backbone(causal_lm):
    """Return the transformer body without the LM head when possible."""
    base = causal_lm.get_base_model() if hasattr(causal_lm, "get_base_model") else causal_lm
    backbone = getattr(base, "model", None)
    if backbone is not None and callable(getattr(backbone, "forward", None)):
        return backbone
    raise ValueError(
        "Could not locate decoder backbone via get_base_model().model; "
        "retrieval needs hidden states without materializing LM logits."
    )


@dataclass(frozen=True)
class RetrievalTaskContext:
    concept_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    query_texts: tuple[str, ...]
    candidate_ids: tuple[tuple[str, ...], ...]
    positive_indices: tuple[int, ...]
    text_by_id: Mapping[str, str]
    tokenizer: object
    batch_size: int = 2
    max_length: int = 512
    score_temperature: float = 0.07

    def __post_init__(self):
        concept_ids = tuple(map(str, self.concept_ids))
        query_ids = tuple(map(str, self.query_ids))
        query_texts = tuple(map(str, self.query_texts))
        candidate_ids = tuple(tuple(map(str, row)) for row in self.candidate_ids)
        positive_indices = tuple(int(x) for x in self.positive_indices)
        if len(concept_ids) < 2 or len(set(concept_ids)) != len(concept_ids):
            raise ValueError("concept_ids must be unique")
        n = len(query_ids)
        if n < 1 or len(query_texts) != n or len(candidate_ids) != n or len(positive_indices) != n:
            raise ValueError("retrieval context arrays must align and be nonempty")
        pool_sizes = {len(x) for x in candidate_ids}
        if len(pool_sizes) != 1 or min(pool_sizes) < 2:
            raise ValueError("all retrieval candidate pools must have one fixed size >=2")
        if any(not 0 <= y < len(c) for y, c in zip(positive_indices, candidate_ids)):
            raise ValueError("positive index outside candidate pool")
        text_by_id = {str(k): str(v) for k, v in self.text_by_id.items()}
        required = set(query_ids)
        for row in candidate_ids:
            required.update(row)
        missing = sorted(required - set(text_by_id))
        if missing:
            raise ValueError(f"missing patent text for IDs: {missing[:5]}")
        if self.batch_size < 1 or self.max_length < 16 or self.score_temperature <= 0:
            raise ValueError("invalid retrieval batch/max_length/temperature")
        object.__setattr__(self, "concept_ids", concept_ids)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "query_texts", query_texts)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "positive_indices", positive_indices)
        object.__setattr__(self, "text_by_id", MappingProxyType(text_by_id))

    @property
    def pool_size(self):
        return len(self.candidate_ids[0])

    def batches(self):
        for start in range(0, len(self.query_ids), self.batch_size):
            stop = min(start + self.batch_size, len(self.query_ids))
            yield {
                "query_ids": self.query_ids[start:stop],
                "query_texts": self.query_texts[start:stop],
                "candidate_ids": self.candidate_ids[start:stop],
                "positive_indices": self.positive_indices[start:stop],
            }


def load_retrieval_context(
    prepared_dir,
    pool_path,
    concept_ids,
    tokenizer,
    *,
    batch_size=2,
    max_length=512,
    score_temperature=0.07,
    max_queries=None,
):
    """Load fixed retrieval pools and only the patent texts referenced by them."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Loading patent retrieval contexts requires pandas") from exc
    pools = load_candidate_pools(pool_path)
    if max_queries is not None:
        if int(max_queries) < 1:
            raise ValueError("max_queries must be positive")
        pools = pools[: int(max_queries)]
    ids = set()
    for row in pools:
        ids.add(str(row["query_id"]))
        ids.update(map(str, row["candidate_ids"]))
    corpus = pd.read_csv(
        Path(prepared_dir) / "corpus.csv.gz",
        usecols=["patent_id", "text"],
        dtype=str,
    )
    corpus["patent_id"] = corpus["patent_id"].astype(str)
    corpus = corpus[corpus["patent_id"].isin(ids)].drop_duplicates("patent_id")
    text_by_id = dict(zip(corpus["patent_id"], corpus["text"].astype(str)))
    query_ids = tuple(str(row["query_id"]) for row in pools)
    candidate_ids = tuple(tuple(map(str, row["candidate_ids"])) for row in pools)
    positive = tuple(int(row.get("positive_index", 0)) for row in pools)
    query_texts = tuple(text_by_id[qid] for qid in query_ids)
    return RetrievalTaskContext(
        tuple(concept_ids),
        query_ids,
        query_texts,
        candidate_ids,
        positive,
        text_by_id,
        tokenizer,
        batch_size=int(batch_size),
        max_length=int(max_length),
        score_temperature=float(score_temperature),
    )


def _build_retriever_class():
    torch, F = _torch()
    import torch.nn as nn

    class GraphConditionedRetriever(nn.Module):
        """Bi-encoder retrieval with graph conditioning only on the query side."""

        def __init__(self, causal_lm, graph_tokenizer):
            super().__init__()
            self.causal_lm = causal_lm
            self.graph_tokenizer = graph_tokenizer
            self.backbone = _unwrap_backbone(causal_lm)
            embedding = self.causal_lm.get_input_embeddings()
            if embedding is None:
                raise ValueError("causal_lm has no input embeddings")
            try:
                device = embedding.weight.device
                if device.type != "meta":
                    self.graph_tokenizer.to(
                        device=device,
                        dtype=embedding.weight.dtype,
                    )
            except AttributeError:
                pass

        @property
        def device(self):
            return self.causal_lm.get_input_embeddings().weight.device

        def _backbone_hidden(self, inputs_embeds, attention_mask):
            out = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                try:
                    hidden = out[0]
                except Exception as exc:
                    raise RuntimeError("backbone did not return last hidden states") from exc
            return hidden

        def encode_candidates(self, input_ids, attention_mask):
            emb = self.causal_lm.get_input_embeddings()(input_ids)
            hidden = self._backbone_hidden(emb, attention_mask)
            return F.normalize(_last_active_pool(hidden, attention_mask), p=2, dim=-1)

        def encode_queries(self, input_ids, attention_mask, global_graph, return_aux=False):
            emb = self.causal_lm.get_input_embeddings()(input_ids)
            graph_tokens, aux = self.graph_tokenizer(
                emb,
                global_graph,
                token_mask=attention_mask.bool(),
                return_aux=True,
            )
            b, k, _ = graph_tokens.shape
            inputs = torch.cat((graph_tokens, emb), dim=1)
            graph_mask = torch.ones((b, k), dtype=attention_mask.dtype, device=attention_mask.device)
            full_mask = torch.cat((graph_mask, attention_mask), dim=1)
            hidden = self._backbone_hidden(inputs, full_mask)
            pooled = F.normalize(_last_active_pool(hidden, full_mask), p=2, dim=-1)
            if return_aux:
                return pooled, aux
            return pooled

    return GraphConditionedRetriever


try:  # pragma: no cover
    GraphConditionedRetriever = _build_retriever_class()
except RuntimeError:  # pragma: no cover
    class GraphConditionedRetriever:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("GraphConditionedRetriever requires PyTorch")


class FrozenGraphRetrievalEvaluator:
    """Fixed-index retrieval reward for Graph-RL."""

    def __init__(self, model: GraphConditionedRetriever, candidate_batch_size=8):
        self.model = model
        self.candidate_batch_size = int(candidate_batch_size)
        if self.candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive")
        self._candidate_cache = {}

    def prepare(self, context: RetrievalTaskContext):
        torch, _ = _torch()
        missing = []
        seen = set()
        for row in context.candidate_ids:
            for pid in row:
                if pid not in self._candidate_cache and pid not in seen:
                    missing.append(pid)
                    seen.add(pid)
        if not missing:
            return {"encoded": 0, "cached": len(self._candidate_cache)}
        self.model.eval()
        device = self.model.device
        with torch.no_grad():
            for start in range(0, len(missing), self.candidate_batch_size):
                ids = missing[start:start + self.candidate_batch_size]
                batch = encode_text_batch(
                    context.tokenizer,
                    [context.text_by_id[x] for x in ids],
                    max_length=context.max_length,
                    device=device,
                )
                vec = self.model.encode_candidates(**batch).detach().float().cpu()
                for pid, row in zip(ids, vec):
                    self._candidate_cache[pid] = row
        return {"encoded": len(missing), "cached": len(self._candidate_cache)}

    def _candidate_tensor(self, candidate_ids, device, dtype):
        torch, _ = _torch()
        rows = []
        for pool in candidate_ids:
            try:
                rows.append(torch.stack([self._candidate_cache[pid] for pid in pool], dim=0))
            except KeyError as exc:
                raise EvaluationError(f"candidate index missing patent {exc.args[0]}") from exc
        return torch.stack(rows, dim=0).to(device=device, dtype=dtype)

    def _score(self, snapshot, context: RetrievalTaskContext, return_activation=False):
        if snapshot.concept_ids != context.concept_ids:
            raise ValueError("Graph/data concept axis mismatch")
        torch, F = _torch()
        self.prepare(context)
        self.model.eval()
        device = self.model.device
        all_losses, all_ranks, activations = [], [], []
        with torch.no_grad():
            for info in context.batches():
                batch = encode_text_batch(
                    context.tokenizer,
                    info["query_texts"],
                    max_length=context.max_length,
                    device=device,
                )
                q, aux = self.model.encode_queries(
                    **batch,
                    global_graph=snapshot.Rho,
                    return_aux=True,
                )
                # Keep the expensive LM/graph forward in BF16/FP16, but compute
                # retrieval scores and NLL in FP32. Graph-RL needs reward differences
                # much finer than BF16's quantization grid.
                q_score = q.float()
                cand = self._candidate_tensor(
                    info["candidate_ids"], device, torch.float32
                )
                scores = torch.einsum("bd,bkd->bk", q_score, cand) / float(context.score_temperature)
                gold = torch.as_tensor(info["positive_indices"], dtype=torch.long, device=device)
                losses = F.cross_entropy(scores, gold, reduction="none")
                if not torch.isfinite(losses).all():
                    raise EvaluationError("nonfinite retrieval loss")
                order = torch.argsort(scores, dim=1, descending=True)
                ranks = (order == gold[:, None]).nonzero(as_tuple=False)[:, 1] + 1
                all_losses.extend(losses.detach().float().cpu().tolist())
                all_ranks.extend(ranks.detach().cpu().tolist())
                if return_activation:
                    activations.append(aux["activations"].detach().float().cpu().numpy())
        loss = np.asarray(all_losses, dtype=float)
        ranks = np.asarray(all_ranks, dtype=int)
        if len(loss) != len(context.query_ids):
            raise EvaluationError("retrieval scorer did not return one loss per query")
        mean_loss = float(loss.mean())
        reciprocal = 1.0 / ranks
        ndcg10 = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0)
        metrics = TaskMetrics(mean_loss, -mean_loss, {
            "mean_nll": mean_loss,
            "mrr": float(reciprocal.mean()),
            "recall_at_1": float((ranks <= 1).mean()),
            "recall_at_10": float((ranks <= 10).mean()),
            "recall_at_50": float((ranks <= 50).mean()),
            "ndcg_at_10": float(ndcg10.mean()),
            "mean_rank": float(ranks.mean()),
            "n_queries": int(len(ranks)),
            "pool_size": int(context.pool_size),
        })
        if return_activation:
            return metrics, np.concatenate(activations, axis=0), loss
        return metrics

    def evaluate_snapshot(self, snapshot, task_context):
        return self._score(snapshot, task_context)

    def evaluate(self, candidate, task_context, baseline):
        if not candidate.valid or candidate.snapshot is None:
            raise ValueError("Cannot evaluate invalid candidate")
        metrics = self._score(candidate.snapshot, task_context)
        return EvaluationResult(
            candidate.candidate_id,
            metrics.task_loss,
            metrics.task_metric,
            baseline.task_loss,
            baseline.task_metric,
            metrics.task_metric - baseline.task_metric,
            metrics.extra_metrics,
        )

    def task_relevance(self, snapshot, task_context):
        _, activation, losses = self._score(snapshot, task_context, return_activation=True)
        return error_weighted_edge_relevance(activation, losses, normalize=True)


class RetrievalTaskRelevanceProvider:
    def __init__(self, evaluator: FrozenGraphRetrievalEvaluator):
        self.evaluator = evaluator
        self._cache = {}

    def __call__(self, state, reward_context, baseline):
        key = (state.state_id, id(reward_context))
        if key not in self._cache:
            self._cache[key] = self.evaluator.task_relevance(state.snapshot, reward_context)
        return {"task_relevance": self._cache[key]}


def adapt_retrieval_model(
    model,
    evaluator: FrozenGraphRetrievalEvaluator,
    snapshot,
    context: RetrievalTaskContext,
    optimizer,
    *,
    steps=50,
    max_grad_norm=1.0,
):
    """Accepted-graph-only LoRA + SoftGraphTokenizer query-side adaptation."""
    if steps < 1:
        raise ValueError("steps must be positive")
    torch, F = _torch()
    evaluator.prepare(context)
    model.train()
    device = model.device
    iterator = iter(context.batches())
    losses_out = []
    for _ in range(int(steps)):
        try:
            info = next(iterator)
        except StopIteration:
            iterator = iter(context.batches())
            info = next(iterator)
        batch = encode_text_batch(
            context.tokenizer,
            info["query_texts"],
            max_length=context.max_length,
            device=device,
        )
        q = model.encode_queries(**batch, global_graph=snapshot.Rho, return_aux=False)
        # FP32 ranking loss gives stable gradients/rewards while the backbone stays
        # in its native BF16/FP16 dtype. Casting q is differentiable.
        q_score = q.float()
        cand = evaluator._candidate_tensor(
            info["candidate_ids"], device, torch.float32
        ).detach()
        scores = torch.einsum("bd,bkd->bk", q_score, cand) / float(context.score_temperature)
        gold = torch.as_tensor(info["positive_indices"], dtype=torch.long, device=device)
        loss = F.cross_entropy(scores, gold)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite retrieval adaptation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
        optimizer.step()
        losses_out.append(float(loss.detach().cpu()))
    return {
        "steps": int(steps),
        "initial_loss": losses_out[0],
        "final_loss": losses_out[-1],
        "mean_loss": float(np.mean(losses_out)),
        "candidate_index_frozen": True,
    }
