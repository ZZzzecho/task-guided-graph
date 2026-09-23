from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from graph_mvp.graph_tokens import SoftGraphTokenizer
from graph_mvp.retrieval_task import (
    GraphConditionedRetriever,
    FrozenGraphRetrievalEvaluator,
    RetrievalTaskContext,
)


class BatchTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, padding=True, truncation=True, max_length=32,
                 add_special_tokens=True, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            ids = [2 + (ord(c) % 20) for c in str(text)]
            if add_special_tokens:
                ids = [2] + ids
            rows.append(ids[:max_length])
        width = max(map(len, rows))
        padded = [x + [0] * (width - len(x)) for x in rows]
        mask = [[1] * len(x) + [0] * (width - len(x)) for x in rows]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class TinyBackbone(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.ff = nn.Linear(hidden, hidden)

    def forward(self, inputs_embeds, attention_mask, use_cache=False, return_dict=True):
        x = self.ff(torch.cumsum(inputs_embeds, dim=1))
        return SimpleNamespace(last_hidden_state=x)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab=32, hidden=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.model = TinyBackbone(hidden)

    def get_input_embeddings(self):
        return self.emb


def test_retrieval_evaluator_runs(state):
    torch.manual_seed(3)
    lm = TinyCausalLM()
    graph_tok = SoftGraphTokenizer(
        torch.randn(3, 8), output_dim=8, num_tokens=2, graph_hidden_dim=6
    )
    retriever = GraphConditionedRetriever(lm, graph_tok)
    tok = BatchTokenizer()
    text = {
        "q1": "routing congestion",
        "q2": "secure network",
        "p1": "routing prior art",
        "p2": "unrelated storage",
        "p3": "security prior art",
        "p4": "another protocol",
    }
    ctx = RetrievalTaskContext(
        state.snapshot.concept_ids,
        ("q1", "q2"),
        (text["q1"], text["q2"]),
        (("p1", "p2"), ("p3", "p4")),
        (0, 0),
        text,
        tok,
        batch_size=2,
        max_length=32,
        score_temperature=0.2,
    )
    ev = FrozenGraphRetrievalEvaluator(retriever, candidate_batch_size=2)
    prep = ev.prepare(ctx)
    assert prep["cached"] == 4
    metrics = ev.evaluate_snapshot(state.snapshot, ctx)
    assert metrics.task_loss == -metrics.task_metric
    assert metrics.extra_metrics["n_queries"] == 2
    assert metrics.extra_metrics["pool_size"] == 2
    rel = ev.task_relevance(state.snapshot, ctx)
    assert rel.shape == (3, 3)
    assert np.isfinite(rel).all()
