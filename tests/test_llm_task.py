from types import SimpleNamespace
import pytest

from graph_mvp.graph_tokens import SoftGraphTokenizer
from graph_mvp.llm_task import (GraphConditionedCausalLM, CausalTaskContext,
                                FrozenGraphCausalEvaluator, encode_binary_batch)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = [2 + (ord(c) % 20) for c in text]
        if add_special_tokens:
            ids = [2] + ids
        return {"input_ids": ids}


def test_binary_batch_masks_gold_answer_from_activation():
    torch = pytest.importorskip("torch")
    batch = encode_binary_batch(TinyTokenizer(), ["pain"], [1], max_length=220)
    assert batch["input_ids"].shape == batch["labels"].shape
    answer = batch["labels"] != -100
    assert answer.any()
    assert not batch["activation_mask"][answer].any()
    assert batch["activation_mask"].any()


def test_graph_conditioned_causal_evaluator_runs(state):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyLM(nn.Module):
        def __init__(self, vocab=32, hidden=6):
            super().__init__()
            self.emb = nn.Embedding(vocab, hidden)
            self.out = nn.Linear(hidden, vocab)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, inputs_embeds, attention_mask, labels):
            logits = self.out(inputs_embeds)
            shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            return SimpleNamespace(logits=logits, loss=loss)

    prototypes = torch.randn(3, 6)
    graph_tok = SoftGraphTokenizer(prototypes, output_dim=6, num_tokens=2, graph_hidden_dim=5)
    wrapper = GraphConditionedCausalLM(TinyLM(), graph_tok)
    ctx = CausalTaskContext(state.snapshot.concept_ids,
                            ("Chief complaint: pain", "Chief complaint: cough"),
                            (0, 1), TinyTokenizer(), batch_size=2, max_length=220)
    evaluator = FrozenGraphCausalEvaluator(wrapper)
    metrics = evaluator.evaluate_snapshot(state.snapshot, ctx)
    assert metrics.task_loss == -metrics.task_metric
    assert metrics.extra_metrics["n_examples"] == 2
    relevance = evaluator.task_relevance(state.snapshot, ctx)
    assert relevance.shape == (3, 3)
    assert torch.isfinite(torch.tensor(metrics.task_loss))
