import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from graph_mvp.task_prototypes import task_lm_concept_prototypes


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [(ord(ch) % 7) + 1 for ch in text][:5]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(16, 6)
        with torch.no_grad():
            self.emb.weight.copy_(torch.arange(16 * 6).reshape(16, 6) / 100)

    def get_input_embeddings(self):
        return self.emb


def test_task_lm_prototypes_are_in_task_embedding_space():
    model = TinyModel()
    tokenizer = TinyTokenizer()
    p = task_lm_concept_prototypes(model, tokenizer, ["fever", "pain"])
    assert p.shape == (2, 6)
    expected_ids = tokenizer("fever")["input_ids"]
    expected = model.emb(expected_ids).mean(dim=1).detach().numpy()[0]
    np.testing.assert_allclose(p[0], expected)
