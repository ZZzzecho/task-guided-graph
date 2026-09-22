import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from graph_mvp.patient_repr import (
    ConceptVocabulary, PatientConceptMatrixBuilder, PatientMatrixCacheWriter,
    PatientMatrixDataset, last_token_pool, save_concept_prototypes,
    load_concept_prototypes,
)


def test_last_token_pool_left_and_right_padding():
    h = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    left = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    pooled_left = last_token_pool(h, left)
    assert torch.equal(pooled_left, h[:, -1])

    right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    pooled_right = last_token_pool(h, right)
    assert torch.equal(pooled_right[0], h[0, 1])
    assert torch.equal(pooled_right[1], h[1, 2])


class FakeEncoder:
    hidden_size = 4


def test_patient_builder_shape_and_attention_normalization():
    prototypes = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ], dtype=np.float32)
    builder = PatientConceptMatrixBuilder(FakeEncoder(), prototypes, temperature=0.2)
    z = torch.tensor([
        [[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [9, 9, 9, 9]],
        [[0., 1, 0, 0], [1, 0., 0, 0], [0, 0, 0, 1.], [9, 9, 9, 9]],
    ])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    h, alpha = builder.from_hidden_states(z, mask)
    assert h.shape == (2, 4, 3)
    assert alpha.shape == (2, 4, 3)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(2, 3), atol=1e-6)
    assert torch.all(alpha[:, -1] == 0)
    assert torch.isfinite(h).all()


def test_concept_prototype_cache_roundtrip(tmp_path):
    vocab = ConceptVocabulary(("c1", "c2"), ("heart failure", "fever"))
    p = np.eye(2, 4, dtype=np.float32)
    path = tmp_path / "prototypes.npz"
    save_concept_prototypes(path, p, vocab, encoder_id="fake")
    loaded, meta = load_concept_prototypes(path)
    np.testing.assert_allclose(loaded, p)
    assert meta["concept_ids"] == ["c1", "c2"]
    assert meta["pooling"] == "qwen3_official_last_token_normalized"


def test_patient_cache_shards_and_materialize(tmp_path):
    writer = PatientMatrixCacheWriter(
        tmp_path, ("c1", "c2", "c3"), hidden_size=4,
        dtype="float16", encoder_id="fake", temperature=0.1,
    )
    x1 = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) / 10
    x2 = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3) / 20
    writer.write_shard(x1, [10, 11], [100, 101])
    writer.write_shard(x2, [12, 13, 14], [102, 103, 104])
    writer.close()

    ds = PatientMatrixDataset(tmp_path)
    assert len(ds) == 5
    assert ds.hidden_size == 4
    assert ds.num_concepts == 3
    assert ds.metadata["source_split"] == "train"
    assert ds.metadata["frozen_during_graph_rl"] is True
    y = ds.materialize(max_samples=4)
    assert y.shape == (4, 4, 3)
    np.testing.assert_allclose(y[:2], x1, atol=1e-3)
    assert isinstance(ds.fingerprint(), str) and len(ds.fingerprint()) == 64
