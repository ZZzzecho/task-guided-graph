import json
from types import SimpleNamespace

import numpy as np
import pytest

from graph_mvp.estimators import _rank_gaussian_tensor
from graph_mvp.representation_diagnostics import ConceptStageAccumulator, diagnose_representations, spectrum


def test_streamed_covariance_matches_direct_sample_axis_centering():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(13, 5, 4))
    acc = ConceptStageAccumulator(4, pairs=6)
    acc.add(x[:3])
    acc.add(x[3:])
    report = acc.report()
    centered = x - x.mean(axis=0)
    covariance = np.einsum("nri,nrj->ij", centered, centered) / (13 * 5)
    np.testing.assert_allclose(report["sample_axis_centered_covariance"]["eigenvalues_descending"],
                               np.linalg.eigvalsh(covariance)[::-1], atol=1e-12)
    direct_cosines = []
    for row in x:
        unit = row / np.linalg.norm(row, axis=0)
        direct_cosines.extend((unit.T @ unit)[np.triu_indices(4, 1)])
    assert report["all_pair_document_cosine_mean"] == pytest.approx(np.mean(direct_cosines))
    assert report["pairwise_document_cosine"]["mean"] == pytest.approx(np.mean(direct_cosines))


def test_rank_one_and_zero_variance_are_distinguished():
    x = np.repeat(np.arange(12).reshape(3, 4, 1), 5, axis=2)
    acc = ConceptStageAccumulator(5)
    acc.add(x)
    report = acc.report()
    assert report["concept_gram"]["lambda1_over_trace"] == pytest.approx(1)
    assert report["concept_gram"]["effective_rank_participation"] == pytest.approx(1)
    assert report["pairwise_sample_axis_correlation"]["mean"] == pytest.approx(1)
    zero = ConceptStageAccumulator(5)
    zero.add(np.zeros((3, 4, 5)))
    assert zero.report()["concept_gram"]["effective_rank_entropy"] == 0
    assert zero.report()["pairwise_document_cosine"]["count"] == 0
    assert zero.report()["pairwise_sample_axis_correlation"]["count"] == 0


def test_all_six_stages_and_rank_gaussian_use_production_path():
    torch = pytest.importorskip("torch")
    from graph_mvp.patient_repr import PatientConceptMatrixBuilder
    z = torch.tensor(np.random.default_rng(3).normal(size=(12, 5, 8)), dtype=torch.float32)
    mask = torch.ones((12, 5), dtype=torch.bool)
    mask[:, -1] = False
    class Encoder:
        hidden_size = 8
        def encode_token_states(self, texts):
            ids = list(map(int, texts))
            return z[ids], mask[ids]
    builder = PatientConceptMatrixBuilder(Encoder(), np.eye(8, dtype=np.float32), representation_dim=4)
    report = diagnose_representations(builder, [list(map(str, range(6))), list(map(str, range(6, 12)))],
                                     cache_dtype="float16", max_samples=12)
    assert set(report["stages"]) == {"A_prototypes", "B_attention", "C_pre_pca", "D_post_pca_pre_l2",
                                     "E_post_l2", "E_cache_quantized", "F_rank_gaussian"}
    h, _ = builder.from_hidden_states(z, mask)
    ranked = _rank_gaussian_tensor(h.numpy().astype(np.float16).astype(np.float32))
    direct = ranked.reshape(-1, 8).T @ ranked.reshape(-1, 8) / (12 * 4)
    np.testing.assert_allclose(report["stages"]["F_rank_gaussian"]["concept_gram"]["eigenvalues_descending"],
                               np.linalg.eigvalsh(direct)[::-1], atol=1e-10)
    assert report["stages"]["B_attention"]["entropy_divided_by_log_active_tokens"]["quantiles"]["max"] <= 1.00001
    assert report["rank_gaussian_input_cache_dtype"] == "float16"
    json.dumps(report, allow_nan=False)


def test_diagnostic_cli_synthetic_and_seeded_sampling(tmp_path):
    pytest.importorskip("torch")
    from scripts.diagnose_representation_collapse import main, sample_training_rows
    out = tmp_path / "report.json"
    main(["--synthetic", "--max-samples", "12", "--representation-dim", "4", "--output", str(out)])
    report = json.loads(out.read_text())
    assert report["n_documents"] == 12 and report["provenance"]["synthetic"] is True
    csv = tmp_path / "train.csv"
    csv.write_text("patent_id,text\n" + "".join(f"p{i},text-{i}\n" for i in range(100)))
    first = sample_training_rows(csv, 12, 17)
    assert first == sample_training_rows(csv, 12, 17)
    assert first != sample_training_rows(csv, 12, 18)
    assert len({r["id"] for r in first}) == 12
