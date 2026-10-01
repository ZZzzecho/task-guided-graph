import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from graph_mvp.attention_design_diagnostics import (AttentionDesignDiagnostics, TensorStageAccumulator,
    annotation_metrics, binary_auc, cosine_logits, logit_profiles, relative_distance, split_evidence)
from graph_mvp.patient_repr import PatientConceptMatrixBuilder
from graph_mvp.representation_diagnostics import ConceptStageAccumulator


def make_builder(temperature=.1):
    class Encoder:
        hidden_size = 4
    return PatientConceptMatrixBuilder(Encoder(), np.eye(4, dtype=np.float32)[:3], temperature=temperature,
        representation_dim=2, projection_matrix=np.eye(4, dtype=np.float32)[:2], projection_mean=np.zeros(4))


def test_absolute_score_offsets_do_not_change_attention_but_profiles_capture_them():
    x = torch.tensor([[.01, .03], [.02, .01], [.03, .02]])
    shifted = x + torch.tensor([.6, -.4])
    torch.testing.assert_close(torch.softmax(x / .1, 0), torch.softmax(shifted / .1, 0))
    a, b = logit_profiles(x), logit_profiles(shifted)
    torch.testing.assert_close(a["std"], b["std"])
    torch.testing.assert_close(a["max_minus_median"], b["max_minus_median"])
    assert not torch.allclose(a["mean"], b["mean"])


def test_tensor_statistics_match_existing_fp64_streaming_with_undefined_vectors():
    x = torch.tensor(np.random.default_rng(3).normal(size=(7, 6, 4)))
    x[0, :, 0] = 0
    fast, reference = TensorStageAccumulator(4, pairs=6), ConceptStageAccumulator(4, pairs=6)
    for batch in (x[:2], x[2:]):
        fast.add_tensor(batch)
        reference.add(batch.numpy())
    a, b = fast.report(), reference.report()
    np.testing.assert_allclose(fast.cross, reference.cross, atol=1e-12)
    assert a["zero_norm_concept_vectors"] == b["zero_norm_concept_vectors"] == 1
    assert a["all_pair_document_cosine_mean"] == pytest.approx(b["all_pair_document_cosine_mean"])
    np.testing.assert_allclose(a["sample_axis_centered_covariance"]["eigenvalues_descending"],
                               b["sample_axis_centered_covariance"]["eigenvalues_descending"], atol=1e-12)


def add_probe_document(probe, z, idx=0, coverage_complete=True):
    mask = torch.ones((1, len(z)), dtype=torch.bool)
    _, alpha, stages = probe.builder.from_hidden_states(z[None], mask, return_stages=True)
    units = [{"text": f"word{i}", "start": i, "end": i + 1} for i in range(len(z))]
    return probe.add_document({"id": str(idx), "row_index": idx}, z, alpha[0],
        {k: v[0] for k, v in stages.items()}, token_units=units, chunk_vectors=z,
        chunk_units=units, coverage={"complete": coverage_complete}, evidence_concepts=["a"])


def test_common_mean_identity_signal_amplitude_and_native_baseline():
    builder = make_builder()
    probe = AttentionDesignDiagnostics(builder, ["a", "b", "c"], pairs=3, temperatures=[.05])
    z = torch.tensor([[100., .2, 0, 0], [100., -.2, .1, 0], [100., .1, -.1, 0]])
    scores, examples = add_probe_document(probe, z)
    add_probe_document(probe, z + torch.tensor([2., 0., 0., 0.]), 1)
    logits = cosine_logits(z, builder.prototypes)
    alpha = torch.softmax(logits / builder.temperature, 0)
    raw = z.T @ alpha
    residual = (z - z.mean(0)).T @ alpha
    torch.testing.assert_close(residual, raw - z.mean(0)[:, None], atol=1e-5, rtol=1e-4)
    report = probe.report()
    assert report["statistics"]["token_mean_energy_fraction"]["mean"] > .999
    assert report["statistics"]["residual_to_raw_norm_ratio"]["mean"] < .01
    assert report["variants"]["fp32_raw_values"]["stages"]["pre_pca"]["concept_gram"]["lambda1_over_trace"] > .99
    assert report["statistics"]["native_vs_fp32_full_relative_distance"]["quantiles"]["max"] < 1e-6
    assert report["variants"]["uniform_raw_values"]["stages"]["pre_pca"]["concept_gram"]["lambda1_over_trace"] == pytest.approx(1)
    assert examples[0]["relation"] == "" and examples[0]["suggestion_is_not_label"]
    assert len(scores) == 3
    json.dumps(report, allow_nan=False)


def test_unit_value_control_holds_attention_fixed_and_removes_norm_dominance():
    z = torch.tensor([[1000., 0., 0., 0.], [0., 1., 0., 0.]])
    p = torch.eye(4)[:2]
    alpha = torch.softmax(cosine_logits(z, p), 0)
    raw = z.T @ alpha
    unit = torch.nn.functional.normalize(z, dim=1).T @ alpha
    assert float(torch.nn.functional.cosine_similarity(raw[:, 0], raw[:, 1], dim=0)) > .999
    assert float(torch.nn.functional.cosine_similarity(unit[:, 0], unit[:, 1], dim=0)) < .8


def test_zero_reference_is_undefined_and_incomplete_chunks_not_labeled_comparable():
    assert torch.isnan(relative_distance(torch.ones((2, 1)), torch.zeros((2, 1)))).all()
    probe = AttentionDesignDiagnostics(make_builder(), ["a", "b", "c"], pairs=3)
    scores, _ = add_probe_document(probe, torch.zeros((2, 4)), coverage_complete=False)
    assert all(s["chunk_max_cosine"] is None and s["raw_to_uniform_relative_distance"] is None for s in scores)
    json.dumps(scores, allow_nan=False)


def test_source_spans_long_sentences_and_explicit_chunk_coverage():
    text = "First routing sentence.\nSecond security sentence! 最后的句子。"
    units, coverage = split_evidence(text, max_chars=16, max_units=64)
    assert coverage["complete"]
    assert all(u["text"] == text[u["start"]:u["end"]] and len(u["text"]) <= 16 for u in units)
    assert "".join(u["text"].replace(" ", "") for u in units) == text.replace(" ", "").replace("\n", "")
    units, coverage = split_evidence(text, max_chars=16, max_units=1)
    assert not coverage["complete"] and coverage["dropped_units"] > 0
    with pytest.raises(ValueError):
        split_evidence(" \n ")


def test_human_evaluation_ties_and_common_subset_without_inferred_negatives():
    assert binary_auc([1, 2], [1, 0]) == pytest.approx(.875)
    assert binary_auc([], [1]) is None
    rows = [{"document_id": "d", "relation": relation, "token_max_cosine": t, "chunk_max_cosine": c}
            for relation, t, c in [("relevant", .8, .9), ("hard_negative", .8, .1),
                                   ("unknown", 1., 1.), ("unrelated", .1, None)]]
    result = annotation_metrics(rows)
    assert result["n_labeled_pairs"] == 3 and result["n_comparable_pairs"] == 2
    assert result["methods"]["token_max_cosine"]["pooled_auc"] == .5
    assert result["methods"]["chunk_max_cosine"]["pooled_auc"] == 1
    assert annotation_metrics([])["status"] == "pending_human_labels"


def test_saved_labels_evaluate_without_loading_encoder_and_reject_bad_pairs(tmp_path, monkeypatch):
    import scripts.diagnose_attention_design as cli
    scores = tmp_path / "scores.jsonl"
    scores.write_text(json.dumps({"document_id": "d", "concept_id": "a", "token_max_cosine": .5,
                                 "chunk_max_cosine": .7}) + "\n")
    labels = tmp_path / "labels.csv"
    labels.write_text("document_id,concept_id,relation\nd,a,relevant\n")
    monkeypatch.setattr(cli, "Qwen3EmbeddingEncoder", lambda *a, **kw: pytest.fail("Must not load encoder"))
    cli.main(["--evaluate-only", str(scores), "--annotations", str(labels), "--output-dir", str(tmp_path / "evaluated")])
    result = json.loads((tmp_path / "evaluated/annotation_metrics.json").read_text())
    assert result["n_labeled_pairs"] == 1
    assert result["methods"]["token_max_cosine"]["pooled_auc"] is None
    with pytest.raises(ValueError, match="absent"):
        cli.evaluate_saved(scores, {("x", "a"): {"relation": "unrelated"}})
    labels.write_text("document_id,concept_id,relation\nd,a,negative\n")
    with pytest.raises(ValueError, match="Relation"):
        cli.load_annotations(labels)


def test_sample_replay_is_exact_and_fails_closed_on_hash_or_ids(tmp_path):
    from graph_mvp.diagnostic_sampling import file_hash
    from scripts.diagnose_attention_design import replay_sample
    source = tmp_path / "train.csv"
    source.write_text("patent_id,text\na,first\nb,second\nc,third\n")
    manifest = tmp_path / "prior.json"
    value = {"provenance": {"train_sha256": file_hash(source)},
             "sample_rows": [{"id": "a", "row_index": 0}, {"id": "c", "row_index": 2}]}
    manifest.write_text(json.dumps(value))
    assert [r["id"] for r in replay_sample(source, manifest, file_hash(source))] == ["a", "c"]
    with pytest.raises(ValueError, match="hash"):
        replay_sample(source, manifest, "other")
    value["sample_rows"][0]["id"] = "wrong"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="ID"):
        replay_sample(source, manifest, file_hash(source))


def test_cli_all_five_probes_and_repeatability(tmp_path):
    from scripts.diagnose_attention_design import main
    paths = [tmp_path / "first", tmp_path / "second"]
    for path in paths:
        main(["--synthetic", "--max-samples", "6", "--representation-dim", "4", "--pairs", "8",
              "--review-documents", "2", "--output-dir", str(path)])
    a = json.loads((paths[0] / "report.json").read_text())
    b = json.loads((paths[1] / "report.json").read_text())
    assert a == b
    assert a["human_evidence"]["status"] == "pending_human_labels"
    assert a["n_documents"] == 6 and len(a["variants"]) == 8
    assert set(a["variants"]["chunk_pooled"]["stages"]) == {
        "pre_pca", "post_pca_pre_l2", "post_l2", "cache_quantized", "rank_gaussian"}
    assert len((paths[0] / "scores.jsonl").read_text().splitlines()) == 6 * 16
    with (paths[0] / "annotations_template.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len({r["document_id"] for r in rows}) == 2 and all(not r["relation"] for r in rows)
    assert json.loads((paths[0] / "run_manifest.json").read_text())["status"] == "complete"
    with pytest.raises(SystemExit):
        main(["--synthetic", "--output-dir", str(paths[0])])


def test_real_cli_replays_sample_excludes_padding_and_exports_labeled_evidence(tmp_path, monkeypatch):
    import re
    import scripts.diagnose_attention_design as cli
    from graph_mvp.patient_repr import ConceptVocabulary, save_concept_prototypes
    from graph_mvp.diagnostic_sampling import file_hash
    class Tokenizer:
        def __init__(self):
            self.words = {}
        def __call__(self, texts, **kwargs):
            rows = []
            for text in texts:
                row = []
                for match in list(re.finditer(r"\S+", text))[:kwargs["max_length"]]:
                    word = match.group()
                    self.words.setdefault(word, len(self.words) + 1)
                    row.append((self.words[word], match.span()))
                rows.append(row)
            width = max(map(len, rows))
            ids = [[0] * (width - len(r)) + [x[0] for x in r] for r in rows]
            masks = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
            offsets = [[(0, 0)] * (width - len(r)) + [x[1] for x in r] for r in rows]
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
                    "offset_mapping": torch.tensor(offsets)}
        def convert_ids_to_tokens(self, ids):
            reverse = {v: k for k, v in self.words.items()}
            return [reverse[i] for i in ids]
    class Encoder:
        hidden_size = 4
        def __init__(self, *args, max_length, **kwargs):
            self.max_length, self.tokenizer = max_length, Tokenizer()
        def encode_token_states(self, texts):
            batch = self.tokenizer(texts, max_length=self.max_length)
            ids = batch["input_ids"].float()
            z = torch.stack([torch.sin(ids), torch.cos(ids), ids / 10, torch.ones_like(ids)], -1)
            z[~batch["attention_mask"].bool()] = 1e6  # Would dominate if padding leaked.
            return z, batch["attention_mask"]
    source = tmp_path / "train.csv"
    source.write_text("patent_id,text\na,Routing packet. Security network.\nb,Short packet.\nc,Other patent.\n")
    prototypes = tmp_path / "prototypes.npz"
    save_concept_prototypes(prototypes, np.eye(4)[:3], ConceptVocabulary(("x", "y", "z"),
        ("routing", "security", "unrelated")), encoder_id="tiny")
    projection = tmp_path / "projection.npz"
    np.savez(projection, mean=np.zeros(4), matrix=np.eye(4)[:2])
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"sample_rows": [{"id": "a", "row_index": 0}, {"id": "b", "row_index": 1}],
        "provenance": {"train_sha256": file_hash(source), "projection_file_sha256": file_hash(projection), "max_length": 8},
        "projection": {"temperature": .1}}))
    labels = tmp_path / "labels.csv"
    labels.write_text("document_id,concept_id,relation\na,x,relevant\na,z,unrelated\n")
    monkeypatch.setattr(cli, "Qwen3EmbeddingEncoder", Encoder)
    output = tmp_path / "run"
    cli.main(["--train", str(source), "--prototypes", str(prototypes), "--model", "tiny", "--max-length", "8",
              "--max-samples", "2", "--sample-manifest", str(prior), "--projection-file", str(projection),
              "--representation-dim", "2", "--annotations", str(labels), "--output-dir", str(output), "--pairs", "3"])
    report = json.loads((output / "report.json").read_text())
    assert report["provenance"]["sampling"] == "exact_prior_training_rows"
    assert [r["active_tokens"] for r in report["sample_rows"]] == [4, 2]
    assert report["statistics"]["token_norm"]["quantiles"]["max"] < 2
    assert report["human_evidence"]["n_comparable_pairs"] == 2
    evidence = json.loads((output / "evidence_review.json").read_text())
    assert {e["relation"] for e in evidence if e["document_id"] == "a"} == {"relevant", "unrelated"}
    assert all(e["chunk_coverage_complete"] for e in evidence)
    assert all(unit["text"] == e["visible_document_text"][unit["start"]:unit["end"]]
               for e in evidence for unit in e["token_evidence"] + e["chunk_evidence"])
