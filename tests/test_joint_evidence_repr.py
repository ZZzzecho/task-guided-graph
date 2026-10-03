import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from graph_mvp.diagnostic_sampling import SyntheticJointEncoder, file_hash
from graph_mvp.estimators import MNGMEstimator, PATIENT_MATRIX_MODE, _rank_gaussian_tensor
from graph_mvp.joint_evidence_repr import (JointEvidenceMatrixBuilder, format_joint_input,
    select_and_pack_evidence, split_document, token_count)
from graph_mvp.patient_repr import (ConceptVocabulary, PatientMatrixDataset,
    build_patient_cache_from_frame, save_concept_prototypes)


class TinyEncoder(SyntheticJointEncoder):
    hidden_size = 4


def make_builder(**kwargs):
    encoder = kwargs.pop("encoder", TinyEncoder())
    return JointEvidenceMatrixBuilder(encoder, np.eye(4)[:3], ["routing", "security", "queue"],
        representation_dim=2, projection_matrix=np.eye(4)[:2],
        projection_mean=np.full(4, 1000.), **kwargs)


def test_token_bounded_chunks_keep_full_unicode_source_and_late_evidence():
    tokenizer = TinyEncoder.Tokenizer()
    text = "First routing sentence.\n" + "Long network " * 40 + "最后的安全证据！"
    chunks = split_document(text, tokenizer, 32)
    assert len(chunks) > 10
    assert all(c["text"] == text[c["start"]:c["end"]] for c in chunks)
    assert all(token_count(tokenizer, c["text"]) <= 32 for c in chunks)
    assert "".join(c["text"] for c in chunks).replace(" ", "") == text.replace(" ", "").replace("\n", "")
    assert chunks[-1]["text"].endswith("！")


def test_packing_stable_ties_duplicate_removal_source_order_and_exact_budget():
    tokenizer = TinyEncoder.Tokenizer()
    text = "First routing sentence. Second security sentence. Third queue sentence."
    chunks = split_document(text, tokenizer, 64)
    duplicate = {**chunks[0], "index": 3}
    result = select_and_pack_evidence("routing", chunks + [duplicate], [.5, .9, .5, .5], tokenizer,
                                      top_k=3, max_input_tokens=80)
    assert result["input_tokens"] <= 80
    assert result["input_text"].startswith("Concept:\nrouting\n")
    assert [x["index"] for x in result["chunks"]] == sorted(x["index"] for x in result["chunks"])
    assert len({" ".join(x["text"].split()) for x in result["chunks"]}) == len(result["chunks"])
    assert result["truncated"]
    assert result["selection_is_not_relevance_label"]
    for c in result["chunks"]:
        assert c["text"] == text[c["start"]:c["end"]]
    tied = select_and_pack_evidence("routing", chunks, [.5, .5, .5], tokenizer, top_k=1)
    assert tied["chunks"][0]["index"] == 0


def test_direct_joint_matrix_is_only_shared_linear_projection_without_mean_or_l2():
    builder = make_builder(encode_batch_size=2)
    h, full, audits = builder.encode_document("Routing packets. Security key exchange. Queue fairness.")
    assert h.shape == (1, 2, 3) and full.shape == (1, 4, 3)
    torch.testing.assert_close(h, full[:, :2, :])
    torch.testing.assert_close(full.norm(dim=1), torch.ones((1, 3)))
    assert not torch.allclose(h.norm(dim=1), torch.ones((1, 3)))
    expected = builder.encoder.encode_pooled([a["input_text"] for a in audits]).T[None]
    torch.testing.assert_close(full, expected)
    assert torch.equal(builder.projection_mean, torch.zeros(4))
    meta = builder.projection_metadata()
    assert meta["residual_subtraction"] is False and meta["score_gating"] is False
    assert meta["center_before_projection"] is False and meta["per_concept_l2_after_projection"] is False
    assert meta["temperature"] is None


def test_encoding_keeps_concept_order_batch_independence_and_uses_full_document():
    text = "Early unrelated sentence. " * 15 + "Late queue scheduling evidence."
    a = make_builder(top_k=1, chunk_max_tokens=32, encode_batch_size=1)
    b = make_builder(top_k=1, chunk_max_tokens=32, encode_batch_size=8)
    ha, _, audits = a.encode_document(text)
    hb, _, _ = b.encode_document(text)
    torch.testing.assert_close(ha, hb)
    assert audits[0]["candidate_chunks"] > 15
    assert [x["concept_text"] for x in audits] == ["routing", "security", "queue"]
    assert a.stats["joint_inputs"] == 3
    assert a.stats["chunk_inputs"] > 15


def test_independent_concept_ranking_can_share_evidence_without_competition():
    tok = TinyEncoder.Tokenizer()
    chunks = split_document("Shared scheduling evidence. Other evidence.", tok)
    a = select_and_pack_evidence("queue", chunks, [.9, .1], tok, top_k=1)
    b = select_and_pack_evidence("quality", chunks, [.8, .2], tok, top_k=1)
    assert a["evidence_text"] == b["evidence_text"]
    assert a["input_text"] != b["input_text"]


def test_undefined_inputs_and_encoder_outputs_fail_explicitly():
    with pytest.raises(ValueError, match="nonempty"):
        split_document(" \n", TinyEncoder.Tokenizer())
    with pytest.raises(ValueError, match="leave room"):
        select_and_pack_evidence("c" * 1000, [{"index": 0, "text": "x", "start": 0}], [1.],
                                 TinyEncoder.Tokenizer())
    with pytest.raises(ValueError, match="aligned"):
        JointEvidenceMatrixBuilder(TinyEncoder(), np.eye(4)[:3], ["x", "y"])
    with pytest.raises(ValueError, match="nonzero"):
        JointEvidenceMatrixBuilder(TinyEncoder(), np.zeros((3, 4)), ["x", "y", "z"])
    builder = make_builder()
    with pytest.raises(ValueError, match="truncate"):
        builder._pooled(["x" * 1000])
    with pytest.raises(ValueError, match="no output"):
        builder.project_full_states(torch.ones((1, 4, 3)), normalize=True)
    builder.encoder.encode_pooled = lambda texts, **kw: torch.zeros((len(texts), 4))
    with pytest.raises(ValueError, match="zero pooled"):
        builder.encode_document("A patent.")


def test_encoder_must_be_frozen_and_inference_stays_eval():
    encoder = TinyEncoder()
    encoder.model = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="frozen"):
        make_builder(encoder=encoder)
    encoder.model.requires_grad_(False)
    builder = make_builder(encoder=encoder)
    builder.encode_document("Routing packet.")
    assert not encoder.model.training


def test_joint_cache_roundtrip_audit_completion_and_existing_mngm(tmp_path):
    import pandas as pd
    from graph_mvp.config import MNGMConfig, SolverConfig
    from graph_mvp.weighted_glasso import penalty_matrix
    frame = pd.DataFrame({"text": [f"Routing case {i}. Security variant {i % 3}." for i in range(12)],
                          "patent_id": [str(i) for i in range(12)]})
    builder = make_builder()
    ds = build_patient_cache_from_frame(frame, builder, tmp_path / "cache", ["r", "s", "q"],
        batch_size=2, shard_size=5, cache_dtype="float32", subject_col="patent_id", stay_col="patent_id")
    x = ds.materialize()
    assert x.shape == (12, 2, 3)
    assert ds.metadata["representation_pipeline"]["cache_build_complete"]
    assert ds.metadata["representation_reduction"] == "provided_linear_joint"
    assert ds.metadata["temperature"] is None
    mean, matrix = ds.load_projection()
    np.testing.assert_array_equal(mean, np.zeros(4))
    np.testing.assert_array_equal(matrix, np.eye(4)[:2])
    audits = [json.loads(s) for s in (tmp_path / "cache/evidence.jsonl").read_text().splitlines()]
    assert len(audits) == 12 and audits[11]["subject_id"] == "11"
    assert len(audits[0]["concept_evidence"]) == 3
    est = MNGMEstimator(x, PATIENT_MATRIX_MODE,
        MNGMConfig(max_iter=12, tol=1e-3, representation_penalty=.5), SolverConfig(max_iter=1000))
    result = est.solve(penalty_matrix(3, .4))
    assert result.converged and result.Theta.shape == (3, 3)


def test_failed_joint_cache_is_rejected_instead_of_training_partial_data(tmp_path):
    import pandas as pd
    frame = pd.DataFrame({"text": ["Routing example.", ""], "id": ["a", "b"]})
    with pytest.raises(ValueError, match="nonempty"):
        build_patient_cache_from_frame(frame, make_builder(), tmp_path / "cache", ["r", "s", "q"],
            batch_size=1, shard_size=1, subject_col="id", stay_col="id")
    with pytest.raises(ValueError, match="incomplete"):
        PatientMatrixDataset(tmp_path / "cache")


def test_fixed_baseline_subtraction_is_redundant_under_existing_rank_transform():
    x = np.random.default_rng(17).normal(size=(12, 4, 3))
    baseline = np.random.default_rng(2).normal(size=(4, 3))
    np.testing.assert_array_equal(_rank_gaussian_tensor(x), _rank_gaussian_tensor(x - baseline))


def test_diagnostic_synthetic_outputs_repeat_and_hashes_validate(tmp_path):
    from scripts.diagnose_joint_evidence import main
    reports = []
    for name in ("a", "b"):
        output = tmp_path / name
        main(["--synthetic", "--max-samples", "6", "--representation-dim", "4",
              "--pairs", "8", "--review-documents", "2", "--output-dir", str(output)])
        reports.append(json.loads((output / "report.json").read_text()))
        manifest = json.loads((output / "run_manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert all(file_hash(output / p) == h for p, h in manifest["outputs"].items())
        assert PatientMatrixDataset(output / "cache").materialize().shape == (6, 4, 16)
        with (output / "annotations_template.csv").open(newline="") as fh:
            assert all(not r["relation"] for r in csv.DictReader(fh))
    assert reports[0] == reports[1]
    assert reports[0]["pipeline"]["encoding_counts"]["joint_inputs"] == 96
    with pytest.raises(SystemExit):
        main(["--synthetic", "--output-dir", str(tmp_path / "a")])


@pytest.mark.parametrize("native_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cache_dtype", ["float16", "float32"])
def test_real_diagnostic_replay_inputs_trace_and_cache_use_one_encoder(tmp_path, monkeypatch, native_dtype, cache_dtype):
    import scripts.diagnose_joint_evidence as cli
    source = tmp_path / "train.csv"
    source.write_text("patent_id,text\na,Routing packet. Security key.\nb,Queue fairness.\nc,Other text.\n")
    encoder = TinyEncoder()
    original_states = encoder.encode_token_states
    original_pooled = encoder.encode_pooled
    def native_states(texts):
        states, mask = original_states(texts)
        return states.to(native_dtype), mask
    def native_pooled(texts, **kwargs):
        return original_pooled(texts, **kwargs).to(native_dtype)
    encoder.encode_token_states = native_states
    encoder.encode_pooled = native_pooled
    protos = tmp_path / "prototypes.npz"
    save_concept_prototypes(protos, np.eye(4)[:3],
        ConceptVocabulary(("r", "s", "q"), ("routing", "security", "queue")), encoder_id="tiny")
    projection = tmp_path / "projection.npz"
    np.savez(projection, mean=np.full(4, 100.), matrix=np.eye(4)[:2])
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"sample_rows": [{"id": "a", "row_index": 0}, {"id": "c", "row_index": 2}],
        "provenance": {"train_sha256": file_hash(source), "prototypes_sha256": file_hash(protos),
                       "projection_file_sha256": file_hash(projection)}}))
    loaded = []
    def load_encoder(*args, **kw):
        loaded.append(args)
        return encoder
    monkeypatch.setattr(cli, "Qwen3EmbeddingEncoder", load_encoder)
    output = tmp_path / "run"
    cli.main(["--train", str(source), "--prototypes", str(protos), "--model", "tiny",
        "--sample-manifest", str(prior), "--max-samples", "2", "--projection-file", str(projection),
        "--representation-dim", "2", "--cache-dtype", cache_dtype, "--output-dir", str(output)])
    report = json.loads((output / "report.json").read_text())
    assert len(loaded) == 1
    assert report["sample_rows"] == [{"id": "a", "row_index": 0}, {"id": "c", "row_index": 2}]
    assert report["provenance"]["sampling"] == "exact_prior_training_rows"
    assert report["pipeline"]["residual_subtraction"] is False
    assert report["pipeline"]["encoder_native_dtypes"] == [str(native_dtype)]
    assert json.loads((output / "run_manifest.json").read_text())["status"] == "complete"
    dataset = PatientMatrixDataset(output / "cache")
    assert dataset.metadata["dtype"] == cache_dtype
    assert np.isfinite(dataset.materialize()).all()
    audits = [json.loads(s) for s in (output / "evidence.jsonl").read_text().splitlines()]
    assert len(audits) == 2
    assert all(c["text"] == report_text[c["start"]:c["end"]]
        for audit, report_text in zip(audits, ["Routing packet. Security key.", "Other text."])
        for trace in audit["concept_evidence"] for c in trace["chunks"])


def test_build_command_selects_joint_mode_and_preserves_legacy_default(tmp_path, monkeypatch):
    import scripts.build_patient_matrices as cli
    source = tmp_path / "train.csv"
    source.write_text("patent_id,text\na,Routing packet.\nb,Security key.\n")
    protos = tmp_path / "prototypes.npz"
    save_concept_prototypes(protos, np.eye(4)[:3],
        ConceptVocabulary(("r", "s", "q"), ("routing", "security", "queue")), encoder_id="tiny")
    monkeypatch.setattr(cli, "Qwen3EmbeddingEncoder", lambda *a, **kw: TinyEncoder())
    common = ["--train", str(source), "--prototypes", str(protos), "--model", "tiny", "--representation-dim", "2"]
    cli.main([*common, "--output", str(tmp_path / "joint"), "--representation-mode", "joint_evidence"])
    joint = PatientMatrixDataset(tmp_path / "joint")
    assert joint.metadata["representation_pipeline"]["representation_mode"] == "joint_evidence"
    cli.main([*common, "--output", str(tmp_path / "legacy")])
    legacy = PatientMatrixDataset(tmp_path / "legacy")
    assert legacy.metadata["representation_pipeline"]["per_concept_l2_after_projection"] is True
