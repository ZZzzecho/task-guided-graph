import gzip
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from graph_mvp.diagnostic_sampling import SyntheticJointEncoder, file_hash
from graph_mvp.joint_evidence_repr import JointEvidenceMatrixBuilder, select_and_pack_evidence, split_document
from graph_mvp.patient_repr import ConceptVocabulary, PatientMatrixDataset, build_patient_cache_from_frame, save_concept_prototypes
from scripts.diagnose_joint_controls import recover_evidence


class Encoder(SyntheticJointEncoder):
    hidden_size = 80
    model_name_or_path = "test80"

    def encode_pooled(self, texts, **kwargs):
        return super().encode_pooled(texts, **kwargs).to(torch.bfloat16)


def prepare(tmp_path, n=2):
    data = tmp_path / "data"
    data.mkdir()
    with gzip.open(data / "train.csv.gz", "wt", encoding="utf-8") as fh:
        fh.write("patent_id,text\n")
        for i in range(n):
            fh.write(f"{i},Routing example {i}. Security keys {i}. Packet queues {i}. More scheduling {i}. Fairness {i}. Access {i}. Wireless {i}.\n")
    vocab = ConceptVocabulary(("r", "s", "q"), ("routing", "security", "queue"))
    (data / "concepts.json").write_text(json.dumps(dict(zip(vocab.concept_ids, vocab.concept_texts))))
    save_concept_prototypes(data / "concept_prototypes.npz", np.eye(80)[:3], vocab, encoder_id="test80")
    projection = tmp_path / "projection.npz"
    np.savez(projection, mean=np.zeros(80), matrix=np.eye(80)[:64])
    config = tmp_path / "config.json"
    config.write_text("{}")
    for split in ("train", "graph", "val", "test"):
        (data / f"retrieval_{split}_candidates.jsonl").write_text("{}\n")
    return ["--data-dir", str(data), "--projection-file", str(projection), "--config", str(config),
            "--qwen-model", "test80", "--task-model", "test_task"]


def test_top6_adds_original_chunks_to_same_F_and_short_doc_is_not_padded():
    encoder = Encoder(1024)
    chunks = split_document("One routing. Two keys. Three queues. Four access. Five fairness. Six network. Seven packets.", encoder.tokenizer)
    scores = np.arange(len(chunks))
    a = select_and_pack_evidence("routing", chunks, scores, encoder.tokenizer, top_k=3, max_input_tokens=1024)
    b = select_and_pack_evidence("routing", chunks, scores, encoder.tokenizer, top_k=6, max_input_tokens=1024)
    assert len(a["chunks"]) == 3 and len(b["chunks"]) == 6
    assert {c["index"] for c in a["chunks"]} < {c["index"] for c in b["chunks"]}
    assert b["requested_top_k"] == 6 and b["selected_chunk_count"] == 6 and not b["truncated"]
    short = select_and_pack_evidence("routing", chunks[:2], scores[:2], encoder.tokenizer, top_k=6, max_input_tokens=1024)
    assert short["selected_chunk_count"] == 2


def test_compact_audit_reconstructs_exact_F_inputs_and_progress_matches_cache(tmp_path):
    import pandas as pd
    encoder = Encoder(1024)
    builder = JointEvidenceMatrixBuilder(encoder, np.eye(80)[:3], ["routing", "security", "queue"],
        top_k=6, representation_dim=64, projection_matrix=np.eye(80)[:64], projection_mean=np.zeros(80))
    text = "Routing first. Security second. Queue third. More access. Fifth network. Sixth packets. Seventh context."
    frame = pd.DataFrame({"text": [text, text + " Final variant."], "id": ["a", "b"]})
    progress = []
    ds = build_patient_cache_from_frame(frame, builder, tmp_path / "cache", ["r", "s", "q"],
        subject_col="id", stay_col="id", cache_dtype="float32", batch_size=1, shard_size=1,
        compact_evidence_audit=True, progress_callback=progress.append)
    assert progress[-1] == {"documents_done": 2, "documents_total": 2, "persisted_documents": 2}
    assert ds.metadata["representation_pipeline"]["evidence_audit_format"] == "source_spans_v1"
    documents = [json.loads(line) for line in (tmp_path / "cache/evidence.jsonl").read_text().splitlines()]
    for doc in documents:
        for row in doc["concept_evidence"]:
            assert "input_text" not in row and all("text" not in c for c in row["chunks"])
            evidence = recover_evidence(doc, row)
            assert len(evidence) and len(row["chunks"]) == 6
    start = documents[0]["concept_evidence"][0]["chunks"][0]["start"]
    source = documents[0]["source_text"]
    documents[0]["source_text"] = source[:start] + "X" + source[start + 1:]
    with pytest.raises(ValueError, match="hash mismatch"):
        for row in documents[0]["concept_evidence"]:
            recover_evidence(documents[0], row)


def test_full_default_plan_uses_2048_docs_top6_and_existing_full_budget(tmp_path):
    from scripts.run_joint_evidence_full import main
    common = prepare(tmp_path, 2048)
    out = tmp_path / "plan"
    main([*common, "--output", str(out), "--plan-only"])
    manifest = json.loads((out / "experiment_manifest.json").read_text())
    assert manifest["status"] == "plan_only" and not (out / "cache").exists()
    assert manifest["signature"]["documents"] == 2048
    build, train = manifest["commands"]["build"], manifest["commands"]["train"]
    def value(command, flag):
        return command[command.index(flag) + 1]
    assert value(build, "--evidence-top-k") == "6" and value(build, "--max-length") == "1024"
    for flag, expected in (("--phases", "8"), ("--max-train-queries", "1024"), ("--warmup-steps", "50"),
                           ("--adapt-steps", "100"), ("--max-mngm-documents", "2048")):
        assert value(train, flag) == expected
    assert "--test-pools" in train and "controls" in manifest["commands"]
    assert manifest["planned_joint_pairs"] == 2048 * 3
    with pytest.raises(SystemExit):
        main([*common, "--output", str(tmp_path / "too_many"), "--documents", "2049", "--plan-only"])


@pytest.mark.parametrize("controls_fail", [False, True])
def test_sequential_pipeline_real_bf16_cache_controls_and_safe_reuse(tmp_path, monkeypatch, controls_fail):
    import scripts.run_joint_evidence_full as pipeline
    import scripts.build_patient_matrices as build
    import scripts.diagnose_joint_controls as controls
    common = prepare(tmp_path)
    monkeypatch.setattr(build, "Qwen3EmbeddingEncoder", lambda *a, **kw: Encoder(kw["max_length"]))
    monkeypatch.setattr(controls, "Qwen3EmbeddingEncoder", lambda *a, **kw: Encoder(kw["max_length"]))
    calls = []
    def run(command, **kwargs):
        module = command[3]
        calls.append(module)
        if module == "scripts.build_patient_matrices":
            build.main(command[4:])
        elif module == "scripts.diagnose_joint_controls":
            if controls_fail:
                raise subprocess.CalledProcessError(1, command)
            controls.main(command[4:])
        else:
            cache = Path(command[command.index("--cache") + 1])
            assert PatientMatrixDataset(cache).materialize().shape == (2, 64, 3)
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            (output / "summary.json").write_text("{}")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    out = tmp_path / "full"
    pipeline.main([*common, "--documents", "2", "--output", str(out)])
    assert calls == ["scripts.build_patient_matrices", "scripts.diagnose_joint_controls", "scripts.run_patent_graph_rl"]
    manifest = json.loads((out / "experiment_manifest.json").read_text())
    assert manifest["status"] == ("training_complete_controls_failed" if controls_fail else "complete")
    if not controls_fail:
        report = json.loads((out / "controls/report.json").read_text())
        assert report["stages"]["matched"]["n_samples"] == 2
        assert len(report["evidence_diversity_all_documents"]) == 2
        assert all(d["donor_row"] != row["row_index"] and d["relation_label"] is None
                   for row in report["comparisons"] for d in row["donors"])
        control_manifest = json.loads((out / "controls/run_manifest.json").read_text())
        assert control_manifest["outputs"]["report.json"] == file_hash(out / "controls/report.json")
    calls.clear()
    pipeline.main([*common, "--documents", "2", "--output", str(tmp_path / "reused"),
                   "--reuse-cache", str(out / "cache"), "--skip-controls"])
    assert calls == ["scripts.run_patent_graph_rl"]
    with pytest.raises(SystemExit):
        pipeline.main([*common, "--documents", "2", "--evidence-top-k", "3",
                       "--output", str(tmp_path / "bad_reuse"), "--reuse-cache", str(out / "cache")])
    matrix_file = out / "cache/shard_00000.npy"
    x = np.load(matrix_file)
    x[0, 0, 0] += .1
    np.save(matrix_file, x)
    with pytest.raises(SystemExit):
        pipeline.main([*common, "--documents", "2", "--output", str(tmp_path / "corrupt_reuse"),
                       "--reuse-cache", str(out / "cache")])


def test_failed_build_never_starts_training_and_persists_failure(tmp_path, monkeypatch):
    import scripts.run_joint_evidence_full as pipeline
    common = prepare(tmp_path)
    calls = []
    def fail(command, **kwargs):
        calls.append(command[3])
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    out = tmp_path / "failed"
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.main([*common, "--documents", "2", "--output", str(out)])
    assert calls == ["scripts.build_patient_matrices"]
    assert json.loads((out / "experiment_manifest.json").read_text())["status"] == "failed"


def test_graph_diagnostics_and_archive_use_actual_effective_covariance_and_B(tmp_path):
    from graph_mvp.types import GraphSnapshot, GraphState
    from scripts.run_patent_graph_rl import _covariance_report, _save_graph
    s = np.array([[2., .4, .1], [.4, 1., .2], [.1, .2, 3.]])
    b = np.diag([1.5, .5])
    snapshot = GraphSnapshot(("r", "s", "q"), s, np.ones((3, 3)) - np.eye(3), np.eye(3),
        estimator_kind="patient_concept_matrix", auxiliary={"representation_precision": b})
    state = GraphState("example", 0, snapshot, {"converged": True})
    report = _covariance_report(state)
    np.testing.assert_allclose(report["concept_covariance"]["eigenvalues_descending"], np.linalg.eigvalsh(s)[::-1])
    np.testing.assert_allclose(report["representation_precision"]["eigenvalues_descending"], [1.5, .5])
    assert report["solver_info"]["converged"]
    _save_graph(tmp_path / "graph.npz", state)
    with np.load(tmp_path / "graph.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["S"], s)
        np.testing.assert_array_equal(saved["representation_precision"], b)
