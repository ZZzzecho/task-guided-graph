import json
import pytest

torch = pytest.importorskip("torch")
from graph_mvp.retrieval_state import FrozenCandidateSnapshot
from scripts import run_patent_graph_rl as runner


@pytest.mark.parametrize("kind", ["trainable_parameter", "frozen_parameter", "buffer"])
def test_snapshot_rejects_meta_before_copy_or_hash(kind):
    model = torch.nn.Linear(3, 2, device="meta" if kind != "buffer" else "cpu")
    if kind == "frozen_parameter":
        model.requires_grad_(False)
    if kind == "buffer":
        model.register_buffer("offloaded_buffer", torch.empty(2, device="meta"))
    with pytest.raises(ValueError, match="materialized.*--device-map cuda:0"):
        FrozenCandidateSnapshot(model)


@pytest.mark.parametrize("status", ["running", "complete"])
def test_exception_records_failure_without_overwriting_completed_run(tmp_path, monkeypatch, status):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps({"status": status, "git_sha": "test"}), encoding="utf-8")
    def fail(argv):
        raise RuntimeError("controlled startup failure")
    monkeypatch.setattr(runner, "main", fail)
    with pytest.raises(RuntimeError, match="controlled startup failure"):
        runner.entrypoint(["--output", str(tmp_path), "--policy", "grpo-linear"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["git_sha"] == "test"
    assert saved["status"] == ("failed" if status == "running" else status)
    if status == "running":
        assert saved["error"] == "RuntimeError: controlled startup failure"
        assert saved["failed_at_utc"]


def test_trial_explicitly_materializes_model_on_gpu():
    from pathlib import Path
    script = Path(runner.__file__).with_name("run_linear_penalty_trial.sh").read_text()
    assert "--device-map cuda:0" in script
