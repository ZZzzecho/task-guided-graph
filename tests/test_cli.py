import json
import numpy as np
import pytest
from graph_mvp.cli import main
from graph_mvp.config import Config


def test_npz_cli_artifacts_and_final_only_holdout(tmp_path, context):
    data = tmp_path / "dataset.npz"
    output = tmp_path / "run"
    np.savez(data, concept_ids=np.array(context.concept_ids), X_train=context.X_train,
             y_train=context.y_train, X_reward=context.X_reward, y_reward=context.y_reward,
             X_test=context.X_reward + .1, y_test=context.y_reward)
    assert main(["--data", str(data), "--rounds", "1", "--num-candidates", "7",
                 "--output", str(output), "--evaluate-test"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["test_evaluated"] is True
    assert "final_test_metrics" in summary
    assert len((output / "trajectory.jsonl").read_text().splitlines()) == 1
    with np.load(output / "final_state.npz", allow_pickle=False) as saved:
        assert saved["Theta"].shape == (3, 3)
        assert saved["concept_ids"].tolist() == list(context.concept_ids)
    with pytest.raises(FileExistsError):
        main(["--data", str(data), "--rounds", "0", "--output", str(output)])


def test_unknown_config_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"typo": {}}')
    with pytest.raises(ValueError):
        Config.load(path)
