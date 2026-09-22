from pathlib import Path
import pandas as pd
import pytest

import graph_mvp.imdb as imdb_mod


def _write_fixture(root: Path, n_per_class=20):
    for split in ("train", "test"):
        for label_name in ("neg", "pos"):
            folder = root / split / label_name
            folder.mkdir(parents=True, exist_ok=True)
            label = 0 if label_name == "neg" else 1
            for i in range(n_per_class):
                text = f"This is review {i}. " + ("bad boring movie" if label == 0 else "great moving movie")
                (folder / f"{i}_{1 if label == 0 else 9}.txt").write_text(text, encoding="utf-8")


def test_prepare_imdb_preserves_official_test(tmp_path, monkeypatch):
    root = tmp_path / "aclImdb"
    _write_fixture(root)

    def small_loader(path):
        rows = []
        for source_split in ("train", "test"):
            for label_name, label in (("neg", 0), ("pos", 1)):
                rows.extend(imdb_mod._read_labeled_dir(Path(path), source_split, label_name, label))
        return pd.DataFrame(rows)

    monkeypatch.setattr(imdb_mod, "load_imdb_labeled", small_loader)
    out = tmp_path / "prepared"
    summary = imdb_mod.prepare_imdb(root, out, seed=11)

    train = pd.read_csv(out / "train.csv.gz")
    graph = pd.read_csv(out / "graph.csv.gz")
    val = pd.read_csv(out / "val.csv.gz")
    test = pd.read_csv(out / "test.csv.gz")

    assert set(test["source_split"]) == {"test"}
    assert set(train["source_split"]) == {"train"}
    assert set(graph["source_split"]) == {"train"}
    assert set(val["source_split"]) == {"train"}
    assert summary["official_test_preserved"] is True

    task = (out / "task.json").read_text(encoding="utf-8")
    assert "NEGATIVE" in task and "POSITIVE" in task

    sets = [set(x["sample_id"]) for x in (train, graph, val, test)]
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            assert a.isdisjoint(b)
