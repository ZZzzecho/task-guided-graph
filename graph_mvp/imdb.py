"""Stanford Large Movie Review Dataset (aclImdb) preprocessing."""
from __future__ import annotations

from pathlib import Path
import json


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("IMDb preprocessing requires pandas; install `pip install -e '.[medical]'`") from exc
    return pd


def _read_labeled_dir(root: Path, source_split: str, label_name: str, label: int):
    folder = root / source_split / label_name
    if not folder.is_dir():
        raise FileNotFoundError(f"IMDb directory not found: {folder}")
    rows = []
    for path in sorted(folder.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        rows.append(
            {
                "sample_id": f"{source_split}/{label_name}/{path.name}",
                "source_split": source_split,
                "label": int(label),
                "text": text,
            }
        )
    return rows


def load_imdb_labeled(root):
    """Load the official 25k train + 25k test labeled reviews."""
    pd = _pd()
    root = Path(root)
    rows = []
    rows += _read_labeled_dir(root, "train", "neg", 0)
    rows += _read_labeled_dir(root, "train", "pos", 1)
    rows += _read_labeled_dir(root, "test", "neg", 0)
    rows += _read_labeled_dir(root, "test", "pos", 1)
    frame = pd.DataFrame(rows)
    if len(frame) != 50000:
        raise ValueError(
            f"Expected 50,000 labeled IMDb reviews, found {len(frame)} under {root}"
        )
    return frame


def _split_official_train(frame, seed, proportions=(0.70, 0.15, 0.15)):
    from sklearn.model_selection import train_test_split

    if len(proportions) != 3 or any(x <= 0 for x in proportions):
        raise ValueError("train/graph/val proportions must be positive")
    total = float(sum(proportions))
    props = tuple(float(x) / total for x in proportions)

    train_frame, temp = train_test_split(
        frame,
        train_size=props[0],
        random_state=seed,
        stratify=frame["label"],
    )
    graph_fraction = props[1] / (props[1] + props[2])
    graph_frame, val_frame = train_test_split(
        temp,
        train_size=graph_fraction,
        random_state=seed + 1,
        stratify=temp["label"],
    )
    return {
        "train": train_frame.sort_values("sample_id").reset_index(drop=True),
        "graph": graph_frame.sort_values("sample_id").reset_index(drop=True),
        "val": val_frame.sort_values("sample_id").reset_index(drop=True),
    }


def _describe(frame):
    counts = frame["label"].value_counts().to_dict()
    lengths = frame["text"].str.len()
    return {
        "n_examples": int(len(frame)),
        "negative": int(counts.get(0, 0)),
        "positive": int(counts.get(1, 0)),
        "positive_fraction": float(frame["label"].mean()),
        "text_chars_mean": float(lengths.mean()),
        "text_chars_median": float(lengths.median()),
        "text_chars_max": int(lengths.max()),
    }


def prepare_imdb(root_dir, output_dir, *, seed=7):
    """Create common train/graph/val/test CSV.GZ files.

    The official aclImdb test split is kept untouched as final test. Only the
    official 25k training reviews are split into train/graph/val.
    """
    frame = load_imdb_labeled(root_dir)
    official_train = frame[frame["source_split"] == "train"].copy()
    official_test = frame[frame["source_split"] == "test"].copy()
    splits = _split_official_train(official_train, seed)
    splits["test"] = official_test.sort_values("sample_id").reset_index(drop=True)

    # Sanity checks: all sample IDs are unique and official test never enters search.
    seen = set()
    for name in ("train", "graph", "val", "test"):
        ids = set(splits[name]["sample_id"])
        if seen & ids:
            raise RuntimeError(f"sample leakage involving {name}")
        seen |= ids

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    keep = ["sample_id", "source_split", "label", "text"]
    frame.loc[:, keep].to_csv(
        output / "cohort.csv.gz", index=False, compression="gzip"
    )
    for name, split in splits.items():
        split.loc[:, keep].to_csv(
            output / f"{name}.csv.gz", index=False, compression="gzip"
        )

    summary = {
        "dataset": "Stanford Large Movie Review Dataset (aclImdb)",
        "task": "binary sentiment classification",
        "label_text": {"0": "NEGATIVE", "1": "POSITIVE"},
        "official_test_preserved": True,
        "splits": {name: _describe(df) for name, df in splits.items()},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
