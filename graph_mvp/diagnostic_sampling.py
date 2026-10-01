"""Training-only sampling shared by standalone diagnostic commands."""
import csv
import gzip
from hashlib import sha256
from pathlib import Path
import random

import numpy as np


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_training_rows(path, n, seed):
    op = gzip.open if str(path).endswith(".gz") else open
    rng, reservoir = random.Random(seed), []
    with op(path, "rt", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if "text" not in (reader.fieldnames or []):
            raise ValueError("Training CSV requires text column")
        for i, row in enumerate(reader):
            item = {"row_index": i, "id": row.get("patent_id", row.get("subject_id", str(i))), "text": row["text"]}
            if not item["text"].strip():
                raise ValueError(f"Empty training text at row {i}")
            if len(reservoir) < n:
                reservoir.append(item)
            else:
                j = rng.randrange(i + 1)
                if j < n:
                    reservoir[j] = item
    if len(reservoir) < n:
        raise ValueError(f"Requested {n} samples, but training CSV has {len(reservoir)}")
    return sorted(reservoir, key=lambda x: x["row_index"])


class SyntheticEncoder:
    """Instrumentation check only; these vectors contain no patent semantics."""
    def __init__(self, hidden_size, seed):
        self.hidden_size = hidden_size
        self.rng = np.random.default_rng(seed)

    def encode_token_states(self, texts):
        import torch
        shared = self.rng.normal(size=(len(texts), 1, self.hidden_size))
        z = shared + .5 * self.rng.normal(size=(len(texts), 12, self.hidden_size))
        mask = torch.ones((len(texts), 12), dtype=torch.bool)
        mask[:, -2:] = False
        return torch.tensor(z, dtype=torch.float32), mask
