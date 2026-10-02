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


def replay_training_rows(path, manifest, expected_hash):
    """Replay a bounded diagnostic's exact source rows, rejecting changed data."""
    import json
    previous = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if previous.get("provenance", {}).get("train_sha256") != expected_hash:
        raise ValueError("Sample manifest training hash does not match --train")
    selected = previous["sample_rows"]
    wanted = {int(r["row_index"]): str(r.get("id", r.get("document_id"))) for r in selected}
    if len(wanted) != len(selected) or not 2 <= len(wanted) <= 512 or min(wanted) < 0:
        raise ValueError("Sample manifest must have 2..512 unique nonnegative row indices")
    op = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with op(path, "rt", encoding="utf-8-sig", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i in wanted:
                doc_id = str(row.get("patent_id", row.get("subject_id", str(i))))
                if doc_id != wanted[i] or not row["text"].strip():
                    raise ValueError("Sample manifest ID/text does not match source row")
                rows.append({"row_index": i, "id": doc_id, "text": row["text"]})
    if len(rows) != len(wanted):
        raise ValueError("Sample manifest contains missing training rows")
    return rows


class SyntheticJointEncoder:
    """Deterministic instrumentation only; random hash vectors have no semantics."""
    hidden_size = 32
    model_name_or_path = "synthetic_joint_instrumentation"

    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            ids = [ord(x) + 1 for x in text]
            return ids + [1] if add_special_tokens else ids

    def __init__(self, max_length=512, seed=17):
        self.max_length, self.seed = max_length, seed
        self.tokenizer = self.Tokenizer()

    def _vector(self, text):
        key = sha256((str(self.seed) + text).encode()).digest()
        return np.random.default_rng(int.from_bytes(key[:8], "little")).normal(size=self.hidden_size)

    def encode_pooled(self, texts, normalize=True):
        import torch
        import torch.nn.functional as F
        x = torch.tensor(np.stack([self._vector(t) for t in texts]), dtype=torch.float32)
        return F.normalize(x, dim=1) if normalize else x

    def encode_token_states(self, texts):
        import torch
        n = max(min(len(t) + 1, self.max_length) for t in texts)
        z = np.zeros((len(texts), n, self.hidden_size), dtype=np.float32)
        mask = np.zeros((len(texts), n), dtype=bool)
        for i, text in enumerate(texts):
            length = min(len(text) + 1, self.max_length)
            for j in range(length):
                z[i, j] = self._vector(text[:j + 1])
            mask[i, :length] = True
        return torch.tensor(z), torch.tensor(mask)


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
