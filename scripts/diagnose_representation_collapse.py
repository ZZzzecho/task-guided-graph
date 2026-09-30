#!/usr/bin/env python
"""Diagnose 128-512 training documents without rebuilding the full cache."""
from __future__ import annotations

import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np

from graph_mvp.patient_repr import Qwen3EmbeddingEncoder, PatientConceptMatrixBuilder, load_concept_prototypes
from graph_mvp.representation_diagnostics import diagnose_representations


def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample_training_rows(path, n, seed):
    """Seeded reservoir sample; no full corpus in memory or pandas dependency."""
    op = gzip.open if str(path).endswith(".gz") else open
    rng = random.Random(seed)
    reservoir = []
    with op(path, "rt", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if "text" not in (reader.fieldnames or []):
            raise ValueError("Training CSV requires text column")
        for i, row in enumerate(reader):
            item = {"row_index": i, "id": row.get("patent_id", row.get("subject_id", str(i))),
                    "text": row["text"]}
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
    """Synthetic instrumentation sanity check, not evidence about Qwen/patents."""
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--prototypes", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--representation-dim", type=int, default=64)
    parser.add_argument("--projection-file", type=Path,
                        help="Reuse the actual cache projection.npz; otherwise fit the same prototype PCA")
    parser.add_argument("--projection-seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-hidden", type=int, default=32)
    parser.add_argument("--synthetic-concepts", type=int, default=16)
    args = parser.parse_args(argv)
    if not 2 <= args.max_samples <= 512 or not 1 <= args.batch_size <= 8 or args.pairs < 1:
        parser.error("Use 2..512 samples, 1..8 batch size, and positive pairs")
    if args.output.exists():
        parser.error("Output exists; choose a new report path")
    provenance = {"synthetic": args.synthetic, "sampling": "seeded_reservoir_of_training_rows"}
    if args.synthetic:
        if any(x is not None for x in (args.train, args.prototypes, args.model)):
            parser.error("--synthetic cannot be combined with real training/model inputs")
        encoder = SyntheticEncoder(args.synthetic_hidden, args.seed)
        rng = np.random.default_rng(args.seed)
        prototypes = rng.normal(size=(args.synthetic_concepts, args.synthetic_hidden)).astype(np.float32)
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
        rows = [{"id": str(i), "row_index": i, "text": f"synthetic-{i}"} for i in range(args.max_samples)]
        provenance["sampling"] = "synthetic_instrumentation_only"
    else:
        if any(x is None for x in (args.train, args.prototypes, args.model)):
            parser.error("Real diagnostics require --train, --prototypes, and --model")
        prototypes, meta = load_concept_prototypes(args.prototypes)
        if str(meta.get("encoder_id")) != args.model:
            parser.error("Prototype encoder ID differs from --model")
        rows = sample_training_rows(args.train, args.max_samples, args.seed)
        provenance.update({"train": str(args.train), "train_sha256": file_hash(args.train),
                           "prototypes": str(args.prototypes), "prototypes_sha256": file_hash(args.prototypes),
                           "encoder_id": args.model, "max_length": args.max_length})
        encoder = Qwen3EmbeddingEncoder(args.model, device=args.device, dtype=args.dtype,
                                        local_files_only=args.local_files_only, max_length=args.max_length)
    mean = matrix = None
    if args.projection_file is not None:
        with np.load(args.projection_file, allow_pickle=False) as projection:
            mean, matrix = projection["mean"], projection["matrix"]
        provenance["projection_file"] = str(args.projection_file)
        provenance["projection_file_sha256"] = file_hash(args.projection_file)
    builder = PatientConceptMatrixBuilder(encoder, prototypes, temperature=args.temperature,
                                          representation_dim=args.representation_dim,
                                          projection_matrix=matrix, projection_mean=mean,
                                          projection_seed=args.projection_seed)
    def batches():
        for start in range(0, len(rows), args.batch_size):
            print(f"diagnostic documents {start+1}..{min(start+args.batch_size, len(rows))}/{len(rows)}", flush=True)
            yield [x["text"] for x in rows[start:start+args.batch_size]]
    report = diagnose_representations(builder, batches(), seed=args.seed, pairs=args.pairs,
                                     cache_dtype=args.cache_dtype, max_samples=args.max_samples)
    report["provenance"] = provenance
    report["sample_rows"] = [{"id": x["id"], "row_index": x["row_index"]} for x in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("stage | mean cosine | lambda1/trace | entropy rank | participation rank")
    for name, stage in report["stages"].items():
        spec = stage["concept_gram"]
        print(f"{name} | {stage['all_pair_document_cosine_mean']} | {spec['lambda1_over_trace']} | "
              f"{spec['effective_rank_entropy']} | {spec['effective_rank_participation']}")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
