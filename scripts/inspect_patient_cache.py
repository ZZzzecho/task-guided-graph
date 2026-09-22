#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from graph_mvp.patient_repr import PatientMatrixDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cache")
    args = p.parse_args()
    ds = PatientMatrixDataset(args.cache)
    out = {
        "n_samples": len(ds),
        "hidden_size": ds.hidden_size,
        "num_concepts": ds.num_concepts,
        "n_shards": len(ds.shards),
        "dtype": ds.metadata["dtype"],
        "encoder_id": ds.metadata["encoder_id"],
        "source_split": ds.metadata.get("source_split"),
        "frozen_during_graph_rl": ds.metadata.get("frozen_during_graph_rl"),
        "fingerprint": ds.fingerprint(),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
