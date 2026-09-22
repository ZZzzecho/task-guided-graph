#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

from graph_mvp.patient_repr import (
    DEFAULT_PATIENT_ENCODER, Qwen3EmbeddingEncoder, PatientConceptMatrixBuilder,
    build_patient_cache_from_frame, load_concept_prototypes,
)


def main():
    p = argparse.ArgumentParser(description="Build frozen Patient-MNGM H_d cache from prepared D_train")
    p.add_argument("--train", type=Path, required=True, help="Prepared train.csv or train.csv.gz containing a text column")
    p.add_argument("--prototypes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_PATIENT_ENCODER)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--cache-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-patients", type=int, default=None, help="Deterministic prefix for smoke/scale tests")
    p.add_argument("--flash-attention", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Install pandas: pip install -e '.[patient]'") from exc
    frame = pd.read_csv(args.train)
    if args.max_patients is not None:
        if args.max_patients < 2:
            raise SystemExit("--max-patients must be >=2")
        frame = frame.iloc[:args.max_patients].copy()
    prototypes, meta = load_concept_prototypes(args.prototypes)
    if str(meta.get("encoder_id")) != str(args.model):
        raise SystemExit(
            f"Prototype encoder mismatch: cache={meta.get('encoder_id')!r}, requested={args.model!r}"
        )
    encoder = Qwen3EmbeddingEncoder(
        args.model, device=args.device, dtype=args.dtype,
        local_files_only=args.local_files_only, max_length=args.max_length,
        attn_implementation="flash_attention_2" if args.flash_attention else None,
    )
    builder = PatientConceptMatrixBuilder(encoder, prototypes, temperature=args.temperature)
    ds = build_patient_cache_from_frame(
        frame, builder, args.output, meta["concept_ids"], batch_size=args.batch_size,
        shard_size=args.shard_size, cache_dtype=args.cache_dtype,
        encoder_id=args.model, overwrite=args.overwrite,
    )
    print(f"cached n={len(ds)} shape=[N,{ds.hidden_size},{ds.num_concepts}] -> {args.output}")


if __name__ == "__main__":
    main()
