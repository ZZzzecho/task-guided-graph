#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

from graph_mvp.patient_repr import (
    DEFAULT_PATIENT_ENCODER, Qwen3EmbeddingEncoder, load_concept_vocabulary,
    build_concept_prototypes, save_concept_prototypes,
)


def main():
    p = argparse.ArgumentParser(description="Build frozen Qwen3-Embedding concept prototypes")
    p.add_argument("--concepts", type=Path, required=True, help="JSON or CSV concept vocabulary")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_PATIENT_ENCODER)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--flash-attention", action="store_true")
    args = p.parse_args()
    vocab = load_concept_vocabulary(args.concepts)
    encoder = Qwen3EmbeddingEncoder(
        args.model, device=args.device, dtype=args.dtype,
        local_files_only=args.local_files_only, max_length=args.max_length,
        attn_implementation="flash_attention_2" if args.flash_attention else None,
    )
    values = build_concept_prototypes(encoder, vocab, batch_size=args.batch_size)
    save_concept_prototypes(args.output, values, vocab, encoder_id=args.model)
    print(f"saved {values.shape} -> {args.output}")


if __name__ == "__main__":
    main()
