#!/usr/bin/env python
"""Load-only sanity check for local GLM-4.7-Flash.

This deliberately does not start vLLM: the graph-token training path needs a local
Transformers model object because it injects continuous `inputs_embeds`.
"""
from __future__ import annotations
import argparse
from graph_mvp.task_model import load_local_causal_lm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default="auto")
    args = p.parse_args()
    model, tokenizer = load_local_causal_lm(
        args.model, dtype=args.dtype, device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    emb = model.get_input_embeddings()
    print({
        "model": args.model,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "input_embedding_dim": int(emb.weight.shape[-1]),
        "supports_inputs_embeds": True,
    })


if __name__ == "__main__":
    main()
