#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Download Qwen3-Embedding-0.6B for offline Patient-MNGM use")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--revision", default=None)
    args = p.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub: pip install -e '.[patient]'") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=str(args.output),
    )
    print(path)


if __name__ == "__main__":
    main()
