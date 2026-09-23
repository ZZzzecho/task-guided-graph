#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from graph_mvp.patent_retrieval import build_fixed_candidate_pools


def main():
    p = argparse.ArgumentParser(description="Build fixed H04L citation-retrieval candidate pools")
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="graph", choices=["train", "graph", "val", "test"])
    p.add_argument("--negatives-per-query", type=int, default=63)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--min-shared-concepts", type=int, default=1)
    args = p.parse_args()

    summary = build_fixed_candidate_pools(
        args.prepared_dir,
        args.output,
        split=args.split,
        negatives_per_query=args.negatives_per_query,
        max_queries=args.max_queries,
        seed=args.seed,
        min_shared_concepts=args.min_shared_concepts,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
