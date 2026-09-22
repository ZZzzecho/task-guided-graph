#!/usr/bin/env python3
from pathlib import Path
import argparse, json
from graph_mvp.qsar import qsar_dataset
from graph_mvp.headroom import interaction_headroom


def main(argv=None):
    p = argparse.ArgumentParser(description="QSAR interaction headroom diagnostic")
    p.add_argument("csv", type=Path)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    result = interaction_headroom(qsar_dataset(args.csv, seed=args.seed))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
