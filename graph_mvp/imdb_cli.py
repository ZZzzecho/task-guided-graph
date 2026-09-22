import argparse
import json

from .imdb import prepare_imdb


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Prepare Stanford aclImdb train/graph/val/test splits"
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to extracted aclImdb directory containing train/ and test/",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    summary = prepare_imdb(args.input, args.output, seed=args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
