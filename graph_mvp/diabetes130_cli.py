import argparse
import json

from .diabetes130 import prepare_diabetes130


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Prepare UCI Diabetes 130-US Hospitals patient-level splits"
    )
    p.add_argument("--input", required=True, help="Path to diabetic_data.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    summary = prepare_diabetes130(
        args.input,
        args.output,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
