import argparse
import json
from .mimic_ed import prepare_mimic_ed


def main(argv=None):
    p = argparse.ArgumentParser(description="Prepare MIMIC-IV-ED HOME vs ADMITTED subject-level splits")
    p.add_argument("--triage", required=True)
    p.add_argument("--edstays", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--without-acuity", action="store_true")
    args = p.parse_args(argv)
    summary = prepare_mimic_ed(args.triage, args.edstays, args.output,
                               seed=args.seed, include_acuity=not args.without_acuity)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
