import argparse
import json

from .patents_h04l import prepare_patents_h04l


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Prepare PatentsView H04L patent corpus and citation-retrieval task"
    )
    p.add_argument("--data-dir", required=True, help="Directory containing PatentsView g_*.tsv.zip files")
    p.add_argument("--output", required=True)
    p.add_argument("--min-year", type=int, default=2005)
    p.add_argument("--max-year", type=int, default=2024)
    p.add_argument("--target-concepts", type=int, default=800)
    p.add_argument("--min-concept-patents", type=int, default=50)
    p.add_argument("--min-abstract-chars", type=int, default=80)
    p.add_argument("--chunksize", type=int, default=250000)
    args = p.parse_args(argv)

    summary = prepare_patents_h04l(
        args.data_dir,
        args.output,
        min_year=args.min_year,
        max_year=args.max_year,
        target_concepts=args.target_concepts,
        min_concept_patents=args.min_concept_patents,
        min_abstract_chars=args.min_abstract_chars,
        chunksize=args.chunksize,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
