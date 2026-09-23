from pathlib import Path
import json

import pandas as pd

from graph_mvp.patent_retrieval import build_fixed_candidate_pools, load_candidate_pools


def test_fixed_candidate_pool_prefers_examiner_and_hard_negatives(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    corpus = pd.DataFrame({
        "patent_id": ["1","2","3","4","5","6"],
        "patent_date": ["2010-01-01","2011-01-01","2012-01-01","2013-01-01","2014-01-01","2015-01-01"],
        "text": ["x"]*6,
        "split": ["train","train","train","graph","graph","graph"],
    })
    corpus.to_csv(root / "corpus.csv.gz", index=False, compression="gzip")
    corpus[corpus["split"]=="graph"].assign(n_internal_citations=1).to_csv(
        root / "retrieval_graph.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame([
        ("1","A"),("1","B"),
        ("2","A"),("2","C"),
        ("3","B"),("3","C"),
        ("4","A"),("4","B"),
        ("5","A"),("5","C"),
        ("6","B"),("6","C"),
    ], columns=["patent_id","cpc_group"]).to_csv(
        root / "concept_assignments.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame([
        ("4","1","cited by examiner"),
        ("4","2","cited by applicant"),
        ("5","2","cited by examiner"),
        ("6","3","cited by examiner"),
    ], columns=["patent_id","citation_patent_id","citation_category"]).to_csv(
        root / "citations.csv.gz", index=False, compression="gzip"
    )

    out = root / "graph_candidates.jsonl"
    summary = build_fixed_candidate_pools(
        root, out, split="graph", negatives_per_query=2, seed=1
    )
    pools = load_candidate_pools(out)
    assert summary["n_queries"] == 3
    q4 = next(x for x in pools if x["query_id"] == "4")
    assert q4["positive_id"] == "1"
    assert q4["candidate_ids"][0] == "1"
    assert "2" not in q4["candidate_ids"][1:]  # applicant-positive excluded from negatives
