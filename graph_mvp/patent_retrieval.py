"""Fixed citation-retrieval candidate pools for H04L patent experiments."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
import random

import numpy as np


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Patent retrieval preparation requires pandas") from exc
    return pd


def _load_assignments(path):
    pd = _pd()
    df = pd.read_csv(path, dtype=str)
    if not {"patent_id", "cpc_group"}.issubset(df.columns):
        raise ValueError("concept_assignments.csv.gz must contain patent_id,cpc_group")
    grouped = defaultdict(set)
    for pid, code in zip(df["patent_id"].astype(str), df["cpc_group"].astype(str)):
        grouped[pid].add(code)
    return grouped


def _load_citation_sets(path):
    pd = _pd()
    df = pd.read_csv(path, dtype=str)
    required = {"patent_id", "citation_patent_id"}
    if not required.issubset(df.columns):
        raise ValueError("citations.csv.gz missing patent IDs")
    all_pos = defaultdict(set)
    examiner_pos = defaultdict(list)
    if "citation_category" in df.columns:
        categories = df["citation_category"].fillna("").astype(str).str.lower()
    else:
        categories = None
    for idx, (q, p) in enumerate(zip(df["patent_id"].astype(str),
                                     df["citation_patent_id"].astype(str))):
        all_pos[q].add(p)
        is_examiner = categories is None or "examiner" in categories.iloc[idx]
        if is_examiner and p not in examiner_pos[q]:
            examiner_pos[q].append(p)
    # If category metadata is absent or a query has no examiner-tagged citation,
    # the caller may still choose to skip that query rather than relabel applicants.
    return examiner_pos, all_pos


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    union = len(a | b)
    return 0.0 if union == 0 else len(a & b) / union


def _build_inverted(assignments, corpus_ids):
    inv = defaultdict(list)
    for pid in corpus_ids:
        for code in assignments.get(pid, ()):
            inv[code].append(pid)
    return inv


def build_fixed_candidate_pools(
    prepared_dir,
    output_path,
    *,
    split="graph",
    negatives_per_query=63,
    max_queries=None,
    seed=17,
    min_shared_concepts=1,
):
    """Build deterministic examiner-citation retrieval pools.

    Hard negatives are earlier H04L patents sharing selected CPC concepts with the
    query. CPC assignments are used only to construct the fixed evaluation pool;
    they are never passed as model activations.
    """
    pd = _pd()
    root = Path(prepared_dir)
    qpath = root / f"retrieval_{split}.csv.gz"
    if not qpath.exists():
        raise FileNotFoundError(qpath)
    queries = pd.read_csv(qpath, dtype=str)
    corpus = pd.read_csv(root / "corpus.csv.gz", dtype=str)
    assignments = _load_assignments(root / "concept_assignments.csv.gz")
    positives, all_citations = _load_citation_sets(root / "citations.csv.gz")

    corpus = corpus[["patent_id", "patent_date"]].copy()
    date_by_id = dict(zip(corpus["patent_id"].astype(str), corpus["patent_date"].astype(str)))
    all_ids = list(corpus["patent_id"].astype(str))
    inverted = _build_inverted(assignments, all_ids)

    rng = random.Random(int(seed))
    records = []
    dropped_no_examiner = 0
    dropped_no_negatives = 0

    for row in queries.itertuples(index=False):
        qid = str(row.patent_id)
        qdate = date_by_id.get(qid, "")
        pos_all = [p for p in positives.get(qid, ()) if p in date_by_id and date_by_id[p] < qdate]
        if not pos_all:
            dropped_no_examiner += 1
            continue
        positive = sorted(pos_all)[0]
        qcodes = assignments.get(qid, set())

        candidates = set()
        for code in qcodes:
            candidates.update(inverted.get(code, ()))
        candidates.discard(qid)
        candidates.difference_update(all_citations.get(qid, set()))
        candidates = [
            pid for pid in candidates
            if date_by_id.get(pid, "") < qdate
            and len(qcodes & assignments.get(pid, set())) >= int(min_shared_concepts)
        ]

        # Rank by CPC overlap, then keep a deterministic randomized tie order.
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda pid: (
                -_jaccard(qcodes, assignments.get(pid, set())),
                date_by_id.get(pid, ""),
                pid,
            )
        )
        negatives = candidates[: int(negatives_per_query)]

        # Fallback: same-domain earlier patents if CPC-overlap pool is too small.
        if len(negatives) < int(negatives_per_query):
            used = set(negatives) | {qid, positive} | set(all_citations.get(qid, set()))
            fallback = [pid for pid in all_ids if pid not in used and date_by_id.get(pid, "") < qdate]
            rng.shuffle(fallback)
            negatives.extend(fallback[: int(negatives_per_query) - len(negatives)])

        if not negatives:
            dropped_no_negatives += 1
            continue

        pool = [positive] + negatives
        records.append({
            "query_id": qid,
            "positive_id": positive,
            "candidate_ids": pool,
            "positive_index": 0,
            "n_query_concepts": len(qcodes),
            "n_positive_concepts": len(assignments.get(positive, set())),
            "positive_cpc_jaccard": _jaccard(qcodes, assignments.get(positive, set())),
            "max_negative_cpc_jaccard": max(
                (_jaccard(qcodes, assignments.get(pid, set())) for pid in negatives),
                default=0.0,
            ),
        })
        if max_queries is not None and len(records) >= int(max_queries):
            break

    if not records:
        raise ValueError("No retrieval candidate pools were constructed")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "split": split,
        "n_queries": len(records),
        "negatives_per_query_requested": int(negatives_per_query),
        "candidate_pool_size_min": min(len(x["candidate_ids"]) for x in records),
        "candidate_pool_size_max": max(len(x["candidate_ids"]) for x in records),
        "mean_positive_cpc_jaccard": float(np.mean([x["positive_cpc_jaccard"] for x in records])),
        "mean_max_negative_cpc_jaccard": float(np.mean([x["max_negative_cpc_jaccard"] for x in records])),
        "dropped_no_examiner_positive": int(dropped_no_examiner),
        "dropped_no_negatives": int(dropped_no_negatives),
        "seed": int(seed),
        "examiner_citation_is_primary_positive": True,
        "cpc_assignments_used_only_for_fixed_hard_negative_sampling": True,
    }
    out.with_suffix(out.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def load_candidate_pools(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("candidate pool file is empty")
    return rows
