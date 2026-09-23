"""PatentsView H04L domain preparation for task-guided graph learning."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import gzip
import json
import re

import numpy as np


H04L = "H04L"
DEFAULT_SPLITS = (0.70, 0.10, 0.10, 0.10)


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Patent preprocessing requires pandas") from exc
    return pd


def _resolve_table(data_dir, stem):
    data_dir = Path(data_dir)
    for name in (f"{stem}.tsv.zip", f"{stem}.tsv", f"{stem}.csv.gz", f"{stem}.csv"):
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {stem} under {data_dir}")


def _iter_table(path, chunksize):
    pd = _pd()
    path = Path(path)
    sep = "\t" if ".tsv" in path.name else ","
    return pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        chunksize=chunksize,
        compression="infer",
        keep_default_na=False,
        low_memory=False,
    )


def _canon_cpc(value):
    return re.sub(r"\s+", "", str(value or "").upper())


def _is_true(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _temporal_split(frame, proportions=DEFAULT_SPLITS):
    props = np.asarray(proportions, dtype=float)
    if props.shape != (4,) or np.any(props <= 0) or not np.isclose(props.sum(), 1.0):
        raise ValueError("split proportions must be four positive values summing to one")
    frame = frame.sort_values(["patent_date", "patent_id"]).reset_index(drop=True)
    n = len(frame)
    cuts = np.floor(np.cumsum(props[:-1]) * n).astype(int)
    labels = np.empty(n, dtype=object)
    a, b, c = cuts.tolist()
    labels[:a] = "train"
    labels[a:b] = "graph"
    labels[b:c] = "val"
    labels[c:] = "test"
    frame = frame.copy()
    frame["split"] = labels
    return frame


def _write_chunked_csv(path, chunks):
    first = True
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        for chunk in chunks:
            if chunk is None or len(chunk) == 0:
                continue
            chunk.to_csv(fh, index=False, header=first)
            first = False


def _load_h04l_titles(path, chunksize):
    titles = {}
    for chunk in _iter_table(path, chunksize):
        required = {"cpc_group", "cpc_group_title"}
        if not required.issubset(chunk.columns):
            raise ValueError(f"g_cpc_title missing columns: {sorted(required - set(chunk.columns))}")
        groups = chunk["cpc_group"].map(_canon_cpc)
        mask = groups.str.startswith(H04L)
        sub = chunk.loc[mask, ["cpc_group", "cpc_group_title"]].copy()
        sub["cpc_group"] = sub["cpc_group"].map(_canon_cpc)
        for row in sub.itertuples(index=False):
            title = str(row.cpc_group_title).strip()
            if row.cpc_group and title:
                titles[row.cpc_group] = title
    return titles


def _extract_h04l_assignments(path, temp_path, chunksize):
    patent_ids = set()
    first = True
    with open(temp_path, "w", encoding="utf-8", newline="") as fh:
        for chunk in _iter_table(path, chunksize):
            required = {"patent_id", "cpc_group"}
            if not required.issubset(chunk.columns):
                raise ValueError(f"g_cpc_current missing columns: {sorted(required - set(chunk.columns))}")
            groups = chunk["cpc_group"].map(_canon_cpc)
            if "cpc_subclass" in chunk.columns:
                subclasses = chunk["cpc_subclass"].map(_canon_cpc)
                mask = (subclasses == H04L) | groups.str.startswith(H04L)
            else:
                mask = groups.str.startswith(H04L)
            cols = ["patent_id", "cpc_group"]
            if "cpc_type" in chunk.columns:
                cols.append("cpc_type")
            sub = chunk.loc[mask, cols].copy()
            if len(sub) == 0:
                continue
            sub["patent_id"] = sub["patent_id"].astype(str).str.strip()
            sub["cpc_group"] = sub["cpc_group"].map(_canon_cpc)
            sub = sub[(sub["patent_id"] != "") & sub["cpc_group"].str.startswith(H04L)]
            patent_ids.update(sub["patent_id"].tolist())
            sub.to_csv(fh, sep="\t", index=False, header=first)
            first = False
    return patent_ids


def _load_patent_metadata(path, patent_ids, min_year, max_year, chunksize):
    pd = _pd()
    parts = []
    for chunk in _iter_table(path, chunksize):
        required = {"patent_id", "patent_date", "patent_title"}
        if not required.issubset(chunk.columns):
            raise ValueError(f"g_patent missing columns: {sorted(required - set(chunk.columns))}")
        mask = chunk["patent_id"].astype(str).isin(patent_ids)
        if "patent_type" in chunk.columns:
            mask &= chunk["patent_type"].astype(str).str.lower().eq("utility")
        if "withdrawn" in chunk.columns:
            mask &= ~chunk["withdrawn"].map(_is_true)
        sub = chunk.loc[mask, ["patent_id", "patent_date", "patent_title"]].copy()
        if len(sub):
            parts.append(sub)
    if not parts:
        raise ValueError("No H04L patent metadata matched the requested filters")
    frame = pd.concat(parts, ignore_index=True)
    frame["patent_date"] = pd.to_datetime(frame["patent_date"], errors="coerce")
    frame = frame.dropna(subset=["patent_date"])
    frame = frame[
        (frame["patent_date"].dt.year >= int(min_year))
        & (frame["patent_date"].dt.year <= int(max_year))
    ].copy()
    frame["patent_date"] = frame["patent_date"].dt.strftime("%Y-%m-%d")
    frame["patent_id"] = frame["patent_id"].astype(str)
    frame["patent_title"] = frame["patent_title"].astype(str).str.strip()
    return frame.drop_duplicates("patent_id")


def _load_abstracts(path, patent_ids, chunksize, min_chars):
    pd = _pd()
    parts = []
    for chunk in _iter_table(path, chunksize):
        required = {"patent_id", "patent_abstract"}
        if not required.issubset(chunk.columns):
            raise ValueError(f"g_patent_abstract missing columns: {sorted(required - set(chunk.columns))}")
        mask = chunk["patent_id"].astype(str).isin(patent_ids)
        sub = chunk.loc[mask, ["patent_id", "patent_abstract"]].copy()
        sub["patent_abstract"] = sub["patent_abstract"].astype(str).str.strip()
        sub = sub[sub["patent_abstract"].str.len() >= int(min_chars)]
        if len(sub):
            parts.append(sub)
    if not parts:
        raise ValueError("No H04L abstracts matched the requested filters")
    return pd.concat(parts, ignore_index=True).drop_duplicates("patent_id")


def _count_train_concepts(assignments_path, train_ids, titles, chunksize):
    counts = Counter()
    for chunk in _iter_table(assignments_path, chunksize):
        sub = chunk[chunk["patent_id"].astype(str).isin(train_ids)].copy()
        if len(sub) == 0:
            continue
        sub["cpc_group"] = sub["cpc_group"].map(_canon_cpc)
        sub = sub[sub["cpc_group"].isin(titles)]
        sub = sub.drop_duplicates(["patent_id", "cpc_group"])
        counts.update(sub["cpc_group"].tolist())
    return counts


def _select_concepts(counts, titles, target_concepts, min_concept_patents):
    ranked = sorted(
        ((code, int(freq)) for code, freq in counts.items() if code in titles),
        key=lambda x: (-x[1], x[0]),
    )
    frequent = [x for x in ranked if x[1] >= int(min_concept_patents)]
    selected = frequent[: int(target_concepts)]
    if len(selected) < int(target_concepts):
        seen = {code for code, _ in selected}
        selected.extend(
            (code, freq) for code, freq in ranked
            if code not in seen
        )
        selected = selected[: int(target_concepts)]
    return [
        {
            "concept_id": code,
            "concept_text": titles[code],
            "cpc_group": code,
            "train_patent_frequency": freq,
        }
        for code, freq in selected
    ]


def _write_selected_assignments(assignments_path, output_path, corpus_ids, selected_codes, chunksize):
    per_patent = Counter()
    total_rows = 0

    def chunks():
        nonlocal total_rows
        for chunk in _iter_table(assignments_path, chunksize):
            sub = chunk[
                chunk["patent_id"].astype(str).isin(corpus_ids)
                & chunk["cpc_group"].map(_canon_cpc).isin(selected_codes)
            ].copy()
            if len(sub) == 0:
                continue
            sub["patent_id"] = sub["patent_id"].astype(str)
            sub["cpc_group"] = sub["cpc_group"].map(_canon_cpc)
            keep = ["patent_id", "cpc_group"]
            if "cpc_type" in sub.columns:
                keep.append("cpc_type")
            sub = sub[keep].drop_duplicates(["patent_id", "cpc_group"])
            total_rows += len(sub)
            per_patent.update(sub["patent_id"].tolist())
            yield sub

    _write_chunked_csv(output_path, chunks())
    return per_patent, total_rows


def _write_internal_citations(path, output_path, corpus_ids, date_by_id, chunksize):
    categories = Counter()
    positive_counts = Counter()
    total = 0

    def chunks():
        nonlocal total
        for chunk in _iter_table(path, chunksize):
            required = {"patent_id", "citation_patent_id"}
            if not required.issubset(chunk.columns):
                raise ValueError(
                    f"g_us_patent_citation missing columns: {sorted(required - set(chunk.columns))}"
                )
            citing = chunk["patent_id"].astype(str)
            cited = chunk["citation_patent_id"].astype(str)
            mask = citing.isin(corpus_ids) & cited.isin(corpus_ids) & (citing != cited)
            cols = ["patent_id", "citation_patent_id"]
            for optional in ("citation_date", "citation_category"):
                if optional in chunk.columns:
                    cols.append(optional)
            sub = chunk.loc[mask, cols].copy()
            if len(sub) == 0:
                continue
            sub["patent_id"] = sub["patent_id"].astype(str)
            sub["citation_patent_id"] = sub["citation_patent_id"].astype(str)
            # Only retain temporally valid prior-art links.
            temporal = [
                date_by_id.get(a, "") > date_by_id.get(b, "")
                for a, b in zip(sub["patent_id"], sub["citation_patent_id"])
            ]
            sub = sub[np.asarray(temporal, dtype=bool)]
            if len(sub) == 0:
                continue
            sub = sub.drop_duplicates(["patent_id", "citation_patent_id"])
            total += len(sub)
            positive_counts.update(sub["patent_id"].tolist())
            if "citation_category" in sub.columns:
                categories.update(sub["citation_category"].astype(str).str.strip().replace("", "UNKNOWN"))
            yield sub

    _write_chunked_csv(output_path, chunks())
    return positive_counts, categories, total


def prepare_patents_h04l(
    data_dir,
    output_dir,
    *,
    min_year=2005,
    max_year=2024,
    target_concepts=800,
    min_concept_patents=50,
    min_abstract_chars=80,
    chunksize=250_000,
    split_proportions=DEFAULT_SPLITS,
):
    """Build an H04L text corpus, CPC concept vocabulary, and citation-retrieval task."""
    pd = _pd()
    data_dir = Path(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    work = output / "_work"
    work.mkdir(exist_ok=True)

    paths = {
        "patent": _resolve_table(data_dir, "g_patent"),
        "abstract": _resolve_table(data_dir, "g_patent_abstract"),
        "cpc": _resolve_table(data_dir, "g_cpc_current"),
        "cpc_title": _resolve_table(data_dir, "g_cpc_title"),
        "citation": _resolve_table(data_dir, "g_us_patent_citation"),
    }

    h04l_assignments = work / "h04l_cpc.tsv"
    h04l_ids = _extract_h04l_assignments(paths["cpc"], h04l_assignments, chunksize)
    titles = _load_h04l_titles(paths["cpc_title"], chunksize)

    meta = _load_patent_metadata(
        paths["patent"], h04l_ids, min_year, max_year, chunksize
    )
    abstracts = _load_abstracts(
        paths["abstract"], set(meta["patent_id"]), chunksize, min_abstract_chars
    )
    corpus = meta.merge(abstracts, on="patent_id", how="inner", validate="one_to_one")
    if len(corpus) < 100:
        raise ValueError("Too few H04L patents with usable abstracts")

    corpus["text"] = (
        "Title: " + corpus["patent_title"].fillna("").astype(str).str.strip()
        + "\n\nAbstract: " + corpus["patent_abstract"].fillna("").astype(str).str.strip()
    )
    corpus = _temporal_split(corpus, split_proportions)

    train_ids = set(corpus.loc[corpus["split"] == "train", "patent_id"].astype(str))
    concept_counts = _count_train_concepts(
        h04l_assignments, train_ids, titles, chunksize
    )
    concepts = _select_concepts(
        concept_counts, titles, target_concepts, min_concept_patents
    )
    if len(concepts) < 2:
        raise ValueError("Too few H04L CPC concepts after filtering")
    selected_codes = {item["concept_id"] for item in concepts}

    concepts_path = output / "concepts.json"
    concepts_path.write_text(
        json.dumps(concepts, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    keep = [
        "patent_id",
        "patent_date",
        "patent_title",
        "patent_abstract",
        "text",
        "split",
    ]
    corpus.loc[:, keep].to_csv(output / "corpus.csv.gz", index=False, compression="gzip")
    for split in ("train", "graph", "val", "test"):
        corpus.loc[corpus["split"] == split, keep].to_csv(
            output / f"{split}.csv.gz", index=False, compression="gzip"
        )

    corpus_ids = set(corpus["patent_id"].astype(str))
    assignment_counts, n_assignment_rows = _write_selected_assignments(
        h04l_assignments,
        output / "concept_assignments.csv.gz",
        corpus_ids,
        selected_codes,
        chunksize,
    )

    date_by_id = dict(zip(corpus["patent_id"].astype(str), corpus["patent_date"].astype(str)))
    positive_counts, citation_categories, n_citations = _write_internal_citations(
        paths["citation"],
        output / "citations.csv.gz",
        corpus_ids,
        date_by_id,
        chunksize,
    )

    for split in ("train", "graph", "val", "test"):
        q = corpus[corpus["split"] == split].copy()
        q["n_internal_citations"] = q["patent_id"].astype(str).map(positive_counts).fillna(0).astype(int)
        q = q[q["n_internal_citations"] > 0]
        q.loc[:, keep + ["n_internal_citations"]].to_csv(
            output / f"retrieval_{split}.csv.gz", index=False, compression="gzip"
        )

    nonzero_assignments = np.asarray(
        [assignment_counts.get(str(pid), 0) for pid in corpus["patent_id"]], dtype=float
    )

    split_summary = {}
    for split in ("train", "graph", "val", "test"):
        df = corpus[corpus["split"] == split]
        qn = int(sum(positive_counts.get(str(x), 0) > 0 for x in df["patent_id"]))
        split_summary[split] = {
            "n_patents": int(len(df)),
            "date_min": str(df["patent_date"].min()) if len(df) else None,
            "date_max": str(df["patent_date"].max()) if len(df) else None,
            "n_retrieval_queries": qn,
        }

    summary = {
        "dataset": "PatentsView H04L granted utility patents",
        "domain": "H04L transmission of digital information",
        "year_range": [int(min_year), int(max_year)],
        "n_h04l_patents_before_text_filter": int(len(h04l_ids)),
        "n_corpus_patents": int(len(corpus)),
        "n_selected_concepts": int(len(concepts)),
        "target_concepts": int(target_concepts),
        "min_concept_patents_requested": int(min_concept_patents),
        "selected_concept_frequency_min": int(min(x["train_patent_frequency"] for x in concepts)),
        "selected_concept_frequency_max": int(max(x["train_patent_frequency"] for x in concepts)),
        "selected_concept_assignments": int(n_assignment_rows),
        "concepts_per_patent_mean": float(nonzero_assignments.mean()) if len(nonzero_assignments) else 0.0,
        "concepts_per_patent_median": float(np.median(nonzero_assignments)) if len(nonzero_assignments) else 0.0,
        "fraction_with_selected_concept": float((nonzero_assignments > 0).mean()) if len(nonzero_assignments) else 0.0,
        "n_internal_citations": int(n_citations),
        "citation_categories": {str(k): int(v) for k, v in citation_categories.items()},
        "splits": split_summary,
        "top_concepts": concepts[:20],
        "notes": {
            "concept_assignments_not_model_input": True,
            "cpc_hierarchy_not_used_for_graph_training": True,
            "official_cpc_assignments_used_for_domain_filtering_and_profiling": True,
            "retrieval_target_is_prior_patent_id_not_cpc_label": True,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    task = {
        "task_type": "citation_retrieval",
        "query_text_field": "text",
        "query_id_field": "patent_id",
        "positive_edge_file": "citations.csv.gz",
        "concept_vocabulary": "concepts.json",
        "concept_assignments_for_analysis_only": "concept_assignments.csv.gz",
        "primary_metrics": ["MRR", "Recall@10", "Recall@50", "NDCG@10"],
        "graph_attribution_rule": (
            "Keep sample text and concept activation fixed; vary only graph edge structure."
        ),
    }
    (output / "task.json").write_text(
        json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    try:
        h04l_assignments.unlink()
        work.rmdir()
    except OSError:
        pass
    return summary
