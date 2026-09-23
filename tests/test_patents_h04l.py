from pathlib import Path
import zipfile

import pandas as pd

from graph_mvp.patents_h04l import prepare_patents_h04l


def _zip_tsv(path: Path, frame: pd.DataFrame):
    tsv_name = path.name.replace(".zip", "")
    tmp = path.parent / tsv_name
    frame.to_csv(tmp, sep="\t", index=False)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp, arcname=tsv_name)
    tmp.unlink()


def test_prepare_patents_h04l_small_fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()

    patent_rows = []
    abstract_rows = []
    cpc_rows = []
    citation_rows = []
    for i in range(120):
        pid = str(1000000 + i)
        year = 2005 + i // 6
        patent_rows.append(
            {
                "patent_id": pid,
                "patent_type": "utility",
                "patent_date": f"{year}-01-01",
                "patent_title": f"Network invention {i}",
                "withdrawn": "0",
            }
        )
        abstract_rows.append(
            {
                "patent_id": pid,
                "patent_abstract": (
                    "A network system for routing packets, controlling congestion, "
                    "managing secure communication, and allocating resources. " * 2
                ),
            }
        )
        cpc_rows.append(
            {
                "patent_id": pid,
                "cpc_subclass": "H04L",
                "cpc_group": f"H04L{41 + (i % 4) * 2}/00",
                "cpc_type": "inventional",
            }
        )
        if i >= 5:
            citation_rows.append(
                {
                    "patent_id": pid,
                    "citation_patent_id": str(1000000 + i - 5),
                    "citation_date": f"{year-1}-01-01",
                    "citation_category": "cited by examiner",
                }
            )

    titles = pd.DataFrame(
        [
            {"cpc_group": "H04L41/00", "cpc_group_title": "Network management"},
            {"cpc_group": "H04L43/00", "cpc_group_title": "Monitoring or testing networks"},
            {"cpc_group": "H04L45/00", "cpc_group_title": "Routing or path finding"},
            {"cpc_group": "H04L47/00", "cpc_group_title": "Traffic control and resource allocation"},
            {"cpc_group": "G06F1/00", "cpc_group_title": "Unrelated computing concept"},
        ]
    )

    _zip_tsv(raw / "g_patent.tsv.zip", pd.DataFrame(patent_rows))
    _zip_tsv(raw / "g_patent_abstract.tsv.zip", pd.DataFrame(abstract_rows))
    _zip_tsv(raw / "g_cpc_current.tsv.zip", pd.DataFrame(cpc_rows))
    _zip_tsv(raw / "g_cpc_title.tsv.zip", titles)
    _zip_tsv(raw / "g_us_patent_citation.tsv.zip", pd.DataFrame(citation_rows))

    out = tmp_path / "prepared"
    summary = prepare_patents_h04l(
        raw,
        out,
        min_year=2005,
        max_year=2024,
        target_concepts=4,
        min_concept_patents=2,
        chunksize=25,
    )

    assert summary["n_corpus_patents"] == 120
    assert summary["n_selected_concepts"] == 4
    assert summary["n_internal_citations"] > 0
    assert summary["fraction_with_selected_concept"] == 1.0
    concepts = pd.read_json(out / "concepts.json")
    assert set(concepts["concept_id"]) == {"H04L41/00", "H04L43/00", "H04L45/00", "H04L47/00"}
    assert (out / "train.csv.gz").exists()
    assert (out / "retrieval_graph.csv.gz").exists()
    assert (out / "citations.csv.gz").exists()
    assert (out / "task.json").exists()
