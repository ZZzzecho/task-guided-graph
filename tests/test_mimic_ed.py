import numpy as np
import pytest

from graph_mvp.mimic_ed import build_mimic_ed_cohort, triage_text, cohort_summary


def _tables():
    pd = pytest.importorskip("pandas")
    rows_t, rows_s = [], []
    stay = 100
    for subject in range(40):
        for visit in range(2):
            stay += 1
            rows_t.append({
                "subject_id": subject, "stay_id": stay,
                "chiefcomplaint": "pain" if visit == 0 else None,
                "temperature": 36.5, "heartrate": 70 + subject % 10,
                "resprate": 18, "o2sat": 98, "sbp": 120, "dbp": 70,
                "pain": "3", "acuity": 2 + subject % 3,
            })
            disp = "ADMITTED" if (subject + visit) % 2 else "HOME"
            rows_s.append({"subject_id": subject, "stay_id": stay, "disposition": disp})
    # One excluded outcome proves the cohort filter is exact.
    stay += 1
    rows_t.append({"subject_id": 999, "stay_id": stay, "chiefcomplaint": "x",
                   "temperature": 36, "heartrate": 60, "resprate": 15, "o2sat": 99,
                   "sbp": 110, "dbp": 70, "pain": "0", "acuity": 3})
    rows_s.append({"subject_id": 999, "stay_id": stay, "disposition": "TRANSFER"})
    return pd.DataFrame(rows_t), pd.DataFrame(rows_s)


def test_mimic_home_admitted_subject_group_split_and_template():
    triage, stays = _tables()
    cohort, splits = build_mimic_ed_cohort(triage, stays, seed=5)
    assert set(cohort["disposition"]) == {"HOME", "ADMITTED"}
    assert 999 not in set(cohort["subject_id"])
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert set(splits[a]["subject_id"]).isdisjoint(set(splits[b]["subject_id"]))
    text = cohort.iloc[0]["text"]
    assert "Chief complaint:" in text and "Acuity:" in text
    assert "disposition" not in text.lower()
    no_acuity = triage_text(cohort.iloc[0], include_acuity=False)
    assert "Acuity:" not in no_acuity
    summary = cohort_summary(cohort, splits)
    assert summary["cohort"]["n_encounters"] == 80
    assert sum(x["n_encounters"] for x in summary["splits"].values()) == 80
