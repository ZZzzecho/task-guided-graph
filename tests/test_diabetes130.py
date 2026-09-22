import pytest

from graph_mvp.diabetes130 import (
    build_diabetes130_cohort,
    hospital_record_text,
)


def _frame():
    pd = pytest.importorskip("pandas")
    rows = []
    encounter = 1000
    for patient in range(80):
        for visit in range(2):
            encounter += 1
            rows.append(
                {
                    "encounter_id": encounter,
                    "patient_nbr": patient,
                    "race": "Caucasian",
                    "gender": "Female" if patient % 2 else "Male",
                    "age": "[60-70)",
                    "admission_type_id": 1,
                    "discharge_disposition_id": 1,
                    "time_in_hospital": 2 + visit,
                    "diag_1": "250.00",
                    "num_medications": 8 + visit,
                    "insulin": "No" if visit == 0 else "Steady",
                    "readmitted": "<30" if (patient + visit) % 7 == 0 else "NO",
                }
            )
    return pd.DataFrame(rows)


def test_diabetes130_patient_split_and_no_target_leakage():
    frame = _frame()
    cohort, splits, features = build_diabetes130_cohort(frame, seed=11)

    assert len(cohort) == 160
    assert set(cohort["label"]).issubset({0, 1})
    assert "readmitted" not in features
    assert "patient_nbr" not in features
    assert "encounter_id" not in features
    assert "discharge_disposition_id" not in features

    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert set(splits[a]["patient_nbr"]).isdisjoint(
                set(splits[b]["patient_nbr"])
            )

    text = cohort.iloc[0]["text"]
    assert "Readmitted:" not in text
    assert "Patient nbr:" not in text
    assert "Encounter id:" not in text
    assert "Discharge disposition id:" not in text
    assert "Diagnosis 1:" in text
    assert "Insulin:" in text


def test_diabetes130_missing_value_serialization():
    pd = pytest.importorskip("pandas")
    row = pd.Series({"race": "?", "age": "[70-80)"})
    text = hospital_record_text(row, ("race", "age"))
    assert "Race: MISSING" in text
    assert "Age: [70-80)" in text
