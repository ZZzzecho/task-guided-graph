"""UCI Diabetes 130-US Hospitals preprocessing for Patient-MNGM experiments."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

TARGET_VALUES = ("<30", ">30", "NO")
EXCLUDED_TEXT_COLUMNS = {
    "encounter_id",
    "patient_nbr",
    "readmitted",
    # This is a post-discharge administrative outcome and can create trivial leakage.
    "discharge_disposition_id",
}


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Diabetes130 preprocessing requires pandas; install `pip install -e '.[medical]'`"
        ) from exc
    return pd


def _format_value(value):
    pd = _pd()
    if pd.isna(value):
        return "MISSING"
    text = str(value).strip()
    if not text or text == "?":
        return "MISSING"
    return text


def _display_name(column: str) -> str:
    special = {
        "A1Cresult": "A1C result",
        "diag_1": "Diagnosis 1",
        "diag_2": "Diagnosis 2",
        "diag_3": "Diagnosis 3",
    }
    return special.get(column, column.replace("_", " ").capitalize())


def hospital_record_text(row, feature_columns):
    """Serialize one encounter deterministically without IDs or the target."""
    return "\n".join(
        f"{_display_name(column)}: {_format_value(row[column])}"
        for column in feature_columns
    )


def _group_split(frame, seed, proportions):
    from sklearn.model_selection import GroupShuffleSplit

    names = ("train", "graph", "val", "test")
    props = np.asarray(proportions, dtype=float)
    if props.shape != (4,) or np.any(props <= 0) or not np.isclose(props.sum(), 1.0):
        raise ValueError("split proportions must be four positive values summing to one")

    idx = np.arange(len(frame))
    groups = frame["patient_nbr"].to_numpy()

    first = GroupShuffleSplit(n_splits=1, train_size=float(props[0]), random_state=seed)
    train_idx, temp_idx = next(first.split(idx, frame["label"], groups))

    temp = frame.iloc[temp_idx]
    temp_idx_local = np.arange(len(temp))
    temp_groups = temp["patient_nbr"].to_numpy()
    graph_fraction = float(props[1] / props[1:].sum())
    second = GroupShuffleSplit(
        n_splits=1, train_size=graph_fraction, random_state=seed + 1
    )
    graph_local, rest_local = next(
        second.split(temp_idx_local, temp["label"], temp_groups)
    )

    rest = temp.iloc[rest_local]
    rest_idx_local = np.arange(len(rest))
    rest_groups = rest["patient_nbr"].to_numpy()
    val_fraction = float(props[2] / (props[2] + props[3]))
    third = GroupShuffleSplit(
        n_splits=1, train_size=val_fraction, random_state=seed + 2
    )
    val_local, test_local = next(
        third.split(rest_idx_local, rest["label"], rest_groups)
    )

    splits = {
        "train": frame.iloc[train_idx].copy(),
        "graph": temp.iloc[graph_local].copy(),
        "val": rest.iloc[val_local].copy(),
        "test": rest.iloc[test_local].copy(),
    }

    patient_sets = {
        name: set(df["patient_nbr"].tolist()) for name, df in splits.items()
    }
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if patient_sets[a] & patient_sets[b]:
                raise RuntimeError(f"patient leakage between {a} and {b}")
    return splits


def build_diabetes130_cohort(
    frame,
    *,
    seed=7,
    split_proportions=(0.70, 0.10, 0.10, 0.10),
):
    """Build a patient-disjoint binary cohort for 30-day readmission.

    label=1 iff the official UCI target `readmitted` is "<30".
    Both ">30" and "NO" are negatives. The target, encounter/patient IDs, and
    discharge_disposition_id are excluded from the serialized patient text.
    """
    required = {"encounter_id", "patient_nbr", "readmitted"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"diabetic_data.csv is missing columns: {sorted(missing)}")

    cohort = frame.copy()
    cohort["readmitted"] = cohort["readmitted"].astype(str).str.strip()
    invalid = sorted(set(cohort["readmitted"]) - set(TARGET_VALUES))
    if invalid:
        raise ValueError(f"unexpected readmitted values: {invalid}")

    cohort["label"] = (cohort["readmitted"] == "<30").astype(np.int64)
    feature_columns = [
        c
        for c in frame.columns
        if c not in EXCLUDED_TEXT_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("no usable feature columns remain after leakage exclusions")

    cohort["text"] = cohort.apply(
        lambda row: hospital_record_text(row, feature_columns), axis=1
    )
    cohort = cohort.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
    splits = _group_split(cohort, seed, split_proportions)
    return cohort, splits, tuple(feature_columns)


def cohort_summary(cohort, splits=None, feature_columns=None):
    def describe(df):
        return {
            "n_encounters": int(len(df)),
            "n_patients": int(df["patient_nbr"].nunique()),
            "positive_30d_readmission": int(df["label"].sum()),
            "positive_fraction": float(df["label"].mean()),
            "encounters_per_patient_mean": float(
                df.groupby("patient_nbr").size().mean()
            ),
            "encounters_per_patient_max": int(
                df.groupby("patient_nbr").size().max()
            ),
        }

    out = {
        "task": "30-day readmission (<30 vs >30/NO)",
        "cohort": describe(cohort),
        "text_excluded_columns": sorted(EXCLUDED_TEXT_COLUMNS),
    }
    if feature_columns is not None:
        out["text_feature_columns"] = list(feature_columns)
    if splits is not None:
        out["splits"] = {name: describe(df) for name, df in splits.items()}
    return out


def prepare_diabetes130(
    input_path,
    output_dir,
    *,
    seed=7,
    split_proportions=(0.70, 0.10, 0.10, 0.10),
):
    """Read UCI diabetic_data.csv and write the common split-file interface."""
    pd = _pd()
    frame = pd.read_csv(input_path)
    cohort, splits, feature_columns = build_diabetes130_cohort(
        frame, seed=seed, split_proportions=split_proportions
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    keep = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "label",
        "text",
        *feature_columns,
    ]
    # Keep unique columns because feature_columns may already contain label-like names
    # in a future source revision.
    keep = list(dict.fromkeys(keep))

    cohort.loc[:, keep].to_csv(
        output / "cohort.csv.gz", index=False, compression="gzip"
    )
    for name, split in splits.items():
        split.loc[:, keep].to_csv(
            output / f"{name}.csv.gz", index=False, compression="gzip"
        )

    summary = cohort_summary(cohort, splits, feature_columns)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
