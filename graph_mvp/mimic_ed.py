"""MIMIC-IV-ED HOME-vs-ADMITTED cohort preparation for the first real experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

TRIAGE_FIELDS = (
    "chiefcomplaint", "temperature", "heartrate", "resprate", "o2sat",
    "sbp", "dbp", "pain", "acuity",
)
TARGET_DISPOSITIONS = ("HOME", "ADMITTED")


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MIMIC preprocessing requires pandas; install `pip install -e '.[medical]'`") from exc
    return pd


def _missing(value):
    try:
        return bool(_pd().isna(value))
    except TypeError:
        return value is None


def _format_value(value):
    if _missing(value) or (isinstance(value, str) and not value.strip()):
        return "MISSING"
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return f"{float(value):g}"
        return "MISSING"
    return str(value).strip()


def triage_text(row, include_acuity=True):
    """Deterministic fixed-order triage text; no post-triage fields are used."""
    items = [
        ("Chief complaint", "chiefcomplaint"),
        ("Temperature", "temperature"),
        ("Heart rate", "heartrate"),
        ("Respiratory rate", "resprate"),
        ("Oxygen saturation", "o2sat"),
        ("Systolic blood pressure", "sbp"),
        ("Diastolic blood pressure", "dbp"),
        ("Pain", "pain"),
    ]
    if include_acuity:
        items.append(("Acuity", "acuity"))
    return "\n".join(f"{label}: {_format_value(row[field])}" for label, field in items)


def _group_split(frame, seed, proportions):
    from sklearn.model_selection import GroupShuffleSplit

    names = ("train", "graph", "val", "test")
    props = np.asarray(proportions, dtype=float)
    if props.shape != (4,) or np.any(props <= 0) or not np.isclose(props.sum(), 1.0):
        raise ValueError("split proportions must be four positive values summing to one")
    groups = frame["subject_id"].to_numpy()
    idx = np.arange(len(frame))

    first = GroupShuffleSplit(n_splits=1, train_size=float(props[0]), random_state=seed)
    train_idx, temp_idx = next(first.split(idx, frame["label"], groups))
    temp = frame.iloc[temp_idx]
    temp_groups = temp["subject_id"].to_numpy()
    temp_local = np.arange(len(temp))
    graph_fraction_of_temp = float(props[1] / props[1:].sum())
    second = GroupShuffleSplit(n_splits=1, train_size=graph_fraction_of_temp, random_state=seed + 1)
    graph_local, rest_local = next(second.split(temp_local, temp["label"], temp_groups))
    rest = temp.iloc[rest_local]
    rest_groups = rest["subject_id"].to_numpy()
    rest_idx = np.arange(len(rest))
    val_fraction = float(props[2] / (props[2] + props[3]))
    third = GroupShuffleSplit(n_splits=1, train_size=val_fraction, random_state=seed + 2)
    val_local, test_local = next(third.split(rest_idx, rest["label"], rest_groups))

    splits = {
        "train": frame.iloc[train_idx].copy(),
        "graph": temp.iloc[graph_local].copy(),
        "val": rest.iloc[val_local].copy(),
        "test": rest.iloc[test_local].copy(),
    }
    subject_sets = {name: set(df["subject_id"].tolist()) for name, df in splits.items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if subject_sets[a] & subject_sets[b]:
                raise RuntimeError(f"subject leakage between {a} and {b}")
    return splits


def build_mimic_ed_cohort(triage, edstays, *, seed=7,
                          split_proportions=(0.70, 0.10, 0.10, 0.10),
                          include_acuity=True):
    """Merge triage + edstays and construct the leakage-controlled binary cohort.

    Accepts pandas DataFrames so the function is unit-testable without PhysioNet.
    Only HOME and ADMITTED are retained.  ED diagnosis/discharge text is not an
    accepted input to this function.
    """
    pd = _pd()
    required_triage = {"subject_id", "stay_id", *TRIAGE_FIELDS}
    required_stays = {"subject_id", "stay_id", "disposition"}
    if not required_triage.issubset(triage.columns):
        raise ValueError(f"triage is missing columns: {sorted(required_triage - set(triage.columns))}")
    if not required_stays.issubset(edstays.columns):
        raise ValueError(f"edstays is missing columns: {sorted(required_stays - set(edstays.columns))}")
    left = triage.loc[:, ["subject_id", "stay_id", *TRIAGE_FIELDS]].copy()
    right = edstays.loc[:, ["subject_id", "stay_id", "disposition"]].copy()
    merged = left.merge(right, on=["subject_id", "stay_id"], how="inner", validate="one_to_one")
    merged["disposition"] = merged["disposition"].astype(str).str.upper().str.strip()
    merged = merged[merged["disposition"].isin(TARGET_DISPOSITIONS)].copy()
    if len(merged) < 8:
        raise ValueError("Too few HOME/ADMITTED encounters after filtering")
    merged["label"] = (merged["disposition"] == "ADMITTED").astype(np.int64)
    merged["text"] = merged.apply(lambda row: triage_text(row, include_acuity), axis=1)
    merged = merged.sort_values(["subject_id", "stay_id"]).reset_index(drop=True)
    splits = _group_split(merged, seed, split_proportions)
    return merged, splits


def cohort_summary(cohort, splits=None):
    pd = _pd()

    def describe(df):
        missing = {field: float(df[field].isna().mean()) for field in TRIAGE_FIELDS}
        # Empty chief complaints count as missing as well.
        if "chiefcomplaint" in df:
            missing["chiefcomplaint"] = float(
                (df["chiefcomplaint"].isna() | (df["chiefcomplaint"].astype(str).str.strip() == "")).mean())
        counts = df["disposition"].value_counts().to_dict()
        encounters = df.groupby("subject_id").size()
        return {
            "n_encounters": int(len(df)),
            "n_subjects": int(df["subject_id"].nunique()),
            "disposition_counts": {str(k): int(v) for k, v in counts.items()},
            "admitted_fraction": float(df["label"].mean()),
            "missing_fraction": missing,
            "encounters_per_subject_mean": float(encounters.mean()),
            "encounters_per_subject_max": int(encounters.max()),
        }

    out = {"cohort": describe(cohort)}
    if splits is not None:
        out["splits"] = {name: describe(df) for name, df in splits.items()}
    return out


def prepare_mimic_ed(triage_path, edstays_path, output_dir, *, seed=7,
                      split_proportions=(0.70, 0.10, 0.10, 0.10),
                      include_acuity=True):
    """Read official CSV/CSV.GZ tables and write compact split CSV.GZ files."""
    pd = _pd()
    triage = pd.read_csv(triage_path)
    edstays = pd.read_csv(edstays_path)
    cohort, splits = build_mimic_ed_cohort(
        triage, edstays, seed=seed, split_proportions=split_proportions,
        include_acuity=include_acuity)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    keep = ["subject_id", "stay_id", "disposition", "label", "text", *TRIAGE_FIELDS]
    cohort.loc[:, keep].to_csv(output / "cohort.csv.gz", index=False, compression="gzip")
    for name, frame in splits.items():
        frame.loc[:, keep].to_csv(output / f"{name}.csv.gz", index=False, compression="gzip")
    summary = cohort_summary(cohort, splits)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
