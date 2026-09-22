"""QSAR Biodegradation adapter for the real-data MVP.

Expected source: UCI dataset 254 `biodeg.csv` (semicolon-delimited, no header,
41 descriptors plus RB/NRB class). The adapter performs a deterministic,
stratified 60/20/20 split by default and writes the project's explicit NPZ
contract. Test data remain isolated from search/reward learning.
"""
from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
from sklearn.model_selection import train_test_split

from .data import Dataset

QSAR_CONCEPT_IDS = (
    "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]", "NssssC", "nCb-", "C%",
    "nCp", "nO", "F03[C-N]", "SdssC", "HyWi_B(m)", "LOC", "SM6_L", "F03[C-O]",
    "Me", "Mi", "nN-N", "nArNO2", "nCRX3", "SpPosA_B(p)", "nCIR", "B01[C-Br]",
    "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d", "B04[C-Br]", "SdO", "TI2_L",
    "nCrt", "C-026", "F02[C-N]", "nHDon", "SpMax_B(m)", "Psi_i_A", "nN",
    "SM6_B(m)", "nArCOOR", "nX",
)


def load_qsar_uci_csv(path: str | Path):
    """Load official UCI semicolon-delimited QSAR data and validate its identity."""
    path = Path(path)
    rows, labels = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        for line_no, row in enumerate(reader, 1):
            if not row or all(not c.strip() for c in row):
                continue
            if len(row) != 42:
                raise ValueError(f"QSAR line {line_no} has {len(row)} fields; expected 42")
            try:
                x = [float(v) for v in row[:41]]
            except ValueError as exc:
                raise ValueError(f"QSAR line {line_no} contains a nonnumeric descriptor") from exc
            label = row[41].strip().upper()
            if label not in {"RB", "NRB"}:
                raise ValueError(f"QSAR line {line_no} label must be RB or NRB, got {row[41]!r}")
            rows.append(x)
            labels.append(1 if label == "RB" else 0)
    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 41 or len(X) < 4 or not np.isfinite(X).all():
        raise ValueError("Invalid QSAR feature matrix")
    if len(np.unique(y)) != 2:
        raise ValueError("QSAR file must contain both RB and NRB classes")
    return X, y


def stratified_dataset(X, y, concept_ids=QSAR_CONCEPT_IDS, seed=20260921,
                       train_fraction=0.6, reward_fraction=0.2):
    """Create deterministic train/reward/test splits without test leakage."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or y.shape != (len(X),) or X.shape[1] != len(concept_ids):
        raise ValueError("Feature/label/concept dimensions do not align")
    if not (0 < train_fraction < 1 and 0 < reward_fraction < 1 and
            train_fraction + reward_fraction < 1):
        raise ValueError("train/reward fractions must be positive and sum to < 1")
    holdout_fraction = 1.0 - train_fraction
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=holdout_fraction, random_state=seed, stratify=y
    )
    reward_share = reward_fraction / holdout_fraction
    X_reward, X_test, y_reward, y_test = train_test_split(
        X_hold, y_hold, train_size=reward_share, random_state=seed + 1, stratify=y_hold
    )
    return Dataset(tuple(concept_ids), X_train, y_train, X_reward, y_reward, X_test, y_test)


def qsar_dataset(path: str | Path, seed=20260921, train_fraction=0.6, reward_fraction=0.2):
    X, y = load_qsar_uci_csv(path)
    return stratified_dataset(X, y, QSAR_CONCEPT_IDS, seed, train_fraction, reward_fraction)


def save_dataset_npz(path: str | Path, dataset: Dataset):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "concept_ids": np.asarray(dataset.concept_ids),
        "X_train": dataset.X_train, "y_train": dataset.y_train,
        "X_reward": dataset.X_reward, "y_reward": dataset.y_reward,
    }
    if dataset.X_test is not None:
        payload.update(X_test=dataset.X_test, y_test=dataset.y_test)
    np.savez_compressed(path, **payload)
    return path
