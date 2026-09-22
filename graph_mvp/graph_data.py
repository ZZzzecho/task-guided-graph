"""Explicit graph-estimation data contract, independent of downstream task data."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .estimators import GRAPH_MODES, VECTOR_MODE, MATRIX_MODES


@dataclass(frozen=True)
class GraphSampleSet:
    concept_ids: tuple[str, ...]
    mode: str
    samples: np.ndarray

    def __post_init__(self):
        ids = tuple(self.concept_ids)
        x = np.asarray(self.samples, dtype=float)
        if self.mode not in GRAPH_MODES:
            raise ValueError(f"graph mode must be one of {GRAPH_MODES}")
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("concept_ids must be unique")
        expected_ndim = 2 if self.mode == VECTOR_MODE else 3
        if x.ndim != expected_ndim or x.shape[-1] != len(ids) or x.shape[0] < 2:
            raise ValueError("graph_samples shape does not match graph mode/concept axis")
        if self.mode in MATRIX_MODES and x.shape[1] < 2:
            raise ValueError("matrix-valued graph samples require representation_dim >= 2")
        if not np.isfinite(x).all():
            raise ValueError("graph_samples must be finite")
        object.__setattr__(self, "concept_ids", ids)
        object.__setattr__(self, "samples", x.copy())


def save_graph_samples(path, sample_set: GraphSampleSet):
    np.savez_compressed(path,
                        graph_mode=np.asarray(sample_set.mode),
                        concept_ids=np.asarray(sample_set.concept_ids),
                        graph_samples=np.asarray(sample_set.samples))


def load_graph_samples(path: str | Path):
    with np.load(path, allow_pickle=False) as raw:
        required = {"graph_mode", "concept_ids", "graph_samples"}
        if required - set(raw.files):
            raise ValueError(f"Missing graph-data NPZ keys: {required - set(raw.files)}")
        mode_raw = raw["graph_mode"]
        mode = str(mode_raw.item()) if mode_raw.ndim == 0 else str(mode_raw.reshape(-1)[0])
        ids = raw["concept_ids"]
        if ids.ndim != 1 or ids.dtype.kind != "U":
            raise ValueError("concept_ids must be a 1D Unicode string array")
        return GraphSampleSet(tuple(ids.tolist()), mode, raw["graph_samples"].copy())


def from_legacy_pqn(y, concept_ids, mode):
    """Convert legacy MNGM tensor [P concepts, R repr, N samples] -> [N,R,P]."""
    x = np.asarray(y, dtype=float)
    if x.ndim != 3 or x.shape[0] != len(concept_ids):
        raise ValueError("legacy tensor must have shape [P,R,N] matching concept_ids")
    return GraphSampleSet(tuple(concept_ids), mode, np.transpose(x, (2, 1, 0)))
