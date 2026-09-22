"""Create small compatible downstream + graph-data fixtures for all three modes."""
from pathlib import Path
import numpy as np
from graph_mvp.graph_data import GraphSampleSet, save_graph_samples
from graph_mvp.estimators import VECTOR_MODE, PATIENT_MATRIX_MODE, BOOTSTRAP_MATRIX_MODE

out = Path(__file__).resolve().parent / "generated"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260921)
p, r = 5, 3
ids = tuple(f"concept_{i}" for i in range(p))

# Downstream patient-level scalar activations and labels.
n = 240
x = rng.normal(size=(n, p))
y = (1.2 * x[:, 0] * x[:, 1] + .4 * x[:, 2] + rng.normal(scale=.5, size=n) > 0).astype(int)
np.savez_compressed(out / "downstream_task.npz",
                    concept_ids=np.asarray(ids),
                    X_train=x[:140], y_train=y[:140],
                    X_reward=x[140:190], y_reward=y[140:190],
                    X_test=x[190:], y_test=y[190:])

save_graph_samples(out / "patient_activation_vector.npz",
                   GraphSampleSet(ids, VECTOR_MODE, x[:140]))

row_cov = np.array([[1., .25, 0.], [.25, 1., .12], [0., .12, 1.]])
col_cov = np.eye(p)
for j in range(p - 1):
    col_cov[j, j + 1] = col_cov[j + 1, j] = .28
lr, lc = np.linalg.cholesky(row_cov), np.linalg.cholesky(col_cov)
patient_tensor = np.stack([lr @ rng.normal(size=(r, p)) @ lc.T for _ in range(140)])
bootstrap_tensor = np.stack([lr @ rng.normal(size=(r, p)) @ lc.T for _ in range(50)])

save_graph_samples(out / "patient_concept_matrix.npz",
                   GraphSampleSet(ids, PATIENT_MATRIX_MODE, patient_tensor))
save_graph_samples(out / "bootstrap_concept_embedding_matrix.npz",
                   GraphSampleSet(ids, BOOTSTRAP_MATRIX_MODE, bootstrap_tensor))

print(out)
