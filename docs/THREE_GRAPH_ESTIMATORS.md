# Three graph-estimation backends in v0.1.2

The RL / reward / acceptance stack always acts on **concept edges**.  What changes is the statistical meaning of one graph-estimation sample.

## 1. Patient scalar activation GGM

Input:

```text
[N patients, P concepts]
```

One row is one patient's scalar activation vector over the fixed concept vocabulary.  The estimator applies the training-sample rank-Gaussian transform, forms a `P x P` covariance, and solves the existing edge-weighted graphical lasso.

Mode string:

```text
patient_activation_vector
```

This is the v0.1.1 estimator and remains the cheapest backend.

## 2. Patient-level MNGM

Input:

```text
[N patients, R representation coordinates, P concepts]
```

One patient contributes one `R x P` matrix.  Column `j` is that patient's **vector-valued, patient-conditioned representation of concept j**.  The graph estimator alternates between:

- a `P x P` concept precision matrix (this is the graph exposed to GRPO/downstream code), and
- an `R x R` representation-axis precision matrix (a nuisance/representation dependency model).

Mode string:

```text
patient_concept_matrix
```

## 3. Bootstrap MNGM

Input has exactly the same tensor shape:

```text
[B bootstrap replicates, R representation coordinates, P concepts]
```

One sample is now one bootstrap realization of the model-derived concept embedding matrix.

Mode string:

```text
bootstrap_concept_embedding_matrix
```

The MNGM solver is **identical** to patient-level MNGM.  Only the statistical semantics of axis 0 change.

---

# MNGM update

For a matrix sample `H_s` of shape `R x P`, v0.1.2 estimates:

- concept precision `Omega_C` (`P x P`),
- representation precision `Omega_R` (`R x R`).

At fixed `Omega_R`, the effective concept covariance is

```text
S_C = (1 / (N R)) sum_s H_s^T Omega_R H_s
```

and the concept update is the existing weighted GLASSO:

```text
Omega_C = argmin -logdet(Omega_C)
                    + tr(S_C Omega_C)
                    + sum_{i<j} Lambda_ij |Omega_C_ij|.
```

At fixed `Omega_C`:

```text
S_R = (1 / (N P)) sum_s H_s Omega_C H_s^T
```

and the representation precision is updated with a scalar representation-axis penalty.

GRPO only changes the **concept penalty matrix `Lambda`**.  It never directly edits either precision matrix.

---

# Relation to the original `bigraph_phy.py` prototype

The uploaded prototype used the same core idea:

1. arrange repeated concept-embedding matrices as `p x q x n`;
2. estimate pairwise Spearman dependence across the replicate axis;
3. apply the Gaussian-copula identity `2 sin(pi rho_s / 6)`;
4. alternate an `A` (concept) precision and `B` (representation) precision through trace contractions;
5. solve a graphical lasso on each axis.

v0.1.2 keeps this statistical structure but changes several implementation details deliberately:

- **Explicit tensor contract:** all matrix modes use `[N, R, P]`, so flattening is always representation-major and block indexing is unambiguous.
- **Two nonparanormal options:** `rank_gaussian` (default, scalable) and `spearman_sine` (legacy-compatible Gaussian-copula estimator).
- **Correct covariance interface:** effective `S_C` / `S_R` are passed directly into the framework's weighted graphical-lasso solver.  They are not passed to `sklearn.GraphicalLasso.fit()` as if covariance rows were raw samples.
- **No min-max covariance rescaling:** positive-definite ridge/eigenvalue stabilization is used instead.
- **Edge-specific concept penalties:** `Lambda_ij` is fully compatible with the GRPO actions already implemented in v0.1.1.
- **Scale handling:** the representation precision can be projected to mean diagonal/trace scale 1 (`scale_constraint="trace"`) to control Matrix-Normal scale non-identifiability.
- **No mandatory CUDA/Numba dependency:** the default implementation is NumPy/SciPy and can later receive a GPU acceleration backend without changing the estimator API.


## Legacy tensor conversion

The original script constructed `y` with shape `[P concepts, R representation dims, N bootstrap samples]`. Convert it without guessing axis order:

```python
from graph_mvp.graph_data import from_legacy_pqn, save_graph_samples
from graph_mvp.estimators import BOOTSTRAP_MATRIX_MODE

sample_set = from_legacy_pqn(y, keywords, BOOTSTRAP_MATRIX_MODE)
save_graph_samples("bootstrap_mngm_graph.npz", sample_set)
```

Internally v0.1.2 always stores matrix samples as `[N,R,P]`.

---

# Graph-data NPZ contract

Graph estimation is now separated from downstream task data.

A graph-data file contains:

```text
graph_mode     scalar Unicode string
concept_ids    [P] Unicode strings
graph_samples  [N,P] or [N,R,P]
```

Example:

```python
from graph_mvp.graph_data import GraphSampleSet, save_graph_samples
from graph_mvp.estimators import PATIENT_MATRIX_MODE

sample_set = GraphSampleSet(
    concept_ids=("chest pain", "dyspnea", "hypoxemia"),
    mode=PATIENT_MATRIX_MODE,
    samples=patient_concept_tensor,   # [N,R,P]
)
save_graph_samples("patient_mngm_graph.npz", sample_set)
```

Then run the normal framework with an independent downstream-task NPZ:

```bash
python run_mvp.py \
  --data downstream_task.npz \
  --graph-data patient_mngm_graph.npz \
  --config configs/mvp.json \
  --policy grpo \
  --output outputs/patient_mngm_grpo
```

The `concept_ids` and their order must match exactly between task data and graph data.
