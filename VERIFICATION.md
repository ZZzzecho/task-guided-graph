# Verification — v0.5.0

Date: 2026-09-30

## Current change

The complete repository test suite passed: `71 passed` on Python 3.14 with
NumPy, SciPy, scikit-learn, PyTorch, and pytest. The targeted initialization
and projection tests passed as part of that suite. `python -m compileall -q
graph_mvp scripts` also passed after repairing a pre-existing syntax error in
the patient runner.

These checks use small synthetic matrices. No H04L cache was regenerated and
no GLM-4.7-Flash retrieval experiment was run for v0.5.0. The v0.4.0 smoke
results do not establish performance for the PCA-projected graph.

## Historical v0.3.0 verification

## Automated tests

```bash
python -m pytest -q
```

Result:

```text
60 passed
```

Coverage includes the complete v0.2 test suite and new Patient-MNGM tests for:

- Qwen3 official last-token pooling semantics;
- patient concept-attention shape and padding masks;
- attention normalization over patient tokens;
- concept prototype persistence;
- sharded Patient matrix cache + deterministic materialization;
- Patient-MNGM estimator smoke solve;
- downstream task-LM prototype space;
- GLM-4.7-Flash LoRA target detection.

## Syntax check

```bash
python -m compileall -q graph_mvp scripts
```

passed.

## Not claimed

No MIMIC-IV-ED data or GLM-4.7-Flash weights are included in this artifact, so the
verification does not claim real medical AUROC/AUPRC or a full 30B model run.
`SERVER_QUICKSTART.md` gives the explicit server smoke sequence.
