# Verification — v0.3.0

Date: 2026-09-22

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
