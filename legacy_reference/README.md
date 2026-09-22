# Legacy reference scripts

These files are preserved only as historical references for the research design.

- `NMF_PK3rd.py`: early sentence-order perturbation + LoRA/LM training prototype. It is useful for tracing how shuffled document versions and concept input embeddings were originally handled.
- `bigraph_phy.py`: early nonparanormal MNGM prototype that loads 50 saved concept-embedding matrices and alternates concept-axis and representation-axis graph estimation.

The current framework **does not execute these scripts as its graph solver**. The normalized implementation is in `graph_mvp/estimators.py` and `graph_mvp/weighted_glasso.py`.
