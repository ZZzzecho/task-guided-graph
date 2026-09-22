# Data directory

Raw third-party datasets are intentionally not bundled in the package.

For QSAR Biodegradation:

```bash
python download_qsar.py
python prepare_qsar.py data/qsar_biodegradation/biodeg.csv --output data/qsar_biodeg_split.npz
```

The official UCI dataset is CC BY 4.0. Keep dataset provenance/citation with experiment outputs.
