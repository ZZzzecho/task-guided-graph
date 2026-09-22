#!/usr/bin/env python3
"""Download the official UCI QSAR archive and verify biodeg.csv.

This helper requires internet access. The core package itself never downloads data
implicitly, so experiments remain reproducible from an explicit local file.
"""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import argparse
from graph_mvp.qsar import load_qsar_uci_csv

URL = "https://archive.ics.uci.edu/static/public/254/qsar%2Bbiodegradation.zip"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("data/qsar_biodegradation"))
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(URL) as response:
        payload = response.read()
    with ZipFile(BytesIO(payload)) as archive:
        names = archive.namelist()
        match = next((n for n in names if n.lower().endswith("biodeg.csv")), None)
        if match is None:
            raise RuntimeError(f"UCI archive did not contain biodeg.csv; files={names}")
        target = args.output_dir / "biodeg.csv"
        target.write_bytes(archive.read(match))
    X, y = load_qsar_uci_csv(target)
    rb, nrb = int(y.sum()), int((1-y).sum())
    if X.shape != (1055, 41) or (rb, nrb) != (356, 699):
        raise RuntimeError(f"Downloaded file failed identity check: shape={X.shape}, RB={rb}, NRB={nrb}")
    print(f"verified {target}: shape={X.shape}, RB={rb}, NRB={nrb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
