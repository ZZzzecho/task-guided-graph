from pathlib import Path
import argparse
from .qsar import qsar_dataset, save_dataset_npz


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert official UCI QSAR biodegradation CSV to pre-split MVP NPZ")
    p.add_argument("csv", type=Path, help="UCI biodeg.csv (41 descriptors + RB/NRB)")
    p.add_argument("--output", type=Path, default=Path("data/qsar_biodeg_split.npz"))
    p.add_argument("--seed", type=int, default=20260921)
    args = p.parse_args(argv)
    ds = qsar_dataset(args.csv, seed=args.seed)
    save_dataset_npz(args.output, ds)
    print(f"saved {args.output} train={len(ds.X_train)} reward={len(ds.X_reward)} test={len(ds.X_test)} p={len(ds.concept_ids)}")
    return 0
