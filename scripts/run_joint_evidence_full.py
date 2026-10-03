#!/usr/bin/env python
"""Build expanded evidence matrices and run the existing full H04L experiment."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from graph_mvp import __version__
from graph_mvp.diagnostic_sampling import file_hash
from graph_mvp.patient_repr import PatientMatrixDataset, load_concept_prototypes, load_concept_vocabulary

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="New experiment directory")
    p.add_argument("--data-dir", type=Path, default=Path("data/patents_h04l"))
    p.add_argument("--config", type=Path, default=Path("configs/patient_h04l_main_v1.json"))
    p.add_argument("--projection-file", type=Path,
                   default=Path("data/patents_h04l/cache_main_4096_r64_pca/projection.npz"))
    p.add_argument("--qwen-model", default="/laijizheng/models/Qwen3-Embedding-0.6B")
    p.add_argument("--task-model", default="/laijizheng/models/GLM-4.7-Flash")
    p.add_argument("--documents", type=int, default=2048)
    p.add_argument("--evidence-top-k", type=int, default=6)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--chunk-max-tokens", type=int, default=128)
    p.add_argument("--encode-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--reuse-cache", type=Path, help="Only a complete cache built by this entry point")
    p.add_argument("--skip-controls", action="store_true")
    p.add_argument("--plan-only", action="store_true", help="Validate input data and save commands; no inference")
    args = p.parse_args(argv)
    if args.documents < 2 or args.evidence_top_k < 1 or args.encode_batch_size < 1 or not 8 <= args.chunk_max_tokens <= args.max_length:
        p.error("Invalid document/evidence/batch/token budget")
    def absolute(path):
        return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    out, data = absolute(args.output), absolute(args.data_dir)
    config, projection = absolute(args.config), absolute(args.projection_file)
    if out.exists() and any(out.iterdir()):
        p.error("Output is nonempty; choose a new experiment directory")
    inputs = {"train": data / "train.csv.gz", "prototypes": data / "concept_prototypes.npz",
              "concepts": data / "concepts.json", "config": config, "projection": projection}
    for split in ("train", "graph", "val", "test"):
        inputs[split + "_pools"] = data / f"retrieval_{split}_candidates.jsonl"
    for name, path in inputs.items():
        if not path.is_file():
            p.error(f"Missing {name}: {path}")
    prototypes, prototype_meta = load_concept_prototypes(inputs["prototypes"])
    vocab = load_concept_vocabulary(inputs["concepts"])
    if str(prototype_meta["encoder_id"]) != args.qwen_model or tuple(prototype_meta["concept_ids"]) != vocab.concept_ids:
        p.error("Prototypes/model/vocabulary do not match")
    # This is the same deterministic training prefix used by the full old runner.
    with gzip.open(inputs["train"], "rt", encoding="utf-8-sig", newline="") as fh:
        rows = csv.DictReader(fh)
        if "text" not in (rows.fieldnames or []):
            p.error("Training CSV requires text")
        for i in range(args.documents):
            row = next(rows, None)
            if row is None or not row["text"].strip():
                p.error(f"Training prefix cannot supply {args.documents} nonempty documents (row {i})")
    signature = {"train_sha256": file_hash(inputs["train"]),
                 "prototypes_sha256": file_hash(inputs["prototypes"]),
                 "projection_sha256": file_hash(projection), "qwen_model": args.qwen_model,
                 "documents": args.documents, "top_k": args.evidence_top_k,
                 "max_length": args.max_length, "chunk_max_tokens": args.chunk_max_tokens,
                 "representation_dim": 64, "dtype": "bfloat16", "cache_dtype": "float32",
                 "encode_batch_size": args.encode_batch_size}
    cache = absolute(args.reuse_cache) if args.reuse_cache else out / "cache"
    if args.reuse_cache:
        ds = PatientMatrixDataset(cache)
        if json.loads((cache / "build_signature.json").read_text(encoding="utf-8")) != signature:
            p.error("Reuse cache build signature differs from requested data/evidence/model/projection")
        if (len(ds) != args.documents or ds.hidden_size != 64 or ds.concept_ids != vocab.concept_ids
            or ds.metadata["representation_pipeline"].get("cache_build_complete") is not True):
            p.error("Reuse cache has wrong size or is incomplete")
        for key, expected in (("top_k", args.evidence_top_k), ("max_input_tokens", args.max_length),
                              ("chunk_max_tokens", args.chunk_max_tokens), ("representation_mode", "joint_evidence")):
            if ds.metadata["representation_pipeline"].get(key) != expected:
                p.error(f"Reuse cache metadata differs: {key}")
        ds.load_projection()
        integrity = json.loads((cache / "build_integrity.json").read_text(encoding="utf-8"))
        if any(file_hash(cache / name) != digest for name, digest in integrity.items()):
            p.error("Reuse cache contents changed after build")
    prefix = [sys.executable, "-u", "-m"]
    build = [*prefix, "scripts.build_patient_matrices", "--train", str(inputs["train"]),
        "--prototypes", str(inputs["prototypes"]), "--output", str(cache),
        "--representation-mode", "joint_evidence", "--model", args.qwen_model,
        "--local-files-only", "--dtype", "bfloat16", "--cache-dtype", "float32",
        "--max-length", str(args.max_length), "--chunk-max-tokens", str(args.chunk_max_tokens),
        "--evidence-top-k", str(args.evidence_top_k), "--encode-batch-size", str(args.encode_batch_size),
        "--batch-size", "1", "--shard-size", "16", "--representation-dim", "64",
        "--projection-file", str(projection), "--max-patients", str(args.documents), "--compact-evidence-audit"]
    controls = [*prefix, "scripts.diagnose_joint_controls", "--cache", str(cache),
        "--prototypes", str(inputs["prototypes"]), "--model", args.qwen_model,
        "--output", str(out / "controls"), "--encode-batch-size", str(args.encode_batch_size), "--seed", str(args.seed)]
    train = [*prefix, "scripts.run_patent_graph_rl", "--cache", str(cache),
        "--concepts", str(inputs["concepts"]), "--data-dir", str(data), "--config", str(config),
        "--task-model", args.task_model, "--task-local-files-only", "--task-seed", str(args.seed),
        "--max-mngm-documents", str(args.documents), "--initial-lambda", "0.8",
        "--max-train-queries", "1024", "--max-reward-queries", "128",
        "--max-val-queries", "256", "--max-test-queries", "1024",
        "--warmup-steps", "50", "--adapt-steps", "100", "--phases", "8",
        "--policy-updates-per-phase", "3", "--num-candidates", "8",
        "--task-batch-size", "2", "--candidate-batch-size", "16", "--task-max-length", "384",
        "--output", str(out / "training")]
    for split in ("train", "graph", "val", "test"):
        train.extend(["--" + split + "-pools", str(inputs[split + "_pools"])])
    stages = ([] if args.reuse_cache else [("build", build)])
    if not args.skip_controls:
        stages.append(("controls", controls))
    stages.append(("train", train))
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "plan_only" if args.plan_only else "running", "version": __version__,
                "signature": signature, "cache": str(cache), "args": vars(args),
                "input_hashes": {k: file_hash(v) for k, v in inputs.items()},
                "commands": {name: command for name, command in stages}, "stages": {},
                "planned_joint_pairs": args.documents * len(prototypes)}
    # Paths in arguments are serialized explicitly, without shell evaluation.
    manifest["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in manifest["args"].items()}
    save(out / "experiment_manifest.json", manifest)
    if args.plan_only:
        for name, command in stages:
            print(f"[{name}] {shlex.join(command)}")
        return
    for name, command in stages:
        print(f"[{name}] starting; progress log: {out / (name + '.log')}", flush=True)
        began = time.monotonic()
        try:
            with (out / (name + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            if name == "build":
                ds = PatientMatrixDataset(cache)
                if len(ds) != args.documents:
                    raise ValueError("Completed cache size differs from experiment budget")
                save(cache / "build_signature.json", signature)
                save(cache / "build_integrity.json", {path.name: file_hash(path) for path in cache.iterdir()
                    if path.is_file() and path.name != "build_integrity.json"})
            if name == "train" and not (out / "training" / "summary.json").is_file():
                raise ValueError("Training process returned without summary.json")
            manifest["stages"][name] = {"status": "complete", "seconds": time.monotonic() - began}
        except Exception as exc:
            manifest["stages"][name] = {"status": "failed", "error": str(exc), "seconds": time.monotonic() - began}
            if name != "controls":
                manifest["status"] = "failed"
                save(out / "experiment_manifest.json", manifest)
                raise
            print("[controls] failed; full training continues. Inspect controls.log.", flush=True)
        save(out / "experiment_manifest.json", manifest)
        print(f"[{name}] {manifest['stages'][name]['status']}", flush=True)
    manifest["status"] = "complete" if manifest["stages"].get("controls", {}).get("status") != "failed" else "training_complete_controls_failed"
    save(out / "experiment_manifest.json", manifest)
    print(f"Full experiment finished: {out / 'training' / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
