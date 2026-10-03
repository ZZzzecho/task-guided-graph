#!/usr/bin/env python
"""Build bounded direct joint-evidence matrices and compare their collapse stats."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from graph_mvp import __version__
from graph_mvp.attention_design_diagnostics import TensorStageAccumulator
from graph_mvp.diagnostic_sampling import (SyntheticJointEncoder, file_hash,
    replay_training_rows, sample_training_rows)
from graph_mvp.estimators import _rank_gaussian_tensor
from graph_mvp.joint_evidence_repr import JointEvidenceMatrixBuilder
from graph_mvp.patient_repr import (PatientConceptMatrixBuilder, PatientMatrixCacheWriter,
    Qwen3EmbeddingEncoder, load_concept_prototypes)
from graph_mvp.representation_diagnostics import distribution


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path)
    p.add_argument("--prototypes", type=Path)
    p.add_argument("--model")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-manifest", type=Path)
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--representation-dim", type=int, default=64)
    p.add_argument("--projection-file", type=Path)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--chunk-max-tokens", type=int, default=128)
    p.add_argument("--evidence-top-k", type=int, default=3)
    p.add_argument("--encode-batch-size", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=16)
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--review-documents", type=int, default=32)
    p.add_argument("--review-concepts", type=int, default=6)
    p.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    p.add_argument("--cache-dtype", choices=("float16", "float32"), default="float32")
    p.add_argument("--device")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--compare-token-attention", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args(argv)
    if args.output_dir.exists():
        p.error("Output directory exists; choose a new directory")
    if (not 2 <= args.max_samples <= 512 or not 1 <= args.encode_batch_size <= 64
        or args.shard_size < 1 or args.pairs < 1 or not 0 <= args.review_documents <= 32
        or not 1 <= args.review_concepts <= 16 or not 1 <= args.evidence_top_k <= 16
        or not 8 <= args.chunk_max_tokens <= args.max_length):
        p.error("Invalid bounded sample, batch, evidence, review, or token limits")
    provenance = {"graph_mvp_version": __version__, "synthetic": args.synthetic,
                  "sampling": "seeded_training_reservoir", "seed": args.seed}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True)
    provenance["git_sha"] = revision.stdout.strip() if revision.returncode == 0 else None
    mean = matrix = None
    if args.projection_file:
        with np.load(args.projection_file, allow_pickle=False) as data:
            mean, matrix = data["mean"], data["matrix"]
        provenance["projection_file_sha256"] = file_hash(args.projection_file)
    if args.synthetic:
        if any(x is not None for x in (args.train, args.prototypes, args.model, args.sample_manifest)):
            p.error("Synthetic instrumentation cannot use real input/model/sample files")
        encoder = SyntheticJointEncoder(args.max_length, args.seed)
        concept_texts = [f"synthetic concept {i}" for i in range(16)]
        concept_ids = [f"c{i}" for i in range(16)]
        prototypes = encoder.encode_pooled(concept_texts).numpy()
        rows = [{"id": str(i), "row_index": i,
                 "text": f"Routing system {i}. Packet queues {i % 7}. Security channel {i % 5}."}
                for i in range(args.max_samples)]
        provenance["sampling"] = "synthetic_instrumentation_only"
    else:
        if any(x is None for x in (args.train, args.prototypes, args.model)):
            p.error("Real run requires --train, --prototypes, --model")
        prototypes, meta = load_concept_prototypes(args.prototypes)
        if str(meta.get("encoder_id")) != args.model:
            p.error("Prototype encoder ID differs from --model")
        concept_ids, concept_texts = meta["concept_ids"], meta["concept_texts"]
        train_hash = file_hash(args.train)
        rows = (replay_training_rows(args.train, args.sample_manifest, train_hash) if args.sample_manifest
                else sample_training_rows(args.train, args.max_samples, args.seed))
        if len(rows) != args.max_samples:
            p.error("--max-samples must match replayed sample count")
        provenance.update(train_sha256=train_hash, prototypes_sha256=file_hash(args.prototypes),
                          encoder_id=args.model, train=str(args.train))
        if args.sample_manifest:
            prior = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
            for key in ("prototypes_sha256", "projection_file_sha256"):
                expected = prior.get("provenance", {}).get(key)
                if expected and expected != provenance.get(key):
                    p.error(f"Replay requires the previous {key}")
            provenance.update(sampling="exact_prior_training_rows",
                              sample_manifest_sha256=file_hash(args.sample_manifest))
        encoder = Qwen3EmbeddingEncoder(args.model, device=args.device, dtype=args.dtype,
            local_files_only=args.local_files_only, max_length=args.max_length)
    if len({r["id"] for r in rows}) != len(rows):
        p.error("Sample document IDs must be unique")
    builder = JointEvidenceMatrixBuilder(encoder, prototypes, concept_texts,
        representation_dim=args.representation_dim, projection_matrix=matrix,
        projection_mean=mean, projection_seed=args.seed, top_k=args.evidence_top_k,
        chunk_max_tokens=args.chunk_max_tokens, encode_batch_size=args.encode_batch_size)
    baseline = (PatientConceptMatrixBuilder(encoder, prototypes, representation_dim=args.representation_dim,
        projection_matrix=matrix, projection_mean=mean, projection_seed=args.seed) if args.compare_token_attention else None)
    stage_names = ["joint_full", "projected", "cache_quantized", "rank_gaussian"]
    if baseline:
        stage_names += ["baseline_token_full", "baseline_token_cache", "baseline_token_rank_gaussian"]
    acc = {name: TensorStageAccumulator(len(concept_ids), pairs=args.pairs, seed=args.seed) for name in stage_names}
    args.output_dir.mkdir(parents=True)
    cache = args.output_dir / "cache"
    pipeline = builder.projection_metadata()
    pipeline.update(cache_build_complete=False, evidence_audit="../evidence.jsonl")
    writer = PatientMatrixCacheWriter(cache, concept_ids, builder.hidden_size, dtype=args.cache_dtype,
        encoder_id=provenance.get("encoder_id", "synthetic_joint_instrumentation"), temperature=None,
        encoder_hidden_size=builder.encoder_hidden_size,
        representation_reduction=builder.projection_kind + "_linear_joint",
        projection_sha256=pipeline["projection_sha256"], representation_pipeline=pipeline)
    writer._flush_metadata()
    np.savez_compressed(cache / "projection.npz", mean=builder.projection_mean.numpy(),
                        matrix=builder.projection_matrix.numpy())
    manifest_path = args.output_dir / "run_manifest.json"
    manifest = {"status": "running", "provenance": provenance,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    write_json(manifest_path, manifest)
    stored, baseline_stored, input_lengths, max_scores, selected_counts = [], [], [], [], []
    buffer, ids, reviews, annotation_rows = [], [], [], []
    try:
        with (args.output_dir / "evidence.jsonl").open("w", encoding="utf-8") as evidence_fh, \
             (args.output_dir / "scores.jsonl").open("w", encoding="utf-8") as score_fh:
            for doc_number, row in enumerate(rows):
                h, full, traces = builder.encode_document(row["text"])
                acc["joint_full"].add_tensor(full)
                acc["projected"].add_tensor(h)
                quantized = h.detach().float().cpu().numpy().astype(args.cache_dtype).astype(np.float32)
                acc["cache_quantized"].add(quantized)
                stored.append(quantized)
                buffer.append(quantized[0])
                ids.append(row["id"])
                evidence_fh.write(json.dumps({"document_id": row["id"], "row_index": row["row_index"],
                    "concept_evidence": traces}, ensure_ascii=False, allow_nan=False) + "\n")
                for i, trace in enumerate(traces):
                    record = {"document_id": row["id"], "concept_id": concept_ids[i],
                        **{k: trace[k] for k in ("input_tokens", "selected_evidence_tokens", "max_score",
                                                "mean_score", "projection_norm", "truncated")},
                        "selected_chunks": len(trace["chunks"]), "relation": ""}
                    score_fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    input_lengths.append(trace["input_tokens"])
                    max_scores.append(trace["max_score"])
                    selected_counts.append(len(trace["chunks"]))
                if doc_number < args.review_documents:
                    order = sorted(range(len(traces)), key=lambda i: (-traces[i]["max_score"], i))
                    half = (args.review_concepts + 1) // 2
                    selected = list(dict.fromkeys(order[:half] + order[-(args.review_concepts - half):]
                                                 if args.review_concepts > half else order[:half]))
                    for i in selected:
                        review = {"document_id": row["id"], "concept_id": concept_ids[i],
                            "document_text": row["text"], "relation": "", **traces[i]}
                        reviews.append(review)
                        annotation_rows.append({"document_id": row["id"], "concept_id": concept_ids[i],
                                                "concept_text": concept_texts[i], "relation": ""})
                if baseline:
                    z, mask = encoder.encode_token_states([row["text"]])
                    bh, _, stages = baseline.from_hidden_states(z, mask, return_stages=True)
                    acc["baseline_token_full"].add_tensor(stages["h_full"])
                    bq = bh.detach().float().cpu().numpy().astype(args.cache_dtype).astype(np.float32)
                    acc["baseline_token_cache"].add(bq)
                    baseline_stored.append(bq)
                if len(buffer) >= args.shard_size:
                    writer.write_shard(np.stack(buffer), ids, ids)
                    buffer, ids = [], []
                print(f"joint evidence {doc_number + 1}/{len(rows)}: chunks={traces[0]['candidate_chunks']} "
                      f"joint_inputs={builder.num_concepts}", flush=True)
        if buffer:
            writer.write_shard(np.stack(buffer), ids, ids)
        for name, matrices in [("rank_gaussian", stored), ("baseline_token_rank_gaussian", baseline_stored)]:
            if matrices:
                ranked = _rank_gaussian_tensor(np.concatenate(matrices))
                for start in range(0, len(ranked), 4):
                    acc[name].add(ranked[start:start + 4])
        report = {"n_documents": len(rows), "num_concepts": len(concept_ids), "seed": args.seed,
            "sample_rows": [{"id": r["id"], "row_index": r["row_index"]} for r in rows],
            "provenance": provenance, "pipeline": builder.projection_metadata(),
            "stages": {k: a.report() for k, a in acc.items()},
            "statistics": {"joint_input_tokens": distribution(input_lengths),
                           "maximum_chunk_cosine": distribution(max_scores),
                           "selected_chunk_count": distribution(selected_counts)},
            "comparison_scope": {"joint": "complete supplied text; selected token-budgeted excerpts",
                "baseline": "production token attention; encoder max_length truncation" if baseline else None,
                "baseline_pipeline": baseline.projection_metadata() if baseline else None},
            "human_evidence": {"status": "pending_human_labels", "n_review_pairs": len(reviews),
                               "selection_is_not_relevance_label": True},
            "limitations": ["Synthetic results test instrumentation, not semantics.",
                "Rank/conditioning gains do not prove semantic quality or MNGM separability.",
                "Existing static-prototype projection may not fit joint representations."]}
        write_json(args.output_dir / "report.json", report)
        write_json(args.output_dir / "evidence_review.json", reviews)
        with (args.output_dir / "annotations_template.csv").open("w", encoding="utf-8", newline="") as fh:
            fields = ["document_id", "concept_id", "concept_text", "relation"]
            csv_writer = csv.DictWriter(fh, fieldnames=fields)
            csv_writer.writeheader()
            csv_writer.writerows(annotation_rows)
        writer.representation_pipeline = builder.projection_metadata()
        writer.representation_pipeline.update(cache_build_complete=True, evidence_audit="../evidence.jsonl")
        writer.close()
        manifest.update(status="complete", outputs={str(x.relative_to(args.output_dir)): file_hash(x)
            for x in sorted(args.output_dir.rglob("*")) if x.is_file() and x != manifest_path})
        write_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(manifest_path, manifest)
        raise
    print(f"Saved joint-evidence report and graph-compatible cache: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
