#!/usr/bin/env python
"""Run five bounded attention-design probes without changing production caches."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np

from graph_mvp import __version__
from graph_mvp.attention_design_diagnostics import AttentionDesignDiagnostics, annotation_metrics, split_evidence
from graph_mvp.patient_repr import (PatientConceptMatrixBuilder, Qwen3EmbeddingEncoder,
                                    last_token_pool, load_concept_prototypes)
from graph_mvp.diagnostic_sampling import SyntheticEncoder, file_hash, sample_training_rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_annotations(path):
    result = {}
    if path is None:
        return result
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not {"document_id", "concept_id", "relation"} <= set(reader.fieldnames or ()):
            raise ValueError("Annotations require document_id,concept_id,relation")
        for row in reader:
            relation = row["relation"].strip()
            if relation not in ("", "unknown", "relevant", "hard_negative", "unrelated"):
                raise ValueError("Relation must be blank/unknown/relevant/hard_negative/unrelated")
            key = (row["document_id"], row["concept_id"])
            if key in result:
                raise ValueError("Duplicate annotation pair")
            result[key] = {"relation": relation, "notes": row.get("notes", "")}
    return result


def evaluate_saved(scores_path, annotations):
    """Human labels can be added later without loading or rerunning Qwen."""
    records, found = [], set()
    with Path(scores_path).open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            key = (row["document_id"], row["concept_id"])
            if key in annotations:
                if key in found:
                    raise ValueError("Duplicate saved document/concept score")
                found.add(key)
                records.append({**row, **annotations[key]})
    if found != set(annotations):
        raise ValueError("Annotations contain document/concept pairs absent from saved scores")
    return annotation_metrics(records)


def replay_sample(path, manifest, expected_hash):
    previous = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if previous.get("provenance", {}).get("train_sha256") != expected_hash:
        raise ValueError("Sample manifest training hash does not match --train")
    selected = previous["sample_rows"]
    wanted = {int(r["row_index"]): str(r.get("id", r.get("document_id"))) for r in selected}
    if len(wanted) != len(selected) or not 2 <= len(wanted) <= 512 or min(wanted) < 0:
        raise ValueError("Sample manifest must have 2..512 unique nonnegative row indices")
    op = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with op(path, "rt", encoding="utf-8-sig", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i in wanted:
                doc_id = str(row.get("patent_id", row.get("subject_id", str(i))))
                if doc_id != wanted[i] or not row["text"].strip():
                    raise ValueError("Sample manifest ID/text does not match source row")
                rows.append({"row_index": i, "id": doc_id, "text": row["text"]})
    if len(rows) != len(wanted) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Sample manifest rows missing or document IDs ambiguous")
    return rows


def tokenize_evidence(encoder, texts, masks):
    """Reproduce the encoder tokenization, assert alignment, retain source spans."""
    batch = encoder.tokenizer(texts, padding=True, truncation=True, max_length=encoder.max_length,
                              return_offsets_mapping=True, return_tensors="pt")
    import torch
    if not torch.equal(batch["attention_mask"].bool(), masks.detach().cpu().bool()):
        raise ValueError("Diagnostic tokenization differs from encoder attention mask")
    units, visible = [], []
    for i, text in enumerate(texts):
        active = batch["attention_mask"][i].bool()
        ids = batch["input_ids"][i][active].tolist()
        offsets = batch["offset_mapping"][i][active].tolist()
        tokens = encoder.tokenizer.convert_ids_to_tokens(ids)
        end = max(stop for start, stop in offsets)
        if end < 1:
            raise ValueError("Fast tokenizer source offsets required for fair chunk comparison")
        begin = min(start for start, stop in offsets if stop > start)
        if text[:begin].strip():
            raise ValueError("Right truncation required: evidence would otherwise include dropped prefix")
        units.append([{"token": token, "start": start, "end": stop,
                       "text": text[start:stop], "special_token": start == stop}
                      for token, (start, stop) in zip(tokens, offsets)])
        visible.append(text[:end])
    return units, visible


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--prototypes", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sample-manifest", type=Path, help="Replay IDs/rows from the previous real A-F report")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--representation-dim", type=int, default=64)
    parser.add_argument("--projection-file", type=Path)
    parser.add_argument("--temperature", type=float, default=.1)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[.1, .05, .02])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--chunk-max-chars", type=int, default=500)
    parser.add_argument("--max-chunks", type=int, default=64)
    parser.add_argument("--token-spectrum-samples", type=int, default=128)
    parser.add_argument("--review-documents", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--evaluate-only", type=Path, help="Existing scores.jsonl; evaluate human labels without model")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output directory exists; choose a new directory")
    if not 2 <= args.max_samples <= 512 or not 1 <= args.batch_size <= 8 or not 0 <= args.review_documents <= 32:
        parser.error("Use 2..512 samples, 1..8 batch size, 0..32 review documents")
    if (args.pairs < 1 or not 1 <= args.top_k <= 10 or not 2 <= args.token_spectrum_samples <= 256
            or not 8 <= args.chunk_max_chars <= 2000 or not 1 <= args.max_chunks <= 256
            or any(not np.isfinite(t) or t <= 0 for t in [args.temperature, *args.temperatures])):
        parser.error("Invalid bounded diagnostic limits or temperatures")
    annotations = load_annotations(args.annotations)
    if args.evaluate_only:
        if args.annotations is None:
            parser.error("--evaluate-only requires --annotations")
        result = evaluate_saved(args.evaluate_only, annotations)
        result["provenance"] = {"scores_sha256": file_hash(args.evaluate_only),
                                 "annotations_sha256": file_hash(args.annotations), "graph_mvp_version": __version__}
        args.output_dir.mkdir(parents=True)
        write_json(args.output_dir / "annotation_metrics.json", result)
        print(f"Saved human-label metrics: {args.output_dir}")
        return
    provenance = {"synthetic": args.synthetic, "graph_mvp_version": __version__,
                  "chunk_splitter": "punctuation_newline_with_bounded_character_windows",
                  "chunk_max_chars": args.chunk_max_chars, "max_chunks": args.max_chunks,
                  "sampling": "seeded_training_reservoir", "max_length": args.max_length}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True)
    provenance["git_sha"] = revision.stdout.strip() if revision.returncode == 0 else None
    if args.synthetic:
        if any(x is not None for x in (args.train, args.prototypes, args.model, args.sample_manifest)):
            parser.error("Synthetic cannot be combined with real model/data/sample inputs")
        encoder = SyntheticEncoder(32, args.seed)
        rng = np.random.default_rng(args.seed)
        prototypes = rng.normal(size=(16, 32)).astype(np.float32)
        concept_ids = [f"c{i}" for i in range(16)]
        concept_texts = [f"synthetic concept {i}" for i in range(16)]
        rows = [{"id": str(i), "row_index": i, "text": "Synthetic routing example. Another security sentence."}
                for i in range(args.max_samples)]
        provenance["sampling"] = "synthetic_instrumentation_only"
    else:
        if any(x is None for x in (args.train, args.prototypes, args.model)):
            parser.error("Real diagnostics require --train, --prototypes, --model")
        prototypes, meta = load_concept_prototypes(args.prototypes)
        if str(meta.get("encoder_id")) != args.model:
            parser.error("Prototype encoder ID differs from --model")
        concept_ids, concept_texts = meta["concept_ids"], meta["concept_texts"]
        train_hash = file_hash(args.train)
        rows = (replay_sample(args.train, args.sample_manifest, train_hash) if args.sample_manifest
                else sample_training_rows(args.train, args.max_samples, args.seed))
        if args.sample_manifest and len(rows) != args.max_samples:
            parser.error("--max-samples must match sample-manifest document count")
        provenance.update({"train": str(args.train), "train_sha256": train_hash,
                           "prototypes_sha256": file_hash(args.prototypes), "encoder_id": args.model})
        if args.sample_manifest:
            prior = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
            if prior["provenance"]["max_length"] != args.max_length or prior["projection"]["temperature"] != args.temperature:
                parser.error("Replay requires the previous max-length and baseline temperature")
            expected = prior["provenance"].get("projection_file_sha256")
            if expected and (not args.projection_file or file_hash(args.projection_file) != expected):
                parser.error("Replay requires the previous projection file")
            expected_prototypes = prior["provenance"].get("prototypes_sha256")
            if expected_prototypes and file_hash(args.prototypes) != expected_prototypes:
                parser.error("Replay requires the previous prototype file")
            if prior.get("seed", args.seed) != args.seed:
                parser.error("Replay requires the previous diagnostic seed")
            provenance.update({"sampling": "exact_prior_training_rows", "sample_manifest_sha256": file_hash(args.sample_manifest)})
        if len({r["id"] for r in rows}) != len(rows):
            parser.error("Sample document IDs must be unique for annotation matching")
        encoder = Qwen3EmbeddingEncoder(args.model, device=args.device, dtype=args.dtype,
                                        local_files_only=args.local_files_only, max_length=args.max_length)
    valid = {(r["id"], cid) for r in rows for cid in concept_ids}
    if not set(annotations) <= valid:
        parser.error("Annotation document/concept IDs must belong to this training sample")
    if len({doc for doc, cid in annotations}) > 32 or len(annotations) > 256:
        parser.error("Human review is bounded to 32 documents and 256 pairs")
    review_documents = {doc for doc, cid in annotations}
    for row in rows:
        if len(review_documents) >= max(args.review_documents, len({doc for doc, cid in annotations})):
            break
        review_documents.add(row["id"])
    mean = matrix = None
    if args.projection_file:
        with np.load(args.projection_file, allow_pickle=False) as projection:
            mean, matrix = projection["mean"], projection["matrix"]
        provenance["projection_file_sha256"] = file_hash(args.projection_file)
    builder = PatientConceptMatrixBuilder(encoder, prototypes, temperature=args.temperature,
                                          representation_dim=args.representation_dim,
                                          projection_matrix=matrix, projection_mean=mean, projection_seed=args.seed)
    probe = AttentionDesignDiagnostics(builder, concept_ids, seed=args.seed, pairs=args.pairs,
                                       temperatures=args.temperatures, cache_dtype=args.cache_dtype,
                                       token_spectrum_samples=args.token_spectrum_samples, top_k=args.top_k)
    import torch
    # Full float32 matmul control rather than TF32; no production settings changed.
    args.output_dir.mkdir(parents=True)
    manifest_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.output_dir / "run_manifest.json", {"status": "running", "args": manifest_args, "provenance": provenance})
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    evidence, labeled = [], []
    try:
        with (args.output_dir / "scores.jsonl").open("x", encoding="utf-8") as scores_file:
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                texts = [r["text"] for r in batch]
                with torch.inference_mode():
                    z, mask = encoder.encode_token_states(texts)
                    if args.sample_manifest:
                        expected_dtypes = prior.get("encoder_native_dtypes", [])
                        if expected_dtypes and str(z.dtype) not in expected_dtypes:
                            raise ValueError("Replay encoder native dtype differs from prior diagnostic")
                    h, alpha, stages = builder.from_hidden_states(z, mask, return_stages=True)
                    if args.synthetic:
                        token_units = [[{"token": f"synthetic-token-{j}", "text": "synthetic", "start": None, "end": None}
                                        for j in range(int(m.sum()))] for m in mask]
                        visible = texts
                    else:
                        token_units, visible = tokenize_evidence(encoder, texts, mask)
                    for i, doc in enumerate(batch):
                        chunks, coverage = split_evidence(visible[i], max_chars=args.chunk_max_chars, max_units=args.max_chunks)
                        vectors = []
                        coverage["encoder_truncated_units"] = 0
                        for c in range(0, len(chunks), args.batch_size):
                            chunk_texts = [u["text"] for u in chunks[c:c + args.batch_size]]
                            chunk_z, chunk_mask = encoder.encode_token_states(chunk_texts)
                            if not args.synthetic:
                                _, encoded_texts = tokenize_evidence(encoder, chunk_texts, chunk_mask)
                                coverage["encoder_truncated_units"] += sum(a != b for a, b in zip(chunk_texts, encoded_texts))
                            vectors.append(last_token_pool(chunk_z, chunk_mask).float())
                        coverage["complete"] = coverage["complete"] and coverage["encoder_truncated_units"] == 0
                        review = [cid for did, cid in annotations if did == doc["id"]]
                        if not review and doc["id"] in review_documents:
                            review = ["__suggest__"]
                        scores, examples = probe.add_document(doc, z[i, mask[i].bool()], alpha[i, mask[i].bool()],
                            {k: v[i] for k, v in stages.items()}, token_units=token_units[i],
                            chunk_vectors=torch.cat(vectors), chunk_units=chunks, coverage=coverage, evidence_concepts=review)
                        for record in scores:
                            scores_file.write(json.dumps(record, allow_nan=False) + "\n")
                            key = (record["document_id"], record["concept_id"])
                            if key in annotations:
                                labeled.append({**record, **annotations[key]})
                        for record in examples:
                            c = concept_ids.index(record["concept_id"])
                            record["concept_text"] = concept_texts[c]
                            record.update(annotations.get((doc["id"], record["concept_id"]), {}))
                            record["visible_document_text"] = visible[i]
                        evidence.extend(examples)
                print(f"attention probes {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
        report = probe.report()
        report["provenance"] = provenance
        report["human_evidence"] = annotation_metrics(labeled)
        if args.annotations:
            report["human_evidence"]["annotations_sha256"] = file_hash(args.annotations)
        write_json(args.output_dir / "report.json", report)
        write_json(args.output_dir / "evidence_review.json", evidence)
        with (args.output_dir / "annotations_template.csv").open("x", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["document_id", "row_index", "concept_id", "relation", "notes"])
            writer.writeheader()
            for e in evidence:
                writer.writerow({k: e.get(k, "") for k in writer.fieldnames})
        write_json(args.output_dir / "run_manifest.json", {"status": "complete", "args": manifest_args, "provenance": provenance,
                    "output_hashes": {name: file_hash(args.output_dir / name) for name in
                                      ("report.json", "scores.jsonl", "evidence_review.json", "annotations_template.csv")}})
    except Exception as exc:
        write_json(args.output_dir / "run_manifest.json", {"status": "failed", "args": manifest_args,
                   "provenance": provenance, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    print(f"Saved all five probes: {args.output_dir}")


if __name__ == "__main__":
    main()
