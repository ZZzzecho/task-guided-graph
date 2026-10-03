#!/usr/bin/env python
"""F sensitivity controls on training documents from the full cache, not labels."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from graph_mvp.diagnostic_sampling import file_hash
from graph_mvp.joint_evidence_repr import (JointEvidenceMatrixBuilder, _fitting_prefix,
    format_joint_input, token_count)
from graph_mvp.patient_repr import PatientMatrixDataset, Qwen3EmbeddingEncoder, load_concept_prototypes
from graph_mvp.representation_diagnostics import ConceptStageAccumulator, distribution


def recover_evidence(document, row):
    if "evidence_text" in row:
        text = row["evidence_text"]
    else:
        text = "\n\n".join(document["source_text"][c["start"]:c["end"]] for c in row["chunks"])
    from hashlib import sha256
    if row.get("evidence_sha256") and sha256(text.encode()).hexdigest() != row["evidence_sha256"]:
        raise ValueError("Reconstructed evidence hash mismatch")
    return text


def cosine(a, b):
    return np.sum(a * b, axis=0) / np.maximum(np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0), 1e-15)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--prototypes", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--documents", type=int, default=32)
    p.add_argument("--concepts", type=int, default=64)
    p.add_argument("--encode-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args(argv)
    if args.documents < 2 or args.concepts < 2 or args.encode_batch_size < 1:
        p.error("Need >=2 control documents/concepts and positive batch size")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Choose a new controls output directory")
    ds = PatientMatrixDataset(args.cache)
    pipeline = ds.metadata["representation_pipeline"]
    if pipeline.get("representation_mode") != "joint_evidence":
        p.error("Controls require a joint-evidence cache")
    prototypes, meta = load_concept_prototypes(args.prototypes)
    if tuple(meta["concept_ids"]) != ds.concept_ids or str(meta["encoder_id"]) != args.model or ds.metadata["encoder_id"] != args.model:
        p.error("Cache/prototypes/model axis or encoder mismatch")
    rng = np.random.default_rng(args.seed)
    selected = sorted(rng.choice(len(ds), min(args.documents, len(ds)), replace=False).tolist())
    concepts = sorted(rng.choice(ds.num_concepts, min(args.concepts, ds.num_concepts), replace=False).tolist())
    documents, diversity = [], []
    wanted = set(selected)
    with (args.cache / "evidence.jsonl").open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            doc = json.loads(line)
            if doc["row_index"] != i:
                raise ValueError("Evidence audit row ordering differs from cache")
            traces = doc["concept_evidence"]
            if [x["concept_index"] for x in traces] != list(range(ds.num_concepts)):
                raise ValueError("Evidence audit concept axis differs")
            hashes = Counter(row.get("evidence_sha256") or recover_evidence(doc, row) for row in traces)
            count = ds.num_concepts
            diversity.append({"row_index": i, "candidate_chunks": traces[0]["candidate_chunks"],
                "selected_chunks_mean": float(np.mean([len(x["chunks"]) for x in traces])),
                "unique_evidence_sets": len(hashes), "dominant_set_fraction": max(hashes.values()) / count,
                "same_evidence_pair_fraction": sum(v * (v - 1) for v in hashes.values()) / (count * (count - 1)),
                "truncated_fraction": float(np.mean([x["truncated"] for x in traces]))})
            if i in wanted:
                documents.append(doc)
    if len(diversity) != len(ds) or len(documents) != len(selected):
        raise ValueError("Evidence audit count differs from complete cache")
    if any("source_text" not in doc for doc in documents):
        p.error("Top-1 replay requires compact source_spans_v1 cache from full entry point")
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output
    manifest = {"status": "running", "cache_fingerprint": ds.fingerprint(),
                "metadata_sha256": file_hash(args.cache / "metadata.json"),
                "prototypes_sha256": file_hash(args.prototypes), "seed": args.seed,
                "sample_rows": selected, "concept_indices": concepts}
    def save_manifest():
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_manifest()
    try:
        mean, matrix = ds.load_projection()
        encoder = Qwen3EmbeddingEncoder(args.model, max_length=pipeline["max_input_tokens"],
            dtype="bfloat16", local_files_only=True)
        texts = [meta["concept_texts"][i] for i in concepts]
        builder = JointEvidenceMatrixBuilder(encoder, prototypes[concepts], texts, top_k=1,
            chunk_max_tokens=pipeline["chunk_max_tokens"], encode_batch_size=args.encode_batch_size,
            representation_dim=ds.hidden_size, projection_mean=mean, projection_matrix=matrix)
        baseline_full = builder._pooled([format_joint_input(c, "") for c in texts]).T[None]
        baseline = builder.project_full_states(baseline_full).detach().float().cpu().numpy()[0]
        evidence = [[recover_evidence(d, d["concept_evidence"][i]) for i in concepts] for d in documents]
        lengths = np.array([[token_count(encoder.tokenizer, e, special=False) for e in row] for row in evidence])
        accumulators = {name: ConceptStageAccumulator(len(concepts), pairs=512, seed=args.seed)
                        for name in ("matched", "top1", "other_document", "concept_only")}
        rows = []
        for j, doc in enumerate(documents):
            matched_full = builder._pooled([format_joint_input(c, e) for c, e in zip(texts, evidence[j])]).T[None]
            matched = builder.project_full_states(matched_full).detach().float().cpu().numpy()[0]
            top1, _, _ = builder.encode_document(doc["source_text"])
            top1 = top1.detach().float().cpu().numpy()[0]
            donor_inputs, donor_rows = [], []
            for k, c in enumerate(texts):
                # Independent donor selection per concept, approximately length matched.
                candidates = rng.permutation([i for i in range(len(documents)) if i != j])
                donor = int(min(candidates, key=lambda i: abs(int(lengths[i, k]) - int(lengths[j, k]))))
                candidate = evidence[donor][k]
                target = int(lengths[j, k])
                def fits(x):
                    return (token_count(encoder.tokenizer, x, special=False) <= target
                            and token_count(encoder.tokenizer, format_joint_input(c, x)) <= encoder.max_length)
                if not fits(candidate):
                    candidate = _fitting_prefix(candidate, fits)
                if not candidate:
                    raise ValueError("Cannot construct nonempty donor evidence")
                donor_inputs.append(format_joint_input(c, candidate))
                donor_rows.append({"concept_index": concepts[k], "donor_row": selected[donor],
                    "matched_evidence_tokens": target,
                    "other_evidence_tokens": token_count(encoder.tokenizer, candidate, special=False),
                    "relation_label": None})
            donor_full = builder._pooled(donor_inputs).T[None]
            donor_h = builder.project_full_states(donor_full).detach().float().cpu().numpy()[0]
            for name, h in (("matched", matched), ("top1", top1), ("other_document", donor_h), ("concept_only", baseline)):
                accumulators[name].add(h[None])
            rows.append({"row_index": selected[j], "subject_id": doc["subject_id"], "donors": donor_rows,
                "matched_vs_top1_cosine": cosine(matched, top1).tolist(),
                "matched_vs_other_cosine": cosine(matched, donor_h).tolist(),
                "matched_vs_concept_only_cosine": cosine(matched, baseline).tolist()})
            print(f"[F controls] {j + 1}/{len(documents)}", flush=True)
        report = {"definition": "Diagnostic only; no baseline subtraction enters production H=W F(c,E).",
                  "limits": "Other-document evidence is not a verified negative. Length matching is approximate. "
                            "Cosine changes establish sensitivity, not semantic correctness or graph benefit. "
                            "All control statistics use a training subset, separate from full training.",
                  "sample_rows": selected, "concept_indices": concepts,
                  "evidence_diversity_all_documents": diversity,
                  "diversity_summary": {key: distribution([d[key] for d in diversity]) for key in
                      ("candidate_chunks", "selected_chunks_mean", "unique_evidence_sets", "dominant_set_fraction",
                       "same_evidence_pair_fraction", "truncated_fraction")},
                  "stages": {name: a.report() for name, a in accumulators.items()}, "comparisons": rows}
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        manifest.update(status="complete", outputs={"report.json": file_hash(out / "report.json")})
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_manifest()


if __name__ == "__main__":
    main()
