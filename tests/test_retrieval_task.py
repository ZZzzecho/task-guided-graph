from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from graph_mvp.graph_tokens import SoftGraphTokenizer
from graph_mvp.downstream import EvaluationError
from graph_mvp.retrieval_task import (
    GraphConditionedRetriever,
    FrozenGraphRetrievalEvaluator,
    RetrievalTaskContext,
    adapt_retrieval_model,
)
from graph_mvp.retrieval_state import (
    RetrievalQueryStream, capture_retrieval_training_state, restore_retrieval_training_state,
)


class BatchTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, padding=True, truncation=True, max_length=32,
                 add_special_tokens=True, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            ids = [2 + (ord(c) % 20) for c in str(text)]
            if add_special_tokens:
                ids = [2] + ids
            rows.append(ids[:max_length])
        width = max(map(len, rows))
        padded = [x + [0] * (width - len(x)) for x in rows]
        mask = [[1] * len(x) + [0] * (width - len(x)) for x in rows]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class TinyBackbone(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.ff = nn.Linear(hidden, hidden)

    def forward(self, inputs_embeds, attention_mask, use_cache=False, return_dict=True):
        x = self.ff(torch.cumsum(inputs_embeds, dim=1))
        return SimpleNamespace(last_hidden_state=x)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab=32, hidden=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.model = TinyBackbone(hidden)

    def get_input_embeddings(self):
        return self.emb


def test_retrieval_evaluator_runs(state):
    torch.manual_seed(3)
    lm = TinyCausalLM()
    graph_tok = SoftGraphTokenizer(
        torch.randn(3, 8), output_dim=8, num_tokens=2, graph_hidden_dim=6
    )
    retriever = GraphConditionedRetriever(lm, graph_tok)
    tok = BatchTokenizer()
    text = {
        "q1": "routing congestion",
        "q2": "secure network",
        "p1": "routing prior art",
        "p2": "unrelated storage",
        "p3": "security prior art",
        "p4": "another protocol",
    }
    ctx = RetrievalTaskContext(
        state.snapshot.concept_ids,
        ("q1", "q2"),
        (text["q1"], text["q2"]),
        (("p1", "p2"), ("p3", "p4")),
        (0, 0),
        text,
        tok,
        batch_size=2,
        max_length=32,
        score_temperature=0.2,
    )
    ev = FrozenGraphRetrievalEvaluator(retriever, candidate_batch_size=2)
    prep = ev.prepare(ctx)
    assert prep["cached"] == 4
    metrics = ev.evaluate_snapshot(state.snapshot, ctx)
    assert metrics.task_loss == -metrics.task_metric
    assert metrics.extra_metrics["n_queries"] == 2
    assert metrics.extra_metrics["pool_size"] == 2
    rel = ev.task_relevance(state.snapshot, ctx)
    assert rel.shape == (3, 3)
    assert np.isfinite(rel).all()


def make_context(state, n=9, batch_size=2, *, candidates=("p1", "p2")):
    ids = tuple(f"q{i}" for i in range(n))
    texts = {q: f"query text {q}" for q in ids}
    texts.update({p: f"patent text {p}" for p in candidates})
    return RetrievalTaskContext(state.snapshot.concept_ids, ids, tuple(texts[q] for q in ids),
                                tuple(candidates for _ in ids), tuple(0 for _ in ids),
                                texts, BatchTokenizer(), batch_size=batch_size, max_length=32)


def make_retriever():
    return GraphConditionedRetriever(TinyCausalLM(), SoftGraphTokenizer(
        torch.randn(3, 8), output_dim=8, num_tokens=2, graph_hidden_dim=6))


def test_training_stream_covers_tail_and_replays_epochs(state):
    ctx = make_context(state, n=9, batch_size=2)
    stream = RetrievalQueryStream(ctx, seed=31)
    other = RetrievalQueryStream(ctx, seed=31)
    flow = []
    for _ in range(5):
        a, b = stream.batch(ctx), other.batch(ctx)
        assert a == b
        flow.extend(a["query_ids"])
        stream.advance()
        other.advance()
    assert len(flow) == len(set(flow)) == 9
    assert set(flow) == set(ctx.query_ids)
    assert flow != list(ctx.query_ids)
    assert stream.summary()["cursor"] == 9
    assert stream.summary()["total_query_presentations"] == 9
    saved = stream.state_dict()
    resumed = RetrievalQueryStream(ctx, seed=31)
    resumed.load_state_dict(saved)
    assert resumed.batch(ctx) == stream.batch(ctx)
    assert resumed.epoch == 1
    with pytest.raises(ValueError, match="mismatch"):
        stream.batch(make_context(state, n=8))


def test_adaptation_continues_across_warmup_and_phases(state):
    torch.manual_seed(17)
    model = make_retriever()
    ctx = make_context(state)
    ev = FrozenGraphRetrievalEvaluator(model)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    first = adapt_retrieval_model(model, ev, state.snapshot, ctx, opt, steps=2, shuffle_seed=31)
    second = adapt_retrieval_model(model, ev, state.snapshot, ctx, opt, steps=3, shuffle_seed=31)
    assert first["coverage"]["unique_queries_seen"] == 4
    assert second["coverage"]["unique_queries_seen"] == 9
    assert second["coverage"]["total_query_presentations"] == 9
    assert second["coverage"]["optimizer_steps"] == 5
    assert second["coverage_before"] == first["coverage"]
    assert second["phase_query_presentations"] == 5
    with pytest.raises(ValueError, match="seed"):
        adapt_retrieval_model(model, ev, state.snapshot, ctx, opt, steps=1, shuffle_seed=32)


def test_late_test_candidates_use_initial_snapshot_and_restore_query_model(state):
    from copy import deepcopy
    torch.manual_seed(17)
    model = make_retriever()
    reference = deepcopy(model)
    ev = FrozenGraphRetrievalEvaluator(model, candidate_batch_size=1)
    old = make_context(state, candidates=("p1", "p2"))
    late = make_context(state, candidates=("p1", "new"))
    ev.prepare(old)
    initial_cached = ev._candidate_cache["p1"].clone()
    fingerprint = ev.candidate_metadata()["fingerprint"]
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    adapt_retrieval_model(model, ev, state.snapshot, old, opt, steps=3)
    query_params = {n: p.clone() for n, p in model.named_parameters()}
    model.train()
    info = ev.prepare(late)
    ref_ev = FrozenGraphRetrievalEvaluator(reference, candidate_batch_size=1)
    ref_ev.prepare(late)
    torch.testing.assert_close(ev._candidate_cache["new"], ref_ev._candidate_cache["new"])
    torch.testing.assert_close(ev._candidate_cache["p1"], initial_cached)
    assert info["encoded"] == 1 and info["fingerprint"] == fingerprint
    assert model.training and model.causal_lm.training
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, query_params[n])
    batch = BatchTokenizer()([late.text_by_id["new"]])
    with torch.no_grad():
        current = model.encode_candidates(**batch)[0]
    assert not torch.allclose(current, ev._candidate_cache["new"])


def test_candidate_cache_rejects_changed_text_and_tokenization(state):
    from dataclasses import replace
    ev = FrozenGraphRetrievalEvaluator(make_retriever())
    ctx = make_context(state)
    ev.prepare(ctx)
    changed = dict(ctx.text_by_id)
    changed["p1"] = "different patent"
    with pytest.raises(ValueError, match="text changed"):
        ev.prepare(replace(ctx, text_by_id=changed))
    with pytest.raises(ValueError, match="max_length"):
        ev.prepare(replace(ctx, max_length=48))


def test_frozen_base_mutation_fails_closed(state):
    model = make_retriever()
    model.causal_lm.emb.weight.requires_grad_(False)
    ev = FrozenGraphRetrievalEvaluator(model)
    ctx = make_context(state)
    ev.prepare(ctx)
    with torch.no_grad():
        model.causal_lm.emb.weight.add_(1)
    with pytest.raises(ValueError, match="base weight changed"):
        ev.prepare(ctx)


def test_candidate_encoding_error_restores_trainable_state_and_modes(state, monkeypatch):
    model = make_retriever()
    ev = FrozenGraphRetrievalEvaluator(model)
    with torch.no_grad():
        model.causal_lm.model.ff.bias.add_(3)
    model.train()
    model.causal_lm.model.eval()  # Deliberately mixed submodule modes.
    params = {n: p.clone() for n, p in model.named_parameters()}
    modes = [m.training for m in model.modules()]
    def broken(**batch):
        raise RuntimeError("injected encoding failure")
    monkeypatch.setattr(model, "encode_candidates", broken)
    with pytest.raises(RuntimeError, match="injected"):
        ev.prepare(make_context(state))
    assert modes == [m.training for m in model.modules()]
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, params[n])


def test_same_initial_state_optimizer_budget_stream_and_rng_replay(state, tmp_path):
    class DropoutBackbone(TinyBackbone):
        def __init__(self, hidden):
            super().__init__(hidden)
            self.dropout = nn.Dropout(.3)
        def forward(self, **kwargs):
            out = super().forward(**kwargs)
            return SimpleNamespace(last_hidden_state=self.dropout(out.last_hidden_state))
    torch.manual_seed(17)
    lm = TinyCausalLM()
    lm.model = DropoutBackbone(8)
    model = GraphConditionedRetriever(lm, SoftGraphTokenizer(
        torch.randn(3, 8), output_dim=8, num_tokens=2, graph_hidden_dim=6))
    ctx = make_context(state)
    ev = FrozenGraphRetrievalEvaluator(model)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    stream = RetrievalQueryStream(ctx, seed=31)
    initial = capture_retrieval_training_state(model, opt, stream,
                                               candidate_encoder_fingerprint=ev.candidate_metadata()["fingerprint"])
    path = tmp_path / "initial.pt"
    torch.save(initial, path)
    initial = torch.load(path, weights_only=True)
    assert not initial["optimizer"]["state"]
    adapt_retrieval_model(model, ev, state.snapshot, ctx, opt, steps=2, training_stream=stream)
    first = adapt_retrieval_model(model, ev, state.snapshot, ctx, opt, steps=3, training_stream=stream)
    final = {n: p.clone() for n, p in model.named_parameters()}
    with pytest.raises(ValueError, match="Incompatible"):
        restore_retrieval_training_state(model, opt, stream, initial, candidate_encoder_fingerprint="wrong")
    restore_retrieval_training_state(model, opt, stream, initial,
                                    candidate_encoder_fingerprint=ev.candidate_metadata()["fingerprint"])
    assert not opt.state
    repeated = adapt_retrieval_model(model, ev, state.snapshot, ctx, opt, steps=5, training_stream=stream)
    assert repeated["final_loss"] == first["final_loss"]
    assert repeated["coverage"] == first["coverage"]
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, final[n], rtol=0, atol=0)


def test_candidate_cache_rejects_mixed_fingerprints(state):
    ev = FrozenGraphRetrievalEvaluator(make_retriever())
    ctx = make_context(state)
    ev.prepare(ctx)
    ev._candidate_cache_fingerprints["p1"] = "different_encoder"
    with pytest.raises(EvaluationError, match="fingerprint mismatch"):
        ev._candidate_tensor(ctx.candidate_ids, torch.device("cpu"), torch.float32)


def test_patent_runner_writes_coverage_snapshots_and_final_only_fixed_test_index(tmp_path, monkeypatch):
    """Real MNGM/GRPO/runner flow with a tiny task backbone and on-disk inputs."""
    import json
    import sys
    from copy import deepcopy
    pd = pytest.importorskip("pandas")
    from graph_mvp.patient_repr import PatientMatrixCacheWriter
    import scripts.run_patent_graph_rl as runner
    root = tmp_path
    cache = root / "cache"
    concepts = ("a", "b", "c")
    writer = PatientMatrixCacheWriter(cache, concepts, hidden_size=4, dtype="float32", encoder_id="tiny")
    matrices = np.random.default_rng(17).normal(size=(12, 4, 3)).astype(np.float32)
    writer.write_shard(matrices, list(range(12)), list(range(12)))
    writer.close()
    (root / "concepts.json").write_text(json.dumps(dict(zip(concepts, ("routing", "security", "network")))))
    texts = {f"q-{split}-{i}": f"query {split} {i}" for split, n in (("train", 7), ("graph", 2), ("val", 2), ("test", 2))
             for i in range(n)}
    texts.update({"p1": "routing patent", "p2": "storage patent", "new": "unseen test patent"})
    pd.DataFrame({"patent_id": list(texts), "text": list(texts.values())}).to_csv(root / "corpus.csv.gz", index=False)
    for split, n in (("train", 7), ("graph", 2), ("val", 2), ("test", 2)):
        rows = [{"query_id": f"q-{split}-{i}", "candidate_ids": ["p1", "new" if split=="test" else "p2"],
                 "positive_index": 0} for i in range(n)]
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "config.json").write_text(json.dumps({"runner": {"seed": 17, "rounds": 1, "initial_lambda": .1},
                                                 "mngm": {"max_iter": 2}}))
    instances = []
    class RecordingEvaluator(FrozenGraphRetrievalEvaluator):
        def __init__(self, model, **kwargs):
            self.reference = deepcopy(model)
            super().__init__(model, **kwargs)
            instances.append(self)
    def load_model(*args, **kwargs):
        lm = TinyCausalLM()
        lm.emb.weight.requires_grad_(False)
        lm.model.ff.weight.requires_grad_(False)
        return lm, BatchTokenizer()
    monkeypatch.setattr(runner, "load_local_causal_lm", load_model)
    monkeypatch.setattr(runner, "attach_task_lora", lambda model, **kwargs: model)
    monkeypatch.setattr(runner, "build_task_soft_graph_tokenizer", lambda *args, **kwargs: SoftGraphTokenizer(
        torch.randn(3, 8), output_dim=8, num_tokens=2, graph_hidden_dim=6))
    monkeypatch.setattr(runner, "FrozenGraphRetrievalEvaluator", RecordingEvaluator)
    finished_search = []
    original_run = runner.GraphPhaseRunner.run
    def record_finished(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        finished_search.append(True)
        return result
    monkeypatch.setattr(runner.GraphPhaseRunner, "run", record_finished)
    original_load_context = runner.load_retrieval_context
    def check_holdout(**kwargs):
        if Path(kwargs["pool_path"]).name == "test.jsonl":
            assert finished_search, "Test pools were loaded during search"
        return original_load_context(**kwargs)
    monkeypatch.setattr(runner, "load_retrieval_context", check_holdout)
    argv = ["run_patent_graph_rl", "--cache", str(cache), "--concepts", str(root / "concepts.json"),
            "--data-dir", str(root), "--config", str(root / "config.json"), "--output", str(root / "run"),
            "--warmup-steps", "2", "--adapt-steps", "1", "--task-batch-size", "2", "--task-max-length", "32",
            "--phases", "1", "--policy-updates-per-phase", "1", "--num-candidates", "2"]
    for split in ("train", "graph", "val", "test"):
        argv += [f"--{split}-pools", str(root / f"{split}.jsonl")]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    summary = json.loads((root / "run/summary.json").read_text())
    manifest = json.loads((root / "run/run_manifest.json").read_text())
    assert summary["candidate_encoder"]["fingerprint"] == manifest["candidate_encoder"]["fingerprint"]
    assert summary["candidate_encoder"]["cache_consistent"] is True
    assert summary["candidate_encoder"]["cached_candidates"] == 3
    assert summary["test_split_touched"] is True
    coverage = summary["task_training"]["coverage"]
    assert coverage["actual_train_pool_size"] == 7
    assert coverage["optimizer_steps"] == 2 + summary["accepted_phases"]
    assert coverage["unique_queries_seen"] == coverage["total_query_presentations"]
    initial = torch.load(root / "run/checkpoints/retrieval_initial.pt", weights_only=True)
    final = torch.load(root / "run/checkpoints/retrieval_final.pt", weights_only=True)
    assert initial["stream"]["optimizer_steps"] == 0
    assert final["stream"]["optimizer_steps"] == coverage["optimizer_steps"]
    test_ctx = original_load_context(prepared_dir=root, pool_path=root / "test.jsonl", concept_ids=concepts,
                                     tokenizer=BatchTokenizer(), batch_size=2, max_length=32)
    ref = FrozenGraphRetrievalEvaluator(instances[0].reference)
    ref.prepare(test_ctx)
    torch.testing.assert_close(instances[0]._candidate_cache["new"], ref._candidate_cache["new"])
