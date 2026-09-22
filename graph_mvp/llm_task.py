"""Generic graph-conditioned causal-LM task adapter for HOME vs ADMITTED.

This module deliberately avoids importing transformers/peft.  Pass any causal LM
that supports `get_input_embeddings()` and `forward(inputs_embeds=..., ...)`,
including a PEFT-wrapped Qwen model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .downstream import TaskMetrics, EvaluationResult, EvaluationError
from .graph_tokens import error_weighted_edge_relevance


LABEL_TEXT = {0: "HOME", 1: "ADMITTED"}
DEFAULT_INSTRUCTION = (
    "Predict the emergency-department disposition using only the triage information. "
    "Answer exactly HOME or ADMITTED.\n\n"
)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Graph-conditioned LLM task code requires PyTorch") from exc
    return torch


@dataclass(frozen=True)
class CausalTaskContext:
    concept_ids: tuple[str, ...]
    texts: tuple[str, ...]
    labels: tuple[int, ...]
    tokenizer: object
    batch_size: int = 8
    max_length: int = 512
    instruction: str = DEFAULT_INSTRUCTION

    def __post_init__(self):
        ids = tuple(self.concept_ids)
        texts = tuple(self.texts)
        labels = tuple(int(x) for x in self.labels)
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("concept_ids must be unique")
        if not texts or len(texts) != len(labels) or any(y not in (0, 1) for y in labels):
            raise ValueError("texts/labels must form a nonempty binary task")
        if self.batch_size < 1 or self.max_length < 8:
            raise ValueError("invalid batch_size/max_length")
        object.__setattr__(self, "concept_ids", ids)
        object.__setattr__(self, "texts", texts)
        object.__setattr__(self, "labels", labels)

    def batches(self, device):
        for start in range(0, len(self.texts), self.batch_size):
            yield encode_binary_batch(
                self.tokenizer,
                self.texts[start:start + self.batch_size],
                self.labels[start:start + self.batch_size],
                instruction=self.instruction,
                max_length=self.max_length,
                device=device,
            )


def _encode_ids(tokenizer, text, add_special_tokens):
    out = tokenizer(text, add_special_tokens=add_special_tokens)
    ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def encode_binary_batch(tokenizer, texts, labels, *, instruction=DEFAULT_INSTRUCTION,
                        max_length=512, device="cpu"):
    """Create causal-LM labels with prompt tokens masked by -100."""
    torch = _torch()
    pad = getattr(tokenizer, "pad_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if pad is None:
        pad = eos
    if pad is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    sequences, targets, activation_masks = [], [], []
    for text, label in zip(texts, labels):
        prefix_ids = _encode_ids(tokenizer, instruction, add_special_tokens=True)
        patient_ids = _encode_ids(tokenizer, str(text).rstrip(), add_special_tokens=False)
        suffix_ids = _encode_ids(tokenizer, "\n\nDisposition:", add_special_tokens=False)
        answer_ids = _encode_ids(tokenizer, " " + LABEL_TEXT[int(label)], add_special_tokens=False)
        if eos is not None:
            answer_ids = answer_ids + [eos]
        if not answer_ids:
            raise ValueError("label tokenized to zero tokens")
        fixed = len(prefix_ids) + len(suffix_ids) + len(answer_ids)
        available_patient = max_length - fixed
        if available_patient < 1:
            raise ValueError("max_length is too small for instruction/suffix/label tokens")
        patient_ids = patient_ids[:available_patient]
        seq = prefix_ids + patient_ids + suffix_ids + answer_ids
        prompt_len = len(prefix_ids) + len(patient_ids) + len(suffix_ids)
        target = [-100] * prompt_len + answer_ids
        # Activation is patient-specific: fixed instruction/suffix and gold answer are excluded.
        activation = ([0] * len(prefix_ids) + [1] * len(patient_ids) +
                      [0] * (len(suffix_ids) + len(answer_ids)))
        sequences.append(seq)
        targets.append(target)
        activation_masks.append(activation)
    length = max(len(x) for x in sequences)
    input_ids, labels_out, attention, activation = [], [], [], []
    for seq, target, mask in zip(sequences, targets, activation_masks):
        n = length - len(seq)
        input_ids.append(seq + [pad] * n)
        labels_out.append(target + [-100] * n)
        attention.append([1] * len(seq) + [0] * n)
        activation.append(mask + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels_out, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "activation_mask": torch.tensor(activation, dtype=torch.bool, device=device),
    }


def per_example_label_logprob(logits, labels):
    """Conditional log p(label tokens | prompt+graph), summed per example."""
    torch = _torch()
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits/labels shape mismatch")
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe = shift_labels.masked_fill(~mask, 0)
    lp = torch.log_softmax(shift_logits, dim=-1)
    chosen = torch.gather(lp, -1, safe.unsqueeze(-1)).squeeze(-1)
    return (chosen * mask).sum(dim=1)


def _build_wrapper_class():
    torch = _torch()
    import torch.nn as nn

    class GraphConditionedCausalLM(nn.Module):
        """Inject sample-conditioned soft graph tokens through `inputs_embeds`."""
        def __init__(self, causal_lm, graph_tokenizer):
            super().__init__()
            self.causal_lm = causal_lm
            self.graph_tokenizer = graph_tokenizer
            # For sharded/device_map causal LMs, custom inputs_embeds enter through
            # the input-embedding device. Keep the small graph tokenizer there.
            embedding = self.causal_lm.get_input_embeddings()
            if embedding is None:
                raise ValueError("causal_lm.get_input_embeddings() returned None")
            try:
                entry_device = embedding.weight.device
                if entry_device.type != "meta":
                    self.graph_tokenizer.to(entry_device)
            except AttributeError:
                pass

        @property
        def device(self):
            embedding = self.causal_lm.get_input_embeddings()
            try:
                return embedding.weight.device
            except (AttributeError, StopIteration):
                try:
                    return next(self.parameters()).device
                except StopIteration:
                    return torch.device("cpu")

        def forward(self, input_ids, attention_mask, labels, global_graph,
                    activation_mask=None, return_graph_aux=False):
            embedding = self.causal_lm.get_input_embeddings()
            token_embeddings = embedding(input_ids)
            graph_tokens, aux = self.graph_tokenizer(
                token_embeddings,
                global_graph,
                return_aux=True,
                token_mask=activation_mask,
            )
            b, k, _ = graph_tokens.shape
            inputs_embeds = torch.cat((graph_tokens, token_embeddings), dim=1)
            graph_attention = torch.ones((b, k), dtype=attention_mask.dtype,
                                         device=attention_mask.device)
            full_attention = torch.cat((graph_attention, attention_mask), dim=1)
            ignore = torch.full((b, k), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat((ignore, labels), dim=1)
            output = self.causal_lm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention,
                labels=full_labels,
            )
            if return_graph_aux:
                return output, full_labels, aux
            return output, full_labels

    return GraphConditionedCausalLM


try:  # pragma: no cover
    GraphConditionedCausalLM = _build_wrapper_class()
except RuntimeError:  # pragma: no cover
    class GraphConditionedCausalLM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("GraphConditionedCausalLM requires PyTorch")


class FrozenGraphCausalEvaluator:
    """Dense binary reward from normalized HOME/ADMITTED label likelihoods.

    For every patient we score both label strings under the same frozen model and
    graph, then normalize the two sequence log-likelihoods with log-softmax.  The
    task utility is mean log p(gold label | x,G), i.e. negative binary NLL.
    """
    def __init__(self, model):
        self.model = model

    def _score(self, snapshot, context: CausalTaskContext, return_activation=False):
        if snapshot.concept_ids != context.concept_ids:
            raise ValueError("Graph/data concept axis mismatch")
        torch = _torch()
        from sklearn.metrics import (roc_auc_score, average_precision_score,
                                     accuracy_score, f1_score)
        self.model.eval()
        gold_logps, admitted_probs, activations = [], [], []
        try:
            device = self.model.device
        except AttributeError:
            device = next(self.model.parameters()).device
        with torch.no_grad():
            for start in range(0, len(context.texts), context.batch_size):
                texts = context.texts[start:start + context.batch_size]
                gold = context.labels[start:start + context.batch_size]
                pair_texts = tuple(text for text in texts for _ in (0, 1))
                pair_labels = tuple(label for _ in texts for label in (0, 1))
                batch = encode_binary_batch(
                    context.tokenizer, pair_texts, pair_labels,
                    instruction=context.instruction, max_length=context.max_length,
                    device=device)
                output, labels, aux = self.model(
                    global_graph=snapshot.Rho,
                    return_graph_aux=True,
                    **batch,
                )
                sequence_lp = per_example_label_logprob(output.logits, labels)
                if not torch.isfinite(sequence_lp).all():
                    raise EvaluationError("Nonfinite label log-likelihood")
                scores = sequence_lp.reshape(len(texts), 2)
                normalized = torch.log_softmax(scores, dim=1)
                gold_index = torch.as_tensor(gold, dtype=torch.long, device=device)
                selected = normalized.gather(1, gold_index[:, None]).squeeze(1)
                gold_logps.extend(selected.detach().cpu().tolist())
                admitted_probs.extend(torch.exp(normalized[:, 1]).detach().cpu().tolist())
                if return_activation:
                    # The two candidate labels have identical patient activation masks.
                    pair_a = aux["activations"].reshape(len(texts), 2, -1)
                    activations.append(pair_a[:, 0].detach().cpu().numpy())
        arr = np.asarray(gold_logps, dtype=float)
        prob = np.asarray(admitted_probs, dtype=float)
        y = np.asarray(context.labels, dtype=int)
        if len(arr) != len(context.texts):
            raise EvaluationError("Evaluation did not return one score per example")
        metric = float(arr.mean())
        pred = (prob >= 0.5).astype(int)
        auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None
        auprc = float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else None
        metrics = TaskMetrics(-metric, metric, {
            "mean_gold_logprob": metric,
            "auroc": auc,
            "auprc": auprc,
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "n_examples": int(len(arr)),
        })
        if return_activation:
            return metrics, np.concatenate(activations, axis=0), -arr
        return metrics

    def evaluate_snapshot(self, snapshot, task_context):
        return self._score(snapshot, task_context)

    def evaluate(self, candidate, task_context, baseline):
        if not candidate.valid or candidate.snapshot is None:
            raise ValueError("Cannot evaluate invalid candidate")
        metrics = self._score(candidate.snapshot, task_context)
        return EvaluationResult(
            candidate.candidate_id,
            metrics.task_loss,
            metrics.task_metric,
            baseline.task_loss,
            baseline.task_metric,
            metrics.task_metric - baseline.task_metric,
            metrics.extra_metrics,
        )

    def task_relevance(self, snapshot, task_context):
        _, activation, losses = self._score(snapshot, task_context, return_activation=True)
        return error_weighted_edge_relevance(activation, losses, normalize=True)


class ErrorWeightedTaskRelevanceProvider:
    """CandidateBuilder feature provider with per-state caching during frozen phases."""
    def __init__(self, evaluator: FrozenGraphCausalEvaluator):
        self.evaluator = evaluator
        self._cache = {}

    def __call__(self, state, reward_context, baseline):
        key = (state.state_id, id(reward_context))
        if key not in self._cache:
            self._cache[key] = self.evaluator.task_relevance(state.snapshot, reward_context)
        return {"task_relevance": self._cache[key]}


def adapt_task_model(model, snapshot, context: CausalTaskContext, optimizer, *, steps=200,
                     max_grad_norm=1.0):
    """Accepted-graph-only task adaptation.

    The caller decides which parameters require gradients (normally task LoRA and
    SoftGraphTokenizer only).  Bootstrap adapter/representations are outside this
    object and therefore cannot be refreshed by this function.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    torch = _torch()
    model.train()
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    # Stream batches instead of materializing the full train split on GPU.
    # When `steps` exceeds one pass, restart the deterministic context iterator.
    iterator = iter(context.batches(device))
    losses = []
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(context.batches(device))
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError("empty training context") from exc
        output, _ = model(global_graph=snapshot.Rho, **batch)
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Nonfinite task adaptation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"steps": steps, "initial_loss": losses[0], "final_loss": losses[-1],
            "mean_loss": float(np.mean(losses))}
