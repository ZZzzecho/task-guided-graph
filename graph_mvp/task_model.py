"""Local Transformers task-model helpers for graph-token injection.

Soft graph tokens require direct `inputs_embeds` access, so the train/eval path must
hold a local causal-LM object. A standard OpenAI-compatible vLLM/SGLang HTTP endpoint
cannot replace this path because chat APIs do not expose arbitrary continuous input
embeddings.
"""
from __future__ import annotations

GLM47_FLASH_MODEL_ID = "zai-org/GLM-4.7-Flash"
GLM47_FLASH_LORA_TARGETS = (
    "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
)


def load_local_causal_lm(model_name_or_path=GLM47_FLASH_MODEL_ID, *,
                         dtype="bfloat16", device_map="auto", local_files_only=False,
                         attn_implementation=None, trust_remote_code=False):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Local task-model loading requires torch + transformers") from exc
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    if dtype not in dtype_map:
        raise ValueError("dtype must be auto, float32, float16, or bfloat16")
    kwargs = {
        "device_map": device_map,
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
    }
    if dtype_map[dtype] != "auto":
        kwargs["torch_dtype"] = dtype_map[dtype]
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=bool(local_files_only),
        trust_remote_code=bool(trust_remote_code),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        # Some tokenizers expose multiple EOS IDs; only set pad when a scalar token is available.
        if isinstance(tokenizer.eos_token_id, int):
            tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return model, tokenizer


def default_lora_targets(model):
    """Return architecture-aware attention LoRA targets that actually exist."""
    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    if model_type == "glm4_moe_lite":
        preferred = GLM47_FLASH_LORA_TARGETS
    else:
        preferred = ("q_proj", "k_proj", "v_proj", "o_proj")
    suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    targets = tuple(x for x in preferred if x in suffixes)
    if not targets:
        raise ValueError(
            f"Could not find supported LoRA attention modules for model_type={model_type!r}; "
            "inspect model.named_modules() and pass target_modules explicitly."
        )
    return targets


def attach_task_lora(model, *, r=16, alpha=32, dropout=0.05, target_modules=None):
    """Attach PEFT LoRA to a local causal LM; GLM-4.7-Flash is auto-detected."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Task LoRA requires peft; install `pip install -e '.[llm]'`") from exc
    if target_modules is None:
        target_modules = default_lora_targets(model)
    cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return get_peft_model(model, cfg)
