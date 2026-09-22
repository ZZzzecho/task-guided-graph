# -*- coding: utf-8 -*-
import os
import re
import json
import math
import random
import argparse
import time
from collections import defaultdict
from datasets import concatenate_datasets
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)

from peft import LoraConfig, get_peft_model


# -----------------------------
# 1) 句子切分 + 重排版本生成
# -----------------------------
def split_sentences_regex(text: str):
    # 你给的版本：中英文标点都支持
    # 注意：有些摘要句子之间可能没有空格，这里也兼容
    return re.split(r'(?<=[.!?，。;])\s+', str(text).strip())

def shuffle_sentences(sentences, rng: random.Random):
    # 重要：不能原地打乱原列表，否则每轮会互相污染
    s = list(sentences)  # copy
    rng.shuffle(s)
    return s

def make_shuffled_versions(
    train_csv: str,
    out_csv: str,
    text_col: str = "medical_abstract",
    n_versions: int = 50,
    seed: int = 123,
):
    df = pd.read_csv(train_csv)
    if text_col not in df.columns:
        raise ValueError(f"找不到列 {text_col}，实际列: {list(df.columns)}")

    texts = df[text_col].astype(str)
    splittext = texts.apply(split_sentences_regex)

    rng = random.Random(seed)

    out = {}
    # 原始（不打乱）也保留一列，便于对照
    out["version_0"] = splittext.apply(lambda s: " ".join([x for x in s if x]))

    for i in range(n_versions):
        # 每个版本用独立 seed，保证可复现且版本间不同
        rng_i = random.Random(seed + i + 1)
        newtext = splittext.apply(lambda s: shuffle_sentences(s, rng_i))
        out[f"version_{i+1}"] = newtext.apply(lambda s: " ".join([x for x in s if x]))

    out_df = pd.DataFrame(out)
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] 生成 {n_versions} 个重排版本 + 原始版本，共 {out_df.shape[1]} 列。保存到: {out_csv}")


# -----------------------------
# 2) 概念集合（短语）读取 + tokenizer ids
# -----------------------------
def _normalize_phrases(phrases):
    seen = set()
    uniq = []
    for p in phrases:
        pp = " ".join(str(p).strip().split())
        if pp and pp not in seen:
            seen.add(pp)
            uniq.append(pp)
    return uniq


def load_phrases_from_qwen_json(path: str, topk_each_class: int = 0):
    """
    支持你前面输出的 qwen2_concepts_by_class.json:
      { "1": [...], "2": [...], ... }
    topk_each_class=0 表示全取；否则每类截断 topk。
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    phrases = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list):
                vv = v if topk_each_class <= 0 else v[:topk_each_class]
                phrases.extend([str(x) for x in vv if str(x).strip()])
    else:
        raise ValueError("概念 JSON 格式不符合预期，应为 dict(class -> list[str])")

    return _normalize_phrases(phrases)


def load_phrases_from_csv(path: str):
    """从 CSV 读取关键词列表，支持第一列或第二列为短语。"""
    df = pd.read_csv(path, header=None, encoding="utf-8")
    if df.shape[1] >= 2:
        phrases = df.iloc[:, 1].astype(str).tolist()
    else:
        phrases = df.iloc[:, 0].astype(str).tolist()
    return _normalize_phrases(phrases)


def load_phrases(path: str, topk_each_class: int = 0):
    if path.lower().endswith(".json"):
        return load_phrases_from_qwen_json(path, topk_each_class=topk_each_class)
    if path.lower().endswith(".csv"):
        return load_phrases_from_csv(path)
    raise ValueError("只能加载 .json 或 .csv 格式的概念词表，path=" + str(path))


def save_phrase_ids(tokenizer, phrases, save_path):
    phrase_ids = {phrase: tokenizer.encode(phrase, add_special_tokens=False) for phrase in phrases}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(phrase_ids, f, ensure_ascii=False, indent=2)
    # 返回 list[list[int]]
    return [v for v in phrase_ids.values()]


# -----------------------------
# 3) 你的 A 学习损失：尽量保持原样
# -----------------------------
from scipy.stats import norm

def huge_npn_gpu_shrinkage(x: torch.Tensor, eps=1e-6):
    """
    你代码的 NPN 变换（会走 numpy->cpu->gpu）；概念数不大时可接受
    """
    n, d = x.shape
    device = x.device
    ranks = torch.argsort(torch.argsort(x, dim=0), dim=0).float() + 1
    uniform = ranks / (n + 1)

    norm_ppf = torch.from_numpy(norm.ppf(uniform.detach().cpu().numpy())).to(device)
    norm_mean = norm_ppf.mean(dim=0, keepdim=True)
    norm_std = norm_ppf.std(dim=0, keepdim=True) + eps
    transformed = (norm_ppf - norm_mean) / norm_std
    return transformed

def compute_laplacian_from_A(A: torch.Tensor, threshold=1e-5):
    with torch.no_grad():
        A_cpu = A.detach().cpu()
        A_nodiag = A_cpu * (1 - torch.eye(A.shape[0]))
        adj = (A_nodiag.abs() > threshold).float()
        deg = torch.diag(adj.sum(dim=1))
        L = deg - adj
    return L.to(A.device, dtype=torch.float32)

def laplacian_regularization_loss(model, phrase_ids, L):
    emb_weight = model.get_input_embeddings().weight
    device = emb_weight.device

    phrase_embs = []
    for ids in phrase_ids:
        token_ids = torch.tensor(ids, dtype=torch.long, device=device)
        token_embs = emb_weight[token_ids]
        phrase_emb = token_embs.mean(dim=0)
        phrase_embs.append(phrase_emb)

    F = torch.stack(phrase_embs, dim=0)
    norms = torch.norm(F, p=2, dim=1, keepdim=True)
    F = F / (norms + 1e-8)
    if F.shape[0] != L.shape[0]:
        L = L[:F.shape[0], :F.shape[0]]

    F = F.to(dtype=torch.float32)
    reg = torch.trace(F.T @ L.to(device, dtype=torch.float32) @ F)
    return reg


class CustomLoss(nn.Module):
    """
    基于你 NMFLora 代码的 CustomLoss，保持同样结构：
    - A 是 p×p（p=概念数）
    - B 是 d×d（d=embed_dim）
    - trace/logdet + L1 + Laplacian
    """
    def __init__(
        self, model, phrase_ids,
        λ=2.5, γ=3.0,
        eps=1e-5,
        lambda_lap=1e-6
    ):
        super().__init__()
        self.model = model
        self.phrase_ids = phrase_ids

        device = model.device if hasattr(model, "device") else "cuda"
        embed_dim = model.get_input_embeddings().weight.shape[1]
        self.embed_dim = embed_dim

        self.p = len(phrase_ids)   # 概念数
        self.q = embed_dim         # 用 embedding 维度作为 q（与你代码一致）

        self.λ = λ
        self.γ = γ
        self.eps = eps
        self.lambda_lap = lambda_lap

        # A/B 上三角参数
        self.triu_size_A = self.p * (self.p + 1) // 2
        self.triu_size_B = embed_dim * (embed_dim + 1) // 2
        self.upper_vec_A = nn.Parameter(torch.zeros(self.triu_size_A, device=device))
        self.upper_vec_B = nn.Parameter(torch.zeros(self.triu_size_B, device=device))

        # 初始化成单位阵
        with torch.no_grad():
            indices_A = torch.triu_indices(self.p, self.p)
            eye_A = torch.eye(self.p, device=device)
            self.upper_vec_A.copy_(eye_A[indices_A[0], indices_A[1]])

            indices_B = torch.triu_indices(embed_dim, embed_dim)
            eye_B = torch.eye(embed_dim, device=device)
            self.upper_vec_B.copy_(eye_B[indices_B[0], indices_B[1]])

        # 初始化短语嵌入快照
        self.init_phrase_embs = self.get_phrase_embeddings().detach()

    def vec_to_sym_matrix(self, vec, size):
        matrix = torch.zeros(size, size, device=vec.device, dtype=vec.dtype)
        indices = torch.triu_indices(size, size)
        matrix[indices[0], indices[1]] = vec
        matrix = matrix + matrix.triu(1).T
        return matrix

    def normalize_phrase_embeddings(self, phrase_emb, epsilon=1e-8):
        norms = torch.norm(phrase_emb, p=2, dim=1, keepdim=True)
        return phrase_emb / (norms + epsilon)

    def get_phrase_embeddings(self):
        emb_weight = self.model.get_input_embeddings().weight
        phrase_embs = []
        for ids in self.phrase_ids:
            token_ids = torch.tensor(ids, dtype=torch.long, device=emb_weight.device)
            token_embs = emb_weight[token_ids]
            phrase_emb = token_embs.mean(dim=0)
            phrase_embs.append(phrase_emb)
        return torch.stack(phrase_embs, dim=0)

    def forward(self, outputs, labels):
        base_loss = outputs.loss

        # 当前短语嵌入
        f_x = self.get_phrase_embeddings()
        f_x = self.normalize_phrase_embeddings(f_x)
        f_x = huge_npn_gpu_shrinkage(f_x)
        f_x = f_x.to(self.model.device, dtype=torch.float32)

        # 计算 Δembedding
        delta_f_x = f_x - self.init_phrase_embs.to(f_x.device, dtype=f_x.dtype)

        # 对称矩阵
        A_sym = self.vec_to_sym_matrix(self.upper_vec_A, self.p)
        B_sym = self.vec_to_sym_matrix(self.upper_vec_B, self.embed_dim)

        # 迹项
        n = delta_f_x.shape[0]
        trace_term = torch.trace(delta_f_x.T @ A_sym @ delta_f_x @ B_sym) / n

        # 正则
        l1_A = torch.sum(torch.abs(A_sym) * (1 - torch.eye(self.p, device=A_sym.device)))
        l1_B = torch.sum(torch.abs(B_sym) * (1 - torch.eye(self.embed_dim, device=B_sym.device)))
        logdet_A = torch.logdet(A_sym + self.eps * torch.eye(self.p, device=A_sym.device))
        logdet_B = torch.logdet(B_sym + self.eps * torch.eye(self.embed_dim, device=A_sym.device))
        logdet_term = -self.q * logdet_A - self.p * logdet_B

        w_loss = logdet_term + trace_term + self.λ * l1_A + self.γ * l1_B

        # 无laplacian时的总损失（你原来代码的样子）

        # 拉普拉斯正则化（由 A 构图）
        L = compute_laplacian_from_A(A_sym)
        laplacian_loss = laplacian_regularization_loss(self.model, self.phrase_ids, L)

        total_loss = base_loss + 0.001 * w_loss + self.lambda_lap * laplacian_loss

        self.loss_debug_info = {
            "total_loss": float(total_loss.detach().cpu()),
            "base_loss": float(base_loss.detach().cpu()),
            "w_loss": float(w_loss.detach().cpu()),
            "l1_A": float(l1_A.detach().cpu()),
            "l1_B": float(l1_B.detach().cpu()),
            "laplacian_loss": float(laplacian_loss.detach().cpu()),
        }
        return total_loss


class CustomTrainer(Trainer):
    def __init__(self, loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        labels = inputs.get("labels")
        loss = self.loss_func(outputs, labels)

        if hasattr(self.loss_func, "loss_debug_info"):
            if self.state.global_step % 20 == 0:
                s = " | ".join([f"{k}: {v:.4f}" for k, v in self.loss_func.loss_debug_info.items()])
                print(f"[Loss Breakdown] {s}")

        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        # 把 A/B 参数也加进优化器
        extra_params = list(self.loss_func.parameters())
        if len(extra_params) > 0:
            print("✅ 添加 A/B 到优化器参数中")
            for group in self.optimizer.param_groups:
                group["params"].extend(extra_params)


# -----------------------------
# 4) dataset 构造：用某一列版本做 LM 训练
# -----------------------------
def build_lm_dataset_from_versions_csv(versions_csv: str, version_col: str, tokenizer, max_length: int = 512):
    df = pd.read_csv(versions_csv)
    if version_col not in df.columns:
        raise ValueError(f"找不到列 {version_col}，实际列: {list(df.columns)}")

    texts = df[version_col].astype(str).tolist()

    def format_function(examples):
        formatted_text = examples["text"] + tokenizer.eos_token
        enc = tokenizer(
            formatted_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0),
        }

    dataset = Dataset.from_dict({"text": texts}).map(format_function, remove_columns=["text"], batched=False)
    return dataset


def initialize_model_qwen2(model_path: str, use_bf16: bool = True):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map=None,
        trust_remote_code=False,
    ).to("cuda")

    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "embed_tokens"],  # 你的原设置
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    print("=== LoRA injected into modules (has lora_A) ===")
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            print(name)
    return model


# -----------------------------
# 5) 训练：可以指定用哪个 version 或“轮流训练多个 version”
# -----------------------------
def train_on_versions(
    model_path: str,
    versions_csv: str,
    phrase_json: str = None,
    out_dir: str = None,
    phrases: list = None,
    n_versions: int = 50,
    topk_each_class: int = 0,
    max_length: int = 512,
    per_device_train_batch_size: int = 48,
    grad_accum: int = 1,
    lr: float = 1e-4,
    epochs: int = 3,
    lambda_lap: float = 1e-6,
    seed: int = 123,
    versions_per_epoch: int = 20,
    use_version0: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 载入概念短语
    if phrases is None:
        if phrase_json is None:
            raise ValueError("需要提供 phrase_json 或直接传入 phrases")
        phrases = load_phrases(phrase_json, topk_each_class=topk_each_class)
    else:
        if not isinstance(phrases, (list, tuple)):
            raise ValueError("phrases 必须是 list 或 tuple")
        phrases = _normalize_phrases(phrases)

    if len(phrases) < 5:
        raise ValueError(f"概念短语太少：{len(phrases)}，请检查输入的词表")

    phrase_ids_path = os.path.join(out_dir, "target_phrase_ids.json")
    phrase_ids = save_phrase_ids(tokenizer, phrases, phrase_ids_path)
    print(f"[INFO] concepts={len(phrases)}; phrase_ids saved: {phrase_ids_path}")

    # 初始化模型 + LoRA
    model = initialize_model_qwen2(model_path, use_bf16=True)

    # 训练参数（尽量保持你原来风格；如果你环境不支持 save_strategy，就改成 save_steps）
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=10000,
        dataloader_num_workers=2,
        optim="adamw_torch",
        fp16=True,
        remove_unused_columns=False,
        output_dir=out_dir,
        report_to="none",
        seed=seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    # 关键：你原代码每次只训练一个 version_i；这里给你两种模式：
    # - 模式1：只用 version_1 训练（最快）
    # - 模式2：依次在 version_1..version_n 上继续训练（更符合“多视角重排”）
    # 我默认模式2：轮流训练所有版本，每个版本训练 epochs/n_versions（这里简化：每个版本都跑 epochs）
    loss_func = CustomLoss(model, phrase_ids, lambda_lap=lambda_lap)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=None,  # 后面动态换
        data_collator=data_collator,
        loss_func=loss_func,
    )

    # 轮流训练多个版本

    # 版本列名列表
    version_cols = [f"version_{i + 1}" for i in range(n_versions)]
    if use_version0:
        version_cols = ["version_0"] + version_cols

    rng = random.Random(seed)

    num_rows = sum(1 for _ in open(versions_csv, "r", encoding="utf-8")) - 1
    num_chosen = min(versions_per_epoch, len(version_cols))
    steps_per_epoch = math.ceil(num_rows * num_chosen / per_device_train_batch_size)
    print(
        f"[INFO] approx {steps_per_epoch} steps/epoch "
        f"({num_rows} rows × {num_chosen} versions / batch {per_device_train_batch_size})"
    )

    # 每个 epoch：随机抽 versions_per_epoch 个 version 拼起来，只训练 1 个 epoch
    original_num_train_epochs = training_args.num_train_epochs
    total_start = time.time()

    for ep in range(original_num_train_epochs):
        epoch_start = time.time()
        chosen = rng.sample(version_cols, k=min(versions_per_epoch, len(version_cols)))
        print(
            f"\n[INFO] Epoch {ep + 1}/{original_num_train_epochs} | sampled versions: {chosen[:5]} ... (total {len(chosen)})")

        # 拼接多个版本的数据集（注意：这里仍会 tokenize -> 但你说不改其他，我就不加缓存）
        dss = []
        for col in chosen:
            ds = build_lm_dataset_from_versions_csv(versions_csv, col, tokenizer, max_length=max_length)
            dss.append(ds)

        mixed = concatenate_datasets(dss)
        # 可选：打乱一下（更随机）
        mixed = mixed.shuffle(seed=seed + ep)

        # 关键：这一轮只训练 1 个 epoch
        trainer.args.num_train_epochs = 1
        trainer.train_dataset = mixed
        trainer.train(resume_from_checkpoint=None)

        epoch_sec = time.time() - epoch_start
        eta_sec = epoch_sec * (original_num_train_epochs - ep - 1)
        print(
            f"[TIME] epoch {ep + 1}/{original_num_train_epochs} 用时 {epoch_sec:.1f}s, "
            f"估计剩余 {eta_sec/60:.1f} 分钟"
        )

    total_sec = time.time() - total_start
    print(f"[TIME] total training time: {total_sec:.1f}s ({total_sec/60:.1f} min)")

    # 恢复（不恢复也无所谓，但写上更干净）
    trainer.args.num_train_epochs = original_num_train_epochs

    # 保存 A
    A_vec = trainer.loss_func.upper_vec_A.detach().cpu().numpy()
    npy_path = os.path.join(out_dir, "learned_upper_vec_A.npy")
    np.save(npy_path, A_vec)
    print("[OK] saved upper_vec_A:", npy_path)

    # 同时保存 A_sym（更方便画图）
    A_sym = trainer.loss_func.vec_to_sym_matrix(trainer.loss_func.upper_vec_A, trainer.loss_func.p)
    A_sym_path = os.path.join(out_dir, "learned_A_sym_big.npy")
    np.save(A_sym_path, A_sym.detach().cpu().numpy())
    print("[OK] saved A_sym:", A_sym_path)

    # 导出 edge list（关系图）
    export_graph_edges(A_sym.detach().cpu().numpy(), phrases, out_dir)


def export_graph_edges(A_sym: np.ndarray, phrases, out_dir: str, top_edges: int = 500, threshold: float = 0.0):
    """
    导出概念关系图边列表（i,j,weight,concept_i,concept_j）
    - 默认取绝对值最大的 top_edges 条边（去掉对角线）
    - threshold>0 只保留 abs(weight)>=threshold
    """
    p = A_sym.shape[0]
    edges = []
    for i in range(p):
        for j in range(i+1, p):
            w = float(A_sym[i, j])
            if threshold > 0 and abs(w) < threshold:
                continue
            edges.append((i, j, w, phrases[i], phrases[j]))

    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    edges = edges[:top_edges]

    out_path = os.path.join(out_dir, "concept_graph_edges.csv")
    pd.DataFrame(edges, columns=["i", "j", "weight", "concept_i", "concept_j"]).to_csv(out_path, index=False)
    print("[OK] saved edge list:", out_path)


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="/laijizheng/QwenGAT/Medical-Abstracts-TC-Corpus-main/medical_tc_train.csv")
    parser.add_argument("--concept_json", type=str, default=None, help="Qwen2 抽取的概念集合 JSON，如 qwen2_concepts_by_class.json")
    parser.add_argument("--phrase_csv", type=str, default=None, help="中文关键词集合 CSV，第一列或第二列为关键词")
    parser.add_argument("--model_path", type=str, default="/laijizheng/models/qwen/Qwen2-7B-Instruct")
    parser.add_argument("--out_dir", type=str, default="/laijizheng/QwenGAT/medical_tc_concept_graph_out(nolaplacian)")

    parser.add_argument("--make_versions", action="store_true", help="先生成 50 个句子重排版本 CSV")
    parser.add_argument("--versions_csv", type=str, default="/laijizheng/QwenGAT/Medical-Abstracts-TC-Corpus-main/50_versions_medical_tc.csv")
    parser.add_argument("--n_versions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)

    # 概念控制
    parser.add_argument("--topk_each_class", type=int, default=0, help="每类最多取多少概念，0=不截断")

    # 训练控制
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1, help="每个版本训练的 epoch 数")
    parser.add_argument("--lambda_lap", type=float, default=1e-6)
    parser.add_argument("--num_versions", type=int, default=50)


    args = parser.parse_args()
    # args.concept_json = '/laijizheng/QwenGAT/Medical-Abstracts-TC-Corpus-main/qwen2_concepts_by_class_big.json'

    if args.num_versions != 50:
        args.n_versions = args.num_versions

    if args.make_versions:
        make_shuffled_versions(
            train_csv=args.train_csv,
            out_csv=args.versions_csv,
            text_col="medical_abstract",
            n_versions=args.n_versions,
            seed=args.seed,
        )

    if not os.path.exists(args.versions_csv):
        raise FileNotFoundError(f"找不到 versions_csv: {args.versions_csv}；请先加 --make_versions 生成")

    if args.concept_json is None and args.phrase_csv is None:
        raise ValueError("请提供 --concept_json 或 --phrase_csv 之一")
    if args.concept_json is not None and args.phrase_csv is not None:
        raise ValueError("请只传入一个：--concept_json 或 --phrase_csv")

    csv_phrases = None
    if args.phrase_csv is not None:
        csv_phrases = load_phrases_from_csv(args.phrase_csv)

    train_on_versions(
        model_path=args.model_path,
        versions_csv=args.versions_csv,
        phrase_json=args.concept_json,
        out_dir=args.out_dir,
        phrases=csv_phrases,
        n_versions=args.n_versions,
        topk_each_class=args.topk_each_class,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        epochs=args.epochs,
        lambda_lap=args.lambda_lap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()



# python NMF_medical_abstract.py \
#   --make_versions \
#   --concept_json /laijizheng/QwenGAT/Medical-Abstracts-TC-Corpus-main/qwen2_concepts_by_class.json

# nohup python NMF_medical_abstract.py \
#   --concept_json /laijizheng/QwenGAT/Medical-Abstracts-TC-Corpus-main/qwen2_concepts_by_class_big.json \
#   --topk_each_class 0 \
#   --n_versions 50 \
#   --epochs 3 \
#   --max_length 384 > medicalabstract_NMF.log 2>&1 &

