"""MicroBERT Teacher-Student Distillation Pipeline.

Harvests dark knowledge from live TypeSafe Jev (Cloud RLCD Teacher)
and distills it into MicroBERT (~1.5M params) on Apple Silicon MPS/CPU.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.rakitkode_domain import (
    CATEGORIES,
    ROLES,
    RISK_LEVELS,
    RAKITKODE_DATASET,
)
from teacher.jev_teacher import JevTeacherClient, DEFAULT_JEV_KEY
from tiny.tokenizer import MicroTokenizer
from tiny.microbert import MicroBERT, MicroBERTConfig

CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"


def collate_batch(samples: List[Tuple[str, np.ndarray, float, int, int, float]], tokenizer: MicroTokenizer, device: str):
    """Pad and collate a list of samples into PyTorch tensors."""
    max_len = 0
    tokenized = []
    for prompt, cat_dist, nt, role_idx, risk_idx, delegate in samples:
        ids = tokenizer.encode(prompt, max_length=192)
        tokenized.append((ids, cat_dist, nt, role_idx, risk_idx, delegate))
        if len(ids) > max_len:
            max_len = len(ids)

    b = len(samples)
    input_ids = torch.zeros((b, max_len), dtype=torch.long, device=device)
    attn_mask = torch.zeros((b, max_len), dtype=torch.float, device=device)
    y_cat = torch.zeros((b, len(CATEGORIES)), dtype=torch.float, device=device)
    y_tools = torch.zeros(b, dtype=torch.float, device=device)
    y_role = torch.zeros(b, dtype=torch.long, device=device)
    y_risk = torch.zeros(b, dtype=torch.long, device=device)
    y_del = torch.zeros(b, dtype=torch.float, device=device)

    for i, (ids, cat_dist, nt, role_idx, risk_idx, delegate) in enumerate(tokenized):
        l = len(ids)
        input_ids[i, :l] = torch.tensor(ids, dtype=torch.long, device=device)
        attn_mask[i, :l] = 1.0
        y_cat[i] = torch.tensor(cat_dist, dtype=torch.float, device=device)
        y_tools[i] = float(nt)
        y_role[i] = int(role_idx)
        y_risk[i] = int(risk_idx)
        y_del[i] = float(delegate)

    return input_ids, attn_mask, y_cat, y_tools, y_role, y_risk, y_del


def train_microbert_distillation(
    api_key: str = DEFAULT_JEV_KEY,
    num_teacher_rollouts: int = 15,
    epochs: int = 60,
    lr: float = 3e-4,
    device_name: Optional[str] = None,
):
    print("=" * 72)
    print("🎓 [Tier 1: Tiny] MicroBERT Knowledge Distillation from TypeSafe Jev")
    print("=" * 72)

    device = device_name or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Compute Device: {device.upper()}")

    tokenizer = MicroTokenizer()
    print(f"📖 Pruned Domain Vocabulary: {tokenizer.vocab_size} tokens")

    # 1. Harvest Teacher Soft Targets
    client = JevTeacherClient(api_key=api_key)
    print(f"📡 Querying TypeSafe Jev API for soft distributions ({num_teacher_rollouts} rollouts)...")

    role_map = {r: i for i, r in enumerate(ROLES)}
    risk_map = {k: i for i, k in enumerate(RISK_LEVELS)}

    dataset_with_teacher = []
    for i, item in enumerate(RAKITKODE_DATASET):
        prompt, hard_cat, nt, role, risk, delegate = item
        role_idx = role_map[role]
        risk_idx = risk_map[risk]

        if i < num_teacher_rollouts:
            res = client.query_category_distribution(prompt)
            if res and "probabilities" in res:
                probs = res["probabilities"]
                soft_vec = np.zeros(len(CATEGORIES), dtype=np.float32)
                for cat_name, p in probs.items():
                    if cat_name in CATEGORIES:
                        soft_vec[CATEGORIES.index(cat_name)] = float(p)
                if np.sum(soft_vec) > 0:
                    soft_vec /= np.sum(soft_vec)
                dataset_with_teacher.append((prompt, soft_vec, nt, role_idx, risk_idx, delegate))
                print(f"   [{i+1}/{num_teacher_rollouts}] Jev Teacher Soft Target -> {res.get('choice')} (conf: {res.get('confidence', 0):.2f})")
                time.sleep(0.08)
                continue

        hard_vec = np.zeros(len(CATEGORIES), dtype=np.float32)
        hard_vec[CATEGORIES.index(hard_cat)] = 1.0
        dataset_with_teacher.append((prompt, hard_vec, nt, role_idx, risk_idx, delegate))

    print(f"\n📊 Prepared {len(dataset_with_teacher)} domain training examples.")

    # 2. Instantiate MicroBERT
    config = MicroBERTConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=192,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=512,
    )
    model = MicroBERT(config).to(device)
    print(f"🏗️ MicroBERT Parameters: {model.num_parameters:,} weights (~{model.disk_size_mb:.2f} MB FP32)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # 3. Training Loop with Hinton KD Loss
    kd_temp = 2.0
    alpha = 0.6  # 60% KD loss, 40% hard loss
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    model.train()
    print("🚀 Training MicroBERT with Hinton KD + Multi-Task Heads...")

    input_ids, attn_mask, y_cat, y_tools, y_role, y_risk, y_del = collate_batch(
        dataset_with_teacher, tokenizer, device
    )

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z_cat, z_tools, z_role, z_risk, z_del = model(input_ids, attn_mask)

        # Hinton KD Loss on Category: KL divergence between student and teacher soft probabilities
        log_p_student = F.log_softmax(z_cat / kd_temp, dim=-1)
        p_teacher = y_cat  # Soft distribution from teacher
        loss_kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (kd_temp ** 2)

        # Cross Entropy / Hard Losses
        target_indices = torch.argmax(y_cat, dim=-1)
        loss_cat_ce = ce_loss(z_cat, target_indices)

        loss_cat = alpha * loss_kd + (1.0 - alpha) * loss_cat_ce
        loss_tools = bce_loss(z_tools, y_tools)
        loss_role = ce_loss(z_role, y_role)
        loss_risk = ce_loss(z_risk, y_risk)
        loss_del = bce_loss(z_del, y_del)

        total_loss = loss_cat + 0.5 * loss_tools + 0.5 * loss_role + 0.5 * loss_risk + 0.3 * loss_del
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                preds = torch.argmax(z_cat, dim=-1)
                acc = (preds == target_indices).float().mean().item() * 100.0
                print(f"   Epoch {epoch:2d}/{epochs:2d} | Loss: {total_loss.item():.4f} (KD: {loss_kd.item():.4f}) | Category Acc: {acc:.1f}%")

    # 4. Save Checkpoints (FP32 & Native Apple Silicon FP16)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    fp32_ckpt = CHECKPOINTS_DIR / "microbert_distilled.pt"
    torch.save(model.state_dict(), fp32_ckpt)
    print(f"💾 Saved FP32 Model Checkpoint: {fp32_ckpt} ({fp32_ckpt.stat().st_size / (1024*1024):.2f} MB)")

    # Native FP16 Checkpoint (Half Precision for Apple Silicon MPS & ANE)
    model_fp16 = MicroBERT(config)
    model_fp16.load_state_dict(torch.load(fp32_ckpt, map_location="cpu"))
    model_fp16.half()
    fp16_ckpt = CHECKPOINTS_DIR / "microbert_distilled_fp16.pt"
    torch.save(model_fp16.state_dict(), fp16_ckpt)
    print(f"⚡ Saved FP16 Model Checkpoint: {fp16_ckpt} ({fp16_ckpt.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Adversarial Negation Benchmark Test
    print("\n" + "=" * 72)
    print("🧪 Adversarial Negation Resolution Stress-Test (MicroBERT)")
    print("=" * 72)
    negation_tests = [
        ("Do NOT modify the file, only view lines 10 to 40", "read"),
        ("Inspect main.go, but under no circumstance edit or rewrite anything", "read"),
        ("Do not run the unit tests yet, first search for where the test is located", "explore"),
        ("Check git status and diff, do NOT commit or push to remote", "git"),
        ("Only build the binary, do NOT run the test suite", "execute"),
    ]

    correct = 0
    for prompt, expected in negation_tests:
        res = model.predict_turn(prompt, tokenizer, device=device)
        matched = (res["category"] == expected)
        if matched:
            correct += 1
        mark = "✅" if matched else "❌"
        print(f"   {mark} \"{prompt[:55]}...\" -> {res['category']} (conf: {res['confidence']:.2f}, expected: {expected})")

    print(f"\n🎯 Negation Resolution Accuracy: {correct}/{len(negation_tests)} ({correct/len(negation_tests)*100:.0f}%)")
    print("=" * 72)


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JEV_KEY
    rolls = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    train_microbert_distillation(api_key=key, num_teacher_rollouts=rolls)
