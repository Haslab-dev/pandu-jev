#!/usr/bin/env python3
"""High-Performance Vectorized Training for Pandu-Jev NLP Snake Decision Model.

Pre-collates and vectorizes all tensors before the training loop for zero-overhead MPS execution.
Teaches Pandu-Jev NLP semantic distinction of safe vs collision routes and food-seeking behavior.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
LAYA_PATH = Path("/Users/hy4-mac-002/hasdev/research/laya-coreml")

for p in [str(SRC_DIR), str(LAYA_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from laya_coreml.snake.game import DIRECTIONS, SnakeGame
from models.jev_nlp import (
    PanduJevNLP,
    PanduJevNLPConfig,
    build_sequence,
    render_options,
)


def generate_training_data(num_episodes: int = 30, max_steps: int = 80) -> List[Dict]:
    """Generate diverse state-decision pairs with target probabilities."""
    data = []
    
    for ep in range(num_episodes):
        game = SnakeGame(24, 16, seed=2000 + ep, initial_length=6)
        
        for _ in range(max_steps):
            if not game.alive:
                break
                
            moves = game.moves()
            safe = [m for m in moves if m.safe]
            preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
            reachable, space = game.food_reachability()
            
            state = (
                f"Safe route: {'yes' if safe else 'no'}. "
                f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
            )
            
            # Semantic criteria
            move_criteria = {}
            target_logits = {}
            for m in moves:
                if not m.legal:
                    move_criteria[m.direction] = "Blocked. Collision."
                    target_logits[m.direction] = -8.0
                elif not m.safe:
                    move_criteria[m.direction] = "Unsafe. Traps the snake."
                    target_logits[m.direction] = -4.0
                elif m.eats:
                    move_criteria[m.direction] = "Safe. Eat food now. Best."
                    target_logits[m.direction] = 6.0
                elif m.direction == preferred:
                    move_criteria[m.direction] = "Safe. Best route to food."
                    target_logits[m.direction] = 4.0
                else:
                    move_criteria[m.direction] = "Safe. Slower route."
                    target_logits[m.direction] = 1.0
                    
            ordered_logits = np.array([target_logits[d] for d in DIRECTIONS], dtype=np.float32)
            exp_z = np.exp(ordered_logits - np.max(ordered_logits))
            target_probs = exp_z / np.sum(exp_z)
            
            data.append({
                "state": state,
                "question": {
                    "type": "choice",
                    "instructions": "Choose the best safe move toward food.",
                    "criteria": move_criteria,
                },
                "target_probs": target_probs,
                "qtype": 0,
                "k": 4,
            })
            
            # Step game
            next_move = preferred if preferred != "NONE" else (safe[0].direction if safe else moves[0].direction)
            game.step(next_move)
            
    return data


def train_fast(episodes: int = 35, epochs: int = 8, batch_size: int = 64, lr: float = 1e-3):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- Fast Vectorized Pandu-Jev NLP Training on {device.upper()} ---")
    
    t0 = time.perf_counter()
    raw_samples = generate_training_data(num_episodes=episodes, max_steps=80)
    print(f"Generated {len(raw_samples)} decision trajectories in {time.perf_counter() - t0:.2f}s")
    
    model = PanduJevNLP().to(device)
    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id or 50283
    
    # 1. Pre-tokenize all sequences
    tokenized_items = []
    for s in raw_samples:
        ids, markers = build_sequence(tokenizer, s["state"], s["question"])
        tokenized_items.append({
            "ids": ids,
            "markers": markers[:4],
            "target": s["target_probs"],
        })
        
    N = len(tokenized_items)
    max_len = max(len(x["ids"]) for x in tokenized_items)
    
    # 2. Pre-allocate contiguous pinned tensors
    input_ids_all = torch.full((N, max_len), pad_id, dtype=torch.long)
    attn_mask_all = torch.zeros((N, max_len), dtype=torch.float32)
    marker_pos_all = torch.zeros((N, 4), dtype=torch.long)
    marker_mask_all = torch.ones((N, 4), dtype=torch.float32)
    qtype_all = torch.zeros(N, dtype=torch.long)
    targets_all = torch.zeros((N, 4), dtype=torch.float32)
    
    for i, item in enumerate(tokenized_items):
        l = len(item["ids"])
        input_ids_all[i, :l] = torch.tensor(item["ids"], dtype=torch.long)
        attn_mask_all[i, :l] = 1.0
        m = item["markers"]
        marker_pos_all[i, :len(m)] = torch.tensor(m, dtype=torch.long)
        targets_all[i] = torch.tensor(item["target"], dtype=torch.float32)
        
    print(f"Tensors pre-allocated: [N={N}, SeqLen={max_len}, Options=4]")
    
    # Move whole dataset to device for zero copy overhead
    input_ids_all = input_ids_all.to(device)
    attn_mask_all = attn_mask_all.to(device)
    marker_pos_all = marker_pos_all.to(device)
    marker_mask_all = marker_mask_all.to(device)
    qtype_all = qtype_all.to(device)
    targets_all = targets_all.to(device)
    
    # 3. Train
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_start = time.perf_counter()
    indices = torch.randperm(N, device=device)
    
    for epoch in range(1, epochs + 1):
        indices = torch.randperm(N, device=device)
        epoch_loss = 0.0
        batches = 0
        ep_t0 = time.perf_counter()
        
        for start_idx in range(0, N, batch_size):
            b_idx = indices[start_idx : start_idx + batch_size]
            
            b_ids = input_ids_all[b_idx]
            b_mask = attn_mask_all[b_idx]
            b_mpos = marker_pos_all[b_idx]
            b_mmask = marker_mask_all[b_idx]
            b_qtype = qtype_all[b_idx]
            b_target = targets_all[b_idx]
            
            optimizer.zero_grad()
            logits, _ = model(b_ids, b_mask, b_mpos, b_mmask, b_qtype)
            
            # Fully vectorized KL Divergence Loss
            log_probs = F.log_softmax(logits[:, :4], dim=-1)
            loss = F.kl_div(log_probs, b_target, reduction="batchmean")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        scheduler.step()
        ep_elapsed = time.perf_counter() - ep_t0
        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss / batches:.4f} | Time: {ep_elapsed:.3f}s | Speed: {N / ep_elapsed:.0f} states/s", flush=True)
        
    total_time = time.perf_counter() - train_start
    print(f"\n--- Training Complete in {total_time:.2f}s! Final Loss: {epoch_loss/batches:.4f} ---")
    
    # 4. Save checkpoint
    out_dir = PROJECT_ROOT / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "pandu_snake_nlp.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved trained weights to {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    
    # 5. Quick validation check
    model.eval()
    with torch.no_grad():
        val_sample = raw_samples[0]
        val_output = model.predict(val_sample["state"], {"move": val_sample["question"]})
        probs = val_output["answers"]["move"]["probabilities"]
        print(f"\nValidation test on sample 0:")
        print(f"Learned Probabilities: {probs}")
        print(f"Target Probabilities:  {{DIRECTIONS[i]: round(float(val_sample['target_probs'][i]), 4) for i in range(4)}}")


if __name__ == "__main__":
    train_fast(episodes=35, epochs=8, batch_size=64, lr=1e-3)
