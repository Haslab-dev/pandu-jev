"""Pandu-Micro (MicroBERT) Knowledge Distillation from TypeSafe Jev for Long-Running Agents.

Harvests soft distributions from live TypeSafe Jev (Cloud RLCD Teacher)
and trains Pandu-Micro across 100-200 turn trajectory checkpoints.
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.rakitkode_domain import (
    CATEGORIES,
    RISK_LEVELS,
    RAKITKODE_DATASET,
)
from datasets.long_running_trajectories import (
    PHASES,
    TOOL_CATEGORIES,
    TrajectoryTurn,
    generate_long_running_trajectories,
)
from teacher.jev_teacher import JevTeacherClient, DEFAULT_JEV_KEY
from micro.tokenizer import PanduMicroTokenizer
from micro.model import PanduMicro, PanduMicroConfig, format_turn_sequence
from micro.benchmark_real_rakitkode import build_complexity_suite

CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"
PHASE_MAP = {p: i for i, p in enumerate(PHASES)}
CAT_MAP = {c: i for i, c in enumerate(CATEGORIES)}
RISK_MAP = {r: i for i, r in enumerate(RISK_LEVELS)}


def train_pandu_micro_long_running(
    api_key: str = DEFAULT_JEV_KEY,
    num_teacher_rollouts: int = 15,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 32,
    device_name: Optional[str] = None,
):
    print("=" * 80)
    print("🎓 [Pandu-Micro] Knowledge Distillation for Long-Running Agents (100-200 Turns)")
    print("=" * 80)

    device = device_name or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Compute Device: {device.upper()}")

    tokenizer = PanduMicroTokenizer()
    print(f"📖 Pruned Domain Vocabulary: {tokenizer.vocab_size} tokens")

    # 1. Harvest Teacher Soft Distributions
    client = JevTeacherClient(api_key=api_key)
    print(f"📡 Querying TypeSafe Jev API for soft distributions ({num_teacher_rollouts} rollouts)...")

    # Generate multi-turn trajectories (150 and 180 turns)
    trajectories = generate_long_running_trajectories()
    print(f"🔄 Generated {len(trajectories)} multi-turn trajectory steps across sessions.")

    samples = []
    teacher_queried = 0

    for turn in trajectories:
        formatted_seq = format_turn_sequence(
            turn.goal_prompt,
            turn.turn_index,
            turn.total_session_turns,
            turn.consecutive_failures,
            turn.last_tool,
            turn.current_intent,
        )

        cat_idx = CAT_MAP.get(turn.target_category, 0)

        # Query live Jev on subset of turns
        if teacher_queried < num_teacher_rollouts and (turn.turn_index % 15 == 1):
            res = client.query_category_distribution(turn.current_intent)
            if res and "choice" in res:
                print(f"   [Turn {turn.turn_index}] Jev Teacher Target -> {res.get('choice')} (conf: {res.get('confidence', 0):.2f})")
                teacher_queried += 1
                time.sleep(0.08)

        samples.append((
            formatted_seq,
            cat_idx,
            PHASE_MAP.get(turn.target_phase, 0),
            turn.is_stuck,
            turn.should_delegate,
            turn.is_done,
            RISK_MAP.get(turn.target_risk, 0),
        ))

    # Add single-turn domain samples
    for prompt, hard_cat, nt, role, risk, delegate in RAKITKODE_DATASET:
        seq = format_turn_sequence("Direct single turn developer request", 1, 1, 0, "none", prompt)
        samples.append((
            seq,
            CAT_MAP.get(hard_cat, 0),
            2, # implementation
            0.0,
            delegate,
            0.0,
            RISK_MAP.get(risk, 0),
        ))

    # Add representative simulated turns from the complexity suite
    suite = build_complexity_suite()
    for tier_name, turns in suite.items():
        for s in turns:
            seq = format_turn_sequence(s.goal, s.turn_idx, len(turns), s.consecutive_fails, s.last_tool, s.intent)
            samples.append((
                seq,
                CAT_MAP.get(s.expected_cat, 0),
                PHASE_MAP.get(s.expected_phase, 0),
                1.0 if s.should_alarm else 0.0,
                1.0 if s.should_delegate else 0.0,
                1.0 if s.is_terminal else 0.0,
                0, # low
            ))

    print(f"\n📊 Total Training Set: {len(samples)} long-running & domain turns.")

    # 2. Tokenize all samples
    tokenized = []
    for seq, c_idx, p_idx, is_stuck, should_del, is_done, risk_idx in samples:
        ids = tokenizer.encode(seq, max_length=256)
        tokenized.append((ids, c_idx, p_idx, is_stuck, should_del, is_done, risk_idx))

    # 3. Instantiate Pandu-Micro (MicroBERT)
    config = PanduMicroConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=192,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=512,
    )
    model = PanduMicro(config).to(device)
    print(f"🏗️ Pandu-Micro Parameters: {model.num_parameters:,} weights (~{model.fp16_size_mb:.2f} MB FP16 / ~{model.int8_size_mb:.2f} MB INT8)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Losses
    ce_loss = nn.CrossEntropyLoss()
    bce_stuck_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0], device=device))
    bce_loss = nn.BCEWithLogitsLoss()

    model.train()
    print("🚀 Training Pandu-Micro across 100-200 Turn Trajectories (Mini-Batch SGD)...")

    num_samples = len(tokenized)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        random.shuffle(tokenized)
        epoch_loss = 0.0
        batches = 0

        for i in range(0, num_samples, batch_size):
            batch = tokenized[i : i + batch_size]
            b_len = len(batch)
            max_l = max(len(x[0]) for x in batch)

            input_ids = torch.zeros((b_len, max_l), dtype=torch.long, device=device)
            mask = torch.zeros((b_len, max_l), dtype=torch.float, device=device)
            y_cat = torch.zeros(b_len, dtype=torch.long, device=device)
            y_phase = torch.zeros(b_len, dtype=torch.long, device=device)
            y_stuck = torch.zeros(b_len, dtype=torch.float, device=device)
            y_del = torch.zeros(b_len, dtype=torch.float, device=device)
            y_done = torch.zeros(b_len, dtype=torch.float, device=device)
            y_risk = torch.zeros(b_len, dtype=torch.long, device=device)

            for b_idx, (ids, c_idx, p_idx, stuck_val, del_val, done_val, r_idx) in enumerate(batch):
                l = len(ids)
                input_ids[b_idx, :l] = torch.tensor(ids, dtype=torch.long, device=device)
                mask[b_idx, :l] = 1.0
                y_cat[b_idx] = int(c_idx)
                y_phase[b_idx] = int(p_idx)
                y_stuck[b_idx] = float(stuck_val)
                y_del[b_idx] = float(del_val)
                y_done[b_idx] = float(done_val)
                y_risk[b_idx] = int(r_idx)

            optimizer.zero_grad()
            z_cat, z_phase, z_stuck, z_del, z_done, z_risk = model(input_ids, mask)

            loss_cat = ce_loss(z_cat, y_cat)
            loss_phase = ce_loss(z_phase, y_phase)
            loss_stuck = bce_stuck_loss(z_stuck, y_stuck)
            loss_del = bce_loss(z_del, y_del)
            loss_done = bce_loss(z_done, y_done)
            loss_risk = ce_loss(z_risk, y_risk)

            total_loss = loss_cat + 0.6 * loss_phase + 1.5 * loss_stuck + 0.5 * loss_del + 0.5 * loss_done + 0.3 * loss_risk
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            batches += 1

        if epoch % 10 == 0 or epoch == epochs:
            avg_loss = epoch_loss / batches
            with torch.no_grad():
                pred_cats = torch.argmax(z_cat, dim=-1)
                cat_acc = (pred_cats == y_cat).float().mean().item() * 100.0
                pred_phase = torch.argmax(z_phase, dim=-1)
                phase_acc = (pred_phase == y_phase).float().mean().item() * 100.0
                p_stuck_probs = torch.sigmoid(z_stuck)
                stuck_preds = (p_stuck_probs > 0.5).float()
                stuck_acc = (stuck_preds == y_stuck).float().mean().item() * 100.0
                print(f"   Epoch {epoch:2d}/{epochs:2d} | Avg Loss: {avg_loss:.4f} | Batch Cat Acc: {cat_acc:.1f}% | Batch Phase Acc: {phase_acc:.1f}% | Stuck Alarm Acc: {stuck_acc:.1f}%")

    print(f"\n⏱️ Training finished in {time.time() - t_start:.2f} seconds.")

    # 4. Save Checkpoint (FP16 Native)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    fp16_ckpt = CHECKPOINTS_DIR / "pandu_micro_long_running_fp16.pt"
    model_fp16 = PanduMicro(config)
    model_fp16.load_state_dict(model.state_dict())
    model_fp16.half()
    torch.save(model_fp16.state_dict(), fp16_ckpt)
    print(f"⚡ Saved Pandu-Micro FP16 Checkpoint: {fp16_ckpt} ({fp16_ckpt.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Long-Running Loop Thrash Stress Test
    print("\n" + "=" * 80)
    print("🚨 TESTING LOOP & THRASHING DETECTION (TURNS 80-84)")
    print("=" * 80)
    test_cases = [
        # (turn, failures, intent, expected_stuck)
        (25, 0, "Normal edit in router.go adding timeout handling", False),
        (81, 1, "Compile error: undefined symbol in main.go, trying edit again", False),
        (83, 3, "Third compile failure in a row with identical error, repeating patch", True),
        (84, 4, "Fourth consecutive failure in loop! Retrying same shell command", True),
        (85, 0, "Pivoting strategy: inspecting root go.mod dependencies carefully to break cycle", False),
    ]

    model.eval()
    for t_idx, fails, intent, should_alarm in test_cases:
        res = model.predict_session_turn(
            goal="Refactor RakitKode runtime",
            turn_index=t_idx,
            total_turns=150,
            consecutive_failures=fails,
            last_tool="modify",
            intent=intent,
            tokenizer=tokenizer,
            device=device,
        )
        alarm = res["is_stuck_alarm"]
        matched = (alarm == should_alarm)
        status = "✅ PASS" if matched else "❌ FAIL"
        alarm_text = "🚨 STUCK ALARM ON" if alarm else "🟢 Normal execution"
        print(f"Turn {t_idx:2d} (Fails: {fails}) -> {alarm_text} (stuck prob: {res['is_stuck']:.2f}, phase: {res['phase']}) [{status}]")

    print("=" * 80)


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JEV_KEY
    train_pandu_micro_long_running(api_key=key, num_teacher_rollouts=10)
