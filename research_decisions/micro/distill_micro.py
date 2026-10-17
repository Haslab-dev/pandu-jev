"""Teacher-Student Knowledge Distillation for Micro Decision Model (<100KB).

Harvests soft probability distributions from live TypeSafe Jev (Teacher)
and distills into the ultra-compact Student model using Hinton KD + Brier loss.
"""

import os
import sys
import time
import numpy as np
from typing import Optional

from datasets.rakitkode_domain import (
    CATEGORIES,
    ROLES,
    RISK_LEVELS,
    RAKITKODE_DATASET,
)
from teacher.jev_teacher import JevTeacherClient, DEFAULT_JEV_KEY
from micro.model import (
    FEATURE_SIZE,
    VOCABULARY,
    RLCDSmallNeuralModel,
    extract_features,
)
from micro.export_rakitkode import export_to_rakitkode


def run_micro_distillation(
    api_key: str = DEFAULT_JEV_KEY,
    num_rollouts: int = 15,
    epochs: int = 350,
):
    print("=" * 70)
    print("🔬 [Tier 0: Micro] TypeSafe Jev -> Micro Decision Model Distillation")
    print("=" * 70)

    client = JevTeacherClient(api_key=api_key)
    print(f"📡 Querying Teacher Jev API for soft distributions ({num_rollouts} rollouts)...")

    dataset_with_teacher = []
    for i, item in enumerate(RAKITKODE_DATASET):
        prompt, hard_cat, nt, role, risk, delegate = item
        if i < num_rollouts:
            res = client.query_category_distribution(prompt)
            if res and "probabilities" in res:
                probs = res["probabilities"]
                soft_vec = np.zeros(len(CATEGORIES), dtype=np.float32)
                for cat_name, p in probs.items():
                    if cat_name in CATEGORIES:
                        soft_vec[CATEGORIES.index(cat_name)] = float(p)
                if np.sum(soft_vec) > 0:
                    soft_vec /= np.sum(soft_vec)
                dataset_with_teacher.append((prompt, soft_vec, nt, role, risk, delegate))
                print(f"   [{i+1}/{num_rollouts}] Jev Teacher Soft Target -> {res.get('choice')} (conf: {res.get('confidence', 0):.2f})")
                time.sleep(0.08)
                continue

        hard_vec = np.zeros(len(CATEGORIES), dtype=np.float32)
        hard_vec[CATEGORIES.index(hard_cat)] = 1.0
        dataset_with_teacher.append((prompt, hard_vec, nt, role, risk, delegate))

    print(f"\n📊 Prepared {len(dataset_with_teacher)} training samples.")

    # Prepare matrices
    X, y_cat, y_tools, y_role, y_risk, y_del = [], [], [], [], [], []
    role_map = {r: i for i, r in enumerate(ROLES)}
    risk_map = {k: i for i, k in enumerate(RISK_LEVELS)}

    for prompt, cat_dist, nt, role, risk, delegate in dataset_with_teacher:
        X.append(extract_features(prompt))
        y_cat.append(cat_dist)
        y_tools.append([nt])

        r_vec = np.zeros(len(ROLES), dtype=np.float32)
        r_vec[role_map[role]] = 1.0
        y_role.append(r_vec)

        k_vec = np.zeros(len(RISK_LEVELS), dtype=np.float32)
        k_vec[risk_map[risk]] = 1.0
        y_risk.append(k_vec)

        y_del.append([delegate])

    X = np.array(X, dtype=np.float32)
    y_cat = np.array(y_cat, dtype=np.float32)
    y_tools = np.array(y_tools, dtype=np.float32)
    y_role = np.array(y_role, dtype=np.float32)
    y_risk = np.array(y_risk, dtype=np.float32)
    y_del = np.array(y_del, dtype=np.float32)

    student = RLCDSmallNeuralModel(
        in_dim=FEATURE_SIZE,
        hidden_dim=64,
        num_cats=len(CATEGORIES),
        num_roles=len(ROLES),
        num_risks=len(RISK_LEVELS),
    )
    print("🚀 Training Micro Student Model with RLCD Brier Loss...")
    student.train_rlcd(X, y_cat, y_tools, y_role, y_risk, y_del, epochs=epochs, lr=0.008)
    student.calibrate_temperature(X, y_cat)
    print(f"🎯 Calibrated Micro Model Temperature: T={student.temperature:.3f}")

    weights_data = {
        "version": "micro-distilled-jev-v1",
        "vocabulary": VOCABULARY,
        "categories": CATEGORIES,
        "roles": ROLES,
        "risk_levels": RISK_LEVELS,
        "temperature": student.temperature,
        "W1": student.W1.tolist(),
        "b1": student.b1.tolist(),
        "W_cat": student.W_cat.tolist(),
        "b_cat": student.b_cat.tolist(),
        "W_tools": student.W_tools.tolist(),
        "b_tools": student.b_tools.tolist(),
        "W_role": student.W_role.tolist(),
        "b_role": student.b_role.tolist(),
        "W_risk": student.W_risk.tolist(),
        "b_risk": student.b_risk.tolist(),
        "W_delegate": student.W_delegate.tolist(),
        "b_delegate": student.b_delegate.tolist(),
    }
    go_path, json_path = export_to_rakitkode(weights_data)
    print(f"📦 Exported to RakitKode: {go_path}")


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JEV_KEY
    rolls = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_micro_distillation(api_key=key, num_rollouts=rolls)
