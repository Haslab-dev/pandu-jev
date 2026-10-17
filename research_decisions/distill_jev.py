"""Teacher-Student Knowledge Distillation from TypeSafe Jev to Pandu/RakitKode.

Queries the live TypeSafe Jev (Cloud RLCD) API as the Teacher, harvests soft
probability distributions across developer tasks, and distills that dark knowledge
into the ultra-fast in-process Student model using Hinton KD loss + Brier scoring.
"""

import json
import os
import sys
import time
from typing import Dict, List, Any, Optional
import requests
import numpy as np

from salience import extract_salient_context
from export_rakitkode import export_to_rakitkode
from train_rlcd import (
    CATEGORIES,
    ROLES,
    RISK_LEVELS,
    VOCABULARY,
    FEATURE_SIZE,
    RLCDSmallNeuralModel,
    extract_features,
    BASE_DATASET,
    softmax,
    sigmoid,
    relu,
)

DEFAULT_JEV_KEY = (
    os.getenv("TYPESAFE_API_KEY")
    or "apikey_2239cfdecc7e3aa44b668b44ff182a182d5d_0c968efac3285ccfa8427db1765c186c39b18ee372b07b5979befa19e45a317b"
)
DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def query_jev_teacher(
    prompt: str,
    api_key: str = DEFAULT_JEV_KEY,
    endpoint: str = DEFAULT_ENDPOINT,
) -> Optional[Dict[str, Any]]:
    """Query TypeSafe Jev API to obtain Teacher category probabilities and routing."""
    # Truncate prompt to safe bound for HTTP transport if excessively large
    salient_prompt, _ = extract_salient_context(prompt, head_chars=768, tail_chars=768)

    payload = {
        "state": f"User intent and context:\n{salient_prompt}",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Classify the user intent into the appropriate tool category",
                "criteria": {
                    "explore": "Search, find files, directory tree, grep",
                    "read": "Read, view, inspect, examine file contents",
                    "modify": "Edit, replace, refactor, write, fix bugs",
                    "execute": "Run bash, terminal, compile, build, test",
                    "git": "Git operations, commit, diff, branch, status",
                    "web": "Fetch URLs, browse web, HTTP requests",
                    "skills": "Skills management, workflows",
                    "mcp": "Model context protocol, database queries",
                    "none": "General conversation, questions, explanations",
                },
            }
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("answers", {}).get("category", {})
        else:
            print(f"⚠️ Jev API error {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        print(f"⚠️ Jev query failed: {e}")
    return None


def run_distillation_pipeline(
    api_key: str = DEFAULT_JEV_KEY,
    num_rollouts: int = 25,
    epochs: int = 350,
):
    print("=" * 70)
    print("🎓 TypeSafe Jev -> Pandu/RakitKode Knowledge Distillation Pipeline")
    print("=" * 70)

    print(f"📡 Querying Teacher Jev API for soft probability distributions ({num_rollouts} rollouts)...")
    dataset_with_teacher = []

    for i, item in enumerate(BASE_DATASET):
        prompt, hard_cat, nt, role, risk, delegate = item
        # Query Jev for first N samples or all
        if i < num_rollouts:
            teacher_res = query_jev_teacher(prompt, api_key=api_key)
            if teacher_res and "probabilities" in teacher_res:
                probs = teacher_res["probabilities"]
                soft_vec = np.zeros(len(CATEGORIES), dtype=np.float32)
                for cat_name, prob in probs.items():
                    if cat_name in CATEGORIES:
                        soft_vec[CATEGORIES.index(cat_name)] = float(prob)
                # Normalize just in case
                if np.sum(soft_vec) > 0:
                    soft_vec /= np.sum(soft_vec)
                dataset_with_teacher.append((prompt, soft_vec, nt, role, risk, delegate, True))
                print(f"   [{i+1}/{num_rollouts}] Teacher Soft Target for '{prompt[:45]}...' -> {teacher_res.get('choice')} (conf: {teacher_res.get('confidence', 0):.2f})")
                time.sleep(0.1)  # Respect rate limit
                continue

        # Hard label fallback if Jev query not executed or failed
        hard_vec = np.zeros(len(CATEGORIES), dtype=np.float32)
        hard_vec[CATEGORIES.index(hard_cat)] = 1.0
        dataset_with_teacher.append((prompt, hard_vec, nt, role, risk, delegate, False))

    print(f"\n📊 Distillation Dataset: {len(dataset_with_teacher)} samples prepared.")

    # Prepare tensors
    X, y_cat, y_tools, y_role, y_risk, y_del = [], [], [], [], [], []
    role_map = {r: i for i, r in enumerate(ROLES)}
    risk_map = {k: i for i, k in enumerate(RISK_LEVELS)}

    for prompt, cat_dist, nt, role, risk, delegate, is_soft in dataset_with_teacher:
        feats = extract_features(prompt)
        X.append(feats)
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

    # Train Student Model
    print("🚀 Training Distilled Student Model with Hinton KD + RLCD Brier Loss...")
    student = RLCDSmallNeuralModel(
        in_dim=FEATURE_SIZE,
        hidden_dim=64,
        num_cats=len(CATEGORIES),
        num_roles=len(ROLES),
        num_risks=len(RISK_LEVELS),
    )
    student.train_rlcd(X, y_cat, y_tools, y_role, y_risk, y_del, epochs=epochs, lr=0.007)
    student.calibrate_temperature(X, y_cat)

    # Export to RakitKode
    weights_data = {
        "version": "distilled-jev-v1",
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
    print("\n" + "=" * 70)
    print(f"🎉 Distillation complete!")
    print(f"📦 Pure Go Embed: {go_path}")
    print(f"📄 JSON Weights:  {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JEV_KEY
    rollouts = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    run_distillation_pipeline(api_key=key, num_rollouts=rollouts)
