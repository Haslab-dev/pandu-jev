# Pandu Decision Model Research Workspace

Research, distillation, and experimentation lab for zero-token typed decision models inspired by TypeSafe Jev (Cloud RLCD).

---

## 🏛️ Model Tiers Overview

The workspace is organized into three distinct capacity and operational tiers:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                PANDU DECISION MODEL TIERS                                   │
├─────────────────────────┬───────────────────────────┬───────────────────────────────────────┤
│    TIER 0: MICRO        │       TIER 1: TINY        │            TIER 2: BASE               │
│   (Pure Math Reflex)    │   (MicroBERT / Mini-Jev)  │        (Deep Semantic Cortex)         │
├─────────────────────────┼───────────────────────────┼───────────────────────────────────────┤
│ • Parameters: ~50K      │ • Parameters: ~1.5M       │ • Parameters: ~149M                   │
│ • Footprint: < 100 KB   │ • Footprint: ~1.8 MB INT8 │ • Footprint: ~300 MB - 570 MB         │
│ • Engine: Pure NumPy    │ • Engine: PyTorch / MPS   │ • Engine: PyTorch / CoreML ANE        │
│ • Latency: ~0.04 ms     │ • Latency: ~0.8 ms        │ • Latency: ~23 ms                     │
│ • Zero Dependencies     │ • Pruned Vocab (2,048)    │ • Full Pre-trained Transformer        │
│ • In-Binary Embed       │ • Full Attention (Negation)│ • Architecture Proofs & Vuln Triage  │
└─────────────────────────┴───────────────────────────┴───────────────────────────────────────┘
```

---

## 📁 Directory Structure

```text
research_decisions/
├── datasets/
│   ├── salience.py           # Context salience, zoning, and token compaction
│   └── rakitkode_domain.py   # Developer task prompts, negation tests, security hazards
├── teacher/
│   └── jev_teacher.py        # Live TypeSafe Jev API client (model: jev-latest)
├── micro/                    # [Tier 0] <100 KB Pure Math Reflex
│   ├── model.py              # 5-head neural model with analytical Brier scoring
│   ├── distill_micro.py      # Distillation harness querying Jev as Teacher
│   └── export_rakitkode.py   # Code generator exporting pure Go byte arrays
├── tiny/                     # [Tier 1] ~1.5M Params / ~1.8 MB INT8 MicroBERT
│   ├── tokenizer.py          # Domain-specific compact tokenizer (2,048 tokens)
│   ├── microbert.py          # 3-layer Transformer Encoder (d=192, 4 heads)
│   └── distill_microbert.py  # Hinton KD distillation pipeline from Jev into MicroBERT
└── base/                     # [Tier 2] 149M ModernBERT-Base
    └── base_router.py        # High-level wrapper for deep reasoning & security triage
```

---

## 🚀 Running the Experiments

### 1. Tier 0: Micro Model (<100 KB, ~50 µs)
Distill from TypeSafe Jev into the pure math micro model and export to Go:
```bash
python3 -m micro.distill_micro [JEV_API_KEY] [ROLLOUTS]
```

### 2. Tier 1: MicroBERT (~1.8 MB INT8, ~0.8 ms)
Distill from TypeSafe Jev into MicroBERT with full self-attention and negation resolution:
```bash
.venv/bin/python -m tiny.distill_microbert [JEV_API_KEY] [ROLLOUTS]
```

### 3. Tier 2: Base Model (149M ModernBERT-Base)
Evaluate deep reasoning and security vulnerability triage:
```bash
.venv/bin/python -m base.base_router
```
