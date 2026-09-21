# Pandu (pandu-jev) 🚀

> **Pandu (pandu-jev): Tiny, Local, Zero-API-Cost Policy Networks, Uncertainty Calibration, and Grounded Language Cortex**

Pandu explores how minimalist local neural network policies (~2.9K–5K parameters) can serve as ultra-fast, zero-cost motor reflexes in closed-loop embodied environments, leveraging calibrated uncertainty for hybrid fallback and grounding semantic representations from frozen foundation language models via the Canonical Intent Protocol.

---

## ⚡ Key Highlights

- **Ultra-Compact Footprint:**
  - **Pandu Core:** **2,916 parameters** (~11 KB) — zero external dependencies.
  - **Grounded Pandu Policy:** **4,980 parameters** (~19 KB) — grounds 16d environment state + 16d semantic intent.
- **Microsecond Reflex Latency:** **0.017 – 0.024 ms (17–24 microseconds)** policy evaluation on standard CPU/Apple Silicon.
- **Zero API Cost:** Runs 100% locally with zero external API calls or token billing.
- **Calibrated Uncertainty:** Expected Calibration Error (ECE) of **0.86%**; error detection AUROC of **0.952**.
- **Hybrid Fallback Advantage:** Matches **100% Teacher-grade success** while reducing average latency by **70.3%** and monetary cost by **69.1%**.
- **Canonical Intent Protocol:** Decouples cognitive language understanding from real-time control, ensuring complete invariance to linguistic variation across 5 OOD stress tiers.

---

## 📂 Project Structure

```text
pandu-jev/
├── env/                           # Universal closed-loop environments
│   ├── base.py                    # Abstract universal environment interface
│   ├── gridworld.py               # GridWorld with dynamic hazards and procedural maps
│   └── racing.py                  # 2D continuous vehicle dynamics environment
├── models/                        # Policy and adapter architectures
│   ├── policy.py                  # Canonical TinyPolicy (2.9K) and scalable MLP tiers
│   └── grounded_policy.py         # GroundedPanduPolicy (4.9K) & CanonicalIntentProtocol
├── expert/                        # Demonstration oracles
│   └── astar.py                   # Optimal A* pathfinding oracle
├── datasets/                      # Demonstration collection & serialization
│   ├── trajectory_dataset.py      # Core spatial expert trajectories
│   └── language_trajectory_dataset.py # 5-tier OOD language-grounded trajectories
├── training/                      # Training pipelines
│   ├── bc.py                      # Supervised Behavioral Cloning + temperature scaling
│   └── ppo.py                     # Reinforcement Learning (PPO + GAE actor-critic)
├── evaluation/                    # Calibration & uncertainty metrics
│   ├── calibration.py             # ECE, MCE, Brier score, reliability diagrams
│   ├── uncertainty.py             # OOD stress regimes (unseen, blocked, noisy)
│   └── stress_testing.py          # Environmental robustness suite (Versions A–E)
├── experiments/                   # Empirical research benchmark suites
│   ├── model_scaling.py           # Parameter scaling laws (3K, 50K, 1M, 5M)
│   ├── hybrid_fallback.py         # Teacher vs Pandu vs Hybrid fallback runtime
│   ├── grounded_language_experiment.py # 5-tier OOD grounded language benchmark
│   ├── test_modernbert_tiny.py    # Standalone ModernBERT-Tiny HF benchmark
│   ├── test_smollm2_135m.py       # Standalone SmolLM2-135M HF benchmark
│   ├── test_qwen3_0_6b.py         # Standalone Qwen3-0.6B HF benchmark
│   ├── run_all_tests.py           # Unified test runner
│   └── run_research.py            # Master empirical research pipeline
├── docs/                          # Research specifications & empirical papers
│   ├── PLAN.md                    # Research and prototype plan
│   ├── RESEARCH.md                # Publication-grade empirical research report
│   └── GROUNDED_LANGUAGE_ARCHITECTURE.md # Grounded Language Cortex specification
├── tests/                         # Automated Pytest suite
│   ├── test_all.py                # Core unit & integration tests
│   └── test_grounded_language.py  # Grounded policy, adapter & protocol tests
├── cli.py                         # Click/Rich interactive CLI interface
├── pandu-jev                      # Executable CLI wrapper
└── pyproject.toml                 # Packaging configuration
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Haslab-dev/pandu-jev.git
cd pandu-jev

# Create virtual environment and install editable package
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Interactive CLI Play

Experience the local policy navigating with live calibrated confidence and step visualization:

```bash
pandu-jev play gridworld
```
*(or `./pandu-jev play gridworld`)*

### 3. Run Benchmarks

```bash
# Run quick benchmark suite
pandu-jev benchmark --quick

# Run Grounded Language Cortex & Canonical Intent Protocol benchmark
pandu-jev test-language

# Run master empirical research suite (Phases 3-14)
python experiments/run_research.py
```

### 4. Run Automated Test Suite

```bash
# Run all 13 unit tests via PyTest
pytest tests/ -v

# Run unified test runner (Unit tests + HF models + Language cortex)
pandu-jev test-all
```

---

## 🧠 Architectural Paradigm

```text
        KNOWLEDGE (Cognitive Cortex)
                     │
            Foundation LM (Frozen)
            SmolLM2 / ModernBERT / Qwen3
                     │
                 semantics
                     │
                     ▼
        ┌─────────────────────────┐
        │ Pandu Intent Protocol   │  (Canonical Semantic Boundary)
        └────────────┬────────────┘
                     │
                     ▼
        CONTROL (Motor Reflex Policy)
        ┌─────────────────────────┐
        │          Pandu          │  (Trainable ~4,980 params)
        └────────────┬────────────┘
                     │
                fast action (<0.03 ms)
                     │
                     ▼
                   WORLD
```

> **"The LLM thinks about what is meant. Pandu decides what to do right now."**

---

## 📚 Publications & Technical Reports

- Comprehensive Empirical Research Paper: [docs/RESEARCH.md](file:///Users/hy4-mac-002/hasdev/research/mini-jev/docs/RESEARCH.md)
- Grounded Language Cortex Specification: [docs/GROUNDED_LANGUAGE_ARCHITECTURE.md](file:///Users/hy4-mac-002/hasdev/research/mini-jev/docs/GROUNDED_LANGUAGE_ARCHITECTURE.md)
- Development & Prototype Plan: [docs/PLAN.md](file:///Users/hy4-mac-002/hasdev/research/mini-jev/docs/PLAN.md)
