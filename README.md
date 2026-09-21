# pandu-jev 🚀

> **Mini-Jev: Tiny, Local, Zero-API-Cost Policy & Uncertainty Research Suite**

Mini-Jev explores how minimalist local neural network policies can serve as ultra-fast, zero-cost reflexes in closed-loop agent environments, using calibrated confidence to trigger selective fallbacks to expensive teacher models or LLMs.

---

## Key Highlights

- **Ultra-Compact Footprint:** Canonical policy contains **2,916 parameters** (~11 KB) — well under the 1M parameter target.
- **Microsecond Latency:** **0.029 ms (29 microseconds)** local inference time on standard CPU/MPS.
- **Zero API Cost:** Runs 100% locally with zero external API calls or token usage.
- **High Closed-Loop Success:** **92.0% success rate** on complex obstacle mazes.
- **Calibrated Uncertainty:** Expected Calibration Error (ECE) of **0.86%**; error detection AUROC of **0.952**.
- **Hybrid Fallback Advantage:** Matches **100% Teacher-grade success** while reducing average decision latency by **70.3%** and API/token monetary expenditure by **69.1%**.

---

## Project Structure

```text
mini-jev/
├── env/                     # Universal closed-loop environments
│   ├── base.py              # Abstract universal environment interface
│   ├── gridworld.py         # GridWorld with walls, procedural maps, dynamic hazards
│   └── racing.py            # 2D continuous vehicle dynamics racing environment
├── models/                  # Policy architectures
│   └── policy.py            # Canonical TinyPolicy (2.9K) and scalable MLP tiers
├── expert/                  # Expert teacher solvers
│   └── astar.py             # Optimal A* pathfinding oracle
├── datasets/                # Demonstration data collection & serialization
│   └── trajectory_dataset.py
├── training/                # Training pipelines
│   ├── bc.py                # Supervised Behavioral Cloning + temperature calibration
│   └── ppo.py               # Reinforcement Learning (PPO + GAE actor-critic)
├── evaluation/              # Calibration & uncertainty metrics
│   ├── calibration.py       # ECE, MCE, Brier score, reliability diagrams
│   ├── uncertainty.py       # 5 OOD stress regimes (unseen, large, blocked, noisy)
│   └── stress_testing.py   # Versions A through E environmental stress suite
├── experiments/             # Benchmarking suites
│   ├── model_scaling.py     # Parameter scaling laws (3K, 50K, 1M, 5M)
│   ├── hybrid_fallback.py   # Head-to-head Teacher vs Mini-Jev vs Hybrid runtime
│   └── run_research.py      # Automated master experiment runner
├── docs/
│   ├── PLAN.md              # Original research and prototype plan
│   └── RESEARCH.md          # Comprehensive empirical research publication report
├── tests/
│   └── test_all.py          # Pytest verification suite
├── cli.py                   # Click/Rich interactive CLI interface
└── pyproject.toml           # Packaging metadata
```

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mini-jev.git
cd mini-jev

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### 2. Interactive CLI Play

Run an interactive closed-loop session with live probability meters, calibrated confidence, and real-time step visualization:

```bash
mini-jev play gridworld
```

### 3. Run Benchmarks

```bash
# Run quick benchmark in terminal
mini-jev benchmark --quick

# Run master empirical research suite (Phases 3-14)
python experiments/run_research.py
```

### 4. Run Test Suite

```bash
pytest tests/ -v
```

---

## Research Documentation

Read the full publication-grade research report in [docs/RESEARCH.md](file:///Users/hy4-mac-002/hasdev/research/mini-jev/docs/RESEARCH.md).
