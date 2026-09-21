# Pandu (pandu-jev) Kanban

**Project:** Pandu (pandu-jev) — Tiny Local Zero-Cost Policy Model & Uncertainty Research
**Goal:** Build a tiny, local, zero-API-cost decision/policy model that acts in closed-loop environments, learns from expert demonstrations and RL, exposes calibrated confidence, and benchmarks hybrid fallback against expensive teacher models.

**References:**
- [PLAN.md](file:///Users/hy4-mac-002/hasdev/research/mini-jev/docs/PLAN.md)
- [RESEARCH.md](file:///Users/hy4-mac-002/hasdev/research/mini-jev/docs/RESEARCH.md)
- [README.md](file:///Users/hy4-mac-002/hasdev/research/mini-jev/README.md)

---

## Kanban Board Overview

```mermaid
kanban
  Backlog
  To Do
  In Progress
  Blocked
  Done
    [ENV-001] Universal Interface & Scaffold
    [ENV-002] GridWorld Environment Implementation
    [ENV-003] 2D Continuous Racing Environment
    [EXP-001] A* Expert Policy and Dataset Generator
    [MOD-001] Tiny Policy Networks Architecture
    [TRN-001] Behavioral Cloning Training Pipeline
    [CAL-001] Calibration and Uncertainty Evaluation
    [RL-001] Reinforcement Learning PPO Policy Gradient
    [STR-001] Stress Testing Suite
    [EXP-002] Model Parameter Scaling Benchmark
    [HYB-001] Hybrid Fallback Runtime Benchmark
    [CLI-001] Interactive CLI pandu-jev play gridworld
    [RES-001] Research Report and Empirical Study
    [EXP-003] HuggingFace Language Policy Tests
    [LNG-001] Grounded Language Cortex Architecture & Experiments
    [REF-001] Codebase Architecture Polish & Packaging Clean-up
```

---

## Detailed Task Columns

### 📋 Backlog

| Task ID | Work Item | Scope / Files Affected | Priority | Dependencies / Notes |
| :------ | :-------- | :--------------------- | :------: | :------------------- |

### 📝 To Do

| Task ID | Work Item | Scope / Files Affected | Priority | Dependencies / Notes |
| :------ | :-------- | :--------------------- | :------: | :------------------- |

### 🚧 In Progress

| Task ID | Work Item | Scope / Files Affected | Owner | Status |
| :------ | :-------- | :--------------------- | :---- | :----- |

### 🚫 Blocked

| Task ID | Work Item | Blocker | Required Action | Status |
| :------ | :-------- | :------ | :-------------- | :----: |

### ✅ Done

| Task ID | Work Item | Completed Scope | Evidence |
| :------ | :-------- | :-------------- | :------- |
| **ENV-001** | Universal Interface & Architecture Scaffold | `env/base.py`, directory structure | Abstract Environment class with `observe()`, `step()`, `get_feature_vector()` implemented; all tests pass |
| **ENV-002** | GridWorld Environment Implementation | `env/gridworld.py`, `tests/test_all.py` | 16-dim normalized spatial features, dynamic obstacles, procedural generator verified; 8/8 pytest pass |
| **ENV-003** | 2D Continuous Dynamics Racing Environment | `env/racing.py`, `tests/test_all.py` | Continuous Newtonian vehicle dynamics (speed, heading, track offset, curvature) verified |
| **EXP-001** | A* Expert Policy and Dataset Generator | `expert/astar.py`, `datasets/trajectory_dataset.py` | A* pathfinding achieved 100% oracle success rate; collected 23,271 transitions across 2,500 episodes |
| **MOD-001** | Tiny Policy Networks Architecture (<1M params) | `models/policy.py` | Canonical 2,916-parameter TinyPolicy implemented and verified; scalable tiers (3K, 50K, 1M, 5M) built |
| **TRN-001** | Behavioral Cloning Training Pipeline | `training/bc.py` | Supervised training achieved 90.4% holdout val acc; 92.0% closed-loop success in rollout at 0.029 ms latency |
| **CAL-001** | Calibration and Uncertainty Evaluation | `evaluation/calibration.py`, `evaluation/uncertainty.py` | Uncalibrated ECE 0.86%, calibrated ECE 1.46%; error detection AUROC 0.952 (in-dist) and 1.000 (blocked paths) |
| **RL-001** | Reinforcement Learning PPO Pipeline | `training/ppo.py` | PPO with GAE actor-critic converged to 62.0% success rate without expert demonstrations |
| **STR-001** | Stress Testing Suite (partial obs, noisy, dynamic) | `evaluation/stress_testing.py` | Verified across Versions A-E; Version A (100%), Version C (88%), Version E (100%); Version D drops conf to 0.648 |
| **EXP-002** | Model Parameter Scaling Benchmark | `experiments/model_scaling.py` | Benchmarked 3K, 50K, 1M, 5M params: Policy-3K (90% succ, 0.024ms) matches Policy-5M (92.5% succ, 0.849ms) |
| **HYB-001** | Hybrid Fallback Runtime Benchmark | `experiments/hybrid_fallback.py` | Hybrid fallback (τ=0.85) matched 100.0% Teacher success while cutting latency by 70.3% and cost by 69.1% |
| **CLI-001** | Interactive CLI `pandu-jev play gridworld` | `cli.py`, `pyproject.toml` | Full CLI working with commands `play`, `train`, `benchmark`, `ppo`; real-time step visualization verified |
| **RES-001** | Research Report & Empirical Documentation | `docs/RESEARCH.md` | Comprehensive publication-grade empirical research paper written with full benchmark tables and figures |
| **EXP-003** | HuggingFace Language Policy Tests | `experiments/test_modernbert_tiny.py`, `experiments/test_smollm2_135m.py`, `experiments/test_qwen3_0_6b.py` | Tested 3 HF architectures: ModernBERT (19.3M, 52.9ms), SmolLM2 (134.5M, 51.5ms), Qwen3-0.6B (596.0M, 87-133ms) |
| **LNG-001** | Grounded Language Cortex & Semantic Latent Protocol | `models/grounded_policy.py`, `datasets/language_trajectory_dataset.py`, `experiments/grounded_language_experiment.py`, `docs/GROUNDED_LANGUAGE_ARCHITECTURE.md` | Benchmarked 4 modes: Pandu Core (83.8%, 0.024ms), ModernBERT+Pandu (85.1%, 84.1% OOD), SmolLM2+Pandu (85.6%, 84.8% OOD), Structured Intent+Pandu (84.8%, 0.020ms) |
| **REF-001** | Codebase Architecture Polish & Packaging Clean-up | `models/__init__.py`, `datasets/__init__.py`, `expert/__init__.py`, `training/__init__.py`, `evaluation/__init__.py`, `experiments/__init__.py`, `cli.py`, `README.md`, `.gitignore` | Standardized modular packages with clean exports, removed legacy egg-info, added CLI test-language/test-all, updated README & comprehensive .gitignore |

---

## Verification & Quality Gates

- [x] All relevant test suites pass (`pytest tests/ -v`: 8 passed in 6.33s)
- [x] Zero API cost; runs entirely locally on CPU/MPS (29 microseconds per step)
- [x] Universal interface `State -> Policy -> Action -> Environment -> State` strictly maintained
- [x] Expected Calibration Error (ECE: 0.86%), Brier score (0.064), and accuracy metrics computed and reported
- [x] CLI runs out-of-the-box (`pandu-jev play gridworld`, `pandu-jev benchmark`, `mini-jev play gridworld`)
- [x] Factual evidence documented for completed tasks
