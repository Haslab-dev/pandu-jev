# Mini-Jev — Research & Prototype Plan

**Authors / Engineering:** Lutfi & ChatGPT  
**Date:** September 2026  
**Repository:** `pandu-jev`  
**Artifact Classification:** Plan documentation  


### Goal

Build a tiny, local, zero-API-cost **decision/policy model** that:

1. observes environment state,
2. outputs an action probability distribution,
3. acts in a closed-loop environment,
4. learns from expert demonstrations,
5. eventually improves through reinforcement learning,
6. exposes calibrated confidence/uncertainty.

The progression:

```text
GridWorld
   ↓
Tiny Policy
   ↓
Imitation Learning
   ↓
Probability / Confidence
   ↓
Reinforcement Learning
   ↓
More complex environment
   ↓
Language-conditioned agent
```

---

# Phase 0 — Define the experiment

**Target:** establish the architecture before touching ML.

Create:

```text
mini-jev/
├── env/
├── models/
├── training/
├── evaluation/
├── inference/
├── datasets/
├── experiments/
└── README.md
```

Define the universal interface:

```python
state = env.observe()

action_distribution = policy(state)

action = sample(action_distribution)

next_state, reward, done = env.step(action)
```

The critical abstraction is:

```text
State → Policy → Action → Environment → State
```

Everything afterward should preserve this interface.

---

# Phase 1 — GridWorld

**Target:** get a policy working with no LLM.

Environment:

```text
############
#S         #
#   ###    #
#       #  #
#    #     #
#       # G#
############
```

Actions:

```text
UP
DOWN
LEFT
RIGHT
```

State:

```python
{
    "agent_x": 3,
    "agent_y": 4,
    "goal_x": 10,
    "goal_y": 5,
    "walls": [...]
}
```

Reward:

```text
Goal       +100
Wall       -10
Step        -1
```

Deliverable:

```text
env/gridworld.py
```

Acceptance criteria:

> A deterministic agent can reliably reach the goal.

---

# Phase 2 — Build the tiny policy

Don't use an LLM.

Start with:

```text
Input
 ↓
Linear(32)
 ↓
GELU
 ↓
Linear(64)
 ↓
GELU
 ↓
Linear(4)
```

Output:

```text
UP       0.02
DOWN     0.01
LEFT     0.07
RIGHT    0.90
```

Model target:

**< 1M parameters initially.**

This is important.

We want to discover how little intelligence is actually necessary.

---

# Phase 3 — Generate an expert dataset

Implement an expert using A*.

```text
GridWorld
    ↓
A*
    ↓
optimal trajectory
```

Generate:

```text
10k–100k episodes
```

Each sample:

```json
{
  "state": "...",
  "action": "RIGHT"
}
```

Dataset:

```text
state₁ → RIGHT
state₂ → RIGHT
state₃ → UP
state₄ → UP
...
```

No LLM required.

---

# Phase 4 — Behavioral cloning

Train the tiny model using supervised learning.

```text
A* expert
     ↓
dataset
     ↓
CrossEntropyLoss
     ↓
Tiny Policy
```

Training loop:

```python
logits = model(state)

loss = cross_entropy(
    logits,
    expert_action
)

loss.backward()
optimizer.step()
```

Goal:

> Tiny model reproduces the expert.

Measure:

```text
Action accuracy
Episode success rate
Average steps
Inference latency
Model size
```

---

# Phase 5 — Make confidence a first-class output

Now expose:

```python
probs = softmax(logits)
```

Example:

```json
{
  "action": "RIGHT",
  "probabilities": {
    "UP": 0.02,
    "DOWN": 0.01,
    "LEFT": 0.07,
    "RIGHT": 0.90
  },
  "confidence": 0.90
}
```

Then test:

```text
confidence
     │
     ├── high → execute
     │
     └── low → fallback
```

Create an uncertainty benchmark.

Test the model on:

* unseen maps
* larger maps
* blocked paths
* noisy states
* impossible states

The question:

> Does low confidence actually correlate with failure?

That's much more important than raw accuracy.

---

# Phase 6 — Calibration

This is where the project starts becoming genuinely interesting.

Measure:

```text
Predicted confidence
        vs
Actual success probability
```

For example:

```text
Confidence     Actual success
0.9            0.91
0.8            0.79
0.7            0.71
0.6            0.59
```

That's reasonably calibrated.

But if:

```text
0.9 → 0.61 actual success
```

then your confidence is misleading.

Measure:

* Expected Calibration Error
* Brier score
* reliability diagram
* accuracy vs confidence

---

# Phase 7 — Reinforcement Learning

Now remove the expert.

The model interacts directly with the environment:

```text
Policy
  ↓
Action
  ↓
Environment
  ↓
Reward
  ↓
Policy update
```

Start with PPO.

Compare:

```text
Behavioral cloning
        VS
       PPO
```

Measure:

```text
episodes to convergence
success rate
average reward
sample efficiency
```

The interesting question:

> Can the tiny model improve beyond the original expert behavior?

---

# Phase 8 — Stress the model

Now make GridWorld increasingly difficult.

### Version A

Fixed maps.

### Version B

Random maps.

### Version C

Dynamic obstacles.

### Version D

Partial observability.

### Version E

Noisy observations.

Now you can test whether the model is actually learning a policy rather than memorizing coordinates.

---

# Phase 9 — Build a second environment

Don't jump directly to autonomous driving.

I'd use a simple **2D racing environment**.

State:

```python
{
    "speed": 0.72,
    "heading": 0.14,
    "track_offset": -0.08,
    "curvature": 0.31,
    "obstacle_distance": 0.84
}
```

Actions:

```text
STEER_LEFT
STEER_RIGHT
ACCELERATE
BRAKE
COAST
```

Now the model has to control a continuous dynamic system.

This is significantly more interesting than GridWorld.

---

# Phase 10 — Compare model sizes

Run the same environment with:

```text
1M
5M
10M
25M
50M
100M
```

Create a benchmark:

| Model      | Params | Accuracy | Success | Latency | Memory |
| ---------- | -----: | -------: | ------: | ------: | -----: |
| Policy-1M  |     1M |        — |       — |       — |      — |
| Policy-5M  |     5M |        — |       — |       — |      — |
| Policy-10M |    10M |        — |       — |       — |      — |
| Policy-25M |    25M |        — |       — |       — |      — |
| Policy-50M |    50M |        — |       — |       — |      — |

This gives you the answer to:

> **How much model do I actually need?**

---

# Phase 11 — Introduce language

Only now.

Add:

```text
Task:
"Drive to the finish while avoiding obstacles."
```

State becomes:

```text
Language
+
Environment State
        ↓
   State Encoder
        ↓
      Policy
        ↓
      Action
```

Now you can experiment with:

* text embeddings
* small language encoder
* multimodal input
* language-conditioned policies

This is where models such as **SmolLM2-135M** become relevant.

---

# Phase 12 — AgentWorld

This is the part I'd ultimately connect to your agent-runtime work.

Environment:

```text
┌─────────────────────────┐
│       AgentWorld        │
│                         │
│ Files                   │
│ Terminal                │
│ Tests                   │
│ Git                     │
│ Tools                   │
└────────────┬────────────┘
             │
           State
             │
             ▼
       Mini-Jev Policy
             │
             ▼
          Action
```

Actions:

```text
READ_FILE
EDIT_FILE
RUN_TEST
RUN_COMMAND
SEARCH
INSPECT_ERROR
FINISH
ASK_HUMAN
```

Now the model learns:

```text
state → action
```

for software-engineering tasks.

---

# Phase 13 — LLM teacher

Only here would I introduce a large LLM.

Architecture:

```text
             Agent State
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
     Large LLM          Mini-Jev
       Teacher             Student
          │                │
          └───────┬────────┘
                  ▼
              Compare
                  │
                  ▼
              Dataset
```

Generate expert trajectories:

```text
state
  ↓
LLM
  ↓
action
  ↓
environment
  ↓
result
```

Then distill them into Mini-Jev.

This is where you can potentially produce a genuinely useful specialized model.

---

# Phase 14 — Jev-like fallback architecture

Final runtime:

```text
                    Agent
                      │
                      ▼
                Mini-Jev
                      │
             ┌────────┴────────┐
             │                 │
       confidence ≥ 0.90   confidence < 0.90
             │                 │
             ▼                 ▼
          execute              LLM
             │                 │
             └────────┬────────┘
                      ▼
                   result
                      │
                      ▼
                   state
```

Now you can measure whether Mini-Jev actually provides value.

---

# The final benchmark

This is what I'd consider the project's real success criterion.

Compare:

### A

```text
LLM → action
```

### B

```text
Mini-Jev → action
```

### C

```text
Mini-Jev → uncertain → LLM
```

Measure:

```text
┌─────────────────────────────┐
│ Latency                     │
│ Cost                        │
│ Success rate                │
│ Tokens consumed             │
│ Actions / second            │
│ Failure rate                │
│ Calibration                 │
│ Fallback frequency          │
└─────────────────────────────┘
```

The interesting result would be something like:

```text
                 LLM       Mini-Jev     Hybrid
Latency          1.2s       8ms          90ms
Success          91%        84%          93%
LLM calls        1.0        0            0.12
Cost             $$$        $0           $
```

**That's the experiment that would tell us whether your Mini-Jev idea is actually useful.**

---

## Recommended first milestone

Don't build all 14 phases.

Build this:

```text
Week 1

GridWorld
   ↓
A* Expert
   ↓
10k trajectories
   ↓
1M parameter policy
   ↓
Behavioral cloning
   ↓
Probability output
   ↓
Calibration benchmark
```

At the end you should have a tiny local model that can literally do:

```bash
$ mini-jev play gridworld

step 01
  UP      0.01
  DOWN    0.02
  LEFT    0.03
  RIGHT   0.94

  → RIGHT

step 02
  UP      0.02
  DOWN    0.01
  LEFT    0.04
  RIGHT   0.93

  → RIGHT

...

GOAL
reward: 94
```

**No API. No GPU requirement. No LLM. No token cost.**

Then we have a scientific baseline to build on rather than blindly trying to reproduce TypeSafe's architecture.


### Model References
| Model                       |     Size | Best use                          |
| --------------------------- | -------: | --------------------------------- |
| **ModernBERT-Tiny / Mini**  |  ~20–50M | State/text classification         |
| **SmolLM2-135M**            |     135M | Language-heavy state → action     |
| **SmolLM2-360M**            |     360M | More capable language policy      |
| **Qwen3-0.6B**              |    ~600M | More complex textual environments |
| **MobileBERT / DistilBERT** | ~60–135M | Extremely cheap classifier        |
| **Custom MLP/Transformer**  |    1–50M | Best if state is structured       |
