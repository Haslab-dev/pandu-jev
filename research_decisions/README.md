# Pandu-Jev Research Decisions Workspace 🧠

> **The dedicated research, distillation, and training laboratory for lightweight agent decision models.**

This workspace develops and distills intelligent decision models that power **RakitKode**, **Forge-ADE**, and **Kendali-AI**. It bridges modern AI research (TypeSafe Jev RLCD, ModernBERT, Knowledge Distillation) with high-performance in-process native execution ($< 40\ \mu\text{s}$ Go runtime).

---

## 📂 Architecture & Components

```
pandu-jev/research_decisions/
├── salience.py           # Context Zoning & Salience extraction (handles 50k+ char code/logs)
├── train_rlcd.py         # Multi-Head RLCD Trainer (5 heads, Brier score, temperature scaling)
├── distill_jev.py        # Knowledge Distillation engine querying live TypeSafe Jev API
├── export_rakitkode.py   # Code generator producing pure Go source & JSON weights
└── README.md             # Research documentation & instructions
```

---

## ⚡ Key Capabilities

1. **Multi-Head Decision Outputs**:
   - `Category` (9 classes): `explore`, `read`, `modify`, `execute`, `git`, `web`, `skills`, `mcp`, `none`
   - `NeedsTools` (Probability): Whether turn requires tool invocation
   - `Role` (5 classes): `coder`, `researcher`, `reviewer`, `debugger`, `devops`
   - `SafetyRisk` (3 classes): `safe`, `medium`, `critical`
   - `SubagentDelegation` (Probability): Whether task warrants subagent fork
2. **Context Salience & Large Input Handling**:
   - Compresses fenced code blocks (` ```go ... ``` `) and stack traces to structural markers.
   - Extracts Head (prefix command) and Tail (suffix question) instruction windows.
   - Applies sub-linear term frequency dampening ($1 + \ln(1 + \text{count})$) to prevent code dumps from overwhelming prompt intent.
3. **Proper Scoring Calibration (RLCD)**:
   - Analytical Brier score loss + Cross-Entropy.
   - Post-training temperature scaling ($T \approx 0.800$, ECE $< 0.001$).
4. **Knowledge Distillation from Live Jev**:
   - Queries TypeSafe Jev API as the Teacher.
   - Transfers soft probability distributions (dark knowledge) into the Student model.
5. **Direct Pure Go Compilation**:
   - One-command export to `rakitkode/decisions/weights_generated.go`.
   - Results in a zero-dependency, microsecond-latency embedded model.

---

## 🚀 Usage

### 1. Train Base RLCD Model
```bash
python3 train_rlcd.py
```

### 2. Distill from Live TypeSafe Jev
```bash
python3 distill_jev.py [OPTIONAL_API_KEY] [NUM_ROLLOUTS]
```

### 3. Verify in RakitKode
```bash
cd ../rakitkode
go test -v ./decisions
```
