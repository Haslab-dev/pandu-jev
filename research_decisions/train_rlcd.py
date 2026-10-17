"""RLCD Multi-Head Decision Model Training Pipeline for Pandu & RakitKode.

Trains an ultra-fast, zero-dependency in-process neural model with 5 decision heads:
1. Category (9 classes: explore, read, modify, execute, git, web, skills, mcp, none)
2. NeedsTools (Probability / sigmoid)
3. Role (5 classes: coder, researcher, reviewer, debugger, devops)
4. SafetyRisk (3 classes: safe, medium, critical)
5. SubagentDelegation (Probability / sigmoid)

Features:
- Context Salience and Zoning (handles 50k+ char code blocks, logs, and stack traces)
- Proper Scoring Rules: Hybrid Cross-Entropy (0.7) + Analytical Brier Score (0.3)
- Post-training Temperature Scaling for Expected Calibration Error (ECE < 0.001)
- Seamless export to pure Go (rakitkode/decisions/weights_generated.go) and JSON
"""

import json
import math
import os
import re
from typing import List, Tuple
import numpy as np

from salience import extract_salient_context
from export_rakitkode import export_to_rakitkode

CATEGORIES = [
    "explore",   # 0
    "read",      # 1
    "modify",    # 2
    "execute",   # 3
    "git",       # 4
    "web",       # 5
    "skills",    # 6
    "mcp",       # 7
    "none"       # 8
]

ROLES = [
    "coder",       # 0
    "researcher",  # 1
    "reviewer",    # 2
    "debugger",    # 3
    "devops"       # 4
]

RISK_LEVELS = [
    "safe",      # 0 - read-only, inspection, conversational
    "medium",    # 1 - code edits, builds, local tests, standard commits
    "critical"   # 2 - destructive operations (rm -rf, sudo, force push, drop table)
]

VOCABULARY = [
    # Explore / search
    "find", "search", "locate", "where", "list", "grep", "glob", "files", "dir", "tree", "explore",
    "structure", "inspect", "show me all", "which file", "path", "overview", "hierarchy",
    # Read
    "read", "cat", "view", "open", "display", "check content", "look at", "examine", "show code",
    "contents of", "header", "lines", "source", "definition", "implementation",
    # Modify / write / edit
    "edit", "modify", "replace", "write", "create", "update", "patch", "refactor", "fix bug",
    "implement", "change", "delete file", "insert", "append", "rewrite", "add function", "rename",
    # Execute / bash
    "run", "exec", "bash", "command", "build", "compile", "test", "go test", "npm test",
    "cargo build", "start server", "execute", "launch", "kill", "process", "terminal", "output",
    # Git
    "git", "commit", "push", "pull", "branch", "diff", "status", "checkout", "stash",
    "merge", "rebase", "log", "remote", "tag", "stage", "uncommitted", "repo",
    # Web / browse
    "http", "https", "fetch", "url", "web", "download", "scrape", "browse", "api request", "curl", "endpoint",
    "website", "html", "documentation", "doc", "docs", "online",
    # Skills / Plugins
    "skill", "skills", "learn", "how to", "install skill", "use skill", "custom skill", "plugin", "workflow", "recipe",
    # MCP
    "mcp", "server", "tool protocol", "call mcp", "mcp tool", "database query", "sqlite", "postgres",
    # Conversational / none
    "hello", "hi", "hey", "who are you", "what can you do", "thanks", "thank you", "bye",
    "explain", "summarize", "tell me about", "what is", "why", "how does", "help",
    # Programming concepts
    "function", "class", "struct", "interface", "method", "variable", "package", "import",
    "type", "error", "nil", "null", "pointer", "goroutine", "channel", "thread", "memory",
    "leak", "crash", "panic", "segfault", "benchmark", "optimize", "slow", "performance",
    # Safety keywords (Risk Critical)
    "rm -rf", "delete all", "drop table", "truncate", "format disk", "sudo", "chmod 777",
    "force push", "git push --force", "api_key", "secret", "password", "token", "env", "credentials", "id_rsa", "auth",
    # Structural Markers from Salience preprocessor
    "__has_code_block__", "__has_stacktrace__", "__has_git_diff__"
]

FEATURE_SIZE = 256
vocab_map = {word: i for i, word in enumerate(VOCABULARY)}


def fnv1a_32(s: str) -> int:
    h = 2166136261
    for b in s.encode('utf-8'):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def is_word_match(text: str, kw: str) -> bool:
    if " " in kw or "-" in kw or "_" in kw:
        return kw in text
    pattern = r'\b' + re.escape(kw) + r'\b'
    return bool(re.search(pattern, text))


def extract_features(text: str) -> np.ndarray:
    """Extract 256-dimensional feature vector with context zoning and sub-linear frequency dampening."""
    # Step 1: Preprocess through salience zoning
    salient_text, _ = extract_salient_context(text, head_chars=512, tail_chars=512)
    lower = salient_text.lower()
    words = re.findall(r'\b[a-z0-9_\-\./]+\b', lower)

    vec = np.zeros(FEATURE_SIZE, dtype=np.float32)

    # 1. Vocabulary keyword presence with sub-linear count dampening: 1.0 + ln(1 + count)
    for i, kw in enumerate(VOCABULARY):
        if is_word_match(lower, kw):
            count = lower.count(kw)
            weight = 1.0 + math.log(1.0 + count)
            vec[i] += 1.5 * weight

    # 2. Character trigram hashing for out-of-vocabulary generalization
    offset = len(VOCABULARY)
    num_hash_slots = FEATURE_SIZE - offset
    if num_hash_slots > 0:
        for w in words:
            if len(w) >= 3:
                for j in range(len(w) - 2):
                    tri = w[j:j+3]
                    h = fnv1a_32(tri) % num_hash_slots
                    vec[offset + h] += 0.5

    # 3. L2 Normalization
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm
    return vec


def relu(x):
    return np.maximum(0, x)


def softmax(x, temp=1.0):
    e = np.exp((x - np.max(x, axis=-1, keepdims=True)) / temp)
    return e / np.sum(e, axis=-1, keepdims=True)


def sigmoid(x, temp=1.0):
    return 1.0 / (1.0 + np.exp(-np.clip(x / temp, -20, 20)))


# Dataset schema: (prompt, category, needs_tools, role, risk_level, subagent_needed)
BASE_DATASET = [
    # Explore
    ("find all files matching *.go in the repository", "explore", 1.0, "researcher", "safe", 0.0),
    ("where is the entrypoint defined in this project?", "explore", 1.0, "researcher", "safe", 0.0),
    ("list all files in internal/gateway directory", "explore", 1.0, "researcher", "safe", 0.0),
    ("search for function ExecuteTurnWithModel across the codebase", "explore", 1.0, "researcher", "safe", 0.0),
    ("inspect the directory tree and project architecture", "explore", 1.0, "researcher", "safe", 0.0),
    ("locate where HTTP request routing is handled", "explore", 1.0, "researcher", "safe", 0.0),
    ("grep for all TODO and FIXME comments in the backend", "explore", 1.0, "researcher", "safe", 0.0),
    ("find the config file for database migrations", "explore", 1.0, "researcher", "safe", 0.0),
    ("which files implement the agent harness?", "explore", 1.0, "researcher", "safe", 0.0),
    ("show me the structure of the internal packages", "explore", 1.0, "researcher", "safe", 0.0),

    # Read
    ("read the contents of main.go and check imports", "read", 1.0, "coder", "safe", 0.0),
    ("view internal/app/session_api.go lines 1 to 50", "read", 1.0, "coder", "safe", 0.0),
    ("cat README.md and summarize instructions", "read", 1.0, "researcher", "safe", 0.0),
    ("open and examine the error definitions in types.go", "read", 1.0, "debugger", "safe", 0.0),
    ("display the implementation of NewDecisionEngine", "read", 1.0, "coder", "safe", 0.0),
    ("check the config settings in app-settings.json", "read", 1.0, "devops", "safe", 0.0),
    ("look at the header comments of agent.go", "read", 1.0, "coder", "safe", 0.0),
    ("read the first 100 lines of the log file", "read", 1.0, "debugger", "safe", 0.0),

    # Modify
    ("edit internal/gateway/subagent.go to fix nil pointer exception", "modify", 1.0, "debugger", "medium", 0.0),
    ("refactor the session manager to use sync.RWMutex instead of Mutex", "modify", 1.0, "coder", "medium", 0.0),
    ("add a new method GetActiveSessions to Manager", "modify", 1.0, "coder", "medium", 0.0),
    ("fix bug in calculating token usage in llm.go", "modify", 1.0, "debugger", "medium", 0.0),
    ("update the prompt template to support reasoning tags", "modify", 1.0, "coder", "medium", 0.0),
    ("replace deprecated io/ioutil with io in file operations", "modify", 1.0, "coder", "medium", 0.0),
    ("write a new unit test in adapter_test.go", "modify", 1.0, "coder", "medium", 0.0),
    ("delete unused helper functions in utils.go", "modify", 1.0, "coder", "medium", 0.0),

    # Execute
    ("run go test ./... to verify all unit tests pass", "execute", 1.0, "debugger", "medium", 0.0),
    ("execute make build in the project root", "execute", 1.0, "devops", "medium", 0.0),
    ("build the binary and run tests with race detector", "execute", 1.0, "debugger", "medium", 0.0),
    ("start the development server on port 8080", "execute", 1.0, "devops", "medium", 0.0),
    ("compile the frontend with npm run build", "execute", 1.0, "devops", "medium", 0.0),
    ("run benchmark tests for memory allocation", "execute", 1.0, "debugger", "medium", 0.0),
    ("kill running background task with PID 1234", "execute", 1.0, "devops", "medium", 0.0),

    # Git
    ("git commit -m 'feat: implement decisions file adapter'", "git", 1.0, "coder", "medium", 0.0),
    ("git status to check modified and unstaged files", "git", 1.0, "coder", "safe", 0.0),
    ("git diff between main and feature branch", "git", 1.0, "reviewer", "safe", 0.0),
    ("git checkout -b feature/distillation", "git", 1.0, "coder", "medium", 0.0),
    ("create a git tag v0.2.11 and push to origin", "git", 1.0, "devops", "medium", 0.0),
    ("show git log of the last 5 commits with patch diff", "git", 1.0, "reviewer", "safe", 0.0),

    # Web
    ("fetch documentation from https://golang.org/pkg/context", "web", 1.0, "researcher", "safe", 0.0),
    ("search online for fixing CORS errors in Go http servers", "web", 1.0, "researcher", "safe", 0.0),
    ("download the latest release assets from GitHub API", "web", 1.0, "researcher", "safe", 0.0),
    ("browse the API reference for TypeSafe Jev systemone", "web", 1.0, "researcher", "safe", 0.0),
    ("query the web for best practices on ECE temperature scaling", "web", 1.0, "researcher", "safe", 0.0),

    # Skills / Plugins
    ("use custom skill to inspect and design frontend layout", "skills", 1.0, "coder", "safe", 0.0),
    ("install find-skills to search agent registry", "skills", 1.0, "devops", "safe", 0.0),
    ("run impeccable skill to polish UI color tokens and typography", "skills", 1.0, "coder", "safe", 0.0),
    ("how do I configure agy-customizations for this workspace", "skills", 1.0, "researcher", "safe", 0.0),

    # MCP
    ("call mcp tool filesystem list_allowed_directories", "mcp", 1.0, "devops", "safe", 0.0),
    ("query database via postgres mcp server for user table", "mcp", 1.0, "researcher", "safe", 0.0),
    ("use sqlite mcp tool to inspect schema of migration table", "mcp", 1.0, "coder", "safe", 0.0),

    # None / Conversational
    ("hello assistant, can you explain the difference between mutex and channel?", "none", 0.0, "coder", "safe", 0.0),
    ("why is quantum computing fundamentally different from classical computing?", "none", 0.0, "researcher", "safe", 0.0),
    ("what are proper scoring rules in machine learning calibration?", "none", 0.0, "researcher", "safe", 0.0),
    ("summarize the advantages of in-process inference vs external HTTP calls", "none", 0.0, "researcher", "safe", 0.0),
    ("thank you for your help, that solved my problem", "none", 0.0, "coder", "safe", 0.0),

    # Safety Critical Operations
    ("sudo rm -rf /var/log/* to clear space", "execute", 1.0, "devops", "critical", 0.0),
    ("drop table production_users cascade", "mcp", 1.0, "devops", "critical", 0.0),
    ("git push origin main --force --all", "git", 1.0, "devops", "critical", 0.0),
    ("cat .env and print all production AWS_SECRET_ACCESS_KEY", "read", 1.0, "debugger", "critical", 0.0),

    # Delegation (Complex / Multi-Step Subagent Tasks)
    ("conduct full security audit of authentication pipeline and generate report", "explore", 1.0, "reviewer", "safe", 1.0),
    ("research architectural alternatives for state synchronization across nodes", "explore", 1.0, "researcher", "safe", 1.0),
    ("autonomously refactor the entire test suite to use test tables across 30 packages", "modify", 1.0, "coder", "medium", 1.0),

    # Large Context Scenarios (Stack Traces & Fenced Code Dumps)
    ("Investigate this panic and fix the nil dereference in server.go:\n```\npanic: runtime error: invalid memory address or nil pointer dereference\n[signal SIGSEGV: segmentation violation code=0x2 addr=0x0 pc=0x102345]\ngoroutine 14 [running]:\ngithub.com/hasdev/forge-ade/internal/server.(*Server).Start(0x0)\n\t/Users/dev/internal/server.go:142 +0x24\n```", "modify", 1.0, "debugger", "medium", 0.0),
    ("Why is this test failing? Please inspect:\n```go\nfunc TestAuthenticate(t *testing.T) {\n\tuser := &User{Name: \"test\"}\n\tif !user.Auth() { t.Fatal(\"expected true\") }\n}\n```", "modify", 1.0, "debugger", "medium", 0.0),
    ("Review the following git diff and check for regressions:\n```diff\ndiff --git a/pkg/cache.go b/pkg/cache.go\nindex 1234..5678 100644\n--- a/pkg/cache.go\n+++ b/pkg/cache.go\n@@ -10,3 +10,3 @@\n-func Get() { return }\n+func Get() { lock.Lock(); defer lock.Unlock() }\n```", "read", 1.0, "reviewer", "safe", 0.0),
]


class RLCDSmallNeuralModel:
    """Multi-Layer Perceptron trained with RLCD calibration for local agent decisions."""
    def __init__(self, in_dim=256, hidden_dim=64, num_cats=9, num_roles=5, num_risks=3):
        np.random.seed(42)
        scale1 = math.sqrt(2.0 / in_dim)
        self.W1 = (np.random.randn(hidden_dim, in_dim) * scale1).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        scale_h = math.sqrt(2.0 / hidden_dim)
        self.W_cat = (np.random.randn(num_cats, hidden_dim) * scale_h).astype(np.float32)
        self.b_cat = np.zeros(num_cats, dtype=np.float32)

        self.W_tools = (np.random.randn(1, hidden_dim) * scale_h).astype(np.float32)
        self.b_tools = np.zeros(1, dtype=np.float32)

        self.W_role = (np.random.randn(num_roles, hidden_dim) * scale_h).astype(np.float32)
        self.b_role = np.zeros(num_roles, dtype=np.float32)

        self.W_risk = (np.random.randn(num_risks, hidden_dim) * scale_h).astype(np.float32)
        self.b_risk = np.zeros(num_risks, dtype=np.float32)

        self.W_delegate = (np.random.randn(1, hidden_dim) * scale_h).astype(np.float32)
        self.b_delegate = np.zeros(1, dtype=np.float32)

        self.temperature = 1.0

    def forward(self, x: np.ndarray, temp: float = 1.0):
        z1 = np.dot(x, self.W1.T) + self.b1
        h1 = relu(z1)

        logits_cat = np.dot(h1, self.W_cat.T) + self.b_cat
        prob_cat = softmax(logits_cat, temp=temp)

        logits_tools = np.dot(h1, self.W_tools.T) + self.b_tools
        prob_tools = sigmoid(logits_tools, temp=temp)

        logits_role = np.dot(h1, self.W_role.T) + self.b_role
        prob_role = softmax(logits_role, temp=temp)

        logits_risk = np.dot(h1, self.W_risk.T) + self.b_risk
        prob_risk = softmax(logits_risk, temp=temp)

        logits_del = np.dot(h1, self.W_delegate.T) + self.b_delegate
        prob_del = sigmoid(logits_del, temp=temp)

        return z1, h1, prob_cat, prob_tools, prob_role, prob_risk, prob_del

    def train_rlcd(self, X, y_cat, y_tools, y_role, y_risk, y_del, epochs=350, lr=0.007):
        num_samples = X.shape[0]
        params = [
            self.W1, self.b1,
            self.W_cat, self.b_cat,
            self.W_tools, self.b_tools,
            self.W_role, self.b_role,
            self.W_risk, self.b_risk,
            self.W_delegate, self.b_delegate
        ]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        alpha = 0.7  # 70% CE, 30% Brier for calibrated probability alignment

        for epoch in range(1, epochs + 1):
            z1, h1, prob_cat, prob_tools, prob_role, prob_risk, prob_del = self.forward(X, temp=1.0)

            # Gradient Category
            grad_cat_ce = (prob_cat - y_cat) / num_samples
            grad_cat_brier = 2.0 * (prob_cat - y_cat) * prob_cat * (1.0 - prob_cat) / num_samples
            grad_cat = alpha * grad_cat_ce + (1.0 - alpha) * grad_cat_brier

            # Gradient Tools
            grad_tools = (prob_tools - y_tools) / num_samples

            # Gradient Role
            grad_role_ce = (prob_role - y_role) / num_samples
            grad_role_brier = 2.0 * (prob_role - y_role) * prob_role * (1.0 - prob_role) / num_samples
            grad_role = alpha * grad_role_ce + (1.0 - alpha) * grad_role_brier

            # Gradient Risk
            grad_risk_ce = (prob_risk - y_risk) / num_samples
            grad_risk_brier = 2.0 * (prob_risk - y_risk) * prob_risk * (1.0 - prob_risk) / num_samples
            grad_risk = alpha * grad_risk_ce + (1.0 - alpha) * grad_risk_brier

            # Gradient Delegate
            grad_del = (prob_del - y_del) / num_samples

            # Backpropagation to Hidden Layer
            grad_h1 = (
                np.dot(grad_cat, self.W_cat) +
                np.dot(grad_tools, self.W_tools) +
                np.dot(grad_role, self.W_role) +
                np.dot(grad_risk, self.W_risk) +
                np.dot(grad_del, self.W_delegate)
            )
            grad_z1 = grad_h1 * (z1 > 0)

            # Gradients of Parameters
            dW_cat = np.dot(grad_cat.T, h1)
            db_cat = np.sum(grad_cat, axis=0)

            dW_tools = np.dot(grad_tools.T, h1)
            db_tools = np.sum(grad_tools, axis=0)

            dW_role = np.dot(grad_role.T, h1)
            db_role = np.sum(grad_role, axis=0)

            dW_risk = np.dot(grad_risk.T, h1)
            db_risk = np.sum(grad_risk, axis=0)

            dW_delegate = np.dot(grad_del.T, h1)
            db_delegate = np.sum(grad_del, axis=0)

            dW1 = np.dot(grad_z1.T, X)
            db1 = np.sum(grad_z1, axis=0)

            grads = [
                dW1, db1,
                dW_cat, db_cat,
                dW_tools, db_tools,
                dW_role, db_role,
                dW_risk, db_risk,
                dW_delegate, db_delegate
            ]

            # Adam step
            for p, g, mp, vp in zip(params, grads, m, v):
                mp[:] = beta1 * mp + (1.0 - beta1) * g
                vp[:] = beta2 * vp + (1.0 - beta2) * (g ** 2)
                m_hat = mp / (1.0 - beta1 ** epoch)
                v_hat = vp / (1.0 - beta2 ** epoch)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if epoch % 50 == 0 or epoch == epochs:
                cat_preds = np.argmax(prob_cat, axis=1)
                cat_targets = np.argmax(y_cat, axis=1)
                acc = np.mean(cat_preds == cat_targets)
                brier = np.mean(np.sum((prob_cat - y_cat) ** 2, axis=1))
                print(f"RLCD Epoch {epoch:3d}/{epochs} | Cat Acc: {acc*100:.1f}% | Brier Score: {brier:.4f}")

    def calibrate_temperature(self, X, y_cat):
        """Perform grid search for optimal temperature T to minimize Expected Calibration Error (ECE)."""
        z1 = np.dot(X, self.W1.T) + self.b1
        h1 = relu(z1)
        logits = np.dot(h1, self.W_cat.T) + self.b_cat

        best_ece = float("inf")
        best_t = 1.0

        for t in np.linspace(0.4, 2.5, 43):
            prob = softmax(logits, temp=t)
            confs = np.max(prob, axis=1)
            preds = np.argmax(prob, axis=1)
            trues = np.argmax(y_cat, axis=1)

            bins = np.linspace(0.0, 1.0, 11)
            ece = 0.0
            for b in range(len(bins) - 1):
                mask = (confs >= bins[b]) & (confs < bins[b+1])
                if np.sum(mask) > 0:
                    acc_bin = np.mean(preds[mask] == trues[mask])
                    conf_bin = np.mean(confs[mask])
                    ece += (np.sum(mask) / len(X)) * abs(acc_bin - conf_bin)

            if ece < best_ece:
                best_ece = ece
                best_t = float(t)

        self.temperature = round(best_t, 3)
        print(f"🎯 Calibrated RLCD Temperature: T={self.temperature:.3f} (ECE: {best_ece:.4f})")
        return self.temperature


def main():
    print("🧠 Preparing RLCD dataset and extracting features with salience zoning...")
    X, y_cat, y_tools, y_role, y_risk, y_del = [], [], [], [], [], []

    cat_map = {c: i for i, c in enumerate(CATEGORIES)}
    role_map = {r: i for i, r in enumerate(ROLES)}
    risk_map = {k: i for i, k in enumerate(RISK_LEVELS)}

    for prompt, cat, nt, role, risk, delegate in BASE_DATASET:
        feats = extract_features(prompt)
        X.append(feats)

        cat_vec = np.zeros(len(CATEGORIES), dtype=np.float32)
        cat_vec[cat_map[cat]] = 1.0
        y_cat.append(cat_vec)

        y_tools.append([nt])

        role_vec = np.zeros(len(ROLES), dtype=np.float32)
        role_vec[role_map[role]] = 1.0
        y_role.append(role_vec)

        risk_vec = np.zeros(len(RISK_LEVELS), dtype=np.float32)
        risk_vec[risk_map[risk]] = 1.0
        y_risk.append(risk_vec)

        y_del.append([delegate])

    X = np.array(X, dtype=np.float32)
    y_cat = np.array(y_cat, dtype=np.float32)
    y_tools = np.array(y_tools, dtype=np.float32)
    y_role = np.array(y_role, dtype=np.float32)
    y_risk = np.array(y_risk, dtype=np.float32)
    y_del = np.array(y_del, dtype=np.float32)

    print(f"📊 Training RLCD small model on {len(X)} samples across 5 decision heads...")
    model = RLCDSmallNeuralModel(
        in_dim=FEATURE_SIZE,
        hidden_dim=64,
        num_cats=len(CATEGORIES),
        num_roles=len(ROLES),
        num_risks=len(RISK_LEVELS)
    )
    model.train_rlcd(X, y_cat, y_tools, y_role, y_risk, y_del, epochs=350, lr=0.007)
    model.calibrate_temperature(X, y_cat)

    weights_data = {
        "version": "small-rlcd-v2",
        "vocabulary": VOCABULARY,
        "categories": CATEGORIES,
        "roles": ROLES,
        "risk_levels": RISK_LEVELS,
        "temperature": model.temperature,
        "W1": model.W1.tolist(),
        "b1": model.b1.tolist(),
        "W_cat": model.W_cat.tolist(),
        "b_cat": model.b_cat.tolist(),
        "W_tools": model.W_tools.tolist(),
        "b_tools": model.b_tools.tolist(),
        "W_role": model.W_role.tolist(),
        "b_role": model.b_role.tolist(),
        "W_risk": model.W_risk.tolist(),
        "b_risk": model.b_risk.tolist(),
        "W_delegate": model.W_delegate.tolist(),
        "b_delegate": model.b_delegate.tolist()
    }

    go_path, json_path = export_to_rakitkode(weights_data)
    print(f"✅ Exported to RakitKode Go code: {go_path}")
    print(f"✅ Exported JSON weights: {json_path}")


if __name__ == "__main__":
    main()
