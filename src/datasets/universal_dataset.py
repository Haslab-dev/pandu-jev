"""Universal Multi-Domain TypeSafe System One Dataset Generator.

Generates rich, high-quality semantic decision datasets for training Pandu:
1. Developer Model Routing (local_reflex vs fast_coder vs deep_reasoner vs human_escalation)
2. Agentic Tool Dispatching (read_file, edit_file, run_command, search_code, ask_user, finish_task)
3. Code Review & Vulnerability Triage (approve, request_changes, security_escalation)
4. Multi-Domain Intent & Triage (billing, bug, feature, incident)
5. Embodied Reflex Retention (Snake spatial safety & Chess tactical intent)

All samples follow TypeSafe System One primitives: Choice, Noul, Score.
"""

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# ============================================================================
# 1. Developer Model Routing Scenarios
# ============================================================================

MODEL_ROUTING_TEMPLATES = [
    # Local Reflex (Pandu)
    {
        "state": "User requested: 'Add type hints to this 15-line helper function in utils.py.' File content is already loaded in memory.",
        "target_model": "local_reflex",
        "reasoning": False,
        "high_risk": False,
        "difficulty": 0,  # Level 0 (Trivial)
    },
    {
        "state": "User requested: 'Fix typo in README.md from Recurrant to Recurrent.'",
        "target_model": "local_reflex",
        "reasoning": False,
        "high_risk": False,
        "difficulty": 0,
    },
    {
        "state": "User requested: 'Format this Python dictionary using standard PEP8 formatting and sort the keys alphabetically.'",
        "target_model": "local_reflex",
        "reasoning": False,
        "high_risk": False,
        "difficulty": 0,
    },
    {
        "state": "Linter reports: 'Unused import os on line 3 in src/main.py'. Request is to remove the unused import.",
        "target_model": "local_reflex",
        "reasoning": False,
        "high_risk": False,
        "difficulty": 0,
    },
    {
        "state": "User asks: 'Extract all email addresses from this text using a Python regex.'",
        "target_model": "local_reflex",
        "reasoning": False,
        "high_risk": False,
        "difficulty": 1,
    },
    # Fast Coder
    {
        "state": "User requested: 'Write a unit test with pytest for this calculate_discount function handling 0%, 50%, and invalid negative inputs.'",
        "target_model": "fast_coder",
        "reasoning": False,
        "high_risk": False,
        "difficulty": 1,
    },
    {
        "state": "User requested: 'Implement an Express.js healthcheck endpoint /api/health returning uptime, memory usage, and status 200 OK.'",
        "target_model": "fast_coder",
        "reasoning": False,
        "high_risk": False,
        "difficulty": 1,
    },
    {
        "state": "User requested: 'Convert this JSON schema definition into a Pydantic v2 BaseModel with Field validators.'",
        "target_model": "fast_coder",
        "reasoning": False,
        "high_risk": False,
        "difficulty": 2,
    },
    {
        "state": "User requested: 'Add docstrings with Sphinx format explaining parameters and return values for these 4 methods in AuthManager.'",
        "target_model": "fast_coder",
        "reasoning": False,
        "high_risk": False,
        "difficulty": 1,
    },
    # Deep Reasoner
    {
        "state": "User requested: 'Architect a distributed consensus protocol with Raft to handle network partitioning and Byzantine edge cases across 7 nodes.'",
        "target_model": "deep_reasoner",
        "reasoning": True,
        "high_risk": True,
        "difficulty": 4,
    },
    {
        "state": "User requested: 'Debug this intermittent concurrency deadlock occurring in the Go goroutine worker pool under 100K QPS load. Trace stack dump shows circular mutex locking between pool.sync and queue.lock.'",
        "target_model": "deep_reasoner",
        "reasoning": True,
        "high_risk": False,
        "difficulty": 4,
    },
    {
        "state": "User requested: 'Refactor our monolithic Django authentication layer into an OAuth2/OIDC provider with PKCE, JWT token rotation, and zero downtime database migration across 5 million active users.'",
        "target_model": "deep_reasoner",
        "reasoning": True,
        "high_risk": True,
        "difficulty": 4,
    },
    {
        "state": "User requested: 'Derive the mathematical proof for the convergence bound of this custom loss function under non-convex SGD.'",
        "target_model": "deep_reasoner",
        "reasoning": True,
        "high_risk": False,
        "difficulty": 4,
    },
    # Human Escalation
    {
        "state": "User requested: 'Execute DROP DATABASE production_users CASCADE; and delete all backup snapshots in S3 bucket.'",
        "target_model": "human_escalation",
        "reasoning": False,
        "high_risk": True,
        "difficulty": 3,
    },
    {
        "state": "User requested: 'Deploy unverified experimental code directly to main branch and disable CI/CD branch protection rules.'",
        "target_model": "human_escalation",
        "reasoning": False,
        "high_risk": True,
        "difficulty": 3,
    },
    {
        "state": "User prompt is vague: 'Make the whole website better and fix everything that looks bad.' No specific requirements or files specified.",
        "target_model": "human_escalation",
        "reasoning": False,
        "high_risk": False,
        "difficulty": 2,
    },
]

MODEL_ROUTING_CRITERIA = {
    "local_reflex": "Local fast reflex model. Best for instant formatting, lints, typing, regex, and deterministic small edits (<30ms, $0.00).",
    "fast_coder": "Cloud fast coder tier. Best for boilerplate generation, standard unit tests, API endpoints, and routine implementations.",
    "deep_reasoner": "Frontier deep reasoning model. Required for complex architectural design, multi-threaded deadlocks, mathematical proofs, and deep refactors.",
    "human_escalation": "Escalate to human review. Required for destructive database actions, high-risk production deployments, or ambiguous underspecified instructions.",
}

DIFFICULTY_LEVELS = [
    "level 0: Trivial edit, formatting, or single-token syntax fix.",
    "level 1: Straightforward routine implementation or basic unit test.",
    "level 2: Multi-step feature implementation within known patterns.",
    "level 3: Complex multi-file feature or high-risk operational action.",
    "level 4: Deep architectural redesign, concurrency deadlock, or algorithmic proof.",
]


# ============================================================================
# 2. Agentic Fast Tool Dispatching Scenarios
# ============================================================================

TOOL_DISPATCH_TEMPLATES = [
    {
        "state": "Agent goal: 'Fix TypeError in src/models/policy.py on line 88'. Current agent status: No files have been read yet. Target file path is known.",
        "target_tool": "read_file",
        "needs_read": True,
        "is_done": False,
    },
    {
        "state": "Agent has viewed src/models/policy.py lines 80-100. Bug identified: variable 'temp' is used before assignment. Replacement chunk is formulated.",
        "target_tool": "edit_file",
        "needs_read": False,
        "is_done": False,
    },
    {
        "state": "Agent has viewed code lines 1-50. Bug is identified and ready to fix. Target file path and line numbers are known.",
        "target_tool": "edit_file",
        "needs_read": False,
        "is_done": False,
    },
    {
        "state": "Agent has applied code edit to src/models/policy.py. Next step is to verify whether tests now pass.",
        "target_tool": "run_command",
        "needs_read": False,
        "is_done": False,
    },
    {
        "state": "Agent finished making code changes to fix the bug. Next step is to execute pytest command to verify.",
        "target_tool": "run_command",
        "needs_read": False,
        "is_done": False,
    },
    {
        "state": "User requested: 'Where is the database connection pool initialized in this repository?' Agent has not searched the codebase yet.",
        "target_tool": "search_code",
        "needs_read": False,
        "is_done": False,
    },
    {
        "state": "User prompt: 'Delete user accounts older than 2 years'. Requirement is ambiguous regarding soft delete vs hard delete, and no confirmation was provided.",
        "target_tool": "ask_user",
        "needs_read": False,
        "is_done": False,
    },
    {
        "state": "All tests have passed (34 passed in 2.1s). Code changes are committed and verified. User goal is fully achieved.",
        "target_tool": "finish_task",
        "needs_read": False,
        "is_done": True,
    },
    {
        "state": "Agent goal: 'Find the latest docs for typesafe-sdk on PyPI'. Local codebase does not have external documentation.",
        "target_tool": "web_search",
        "needs_read": False,
        "is_done": False,
    },
]

TOOL_DISPATCH_CRITERIA = {
    "read_file": "Inspect contents of a file when code needs examination before modification.",
    "edit_file": "Apply a targeted modification or bugfix to an existing file.",
    "run_command": "Execute test suites, build tools, linters, or terminal commands to verify behavior.",
    "search_code": "Search across the codebase for function definitions, classes, or keywords.",
    "web_search": "Search external web sources for documentation, packages, or error solutions.",
    "ask_user": "Prompt user for clarification when requirements are ambiguous or decisions require authorization.",
    "finish_task": "Declare task complete after all requirements are met and verified.",
}


# ============================================================================
# 3. Code Review & Vulnerability Triage
# ============================================================================

CODE_REVIEW_TEMPLATES = [
    {
        "state": "Code snippet under review:\ncursor.execute(f\"SELECT * FROM users WHERE username = '{user_input}'\")\nUser input is concatenated directly into SQL statement with critical SQL injection hazard.",
        "verdict": "security_escalation",
        "vulnerability": True,
        "quality": 0,  # Level 0 (Critical vulnerability)
    },
    {
        "state": "Code snippet under review:\napi_key = \"sk-proj-9923847298374928374928374\"\nHardcoded production API secret committed directly in source code.",
        "verdict": "security_escalation",
        "vulnerability": True,
        "quality": 0,
    },
    {
        "state": "Code snippet under review:\ndef calculate_total(items):\n    return sum(item.price for item in items)\nClean Python function with clear naming and list comprehension.",
        "verdict": "approve",
        "vulnerability": False,
        "quality": 4,  # Level 4 (Production ready)
    },
    {
        "state": "Code snippet under review:\nclass UserSession:\n    def __init__(self, user_id: str):\n        self.user_id = user_id\n        self.created_at = time.time()\nClean class constructor with typed parameters.",
        "verdict": "approve",
        "vulnerability": False,
        "quality": 4,
    },
    {
        "state": "Code snippet under review:\ndef format_currency(cents: int) -> str:\n    return f\"${cents / 100:.2f}\"\nPure deterministic math formatting function.",
        "verdict": "approve",
        "vulnerability": False,
        "quality": 4,
    },
    {
        "state": "Code snippet under review:\ndef get_user(id):\n    user = db.find(id)\n    return user.name\nMissing null check if user is None, causing potential AttributeError.",
        "verdict": "request_changes",
        "vulnerability": False,
        "quality": 2,
    },
    {
        "state": "Code snippet under review:\nimport subprocess\nsubprocess.run(f\"rm -rf {user_provided_dir}\", shell=True)\nCommand injection hazard via unsanitized shell=True.",
        "verdict": "security_escalation",
        "vulnerability": True,
        "quality": 0,
    },
    {
        "state": "Code snippet: 'with open(path, \"r\") as f: data = json.load(f)'. Standard idiomatic context manager for file I/O.",
        "verdict": "approve",
        "vulnerability": False,
        "quality": 4,
    },
]

CODE_REVIEW_CRITERIA = {
    "approve": "Code is clean, safe, and ready to merge into production.",
    "request_changes": "Code contains non-critical bugs, missing null checks, or styling issues that require developer revision.",
    "security_escalation": "Code contains critical security vulnerabilities (SQL injection, hardcoded secrets, shell injection) requiring immediate escalation.",
}

CODE_QUALITY_LEVELS = [
    "level 0: Critical security vulnerability or fatal crash bug.",
    "level 1: Serious logical defect or uncaught exception risk.",
    "level 2: Flawed edge-case handling or missing validation.",
    "level 3: Functional code with minor stylistic or performance debt.",
    "level 4: High-quality, safe, and production-ready code.",
]


# ============================================================================
# 4. Multi-Domain Customer & Infrastructure Triage
# ============================================================================

TRIAGE_TEMPLATES = [
    {
        "state": "Support ticket: 'Our payment processor returned error code 500 on all transactions in the last 15 minutes. Customers cannot checkout.'",
        "category": "security_incident",
        "urgent": True,
    },
    {
        "state": "Support ticket: 'Can we get an invoice for our monthly subscription sent to billing@company.com?'",
        "category": "billing",
        "urgent": False,
    },
    {
        "state": "Support ticket: 'Would love to see dark mode added to the mobile app settings.'",
        "category": "feature_request",
        "urgent": False,
    },
    {
        "state": "Support ticket: 'When clicking the export button on Safari 17, the download modal spins indefinitely.'",
        "category": "technical_bug",
        "urgent": False,
    },
    {
        "state": "Alert: 'CPU load 99% on production database cluster. Replica lag exceeds 120 seconds. P99 latency degraded to 4200ms.'",
        "category": "security_incident",
        "urgent": True,
    },
]

TRIAGE_CRITERIA = {
    "billing": "Invoice, pricing, payment card, or subscription questions.",
    "technical_bug": "Defect, error, unexpected behavior, or browser incompatibility in application.",
    "feature_request": "Enhancement proposal or new capability suggestion.",
    "security_incident": "Production outage, data breach alert, or severe revenue-impacting system disruption.",
}


# ============================================================================
# Universal Dataset Generator Function
# ============================================================================

def generate_universal_training_samples(multiplier: int = 5, seed: int = 42) -> List[Dict[str, Any]]:
    """Synthesizes balanced, multi-task System One training samples."""
    random.seed(seed)
    samples = []

    for _ in range(multiplier):
        # 1. Model Routing Samples
        for t in MODEL_ROUTING_TEMPLATES:
            # Perturb wording slightly for data augmentation
            state = t["state"]
            if random.random() < 0.3:
                state = state + f" Context: Active workspace branch '{random.choice(['main', 'feat-auth', 'fix-db'])}'."

            target_probs = {}
            for k in MODEL_ROUTING_CRITERIA:
                target_probs[k] = 0.90 if k == t["target_model"] else (0.10 / (len(MODEL_ROUTING_CRITERIA) - 1))

            samples.append({
                "domain": "model_routing",
                "state": state,
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Select the most appropriate model tier to handle this developer request.",
                        "criteria": MODEL_ROUTING_CRITERIA,
                    },
                    "reasoning": {
                        "type": "noul",
                        "instructions": "Does this task require complex multi-step reasoning or deep architectural proofs?",
                    },
                    "risk": {
                        "type": "noul",
                        "instructions": "Is this action potentially destructive or safety-critical?",
                    },
                    "difficulty": {
                        "type": "score",
                        "instructions": "Rate the technical complexity level of this developer request.",
                        "criteria": DIFFICULTY_LEVELS,
                    },
                },
                "targets": {
                    "route": target_probs,
                    "reasoning": 0.95 if t["reasoning"] else 0.05,
                    "risk": 0.95 if t["high_risk"] else 0.05,
                    "difficulty": t["difficulty"],
                },
            })

        # 2. Tool Dispatching Samples
        for t in TOOL_DISPATCH_TEMPLATES:
            state = t["state"]
            target_probs = {}
            for k in TOOL_DISPATCH_CRITERIA:
                target_probs[k] = 0.92 if k == t["target_tool"] else (0.08 / (len(TOOL_DISPATCH_CRITERIA) - 1))

            samples.append({
                "domain": "tool_dispatch",
                "state": state,
                "questions": {
                    "tool": {
                        "type": "choice",
                        "instructions": "Select the immediate next tool the coding agent should execute.",
                        "criteria": TOOL_DISPATCH_CRITERIA,
                    },
                    "needs_read": {
                        "type": "noul",
                        "instructions": "Does the agent need to inspect or read code before proceeding?",
                    },
                    "is_done": {
                        "type": "noul",
                        "instructions": "Has the task objective been fully accomplished and verified?",
                    },
                },
                "targets": {
                    "tool": target_probs,
                    "needs_read": 0.95 if t["needs_read"] else 0.05,
                    "is_done": 0.95 if t["is_done"] else 0.05,
                },
            })

        # 3. Code Review Samples
        for t in CODE_REVIEW_TEMPLATES:
            state = t["state"]
            target_probs = {}
            for k in CODE_REVIEW_CRITERIA:
                target_probs[k] = 0.92 if k == t["verdict"] else (0.08 / (len(CODE_REVIEW_CRITERIA) - 1))

            samples.append({
                "domain": "code_review",
                "state": state,
                "questions": {
                    "verdict": {
                        "type": "choice",
                        "instructions": "Determine the pull request code review verdict for this snippet.",
                        "criteria": CODE_REVIEW_CRITERIA,
                    },
                    "vulnerable": {
                        "type": "noul",
                        "instructions": "Does this code contain a critical security vulnerability or injection hazard?",
                    },
                    "quality": {
                        "type": "score",
                        "instructions": "Rate the overall safety and quality level of this code snippet.",
                        "criteria": CODE_QUALITY_LEVELS,
                    },
                },
                "targets": {
                    "verdict": target_probs,
                    "vulnerable": 0.95 if t["vulnerability"] else 0.05,
                    "quality": t["quality"],
                },
            })

        # 4. Multi-Domain Triage Samples
        for t in TRIAGE_TEMPLATES:
            state = t["state"]
            target_probs = {}
            for k in TRIAGE_CRITERIA:
                target_probs[k] = 0.92 if k == t["category"] else (0.08 / (len(TRIAGE_CRITERIA) - 1))

            samples.append({
                "domain": "incident_triage",
                "state": state,
                "questions": {
                    "category": {
                        "type": "choice",
                        "instructions": "Categorize this incident or user request into the correct operational queue.",
                        "criteria": TRIAGE_CRITERIA,
                    },
                    "urgent": {
                        "type": "noul",
                        "instructions": "Is this issue an urgent disruption requiring immediate emergency response?",
                    },
                },
                "targets": {
                    "category": target_probs,
                    "urgent": 0.95 if t["urgent"] else 0.05,
                },
            })

    random.shuffle(samples)
    return samples
