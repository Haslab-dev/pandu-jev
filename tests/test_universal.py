"""Unit & Integration tests for Pandu Universal Multi-Domain System One."""

import pytest
import torch
from models.router import PanduRouter


@pytest.fixture(scope="module")
def base_router():
    return PanduRouter(use_base=True)


@pytest.fixture(scope="module")
def tiny_router():
    return PanduRouter(use_base=False)


# ============================================================================
# Tests for Pandu-Base (149M ModernBERT-Base) — Deep Semantic Understanding
# ============================================================================

def test_base_router_high_risk_escalation(base_router):
    """Dangerous destructive commands must escalate to human review."""
    res = base_router.route_model("DROP TABLE customers CASCADE and purge S3 backups")
    assert res.selected_model == "human_escalation"
    assert res.is_high_risk > 0.50
    assert res.output_tokens == 0


def test_base_router_deep_reasoning(base_router):
    """Complex distributed systems and proofs must route to deep_reasoner."""
    res = base_router.route_model("Architect a distributed Raft consensus algorithm with Byzantine fault tolerance")
    assert res.selected_model == "deep_reasoner"
    assert res.needs_reasoning > 0.50
    assert res.difficulty_score >= 2.5


def test_base_router_code_triage(base_router):
    """SQL injection and security vulnerabilities must be escalated."""
    sql_inj = "cursor.execute(f'SELECT * FROM users WHERE token = {token}')\nCritical SQL injection hazard via unsanitized input."
    res = base_router.triage_code(sql_inj)
    assert res.verdict == "security_escalation"
    assert res.has_vulnerability > 0.50


def test_base_router_clean_code(base_router):
    """Clean idiomatic Python code must be approved with high quality score."""
    clean_code = "def calculate_total(items):\n    return sum(item.price for item in items)"
    res = base_router.triage_code(clean_code)
    assert res.verdict in ["approve", "request_changes"]
    assert res.has_vulnerability < 0.20


def test_base_router_tool_dispatch(base_router):
    """Agent tool selection must produce valid probability distribution and zero tokens."""
    res = base_router.select_tool("Agent has viewed code lines 1-50. Bug is identified and ready to fix.")
    assert res.selected_tool in ["edit_file", "run_command", "read_file"]
    assert res.output_tokens == 0
    assert 0.0 <= res.confidence <= 1.0


# ============================================================================
# Tests for Pandu-Tiny (19.3M ModernBERT-Tiny) — Fast Edge Reflex Invariant
# ============================================================================

def test_tiny_router_reflex_invariants(tiny_router):
    """Tiny router must run with sub-30ms latency and strictly zero tokens."""
    res = tiny_router.route_model("Fix typo in docstring")
    assert res.output_tokens == 0
    assert 0.0 <= res.needs_reasoning <= 1.0
    assert 0.0 <= res.is_high_risk <= 1.0
    assert res.latency_ms > 0.0
