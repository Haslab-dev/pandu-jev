"""Base Decision Model Tier (149M ModernBERT-Base, ~300 MB - 570 MB).

Provides deep semantic code comprehension, architecture proofs,
and security vulnerability triage running on local Apple Silicon MPS.
"""

from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BaseDecisionRouter:
    """Wrapper for 149M ModernBERT-Base developer routing & security triage."""

    def __init__(self, device: Optional[str] = None):
        self.device = device
        # Lazy load to avoid importing heavy transformers when using Micro/Tiny
        from models.router import PanduRouter
        self.router = PanduRouter(use_base=True, device=device)

    def route_task(self, task: str) -> Dict[str, Any]:
        """Route task across model tiers (local_reflex, fast_coder, deep_reasoner, human_escalation)."""
        res = self.router.route_model(task)
        return {
            "selected_model": res.selected_model,
            "needs_reasoning": res.needs_reasoning,
            "is_high_risk": res.is_high_risk,
            "difficulty_score": res.difficulty_score,
            "confidence": res.confidence,
            "latency_ms": res.latency_ms,
            "output_tokens": 0,
        }

    def triage_security(self, code_snippet: str) -> Dict[str, Any]:
        """Evaluate code quality and security injection vulnerabilities."""
        res = self.router.triage_code(code_snippet)
        return {
            "verdict": res.verdict,
            "has_vulnerability": res.has_vulnerability,
            "quality_score": res.quality_score,
            "confidence": res.confidence,
            "latency_ms": res.latency_ms,
            "output_tokens": 0,
        }
