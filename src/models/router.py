"""High-Level Developer Router & Fast Tool Dispatcher API for Pandu.

Uses Pandu-Jev as an ultra-fast (<25ms), zero-token, local TypeSafe System One
primitive to route software engineering tasks, dispatch agent tools, and triage code.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from rich.console import Console

from datasets.universal_dataset import (
    CODE_QUALITY_LEVELS,
    CODE_REVIEW_CRITERIA,
    DIFFICULTY_LEVELS,
    MODEL_ROUTING_CRITERIA,
    TOOL_DISPATCH_CRITERIA,
)
from models.jev_nlp import PanduJevNLP, PanduJevNLPConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ModelRouteDecision:
    selected_model: str
    probabilities: Dict[str, float]
    needs_reasoning: float
    is_high_risk: float
    difficulty_score: float
    confidence: float
    latency_ms: float
    output_tokens: int = 0


@dataclass
class ToolDispatchDecision:
    selected_tool: str
    probabilities: Dict[str, float]
    needs_read: float
    is_done: float
    confidence: float
    latency_ms: float
    output_tokens: int = 0


@dataclass
class CodeTriageDecision:
    verdict: str
    probabilities: Dict[str, float]
    has_vulnerability: float
    quality_score: float
    confidence: float
    latency_ms: float
    output_tokens: int = 0


class PanduRouter:
    """Developer Router & Fast Tool Dispatcher powered by local Pandu-Jev System One."""

    def __init__(
        self,
        use_base: bool = False,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.use_base = use_base
        config = PanduJevNLPConfig(use_pretrained_base=use_base)
        self.model = PanduJevNLP(config).to(self.device)

        # Load checkpoint if exists
        ckpt = checkpoint_path
        if not ckpt:
            def_name = "pandu_universal_base.pt" if use_base else "pandu_universal_tiny.pt"
            candidate = PROJECT_ROOT / "checkpoints" / def_name
            if candidate.is_file():
                ckpt = candidate

        if ckpt and Path(ckpt).is_file():
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        self.model.eval()
        self.metadata = {
            "tier": "ModernBERT-Base (149M)" if use_base else "ModernBERT-Tiny (19.3M)",
            "device": self.device.upper(),
            "checkpoint": str(ckpt) if ckpt else "none (zero-shot base)",
            "output_tokens": 0,
        }

    def route_model(self, task_description: str) -> ModelRouteDecision:
        """Route a software task to the optimal model tier in <25ms."""
        state = f"Task Description: {task_description}"
        questions = {
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
        }

        out = self.model.predict(state, questions)
        route_ans = out["answers"]["route"]
        reason_ans = out["answers"]["reasoning"]
        risk_ans = out["answers"]["risk"]
        diff_ans = out["answers"]["difficulty"]

        return ModelRouteDecision(
            selected_model=route_ans["choice"],
            probabilities=route_ans["probabilities"],
            needs_reasoning=reason_ans["noul"],
            is_high_risk=risk_ans["noul"],
            difficulty_score=diff_ans.get("score", 0.0),
            confidence=route_ans["confidence"],
            latency_ms=out["latency_ms"],
        )

    def select_tool(
        self,
        agent_state: str,
        available_tools: Optional[List[str]] = None,
    ) -> ToolDispatchDecision:
        """Dispatch the immediate next tool for an autonomous coding agent."""
        criteria = TOOL_DISPATCH_CRITERIA
        if available_tools:
            criteria = {k: v for k, v in criteria.items() if k in available_tools}

        questions = {
            "tool": {
                "type": "choice",
                "instructions": "Select the immediate next tool the coding agent should execute.",
                "criteria": criteria,
            },
            "needs_read": {
                "type": "noul",
                "instructions": "Does the agent need to inspect or read code before proceeding?",
            },
            "is_done": {
                "type": "noul",
                "instructions": "Has the task objective been fully accomplished and verified?",
            },
        }

        out = self.model.predict(agent_state, questions)
        tool_ans = out["answers"]["tool"]
        read_ans = out["answers"]["needs_read"]
        done_ans = out["answers"]["is_done"]

        return ToolDispatchDecision(
            selected_tool=tool_ans["choice"],
            probabilities=tool_ans["probabilities"],
            needs_read=read_ans["noul"],
            is_done=done_ans["noul"],
            confidence=tool_ans["confidence"],
            latency_ms=out["latency_ms"],
        )

    def triage_code(self, code_snippet: str) -> CodeTriageDecision:
        """Evaluate code quality and security vulnerability in <25ms."""
        state = f"Code snippet under review:\n{code_snippet}"
        questions = {
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
        }

        out = self.model.predict(state, questions)
        verdict_ans = out["answers"]["verdict"]
        vuln_ans = out["answers"]["vulnerable"]
        qual_ans = out["answers"]["quality"]

        verdict = verdict_ans["choice"]
        if vuln_ans["noul"] > 0.50 and verdict == "approve":
            verdict = "security_escalation"

        return CodeTriageDecision(
            verdict=verdict,
            probabilities=verdict_ans["probabilities"],
            has_vulnerability=vuln_ans["noul"],
            quality_score=qual_ans.get("score", 0.0),
            confidence=verdict_ans["confidence"],
            latency_ms=out["latency_ms"],
        )
