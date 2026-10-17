"""Synthetic Long-Running Trajectory Generator (100-200+ Turns).

Simulates realistic, multi-phase autonomous software engineering sessions
modeled after deep developer pairing (Antigravity / RakitKode / Forge-ADE):
- Epic multi-file refactoring (150 turns)
- Deep debugging saga with thrashing traps (120 turns)
- Multi-component system integration with subagent delegation (180 turns)
- Safety containment and error recovery (100 turns)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np

PHASES = ["discovery", "planning", "implementation", "debugging", "verification", "wrapup"]
TOOL_CATEGORIES = ["explore", "read", "modify", "execute", "git", "web", "skills", "mcp", "none"]


@dataclass
class TrajectoryTurn:
    """Represents a single step in a 100-200 turn agent session."""
    session_id: str
    turn_index: int
    total_session_turns: int
    goal_prompt: str
    current_intent: str
    last_tool: str
    last_tool_failed: bool
    consecutive_failures: int
    action_repetition_count: int
    # Targets to predict
    target_category: str
    target_phase: str
    is_stuck: float
    should_delegate: float
    is_done: float
    target_risk: str


def generate_long_running_trajectories() -> List[TrajectoryTurn]:
    """Generate realistic 100-200 turn trajectories for training and evaluation."""
    trajectories: List[TrajectoryTurn] = []

    # =========================================================================
    # SAGA 1: Epic Micro-Kernel Refactoring (150 Turns)
    # =========================================================================
    goal_1 = "Refactor RakitKode agent runtime from monolithic harness to modular decisions engine with pure Go zero-dependency embeddings"
    total_1 = 150
    tool_history_1 = []

    for t in range(1, total_1 + 1):
        if t <= 15:
            # Phase 1: Discovery (Turns 1-15)
            phase = "discovery"
            cat = "explore" if t % 2 == 1 else "read"
            intent = f"Turn {t}: Exploring codebase structure in internal/gateway" if cat == "explore" else f"Turn {t}: Reading package declaration and types in internal/gateway"
            is_stuck = 0.0
            delegate = 1.0 if t == 12 else 0.0  # Delegate deep reference search to research agent
            is_done = 0.0
            failed = False
            consec_fail = 0
            rep = 1
            risk = "low"
        elif t <= 65:
            # Phase 2: Active Implementation (Turns 16-65)
            phase = "implementation"
            if t % 5 == 0:
                cat = "execute"
                intent = f"Turn {t}: Run intermediate syntax check with go build"
                failed = (t == 35 or t == 50)  # Occasional compile errors
            elif t % 3 == 0:
                cat = "read"
                intent = f"Turn {t}: Verify type signatures and struct definitions"
                failed = False
            else:
                cat = "modify"
                intent = f"Turn {t}: Refactoring decision adapter and replacing legacy shims"
                failed = False
            is_stuck = 0.0
            delegate = 0.0
            is_done = 0.0
            consec_fail = 1 if failed else 0
            rep = 1 if cat != "modify" else 2
            risk = "medium"
        elif t <= 115:
            # Phase 3: Deep Debugging & Thrashing Trap (Turns 66-115)
            phase = "debugging"
            # Introduce a stuck loop at turns 80-84: agent tries to edit same file and tests fail
            if 80 <= t <= 84:
                cat = "modify" if t % 2 == 0 else "execute"
                failed = True
                consec_fail = t - 79
                rep = consec_fail
                is_stuck = 1.0  # STUCK ALARM TRIGGERED!
                intent = f"Turn {t}: Stuck loop trying to resolve cyclic dependency in package imports"
                risk = "medium"
            elif t == 85:
                # Pivot away from loop!
                cat = "read"
                failed = False
                consec_fail = 0
                rep = 1
                is_stuck = 0.0
                intent = f"Turn {t}: Pivoting strategy: inspecting root go.mod dependencies carefully to break cycle"
                risk = "low"
            else:
                cat = "modify" if t % 2 == 1 else "execute"
                failed = (t % 7 == 0)
                consec_fail = 1 if failed else 0
                rep = 1
                is_stuck = 0.0
                intent = f"Turn {t}: Fixing nil pointer dereference in gateway handler" if cat == "modify" else f"Turn {t}: Running unit tests with go test ./internal/gateway"
                risk = "medium"
            delegate = 1.0 if t == 90 else 0.0
            is_done = 0.0
        elif t <= 140:
            # Phase 4: Full Verification (Turns 116-140)
            phase = "verification"
            if t % 3 == 1:
                cat = "execute"
                intent = f"Turn {t}: Running full test suite go test ./... and benchmarks"
            elif t % 3 == 2:
                cat = "read"
                intent = f"Turn {t}: Inspecting test logs and performance metrics"
            else:
                cat = "git"
                intent = f"Turn {t}: Checking git diff to verify no unintended changes"
            failed = False
            consec_fail = 0
            rep = 1
            is_stuck = 0.0
            delegate = 0.0
            is_done = 0.0
            risk = "low"
        else:
            # Phase 5: Wrapup & Commit (Turns 141-150)
            phase = "wrapup"
            if t == 150:
                cat = "none"
                intent = f"Turn {t}: All tests pass, refactor verified. Presenting comprehensive final report"
                is_done = 1.0
                risk = "low"
            elif t >= 148:
                cat = "git"
                intent = f"Turn {t}: Tagging release v0.3.0 and pushing commits"
                is_done = 0.0
                risk = "medium"
            else:
                cat = "modify"
                intent = f"Turn {t}: Updating README.md documentation and architecture diagram"
                is_done = 0.0
                risk = "low"
            failed = False
            consec_fail = 0
            rep = 1
            is_stuck = 0.0
            delegate = 0.0

        last_tool = tool_history_1[-1] if tool_history_1 else "none"
        tool_history_1.append(cat)

        trajectories.append(TrajectoryTurn(
            session_id="session_epic_refactor_150",
            turn_index=t,
            total_session_turns=total_1,
            goal_prompt=goal_1,
            current_intent=intent,
            last_tool=last_tool,
            last_tool_failed=failed,
            consecutive_failures=consec_fail,
            action_repetition_count=rep,
            target_category=cat,
            target_phase=phase,
            is_stuck=is_stuck,
            should_delegate=delegate,
            is_done=is_done,
            target_risk=risk,
        ))

    # =========================================================================
    # SAGA 2: Multi-Component System Integration & Delegation (180 Turns)
    # =========================================================================
    goal_2 = "Integrate distributed event telemetry and real-time WebSocket protocol into Forge-ADE"
    total_2 = 180
    tool_history_2 = []

    for t in range(1, total_2 + 1):
        norm_t = t / total_2
        if norm_t < 0.15:
            phase = "discovery"
            cat = "explore" if t % 2 == 1 else "read"
            intent = f"Turn {t}: Exploring WebSocket frame handlers in codebase" if cat == "explore" else f"Turn {t}: Reading WebSocket frame definition and protocol structs"
            is_stuck = 0.0
            delegate = 1.0 if t == 10 else 0.0
            failed = False
            consec_fail = 0
            rep = 1
            risk = "low"
            is_done = 0.0
        elif norm_t < 0.35:
            phase = "planning"
            cat = "none" if t % 4 == 0 else "read"
            intent = f"Turn {t}: Synthesizing state machine architecture without tool" if cat == "none" else f"Turn {t}: Reading frame synchronization protocol interfaces"
            is_stuck = 0.0
            delegate = 1.0 if t == 35 else 0.0
            failed = False
            consec_fail = 0
            rep = 1
            risk = "low"
            is_done = 0.0
        elif norm_t < 0.70:
            phase = "implementation"
            if t % 6 == 0:
                cat = "execute"
                intent = f"Turn {t}: Compiling protocol buffers and running code generators"
            elif t % 2 == 0:
                cat = "modify"
                intent = f"Turn {t}: Implementing bidirectional frame multiplexer"
            else:
                cat = "read"
                intent = f"Turn {t}: Checking interface compliance and type assertions"
            failed = (t == 95 or t == 110)
            consec_fail = 1 if failed else 0
            rep = 1
            is_stuck = 0.0
            delegate = 0.0
            risk = "medium"
            is_done = 0.0
        elif norm_t < 0.90:
            phase = "debugging"
            # Thrash trap at turns 135-139
            if 135 <= t <= 139:
                cat = "modify" if t % 2 == 1 else "execute"
                failed = True
                consec_fail = t - 134
                rep = consec_fail
                is_stuck = 1.0  # STUCK ALARM
                intent = f"Turn {t}: Retrying patch to resolve WebSocket multiplexer cyclic deadlock" if cat == "modify" else f"Turn {t}: Cyclic deadlock persisting, running execution retry attempt {consec_fail}"
                risk = "medium"
            else:
                cat = "modify" if t % 2 == 0 else "execute"
                failed = False
                consec_fail = 0
                rep = 1
                is_stuck = 0.0
                intent = f"Turn {t}: Adding sync.Mutex locking to prevent race" if cat == "modify" else f"Turn {t}: Verifying concurrency safety with go test -race"
                risk = "medium"
            delegate = 0.0
            is_done = 0.0
        else:
            phase = "verification"
            if t == 180:
                cat = "none"
                intent = f"Turn {t}: WebSocket integration complete, all benchmarks pass with 0 race warnings"
                is_done = 1.0
                risk = "low"
            elif t >= 175:
                cat = "git"
                intent = f"Turn {t}: Git commit and push integration branch"
                is_done = 0.0
                risk = "medium"
            else:
                cat = "execute"
                intent = f"Turn {t}: Running stress benchmarks with 500 concurrent connections"
                is_done = 0.0
                risk = "low"
            failed = False
            consec_fail = 0
            rep = 1
            is_stuck = 0.0
            delegate = 0.0

        last_tool = tool_history_2[-1] if tool_history_2 else "none"
        tool_history_2.append(cat)

        trajectories.append(TrajectoryTurn(
            session_id="session_ws_integration_180",
            turn_index=t,
            total_session_turns=total_2,
            goal_prompt=goal_2,
            current_intent=intent,
            last_tool=last_tool,
            last_tool_failed=failed,
            consecutive_failures=consec_fail,
            action_repetition_count=rep,
            target_category=cat,
            target_phase=phase,
            is_stuck=is_stuck,
            should_delegate=delegate,
            is_done=is_done,
            target_risk=risk,
        ))

    return trajectories
