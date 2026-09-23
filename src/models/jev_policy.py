"""TypeSafe Jev System One Policy Adapter for Snake Arena.

Connects to TypeSafe's live System One flagship model (jev-1.13) via typesafe-sdk
and converts game state into typed Choice and Noul questions.
"""

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Load .env if present
env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from typesafe_sdk import Choice, Noul, TypeSafeClient

# Reference to Laya / Pandu structures
LAYA_PATH = Path("/Users/hy4-mac-002/hasdev/research/laya-coreml")
if str(LAYA_PATH) not in sys.path:
    sys.path.insert(0, str(LAYA_PATH))

from laya_coreml.snake.game import DIRECTIONS, SnakeGame
from laya_coreml.snake.policy import Decision


class JevPolicy:
    """TypeSafe Jev System One Policy Adapter for Snake Arena."""

    def __init__(self, api_key: Optional[str] = None, model: str = "jev-latest", guarded: bool = True):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TYPESAFE_API_KEY is required to initialize JevPolicy. "
                "Set it in .env or pass api_key parameter."
            )
        self.model = model
        self.guarded = guarded
        self.client = TypeSafeClient(api_key=self.api_key)
        self.metadata = {
            "name": f"TypeSafe {model}",
            "hardware": "TypeSafe Cloud API",
            "engine": "Jev (System One)",
            "guarded": self.guarded,
        }

    def decide(self, game: SnakeGame) -> Decision:
        """Query Jev System One API for next move."""
        started = time.perf_counter()
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        
        if not safe and self.guarded:
            raise RuntimeError("Cycle safety invariant violated: no safe action")
            
        preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
        reachable, space = game.food_reachability()

        # Format semantic criteria matching TypeSafe skills best practice
        descriptions = {}
        for m in moves:
            if not m.legal:
                descriptions[m.direction] = "Blocked by wall or snake body. Collision hazard. Do not move here."
            elif not m.safe:
                descriptions[m.direction] = "Unsafe route. High risk of dead end or trapping the snake."
            elif m.eats:
                descriptions[m.direction] = "Safe route that reaches food immediately. Best move."
            elif m.direction == preferred:
                descriptions[m.direction] = "Safe route that advances closest toward food."
            else:
                descriptions[m.direction] = "Safe open route but slower progress toward food."

        state = (
            f"Snake game board status. Safe directions count: {len(safe)}. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}. "
            f"Open space: {space} cells. Current snake length: {len(game.body)}. "
            f"{'Safe route forward exists.' if safe else 'Snake is currently trapped.'}"
        )

        questions = {
            "move": Choice(
                instructions="Select the best safe move that makes progress toward food and avoids collisions.",
                criteria=descriptions,
            ),
            "risk": Noul(
                instructions="Is a safe forward route available without collision or immediate trapping?",
            ),
            "food": Noul(
                instructions="Is food reachable through currently available empty cells?",
            ),
        }

        inference_start = time.perf_counter()
        response = self.client.system_one(state=state, questions=questions, model=self.model)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        # Parse Choice
        move_ans = response.choices["move"]
        raw_probs = dict(move_ans.probabilities)
        probabilities = {d: float(raw_probs.get(d, 0.0)) for d in DIRECTIONS}
        # Normalize in case of slight numerical drift
        total_p = sum(probabilities.values()) or 1.0
        probabilities = {d: p / total_p for d, p in probabilities.items()}

        # Parse Nouls
        risk_noul = float(response.nouls["risk"].noul)
        food_noul = float(response.nouls["food"].noul)

        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        allowed = [m.direction for m in safe]
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and proposed not in allowed
            else proposed
        )

        self.last_request_id = getattr(response, "request_id", "")
        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        self.total_input_tokens = getattr(self, "total_input_tokens", 0) + input_tokens
        self.total_output_tokens = getattr(self, "total_output_tokens", 0) + output_tokens

        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=round(1.0 - risk_noul, 4),
            food_reachable=round(food_noul, 4),
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            safe_count=len(safe),
            planner_best=preferred,
        )

    def close(self):
        if hasattr(self, "client") and self.client:
            self.client.close()

    def __del__(self):
        self.close()
