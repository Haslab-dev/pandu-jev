"""2D Racing Environment with continuous vehicle dynamics (Phase 9).

State:
    speed: normalized forward velocity [0.0, 1.0]
    heading: car orientation relative to track tangent [-1.0, 1.0]
    track_offset: lateral deviation from track centerline [-1.0, 1.0]
    curvature: upcoming track bend rate [-1.0, 1.0]
    obstacle_distance: normalized distance to nearest forward hazard [0.0, 1.0]

Actions:
    0: STEER_LEFT
    1: STEER_RIGHT
    2: ACCELERATE
    3: BRAKE
    4: COAST
"""

from enum import IntEnum
import math
import random
from typing import Any, Dict, Optional, Tuple
import numpy as np

from env.base import Environment


class RacingAction(IntEnum):
    STEER_LEFT = 0
    STEER_RIGHT = 1
    ACCELERATE = 2
    BRAKE = 3
    COAST = 4

    @classmethod
    def from_str(cls, name: str) -> "RacingAction":
        name = name.strip().upper()
        mapping = {
            "STEER_LEFT": cls.STEER_LEFT,
            "STEER_RIGHT": cls.STEER_RIGHT,
            "ACCELERATE": cls.ACCELERATE,
            "BRAKE": cls.BRAKE,
            "COAST": cls.COAST,
        }
        if name not in mapping:
            raise ValueError(f"Unknown racing action: {name}")
        return mapping[name]


class RacingEnv(Environment):
    """Continuous dynamics 2D racing environment."""

    MAX_SPEED = 1.0
    MIN_SPEED = 0.0
    ACCEL_RATE = 0.08
    BRAKE_RATE = 0.12
    DRAG_RATE = 0.01
    STEER_RATE = 0.10

    def __init__(self, max_steps: int = 200, seed: Optional[int] = None):
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.step_count = 0
        self.done = False

        # State variables
        self.speed = 0.2
        self.heading = 0.0
        self.track_offset = 0.0
        self.curvature = 0.0
        self.obstacle_distance = 1.0
        self.lap_progress = 0.0

        self.reset(seed=seed)

    @property
    def action_space_size(self) -> int:
        return 5

    @property
    def feature_dim(self) -> int:
        return 5

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = random.Random(seed)

        self.step_count = 0
        self.done = False
        self.speed = 0.3
        self.heading = 0.0
        self.track_offset = self._rng.uniform(-0.1, 0.1)
        self.curvature = self._rng.uniform(-0.3, 0.3)
        self.obstacle_distance = self._rng.uniform(0.6, 1.0)
        self.lap_progress = 0.0

        return self.observe()

    def observe(self) -> Dict[str, Any]:
        return {
            "speed": float(self.speed),
            "heading": float(self.heading),
            "track_offset": float(self.track_offset),
            "curvature": float(self.curvature),
            "obstacle_distance": float(self.obstacle_distance),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }

    def get_feature_vector(self) -> np.ndarray:
        return np.array(
            [
                self.speed,
                self.heading,
                self.track_offset,
                self.curvature,
                self.obstacle_distance,
            ],
            dtype=np.float32,
        )

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            return self.observe(), 0.0, True, {"msg": "already done"}

        if isinstance(action, str):
            action = RacingAction.from_str(action)
        elif isinstance(action, (int, np.integer)):
            action = RacingAction(int(action))

        self.step_count += 1

        # Apply controls
        if action == RacingAction.STEER_LEFT:
            self.heading -= self.STEER_RATE
        elif action == RacingAction.STEER_RIGHT:
            self.heading += self.STEER_RATE
        elif action == RacingAction.ACCELERATE:
            self.speed = min(self.MAX_SPEED, self.speed + self.ACCEL_RATE)
        elif action == RacingAction.BRAKE:
            self.speed = max(self.MIN_SPEED, self.speed - self.BRAKE_RATE)
        elif action == RacingAction.COAST:
            self.speed = max(self.MIN_SPEED, self.speed - self.DRAG_RATE)

        # Physics update
        # Lateral movement is driven by heading mismatch relative to curvature
        slip_angle = self.heading - self.curvature
        self.track_offset += self.speed * math.sin(slip_angle) * 0.5

        # Heading naturally decays toward forward alignment via caster angle
        self.heading *= 0.85

        # Advance track curvature smoothly
        self.curvature += self._rng.uniform(-0.08, 0.08)
        self.curvature = max(-0.8, min(0.8, self.curvature))

        # Update obstacle
        self.obstacle_distance -= self.speed * 0.08
        if self.obstacle_distance <= 0.0:
            # Respawn new obstacle ahead
            self.obstacle_distance = self._rng.uniform(0.7, 1.2)

        self.lap_progress += self.speed

        # Reward computation
        # Reward high speed + staying centered
        crashed_off_track = abs(self.track_offset) > 1.0
        crashed_obstacle = self.obstacle_distance < 0.08 and abs(self.track_offset) < 0.35

        reward = self.speed * 1.5 - abs(self.track_offset) * 0.8

        info = {
            "crashed": False,
            "lap_completed": False,
            "action": action.name,
        }

        if crashed_off_track or crashed_obstacle:
            reward -= 50.0
            self.done = True
            info["crashed"] = True
            info["crash_reason"] = "off_track" if crashed_off_track else "obstacle_hit"
        elif self.step_count >= self.max_steps:
            reward += 100.0  # Finished lap successfully
            self.done = True
            info["lap_completed"] = True

        return self.observe(), reward, self.done, info

    def render_ascii(self) -> str:
        track_width = 21
        center = track_width // 2
        car_col = int(center + self.track_offset * (center - 2))
        car_col = max(1, min(track_width - 2, car_col))

        row = [" "] * track_width
        row[0] = "|"
        row[-1] = "|"
        row[center] = "."
        row[car_col] = "V"

        status = f"Speed: {self.speed:.2f} | Offset: {self.track_offset:+.2f} | Curv: {self.curvature:+.2f} | ObsDist: {self.obstacle_distance:.2f}"
        return f"{''.join(row)}\n{status}"
