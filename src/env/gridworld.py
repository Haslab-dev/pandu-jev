"""GridWorld Environment implementation for Pandu (pandu-jev).

Follows the universal interface:
    state = env.observe()
    action_distribution = policy(state)
    action = sample(action_distribution)
    next_state, reward, done, info = env.step(action)
"""

from enum import IntEnum
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from env.base import Environment


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @classmethod
    def from_str(cls, name: str) -> "Action":
        name = name.strip().upper()
        mapping = {
            "UP": cls.UP,
            "DOWN": cls.DOWN,
            "LEFT": cls.LEFT,
            "RIGHT": cls.RIGHT,
        }
        if name not in mapping:
            raise ValueError(f"Unknown action: {name}")
        return mapping[name]

    def delta(self) -> Tuple[int, int]:
        """Returns (dx, dy). Note: y increases downward."""
        if self == Action.UP:
            return (0, -1)
        elif self == Action.DOWN:
            return (0, 1)
        elif self == Action.LEFT:
            return (-1, 0)
        elif self == Action.RIGHT:
            return (1, 0)
        return (0, 0)


DEFAULT_MAP = [
    "############",
    "#S         #",
    "#   ###    #",
    "#       #  #",
    "#    #     #",
    "#       # G#",
    "############",
]


class GridWorld(Environment):
    """2D Grid navigation environment with obstacles, step penalties, and goal rewards."""

    REWARD_GOAL = 100.0
    REWARD_WALL = -10.0
    REWARD_STEP = -1.0

    def __init__(
        self,
        map_layout: Optional[List[str]] = None,
        max_steps: int = 150,
        random_start_goal: bool = False,
        dynamic_obstacles: bool = False,
        partial_obs: bool = False,
        obs_radius: int = 3,
        noise_level: float = 0.0,
        feature_dim: int = 16,
        seed: Optional[int] = None,
    ):
        self.raw_layout = [row for row in (map_layout or DEFAULT_MAP)]
        self.height = len(self.raw_layout)
        self.width = max(len(row) for row in self.raw_layout)
        self.max_steps = max_steps
        self.random_start_goal = random_start_goal
        self.dynamic_obstacles = dynamic_obstacles
        self.partial_obs = partial_obs
        self.obs_radius = obs_radius
        self.noise_level = noise_level
        self._feature_dim = feature_dim
        self._rng = random.Random(seed)

        # Parse static map
        self.static_walls: Set[Tuple[int, int]] = set()
        self.default_start: Tuple[int, int] = (1, 1)
        self.default_goal: Tuple[int, int] = (self.width - 2, self.height - 2)

        for y, row in enumerate(self.raw_layout):
            for x, char in enumerate(row):
                if char == "#":
                    self.static_walls.add((x, y))
                elif char == "S":
                    self.default_start = (x, y)
                elif char == "G":
                    self.default_goal = (x, y)

        self.agent_x = self.default_start[0]
        self.agent_y = self.default_start[1]
        self.goal_x = self.default_goal[0]
        self.goal_y = self.default_goal[1]
        self.dynamic_wall_pos: Optional[Tuple[int, int]] = None
        self.dynamic_patrol_dir = 1
        self.step_count = 0
        self.done = False

        self.reset(seed=seed)

    @property
    def action_space_size(self) -> int:
        return 4

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def get_valid_open_cells(self) -> List[Tuple[int, int]]:
        cells = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self.static_walls:
                    cells.append((x, y))
        return cells

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = random.Random(seed)

        self.step_count = 0
        self.done = False

        if self.random_start_goal:
            open_cells = self.get_valid_open_cells()
            if len(open_cells) >= 2:
                samples = self._rng.sample(open_cells, 2)
                self.agent_x, self.agent_y = samples[0]
                self.goal_x, self.goal_y = samples[1]
            else:
                self.agent_x, self.agent_y = self.default_start
                self.goal_x, self.goal_y = self.default_goal
        else:
            self.agent_x, self.agent_y = self.default_start
            self.goal_x, self.goal_y = self.default_goal

        if self.dynamic_obstacles:
            open_cells = [
                c for c in self.get_valid_open_cells()
                if c != (self.agent_x, self.agent_y) and c != (self.goal_x, self.goal_y)
            ]
            if open_cells:
                self.dynamic_wall_pos = self._rng.choice(open_cells)
                self.dynamic_patrol_dir = 1
        else:
            self.dynamic_wall_pos = None

        return self.observe()

    @property
    def current_walls(self) -> Set[Tuple[int, int]]:
        walls = set(self.static_walls)
        if self.dynamic_obstacles and self.dynamic_wall_pos is not None:
            walls.add(self.dynamic_wall_pos)
        return walls

    def observe(self) -> Dict[str, Any]:
        # Handle partial observability: goal coordinates might be masked
        visible_goal_x = self.goal_x
        visible_goal_y = self.goal_y
        goal_detected = True

        if self.partial_obs:
            manhattan = abs(self.agent_x - self.goal_x) + abs(self.agent_y - self.goal_y)
            if manhattan > self.obs_radius:
                visible_goal_x = -1
                visible_goal_y = -1
                goal_detected = False

        return {
            "agent_x": self.agent_x,
            "agent_y": self.agent_y,
            "goal_x": visible_goal_x,
            "goal_y": visible_goal_y,
            "true_goal_x": self.goal_x,
            "true_goal_y": self.goal_y,
            "goal_detected": goal_detected,
            "walls": sorted(list(self.current_walls)),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "width": self.width,
            "height": self.height,
        }

    def _step_dynamic_obstacle(self) -> None:
        if not self.dynamic_obstacles or self.dynamic_wall_pos is None:
            return
        x, y = self.dynamic_wall_pos
        ny = y + self.dynamic_patrol_dir
        if (x, ny) in self.static_walls or ny <= 0 or ny >= self.height - 1:
            self.dynamic_patrol_dir *= -1
            ny = y + self.dynamic_patrol_dir

        if (x, ny) not in self.static_walls and (x, ny) != (self.goal_x, self.goal_y):
            self.dynamic_wall_pos = (x, ny)

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            return self.observe(), 0.0, True, {"msg": "already done"}

        if isinstance(action, str):
            action = Action.from_str(action)
        elif isinstance(action, (int, np.integer)):
            action = Action(int(action))

        self.step_count += 1
        dx, dy = action.delta()
        target_x = self.agent_x + dx
        target_y = self.agent_y + dy

        reward = self.REWARD_STEP
        hit_wall = False
        reached_goal = False

        # Check wall collision or out of bounds
        if (
            (target_x, target_y) in self.current_walls
            or target_x < 0
            or target_x >= self.width
            or target_y < 0
            or target_y >= self.height
        ):
            reward += self.REWARD_WALL
            hit_wall = True
            # agent remains at current position
        else:
            self.agent_x = target_x
            self.agent_y = target_y

        # Check goal reached
        if self.agent_x == self.goal_x and self.agent_y == self.goal_y:
            reward += self.REWARD_GOAL
            reached_goal = True
            self.done = True

        # Check max steps limit
        if self.step_count >= self.max_steps:
            self.done = True

        # Update dynamic obstacles
        self._step_dynamic_obstacle()

        info = {
            "hit_wall": hit_wall,
            "reached_goal": reached_goal,
            "step_count": self.step_count,
            "action": action.name,
        }

        return self.observe(), reward, self.done, info

    def get_feature_vector(self, dim: Optional[int] = None) -> np.ndarray:
        """Normalized feature vector supporting multiple representation dimensions:
        - 16d: Canonical baseline (agent/goal coords, distances, immediate wall sensors, 4 cardinal raycasts)
        - 32d: + 4 diagonal raycasts, 4 dynamic obstacle features, 8-cell 3x3 local occupancy patch
        - 64d: + 16-cell 5x5 local occupancy ring, 8 goal-direction ray projections, 8 corridor clearance depths
        - 128d: + 24-cell 7x7 local occupancy ring, 16-ray circular rangefinder, 24 harmonic Fourier encodings
        """
        target_dim = dim if dim is not None else self._feature_dim
        walls = self.current_walls
        w = max(1.0, float(self.width))
        h = max(1.0, float(self.height))
        diag = math.sqrt(w * w + h * h)

        gx = self.goal_x
        gy = self.goal_y

        if self.partial_obs:
            manhattan = abs(self.agent_x - self.goal_x) + abs(self.agent_y - self.goal_y)
            if manhattan > self.obs_radius:
                gx = self.agent_x
                gy = self.agent_y

        dx = gx - self.agent_x
        dy = gy - self.agent_y
        euc_dist = math.hypot(dx, dy) / max(1.0, diag)
        man_dist = (abs(dx) + abs(dy)) / max(1.0, w + h)

        # Immediate adjacent wall collision sensor
        wall_up = 1.0 if (self.agent_x, self.agent_y - 1) in walls or self.agent_y <= 0 else 0.0
        wall_down = 1.0 if (self.agent_x, self.agent_y + 1) in walls or self.agent_y >= self.height - 1 else 0.0
        wall_left = 1.0 if (self.agent_x - 1, self.agent_y) in walls or self.agent_x <= 0 else 0.0
        wall_right = 1.0 if (self.agent_x + 1, self.agent_y) in walls or self.agent_x >= self.width - 1 else 0.0

        # Raycasts: distance to nearest wall in each direction
        dist_up = 0
        for y in range(self.agent_y - 1, -1, -1):
            dist_up += 1
            if (self.agent_x, y) in walls:
                break
        norm_ray_up = dist_up / h

        dist_down = 0
        for y in range(self.agent_y + 1, self.height):
            dist_down += 1
            if (self.agent_x, y) in walls:
                break
        norm_ray_down = dist_down / h

        dist_left = 0
        for x in range(self.agent_x - 1, -1, -1):
            dist_left += 1
            if (x, self.agent_y) in walls:
                break
        norm_ray_left = dist_left / w

        dist_right = 0
        for x in range(self.agent_x + 1, self.width):
            dist_right += 1
            if (x, self.agent_y) in walls:
                break
        norm_ray_right = dist_right / w

        base_16 = [
            self.agent_x / w,
            self.agent_y / h,
            gx / w,
            gy / h,
            dx / w,
            dy / h,
            euc_dist,
            man_dist,
            wall_up,
            wall_down,
            wall_left,
            wall_right,
            norm_ray_up,
            norm_ray_down,
            norm_ray_left,
            norm_ray_right,
        ]

        if target_dim <= 16:
            vec = np.array(base_16, dtype=np.float32)
        else:
            # 32-dim features: 4 diagonal raycasts + 4 dynamic obstacle features + 8-cell 3x3 local patch
            def ray_dist(step_x: int, step_y: int) -> float:
                cx, cy = self.agent_x, self.agent_y
                steps = 0
                while True:
                    cx += step_x
                    cy += step_y
                    steps += 1
                    if (cx, cy) in walls or cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                        break
                return steps / diag

            ray_nw = ray_dist(-1, -1)
            ray_ne = ray_dist(1, -1)
            ray_sw = ray_dist(-1, 1)
            ray_se = ray_dist(1, 1)

            if self.dynamic_wall_pos is not None:
                dyn_dx = (self.dynamic_wall_pos[0] - self.agent_x) / w
                dyn_dy = (self.dynamic_wall_pos[1] - self.agent_y) / h
                dyn_dist = math.hypot(self.dynamic_wall_pos[0] - self.agent_x, self.dynamic_wall_pos[1] - self.agent_y) / diag
                dyn_active = 1.0
            else:
                dyn_dx, dyn_dy, dyn_dist, dyn_active = 0.0, 0.0, 1.0, 0.0

            offsets_3x3 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
            patch_3x3 = [
                1.0 if (self.agent_x + ox, self.agent_y + oy) in walls or (
                    self.agent_x + ox < 0 or self.agent_x + ox >= self.width or self.agent_y + oy < 0 or self.agent_y + oy >= self.height
                ) else 0.0
                for ox, oy in offsets_3x3
            ]

            feat_32 = base_16 + [
                ray_nw, ray_ne, ray_sw, ray_se,
                dyn_dx, dyn_dy, dyn_dist, dyn_active,
            ] + patch_3x3

            if target_dim <= 32:
                vec = np.array(feat_32, dtype=np.float32)
            else:
                # 64-dim features: 16-cell 5x5 ring + 8 goal projections + 8 corridor clearance
                patch_5x5_ring = []
                for oy in range(-2, 3):
                    for ox in range(-2, 3):
                        if max(abs(ox), abs(oy)) == 2:
                            is_blocked = 1.0 if (self.agent_x + ox, self.agent_y + oy) in walls or (
                                self.agent_x + ox < 0 or self.agent_x + ox >= self.width or self.agent_y + oy < 0 or self.agent_y + oy >= self.height
                            ) else 0.0
                            patch_5x5_ring.append(is_blocked)

                hyp = math.hypot(dx, dy)
                ugx = (dx / hyp) if hyp > 1e-6 else 0.0
                ugy = (dy / hyp) if hyp > 1e-6 else 0.0
                ray_dirs = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
                goal_projections = [
                    (rx * ugx + ry * ugy) / math.hypot(rx, ry)
                    for rx, ry in ray_dirs
                ]
                corridor_depths = [norm_ray_up, norm_ray_down, norm_ray_left, norm_ray_right, ray_nw, ray_ne, ray_sw, ray_se]

                feat_64 = feat_32 + patch_5x5_ring + goal_projections + corridor_depths

                if target_dim <= 64:
                    vec = np.array(feat_64, dtype=np.float32)
                else:
                    # 128-dim features: 24-cell 7x7 ring + 16-ray circular rangefinder + 24 Fourier encodings
                    patch_7x7_ring = []
                    for oy in range(-3, 4):
                        for ox in range(-3, 4):
                            if max(abs(ox), abs(oy)) == 3:
                                is_blocked = 1.0 if (self.agent_x + ox, self.agent_y + oy) in walls or (
                                    self.agent_x + ox < 0 or self.agent_x + ox >= self.width or self.agent_y + oy < 0 or self.agent_y + oy >= self.height
                                ) else 0.0
                                patch_7x7_ring.append(is_blocked)

                    # 16-ray circular rangefinder
                    lidar_16 = []
                    for k in range(16):
                        angle = k * (2.0 * math.pi / 16.0)
                        cos_a, sin_a = math.cos(angle), math.sin(angle)
                        step = 0.0
                        while step < diag:
                            step += 0.5
                            sx = int(round(self.agent_x + cos_a * step))
                            sy = int(round(self.agent_y + sin_a * step))
                            if (sx, sy) in walls or sx < 0 or sx >= self.width or sy < 0 or sy >= self.height:
                                break
                        lidar_16.append(min(1.0, step / diag))

                    # 24 harmonic Fourier encodings
                    fourier_24 = []
                    norm_ax = self.agent_x / w
                    norm_ay = self.agent_y / h
                    for freq in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
                        fourier_24.append(math.sin(freq * math.pi * norm_ax))
                        fourier_24.append(math.cos(freq * math.pi * norm_ax))
                        fourier_24.append(math.sin(freq * math.pi * norm_ay))
                        fourier_24.append(math.cos(freq * math.pi * norm_ay))

                    feat_128 = feat_64 + patch_7x7_ring + lidar_16 + fourier_24
                    vec = np.array(feat_128, dtype=np.float32)

        if self.noise_level > 0.0:
            noise = np.random.normal(0.0, self.noise_level, size=vec.shape).astype(np.float32)
            vec = vec + noise

        return vec

    def render_ascii(self) -> str:
        """Render the grid as an ASCII string."""
        lines = []
        walls = self.current_walls
        for y in range(self.height):
            row_chars = []
            for x in range(self.width):
                if (x, y) == (self.agent_x, self.agent_y):
                    row_chars.append("A")
                elif (x, y) == (self.goal_x, self.goal_y):
                    row_chars.append("G")
                elif (x, y) in walls:
                    row_chars.append("#")
                else:
                    row_chars.append(" ")
            lines.append("".join(row_chars))
        return "\n".join(lines)


def create_random_gridworld(
    width: int = 12,
    height: int = 7,
    wall_prob: float = 0.2,
    feature_dim: int = 16,
    seed: Optional[int] = None,
) -> GridWorld:
    """Generate a random GridWorld map guaranteed to have an open path."""
    rng = random.Random(seed)
    max_retries = 100

    for _ in range(max_retries):
        layout = []
        for y in range(height):
            row = []
            for x in range(width):
                if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                    row.append("#")
                elif rng.random() < wall_prob:
                    row.append("#")
                else:
                    row.append(" ")
            layout.append(row)

        # Set S and G
        start = (1, 1)
        goal = (width - 2, height - 2)
        layout[start[1]][start[0]] = "S"
        layout[goal[1]][goal[0]] = "G"

        # Check BFS path
        from collections import deque
        q = deque([start])
        visited = {start}
        found = False
        while q:
            cx, cy = q.popleft()
            if (cx, cy) == goal:
                found = True
                break
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if layout[ny][nx] != "#" and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))

        if found:
            map_str = ["".join(row) for row in layout]
            return GridWorld(map_layout=map_str, feature_dim=feature_dim, seed=seed)

    # Fallback to default
    return GridWorld(feature_dim=feature_dim, seed=seed)
