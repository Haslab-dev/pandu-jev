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
        # 2 (agent_xy) + 2 (goal_xy) + 2 (delta_xy) + 2 (distances) + 4 (immediate wall) + 4 (raycast wall) = 16
        return 16

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

    def get_feature_vector(self) -> np.ndarray:
        """Normalized 16-dimensional feature vector:

        [0]: agent_x / width
        [1]: agent_y / height
        [2]: goal_x / width (or agent_x if unobserved)
        [3]: goal_y / height (or agent_y if unobserved)
        [4]: relative dx / width
        [5]: relative dy / height
        [6]: normalized euclidean distance to goal
        [7]: normalized manhattan distance to goal
        [8..11]: immediate wall flags (UP, DOWN, LEFT, RIGHT: 1 if wall, 0 otherwise)
        [12..15]: raycast normalized distance to wall (UP, DOWN, LEFT, RIGHT)
        """
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
        # Ray UP
        dist_up = 0
        for y in range(self.agent_y - 1, -1, -1):
            dist_up += 1
            if (self.agent_x, y) in walls:
                break
        norm_ray_up = dist_up / h

        # Ray DOWN
        dist_down = 0
        for y in range(self.agent_y + 1, self.height):
            dist_down += 1
            if (self.agent_x, y) in walls:
                break
        norm_ray_down = dist_down / h

        # Ray LEFT
        dist_left = 0
        for x in range(self.agent_x - 1, -1, -1):
            dist_left += 1
            if (x, self.agent_y) in walls:
                break
        norm_ray_left = dist_left / w

        # Ray RIGHT
        dist_right = 0
        for x in range(self.agent_x + 1, self.width):
            dist_right += 1
            if (x, self.agent_y) in walls:
                break
        norm_ray_right = dist_right / w

        vec = np.array(
            [
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
            ],
            dtype=np.float32,
        )

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
            return GridWorld(map_layout=map_str, seed=seed)

    # Fallback to default
    return GridWorld(seed=seed)
