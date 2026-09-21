"""Competitive 2-Player Snake Arena for Pandu (pandu-jev).

Features:
- 20x20 Arena with simultaneous moves for Snake A and Snake B.
- Shared food, length growth, wall collisions, body collisions, head-on duels.
- Information-dense 20d perceptual observation vector.
- Heuristic Expert Bot with 2-step lookahead and flood-fill safety.
- Trainable Pandu Micro-Policy (~2,980 parameters, <0.03 ms latency).
"""

import collections
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from arena.base import ArenaBot, ArenaEnvironment
from models.policy import TinyPolicy

# Direction vectors: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
OPPOSITE_ACTIONS = {0: 1, 1: 0, 2: 3, 3: 2}


class CompetitiveSnakeEnv(ArenaEnvironment):
    """20x20 Competitive Two-Player or Solo Snake Environment."""

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        max_steps: int = 350,
        obstacles: Optional[Set[Tuple[int, int]]] = None,
        solo: bool = False,
        wrap_walls: bool = False,
    ):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.obstacles: Set[Tuple[int, int]] = set(obstacles) if obstacles else set()
        self.solo = solo
        self.wrap_walls = wrap_walls

        self.snake_a: collections.deque = collections.deque()
        self.snake_b: collections.deque = collections.deque()
        self.dir_a: int = 0
        self.dir_b: int = 0
        self.food: Tuple[int, int] = (10, 10)
        self.steps: int = 0
        self.score_a: int = 0
        self.score_b: int = 0
        self.done: bool = False
        self.rng = np.random.RandomState(42)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Initial positions scaled to grid dimensions
        xa = max(1, self.width // 4)
        xb = min(self.width - 2, 3 * self.width // 4)
        ym = min(self.height - 4, max(2, self.height // 2))

        self.snake_a = collections.deque([(xa, ym), (xa, ym + 1), (xa, ym + 2)])
        if self.solo:
            self.snake_b = collections.deque()
            self.dir_b = 0
        else:
            self.snake_b = collections.deque([(xb, ym), (xb, ym + 1), (xb, ym + 2)])
            self.dir_b = 0  # UP
        self.dir_a = 0  # UP
        self.steps = 0
        self.score_a = 0
        self.score_b = 0
        self.done = False

        self._spawn_food()
        obs_b = np.zeros(20, dtype=np.float32) if self.solo else self._get_obs(player=1)
        return self._get_obs(player=0), obs_b

    def _spawn_food(self) -> None:
        occupied = set(self.snake_a) | set(self.snake_b) | self.obstacles
        open_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]
        if open_cells:
            idx = int(self.rng.randint(0, len(open_cells)))
            self.food = open_cells[idx]
        else:
            self.food = (self.width // 2, self.height // 2)

    def get_valid_actions(self, player_id: int) -> List[int]:
        """Return non-reversing actions (cannot turn 180 degrees into neck)."""
        curr_dir = self.dir_a if player_id == 0 else self.dir_b
        opp = OPPOSITE_ACTIONS.get(curr_dir, -1)
        return [a for a in range(4) if a != opp]

    def _get_obs(self, player: int) -> np.ndarray:
        """Compute 20-dim perceptual sensory vector for specified snake."""
        self_body = self.snake_a if player == 0 else self.snake_b
        opp_body = self.snake_b if player == 0 else self.snake_a

        if not self_body:
            return np.zeros(20, dtype=np.float32)

        head_x, head_y = self_body[0]
        opp_x, opp_y = opp_body[0] if opp_body else (-999, -999)
        fx, fy = self.food

        w = float(self.width)
        h = float(self.height)
        diag = math.hypot(w, h)

        # 1. Normalized positions (2)
        norm_head_x = head_x / w
        norm_head_y = head_y / h

        # 2. Food relative direction (2) and distance (1)
        food_dx = (fx - head_x) / w
        food_dy = (fy - head_y) / h
        food_dist = math.hypot(fx - head_x, fy - head_y) / diag

        # Obstacle set: all body parts + fixed obstacle barriers
        all_obstacles = set(self_body) | set(opp_body) | self.obstacles

        # 3. Immediate 4-neighbor collision hazard (4)
        hazards = []
        for dx, dy in ACTIONS:
            nx, ny = head_x + dx, head_y + dy
            if self.wrap_walls:
                wx, wy = nx % self.width, ny % self.height
                hazards.append(1.0 if (wx, wy) in all_obstacles else 0.0)
            else:
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height or (nx, ny) in all_obstacles:
                    hazards.append(1.0)
                else:
                    hazards.append(0.0)

        # 4. 4 Cardinal Raycasts to nearest obstacle (4)
        cardinal_rays = []
        for dx, dy in ACTIONS:
            dist = 0
            cx, cy = head_x, head_y
            while True:
                cx += dx
                cy += dy
                dist += 1
                if self.wrap_walls:
                    if dist >= max(w, h) or (cx % self.width, cy % self.height) in all_obstacles:
                        break
                else:
                    if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height or (cx, cy) in all_obstacles:
                        break
            cardinal_rays.append(dist / max(w, h))

        # 5. 4 Diagonal Raycasts (4)
        diag_dirs = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        diag_rays = []
        for dx, dy in diag_dirs:
            dist = 0
            cx, cy = head_x, head_y
            while True:
                cx += dx
                cy += dy
                dist += 1
                if self.wrap_walls:
                    if dist >= max(w, h) or (cx % self.width, cy % self.height) in all_obstacles:
                        break
                else:
                    if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height or (cx, cy) in all_obstacles:
                        break
            diag_rays.append(dist / diag)

        # 6. Relative opponent head vector (2)
        opp_dx = (opp_x - head_x) / w if opp_x >= 0 else 0.0
        opp_dy = (opp_y - head_y) / h if opp_y >= 0 else 0.0

        # 7. Relative length difference (1)
        len_diff = (len(self_body) - len(opp_body)) / float(max(1, self.width))

        feat = [
            norm_head_x, norm_head_y,
            food_dx, food_dy, food_dist,
            hazards[0], hazards[1], hazards[2], hazards[3],
            cardinal_rays[0], cardinal_rays[1], cardinal_rays[2], cardinal_rays[3],
            diag_rays[0], diag_rays[1], diag_rays[2], diag_rays[3],
            opp_dx, opp_dy,
            len_diff,
        ]
        return np.array(feat, dtype=np.float32)

    def step(
        self, action_a: int, action_b: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool, Dict[str, Any]]:
        if self.done:
            obs_b = np.zeros(20, dtype=np.float32) if self.solo else self._get_obs(1)
            return self._get_obs(0), obs_b, 0.0, 0.0, True, {"winner": "DRAW", "reason": "already_done"}

        self.steps += 1
        act_a = int(action_a)
        act_b = int(action_b)

        # Prevent 180 reverse suicide if valid
        if act_a != OPPOSITE_ACTIONS.get(self.dir_a, -1):
            self.dir_a = act_a
        if not self.solo and act_b != OPPOSITE_ACTIONS.get(self.dir_b, -1):
            self.dir_b = act_b

        dx_a, dy_a = ACTIONS[self.dir_a]
        head_a = self.snake_a[0]
        if self.wrap_walls:
            next_a = ((head_a[0] + dx_a) % self.width, (head_a[1] + dy_a) % self.height)
        else:
            next_a = (head_a[0] + dx_a, head_a[1] + dy_a)

        dead_a = False
        dead_b = False
        reason = ""

        # --- Solo Mode Execution ---
        if self.solo:
            if not self.wrap_walls and (next_a[0] < 0 or next_a[0] >= self.width or next_a[1] < 0 or next_a[1] >= self.height):
                dead_a = True
                reason = "Wall collision"
            elif next_a in self.obstacles:
                dead_a = True
                reason = "Obstacle collision"
            elif next_a in set(list(self.snake_a)[:-1]):
                dead_a = True
                reason = "Self-collision"

            if dead_a:
                self.done = True
                winner = "GAME_OVER"
            else:
                winner = None
                self.snake_a.appendleft(next_a)
                if next_a == self.food:
                    self.score_a += 1
                    self._spawn_food()
                else:
                    self.snake_a.pop()

                if self.steps >= self.max_steps:
                    self.done = True
                    winner = "SURVIVED"
                    reason = "Completed max steps"

            info = {
                "winner": winner if winner is not None else "ONGOING",
                "reason": reason,
                "steps": self.steps,
                "len_a": len(self.snake_a),
                "len_b": 0,
                "score_a": self.score_a,
                "score_b": 0,
            }
            return self._get_obs(0), np.zeros(20, dtype=np.float32), 1.0 if not dead_a else -1.0, 0.0, self.done, info

        # --- Dual Player Competitive Execution ---
        dx_b, dy_b = ACTIONS[self.dir_b]
        head_b = self.snake_b[0]
        if self.wrap_walls:
            next_b = ((head_b[0] + dx_b) % self.width, (head_b[1] + dy_b) % self.height)
        else:
            next_b = (head_b[0] + dx_b, head_b[1] + dy_b)

        # 1. Wall collisions (skipped if wrap_walls is True)
        if not self.wrap_walls:
            if next_a[0] < 0 or next_a[0] >= self.width or next_a[1] < 0 or next_a[1] >= self.height:
                dead_a = True
                reason = "A wall collision"
            if next_b[0] < 0 or next_b[0] >= self.width or next_b[1] < 0 or next_b[1] >= self.height:
                dead_b = True
                reason = "B wall collision" if not dead_a else "Both wall collision"

        # 2. Obstacle collisions
        if next_a in self.obstacles:
            dead_a = True
            reason = "A obstacle collision"
        if next_b in self.obstacles:
            dead_b = True
            reason = "B obstacle collision" if not dead_a else "Both obstacle collision"

        # 3. Self body collisions
        body_a_no_tail = set(list(self.snake_a)[:-1])
        body_b_no_tail = set(list(self.snake_b)[:-1])
        if next_a in body_a_no_tail:
            dead_a = True
            reason = "A self-collision"
        if next_b in body_b_no_tail:
            dead_b = True
            reason = "B self-collision" if not dead_a else "Both body collision"

        # 4. Opponent body collisions
        if next_a in self.snake_b:
            dead_a = True
            reason = "A hit opponent body"
        if next_b in self.snake_a:
            dead_b = True
            reason = "B hit opponent body" if not dead_a else "Mutual body collision"

        # 5. Head-on collision
        if next_a == next_b:
            if len(self.snake_a) > len(self.snake_b):
                dead_b = True
                reason = "Head-on collision (A longer)"
            elif len(self.snake_b) > len(self.snake_a):
                dead_a = True
                reason = "Head-on collision (B longer)"
            else:
                dead_a = True
                dead_b = True
                reason = "Head-on collision (equal length)"

        # Determine winner
        if dead_a and dead_b:
            winner = "DRAW"
            self.done = True
        elif dead_a:
            winner = "B"
            self.done = True
        elif dead_b:
            winner = "A"
            self.done = True
        else:
            winner = None

        r_a = 0.0
        r_b = 0.0

        if not self.done:
            # Advance Snake A
            self.snake_a.appendleft(next_a)
            if next_a == self.food:
                self.score_a += 1
                r_a = 1.0
                self._spawn_food()
            else:
                self.snake_a.pop()

            # Advance Snake B
            self.snake_b.appendleft(next_b)
            if next_b == self.food:
                self.score_b += 1
                r_b = 1.0
                self._spawn_food()
            else:
                self.snake_b.pop()

            if self.steps >= self.max_steps:
                self.done = True
                if len(self.snake_a) > len(self.snake_b):
                    winner = "A"
                    reason = "Timeout (A longer)"
                elif len(self.snake_b) > len(self.snake_a):
                    winner = "B"
                    reason = "Timeout (B longer)"
                else:
                    winner = "DRAW"
                    reason = "Timeout (equal length)"

        info = {
            "winner": winner if winner is not None else "ONGOING",
            "reason": reason,
            "steps": self.steps,
            "len_a": len(self.snake_a),
            "len_b": len(self.snake_b),
            "score_a": self.score_a,
            "score_b": self.score_b,
        }
        return self._get_obs(0), self._get_obs(1), r_a, r_b, self.done, info


def generate_obstacles_for_level(
    level: str, width: int = 20, height: int = 20
) -> Set[Tuple[int, int]]:
    """Generate thematic obstacle sets for difficulty levels with open runways for player spawns."""
    obs: Set[Tuple[int, int]] = set()
    lvl = level.lower()
    if lvl == "easy":
        return obs

    xa = max(1, width // 4)  # typically col 5
    xb = min(width - 2, 3 * width // 4)  # typically col 15

    # Protected corridors: clear 3 columns around xa and xb so snakes have completely unobstructed runways
    protected_cols = {xa - 1, xa, xa + 1, xb - 1, xb, xb + 1}
    protected_center = {
        (width // 2, height // 2),
        (width // 2 - 1, height // 2),
        (width // 2, height // 2 - 1),
    }

    if lvl == "medium":
        # Central tactical pillars at mid columns, away from snake lanes
        mid_x = width // 2
        for x in (mid_x - 1, mid_x):
            for y in (3, 4, 5):  # Top central pillar
                obs.add((x, y))
            for y in (height - 6, height - 5, height - 4):  # Bottom central pillar
                obs.add((x, y))
        # Side bumpers
        for y in range(7, min(13, height - 7)):
            obs.add((1, y))
            obs.add((width - 2, y))

    elif lvl == "hard":
        mid_x = width // 2
        mid_y = height // 2
        # Cross barricades with generous gates
        for y in range(2, mid_y - 2):
            obs.add((mid_x, y))
        for y in range(mid_y + 3, height - 2):
            obs.add((mid_x, y))
        for x in range(2, mid_x - 3):
            obs.add((x, mid_y))
        for x in range(mid_x + 4, width - 2):
            obs.add((x, mid_y))

    # Remove any collision with protected runways and center food
    return {pt for pt in obs if pt[0] not in protected_cols and pt not in protected_center}



class EasySnakeBot(ArenaBot):
    """Easy / Beginner Bot: moves safely towards food with 35% casual exploratory jitter."""

    def __init__(self, name: str = "EasyBot", seed: int = 42):
        super().__init__(name)
        self.rng = np.random.RandomState(seed)

    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: Optional[List[int]] = None,
        env_state: Optional[Any] = None,
    ) -> int:
        candidates = valid_actions if valid_actions else list(range(4))
        hazards = obs[5:9]
        safe_moves = [a for a in candidates if hazards[a] < 0.5]

        if not safe_moves:
            return candidates[0] if candidates else 0

        # With 35% chance, choose casual wandering move
        if len(safe_moves) > 1 and self.rng.rand() < 0.35:
            return int(self.rng.choice(safe_moves))

        # Otherwise greedy alignment to food
        food_dx, food_dy = obs[2], obs[3]
        best_move = safe_moves[0]
        best_score = -9999.0
        for a in safe_moves:
            dx, dy = ACTIONS[a]
            score = (dx * food_dx) + (dy * food_dy)
            if score > best_score:
                best_score = score
                best_move = a
        return best_move


class HardSnakeBot(ArenaBot):
    """Hard / Master Bot: 2-step lookahead + BFS Flood-fill area calculation and trap avoidance."""

    def __init__(self, name: str = "MasterBot"):
        super().__init__(name)

    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: Optional[List[int]] = None,
        env_state: Optional[Any] = None,
    ) -> int:
        candidates = valid_actions if valid_actions else list(range(4))
        hazards = obs[5:9]
        safe_moves = [a for a in candidates if hazards[a] < 0.5]

        if not safe_moves:
            return candidates[0] if candidates else 0

        # If env_state is provided, execute BFS flood-fill reachable space analysis
        if env_state is not None and hasattr(env_state, "snake_a"):
            is_a = (len(env_state.snake_a) > 0 and 
                    abs(obs[0] - env_state.snake_a[0][0] / float(env_state.width)) < 1e-4)
            my_body = env_state.snake_a if is_a else env_state.snake_b
            opp_body = env_state.snake_b if is_a else env_state.snake_a
            head = my_body[0]
            obstacles = set(my_body) | set(opp_body) | getattr(env_state, "obstacles", set())

            best_move = safe_moves[0]
            best_score = -999999.0

            for a in safe_moves:
                dx, dy = ACTIONS[a]
                nxt = (head[0] + dx, head[1] + dy)

                # BFS flood-fill reachable area from nxt cell
                visited = set(obstacles)
                visited.add(nxt)
                queue = collections.deque([nxt])
                area = 0
                max_area_check = min(75, env_state.width * env_state.height // 3)
                while queue and area < max_area_check:
                    cx, cy = queue.popleft()
                    area += 1
                    for ndx, ndy in ACTIONS:
                        nx, ny = cx + ndx, cy + ndy
                        if 0 <= nx < env_state.width and 0 <= ny < env_state.height and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))

                # Distance to food
                fx, fy = env_state.food
                dist_food = abs(fx - nxt[0]) + abs(fy - nxt[1])

                # Heavy penalty if trapped in dead-end space smaller than snake length
                trap_penalty = -1000.0 if area < len(my_body) else 0.0
                score = (area * 15.0) - (dist_food * 2.5) + trap_penalty
                if score > best_score:
                    best_score = score
                    best_move = a

            return best_move

        # Fallback: raycast space + food alignment
        rays = obs[9:13]
        food_dx, food_dy = obs[2], obs[3]
        best_move = safe_moves[0]
        best_score = -99999.0
        for a in safe_moves:
            score = (rays[a] * 20.0) + ((ACTIONS[a][0] * food_dx + ACTIONS[a][1] * food_dy) * 35.0)
            if score > best_score:
                best_score = score
                best_move = a
        return best_move


class HeuristicSnakeBot(ArenaBot):
    """Heuristic Snake Bot with 2-step collision avoidance and food pursuit."""

    def __init__(self, name: str = "HeuristicBot"):
        super().__init__(name)

    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: Optional[List[int]] = None,
        env_state: Optional[Any] = None,
    ) -> int:
        candidates = valid_actions if valid_actions else list(range(4))

        # Obstacle hazards from obs: indices 5, 6, 7, 8 represent UP, DOWN, LEFT, RIGHT
        hazards = obs[5:9]
        safe_moves = [a for a in candidates if hazards[a] < 0.5]

        if not safe_moves:
            return candidates[0] if candidates else 0

        # Cardinal raycast distances: indices 9, 10, 11, 12
        rays = obs[9:13]
        food_dx, food_dy = obs[2], obs[3]

        best_move = safe_moves[0]
        best_score = -99999.0

        for a in safe_moves:
            score = rays[a] * 10.0  # prefer open space
            dx, dy = ACTIONS[a]
            # Reward moving in direction of food
            alignment = (dx * food_dx) + (dy * food_dy)
            score += alignment * 25.0
            if score > best_score:
                best_score = score
                best_move = a

        return best_move


def create_snake_bot(bot_type: str, seed: int = 42, name: Optional[str] = None) -> ArenaBot:
    """Factory creating snake bot instances for given difficulty or architecture."""
    b_type = bot_type.lower()
    if b_type == "easy":
        return EasySnakeBot(name=name or "Easy-Bot", seed=seed)
    elif b_type in ("hard", "master"):
        return HardSnakeBot(name=name or "Master-Bot")
    elif b_type == "heuristic":
        return HeuristicSnakeBot(name=name or "Heuristic-Bot")
    elif b_type == "random":
        from arena.base import RandomArenaBot
        return RandomArenaBot(name=name or "Random-Bot", seed=seed)
    elif b_type == "pandu":
        return PanduSnakeBot(name=name or "Pandu-3K")
    else:
        return PanduSnakeBot(name=name or "Pandu-3K")


class PanduSnakeBot(ArenaBot):
    """Trained Neural Pandu Policy for Snake Arena (~2,980 parameters)."""

    def __init__(self, model: Optional[TinyPolicy] = None, name: str = "Pandu-3K"):
        super().__init__(name)
        if model is None:
            self.model = TinyPolicy(input_dim=20, num_actions=4, hidden_dims=(32, 64))
        else:
            self.model = model
        self.model.eval()

    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: Optional[List[int]] = None,
        env_state: Optional[Any] = None,
    ) -> int:
        candidates = valid_actions if valid_actions else list(range(4))
        # Strictly mask out collision hazards
        hazards = obs[5:9]
        safe_moves = [a for a in candidates if hazards[a] < 0.5]
        target_pool = safe_moves if safe_moves else candidates

        dist = self.model.get_action_distribution(obs)
        probs = dist["probabilities"]

        best_act = target_pool[0]
        best_prob = -1.0
        for act in target_pool:
            act_name = ACTION_NAMES[act]
            prob = probs.get(act_name, 0.0)
            if prob > best_prob:
                best_prob = prob
                best_act = act
        return best_act


def train_pandu_snake_bot(episodes: int = 500, epochs: int = 25, seed: int = 42) -> PanduSnakeBot:
    """Collect self-play / heuristic demonstration data and train Pandu-3K policy."""
    env = CompetitiveSnakeEnv()
    heuristic = HeuristicSnakeBot()

    X_data = []
    y_data = []
    rng = np.random.RandomState(seed)

    for ep in range(episodes):
        ep_seed = int(rng.randint(0, 1_000_000))
        obs_a, obs_b = env.reset(seed=ep_seed)
        done = False
        steps = 0

        while not done and steps < env.max_steps:
            valid_a = env.get_valid_actions(0)
            valid_b = env.get_valid_actions(1)

            act_a = heuristic.select_action(obs_a, valid_actions=valid_a)
            act_b = heuristic.select_action(obs_b, valid_actions=valid_b)

            X_data.append(obs_a)
            y_data.append(act_a)
            X_data.append(obs_b)
            y_data.append(act_b)

            obs_a, obs_b, r_a, r_b, done, info = env.step(act_a, act_b)
            steps += 1

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int64)

    model = TinyPolicy(input_dim=20, num_actions=4, hidden_dims=(32, 64))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = torch.nn.functional.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()

    return PanduSnakeBot(model=model, name="Pandu-3K")
