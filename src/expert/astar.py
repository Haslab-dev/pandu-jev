"""A* Expert solver and trajectory dataset generation for GridWorld."""

from heapq import heappush, heappop
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from env.gridworld import Action, GridWorld


class AStarExpert:
    """Optimal pathfinding expert using A* with Manhattan distance heuristic."""

    def __init__(self, heuristic_weight: float = 1.0):
        self.heuristic_weight = heuristic_weight

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return self.heuristic_weight * (abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        width: int,
        height: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """Compute the shortest collision-free path from start to goal."""
        if start == goal:
            return [start]

        # Priority queue stores (f_score, counter, current_node)
        counter = 0
        frontier = [(self.heuristic(start, goal), counter, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed_set: Set[Tuple[int, int]] = set()

        directions = [
            (0, -1),  # UP
            (0, 1),   # DOWN
            (-1, 0),  # LEFT
            (1, 0),   # RIGHT
        ]

        while frontier:
            _, _, current = heappop(frontier)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            cx, cy = current
            for dx, dy in directions:
                neighbor = (cx + dx, cy + dy)
                nx, ny = neighbor

                if nx < 0 or nx >= width or ny < 0 or ny >= height or neighbor in walls:
                    continue

                tentative_g = g_score[current] + 1.0
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    counter += 1
                    heappush(frontier, (f, counter, neighbor))

        return None

    def get_action(self, env: GridWorld) -> Optional[Action]:
        """Returns the optimal next Action for the current environment state."""
        start = (env.agent_x, env.agent_y)
        goal = (env.goal_x, env.goal_y)

        if start == goal:
            return None

        path = self.find_path(
            start=start,
            goal=goal,
            walls=env.current_walls,
            width=env.width,
            height=env.height,
        )

        if not path or len(path) < 2:
            return None

        next_pos = path[1]
        dx = next_pos[0] - start[0]
        dy = next_pos[1] - start[1]

        if dx == 0 and dy == -1:
            return Action.UP
        elif dx == 0 and dy == 1:
            return Action.DOWN
        elif dx == -1 and dy == 0:
            return Action.LEFT
        elif dx == 1 and dy == 0:
            return Action.RIGHT

        return None
