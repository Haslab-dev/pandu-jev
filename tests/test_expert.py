"""Tests for A* pathfinding expert oracle."""

import pytest
from env.gridworld import GridWorld, Action
from expert.astar import AStarExpert


def test_astar_pathfinding():
    env = GridWorld(seed=42)
    expert = AStarExpert()
    path = expert.find_path(
        start=(1, 1),
        goal=(env.goal_x, env.goal_y),
        walls=env.current_walls,
        width=env.width,
        height=env.height,
    )
    assert path is not None
    assert path[0] == (1, 1)
    assert path[-1] == (env.goal_x, env.goal_y)

    action = expert.get_action(env)
    assert action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
