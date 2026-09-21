"""Tests for Multi-Agent Arena (Snake, Micro Chess, Evolution)."""

import numpy as np
import pytest

from arena.base import ArenaMatch, ArenaTournament, RandomArenaBot
from arena.chess import CompetitiveChessEnv, HeuristicChessBot, PanduChessBot, board_to_feature_vector
from arena.evolution import EvolutionaryPolicyLeague
from arena.snake import CompetitiveSnakeEnv, HeuristicSnakeBot, PanduSnakeBot


def test_snake_environment_mechanics():
    env = CompetitiveSnakeEnv(width=10, height=10)
    obs_a, obs_b = env.reset(seed=123)

    assert obs_a.shape == (20,)
    assert obs_b.shape == (20,)

    # Valid moves should exclude reversing 180 deg
    valid_a = env.get_valid_actions(0)
    assert len(valid_a) == 3

    # Step simulation
    next_obs_a, next_obs_b, r_a, r_b, done, info = env.step(valid_a[0], valid_a[0])
    assert not done
    assert next_obs_a.shape == (20,)


def test_snake_bots_execution():
    env = CompetitiveSnakeEnv(width=12, height=12)
    obs_a, obs_b = env.reset(seed=42)

    bot_heuristic = HeuristicSnakeBot("Heuristic")
    bot_random = RandomArenaBot("Random")
    bot_pandu = PanduSnakeBot(name="Pandu-Test")

    act_h = bot_heuristic.select_action(obs_a, env.get_valid_actions(0))
    act_r = bot_random.select_action(obs_b, env.get_valid_actions(1))
    act_p = bot_pandu.select_action(obs_a, env.get_valid_actions(0))

    assert 0 <= act_h < 4
    assert 0 <= act_r < 4
    assert 0 <= act_p < 4


def test_chess_environment_and_features():
    env = CompetitiveChessEnv(max_plies=20)
    obs_w, obs_b = env.reset(seed=42)

    assert obs_w.shape == (68,)
    assert obs_b.shape == (68,)

    # White has 20 opening legal moves in chess
    valid_w = env.get_valid_actions(0)
    assert len(valid_w) == 20

    # Black has 0 legal moves on White's turn
    valid_b = env.get_valid_actions(1)
    assert len(valid_b) == 0

    # Play move
    first_move = valid_w[0]
    next_w, next_b, r_a, r_b, done, info = env.step(first_move, None)
    assert not done
    assert info["turn"] == "BLACK"

    # Now Black has 20 legal moves
    valid_b_after = env.get_valid_actions(1)
    assert len(valid_b_after) == 20


def test_chess_bots_decision():
    env = CompetitiveChessEnv()
    obs_w, _ = env.reset(seed=42)

    bot_pandu = PanduChessBot(name="Pandu-Chess")
    bot_heur = HeuristicChessBot(name="Heur-Chess")

    legals = env.get_valid_actions(0)
    move_p = bot_pandu.select_action(obs_w, legals)
    move_h = bot_heur.select_action(obs_w, legals)

    assert move_p in legals
    assert move_h in legals


def test_evolutionary_league_step():
    league = EvolutionaryPolicyLeague(population_size=4, input_dim=20, num_actions=4, hidden_dims=(16, 16))
    stats = league.run_generation(games_per_pair=2)
    assert "best_fitness" in stats
    assert "avg_fitness" in stats
    assert len(league.population) == 4


def test_chess_gui_engine_mechanics():
    from arena.chess_gui import ChessGameEngine

    engine = ChessGameEngine(white_type="pandu", black_type="heuristic")
    state = engine.get_state()

    assert state["turn"] == "white"
    assert not state["is_game_over"]
    assert len(state["legal_moves"]) == 20
    assert state["material"]["diff"] == 0.0

    # Execute auto step (Pandu plays White)
    next_state = engine.step()
    assert next_state["step_count"] == 1
    assert next_state["turn"] == "black"
    assert next_state["last_move"] is not None
    assert len(next_state["move_history"]) == 1

    # Execute next auto step (Heuristic plays Black)
    next_state2 = engine.step()
    assert next_state2["step_count"] == 2
    assert next_state2["turn"] == "white"
    assert len(next_state2["move_history"]) == 2

    # Reset
    reset_state = engine.reset(white_type="human", black_type="pandu")
    assert reset_state["step_count"] == 0
    assert reset_state["is_human_turn"] is True


def test_chess_gui_human_and_bot_turns():
    from arena.chess_gui import ChessGameEngine

    engine = ChessGameEngine(white_type="human", black_type="pandu")
    state = engine.get_state()
    assert state["is_human_turn"] is True

    # Attempt illegal move
    ok, msg, _ = engine.make_human_move("e2e5")
    assert not ok

    # Legal human move: 1. e2e4
    ok, msg, state_after_human = engine.make_human_move("e2e4")
    assert ok
    assert state_after_human["turn"] == "black"
    assert state_after_human["is_human_turn"] is False
    assert state_after_human["last_move"]["san"] == "e4"

    # Bot replies
    state_after_bot = engine.step()
    assert state_after_bot["turn"] == "white"
    assert state_after_bot["is_human_turn"] is True
    assert state_after_bot["step_count"] == 2

    # Verify move history contains FEN for replay / preview scrubber
    assert len(state_after_bot["move_history"]) == 2
    for entry in state_after_bot["move_history"]:
        assert "fen" in entry
        assert " " in entry["fen"]
        assert "ply" in entry
        assert "san" in entry


def test_snake_difficulty_levels_and_bots():
    from arena.snake import (
        CompetitiveSnakeEnv,
        EasySnakeBot,
        HardSnakeBot,
        generate_obstacles_for_level,
    )

    # Obstacles per level
    obs_easy = generate_obstacles_for_level("easy")
    obs_med = generate_obstacles_for_level("medium")
    obs_hard = generate_obstacles_for_level("hard")

    assert len(obs_easy) == 0
    assert len(obs_med) == 24
    assert len(obs_hard) == 14

    # Bot action decisions
    env = CompetitiveSnakeEnv(obstacles=obs_med, solo=False)
    obs_a, obs_b = env.reset(seed=42)

    easy_bot = EasySnakeBot("Easy", seed=42)
    hard_bot = HardSnakeBot("Hard")

    act_easy = easy_bot.select_action(obs_a, env.get_valid_actions(0), env_state=env)
    act_hard = hard_bot.select_action(obs_b, env.get_valid_actions(1), env_state=env)

    assert 0 <= act_easy < 4
    assert 0 <= act_hard < 4


def test_snake_gui_engine_modes_and_levels():
    from arena.snake_gui import SnakeGameEngine

    # Medium vs bot default
    engine = SnakeGameEngine(level="medium", mode="vs_bot", bot_a_type="human", bot_b_type="pandu")
    st = engine.get_state()
    assert st["level"] == "medium"
    assert st["mode"] == "vs_bot"
    assert st["speed_ms"] == 150
    assert len(st["obstacles"]) == 24
    assert st["wrap_walls"] is False
    assert st["is_game_over"] is False

    # Human action buffer & step
    engine.set_human_action(3)  # RIGHT
    next_st = engine.step()
    assert next_st["step_count"] == 1

    # Switch to Easy Solo mode
    solo_st = engine.reset(level="easy", mode="solo")
    assert solo_st["level"] == "easy"
    assert solo_st["mode"] == "solo"
    assert solo_st["speed_ms"] == 240
    assert len(solo_st["obstacles"]) == 0
    assert solo_st["wrap_walls"] is True
    assert len(solo_st["snake_b"]) == 0  # Solo mode has no Snake B

    # Switch to Hard Bot vs Bot mode
    bvb_st = engine.reset(level="hard", mode="bot_vs_bot", bot_a_type="pandu", bot_b_type="hard")
    assert bvb_st["level"] == "hard"
    assert bvb_st["mode"] == "bot_vs_bot"
    assert bvb_st["speed_ms"] == 90
    assert len(bvb_st["obstacles"]) == 14
    assert bvb_st["player_b_name"] == "Bot-B (Hard)"
    assert bvb_st["wrap_walls"] is False

