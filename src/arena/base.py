"""Universal Multi-Agent Arena Framework for Pandu (pandu-jev).

Provides:
- ArenaEnvironment: Abstract protocol for 2-player closed-loop games.
- ArenaBot: Protocol for policy models, heuristics, and random baselines.
- ArenaMatch: Runs a single head-to-head match between Bot A and Bot B.
- ArenaTournament: Runs an N-game tournament with slot-swapping (fairness)
  and produces controller-grade competitive metrics (wins, draws, latency, turns).
"""

import abc
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class ArenaEnvironment(abc.ABC):
    """Abstract 2-player competitive environment."""

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset environment. Returns (obs_a, obs_b)."""
        pass

    @abc.abstractmethod
    def step(
        self, action_a: Any, action_b: Any
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool, Dict[str, Any]]:
        """Simultaneous or turn-based step.

        Returns: (obs_a, obs_b, reward_a, reward_b, done, info).
        info MUST contain:
          - "winner": "A", "B", or "DRAW"
          - "reason": str (e.g. "wall_collision", "checkmate", "timeout")
        """
        pass

    @abc.abstractmethod
    def get_valid_actions(self, player_id: int) -> List[Any]:
        """Return list of valid legal action candidates for player (0=A, 1=B)."""
        pass


class ArenaBot(abc.ABC):
    """Abstract decision maker in the Arena."""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: Optional[List[Any]] = None,
        env_state: Optional[Any] = None,
    ) -> Any:
        """Select action given observation and valid action set."""
        pass


class RandomArenaBot(ArenaBot):
    """Selects a random legal action."""

    def __init__(self, name: str = "RandomBot", seed: int = 42):
        super().__init__(name)
        self.rng = np.random.RandomState(seed)

    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: Optional[List[Any]] = None,
        env_state: Optional[Any] = None,
    ) -> Any:
        if valid_actions is not None and len(valid_actions) > 0:
            idx = int(self.rng.randint(0, len(valid_actions)))
            return valid_actions[idx]
        return 0


class ArenaMatch:
    """Manages a single match between two bots."""

    def __init__(self, env: ArenaEnvironment, bot_a: ArenaBot, bot_b: ArenaBot):
        self.env = env
        self.bot_a = bot_a
        self.bot_b = bot_b

    def play(self, seed: Optional[int] = None, max_steps: int = 500) -> Dict[str, Any]:
        obs_a, obs_b = self.env.reset(seed=seed)
        done = False
        step = 0

        latencies_a: List[float] = []
        latencies_b: List[float] = []

        while not done and step < max_steps:
            valid_a = self.env.get_valid_actions(player_id=0)
            valid_b = self.env.get_valid_actions(player_id=1)

            board_state = getattr(self.env, "board", None)
            t0 = time.perf_counter()
            act_a = self.bot_a.select_action(obs_a, valid_actions=valid_a, env_state=board_state)
            t1 = time.perf_counter()
            latencies_a.append((t1 - t0) * 1000.0)

            t0 = time.perf_counter()
            act_b = self.bot_b.select_action(obs_b, valid_actions=valid_b, env_state=board_state)
            t1 = time.perf_counter()
            latencies_b.append((t1 - t0) * 1000.0)

            obs_a, obs_b, r_a, r_b, done, info = self.env.step(act_a, act_b)
            step += 1

        winner = info.get("winner", "DRAW")
        if not done and step >= max_steps:
            winner = info.get("winner_on_timeout", "DRAW")

        return {
            "winner": winner,
            "reason": info.get("reason", "timeout" if step >= max_steps else "unknown"),
            "steps": step,
            "mean_latency_a_ms": float(np.mean(latencies_a)) if latencies_a else 0.0,
            "mean_latency_b_ms": float(np.mean(latencies_b)) if latencies_b else 0.0,
            "info": info,
        }


class ArenaTournament:
    """Runs an N-game competitive tournament between two bot architectures."""

    def __init__(
        self,
        env_factory: Any,
        bot_a: ArenaBot,
        bot_b: ArenaBot,
        total_games: int = 100,
        alternate_slots: bool = True,
        seed: int = 42,
    ):
        self.env_factory = env_factory
        self.bot_a = bot_a
        self.bot_b = bot_b
        self.total_games = total_games
        self.alternate_slots = alternate_slots
        self.seed = seed

    def run(self) -> Dict[str, Any]:
        rng = np.random.RandomState(self.seed)
        a_wins = 0
        b_wins = 0
        draws = 0
        turns_list = []
        lat_a_list = []
        lat_b_list = []
        match_records = []

        for i in range(self.total_games):
            game_seed = int(rng.randint(0, 1_000_000))
            env = self.env_factory()

            # Alternate slots to ensure no first-player or side advantage
            if self.alternate_slots and (i % 2 == 1):
                match = ArenaMatch(env, bot_a=self.bot_b, bot_b=self.bot_a)
                res = match.play(seed=game_seed)
                # Slot B is bot_a here
                if res["winner"] == "A":
                    b_wins += 1
                elif res["winner"] == "B":
                    a_wins += 1
                else:
                    draws += 1
                lat_a_list.append(res["mean_latency_b_ms"])
                lat_b_list.append(res["mean_latency_a_ms"])
            else:
                match = ArenaMatch(env, bot_a=self.bot_a, bot_b=self.bot_b)
                res = match.play(seed=game_seed)
                if res["winner"] == "A":
                    a_wins += 1
                elif res["winner"] == "B":
                    b_wins += 1
                else:
                    draws += 1
                lat_a_list.append(res["mean_latency_a_ms"])
                lat_b_list.append(res["mean_latency_b_ms"])

            turns_list.append(res["steps"])
            match_records.append(res)

        win_rate_a = (a_wins / max(1, self.total_games)) * 100.0
        win_rate_b = (b_wins / max(1, self.total_games)) * 100.0
        draw_rate = (draws / max(1, self.total_games)) * 100.0

        summary = {
            "total_games": self.total_games,
            "bot_a": self.bot_a.name,
            "bot_b": self.bot_b.name,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "win_rate_a": win_rate_a,
            "win_rate_b": win_rate_b,
            "draw_rate": draw_rate,
            "mean_turns": float(np.mean(turns_list)),
            "mean_latency_a_ms": float(np.mean(lat_a_list)),
            "mean_latency_b_ms": float(np.mean(lat_b_list)),
        }
        return summary

    def print_summary(self, summary: Dict[str, Any], title: str = "Pandu Arena") -> None:
        """Render Rich summary table matching Pandu CLI specification."""
        table = Table(title=title, header_style="bold cyan")
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")

        table.add_row("Total Games", str(summary["total_games"]))
        table.add_row("Bot A", summary["bot_a"])
        table.add_row("Bot B", summary["bot_b"])
        table.add_row("A Wins", f"{summary['a_wins']} ({summary['win_rate_a']:.1f}%)")
        table.add_row("B Wins", f"{summary['b_wins']} ({summary['win_rate_b']:.1f}%)")
        table.add_row("Draws", f"{summary['draws']} ({summary['draw_rate']:.1f}%)")
        table.add_row("Avg Turns / Game", f"{summary['mean_turns']:.1f}")
        table.add_row("Bot A Decision Latency", f"{summary['mean_latency_a_ms']:.3f} ms")
        table.add_row("Bot B Decision Latency", f"{summary['mean_latency_b_ms']:.3f} ms")
        console.print(table)
