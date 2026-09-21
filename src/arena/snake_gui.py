"""Interactive Web GUI and Turn-Based / Real-Time Runner for Pandu Snake Arena.

Features:
- Zero-dependency standard library HTTP server (`http.server`).
- Difficulty levels: Easy (200ms, open arena, novice bot),
                     Medium (120ms, tactical pillars, Pandu-3K bot),
                     Hard (65ms, maze obstacles, master BFS bot).
- Modes: Solo (Single Player Classic), Player vs AI Bot (Duel), and AI vs AI (Spectator).
- Real-time sub-millisecond bot telemetry, animated canvas grid, WebAudio synth sfx.
"""

import json
import os
import sys
import time
import webbrowser
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np

from arena.snake import (
    ACTIONS,
    ACTION_NAMES,
    OPPOSITE_ACTIONS,
    CompetitiveSnakeEnv,
    EasySnakeBot,
    HardSnakeBot,
    HeuristicSnakeBot,
    PanduSnakeBot,
    create_snake_bot,
    generate_obstacles_for_level,
)


LEVEL_SPEEDS = {
    "easy": 240,
    "medium": 150,
    "hard": 90,
}

LEVEL_DEFAULT_BOTS = {
    "easy": "easy",
    "medium": "heuristic",
    "hard": "hard",
}


class SnakeGameEngine:
    """Manages the Snake environment state, difficulty levels, bots, and interactive player dispatch."""

    def __init__(
        self,
        level: str = "medium",
        mode: str = "vs_bot",
        bot_a_type: str = "human",
        bot_b_type: Optional[str] = None,
        width: int = 20,
        height: int = 20,
        seed: int = 42,
        wrap_walls: Optional[bool] = None,
    ):
        self.width = width
        self.height = height
        self.seed = seed
        self.level = level.lower() if level.lower() in ("easy", "medium", "hard") else "medium"
        self.mode = mode.lower() if mode.lower() in ("solo", "vs_bot", "bot_vs_bot") else "vs_bot"
        self.bot_a_type = bot_a_type
        self.bot_b_type = bot_b_type or LEVEL_DEFAULT_BOTS.get(self.level, "heuristic")
        self.wrap_walls = (self.level == "easy") if wrap_walls is None else bool(wrap_walls)

        self.speed_ms = LEVEL_SPEEDS.get(self.level, 150)
        self.high_score = 0
        self.human_action_a: int = 0  # Default UP
        self.latencies: Dict[str, float] = {"a": 0.0, "b": 0.0}

        self._init_environment_and_bots()

    def _init_environment_and_bots(self):
        obstacles = generate_obstacles_for_level(self.level, self.width, self.height)
        is_solo = (self.mode == "solo")
        self.env = CompetitiveSnakeEnv(
            width=self.width,
            height=self.height,
            max_steps=500,
            obstacles=obstacles,
            solo=is_solo,
            wrap_walls=self.wrap_walls,
        )
        self.obs_a, self.obs_b = self.env.reset(seed=self.seed)
        self.human_action_a = self.env.dir_a

        # Setup Bot A
        if self.mode == "bot_vs_bot" or self.bot_a_type != "human":
            self.bot_a = create_snake_bot(self.bot_a_type, seed=self.seed, name=f"Bot-A ({self.bot_a_type.title()})")
        else:
            self.bot_a = None

        # Setup Bot B
        if not is_solo:
            effective_b = self.bot_b_type or LEVEL_DEFAULT_BOTS.get(self.level, "heuristic")
            self.bot_b = create_snake_bot(effective_b, seed=self.seed + 1, name=f"Bot-B ({effective_b.title()})")
        else:
            self.bot_b = None

        self.latencies = {"a": 0.0, "b": 0.0}

    def reset(
        self,
        level: Optional[str] = None,
        mode: Optional[str] = None,
        bot_a_type: Optional[str] = None,
        bot_b_type: Optional[str] = None,
        wrap_walls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Reset game to initial state with optional new level or mode configurations."""
        if level:
            self.level = level.lower()
            self.speed_ms = LEVEL_SPEEDS.get(self.level, 150)
            if wrap_walls is None:
                self.wrap_walls = (self.level == "easy")
        if mode:
            self.mode = mode.lower()
        if bot_a_type:
            self.bot_a_type = bot_a_type
        if bot_b_type:
            self.bot_b_type = bot_b_type
        if wrap_walls is not None:
            self.wrap_walls = bool(wrap_walls)

        self._init_environment_and_bots()
        return self.get_state()

    def set_human_action(self, action: int) -> bool:
        """Buffer direction input from human player for Snake A."""
        act = int(action)
        if 0 <= act < 4:
            # Check 180 reverse against current direction
            if act != OPPOSITE_ACTIONS.get(self.env.dir_a, -1):
                self.human_action_a = act
                return True
        return False

    def step(self, action_a: Optional[int] = None) -> Dict[str, Any]:
        """Advance game by one tick."""
        if self.env.done:
            return self.get_state()

        if action_a is not None:
            self.set_human_action(action_a)

        # Decide Action A
        if self.bot_a is not None:
            valid_a = self.env.get_valid_actions(0)
            t0 = time.perf_counter()
            act_a = self.bot_a.select_action(self.obs_a, valid_actions=valid_a, env_state=self.env)
            self.latencies["a"] = (time.perf_counter() - t0) * 1000.0
        else:
            act_a = self.human_action_a
            self.latencies["a"] = 0.0

        # Decide Action B (if not solo)
        if not self.env.solo and self.bot_b is not None:
            valid_b = self.env.get_valid_actions(1)
            t0 = time.perf_counter()
            act_b = self.bot_b.select_action(self.obs_b, valid_actions=valid_b, env_state=self.env)
            self.latencies["b"] = (time.perf_counter() - t0) * 1000.0
        else:
            act_b = 0
            self.latencies["b"] = 0.0

        # Advance environment
        self.obs_a, self.obs_b, r_a, r_b, done, info = self.env.step(act_a, act_b)

        # Update high score
        curr_max = max(self.env.score_a, self.env.score_b)
        if curr_max > self.high_score:
            self.high_score = curr_max

        return self.get_state(info=info)

    def get_state(self, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Serialize current board and players to JSON dictionary for frontend rendering."""
        snake_a_pts = list(self.env.snake_a)
        snake_b_pts = list(self.env.snake_b)
        obstacles_pts = list(self.env.obstacles)

        winner = "ONGOING"
        reason = ""
        if info:
            winner = info.get("winner", "ONGOING")
            reason = info.get("reason", "")
        elif self.env.done:
            if self.env.solo:
                winner = "GAME_OVER" if len(self.env.snake_a) == 0 else "SURVIVED"
            elif len(self.env.snake_a) > len(self.env.snake_b):
                winner = "A"
            elif len(self.env.snake_b) > len(self.env.snake_a):
                winner = "B"
            else:
                winner = "DRAW"

        # Determine readable status text
        if not self.env.done:
            if self.mode == "solo":
                status_text = f"Level: {self.level.title()} • Score: {self.env.score_a}"
            elif self.mode == "vs_bot":
                status_text = f"Duel ({self.level.title()}) • You vs {self.bot_b.name if self.bot_b else 'AI'}"
            else:
                status_text = f"Spectator • {self.bot_a.name if self.bot_a else 'A'} vs {self.bot_b.name if self.bot_b else 'B'}"
        else:
            if self.mode == "solo":
                status_text = f"Game Over! Final Score: {self.env.score_a} ({reason})"
            else:
                if winner == "A":
                    winner_name = "Player A (You)" if self.mode == "vs_bot" else "Snake A"
                    status_text = f"Victory! {winner_name} Wins! ({reason})"
                elif winner == "B":
                    winner_name = self.bot_b.name if self.bot_b else "Snake B"
                    status_text = f"Victory! {winner_name} Wins! ({reason})"
                else:
                    status_text = f"Match Drawn ({reason})"

        player_a_label = "Human (You)" if self.bot_a is None else getattr(self.bot_a, "name", "Bot-A")
        player_b_label = getattr(self.bot_b, "name", "Bot-B") if self.bot_b else "Disabled"

        return {
            "width": self.width,
            "height": self.height,
            "level": self.level,
            "mode": self.mode,
            "speed_ms": self.speed_ms,
            "step_count": self.env.steps,
            "is_game_over": self.env.done,
            "winner": winner,
            "reason": reason,
            "status_text": status_text,
            "snake_a": snake_a_pts,
            "snake_b": snake_b_pts,
            "dir_a": self.env.dir_a,
            "dir_b": self.env.dir_b,
            "food": list(self.env.food),
            "obstacles": obstacles_pts,
            "score_a": self.env.score_a,
            "score_b": self.env.score_b,
            "len_a": len(self.env.snake_a),
            "len_b": len(self.env.snake_b),
            "player_a_name": player_a_label,
            "player_b_name": player_b_label,
            "latency_a_ms": round(self.latencies["a"], 3),
            "latency_b_ms": round(self.latencies["b"], 3),
            "high_score": self.high_score,
            "wrap_walls": self.wrap_walls,
        }


class SnakeHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Zero-dependency HTTP handler serving Snake static web assets and REST JSON API."""

    engine: SnakeGameEngine = None
    web_dir: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.web_dir, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stdout request logs for cleaner CLI output."""
        pass

    def send_json(self, data: Any, status: int = HTTPStatus.OK):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            state = self.engine.get_state()
            self.send_json(state)
        elif parsed.path in ("/", "/index.html"):
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = b"{}"
        if content_length > 0:
            post_data = self.rfile.read(content_length)

        try:
            req_json = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            req_json = {}

        if parsed.path == "/api/step":
            action_a = req_json.get("action_a")
            state = self.engine.step(action_a=action_a)
            self.send_json(state)
        elif parsed.path == "/api/action":
            act = req_json.get("action")
            if act is None and "direction" in req_json:
                dir_str = str(req_json["direction"]).upper()
                if dir_str in ACTION_NAMES:
                    act = ACTION_NAMES.index(dir_str)
            if act is not None:
                ok = self.engine.set_human_action(int(act))
                self.send_json({"status": "ok" if ok else "ignored", "action": act})
            else:
                self.send_json({"status": "error", "message": "Missing action"}, status=HTTPStatus.BAD_REQUEST)
        elif parsed.path == "/api/reset":
            lvl = req_json.get("level")
            mode = req_json.get("mode")
            bot_a = req_json.get("bot_a_type")
            bot_b = req_json.get("bot_b_type")
            wrap_w = req_json.get("wrap_walls")
            state = self.engine.reset(
                level=lvl, mode=mode, bot_a_type=bot_a, bot_b_type=bot_b, wrap_walls=wrap_w
            )
            self.send_json({"status": "ok", "state": state})
        elif parsed.path == "/api/config":
            if "wrap_walls" in req_json:
                self.engine.wrap_walls = bool(req_json["wrap_walls"])
                self.engine.env.wrap_walls = self.engine.wrap_walls
            if "speed_ms" in req_json:
                self.engine.speed_ms = int(req_json["speed_ms"])
            self.send_json({"status": "ok", "state": self.engine.get_state()})
        else:
            self.send_json({"status": "error", "message": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)


def start_snake_gui_server(
    port: int = 8081,
    level: str = "medium",
    mode: str = "vs_bot",
    bot_a: str = "human",
    bot_b: Optional[str] = None,
    open_browser: bool = True,
):
    """Launch the Snake GUI web server."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    web_dir = os.path.join(current_dir, "web_snake")

    engine = SnakeGameEngine(
        level=level,
        mode=mode,
        bot_a_type=bot_a,
        bot_b_type=bot_b,
        width=20,
        height=20,
    )

    SnakeHTTPRequestHandler.engine = engine
    SnakeHTTPRequestHandler.web_dir = web_dir

    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, SnakeHTTPRequestHandler)

    url = f"http://127.0.0.1:{port}"
    print(f"\n========================================================")
    print(f"🐍 Pandu Snake Arena GUI running at: {url}")
    print(f"🎮 Level: {engine.level.upper()} | Mode: {engine.mode.upper()}")
    print(f"⌨️ Controls: WASD or Arrow Keys | Space: Pause | R: Reset")
    print(f"Press Ctrl+C to shutdown the server.")
    print(f"========================================================\n")

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Pandu Snake GUI server...")
        httpd.server_close()
