"""Interactive Web GUI and Turn-Based Auto-Runner for Pandu Chess Arena.

Features:
- Zero-dependency standard library HTTP server (`http.server`).
- Auto-run turn-based execution (Bot vs Bot, Pandu vs Pandu, Pandu vs Heuristic).
- Human vs Bot interactive gameplay with click-to-move and drag-and-drop.
- Real-time telemetry: decision latency in ms, neural/material evaluation bar, captured pieces.
- Move history log in standard algebraic notation (SAN).
"""

import json
import os
import sys
import time
import webbrowser
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import chess
import numpy as np

from arena.base import RandomArenaBot
from arena.chess import (
    PIECE_VALUES,
    HeuristicChessBot,
    PanduChessBot,
    PanduChessPolicyBot,
    board_to_feature_vector,
)


class ChessGameEngine:
    """Manages the chess board state, bot instances, and turn-based execution."""

    def __init__(
        self,
        white_type: str = "pandu_deep",
        black_type: str = "heuristic",
        seed: int = 42,
    ):
        self.seed = seed
        self.board = chess.Board()
        self.white_type = white_type
        self.black_type = black_type
        self.white_bot = self._create_bot(white_type, "White")
        self.black_bot = self._create_bot(black_type, "Black")
        self.move_history: List[Dict[str, Any]] = []
        self.last_move: Optional[Dict[str, Any]] = None
        self.latencies: Dict[str, float] = {"white": 0.0, "black": 0.0}
        self.step_count = 0

    def _create_bot(self, bot_type: str, side_label: str):
        b_type = (bot_type or "").lower().strip()
        if b_type in ("pandu_deep", "pandu-deep", "deep", "pandu_curriculum", "pandu_policy", "policy"):
            return PanduChessPolicyBot(name=f"Pandu-Deep-{side_label}")
        elif b_type in ("pandu", "pandu_3k", "pandu-3k", "neural"):
            return PanduChessBot(name=f"Pandu-3K-{side_label}")
        elif b_type in ("heuristic", "heur"):
            return HeuristicChessBot(name=f"Heuristic-{side_label}")
        elif b_type in ("random", "rand"):
            return RandomArenaBot(name=f"Random-{side_label}", seed=self.seed)
        elif b_type == "human":
            return None
        else:
            return PanduChessPolicyBot(name=f"Pandu-Deep-{side_label}")

    def reset(self, white_type: Optional[str] = None, black_type: Optional[str] = None):
        """Reset game to initial position, optionally updating bot types."""
        if white_type:
            self.white_type = white_type
            self.white_bot = self._create_bot(white_type, "White")
        if black_type:
            self.black_type = black_type
            self.black_bot = self._create_bot(black_type, "Black")

        self.board.reset()
        self.move_history = []
        self.last_move = None
        self.latencies = {"white": 0.0, "black": 0.0}
        self.step_count = 0
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        # Material calculation
        white_mat = 0.0
        black_mat = 0.0
        piece_counts: Dict[str, int] = {
            "P": 0, "N": 0, "B": 0, "R": 0, "Q": 0,
            "p": 0, "n": 0, "b": 0, "r": 0, "q": 0,
        }
        for sq in chess.SQUARES:
            p = self.board.piece_at(sq)
            if p is not None:
                val = PIECE_VALUES.get(p.piece_type, 0.0)
                symbol = p.symbol()
                if p.color == chess.WHITE:
                    white_mat += val
                    if symbol in piece_counts:
                        piece_counts[symbol] += 1
                else:
                    black_mat += val
                    if symbol in piece_counts:
                        piece_counts[symbol] += 1

        is_over = self.board.is_game_over()
        result = "*"
        status_text = "White to move"
        if self.board.turn == chess.BLACK:
            status_text = "Black to move"

        if self.board.is_checkmate():
            is_over = True
            result = "1-0" if self.board.turn == chess.BLACK else "0-1"
            winner_name = "White" if result == "1-0" else "Black"
            status_text = f"Checkmate! {winner_name} wins"
        elif self.board.is_stalemate():
            is_over = True
            result = "1/2-1/2"
            status_text = "Draw by Stalemate"
        elif self.board.is_insufficient_material():
            is_over = True
            result = "1/2-1/2"
            status_text = "Draw by Insufficient Material"
        elif self.board.can_claim_threefold_repetition():
            is_over = True
            result = "1/2-1/2"
            status_text = "Draw by Threefold Repetition"
        elif self.board.can_claim_fifty_moves():
            is_over = True
            result = "1/2-1/2"
            status_text = "Draw by 50-Move Rule"
        elif self.step_count >= 150:
            is_over = True
            diff = white_mat - black_mat
            if diff > 1.5:
                result = "1-0"
                status_text = f"Timeout Adjudication: White wins (+{diff:.1f})"
            elif diff < -1.5:
                result = "0-1"
                status_text = f"Timeout Adjudication: Black wins (+{-diff:.1f})"
            else:
                result = "1/2-1/2"
                status_text = "Draw by Timeout (Equal Material)"
        elif self.board.is_check():
            turn_str = "White" if self.board.turn == chess.WHITE else "Black"
            status_text = f"Check! ({turn_str})"

        # Calculate captured pieces relative to standard 8P, 2N, 2B, 2R, 1Q
        captured_white = {
            "p": max(0, 8 - piece_counts["p"]),
            "n": max(0, 2 - piece_counts["n"]),
            "b": max(0, 2 - piece_counts["b"]),
            "r": max(0, 2 - piece_counts["r"]),
            "q": max(0, 1 - piece_counts["q"]),
        }
        captured_black = {
            "P": max(0, 8 - piece_counts["P"]),
            "N": max(0, 2 - piece_counts["N"]),
            "B": max(0, 2 - piece_counts["B"]),
            "R": max(0, 2 - piece_counts["R"]),
            "Q": max(0, 1 - piece_counts["Q"]),
        }

        # Format legal moves
        legal_moves_list = []
        if not is_over:
            for m in self.board.legal_moves:
                legal_moves_list.append({
                    "uci": m.uci(),
                    "from": chess.square_name(m.from_square),
                    "to": chess.square_name(m.to_square),
                    "san": self.board.san(m),
                    "promotion": m.promotion is not None,
                })

        active_bot = self.white_bot if self.board.turn == chess.WHITE else self.black_bot
        active_type = self.white_type if self.board.turn == chess.WHITE else self.black_type

        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "is_game_over": is_over,
            "result": result,
            "status_text": status_text,
            "in_check": self.board.is_check(),
            "last_move": self.last_move,
            "last_intent": self.last_move.get("intent") if self.last_move else None,
            "last_confidence": self.last_move.get("confidence") if self.last_move else None,
            "last_eval": self.last_move.get("eval_value") if self.last_move else None,
            "move_history": self.move_history,
            "legal_moves": legal_moves_list,
            "step_count": self.step_count,
            "white_type": self.white_type,
            "black_type": self.black_type,
            "white_name": getattr(self.white_bot, "name", "Human") if self.white_bot else "Human",
            "black_name": getattr(self.black_bot, "name", "Human") if self.black_bot else "Human",
            "active_type": active_type,
            "is_human_turn": active_bot is None,
            "white_latency_ms": round(self.latencies["white"], 3),
            "black_latency_ms": round(self.latencies["black"], 3),
            "material": {
                "white": white_mat,
                "black": black_mat,
                "diff": round(white_mat - black_mat, 1),
                "captured_by_white": captured_white,
                "captured_by_black": captured_black,
            },
        }

    def step(self) -> Dict[str, Any]:
        """Execute one turn using the active player's bot."""
        if self.board.is_game_over():
            return self.get_state()

        is_white = self.board.turn == chess.WHITE
        active_bot = self.white_bot if is_white else self.black_bot
        color_key = "white" if is_white else "black"

        if active_bot is None:
            # Waiting for human input
            return self.get_state()

        # Bot selects action
        obs = board_to_feature_vector(self.board, self.board.turn)
        legals = list(self.board.legal_moves)
        if not legals:
            return self.get_state()

        t0 = time.perf_counter()
        chosen_move = active_bot.select_action(obs, valid_actions=legals, env_state=self.board)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.latencies[color_key] = dt_ms

        # Push move
        if chosen_move not in self.board.legal_moves:
            chosen_move = legals[0]

        san_str = self.board.san(chosen_move)
        from_sq_str = chess.square_name(chosen_move.from_square)
        to_sq_str = chess.square_name(chosen_move.to_square)

        # Retrieve policy decision metadata if available
        decision = getattr(active_bot, "last_decision", None)
        intent = getattr(decision, "intent", None)
        conf = getattr(decision, "confidence", None)
        eval_val = getattr(decision, "value", None)

        self.last_move = {
            "from": from_sq_str,
            "to": to_sq_str,
            "uci": chosen_move.uci(),
            "san": san_str,
            "turn": color_key,
            "intent": intent,
            "confidence": round(conf, 3) if conf is not None else None,
            "eval_value": round(eval_val, 3) if eval_val is not None else None,
        }

        self.board.push(chosen_move)
        self.step_count += 1

        self.move_history.append({
            "ply": self.step_count,
            "move_number": (self.step_count + 1) // 2,
            "turn": color_key,
            "san": san_str,
            "uci": chosen_move.uci(),
            "from": from_sq_str,
            "to": to_sq_str,
            "fen": self.board.fen(),
            "latency_ms": round(dt_ms, 3),
            "intent": intent,
            "confidence": round(conf, 3) if conf is not None else None,
            "eval_value": round(eval_val, 3) if eval_val is not None else None,
        })

        return self.get_state()

    def make_human_move(self, uci_str: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Apply a move submitted by human player."""
        if self.board.is_game_over():
            return False, "Game is already over", self.get_state()

        is_white = self.board.turn == chess.WHITE
        active_bot = self.white_bot if is_white else self.black_bot
        if active_bot is not None:
            return False, "Not human's turn", self.get_state()

        try:
            move = chess.Move.from_uci(uci_str)
        except Exception:
            return False, "Invalid UCI format", self.get_state()

        # Handle promotion fallback: default to queen if pawn on 7th rank
        if move not in self.board.legal_moves:
            promo_move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if promo_move in self.board.legal_moves:
                move = promo_move

        if move not in self.board.legal_moves:
            return False, "Illegal move", self.get_state()

        color_key = "white" if is_white else "black"
        san_str = self.board.san(move)
        from_sq_str = chess.square_name(move.from_square)
        to_sq_str = chess.square_name(move.to_square)

        self.last_move = {
            "from": from_sq_str,
            "to": to_sq_str,
            "uci": move.uci(),
            "san": san_str,
            "turn": color_key,
        }

        self.board.push(move)
        self.step_count += 1
        self.latencies[color_key] = 0.0

        self.move_history.append({
            "ply": self.step_count,
            "move_number": (self.step_count + 1) // 2,
            "turn": color_key,
            "san": san_str,
            "uci": move.uci(),
            "from": from_sq_str,
            "to": to_sq_str,
            "fen": self.board.fen(),
            "latency_ms": 0.0,
        })

        return True, "Move applied", self.get_state()


class ChessHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Zero-dependency HTTP handler serving static web assets and REST JSON API."""

    engine: ChessGameEngine = None  # Class-level engine singleton
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
        elif parsed.path == "/" or parsed.path == "/index.html":
            # Serve index.html
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
            state = self.engine.step()
            self.send_json(state)
        elif parsed.path == "/api/move":
            uci = req_json.get("uci", "")
            if not uci:
                from_sq = req_json.get("from", "")
                to_sq = req_json.get("to", "")
                promo = req_json.get("promotion", "")
                uci = f"{from_sq}{to_sq}{promo}"
            success, msg, state = self.engine.make_human_move(uci)
            if success:
                self.send_json({"status": "ok", "message": msg, "state": state})
            else:
                self.send_json({"status": "error", "message": msg, "state": state}, status=HTTPStatus.BAD_REQUEST)
        elif parsed.path == "/api/reset":
            white_bot = req_json.get("white_type", "pandu")
            black_bot = req_json.get("black_type", "heuristic")
            state = self.engine.reset(white_type=white_bot, black_type=black_bot)
            self.send_json({"status": "ok", "state": state})
        else:
            self.send_json({"error": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)


def start_chess_gui_server(
    port: int = 8080,
    white_type: str = "pandu",
    black_type: str = "heuristic",
    open_browser: bool = True,
    seed: int = 42,
):
    """Launch interactive Chess GUI web server."""
    web_dir = os.path.join(os.path.dirname(__file__), "web")
    os.makedirs(web_dir, exist_ok=True)

    engine = ChessGameEngine(white_type=white_type, black_type=black_type, seed=seed)
    ChessHTTPRequestHandler.engine = engine
    ChessHTTPRequestHandler.web_dir = web_dir

    server = HTTPServer(("127.0.0.1", port), ChessHTTPRequestHandler)
    url = f"http://127.0.0.1:{port}"

    print(f"\n==================================================")
    print(f"  Pandu Chess Arena Interactive Web GUI Running")
    print(f"  URL: {url}")
    print(f"  White: {white_type.upper()} | Black: {black_type.upper()}")
    print(f"  Auto-Run & Human-vs-Bot Play Enabled")
    print(f"  Press Ctrl+C in terminal to stop server")
    print(f"==================================================\n")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Pandu Chess GUI server...")
        server.server_close()
