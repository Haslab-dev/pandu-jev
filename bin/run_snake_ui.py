#!/usr/bin/env python3
"""Interactive Terminal Snake UI powered by Pandu-Jev NLP (~19.3M).

Uses the rich terminal dashboard from Laya-CoreML with live board rendering,
decision probability bars, latency telemetry, and cycle safety indicators.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time
from collections import deque

import torch
from rich.console import Console
from rich.live import Live

# Ensure paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
LAYA_PATH = Path("/Users/hy4-mac-002/hasdev/research/laya-coreml")

for p in [str(SRC_DIR), str(LAYA_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from laya_coreml.snake.cli import Keyboard, positive
from laya_coreml.snake.game import SnakeGame
from laya_coreml.snake.policy import LayaPolicy
from laya_coreml.snake.ui import BG, compose, layout_size
from models.jev_nlp import PanduJevNLP, PanduJevNLPConfig


def main():
    parser = argparse.ArgumentParser(
        prog="pandu-jev-snake-ui",
        description="Run live Snake terminal UI with Pandu-Jev NLP decision model.",
    )
    parser.add_argument("--width", type=int, default=24, help="Grid width (default: 24)")
    parser.add_argument("--height", type=int, default=16, help="Grid height (default: 16)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    parser.add_argument("--initial-length", type=int, default=6, help="Initial snake length")
    parser.add_argument("--fps", type=positive, default=12, help="Target moves per second (default: 12)")
    parser.add_argument("--max-speed", action="store_true", help="Uncapped speed (run as fast as model decides)")
    parser.add_argument("--duration", type=positive, help="Stop after N seconds")
    parser.add_argument("--steps", type=int, help="Stop after N steps")
    parser.add_argument("--unassisted", action="store_true", help="Disable cycle safety shield (raw top-1)")
    parser.add_argument("--headless", action="store_true", help="Headless benchmark mode")
    parser.add_argument("--no-alt-screen", action="store_true", help="Keep game frame in scrollback")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    console = Console(style=f"on {BG}", highlight=False)

    if not args.headless and not console.is_terminal:
        console.print("[yellow]Warning: Non-TTY environment detected. Running headless fallback.[/yellow]")
        args.headless = True

    console.print(f"[bold cyan]Initializing Pandu-Jev NLP on {device.upper()}...[/bold cyan]")
    model = PanduJevNLP().to(device)
    model.eval()

    # Wrap in LayaPolicy
    policy = LayaPolicy.__new__(LayaPolicy)
    policy.agent = model
    policy.guarded = not args.unassisted
    policy.prompt = "compact"
    policy.metadata = {
        "hardware": f"Apple Silicon ({device.upper()})",
        "engine": "Pandu-Jev NLP (19.3M ModernBERT-Tiny)",
        "guarded": policy.guarded,
    }

    game = SnakeGame(args.width, args.height, args.seed, args.initial_length)

    # Warmup
    warm = SnakeGame(args.width, args.height, args.seed + 10000, args.initial_length)
    for _ in range(5):
        d = policy.decide(warm)
        warm.step(d.executed)

    started = time.perf_counter()
    stats = {
        "hardware": policy.metadata["hardware"],
        "engine": policy.metadata["engine"],
        "guarded": policy.guarded,
        "interventions": 0,
        "best": 0,
        "round": 1,
        "paused": False,
        "elapsed": 0,
        "steps_per_second": 0,
    }

    calls = total_steps = deaths = 0
    timestamps = deque(maxlen=60)
    inference = []
    displayed_board, displayed_decision = game.snapshot(), {}

    live = (
        Live(
            console=console,
            screen=not args.no_alt_screen,
            auto_refresh=False,
            vertical_overflow="crop",
        )
        if not args.headless
        else None
    )

    quit_requested = False
    console.print("[green]Ready! Starting game loop. Press Space to pause, +/- to adjust speed, Q to quit.[/green]")
    time.sleep(0.5)

    try:
        with Keyboard() as keys, live if live else nullcontext():
            while not quit_requested:
                now = time.perf_counter()
                if (args.duration and now - started >= args.duration) or (
                    args.steps and total_steps >= args.steps
                ):
                    break

                pressed = keys.read().lower()
                if "q" in pressed or "\x03" in pressed:
                    break
                if " " in pressed:
                    stats["paused"] = not stats["paused"]
                if "\x1b[a" in pressed or "+" in pressed:
                    args.fps = min(240, args.fps + 2)
                if "\x1b[b" in pressed or "-" in pressed:
                    args.fps = max(1, args.fps - 2)
                if "r" in pressed:
                    stats["round"] += 1
                    game = SnakeGame(
                        args.width, args.height, args.seed + stats["round"] - 1, args.initial_length
                    )
                    displayed_board, displayed_decision = game.snapshot(), {}

                if stats["paused"]:
                    stats["elapsed"] = now - started
                    if live:
                        live.update(
                            compose(displayed_board, displayed_decision, stats).rich_text(),
                            refresh=True,
                        )
                    time.sleep(0.03)
                    continue

                min_w, min_h = layout_size(game.width, game.height)
                if live and (console.width < min_w or console.height < min_h):
                    live.update(
                        f"Resize terminal to at least {min_w} columns × {min_h} rows.\n"
                        "The game is waiting. Q quits.",
                        refresh=True,
                    )
                    time.sleep(0.1)
                    continue

                decision = policy.decide(game)
                calls += 1
                inference.append(decision.inference_ms)
                stats["interventions"] += decision.intervened
                shown = time.perf_counter()
                timestamps.append(shown)
                stats["elapsed"] = shown - started
                stats["steps_per_second"] = (
                    (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                    if len(timestamps) > 1
                    else 0
                )
                stats["best"] = max(stats["best"], game.score)
                board = game.snapshot()
                displayed_board, displayed_decision = board, decision.to_dict()

                if live:
                    canvas = compose(board, decision.to_dict(), stats)
                    live.update(canvas.rich_text(), refresh=True)

                if not args.max_speed:
                    remaining = (1.0 / args.fps) - (time.perf_counter() - now)
                    if remaining > 0:
                        time.sleep(remaining)

                game.step(decision.executed)
                total_steps += 1
                stats["best"] = max(stats["best"], game.score)

                if not game.alive or game.won:
                    deaths += not game.alive
                    if args.unassisted:
                        break
                    if live:
                        live.update(compose(game.snapshot(), {}, stats).rich_text(), refresh=True)
                        time.sleep(1)
                    stats["round"] += 1
                    game = SnakeGame(
                        args.width, args.height, args.seed + stats["round"] - 1, args.initial_length
                    )
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.perf_counter() - started
        summary = {
            "model": "Pandu-Jev NLP (19.3M ModernBERT-Tiny)",
            "steps": total_steps,
            "inference_calls": calls,
            "seconds": round(elapsed, 2),
            "steps_per_second": round(total_steps / elapsed, 2) if elapsed else 0,
            "score": game.score,
            "length": len(game.body),
            "best_score": stats["best"],
            "interventions": stats["interventions"],
            "deaths": deaths,
            "guarded": policy.guarded,
            "mean_inference_ms": round(sum(inference) / len(inference), 2) if inference else None,
        }
        print("\n" + json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
