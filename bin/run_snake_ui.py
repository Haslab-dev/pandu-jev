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
from laya_coreml.snake.ui import BG, compose as orig_compose, layout_size, MUTED, GREEN
from models.jev_nlp import PanduJevNLP, PanduJevNLPConfig


def pandu_compose(game, decision, stats):
    canvas = orig_compose(game, decision, stats)
    width, height = layout_size(game["width"], game["height"])
    right = max(58, game["width"] * 2 + 10)
    engine_name = stats.get("engine", "")
    guarded = stats.get("guarded", True)
    elapsed = stats.get("elapsed", 0)
    clock = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"

    is_typesafe_jev = "System One" in engine_name or "typesafe" in engine_name.lower()

    if is_typesafe_jev:
        canvas.put(1, 3, "JEV (TYPESAFE)  /  CLOUD SYSTEM ONE (API)           ", MUTED)
        canvas.put(4, right, "Jev-latest (Cloud API)", GREEN)
        canvas.put(24, right, "API TOKENS        ", MUTED)
        canvas.put(24, right + 18, f"{decision.get('input_tokens', 0)} in / {decision.get('output_tokens', 0)} out ", GREEN)
        canvas.put(25, right, "NETWORK           ", MUTED)
        canvas.put(25, right + 18, "ONLINE (HTTPS TLS)", GREEN)
        canvas.put(26, right, "MODEL             ", MUTED)
        canvas.put(26, right + 18, "jev-latest (Cloud)", MUTED)
        req_id = stats.get("last_request_id", "")
        if req_id:
            canvas.put(27, right, "REQ ID            ", MUTED)
            canvas.put(27, right + 18, f"{req_id[:16]}...", MUTED)
        canvas.put(28, right, "Jev + cycle safety    " if guarded else "Jev · shield OFF    ", MUTED)
        canvas.put(height - 2, right, f"ESTIMATES BY JEV             {clock}", MUTED)
    else:
        is_base = "Base" in engine_name or "149M" in engine_name
        title = "PANDU BASE  /  LOCAL 149M MODERNBERT (MPS)    " if is_base else "PANDU LITE  /  LOCAL 19.3M MODERNBERT (MPS)    "
        name_tag = "Pandu Base (149M MPS)" if is_base else "Pandu Lite (19.3M MPS)"
        canvas.put(1, 3, title, MUTED)
        canvas.put(4, right, name_tag, GREEN)
        canvas.put(28, right, "Pandu + cycle safety  " if guarded else "Pandu · shield OFF  ", MUTED)
        canvas.put(height - 2, right, f"ESTIMATES BY PANDU           {clock}", MUTED)
    return canvas


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
    parser.add_argument("--base", action="store_true", help="Use 149M answerdotai/ModernBERT-base pre-trained backbone")
    parser.add_argument("--jev", action="store_true", help="Use TypeSafe Jev System One cloud flagship API")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    console = Console(style=f"on {BG}", highlight=False)

    if not args.headless and not console.is_terminal:
        console.print("[yellow]Warning: Non-TTY environment detected. Running headless fallback.[/yellow]")
        args.headless = True

    if args.jev:
        from models.jev_policy import JevPolicy
        console.print("[bold cyan]Initializing TypeSafe Jev System One Client...[/bold cyan]")
        policy = JevPolicy(guarded=not args.unassisted)
    else:
        model_name = "ModernBERT-Base (149M)" if args.base else "Pandu-Jev NLP (19.3M)"
        console.print(f"[bold cyan]Initializing {model_name} on {device.upper()}...[/bold cyan]")
        
        config = PanduJevNLPConfig(use_pretrained_base=args.base)
        model = PanduJevNLP(config).to(device)
        
        ckpt_name = "pandu_snake_base.pt" if args.base else "pandu_snake_nlp.pt"
        ckpt_path = PROJECT_ROOT / "checkpoints" / ckpt_name
        if ckpt_path.is_file():
            console.print(f"[bold green]Loaded trained checkpoint from {ckpt_path.name}![/bold green]")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            console.print(f"[yellow]No checkpoint found at {ckpt_name}, using base weights.[/yellow]")
            
        model.eval()

        # Wrap in LayaPolicy
        policy = LayaPolicy.__new__(LayaPolicy)
        policy.agent = model
        policy.guarded = not args.unassisted
        policy.prompt = "compact"
        policy.metadata = {
            "name": "Pandu Base (149M)" if args.base else "Pandu Lite (19.3M)",
            "hardware": f"Apple Silicon ({device.upper()})",
            "engine": "Pandu-Base (149M)" if args.base else "Pandu-Lite (19.3M)",
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
                            pandu_compose(displayed_board, displayed_decision, stats).rich_text(),
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
                stats["last_request_id"] = getattr(policy, "last_request_id", "")
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
                    canvas = pandu_compose(board, decision.to_dict(), stats)
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
                        live.update(pandu_compose(game.snapshot(), {}, stats).rich_text(), refresh=True)
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
            "model": policy.metadata.get("name", "Pandu-Jev NLP"),
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
