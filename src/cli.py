"""Pandu (pandu-jev) Command-Line Interface.

Provides:
- pandu-jev play gridworld (Step-by-step policy rollout with confidence display)
- pandu-jev train (Generate dataset & train TinyPolicy)
- pandu-jev benchmark (Run calibration, scaling, uncertainty, and hybrid fallback)
- pandu-jev ppo (Train policy via Reinforcement Learning)
"""

import os
import sys
import time
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import torch

from env.gridworld import GridWorld, Action
from env.racing import RacingEnv
from models.policy import TinyPolicy, build_scaled_model
from expert.astar import AStarExpert
from datasets.trajectory import collect_expert_trajectories, save_dataset, load_dataset
from training.bc import train_behavioral_cloning, evaluate_policy_closed_loop
from training.ppo import train_ppo
from evaluation.calibration import compute_calibration_metrics, format_reliability_table
from evaluation.uncertainty import run_uncertainty_benchmark
from evaluation.stress_testing import run_stress_test_suite
from experiments.benchmarks.model_scaling import run_model_scaling_benchmark
from experiments.benchmarks.hybrid_fallback import run_hybrid_fallback_benchmark

console = Console()
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "tiny_policy.pt")
DATASET_PATH = "data/expert_trajectories.npz"


def get_or_train_model() -> TinyPolicy:
    """Load cached model or quickly train one if missing."""
    model = TinyPolicy(input_dim=16, num_actions=4)

    if os.path.exists(CHECKPOINT_PATH):
        try:
            state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            return model
        except Exception:
            pass

    console.print("[yellow]No checkpoint found. Training a fresh TinyPolicy on expert demonstrations...[/yellow]")
    if os.path.exists(DATASET_PATH):
        X, y, _ = load_dataset(DATASET_PATH)
    else:
        console.print("[cyan]Generating 1,500 A* expert demonstration episodes...[/cyan]")
        X, y, stats = collect_expert_trajectories(num_episodes=1500, seed=42)
        save_dataset(DATASET_PATH, X, y, stats)

    train_behavioral_cloning(model, X, y, epochs=35, batch_size=64, device=torch.device("cpu"))
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    console.print(f"[green]Trained and saved TinyPolicy ({model.count_parameters():,} params) to {CHECKPOINT_PATH}[/green]")
    return model


@click.group()
def cli():
    """Pandu (pandu-jev): Tiny, Local, Zero-Cost Policy & Uncertainty Research Suite."""
    pass


@cli.command("play")
@click.argument("environment", default="gridworld")
@click.option("--delay", default=0.15, help="Delay between steps in seconds.")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
@click.option("--random-map", is_flag=True, help="Use procedural random map instead of default.")
def play(environment: str, delay: float, seed: Optional[int], random_map: bool):
    """Play an environment interactively with the trained TinyPolicy."""
    if environment.lower() == "gridworld":
        model = get_or_train_model()

        if random_map:
            from env.gridworld import create_random_gridworld
            env = create_random_gridworld(width=12, height=7, wall_prob=0.18, seed=seed)
        else:
            env = GridWorld(random_start_goal=False, seed=seed)

        console.print(Panel.fit("[bold green]Pandu (pandu-jev) Playing GridWorld[/bold green]\n"
                                f"Model Size: {model.count_parameters():,} parameters | Cost: $0.00 | Local Inference"))

        obs = env.observe()
        step = 0
        total_reward = 0.0

        console.print("\n[bold]Initial Map State:[/bold]")
        console.print(env.render_ascii(), style="cyan")

        while not env.done and step < env.max_steps:
            step += 1
            feat = env.get_feature_vector()
            dist = model.get_action_distribution(feat)

            console.print(f"\n[bold]step {step:02d}[/bold]")
            for act_name, prob in dist["probabilities"].items():
                bar = "█" * int(prob * 20)
                color = "green" if act_name == dist["action"] else "dim"
                console.print(f"  [{color}]{act_name:5s}  {prob:0.2f}  |{bar:<20}|[/{color}]")

            conf = dist["confidence"]
            conf_color = "green" if conf >= 0.8 else ("yellow" if conf >= 0.5 else "red")
            console.print(f"  Confidence: [{conf_color}]{conf*100:.1f}%[/{conf_color}] (Entropy: {dist['entropy']:.2f})")
            console.print(f"  → [bold cyan]{dist['action']}[/bold cyan]")

            obs, r, done, info = env.step(dist["action_idx"])
            total_reward += r

            if info.get("hit_wall", False):
                console.print("  [red]⚠ Wall collision! (-10 penalty)[/red]")

            if delay > 0:
                time.sleep(delay)

        if info.get("reached_goal", False):
            console.print(Panel.fit(f"[bold green]★ GOAL REACHED! ★[/bold green]\n"
                                    f"Total Steps: {step}\nTotal Reward: {total_reward:+.1f}"))
        else:
            console.print(Panel.fit(f"[bold red]Episode Terminated (Max steps reached)[/bold red]\n"
                                    f"Total Steps: {step}\nTotal Reward: {total_reward:+.1f}"))

    elif environment.lower() == "racing":
        env = RacingEnv(seed=seed)
        console.print(Panel.fit("[bold green]Pandu (pandu-jev) in 2D Racing Environment[/bold green]"))
        step = 0
        while not env.done and step < 50:
            step += 1
            # Simple heuristic or policy
            feat = env.get_feature_vector()
            # Steer toward center
            if feat[2] > 0.1:
                action = 0  # LEFT
            elif feat[2] < -0.1:
                action = 1  # RIGHT
            else:
                action = 2  # ACCEL
            obs, r, done, info = env.step(action)
            console.print(f"Step {step:02d} | Action: {info['action']} | R: {r:+.2f}")
            console.print(env.render_ascii())
            time.sleep(delay)
    else:
        console.print(f"[red]Unknown environment: {environment}[/red]")


@cli.command("train")
@click.option("--episodes", default=2000, help="Number of expert demonstration episodes to collect.")
@click.option("--epochs", default=40, help="Training epochs for behavioral cloning.")
def train(episodes: int, epochs: int):
    """Collect expert trajectories and train TinyPolicy."""
    console.print(f"[bold cyan]Collecting {episodes:,} expert demonstration episodes via A*...[/bold cyan]")
    X, y, stats = collect_expert_trajectories(num_episodes=episodes, seed=42)
    save_dataset(DATASET_PATH, X, y, stats)
    console.print(f"[green]Dataset generated: {len(y):,} samples across {episodes:,} episodes.[/green]")

    model = TinyPolicy(input_dim=16, num_actions=4)
    console.print(f"[bold cyan]Training TinyPolicy ({model.count_parameters():,} params) for {epochs} epochs...[/bold cyan]")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"Using compute device: [magenta]{device}[/magenta]")

    res = train_behavioral_cloning(model, X, y, epochs=epochs, batch_size=64, device=device)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)

    console.print(Panel.fit(
        f"[bold green]Training Completed Successfully![/bold green]\n"
        f"Validation Accuracy: {res['final_val_acc']*100:.2f}%\n"
        f"Optimal Temperature: {res['optimal_temperature']:.3f}\n"
        f"Training Duration: {res['training_time_seconds']:.2f}s\n"
        f"Model saved to: {CHECKPOINT_PATH}"
    ))


@cli.command("benchmark")
@click.option("--quick", is_flag=True, help="Run faster benchmark with smaller sample sizes.")
def benchmark(quick: bool):
    """Run complete empirical research benchmarks across all phases."""
    model = get_or_train_model()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    console.print(Panel.fit("[bold magenta]Running Pandu (pandu-jev) Research Benchmark Suite[/bold magenta]\n"
                            "Phases: Calibration, Uncertainty, Parameter Scaling, and Hybrid Fallback"))

    # 1. Closed Loop Baseline
    console.print("\n[bold cyan]1. Closed-Loop Rollout Evaluation[/bold cyan]")
    n_eval = 25 if quick else 100
    baseline_eval = evaluate_policy_closed_loop(model, num_episodes=n_eval, device=torch.device("cpu"))
    table_base = Table(title="Closed-Loop Baseline Performance")
    table_base.add_column("Metric", style="cyan")
    table_base.add_column("Value", style="green")
    table_base.add_row("Success Rate", f"{baseline_eval['success_rate']*100:.1f}%")
    table_base.add_row("Average Reward", f"{baseline_eval['avg_reward']:+.1f}")
    table_base.add_row("Average Steps", f"{baseline_eval['avg_steps']:.1f}")
    table_base.add_row("Mean Inference Latency", f"{baseline_eval['avg_latency_ms']:.3f} ms")
    table_base.add_row("P95 Latency", f"{baseline_eval['p95_latency_ms']:.3f} ms")
    console.print(table_base)

    # 2. Uncertainty & OOD Benchmark
    console.print("\n[bold cyan]2. Uncertainty & Out-of-Distribution Benchmark (Phase 5 & 6)[/bold cyan]")
    n_ood = 15 if quick else 40
    ood_res = run_uncertainty_benchmark(model, num_episodes_per_regime=n_ood)
    table_ood = Table(title="Uncertainty Across Regimes (Does low confidence correlate with failure?)")
    table_ood.add_column("Regime", style="bold")
    table_ood.add_column("Success", justify="right")
    table_ood.add_column("Mean Conf", justify="right")
    table_ood.add_column("Agreement", justify="right")
    table_ood.add_column("ECE", justify="right")
    table_ood.add_column("Brier", justify="right")
    table_ood.add_column("AUROC (Err Detect)", justify="right", style="green")

    for reg, d in ood_res.items():
        table_ood.add_row(
            reg,
            f"{d['episode_success_rate']*100:.1f}%",
            f"{d['mean_confidence']:.3f}",
            f"{d['action_agreement_rate']*100:.1f}%",
            f"{d['ece']:.3f}",
            f"{d['brier_score']:.3f}",
            f"{d['auroc_error_detection']:.3f}",
        )
    console.print(table_ood)

    # 3. Stress Testing Suite
    console.print("\n[bold cyan]3. Stress Testing Suite (Phase 8)[/bold cyan]")
    stress_res = run_stress_test_suite(model, num_episodes_per_variant=n_ood)
    table_stress = Table(title="Robustness Across Environmental Variations")
    table_stress.add_column("Variant", style="bold")
    table_stress.add_column("Success Rate", justify="right")
    table_stress.add_column("Avg Reward", justify="right")
    table_stress.add_column("Wall Hits/Ep", justify="right")
    table_stress.add_column("Mean Conf", justify="right")

    for var, d in stress_res.items():
        table_stress.add_row(
            var,
            f"{d['success_rate']*100:.1f}%",
            f"{d['avg_reward']:+.1f}",
            f"{d['avg_wall_hits']:.2f}",
            f"{d['mean_confidence']:.3f}",
        )
    console.print(table_stress)

    # 4. Hybrid Fallback Benchmark
    console.print("\n[bold cyan]4. Hybrid Fallback Architecture Benchmark (Phase 14 & Final Benchmark)[/bold cyan]")
    hyb_res = run_hybrid_fallback_benchmark(model, num_episodes=n_eval, confidence_threshold=0.85)
    table_hyb = Table(title="Teacher vs Pandu (pandu-jev) vs Hybrid Fallback")
    table_hyb.add_column("Architecture", style="bold")
    table_hyb.add_column("Success Rate", justify="right")
    table_hyb.add_column("Avg Latency", justify="right")
    table_hyb.add_column("Cost / Ep", justify="right")
    table_hyb.add_column("Fallback %", justify="right")
    table_hyb.add_column("Actions/sec", justify="right")

    for arch, d in hyb_res.items():
        table_hyb.add_row(
            arch,
            f"{d['success_rate']*100:.1f}%",
            f"{d['avg_latency_ms']:.2f} ms",
            f"${d['cost_per_episode_usd']:.4f}",
            f"{d['fallback_frequency']*100:.1f}%",
            f"{d['actions_per_second']:.0f}",
        )
    console.print(table_hyb)


@cli.command("ppo")
@click.option("--episodes", default=300, help="Number of RL episodes to train.")
def ppo(episodes: int):
    """Train policy directly via PPO reinforcement learning."""
    console.print(f"[bold cyan]Starting PPO Reinforcement Learning for {episodes} episodes...[/bold cyan]")
    model, stats = train_ppo(total_episodes=episodes, seed=42)
    console.print(Panel.fit(
        f"[bold green]PPO Training Finished![/bold green]\n"
        f"Final Moving Avg Reward: {stats['final_avg_reward']:+.2f}\n"
        f"Final Success Rate: {stats['final_success_rate']*100:.1f}%\n"
        f"Total Episodes: {stats['total_episodes']}"
    ))


@cli.command("test-language")
def test_language():
    """Run the Grounded Language Cortex & Canonical Intent Protocol benchmark."""
    from experiments.language.grounded_cortex import main as run_lang_exp
    run_lang_exp()


@cli.command("test-representation")
@click.option("--episodes", default=1000, help="Number of expert demonstration episodes.")
def test_representation(episodes: int):
    """Run Environment Representation Scaling Benchmark (16d -> 32d -> 64d -> 128d)."""
    from experiments.benchmarks.representation_scaling import run_representation_scaling_benchmark
    run_representation_scaling_benchmark(num_episodes=episodes, epochs=30, eval_episodes=80)


@cli.command("test-memory")
def test_memory():
    """Run 5-Way Ablation Benchmark: State vs Language vs Recurrent Memory."""
    from experiments.benchmarks.recurrent_memory import run_5way_ablation_benchmark
    run_5way_ablation_benchmark()


@cli.command("test-all")
def test_all():
    """Execute all test suites (unit tests + language models) in one command."""
    from experiments.run_all_tests import main as run_all
    run_all()


@cli.group("arena")
def arena():
    """Multi-Agent Closed-Loop Arena (Competitive Snake, Micro Chess, Evolution)."""
    pass


@arena.command("snake")
@click.option("--bot-a", default="pandu", type=click.Choice(["pandu", "heuristic", "random"]), help="Architecture for Bot A.")
@click.option("--bot-b", default="heuristic", type=click.Choice(["pandu", "heuristic", "random"]), help="Architecture for Bot B.")
@click.option("--games", default=50, help="Number of competitive games.")
@click.option("--seed", default=42, help="Tournament random seed.")
def arena_snake(bot_a: str, bot_b: str, games: int, seed: int):
    """Run Competitive 2-Player Snake Tournament."""
    from arena.base import ArenaTournament, RandomArenaBot
    from arena.snake import CompetitiveSnakeEnv, HeuristicSnakeBot, train_pandu_snake_bot

    def make_bot(choice: str, name_prefix: str):
        if choice == "pandu":
            return train_pandu_snake_bot(episodes=150, epochs=20, seed=seed)
        elif choice == "heuristic":
            return HeuristicSnakeBot(name=f"Heuristic-{name_prefix}")
        else:
            return RandomArenaBot(name=f"Random-{name_prefix}", seed=seed)

    bot1 = make_bot(bot_a, "A")
    bot2 = make_bot(bot_b, "B")

    tournament = ArenaTournament(
        CompetitiveSnakeEnv,
        bot_a=bot1,
        bot_b=bot2,
        total_games=games,
        seed=seed,
    )
    summary = tournament.run()
    tournament.print_summary(summary, title=f"Pandu Arena: Snake ({bot_a.upper()} vs {bot_b.upper()})")


@arena.command("chess")
@click.option("--bot-a", default="pandu", type=click.Choice(["pandu", "heuristic", "random"]), help="Architecture for Bot A.")
@click.option("--bot-b", default="random", type=click.Choice(["pandu", "heuristic", "random"]), help="Architecture for Bot B.")
@click.option("--games", default=20, help="Number of competitive chess games.")
@click.option("--seed", default=42, help="Tournament random seed.")
def arena_chess(bot_a: str, bot_b: str, games: int, seed: int):
    """Run Competitive Micro Chess Tournament (Environment enforces rules, Pandu decides moves)."""
    from arena.base import ArenaTournament, RandomArenaBot
    from arena.chess import CompetitiveChessEnv, HeuristicChessBot, PanduChessBot

    def make_bot(choice: str, name_prefix: str):
        if choice == "pandu":
            return PanduChessBot(name=f"Pandu-Chess-{name_prefix}")
        elif choice == "heuristic":
            return HeuristicChessBot(name=f"Heuristic-{name_prefix}")
        else:
            return RandomArenaBot(name=f"Random-{name_prefix}", seed=seed)

    bot1 = make_bot(bot_a, "A")
    bot2 = make_bot(bot_b, "B")

    tournament = ArenaTournament(
        CompetitiveChessEnv,
        bot_a=bot1,
        bot_b=bot2,
        total_games=games,
        seed=seed,
    )
    summary = tournament.run()
    tournament.print_summary(summary, title=f"Pandu Arena: Chess ({bot_a.upper()} vs {bot_b.upper()})")


@arena.command("evolve")
@click.option("--generations", default=5, help="Number of evolutionary tournament generations.")
@click.option("--pop-size", default=6, help="Population size.")
def arena_evolve(generations: int, pop_size: int):
    """Run Evolutionary Self-Play League tournament."""
    from arena.evolution import run_evolutionary_experiment
    run_evolutionary_experiment(generations=generations, population_size=pop_size)


@arena.command("chess-gui")
@click.option("--port", default=8080, help="Port to bind the web GUI server.")
@click.option("--white", default="pandu", type=click.Choice(["pandu", "heuristic", "random", "human"]), help="White player agent.")
@click.option("--black", default="heuristic", type=click.Choice(["pandu", "heuristic", "random", "human"]), help="Black player agent.")
@click.option("--no-browser", is_flag=True, default=False, help="Do not open browser automatically.")
def arena_chess_gui(port: int, white: str, black: str, no_browser: bool):
    """Launch interactive Chess GUI with turn-based auto-run and human play."""
    from arena.chess_gui import start_chess_gui_server
    start_chess_gui_server(
        port=port,
        white_type=white,
        black_type=black,
        open_browser=not no_browser,
    )


@arena.command("snake-gui")
@click.option("--port", default=8081, help="Port to bind the web GUI server.")
@click.option("--level", default="medium", type=click.Choice(["easy", "medium", "hard"]), help="Difficulty level.")
@click.option("--mode", default="vs_bot", type=click.Choice(["solo", "vs_bot", "bot_vs_bot"]), help="Gameplay mode.")
@click.option("--bot-a", default="human", type=click.Choice(["human", "pandu", "heuristic", "easy", "hard", "random"]), help="Agent for Snake A.")
@click.option("--bot-b", default="pandu", type=click.Choice(["pandu", "heuristic", "easy", "hard", "random"]), help="Agent for Snake B.")
@click.option("--no-browser", is_flag=True, default=False, help="Do not open browser automatically.")
def arena_snake_gui(port: int, level: str, mode: str, bot_a: str, bot_b: str, no_browser: bool):
    """Launch interactive Snake GUI with Easy, Medium, Hard levels and real-time play."""
    from arena.snake_gui import start_snake_gui_server
    start_snake_gui_server(
        port=port,
        level=level,
        mode=mode,
        bot_a=bot_a,
        bot_b=bot_b,
        open_browser=not no_browser,
    )


@arena.command("snake-terminal")
@click.option("--fps", default=12, help="Moves per second (default: 12)")
@click.option("--max-speed", is_flag=True, help="Uncapped speed (run as fast as model decides)")
@click.option("--seed", default=7, help="Random seed (default: 7)")
@click.option("--unassisted", is_flag=True, help="Disable cycle safety shield (raw top-1)")
def arena_snake_terminal(fps: int, max_speed: bool, seed: int, unassisted: bool):
    """Launch interactive terminal Snake dashboard UI powered by Pandu-Jev NLP."""
    import subprocess
    cmd = [sys.executable, "bin/run_snake_ui.py", "--fps", str(fps), "--seed", str(seed)]
    if max_speed:
        cmd.append("--max-speed")
    if unassisted:
        cmd.append("--unassisted")
    subprocess.run(cmd)


@arena.command("duel")
def arena_duel():
    """Launch dual side-by-side Terminal windows: Pandu-Jev NLP vs Laya CoreML."""
    import subprocess
    script_path = Path(__file__).resolve().parent.parent / "bin" / "run_side_by_side.sh"
    console.print("[bold cyan]Launching Pandu-Jev NLP vs Laya CoreML side-by-side duel...[/bold cyan]")
    subprocess.run(["bash", str(script_path)])


@arena.command("trio")
def arena_trio():
    """Launch 3 Terminal windows: Pandu-Jev vs Laya-CoreML vs Jev (Max Speed)."""
    import subprocess
    script_path = Path(__file__).resolve().parent.parent / "bin" / "run_trio.sh"
    console.print("[bold cyan]Launching 3-Way Snake Arena (Pandu vs Laya vs Jev Max Speed)...[/bold cyan]")
    subprocess.run(["bash", str(script_path)])


@arena.command("snake-jev")
@click.option("--fps", default=4, help="Moves per second (default: 4 for cloud latency)")
@click.option("--seed", default=7, help="Random seed (default: 7)")
def arena_snake_jev(fps: int, seed: int):
    """Launch interactive terminal Snake dashboard UI powered by TypeSafe Jev System One."""
    import subprocess
    cmd = [sys.executable, "bin/run_snake_ui.py", "--jev", "--fps", str(fps), "--seed", str(seed)]
    subprocess.run(cmd)


@cli.command("train-chess")
@click.option("--samples", default=250, help="Number of curriculum samples per tier.")
@click.option("--epochs", default=5, help="Training epochs per curriculum tier.")
@click.option("--save-path", default="experiments/results/pandu_chess_curriculum.pt", help="Output path for weights.")
def train_chess(samples: int, epochs: int, save_path: str):
    """Train Pandu Chess policy using the 5-phase cumulative curriculum."""
    from training.chess_curriculum import ChessCurriculumTrainer
    console.print(f"[bold cyan]Starting Pandu Chess 5-Phase Curriculum Training...[/bold cyan]")
    trainer = ChessCurriculumTrainer()
    metrics = trainer.run_curriculum(samples_per_tier=samples, epochs_per_tier=epochs, save_path=save_path)

    table = Table(title="Chess Curriculum Training Progression")
    table.add_column("Tier", style="cyan")
    table.add_column("Samples", style="magenta")
    table.add_column("Accuracy", style="green")
    table.add_column("Mate-in-1", style="yellow")
    table.add_column("Final Loss", style="white")
    table.add_column("Time (s)", style="blue")

    for m in metrics:
        table.add_row(
            m.tier_name,
            f"{m.samples_count:,}",
            f"{m.target_move_accuracy*100:.1f}%",
            f"{m.mate_in_1_accuracy*100:.1f}%",
            f"{m.final_loss:.4f}",
            f"{m.training_time_sec:.2f}",
        )
    console.print(table)
    console.print(f"[bold green]Saved model weights to: {save_path}[/bold green]")


@cli.command("benchmark-chess")
@click.option("--quick", is_flag=True, help="Run faster benchmark with smaller sample sizes.")
def benchmark_chess(quick: bool):
    """Run EXP-009 micro-policy chess benchmark suite."""
    from experiments.benchmarks.exp009_chess_benchmark import run_chess_benchmark
    console.print(f"[bold cyan]Running EXP-009 Micro-Policy Chess Benchmark Suite...[/bold cyan]")
    res = run_chess_benchmark(quick=quick)

    table = Table(title="EXP-009 Chess Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Result", style="green")
    table.add_row("Total Parameters", f"{res.total_parameters:,} (<50K budget)")
    table.add_row("Legal Move Rate", f"{res.legal_move_rate*100:.1f}% (Guaranteed)")
    table.add_row("Mate-in-1 Accuracy", f"{res.mate_in_1_accuracy*100:.1f}%")
    table.add_row("Tactical Capture Accuracy", f"{res.tactical_capture_accuracy*100:.1f}%")
    table.add_row("Average Latency", f"{res.avg_latency_ms:.3f} ms (p95: {res.p95_latency_ms:.3f} ms)")
    table.add_row("Win Rate vs Random", f"{res.win_rate_vs_random*100:.1f}% (Draws: {res.draw_rate_vs_random*100:.1f}%)")
    table.add_row("Win Rate vs Heuristic", f"{res.win_rate_vs_heuristic*100:.1f}% (Draws: {res.draw_rate_vs_heuristic*100:.1f}%)")
    table.add_row("Self-Play Draw Rate", f"{res.self_play_draw_rate*100:.1f}%")
    console.print(table)


@cli.command("benchmark-chess-deep")
@click.option("--quick", is_flag=True, help="Run faster benchmark with smaller sample sizes.")
def benchmark_chess_deep(quick: bool):
    """Run EXP-010 capacity-controlled deep curriculum chess benchmark suite."""
    from experiments.benchmarks.exp010_chess_capacity_scaling import run_capacity_scaling_experiment
    console.print(f"[bold cyan]Running EXP-010 Capacity-Controlled Deep Chess Benchmark...[/bold cyan]")
    res = run_capacity_scaling_experiment(quick=quick)

    table = Table(title="EXP-010 Deep Curriculum Benchmark Results")
    table.add_column("Evaluation Metric", style="cyan")
    table.add_column("Measured Performance", style="green")
    table.add_row("Total Parameters", f"{res.total_parameters:,} (<50K budget)")
    table.add_row("Legal Move Rate", f"{res.legal_move_rate*100:.1f}% (Guaranteed)")
    table.add_row("Mate-in-1 Accuracy", f"{res.mate_in_1_accuracy*100:.1f}%")
    table.add_row("Tactical Depth Accuracy", f"{res.tactical_depth_accuracy*100:.1f}%")
    table.add_row("Endgame Competence", f"{res.endgame_competence_accuracy*100:.1f}%")
    table.add_row("Tactical Capture Accuracy", f"{res.tactical_capture_accuracy*100:.1f}%")
    table.add_row("Strategic Intent Accuracy", f"{res.strategic_intent_accuracy*100:.1f}%")
    table.add_row("Average Latency", f"{res.avg_latency_ms:.3f} ms (p95: {res.p95_latency_ms:.3f} ms)")
    table.add_row("Win Rate vs Random", f"{res.win_rate_vs_random*100:.1f}% (Draws: {res.draw_rate_vs_random*100:.1f}%)")
    table.add_row("Self-Play Draw Rate", f"{res.self_play_draw_rate*100:.1f}%")
    console.print(table)


@cli.command("benchmark-jev")
def benchmark_jev():
    """Run EXP-011 empirical benchmark: Pandu-Jev NLP vs Laya-CoreML."""
    from experiments.benchmarks.exp011_pandu_vs_laya import run_exp011_benchmark
    run_exp011_benchmark()


@cli.command("benchmark-three-way")
@click.option("--seeds", default="101,102", help="Comma-separated random seeds (default: 101,102)")
@click.option("--steps", default=8, help="Steps per episode (default: 8)")
def benchmark_three_way(seeds: str, steps: int):
    """Run EXP-012 Three-Way Arena: Pandu-Jev vs Laya-CoreML vs Jev (TypeSafe)."""
    from experiments.benchmarks.exp012_three_way_arena import run_three_way_benchmark
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    run_three_way_benchmark(seeds=seed_list, steps_per_seed=steps)


@cli.command("route-model")
@click.argument("prompt")
@click.option("--base", is_flag=True, help="Use 149M ModernBERT-Base")
def route_model_cmd(prompt: str, base: bool):
    """Route a software task to the optimal model tier in <25ms."""
    from models.router import PanduRouter
    router = PanduRouter(use_base=base)
    res = router.route_model(prompt)

    table = Table(title=f"Pandu TypeSafe Router ({router.metadata['tier']} on {router.metadata['device']})")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Task Prompt", prompt)
    table.add_row("Selected Tier", f"[bold green]{res.selected_model}[/bold green]")
    table.add_row("Confidence", f"{res.confidence:.4f}")
    table.add_row("Requires Reasoning (Noul)", f"{res.needs_reasoning:.4f}")
    table.add_row("Is High Risk (Noul)", f"{res.is_high_risk:.4f}")
    table.add_row("Complexity Level (Score)", f"{res.difficulty_score:.2f} / 4.0")
    table.add_row("Latency", f"{res.latency_ms:.2f} ms")
    table.add_row("Tokens Generated", f"{res.output_tokens} (Zero-Token Invariant)")
    console.print(table)


@cli.command("route-tool")
@click.argument("agent_state")
@click.option("--base", is_flag=True, help="Use 149M ModernBERT-Base")
def route_tool_cmd(agent_state: str, base: bool):
    """Dispatch the immediate next tool for an autonomous coding agent."""
    from models.router import PanduRouter
    router = PanduRouter(use_base=base)
    res = router.select_tool(agent_state)

    table = Table(title=f"Pandu Fast Tool Dispatcher ({router.metadata['tier']} on {router.metadata['device']})")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Agent State", agent_state)
    table.add_row("Selected Tool", f"[bold green]{res.selected_tool}[/bold green]")
    table.add_row("Confidence", f"{res.confidence:.4f}")
    table.add_row("Needs Code Read (Noul)", f"{res.needs_read:.4f}")
    table.add_row("Task Complete (Noul)", f"{res.is_done:.4f}")
    table.add_row("Latency", f"{res.latency_ms:.2f} ms")
    console.print(table)


@cli.command("triage-code")
@click.argument("code_snippet")
@click.option("--base", is_flag=True, help="Use 149M ModernBERT-Base")
def triage_code_cmd(code_snippet: str, base: bool):
    """Evaluate code quality and security vulnerability in <25ms."""
    from models.router import PanduRouter
    router = PanduRouter(use_base=base)
    res = router.triage_code(code_snippet)

    table = Table(title=f"Pandu Code Triage ({router.metadata['tier']} on {router.metadata['device']})")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Verdict", f"[bold green]{res.verdict}[/bold green]")
    table.add_row("Confidence", f"{res.confidence:.4f}")
    table.add_row("Has Vulnerability (Noul)", f"{res.has_vulnerability:.4f}")
    table.add_row("Quality Level (Score)", f"{res.quality_score:.2f} / 4.0")
    table.add_row("Latency", f"{res.latency_ms:.2f} ms")
    console.print(table)


@cli.command("train-universal")
@click.option("--base", is_flag=True, help="Train 149M ModernBERT-Base")
@click.option("--epochs", default=8, help="Number of training epochs")
@click.option("--lr", default=2e-4, help="Learning rate")
def train_universal_cmd(base: bool, epochs: int, lr: float):
    """Train Pandu Universal Model on multi-domain System One tasks."""
    from training.universal_trainer import train_universal_pandu
    train_universal_pandu(use_base=base, epochs=epochs, lr=lr)


def main():
    cli()


if __name__ == "__main__":
    main()

