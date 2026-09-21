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


@cli.command("test-all")
def test_all():
    """Execute all test suites (unit tests + language models) in one command."""
    from experiments.run_all_tests import main as run_all
    run_all()


def main():
    cli()


if __name__ == "__main__":
    main()
