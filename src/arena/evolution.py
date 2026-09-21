"""Self-Play & Evolutionary Tournament Engine for Pandu (pandu-jev).

Provides:
- EvolutionaryPolicyLeague: Maintains a population of Pandu micro-policies.
- Population Tournament: Round-robin evaluation between population members.
- Evolutionary Update: Fitness-weighted mutation / selection to breed
  progressively superior micro-policies without external reward shaping or APIs.
"""

import copy
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arena.base import ArenaMatch
from arena.snake import CompetitiveSnakeEnv, PanduSnakeBot
from models.policy import TinyPolicy

console = Console()


class EvolutionaryPolicyLeague:
    """Manages an evolutionary population of micro-policies."""

    def __init__(
        self,
        population_size: int = 6,
        input_dim: int = 20,
        num_actions: int = 4,
        hidden_dims: Tuple[int, int] = (32, 64),
        seed: int = 42,
    ):
        self.pop_size = population_size
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dims = hidden_dims
        self.rng = np.random.RandomState(seed)

        # Initialize population
        self.population: List[TinyPolicy] = []
        for i in range(population_size):
            p = TinyPolicy(input_dim=input_dim, num_actions=num_actions, hidden_dims=hidden_dims)
            # Add diverse initial weights
            with torch.no_grad():
                for param in p.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            self.population.append(p)

    def run_generation(
        self,
        games_per_pair: int = 4,
        mutation_rate: float = 0.05,
        elite_count: int = 2,
    ) -> Dict[str, Any]:
        """Evaluate population round-robin and produce the next generation."""
        n = len(self.population)
        scores = np.zeros(n, dtype=np.float32)
        wins = np.zeros(n, dtype=np.int32)
        food_eaten = np.zeros(n, dtype=np.int32)

        # Round-robin tournament
        for i in range(n):
            for j in range(i + 1, n):
                bot_i = PanduSnakeBot(self.population[i], name=f"Agent-{i}")
                bot_j = PanduSnakeBot(self.population[j], name=f"Agent-{j}")

                for g in range(games_per_pair):
                    env = CompetitiveSnakeEnv()
                    # Alternate starting sides
                    if g % 2 == 0:
                        match = ArenaMatch(env, bot_i, bot_j)
                        res = match.play(max_steps=250)
                        if res["winner"] == "A":
                            wins[i] += 1
                            scores[i] += 3.0
                        elif res["winner"] == "B":
                            wins[j] += 1
                            scores[j] += 3.0
                        else:
                            scores[i] += 1.0
                            scores[j] += 1.0
                        food_eaten[i] += res["info"]["score_a"]
                        food_eaten[j] += res["info"]["score_b"]
                    else:
                        match = ArenaMatch(env, bot_j, bot_i)
                        res = match.play(max_steps=250)
                        if res["winner"] == "A":
                            wins[j] += 1
                            scores[j] += 3.0
                        elif res["winner"] == "B":
                            wins[i] += 1
                            scores[i] += 3.0
                        else:
                            scores[i] += 1.0
                            scores[j] += 1.0
                        food_eaten[j] += res["info"]["score_a"]
                        food_eaten[i] += res["info"]["score_b"]

        # Composite fitness: match points + food eaten * 5
        fitness = scores + (food_eaten * 5.0)

        # Rank population
        ranking = np.argsort(-fitness)
        best_idx = int(ranking[0])
        best_fitness = float(fitness[best_idx])
        avg_fitness = float(np.mean(fitness))

        # Breed next generation (elitism + Gaussian mutation)
        new_population = []
        for idx in ranking[:elite_count]:
            new_population.append(copy.deepcopy(self.population[idx]))

        while len(new_population) < self.pop_size:
            # Pick a parent from top half
            parent_idx = int(ranking[self.rng.randint(0, max(1, self.pop_size // 2))])
            child = copy.deepcopy(self.population[parent_idx])
            with torch.no_grad():
                for param in child.parameters():
                    noise = torch.randn_like(param) * mutation_rate
                    param.add_(noise)
            new_population.append(child)

        self.population = new_population

        return {
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "best_bot_id": best_idx,
            "total_food": int(np.sum(food_eaten)),
            "total_wins": int(np.sum(wins)),
        }


def run_evolutionary_experiment(generations: int = 5, population_size: int = 6) -> List[Dict[str, Any]]:
    """Run multi-generation evolution of Pandu micro-policies."""
    console.print(Panel(
        f"[bold cyan]Pandu Evolutionary Self-Play League[/bold cyan]\n"
        f"[italic]Evolving {population_size} micro-policies across {generations} tournament generations[/italic]",
        border_style="cyan",
    ))

    league = EvolutionaryPolicyLeague(population_size=population_size)
    history = []

    table = Table(title="Evolutionary Generation Progress", header_style="bold magenta")
    table.add_column("Gen", justify="center")
    table.add_column("Best Fitness", justify="right")
    table.add_column("Avg Fitness", justify="right")
    table.add_column("Total Food Eaten", justify="right")
    table.add_column("Best Agent", justify="center")

    t0 = time.time()
    for gen in range(generations):
        stats = league.run_generation(games_per_pair=2)
        history.append(stats)
        table.add_row(
            f"Gen {gen}",
            f"{stats['best_fitness']:.1f}",
            f"{stats['avg_fitness']:.1f}",
            str(stats["total_food"]),
            f"Agent-{stats['best_bot_id']}",
        )

    console.print(table)
    console.print(f"[bold green]✓ Evolution finished in {time.time() - t0:.2f}s[/bold green]")
    return history
