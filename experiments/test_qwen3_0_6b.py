"""Test 3: Qwen 0.6B (Qwen/Qwen3-0.6B) Policy via Hugging Face.

Architecture: Qwen3-0.6B (~590M parameters)
Task: Multi-turn reasoning, language-conditioned decision making, and action probability extraction.

Usage:
    python experiments/test_qwen3_0_6b.py
    python experiments/test_qwen3_0_6b.py --download-weights
"""

import time
import math
from typing import Dict, Any, List
import click
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

console = Console()

MODEL_ID = "Qwen/Qwen3-0.6B"
ACTION_CANDIDATES = ["UP", "DOWN", "LEFT", "RIGHT"]


def build_qwen_prompt(obs: Dict[str, Any], instruction: str, tokenizer: Any) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert autonomous reasoning policy. Given a state description "
                "and objective, select the optimal next movement. "
                "Select exactly one action from: UP, DOWN, LEFT, RIGHT."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Mission: {instruction}\n"
                f"Agent Coordinates: ({obs['agent_x']}, {obs['agent_y']})\n"
                f"Target Coordinates: ({obs['goal_x']}, {obs['goal_y']})\n"
                f"Adjacent Obstacles: {obs['walls_near']}\n"
                f"Optimal Next Action:"
            ),
        },
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Mission: {instruction}. Pos: ({obs['agent_x']},{obs['agent_y']}). Goal: ({obs['goal_x']},{obs['goal_y']}). Action:"


@click.command()
@click.option("--download-weights", is_flag=True, help="Download full 980MB weights from HuggingFace Hub.")
def main(download_weights: bool):
    console.print(Panel.fit(
        f"[bold cyan]Test 3: Qwen 0.6B from Hugging Face[/bold cyan]\n"
        f"Model ID: [yellow]{MODEL_ID}[/yellow]\n"
        f"Role: High-Capacity Language & Multi-Step Reasoning Policy (~500M Parameters)"
    ))

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    console.print(f"Hardware Compute Device: [magenta]{device}[/magenta] (Precision: {dtype})")

    # Load Tokenizer from Hugging Face Hub
    console.print(f"Loading Hugging Face Tokenizer: [cyan]{MODEL_ID}[/cyan]...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if download_weights:
        console.print(f"Downloading full weights for [yellow]{MODEL_ID}[/yellow] from HuggingFace Hub...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if device != "cpu" else None,
        )
    else:
        console.print(f"Instantiating Qwen 0.6B architecture from Hugging Face config...")
        config = AutoConfig.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_config(config)
        model.to(dtype=dtype, device=device)

    if device == "cpu":
        model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    console.print(f"Total Parameters: [bold green]{n_params:,}[/bold green] (~{n_params/1e6:.1f}M params)")
    console.print(f"Memory Footprint ({dtype}): [bold green]{mem_mb:.2f} MB[/bold green]\n")

    # Map candidate tokens
    candidate_token_ids = {}
    for act in ACTION_CANDIDATES:
        toks = tokenizer.encode(" " + act, add_special_tokens=False) or tokenizer.encode(act, add_special_tokens=False)
        candidate_token_ids[act] = toks[-1]

    # Scenarios
    test_cases = [
        {
            "desc": "Direct Corridor East",
            "instruction": "Advance eastward directly towards the extraction point.",
            "obs": {"agent_x": 1, "agent_y": 5, "goal_x": 10, "goal_y": 5, "walls_near": "UP, DOWN"},
        },
        {
            "desc": "Hazard Evasion South",
            "instruction": "Bypass the barrier and head south through the open channel.",
            "obs": {"agent_x": 3, "agent_y": 1, "goal_x": 3, "goal_y": 6, "walls_near": "RIGHT, UP"},
        },
        {
            "desc": "Complex Maze Detour",
            "instruction": "Route around impassable blockage to maintain trajectory.",
            "obs": {"agent_x": 6, "agent_y": 4, "goal_x": 2, "goal_y": 2, "walls_near": "UP, RIGHT"},
        },
    ]

    # Warmup
    dummy_input = tokenizer("Warmup query", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**dummy_input)

    table = Table(title="Qwen 0.6B Language-Conditioned Policy Decisions")
    table.add_column("Scenario", style="bold")
    table.add_column("Instruction", style="dim")
    table.add_column("Predicted", style="cyan")
    table.add_column("Candidate Probabilities", style="magenta")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Latency", justify="right", style="yellow")

    latencies = []

    for case in test_cases:
        prompt_text = build_qwen_prompt(case["obs"], case["instruction"], tokenizer)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            act_logits = torch.tensor([next_token_logits[candidate_token_ids[act]].item() for act in ACTION_CANDIDATES])
            act_probs = F.softmax(act_logits, dim=-1).cpu().numpy()
        t_end = time.perf_counter()

        lat_ms = (t_end - t_start) * 1000.0
        latencies.append(lat_ms)

        best_idx = int(act_probs.argmax())
        best_action = ACTION_CANDIDATES[best_idx]
        confidence = float(act_probs[best_idx])

        prob_str = " | ".join(f"{ACTION_CANDIDATES[i]}:{act_probs[i]*100:.1f}%" for i in range(4))

        table.add_row(
            case["desc"],
            case["instruction"][:30] + "...",
            best_action,
            prob_str,
            f"{confidence*100:.1f}%",
            f"{lat_ms:.2f} ms",
        )

    console.print(table)

    avg_lat = sum(latencies) / len(latencies)
    throughput = 1000.0 / max(0.001, avg_lat)

    console.print(Panel.fit(
        f"[bold green]Qwen 0.6B Benchmark Summary[/bold green]\n"
        f"• Steady-State Latency: [bold]{avg_lat:.2f} ms[/bold]\n"
        f"• Throughput: [bold]~{throughput:.0f} decisions/sec[/bold]\n"
        f"• Parameter Budget: [bold]{n_params:,}[/bold] ({mem_mb:.1f} MB in {dtype})\n"
        f"• Suitability: High-capacity multi-step reasoning policy; ideal for deliberate local fallback."
    ))


if __name__ == "__main__":
    main()
