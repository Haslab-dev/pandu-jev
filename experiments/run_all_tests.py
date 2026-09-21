"""Unified test runner for Pandu (pandu-jev).

Runs:
1. Core unit & integration test suite (pytest)
2. Language-conditioned HuggingFace model tests:
   - ModernBERT-Tiny
   - SmolLM2-135M
   - Qwen3-0.6B

Usage:
    python experiments/run_all_tests.py
"""

import sys
import subprocess
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

TEST_SUITES = [
    ("Core Modular Unit Tests", [sys.executable, "-m", "pytest", "tests/", "-v"]),
    ("HuggingFace Test 1: ModernBERT-Tiny", [sys.executable, "experiments/language/modernbert.py"]),
    ("HuggingFace Test 2: SmolLM2-135M", [sys.executable, "experiments/language/smollm2.py"]),
    ("HuggingFace Test 3: Qwen3-0.6B", [sys.executable, "experiments/language/qwen3.py"]),
    ("Grounded Language Cortex Benchmark", [sys.executable, "experiments/language/grounded_cortex.py"]),
]


def main():
    console.print(Panel.fit(
        "[bold cyan]Pandu (pandu-jev) — Unified Test Runner[/bold cyan]\n"
        "Executing all unit tests, calibration checks, and Hugging Face language model tests...",
        border_style="cyan"
    ))

    results = []
    total_start = time.perf_counter()

    for name, cmd in TEST_SUITES:
        console.rule(f"[bold yellow]{name}[/bold yellow]")
        start = time.perf_counter()
        proc = subprocess.run(cmd)
        elapsed = time.perf_counter() - start
        status = "PASSED" if proc.returncode == 0 else "FAILED"
        results.append((name, status, f"{elapsed:.2f}s"))
        if proc.returncode != 0:
            console.print(f"[bold red]❌ {name} failed with exit code {proc.returncode}[/bold red]")

    total_elapsed = time.perf_counter() - total_start

    # Summary Table
    table = Table(title="Test Execution Summary", border_style="green")
    table.add_column("Test Suite", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")

    all_passed = True
    for name, status, duration in results:
        if status == "PASSED":
            st_str = "[bold green]✔ PASSED[/bold green]"
        else:
            st_str = "[bold red]✖ FAILED[/bold red]"
            all_passed = False
        table.add_row(name, st_str, duration)

    console.print()
    console.print(table)
    console.print(f"[dim]Total time: {total_elapsed:.2f}s[/dim]\n")

    if all_passed:
        console.print("[bold green]🎉 All test suites completed successfully![/bold green]")
        sys.exit(0)
    else:
        console.print("[bold red]💥 Some tests failed. Check the logs above.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
