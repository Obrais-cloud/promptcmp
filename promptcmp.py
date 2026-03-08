#!/usr/bin/env python3
"""
promptcmp — compare responses from multiple local Ollama models side-by-side.

Usage:
    promptcmp "your prompt here"
    promptcmp "your prompt" --models deepseek-r1:latest,llama3.3:latest
    echo "your prompt" | promptcmp
    promptcmp "your prompt" --system "You are a concise assistant."
    promptcmp "your prompt" --save
    promptcmp --list
"""

import argparse
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import requests
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

OLLAMA_BASE = "http://localhost:11434"
console = Console()


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def list_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
        # Filter out cloud/remote models (tiny placeholder size < 1MB)
        local = [m["name"] for m in models if m.get("size", 0) > 1_000_000]
        return local or [m["name"] for m in models]
    except Exception as e:
        console.print(f"[red]Cannot reach Ollama at {OLLAMA_BASE}: {e}[/red]")
        sys.exit(1)


def strip_think(text: str) -> str:
    """Strip <think>...</think> reasoning blocks."""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think(?:ing)?>.*$", "", text, flags=re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    model: str
    response: str = ""
    elapsed_s: float = 0.0
    tokens: int = 0
    tps: float = 0.0
    error: str = ""
    done: bool = False

    @property
    def short_name(self) -> str:
        # Truncate long model names for display
        return self.model if len(self.model) <= 28 else self.model[:25] + "…"


# ---------------------------------------------------------------------------
# Query runner (streaming)
# ---------------------------------------------------------------------------

def query_stream(model: str, prompt: str, system: str, result: ModelResponse, num_ctx: int = 0):
    """Run a streaming query and populate result in-place."""
    options: dict = {"temperature": 0.7, "num_predict": 2048}
    if num_ctx:
        options["num_ctx"] = num_ctx
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": options,
    }
    if system:
        payload["system"] = system

    t0 = time.monotonic()
    try:
        with requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            stream=True,
            timeout=180,
        ) as resp:
            if not resp.ok:
                try:
                    err_body = resp.json().get("error", resp.text[:120])
                except Exception:
                    err_body = resp.text[:120]
                # Provide actionable hint for OOM crashes
                if "resource limitations" in err_body or "unexpectedly stopped" in err_body:
                    err_body += "\n[hint: try --num-ctx 4096 to reduce KV cache memory]"
                result.error = err_body
                result.elapsed_s = time.monotonic() - t0
                result.done = True
                return
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result.response += chunk.get("response", "")
                if chunk.get("done"):
                    result.elapsed_s = time.monotonic() - t0
                    result.tokens = chunk.get("eval_count", 0)
                    result.tps = result.tokens / result.elapsed_s if result.elapsed_s > 0 else 0
                    break
    except requests.Timeout:
        result.error = "timeout"
        result.elapsed_s = time.monotonic() - t0
    except Exception as e:
        result.error = str(e)
        result.elapsed_s = time.monotonic() - t0
    finally:
        result.done = True


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_response_panel(r: ModelResponse, width: int) -> Panel:
    if r.error:
        body = f"[red]Error: {r.error}[/red]"
    elif not r.done:
        body = "[dim]running…[/dim]"
    else:
        cleaned = strip_think(r.response)
        body = escape(cleaned) if cleaned else "[dim](empty)[/dim]"

    footer = ""
    if r.done and not r.error:
        footer = f"[dim]{r.elapsed_s:.1f}s · {r.tokens} tok · {r.tps:.0f} tok/s[/dim]"

    title = f"[bold cyan]{escape(r.short_name)}[/bold cyan]"
    if r.done and not r.error:
        title += f"  {footer}"

    return Panel(body, title=title, width=width, border_style="cyan" if r.done else "dim")


def run_comparison(
    prompt: str,
    models: list[str],
    system: str,
    save: bool,
    output_path: Path | None,
    no_think: bool,
    sequential: bool = False,
    num_ctx: int = 0,
):
    results = [ModelResponse(model=m) for m in models]

    console.print()
    console.print(Panel(
        f"[bold]{escape(prompt[:200])}{'…' if len(prompt) > 200 else ''}[/bold]",
        title="[bold green]Prompt[/bold green]",
        border_style="green",
    ))
    if system:
        console.print(Panel(
            f"[dim]{escape(system[:120])}[/dim]",
            title="[dim]System[/dim]",
            border_style="dim",
        ))
    console.print(f"[dim]Models: {', '.join(models)}[/dim]")
    console.print()

    term_width = console.width or 120
    col_width = max(40, (term_width - 4) // len(models))

    def build_display():
        panels = [format_response_panel(r, col_width) for r in results]
        if len(panels) == 1:
            return panels[0]
        return Columns(panels, equal=True, expand=True)

    if sequential:
        # Run one model at a time — better for large models sharing limited VRAM
        for r in results:
            with Live(build_display(), console=console, refresh_per_second=4) as live:
                t = threading.Thread(target=query_stream, args=(r.model, prompt, system, r, num_ctx), daemon=True)
                t.start()
                while not r.done:
                    live.update(build_display())
                    time.sleep(0.25)
                live.update(build_display())
    else:
        # Launch all models in parallel
        threads = []
        for r in results:
            t = threading.Thread(target=query_stream, args=(r.model, prompt, system, r, num_ctx), daemon=True)
            threads.append(t)
            t.start()

        with Live(build_display(), console=console, refresh_per_second=4) as live:
            while not all(r.done for r in results):
                live.update(build_display())
                time.sleep(0.25)
            live.update(build_display())

        # Retry any OOM errors sequentially
        failed = [r for r in results if r.error and "unexpectedly stopped" in r.error]
        if failed:
            console.print(f"\n[yellow]Retrying {len(failed)} model(s) sequentially (memory contention)…[/yellow]")
            for r in failed:
                r.error = ""
                r.response = ""
                r.done = False
                with Live(build_display(), console=console, refresh_per_second=4) as live:
                    t = threading.Thread(target=query_stream, args=(r.model, prompt, system, r, num_ctx), daemon=True)
                    t.start()
                    while not r.done:
                        live.update(build_display())
                        time.sleep(0.25)
                    live.update(build_display())

    # Summary table
    console.print()
    console.print(Rule("[dim]Summary[/dim]"))
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Model", style="cyan")
    t.add_column("Time", justify="right")
    t.add_column("Tokens", justify="right")
    t.add_column("tok/s", justify="right")
    t.add_column("Words", justify="right")

    for r in sorted(results, key=lambda x: x.elapsed_s if x.elapsed_s > 0 else 9999):
        if r.error:
            t.add_row(r.short_name, "[red]error[/red]", "—", "—", "—")
        else:
            cleaned = strip_think(r.response) if no_think else r.response
            words = len(cleaned.split())
            fastest = min((x.tps for x in results if x.tps > 0), default=0)
            tps_str = f"[green]{r.tps:.0f}[/green]" if r.tps == fastest and fastest > 0 else f"{r.tps:.0f}"
            t.add_row(r.short_name, f"{r.elapsed_s:.1f}s", str(r.tokens), tps_str, str(words))
    console.print(t)

    if save:
        path = output_path or Path(f"promptcmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        save_markdown(prompt, system, results, path, no_think)
        console.print(f"\n[dim]Saved → {path}[/dim]")


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def save_markdown(
    prompt: str,
    system: str,
    results: list[ModelResponse],
    path: Path,
    no_think: bool,
):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# promptcmp — {ts}",
        "",
        f"**Prompt**: {prompt}",
        "",
    ]
    if system:
        lines += [f"**System**: {system}", ""]

    lines += [
        "## Summary",
        "",
        "| Model | Time | Tokens | tok/s | Words |",
        "|-------|------|--------|-------|-------|",
    ]
    for r in sorted(results, key=lambda x: x.elapsed_s if x.elapsed_s > 0 else 9999):
        if r.error:
            lines.append(f"| `{r.model}` | error | — | — | — |")
        else:
            cleaned = strip_think(r.response) if no_think else r.response
            lines.append(
                f"| `{r.model}` | {r.elapsed_s:.1f}s | {r.tokens} | {r.tps:.0f} | {len(cleaned.split())} |"
            )

    lines += ["", "## Responses", ""]
    for r in results:
        lines.append(f"### {r.model}")
        lines.append("")
        if r.error:
            lines.append(f"*Error: {r.error}*")
        else:
            cleaned = strip_think(r.response) if no_think else r.response
            lines.append(cleaned)
        lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# --bench mode: run localeval benchmarks inside promptcmp
# ---------------------------------------------------------------------------

def _load_localeval():
    """Import localeval module from sibling dir or ~/localeval/."""
    import importlib
    candidates = [
        Path(__file__).parent / "localeval.py",
        Path.home() / "localeval" / "localeval.py",
    ]
    for p in candidates:
        if p.exists():
            parent = str(p.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return importlib.import_module("localeval")
    return None


def run_bench_mode(args):
    """Run localeval benchmark suites and show a leaderboard."""
    le = _load_localeval()
    if le is None:
        console.print(
            "[red]localeval.py not found.[/red]\n"
            "Put it alongside promptcmp.py or in ~/localeval/localeval.py\n"
            "Get it: https://github.com/Obrais-cloud/localeval"
        )
        sys.exit(1)

    suite_arg = args.bench or "all"
    all_suites = list(le.SUITES.keys())
    if suite_arg == "all":
        suite_names = all_suites
    else:
        suite_names = [s.strip() for s in suite_arg.split(",")]
        bad = [s for s in suite_names if s not in le.SUITES]
        if bad:
            console.print(f"[red]Unknown suites: {bad}. Available: {all_suites}[/red]")
            sys.exit(1)

    questions = []
    for s in suite_names:
        questions.extend(le.SUITES[s])

    available = list_models()
    models = [m.strip() for m in args.models.split(",")] if args.models else available

    console.print()
    console.print(Panel(
        f"[bold]Models:[/bold] {', '.join(models)}\n"
        f"[bold]Suites:[/bold] {', '.join(suite_names)}\n"
        f"[bold]Questions:[/bold] {len(questions)} × {len(models)} = "
        f"{len(questions) * len(models)} total queries",
        title="[bold green]promptcmp --bench[/bold green]",
        border_style="green",
    ))

    results = le.run_evals(models, questions)
    le.print_leaderboard(results, suite_names)

    if getattr(args, "save", False) or getattr(args, "output", None):
        out = Path(args.output) if getattr(args, "output", None) else \
            Path(f"promptcmp_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        le.save_markdown(results, suite_names, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="promptcmp — compare responses from multiple local Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  promptcmp "Explain recursion in one sentence."
  promptcmp "Write a haiku about debugging." --models deepseek-r1:latest,llama3.3:latest
  promptcmp "What is 2+2?" --save
  echo "Summarize this" | promptcmp
  promptcmp --list
""",
    )
    parser.add_argument("prompt", nargs="?", help="The prompt to send (or pipe via stdin)")
    parser.add_argument("--models", "-m", help="Comma-separated model names (default: all local)")
    parser.add_argument("--system", "-s", default="", help="System prompt")
    parser.add_argument("--save", action="store_true", help="Save results to a markdown file")
    parser.add_argument("--output", "-o", help="Output file path (implies --save)")
    parser.add_argument("--no-think", action="store_true", default=True,
                        help="Strip <think> blocks from output (default: on)")
    parser.add_argument("--keep-think", action="store_true",
                        help="Keep <think> reasoning blocks in output")
    parser.add_argument("--sequential", "-S", action="store_true",
                        help="Run models one at a time (better for large models with limited VRAM)")
    parser.add_argument("--num-ctx", type=int, default=0,
                        help="Context window size. Reduce (e.g. 4096) to fix OOM on large models (default: model default)")
    parser.add_argument("--bench", "-b", nargs="?", const="all", metavar="SUITE",
                        help="Run structured benchmarks instead of a prompt. "
                             "SUITE: math, reasoning, coding, general, or all (default: all). "
                             "Requires localeval.py in the same dir or ~/localeval/")
    parser.add_argument("--list", "-l", action="store_true", help="List available local models")
    parser.add_argument("--json", action="store_true", help="Also dump JSON results to stdout")

    args = parser.parse_args()

    if args.list:
        models = list_models()
        for m in models:
            console.print(f"  [cyan]•[/cyan] {m}")
        return

    # --bench mode: delegate to localeval
    if args.bench:
        run_bench_mode(args)
        return

    # Resolve prompt
    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        parser.print_help()
        sys.exit(1)

    # Resolve models
    available = list_models()
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
        bad = [m for m in models if m not in available]
        if bad:
            # Try available list (may include cloud models)
            all_models = [m["name"] for m in requests.get(f"{OLLAMA_BASE}/api/tags").json().get("models", [])]
            bad2 = [m for m in models if m not in all_models]
            if bad2:
                console.print(f"[red]Unknown models: {bad2}[/red]")
                console.print(f"Available: {available}")
                sys.exit(1)
    else:
        models = available

    if not models:
        console.print("[red]No local models found. Pull one with: ollama pull llama3.2[/red]")
        sys.exit(1)

    no_think = not args.keep_think
    output_path = Path(args.output) if args.output else None
    save = args.save or bool(args.output)

    run_comparison(prompt, models, args.system, save, output_path, no_think,
                   sequential=args.sequential, num_ctx=args.num_ctx)

    if args.json:
        # Re-run would be needed; just print what we have from the last run
        # This would need refactor to return results — skip for now
        console.print("[dim]--json not yet implemented[/dim]")


if __name__ == "__main__":
    main()
