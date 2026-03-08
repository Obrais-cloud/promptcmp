# promptcmp

Compare responses from multiple local Ollama models side-by-side in your terminal.

Ask a question once, get every model's answer at the same time. Great for choosing the right model for a task, or just seeing how they differ.

```
╭──────────────────────────────────╮  ╭──────────────────────────────────╮
│  deepseek-r1  11s · 422 tok · 37/s│  │  llama3.3  8s · 310 tok · 39/s  │
│                                  │  │                                  │
│  Recursion is a method where a   │  │  Recursion is when a function    │
│  function calls itself to solve  │  │  calls itself to break a problem │
│  a problem by breaking it into   │  │  into smaller pieces until it    │
│  smaller versions…               │  │  hits a base case.               │
╰──────────────────────────────────╯  ╰──────────────────────────────────╘
```

## Features

- Runs all local Ollama models in **parallel** by default
- **Strips `<think>` blocks** from reasoning models (deepseek-r1, qwq, etc.)
- Live streaming output as responses come in
- Auto-retry with sequential fallback if models 500 (memory contention)
- Saves a clean markdown report
- `--sequential` mode for large models sharing limited VRAM

## Requirements

- Python 3.11+
- Ollama running locally
- `pip install requests rich`

## Installation

```bash
git clone https://github.com/Obrais-cloud/promptcmp
cd promptcmp
pip install -e .
```

Or run directly:

```bash
python promptcmp.py "your prompt here"
```

## Usage

```bash
# Compare all local models
promptcmp "Explain recursion in one sentence."

# Specific models
promptcmp "Write a haiku about bugs." --models deepseek-r1:latest,llama3.3:latest

# With a system prompt
promptcmp "Review this code" --system "You are a senior Python developer."

# Pipe a prompt
echo "What is the meaning of life?" | promptcmp

# Save results to markdown
promptcmp "Explain quantum entanglement." --save

# Custom output path
promptcmp "Summarise the CAP theorem." --output cap_results.md

# Sequential mode (better for large models on limited VRAM)
promptcmp "Write a sorting algorithm." --sequential

# List available local models
promptcmp --list

# Keep <think> reasoning blocks in output
promptcmp "Solve this logic puzzle…" --keep-think
```

## Notes on large models

Running models like `llama3.3:70b` concurrently with other large models may cause 500 errors if your system runs out of memory. `promptcmp` will auto-retry failed models sequentially, or use `--sequential` to avoid the issue entirely.

Cloud/remote Ollama models (e.g. `glm-5:cloud`) are automatically excluded from the default model list but can be specified explicitly with `--models`.

## Companion tools

- [localeval](https://github.com/Obrais-cloud/localeval) — structured benchmarks with auto-scoring

## License

MIT
