# automl-research

[![PyPI version](https://img.shields.io/pypi/v/automl-research)](https://pypi.org/project/automl-research/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://pypi.org/project/automl-research/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Autonomous ML experimentation for any project. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the same constraint-driven experiment loop, adapted to work with **any ML project, any framework, any AI agent**.

The key idea: your AI coding agent (Claude Code, Cursor, Aider, Codex, etc.) is already an ML expert. This tool gives it a structured experiment loop — init, preflight, train, decide, repeat — and a research pipeline that fetches real papers. The agent does the thinking. The tool does the bookkeeping.

## Install

```bash
pip install automl-research

# With WandB support:
pip install "automl-research[wandb]"
```

> **Currently on TestPyPI** — while we're in early release:
> ```bash
> pip install -i https://test.pypi.org/simple/ automl-research
> ```

From source:

```bash
git clone https://github.com/paraskoundal/automl-research.git
cd automl-research
pip install -e .
```

## Quick Start

```bash
cd your-ml-project

# 1. Initialize (auto-scans your project)
automl-research init

# 2. Point any AI agent at the generated instructions
#    Open Cursor, Claude Code, or any tool and say:
#    "Read .automl-research/program.md and start experimenting"
```

That's it. The agent reads `program.md`, understands your project config, and starts running experiments autonomously.

## How It Works

```
Think → Modify → Preflight → Train → Decide → Repeat
```

The tool provides CLI commands that form an autonomous experiment loop. Your AI agent calls these commands, interprets the results, and iterates.

### Commands

| Command | What it does |
|---------|-------------|
| `automl-research init` | Scan project, create `.automl-research/` with config + agent instructions |
| `automl-research preflight` | Validate syntax, configs, diff size before training |
| `automl-research train` | Run training with timeout + log capture |
| `automl-research decide -d "..."` | Extract metrics, keep/discard, git commit/revert, log everything |
| `automl-research status` | Show current experiment count, best metric, last result |
| `automl-research summary` | Regenerate experiment summary from results |
| `automl-research analyze` | Generate progress chart |
| `automl-research research -k "..."` | Search papers for experiment ideas (Semantic Scholar, ArXiv, OpenReview) |

### The Decide Command

One command handles the entire keep/discard decision:

```bash
automl-research decide --description "increase lr to 0.01"
```

1. Extracts metrics from the latest run log
2. Compares primary metric to best (respects direction + min improvement threshold)
3. Checks constraints (VRAM, etc.)
4. Git commits (keep) or reverts (discard)
5. Logs to `results.tsv`, updates `state.json` and `summary.md`
6. Generates progress chart
7. Logs to WandB (if enabled)

For crashed runs:

```bash
automl-research decide --crash --description "OOM on wider model"
```

### Research Pipeline

The research command fetches papers from Semantic Scholar, ArXiv, and OpenReview:

```bash
# Search by keyword
automl-research research -k "cosine annealing warm restarts"

# Context-aware: analyzes your experiment history and searches adaptively
automl-research research --deep

# Filter by source
automl-research research -k "optimizer" -s arxiv -s semantic_scholar
```

Papers are written to `ideas.md` with titles, years, citation counts, TLDRs, and code links. Your AI agent reads these and decides what's worth trying — it understands ML better than any keyword filter.

## Project Structure

After `automl-research init`:

```
your_project/
├── (existing files...)
└── .automl-research/
    ├── project.yaml    # Project config (auto-detected + user-refined)
    ├── program.md      # Agent instructions (auto-generated)
    ├── state.json      # Resume state (best metric, experiment count)
    ├── results.tsv     # Full experiment log
    ├── summary.md      # What worked / what didn't
    ├── ideas.md        # Paper-backed experiment ideas
    ├── progress.png    # Progress chart
    └── runs/           # Per-run logs
```

## Templates

Start from a template instead of interactive setup:

```bash
automl-research init --template simple_script     # Single train.py
automl-research init --template pytorch_config     # Config-driven PyTorch
automl-research init --template multi_stage        # Pretrain → finetune
```

## Configuration

`project.yaml` supports:

- **Multiple training speeds**: `command_quick` (fast iteration) + `command_full` (validation)
- **Flexible metric extraction**: log grep, JSON file, CSV file
- **Constraints**: auto-discard experiments that exceed limits (e.g., VRAM)
- **Min improvement threshold**: reject noise
- **File boundaries**: explicit modifiable / read-only / forbidden lists
- **Research config**: keywords, ArXiv categories, paper sources
- **WandB integration**: optional experiment tracking

## Agent-Agnostic

The generated `program.md` works with any AI coding agent — Cursor, Claude Code, Aider, Codex, Windsurf, or anything that can read files and run shell commands. The agent is the intelligence layer; the tool is the data pipeline.

## WandB Integration

Enable in `project.yaml`:

```yaml
tracking:
  wandb:
    enabled: true
    project: "my-automl-experiments"
```

Each experiment becomes a WandB run with metrics, git diff, and progress chart.

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

## License

[MIT](LICENSE)
