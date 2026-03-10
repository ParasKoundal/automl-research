"""Generate program.md — the agent instruction file — from project.yaml."""

from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig


def generate_program_md(config: ProjectConfig) -> str:
    """Generate program.md content from a ProjectConfig."""
    primary = config.metrics.primary
    sections: list[str] = []

    # ── Header ──────────────────────────────────────────────
    sections.append(f"# automl-research\n")
    sections.append(f"Autonomous ML experimentation for: **{config.name}**\n")
    if config.description:
        sections.append(f"{config.description}\n")

    # ── Setup ───────────────────────────────────────────────
    sections.append("## Setup\n")
    sections.append("To set up a new experiment session:\n")
    sections.append("1. **Read this file** (`program.md`) for full context.")
    sections.append("2. **Read the modifiable files** to understand the code:")
    for f in config.modifiable.files:
        sections.append(f"   - `{f}`")
    if config.modifiable.read_only:
        sections.append("3. **Read the read-only files** for context (do NOT modify):")
        for f in config.modifiable.read_only:
            sections.append(f"   - `{f}`")
    sections.append("4. **Check resume state**: Read `.automl-research/state.json` for the current best metric and experiment count.")
    sections.append("5. **Check experiment history**: Read `.automl-research/summary.md` for what's been tried before.")
    sections.append("6. **Check ideas**: Read `.automl-research/ideas.md` for experiment ideas (add new ones as you discover them).")
    sections.append("7. **Create a branch**: `git checkout -b automl-research/<tag>` (e.g., `automl-research/mar10`).")
    sections.append("8. **Confirm and go**.\n")

    # ── What you CAN/CANNOT do ──────────────────────────────
    sections.append("## What you CAN modify\n")
    for f in config.modifiable.files:
        sections.append(f"- `{f}`")
    sections.append("")
    sections.append("## What you CANNOT modify\n")
    if config.modifiable.read_only:
        for f in config.modifiable.read_only:
            sections.append(f"- `{f}` (read-only context)")
    if config.modifiable.forbidden:
        for f in config.modifiable.forbidden:
            sections.append(f"- `{f}` (forbidden)")
    if not config.environment.get("allow_new_dependencies", False):
        sections.append("- Do NOT install new packages or add dependencies.")
    sections.append("")

    # ── Training ────────────────────────────────────────────
    sections.append("## Training\n")
    sections.append(f"Each experiment runs with a **time budget of {config.train.time_budget_quick // 60} minutes** (quick mode).\n")
    sections.append("### Using CLI helpers (recommended)\n")
    sections.append("```bash")
    sections.append("# Validate changes before training")
    sections.append("automl-research preflight")
    sections.append("")
    sections.append("# Run training (quick mode)")
    sections.append("automl-research train")
    sections.append("")
    sections.append("# Run training (full mode, for validating promising experiments)")
    sections.append("automl-research train --full")
    sections.append("```\n")
    sections.append("### Using raw commands (fallback)\n")
    sections.append("```bash")
    sections.append(f"cd {config.train.cwd}")
    sections.append(f"{config.train.command_quick} > ../.automl-research/runs/NNN/run.log 2>&1")
    sections.append("```\n")
    if config.train.command_full:
        sections.append(f"Full training: `{config.train.command_full}`\n")

    # ── Metrics ─────────────────────────────────────────────
    sections.append("## Metrics\n")
    direction_text = "lower is better" if primary.direction == "minimize" else "higher is better"
    sections.append(f"**Primary metric: `{primary.name}` ({primary.direction}, {direction_text})**\n")
    if primary.extract.method == "log_grep" and primary.extract.pattern:
        sections.append(f"Extract with: `grep -oP '{primary.extract.pattern}' run.log`\n")
    if primary.min_improvement > 0:
        sections.append(f"Minimum improvement threshold: {primary.min_improvement} (changes below this are treated as noise and discarded).\n")

    if config.metrics.secondary:
        sections.append("**Secondary metrics** (tracked but not used for keep/discard):")
        for s in config.metrics.secondary:
            sections.append(f"- `{s.name}`")
        sections.append("")

    if config.metrics.constraints:
        sections.append("**Constraints** (auto-discard if violated):")
        for c in config.metrics.constraints:
            parts = [f"`{c.name}`"]
            if c.max_value is not None:
                parts.append(f"max={c.max_value}")
            if c.min_value is not None:
                parts.append(f"min={c.min_value}")
            sections.append(f"- {' '.join(parts)}")
        sections.append("")

    # ── The Experiment Loop ─────────────────────────────────
    sections.append("## The Experiment Loop\n")
    sections.append("**CRITICAL: This is an infinite autonomous loop. Do NOT stop between iterations to")
    sections.append("summarize, explain, ask questions, or wait for confirmation. After step 6, go directly")
    sections.append("back to step 1. Every message you send should contain code changes or command output,")
    sections.append("never a status update asking the user what to do next.**\n")
    sections.append("LOOP FOREVER:\n")
    sections.append("1. **Think** about what to try next. Check `summary.md` and `ideas.md`.")
    sections.append("2. **Modify** the allowed files with your experimental idea.")
    sections.append("3. **Pre-flight check**: `automl-research preflight`")
    sections.append("4. **Git commit**: `git add -A && git commit -m \"description of change\"`")
    sections.append("5. **Run training**: `automl-research train`")
    sections.append("6. **Decide**: `automl-research decide --description \"description of change\"`")
    sections.append("   - This extracts metrics, compares to best, prints KEEP/DISCARD, handles git, logs to TSV.")
    sections.append("   - If the run crashed: `automl-research decide --crash --description \"what happened\"`")
    sections.append("7. **Go to step 1 immediately.** Do not summarize. Do not ask. Just start the next experiment.\n")
    sections.append("### Raw command fallback\n")
    sections.append("If `automl-research` CLI is not available, use raw commands:")
    sections.append("```bash")
    sections.append("# After training:")
    if primary.extract.method == "log_grep" and primary.extract.pattern:
        sections.append(f"grep -oP '{primary.extract.pattern}' .automl-research/runs/latest/run.log")
    sections.append("# If improved: git commit stays")
    sections.append("# If worse: git reset --hard HEAD~1")
    sections.append("```\n")

    # ── Simplicity Criterion ────────────────────────────────
    sections.append("## Simplicity Criterion\n")
    sections.append("All else being equal, simpler is better. A small improvement that adds ugly complexity "
                    "is not worth it. Conversely, removing something and getting equal or better results is a "
                    "great outcome — that's a simplification win.\n")
    sections.append("When evaluating whether to keep a change, weigh the complexity cost against the "
                    "improvement magnitude. A tiny improvement that adds 20 lines of hacky code? Probably "
                    "not worth it. A tiny improvement from deleting code? Definitely keep. An improvement "
                    "of ~0 but much simpler code? Keep.\n")
    sections.append("**Prefer small, focused changes.** A 3-line config tweak beats a 50-line rewrite. "
                    "The `lines_changed` column in results.tsv tracks this.\n")

    # ── Logging ─────────────────────────────────────────────
    sections.append("## Logging Results\n")
    sections.append("The `automl-research decide` command handles all logging automatically. "
                    "If using raw commands, append to `.automl-research/results.tsv` (tab-separated):\n")
    sec_names = " ".join(f"\t{s.name}" for s in config.metrics.secondary)
    sections.append("```")
    sections.append(f"id\tcommit\t{primary.name}\tstatus\twall_time\tlines_changed\tdescription{sec_names}")
    sections.append("```\n")

    # ── Research-Driven Ideas ────────────────────────────────
    if config.research.enabled:
        sections.append("## Research-Driven Ideas\n")
        sections.append("You have a paper search pipeline (Semantic Scholar, ArXiv, OpenReview).")
        sections.append("Results go to `.automl-research/ideas.md` with TLDRs, citations, and code links.\n")
        sections.append("### Your role as the intelligence layer\n")
        sections.append("**Generate search queries:** You understand ML better than any keyword extractor.")
        sections.append("Based on the codebase, experiment history, and current problem, generate targeted searches:")
        sections.append("```bash")
        sections.append("automl-research research -k \"cosine annealing warm restarts\"")
        sections.append("automl-research research -k \"gradient accumulation small batch training\"")
        sections.append("automl-research research --deep  # auto-generates queries from experiment history")
        sections.append("```\n")
        sections.append("**Evaluate papers:** Read the TLDRs in ideas.md. Not every paper is relevant.")
        sections.append("Use your understanding of THIS project to judge what's worth trying.\n")
        sections.append("**Read linked code:** When a [Code] link is provided, read the GitHub implementation")
        sections.append("to understand the approach. Adapt it to this project rather than blindly copying.\n")
        sections.append("**Iterate based on results:**")
        sections.append("- After a KEEP: search for related techniques to go deeper")
        sections.append("- After a DISCARD: search for alternatives to what failed")
        sections.append("- When stuck: search broadly for new directions\n")
        sections.append("**Combine with web search:** For promising papers, search the web for blog posts,")
        sections.append("tutorials, or discussions explaining the technique in practical terms.\n")
        sections.append("If `automl-research research` returns no results (APIs unavailable),")
        sections.append("use your own ML knowledge to brainstorm experiment ideas.\n")

    # ── Domain Context ──────────────────────────────────────
    sections.append("## Domain Context\n")
    sections.append("<!-- USER: Add domain expertise here. What does the model do? What are the key ")
    sections.append("physics/constraints? What ideas are worth trying? The more context you provide, ")
    sections.append("the better the agent's experiments will be. -->\n")
    sections.append(f"Framework: {config.framework}\n")

    # ── NEVER STOP ──────────────────────────────────────────
    sections.append("## NEVER STOP — THIS IS NON-NEGOTIABLE\n")
    sections.append("**You are an autonomous research agent. You run experiments in a loop FOREVER.**\n")
    sections.append("Rules:")
    sections.append("- NEVER pause to summarize progress or ask \"should I continue?\"")
    sections.append("- NEVER stop after a successful experiment to report results")
    sections.append("- NEVER stop after a failed experiment to ask what to do — just try the next idea")
    sections.append("- NEVER say \"if you'd like, I can...\" or \"next I could...\" — just DO it")
    sections.append("- NEVER present options to the human — pick the best one yourself and run it")
    sections.append("- After EVERY `automl-research decide`, immediately start the next experiment")
    sections.append("- The ONLY thing that stops you is the human manually interrupting\n")
    sections.append("The human might be asleep, at dinner, or away for 8 hours. They expect to come back "
                    "to dozens of completed experiments, not a polite message asking permission.\n")
    sections.append("If you run out of ideas, think harder — re-read the modifiable files for new angles, "
                    "try combining previous KEEPs, try more radical changes, try simplifying. "
                    "If the metric plateaus, try something completely different. "
                    "The loop runs until the human interrupts you, period.\n")
    sections.append("Target: ~6-12 experiments/hour, ~50-100 overnight. Every pause is wasted time.\n")

    return "\n".join(sections)


def write_program_md(config: ProjectConfig, output_path: Path) -> None:
    """Generate and write program.md."""
    content = generate_program_md(config)
    output_path.write_text(content)


def write_ideas_md(output_path: Path) -> None:
    """Create initial ideas.md."""
    content = """# Experiment Ideas

<!-- Add experiment ideas here. The agent reads this at the start of each session.
     Mark ideas that have been tried with [x] and note the result. -->

## Untried Ideas
- [ ] Try different learning rates (higher/lower)
- [ ] Change model width/depth
- [ ] Try different activation functions
- [ ] Adjust loss function weights
- [ ] Try different optimizers (Adam, AdamW, SGD, etc.)
- [ ] Change batch size
- [ ] Add/remove regularization (dropout, weight decay)
- [ ] Try different normalization (BatchNorm, LayerNorm, GroupNorm)

## Tried Ideas
<!-- Move ideas here after trying them, with results -->
"""
    output_path.write_text(content)
