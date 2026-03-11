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
    sections.append("**IMPORTANT: Enable auto-approve for terminal commands in your editor settings "
                    "(e.g., Cursor → Yolo Mode, or add `automl-research` and `git` to allowed commands). "
                    "The experiment loop must run without manual permission prompts.**\n")
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
    sections.append("7. **Start experimenting** — the experiment branch was created during `automl-research init`.\n")

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
    sections.append("3. **Run everything in ONE command** (preflight → commit → train → decide):")
    sections.append("```bash")
    sections.append("automl-research preflight && git add -A && git commit -m \"description\" && automl-research train && automl-research decide -d \"description\" -n \"reasoning\"")
    sections.append("```")
    sections.append("   - `-d`: Short description of what changed (e.g., \"increase lr to 0.01\")")
    sections.append("   - `-n`: Your reasoning — why you tried this, what you expected, what you learned from the result")
    sections.append("   - If the run crashed: `automl-research decide --crash -d \"what happened\"`")
    sections.append("4. **Go to step 1 immediately.** Do not summarize. Do not ask. Just start the next experiment.\n")
    sections.append("**IMPORTANT: Run step 3 as a SINGLE chained command (&&). Do NOT run preflight, commit,")
    sections.append("train, and decide as separate commands. One command = one permission prompt = faster loop.**\n")

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
    sections.append("The `automl-research decide` command handles all logging automatically "
                    "(results.tsv, summary.md, progress.png, diff.patch, notes.md, git).\n")

    # ── Research-Driven Ideas ────────────────────────────────
    if config.research.enabled:
        sections.append("## Research-Driven Ideas\n")
        sections.append("You have a paper search pipeline (Semantic Scholar, ArXiv, OpenReview).")
        sections.append("Results go to `.automl-research/ideas.md` with TLDRs, citations, and code links.\n")
        sections.append("### When to use research\n")
        sections.append("Research is part of the experiment loop — use it actively, not as an afterthought:")
        sections.append("- **At session start:** Run `automl-research research --deep` to populate ideas.md before your first experiment")
        sections.append("- **Every 3-5 experiments:** Search for papers related to what's working or failing")
        sections.append("- **After a KEEP:** Search for related techniques to go deeper (e.g., kept a learning rate change → search for scheduling papers)")
        sections.append("- **After 2+ consecutive DISCARDs:** Search for alternative approaches — your current direction may be wrong")
        sections.append("- **When ideas.md is empty:** Run research before guessing — papers beat random exploration\n")
        sections.append("### How to search\n")
        sections.append("**Generate targeted queries** based on what you're learning from experiments:")
        sections.append("```bash")
        sections.append("automl-research research -k \"cosine annealing warm restarts\"")
        sections.append("automl-research research -k \"gradient accumulation small batch training\"")
        sections.append("automl-research research --deep  # auto-generates queries from experiment history")
        sections.append("```\n")
        sections.append("**Then use the results:** Read TLDRs in ideas.md, judge what's relevant to THIS project,")
        sections.append("and follow [Code] links to understand implementations before adapting them.\n")
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
