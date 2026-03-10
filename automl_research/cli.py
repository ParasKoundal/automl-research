"""CLI entry point: automl-research init | preflight | train | decide | status | summary | analyze."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import click
import yaml

from .config import ProjectConfig, find_config, load_config
from .context_builder import generate_program_md, write_ideas_md, write_program_md
from .decider import Decision, _load_state, decide
from .runner import execute_training
from .scanner import scan_project


@click.group()
def main():
    """automl-research: Autonomous ML experimentation for any project."""
    pass


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@main.command()
@click.option("--template", type=click.Choice(["simple_script", "pytorch_config", "multi_stage"]),
              help="Start from a template.")
def init(template: str | None):
    """Initialize automl-research in the current project."""
    project_root = Path.cwd()
    ar_dir = project_root / ".automl-research"

    if ar_dir.exists():
        click.echo(f".automl-research/ already exists at {ar_dir}")
        if not click.confirm("Overwrite?"):
            return

    # Scan project
    click.echo("Scanning project...")
    scan = scan_project(project_root)

    if scan["framework"]:
        click.echo(f"  Framework: {scan['framework']}")
    if scan["training_scripts"]:
        click.echo(f"  Training scripts: {', '.join(scan['training_scripts'][:5])}")
    if scan["config_files"]:
        click.echo(f"  Config files: {', '.join(scan['config_files'][:5])}")
    if scan["model_files"]:
        click.echo(f"  Model files: {', '.join(scan['model_files'][:5])}")

    # Load template or build interactively
    if template:
        template_path = Path(__file__).parent / "templates" / f"{template}.yaml"
        if template_path.exists():
            config_text = template_path.read_text()
            click.echo(f"\nLoaded template: {template}")
        else:
            click.echo(f"Template {template} not found, using interactive setup.")
            template = None

    if not template:
        # Interactive setup
        name = click.prompt("Project name", default=project_root.name)
        desc = click.prompt("Short description", default="")
        framework = click.prompt("Framework", default=scan["framework"] or "pytorch")

        # Training command
        default_script = scan["training_scripts"][0] if scan["training_scripts"] else "train.py"
        train_cmd = click.prompt("Training command (quick mode)", default=f"python {default_script}")
        train_full = click.prompt("Full training command (optional, press Enter to skip)", default="")
        cwd = click.prompt("Working directory", default=".")
        budget = click.prompt("Time budget for quick runs", default="10m")

        # Metric
        metric_name = click.prompt("Primary metric name", default="val_loss")
        direction = click.prompt("Direction", type=click.Choice(["minimize", "maximize"]), default="minimize")
        pattern = click.prompt("Grep pattern to extract metric from log",
                               default=f"{metric_name}\\s*[:=]\\s*([0-9.]+)")

        # Modifiable files
        default_files = scan["config_files"][:3] + scan["model_files"][:2]
        files_str = click.prompt("Modifiable files (comma-separated)",
                                 default=", ".join(default_files[:3]) if default_files else "")
        mod_files = [f.strip() for f in files_str.split(",") if f.strip()]

        config_dict = {
            "project": {"name": name, "description": desc, "framework": framework},
            "train": {
                "command_quick": train_cmd,
                "cwd": cwd,
                "time_budget_quick": budget,
            },
            "metrics": {
                "primary": {
                    "name": metric_name,
                    "direction": direction,
                    "min_improvement": 0.001,
                    "extract": {"method": "log_grep", "pattern": pattern},
                },
                "secondary": [],
                "constraints": [],
            },
            "modifiable": {"files": mod_files, "read_only": [], "forbidden": [".automl-research/**"]},
            "environment": {"allow_new_dependencies": False},
            "preflight": {"max_diff_lines": 50, "validate_syntax": True, "validate_configs": True},
            "tracking": {"wandb": {"enabled": False}},
        }
        if train_full:
            config_dict["train"]["command_full"] = train_full
            config_dict["train"]["time_budget_full"] = "2h"

        config_text = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

    # Create directory structure
    ar_dir.mkdir(parents=True, exist_ok=True)
    (ar_dir / "runs").mkdir(exist_ok=True)

    # Write project.yaml
    config_path = ar_dir / "project.yaml"
    config_path.write_text(config_text)
    click.echo(f"\nCreated {config_path}")

    # Write state.json
    state_path = ar_dir / "state.json"
    state_path.write_text(json.dumps({
        "best_commit": None,
        "best_primary_metric": None,
        "best_experiment_id": 0,
        "total_experiments": 0,
        "current_branch": "",
        "last_experiment": {"id": 0, "status": "", "description": ""},
    }, indent=2) + "\n")

    # Generate program.md
    config = load_config(config_path)
    program_path = ar_dir / "program.md"
    write_program_md(config, program_path)
    click.echo(f"Created {program_path}")

    # Create ideas.md
    ideas_path = ar_dir / "ideas.md"
    write_ideas_md(ideas_path)
    click.echo(f"Created {ideas_path}")

    # Create empty results.tsv and summary.md
    (ar_dir / "summary.md").write_text("# Experiment Summary\n\nNo experiments yet.\n")

    click.echo(f"\nReady! Open your AI tool and tell it:")
    click.echo(f'  "Read .automl-research/program.md and start experimenting"')


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

@main.command()
def preflight():
    """Validate changes before training."""
    config = load_config(find_config())
    errors: list[str] = []
    warnings: list[str] = []

    for fpath in config.modifiable.files:
        full = config.project_root / fpath
        if not full.exists():
            continue

        # Python syntax check
        if config.preflight.validate_syntax and fpath.endswith(".py"):
            try:
                ast.parse(full.read_text())
                click.echo(f"  ✓ {fpath} — syntax OK")
            except SyntaxError as e:
                errors.append(f"  ✗ {fpath} — syntax error: {e}")

        # YAML/JSON validation
        if config.preflight.validate_configs:
            if fpath.endswith((".yaml", ".yml")):
                try:
                    yaml.safe_load(full.read_text())
                    click.echo(f"  ✓ {fpath} — valid YAML")
                except yaml.YAMLError as e:
                    errors.append(f"  ✗ {fpath} — invalid YAML: {e}")
            elif fpath.endswith(".json"):
                try:
                    json.loads(full.read_text())
                    click.echo(f"  ✓ {fpath} — valid JSON")
                except json.JSONDecodeError as e:
                    errors.append(f"  ✗ {fpath} — invalid JSON: {e}")

    # Diff size check
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"], capture_output=True, text=True, cwd=config.project_root
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--stat"], capture_output=True, text=True, cwd=config.project_root
        )
        combined = result.stdout + staged.stdout
        # Count insertions + deletions
        import re
        ins = sum(int(x) for x in re.findall(r"(\d+) insertion", combined))
        dels = sum(int(x) for x in re.findall(r"(\d+) deletion", combined))
        total = ins + dels
        if total > config.preflight.max_diff_lines:
            warnings.append(f"  ⚠ {total} lines changed — consider smaller experiments (max: {config.preflight.max_diff_lines})")
        elif total > 0:
            click.echo(f"  ✓ Diff: {total} lines changed")
        else:
            click.echo(f"  ✓ No changes detected")
    except Exception:
        pass

    for w in warnings:
        click.echo(w)
    for e in errors:
        click.echo(e)

    if errors:
        click.echo("\nPre-flight FAILED. Fix errors before training.")
        sys.exit(1)
    else:
        click.echo("\nAll checks passed.")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@main.command()
@click.option("--full", is_flag=True, help="Use full training command instead of quick mode.")
@click.option("--description", "-d", default="", help="Short description for run directory naming.")
def train(full: bool, description: str):
    """Run training with timeout and log capture."""
    config = load_config(find_config())
    ar_dir = config.project_root / ".automl-research"
    state = _load_state(ar_dir / "state.json")
    run_id = state["total_experiments"] + 1

    # Determine command and budget
    if full and config.train.command_full:
        command = config.train.command_full
        budget = config.train.time_budget_full or config.train.time_budget_quick
        mode = "full"
    else:
        command = config.train.command_quick
        budget = config.train.time_budget_quick
        mode = "quick"

    # Create run directory
    desc_slug = description.replace(" ", "_")[:30] if description else "run"
    run_dir = ar_dir / "runs" / f"{run_id:03d}_{desc_slug}"
    log_path = run_dir / "run.log"

    cwd = str(config.project_root / config.train.cwd)

    click.echo(f"Running: {command}")
    click.echo(f"  Working dir: {config.train.cwd}")
    click.echo(f"  Time budget: {budget // 60}m ({mode} mode)")
    click.echo(f"  Log: {log_path.relative_to(config.project_root)}")

    result = execute_training(
        command=command,
        cwd=cwd,
        log_path=log_path,
        time_budget=budget,
        env_vars={
            "AUTOML_RUN_ID": str(run_id),
            "AUTOML_RUN_DIR": str(run_dir),
        },
    )

    if result.crashed:
        click.echo(f"  ✗ CRASHED (exit code {result.exit_code}, {result.wall_time:.0f}s)")
        if result.error_tail:
            click.echo(f"  Last output:\n{result.error_tail[-500:]}")
    else:
        click.echo(f"  ✓ Completed in {result.wall_time:.0f}s (exit code {result.exit_code})")


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------

@main.command()
@click.option("--description", "-d", required=True, help="Short description of what this experiment tried.")
@click.option("--crash", is_flag=True, help="Log as a crashed experiment.")
@click.option("--wall-time", type=float, default=0.0, help="Wall time in seconds (auto-detected if possible).")
@click.option("--session", default="", help="Session tag for WandB grouping.")
def decide_cmd(description: str, crash: bool, wall_time: float, session: str):
    """Make keep/discard decision for the latest experiment."""
    config = load_config(find_config())

    result = decide(
        config=config,
        description=description,
        is_crash=crash,
        wall_time=wall_time,
        session_tag=session,
    )

    # Display result
    if result.status == "keep":
        click.echo(f"\n  ✅ {result.message}")
    elif result.status == "discard":
        click.echo(f"\n  ❌ {result.message}")
    else:
        click.echo(f"\n  ⚠️  {result.message}")

    # Show metrics
    if result.metrics:
        click.echo("\n  Metrics:")
        for name, val in result.metrics.items():
            if val is not None:
                extra = ""
                if name == config.metrics.primary.name and result.best_value is not None:
                    extra = f"  (best: {result.best_value:.6f})"
                click.echo(f"    {name}: {val:.6f}{extra}")

    if result.lines_changed:
        click.echo(f"  Lines changed: {result.lines_changed}")

    click.echo(f"  Experiment #{result.experiment_id:03d}, commit {result.commit_hash}")
    if result.research_ideas_added:
        click.echo(f"  Research: {result.research_ideas_added} new ideas added to ideas.md")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@main.command()
def status():
    """Show current experiment status."""
    config = load_config(find_config())
    ar_dir = config.project_root / ".automl-research"
    state = _load_state(ar_dir / "state.json")

    total = state.get("total_experiments", 0)
    if total == 0:
        click.echo("No experiments yet. Run 'automl-research train' to start.")
        return

    # Count statuses from TSV
    tsv_path = ar_dir / "results.tsv"
    counts = {"keep": 0, "discard": 0, "crash": 0}
    if tsv_path.exists():
        with open(tsv_path) as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    s = parts[3].strip().lower()
                    counts[s] = counts.get(s, 0) + 1

    branch = state.get("current_branch", "?")
    best_val = state.get("best_primary_metric")
    best_id = state.get("best_experiment_id", 0)
    best_commit = state.get("best_commit", "?")
    last = state.get("last_experiment", {})

    click.echo(f"Branch: {branch}")
    click.echo(f"Experiments: {total} total ({counts['keep']} keep, {counts['discard']} discard, {counts['crash']} crash)")
    if best_val is not None:
        click.echo(f"Best: {config.metrics.primary.name} = {best_val:.6f} (commit {best_commit}, experiment #{best_id:03d})")
    if last.get("id"):
        click.echo(f"Last: #{last['id']:03d} {last.get('status', '?')} — {last.get('description', '')}")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

@main.command()
def summary():
    """Regenerate summary.md from results.tsv."""
    config = load_config(find_config())
    ar_dir = config.project_root / ".automl-research"
    tsv_path = ar_dir / "results.tsv"
    summary_path = ar_dir / "summary.md"

    from .decider import _generate_summary
    text = _generate_summary(tsv_path, config.metrics.primary.name, config.metrics.primary.direction)
    summary_path.write_text(text)
    click.echo(text)
    click.echo(f"\nWritten to {summary_path}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@main.command()
@click.option("--output", "-o", default=None, help="Output path for progress.png")
def analyze(output: str | None):
    """Generate progress chart from results.tsv."""
    config = load_config(find_config())
    ar_dir = config.project_root / ".automl-research"
    tsv_path = ar_dir / "results.tsv"

    if not tsv_path.exists():
        click.echo("No results.tsv found. Run experiments first.")
        return

    from .progress_chart import generate_progress_chart
    import matplotlib.pyplot as plt

    fig = generate_progress_chart(tsv_path, config.metrics.primary.name, config.metrics.primary.direction)
    out_path = Path(output) if output else ar_dir / "progress.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Saved progress chart to {out_path}")


# ---------------------------------------------------------------------------
# research
# ---------------------------------------------------------------------------

@main.command()
@click.option("--keywords", "-k", multiple=True, help="Search keywords (can be repeated).")
@click.option("--deep", is_flag=True, help="Context-aware: analyze what's working and search adaptively.")
@click.option("--max-papers", "-n", type=int, default=None, help="Max papers per source.")
@click.option("--source", "-s", multiple=True, help="Sources: semantic_scholar, arxiv, openreview.")
@click.option("--no-code", is_flag=True, help="Skip Papers With Code lookup (faster).")
@click.option("--clear-cache", is_flag=True, help="Clear cached results before searching.")
@click.option("--dry-run", is_flag=True, help="Preview ideas without updating ideas.md.")
def research(keywords, deep, max_papers, source, no_code, clear_cache, dry_run):
    """Search published papers for experiment ideas."""
    config = load_config(find_config())

    if clear_cache:
        cache_dir = config.project_root / ".automl-research" / "cache" / "research"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            click.echo("Cache cleared.")
        if not keywords and not deep:
            return

    from .researcher import run_research, generate_search_guidance

    click.echo("Searching for experiment ideas...")
    if deep:
        click.echo("  Mode: context-aware (analyzing experiment history)")
    elif keywords:
        click.echo(f"  Keywords: {', '.join(keywords)}")
    else:
        click.echo("  Mode: default search")

    ideas, papers = run_research(
        config=config,
        keywords=list(keywords) if keywords else None,
        deep=deep,
        sources=list(source) if source else None,
        max_papers=max_papers,
        include_code=not no_code,
        dry_run=dry_run,
    )

    click.echo(f"\n  Papers found: {len(papers)}")
    click.echo(f"  Ideas extracted: {len(ideas)}")

    if ideas:
        click.echo("\n  Ideas:")
        for i, idea in enumerate(ideas, 1):
            parts = [f"  {i}. {idea.description}"]
            if idea.paper_url:
                parts.append(f"     [{idea.paper_year}] {idea.paper_url}")
            if idea.code_url:
                parts.append(f"     Code: {idea.code_url}")
            parts.append(f"     ({idea.search_reason})")
            click.echo("\n".join(parts))

        if dry_run:
            click.echo("\n  (dry run — ideas.md not updated)")
        else:
            click.echo(f"\n  Updated .automl-research/ideas.md")
    else:
        click.echo("\n  No new ideas found from papers.")
        # Show web search guidance as fallback
        guidance = generate_search_guidance(config)
        click.echo("\n  Try these web searches manually:")
        for g in guidance:
            click.echo(f"    - {g}")


# Register decide command with proper name
main.add_command(decide_cmd, name="decide")


if __name__ == "__main__":
    main()
