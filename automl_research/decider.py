"""Unified keep/discard decision engine.

One command does: extract metrics → compare → git commit/revert → TSV → state → chart → WandB.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import ProjectConfig
from .metrics import extract_all_metrics
from .progress_chart import generate_progress_chart


@dataclass
class Decision:
    status: str  # "keep" | "discard" | "crash"
    primary_value: Optional[float]
    best_value: Optional[float]
    improvement: Optional[float]
    metrics: dict[str, Optional[float]]
    message: str
    experiment_id: int
    commit_hash: str
    lines_changed: int
    wall_time: float
    research_ideas_added: int = 0


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(*args: str, cwd: str | Path | None = None) -> str:
    r = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True,
        cwd=cwd,
    )
    return r.stdout.strip()


def _git_commit(message: str, cwd: Path) -> str:
    """Commit all changes EXCEPT .automl-research/ metadata."""
    _git("add", "-A", cwd=cwd)
    # Unstage .automl-research/ — metadata is managed separately from experiment code
    _git("reset", "HEAD", "--", ".automl-research", cwd=cwd)
    _git("commit", "-m", message, "--allow-empty", cwd=cwd)
    return _git("rev-parse", "--short", "HEAD", cwd=cwd)


def _git_revert_to(commit: str, cwd: Path) -> None:
    """Revert working tree to a commit, preserving .automl-research/ metadata."""
    import shutil
    import tempfile

    ar_dir = cwd / ".automl-research"
    backup_dir = None

    # Backup .automl-research/ before reset
    if ar_dir.exists():
        backup_dir = Path(tempfile.mkdtemp()) / ".automl-research"
        shutil.copytree(ar_dir, backup_dir)

    _git("reset", "--hard", commit, cwd=cwd)

    # Restore .automl-research/
    if backup_dir and backup_dir.exists():
        if ar_dir.exists():
            shutil.rmtree(ar_dir)
        shutil.copytree(backup_dir, ar_dir)
        shutil.rmtree(backup_dir.parent)


def _git_diff_stat(cwd: Path) -> int:
    """Count lines changed in working tree + staged."""
    output = _git("diff", "HEAD~1", "--stat", cwd=cwd)
    # Last line like: "3 files changed, 10 insertions(+), 2 deletions(-)"
    if not output:
        return 0
    import re
    m = re.search(r"(\d+) insertion", output)
    ins = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) deletion", output)
    dels = int(m.group(1)) if m else 0
    return ins + dels


def _git_diff_text(cwd: Path) -> str:
    return _git("diff", "HEAD~1", cwd=cwd) or _git("diff", "HEAD", cwd=cwd)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {
        "best_commit": None,
        "best_primary_metric": None,
        "best_experiment_id": 0,
        "total_experiments": 0,
        "current_branch": "",
        "last_experiment": {"id": 0, "status": "", "description": ""},
    }


def _save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2) + "\n")


# ---------------------------------------------------------------------------
# TSV logging
# ---------------------------------------------------------------------------

def _append_tsv(
    tsv_path: Path,
    experiment_id: int,
    commit: str,
    primary_metric: str,
    primary_value: Optional[float],
    status: str,
    wall_time: float,
    lines_changed: int,
    description: str,
    secondary_metrics: dict[str, Optional[float]],
    header_order: list[str],
) -> None:
    """Append a row to results.tsv."""
    # Create file with header if it doesn't exist
    if not tsv_path.exists():
        cols = ["id", "commit", primary_metric, "status", "wall_time", "lines_changed", "description"]
        cols += [n for n in header_order if n != primary_metric]
        tsv_path.write_text("\t".join(cols) + "\n")

    pval = f"{primary_value:.6f}" if primary_value is not None else "0.000000"
    parts = [
        f"{experiment_id:03d}",
        commit,
        pval,
        status,
        f"{wall_time:.0f}",
        str(lines_changed),
        description,
    ]
    for name in header_order:
        if name != primary_metric:
            val = secondary_metrics.get(name)
            parts.append(f"{val:.6f}" if val is not None else "0.000000")

    with open(tsv_path, "a") as f:
        f.write("\t".join(parts) + "\n")


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def _generate_summary(tsv_path: Path, primary_metric: str, direction: str) -> str:
    """Generate summary.md from results.tsv."""
    rows: list[dict] = []
    if tsv_path.exists():
        with open(tsv_path) as f:
            header = f.readline().strip().split("\t")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= len(header):
                    rows.append(dict(zip(header, parts)))

    if not rows:
        return "# Experiment Summary\n\nNo experiments yet.\n"

    is_min = direction == "minimize"

    # Find best
    kept = [r for r in rows if r.get("status") == "keep"]
    best_row = None
    if kept:
        best_row = min(kept, key=lambda r: float(r.get(primary_metric, "inf"))) if is_min \
            else max(kept, key=lambda r: float(r.get(primary_metric, "-inf")))

    lines = ["# Experiment Summary (auto-generated)\n"]

    if best_row:
        lines.append(f"## Best result")
        lines.append(f"{primary_metric} = {best_row[primary_metric]} at experiment #{best_row['id']} (commit {best_row['commit']})\n")

    # What worked
    lines.append("## What worked (kept)")
    prev_val = None
    for r in rows:
        if r.get("status") == "keep":
            val = float(r.get(primary_metric, 0))
            delta = ""
            if prev_val is not None:
                d = prev_val - val if is_min else val - prev_val
                delta = f" (Δ{d:+.6f})"
            lines.append(f"- #{r['id']}: {r.get('description', '')} → {primary_metric} {val:.6f}{delta}")
            prev_val = val
    lines.append("")

    # What didn't work
    discarded = [r for r in rows if r.get("status") == "discard"]
    if discarded:
        lines.append("## What didn't work (discarded)")
        for r in discarded:
            lines.append(f"- #{r['id']}: {r.get('description', '')} → {primary_metric} {r.get(primary_metric, '?')}")
        lines.append("")

    # Crashes
    crashes = [r for r in rows if r.get("status") == "crash"]
    if crashes:
        lines.append("## Crashes")
        for r in crashes:
            lines.append(f"- #{r['id']}: {r.get('description', '')}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# WandB logging
# ---------------------------------------------------------------------------

def _log_wandb(
    config: ProjectConfig,
    experiment_id: int,
    description: str,
    status: str,
    metrics: dict[str, Optional[float]],
    commit_hash: str,
    lines_changed: int,
    wall_time: float,
    diff_text: str,
    session_tag: str,
    is_best: bool,
    tsv_path: Path,
) -> None:
    """Log experiment to WandB (if enabled)."""
    if not config.tracking.wandb.enabled:
        return
    try:
        import wandb
    except ImportError:
        return

    run = wandb.init(
        project=config.tracking.wandb.project or f"automl-research-{config.name}",
        entity=config.tracking.wandb.entity,
        name=f"#{experiment_id:03d} {description}",
        group=session_tag,
        tags=[status],
        config={
            "description": description,
            "commit": commit_hash,
            "lines_changed": lines_changed,
            "wall_time_s": wall_time,
            "experiment_id": experiment_id,
        },
        notes=diff_text[:10000] if diff_text else "",
        reinit=True,
    )

    # Log metrics as summary
    summary = {k: v for k, v in metrics.items() if v is not None}
    summary["status"] = status
    summary["is_best"] = is_best
    summary["wall_time_s"] = wall_time
    summary["lines_changed"] = lines_changed
    run.summary.update(summary)

    # Generate and log progress chart
    try:
        fig = generate_progress_chart(
            tsv_path,
            config.metrics.primary.name,
            config.metrics.primary.direction,
        )
        run.log({"progress_chart": wandb.Image(fig)})
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass

    # Log experiment history as WandB Table
    try:
        primary_name = config.metrics.primary.name
        is_min = config.metrics.primary.direction == "minimize"
        columns = ["experiment_id", primary_name, "best_so_far", "status",
                    "lines_changed", "wall_time_s", "description"]
        table = wandb.Table(columns=columns)

        with open(tsv_path) as f:
            header = f.readline().strip().split("\t")
            running_best = float("inf") if is_min else float("-inf")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < len(header):
                    continue
                row = dict(zip(header, parts))
                try:
                    val = float(row.get(primary_name, 0))
                except ValueError:
                    continue
                if row.get("status") == "keep":
                    running_best = min(running_best, val) if is_min else max(running_best, val)
                table.add_data(
                    int(row.get("id", 0)),
                    val,
                    running_best,
                    row.get("status", ""),
                    int(row.get("lines_changed", 0)),
                    float(row.get("wall_time", 0)),
                    row.get("description", ""),
                )
        run.log({"experiment_history": table})
    except Exception:
        pass

    run.finish()


# ---------------------------------------------------------------------------
# Main decide function
# ---------------------------------------------------------------------------

def decide(
    config: ProjectConfig,
    description: str,
    is_crash: bool = False,
    wall_time: float = 0.0,
    session_tag: str = "",
    notes: str = "",
) -> Decision:
    """Make the keep/discard decision for the latest experiment.

    This is the unified command that does everything:
    1. Extract metrics from latest run log
    2. Compare to best (from state.json)
    3. Check constraints + min improvement
    4. Git commit (keep) or revert (discard)
    5. Append to results.tsv
    6. Update state.json + summary.md + progress.png
    7. Log to WandB
    """
    ar_dir = config.project_root / ".automl-research"
    state_path = ar_dir / "state.json"
    tsv_path = ar_dir / "results.tsv"
    summary_path = ar_dir / "summary.md"
    progress_path = ar_dir / "progress.png"

    state = _load_state(state_path)
    experiment_id = state["total_experiments"] + 1
    primary_name = config.metrics.primary.name
    direction = config.metrics.primary.direction
    is_min = direction == "minimize"

    # --- Handle crash ---
    if is_crash:
        # Revert to best commit
        best_commit = state.get("best_commit")
        if best_commit:
            _git_revert_to(best_commit, config.project_root)

        _append_tsv(
            tsv_path, experiment_id, "-------", primary_name, 0.0,
            "crash", wall_time, 0, description,
            {}, [s.name for s in config.metrics.secondary],
        )
        state["total_experiments"] = experiment_id
        state["last_experiment"] = {"id": experiment_id, "status": "crash", "description": description}
        _save_state(state_path, state)
        summary_path.write_text(_generate_summary(tsv_path, primary_name, direction))

        return Decision(
            status="crash", primary_value=None, best_value=state.get("best_primary_metric"),
            improvement=None, metrics={}, message=f"CRASH — reverted, logged #{experiment_id:03d}",
            experiment_id=experiment_id, commit_hash="-------", lines_changed=0, wall_time=wall_time,
        )

    # --- Find latest run log ---
    runs_dir = ar_dir / "runs"
    if runs_dir.exists():
        run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.name)
        latest_log = run_dirs[-1] / "run.log" if run_dirs else None
    else:
        latest_log = None

    if not latest_log or not latest_log.exists():
        return Decision(
            status="crash", primary_value=None, best_value=None,
            improvement=None, metrics={}, message="No run log found. Run 'automl-research train' first.",
            experiment_id=experiment_id, commit_hash="", lines_changed=0, wall_time=0,
        )

    # --- Extract metrics ---
    all_metrics = extract_all_metrics(config.metrics, latest_log)
    primary_value = all_metrics.get(primary_name)

    if primary_value is None:
        # Treat as crash — couldn't extract metric
        best_commit = state.get("best_commit")
        if best_commit:
            _git_revert_to(best_commit, config.project_root)
        msg = f"Could not extract {primary_name} from {latest_log}. Treated as crash."
        _append_tsv(
            tsv_path, experiment_id, "-------", primary_name, 0.0,
            "crash", wall_time, 0, description + " [metric not found]",
            {}, [s.name for s in config.metrics.secondary],
        )
        state["total_experiments"] = experiment_id
        state["last_experiment"] = {"id": experiment_id, "status": "crash", "description": description}
        _save_state(state_path, state)
        return Decision(
            status="crash", primary_value=None, best_value=state.get("best_primary_metric"),
            improvement=None, metrics=all_metrics, message=msg,
            experiment_id=experiment_id, commit_hash="-------", lines_changed=0, wall_time=wall_time,
        )

    best_value = state.get("best_primary_metric")

    # --- Check constraints ---
    for constraint in config.metrics.constraints:
        val = all_metrics.get(constraint.name)
        if val is not None:
            if constraint.max_value is not None and val > constraint.max_value:
                # Constraint violated — discard
                commit_hash = _git_commit(description, config.project_root)
                lines_changed = _git_diff_stat(config.project_root)
                best_commit = state.get("best_commit")
                if best_commit:
                    _git_revert_to(best_commit, config.project_root)
                msg = f"DISCARD — {constraint.name}={val:.1f} exceeds max {constraint.max_value}"
                _append_tsv(
                    tsv_path, experiment_id, commit_hash, primary_name, primary_value,
                    "discard", wall_time, lines_changed, description + f" [{constraint.name} violation]",
                    all_metrics, [s.name for s in config.metrics.secondary],
                )
                state["total_experiments"] = experiment_id
                state["last_experiment"] = {"id": experiment_id, "status": "discard", "description": description}
                _save_state(state_path, state)
                return Decision(
                    status="discard", primary_value=primary_value, best_value=best_value,
                    improvement=None, metrics=all_metrics, message=msg,
                    experiment_id=experiment_id, commit_hash=commit_hash,
                    lines_changed=lines_changed, wall_time=wall_time,
                )

    # --- Decide keep/discard ---
    is_first = best_value is None
    if is_first:
        improved = True
        improvement = 0.0
    else:
        if is_min:
            improvement = best_value - primary_value
        else:
            improvement = primary_value - best_value
        min_imp = config.metrics.primary.min_improvement
        improved = improvement > min_imp

    # Git: commit changes
    commit_hash = _git_commit(description, config.project_root)
    lines_changed = _git_diff_stat(config.project_root)
    diff_text = _git_diff_text(config.project_root)

    # Save diff.patch and notes to run dir (preserves code for ALL experiments, even discarded)
    try:
        runs_dir = ar_dir / "runs"
        if runs_dir.exists():
            run_dirs_all = sorted(runs_dir.iterdir(), key=lambda p: p.name)
            if run_dirs_all:
                run_dir = run_dirs_all[-1]
                if diff_text:
                    (run_dir / "diff.patch").write_text(diff_text)
                if notes:
                    (run_dir / "notes.md").write_text(notes + "\n")
    except Exception:
        pass

    if improved:
        status = "keep"
        msg_parts = []
        if is_first:
            msg_parts.append(f"KEEP — baseline {primary_name}={primary_value:.6f}")
        else:
            msg_parts.append(f"KEEP — {primary_name} improved {best_value:.6f} → {primary_value:.6f}")
        msg = " ".join(msg_parts)

        state["best_commit"] = commit_hash
        state["best_primary_metric"] = primary_value
        state["best_experiment_id"] = experiment_id
    else:
        status = "discard"
        msg = f"DISCARD — {primary_name} {best_value:.6f} → {primary_value:.6f} (no improvement)"
        # Revert
        best_commit = state.get("best_commit")
        if best_commit:
            _git_revert_to(best_commit, config.project_root)

    # --- Log to TSV ---
    _append_tsv(
        tsv_path, experiment_id, commit_hash, primary_name, primary_value,
        status, wall_time, lines_changed, description,
        all_metrics, [s.name for s in config.metrics.secondary],
    )

    # --- Update state ---
    state["total_experiments"] = experiment_id
    state["current_branch"] = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=config.project_root)
    state["last_experiment"] = {"id": experiment_id, "status": status, "description": description}
    _save_state(state_path, state)

    # --- Update summary ---
    summary_path.write_text(_generate_summary(tsv_path, primary_name, direction))

    # --- Generate progress chart ---
    try:
        fig = generate_progress_chart(tsv_path, primary_name, direction)
        fig.savefig(progress_path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass

    # --- WandB ---
    _log_wandb(
        config, experiment_id, description, status, all_metrics,
        commit_hash, lines_changed, wall_time, diff_text,
        session_tag or state.get("current_branch", ""),
        improved, tsv_path,
    )

    # --- Post-decide research hook ---
    research_ideas_added = 0
    if config.research.auto_after_decide and experiment_id % 3 == 0:
        # Run every 3rd experiment to avoid API spam
        try:
            from .researcher import run_research
            kw = [description]  # search based on what was just tried
            ideas, _ = run_research(
                config=config,
                keywords=kw if status == "keep" else None,
                deep=status == "discard",  # explore more when stuck
                max_papers=5,
            )
            research_ideas_added = len(ideas)
        except Exception:
            pass  # never break the loop

    # Track consecutive discards for explore/exploit balance
    if status == "discard" or status == "crash":
        state["_consecutive_discards"] = state.get("_consecutive_discards", 0) + 1
    else:
        state["_consecutive_discards"] = 0
    _save_state(state_path, state)

    return Decision(
        status=status,
        primary_value=primary_value,
        best_value=best_value if not is_first else primary_value,
        improvement=improvement,
        metrics=all_metrics,
        message=msg,
        experiment_id=experiment_id,
        commit_hash=commit_hash,
        lines_changed=lines_changed,
        wall_time=wall_time,
        research_ideas_added=research_ideas_added,
    )
