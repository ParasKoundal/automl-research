"""Generate Karpathy-style progress plot from results.tsv.

Adapted from autoresearch-master/analysis.ipynb:
  - Gray dots for discarded experiments
  - Green dots with description labels for kept experiments
  - Green step-line for running best frontier
  - Crashes filtered out
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def _compute_focused_ylim(
    values: list[float],
    lower_pct: float = 2.0,
    upper_pct: float = 98.0,
    padding_frac: float = 0.15,
) -> tuple[float, float]:
    """Compute y-axis limits focused on the interesting percentile range."""
    arr = np.array(values)
    lo = float(np.percentile(arr, lower_pct))
    hi = float(np.percentile(arr, upper_pct))
    if hi == lo:
        margin = abs(lo) * 0.05 if lo != 0 else 0.1
        return lo - margin, hi + margin
    span = hi - lo
    return lo - span * padding_frac, hi + span * padding_frac


def generate_progress_chart(
    results_tsv: Path,
    primary_metric: str,
    direction: str = "minimize",
) -> plt.Figure:
    """Generate progress chart from results.tsv.

    Returns a matplotlib Figure (caller saves to file or logs to WandB).
    """
    # Parse TSV
    rows: list[dict] = []
    with open(results_tsv) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= len(header):
                rows.append(dict(zip(header, parts)))

    if not rows:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("No experiments yet")
        return fig

    # Extract data
    ids: list[int] = []
    metrics: list[float] = []
    statuses: list[str] = []
    descriptions: list[str] = []

    for row in rows:
        try:
            exp_id = int(row.get("id", 0))
            metric_val = float(row.get(primary_metric, 0))
            status = row.get("status", "").strip().lower()
            desc = row.get("description", "").strip()
        except (ValueError, TypeError):
            continue

        ids.append(exp_id)
        metrics.append(metric_val)
        statuses.append(status)
        descriptions.append(desc)

    # Filter out crashes for the main plot
    valid_ids = []
    valid_metrics = []
    valid_statuses = []
    valid_descs = []
    for i, s, m, d in zip(ids, statuses, metrics, descriptions):
        if s != "crash":
            valid_ids.append(i)
            valid_metrics.append(m)
            valid_statuses.append(s)
            valid_descs.append(d)

    if not valid_ids:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("All experiments crashed")
        return fig

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    baseline_val = valid_metrics[0]
    is_minimize = direction == "minimize"

    # Plot ALL discarded as gray dots (no filtering)
    disc_ids = [valid_ids[i] for i in range(len(valid_ids)) if valid_statuses[i] == "discard"]
    disc_vals = [valid_metrics[i] for i in range(len(valid_ids)) if valid_statuses[i] == "discard"]
    ax.scatter(disc_ids, disc_vals, c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded")

    # Plot ALL kept as prominent green dots
    kept_ids = [valid_ids[i] for i in range(len(valid_ids)) if valid_statuses[i] == "keep"]
    kept_vals = [valid_metrics[i] for i in range(len(valid_ids)) if valid_statuses[i] == "keep"]
    kept_descs = [valid_descs[i] for i in range(len(valid_ids)) if valid_statuses[i] == "keep"]
    ax.scatter(kept_ids, kept_vals, c="#2ecc71", s=50, zorder=4, label="Kept",
               edgecolors="black", linewidths=0.5)

    # Running best step line
    all_kept_ids = [valid_ids[i] for i in range(len(valid_ids)) if valid_statuses[i] == "keep"]
    all_kept_vals = [valid_metrics[i] for i in range(len(valid_ids)) if valid_statuses[i] == "keep"]
    if all_kept_ids:
        running_best = []
        best_so_far = all_kept_vals[0]
        for v in all_kept_vals:
            if is_minimize:
                best_so_far = min(best_so_far, v)
            else:
                best_so_far = max(best_so_far, v)
            running_best.append(best_so_far)
        ax.step(all_kept_ids, running_best, where="post", color="#27ae60",
                linewidth=2, alpha=0.7, zorder=3, label="Running best")

    # Annotate kept experiments with descriptions
    for kid, kval, kdesc in zip(kept_ids, kept_vals, kept_descs):
        if len(kdesc) > 45:
            kdesc = kdesc[:42] + "..."
        ax.annotate(
            kdesc, (kid, kval),
            textcoords="offset points", xytext=(6, 6),
            fontsize=8.0, color="#1a7a3a", alpha=0.9,
            rotation=30, ha="left", va="bottom",
        )

    # Labels and formatting
    n_total = len(ids)
    n_kept = sum(1 for s in statuses if s == "keep")
    n_crash = sum(1 for s in statuses if s == "crash")
    metric_label = f"{primary_metric} ({'lower' if is_minimize else 'higher'} is better)"

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    title = f"AutoML-Research Progress: {n_total} Experiments, {n_kept} Kept"
    if n_crash:
        title += f", {n_crash} Crashes"
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Y-axis: percentile-based zoom to interesting region
    all_plotted_vals = disc_vals + kept_vals
    if len(all_plotted_vals) >= 5:
        y_lo, y_hi = _compute_focused_ylim(all_plotted_vals)
        ax.set_ylim(y_lo, y_hi)
    elif all_plotted_vals:
        y_min = min(all_plotted_vals)
        y_max = max(all_plotted_vals)
        span = y_max - y_min
        margin = span * 0.1 if span > 0 else abs(y_min) * 0.01 or 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

    plt.tight_layout()
    return fig
