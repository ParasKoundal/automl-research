"""Extract metrics from training output (logs, JSON files, CSV files)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional

from .config import MetricExtract, MetricsConfig


def extract_metric(extractor: MetricExtract, log_path: Path) -> Optional[float]:
    """Extract a single metric value using the configured method."""
    if extractor.method == "log_grep":
        return _grep_log(log_path, extractor.pattern)
    elif extractor.method == "json_file":
        return _read_json(extractor.file, extractor.key)
    elif extractor.method == "csv_file":
        return _read_csv(extractor.file, extractor.key)
    return None


def extract_all_metrics(
    metrics_config: MetricsConfig,
    log_path: Path,
) -> dict[str, Optional[float]]:
    """Extract primary + all secondary metrics from a run."""
    result: dict[str, Optional[float]] = {}

    # Primary
    result[metrics_config.primary.name] = extract_metric(
        metrics_config.primary.extract, log_path
    )

    # Secondary
    for sec in metrics_config.secondary:
        result[sec.name] = extract_metric(sec.extract, log_path)

    return result


def _grep_log(log_path: Path, pattern: Optional[str]) -> Optional[float]:
    """Extract a metric by regex matching on the log file.

    Returns the LAST match (final value after training completes).
    """
    if not pattern or not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return None
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (ValueError, TypeError):
        return None


def _read_json(file_path: Optional[str], key: Optional[str]) -> Optional[float]:
    """Extract a metric from a JSON file."""
    if not file_path or not key:
        return None
    p = Path(file_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        # Support nested keys with dot notation: "metrics.val_loss"
        for part in key.split("."):
            data = data[part]
        return float(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _read_csv(file_path: Optional[str], key: Optional[str]) -> Optional[float]:
    """Extract a metric from the last row of a CSV file."""
    if not file_path or not key:
        return None
    p = Path(file_path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            if last_row and key in last_row:
                return float(last_row[key])
    except (OSError, ValueError, KeyError):
        pass
    return None
