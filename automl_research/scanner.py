"""Auto-detect ML project structure during `automl-research init`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def detect_framework(project_root: Path) -> Optional[str]:
    """Detect ML framework from import statements."""
    counts: dict[str, int] = {"pytorch": 0, "tensorflow": 0, "keras": 0, "jax": 0, "sklearn": 0}
    for py in project_root.rglob("*.py"):
        try:
            text = py.read_text(errors="ignore")
        except OSError:
            continue
        if "import torch" in text or "from torch" in text:
            counts["pytorch"] += 1
        if "import tensorflow" in text or "from tensorflow" in text:
            counts["tensorflow"] += 1
        if "import keras" in text or "from keras" in text:
            counts["keras"] += 1
        if "import jax" in text or "from jax" in text:
            counts["jax"] += 1
        if "import sklearn" in text or "from sklearn" in text:
            counts["sklearn"] += 1
    best = max(counts, key=counts.get)  # type: ignore[arg-type]
    return best if counts[best] > 0 else None


def detect_training_scripts(project_root: Path) -> list[str]:
    """Find likely training scripts."""
    candidates: list[str] = []
    for py in sorted(project_root.rglob("*.py")):
        if py.name.startswith(".") or any(p.startswith(".") for p in py.parts):
            continue
        name_lower = py.name.lower()
        # Name-based heuristic
        if any(kw in name_lower for kw in ("train", "fit", "run_experiment")):
            candidates.append(str(py.relative_to(project_root)))
            continue
        # Content-based: has argparse + training-like code
        try:
            text = py.read_text(errors="ignore")[:5000]
        except OSError:
            continue
        if "argparse" in text and ("epoch" in text.lower() or "train" in text.lower()):
            candidates.append(str(py.relative_to(project_root)))
    return candidates


def detect_config_files(project_root: Path) -> list[str]:
    """Find YAML/JSON config files."""
    configs: list[str] = []
    for pattern in ("**/*.yaml", "**/*.yml", "**/*.json"):
        for f in sorted(project_root.glob(pattern)):
            if f.name.startswith(".") or any(p.startswith(".") for p in f.parts):
                continue
            if f.name in ("package.json", "package-lock.json", "tsconfig.json"):
                continue
            rel = str(f.relative_to(project_root))
            if "node_modules" in rel or "__pycache__" in rel:
                continue
            configs.append(rel)
    return configs


def detect_model_files(project_root: Path) -> list[str]:
    """Find files containing model class definitions."""
    models: list[str] = []
    model_re = re.compile(r"class\s+\w+.*\b(Module|Model|Layer|Network)\b")
    for py in sorted(project_root.rglob("*.py")):
        if py.name.startswith(".") or any(p.startswith(".") for p in py.parts):
            continue
        try:
            text = py.read_text(errors="ignore")[:10000]
        except OSError:
            continue
        if model_re.search(text):
            models.append(str(py.relative_to(project_root)))
    return models


# Common metric patterns found in ML training output
METRIC_PATTERNS: list[tuple[str, str, str]] = [
    # (name, regex_pattern, direction)
    ("val_loss", r"(?:val[_\s]?loss|validation[_\s]?loss)\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "minimize"),
    ("val_accuracy", r"(?:val[_\s]?acc|validation[_\s]?accuracy)\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "maximize"),
    ("test_loss", r"test[_\s]?loss\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "minimize"),
    ("val_bpb", r"val_bpb\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "minimize"),
    ("loss", r"\bloss\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "minimize"),
    ("accuracy", r"\baccuracy\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "maximize"),
    ("f1_score", r"f1[_\s]?score\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "maximize"),
    ("mse", r"\bmse\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "minimize"),
    ("mae", r"\bmae\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "minimize"),
    ("auc", r"\bauc\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "maximize"),
    ("total_loss", r"(?:Average\s+)?(?:Validation\s+)?total[_\s]?loss\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "minimize"),
    ("peak_vram_mb", r"peak_vram_mb\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", "minimize"),
]


def detect_metrics_from_log(log_text: str) -> list[dict]:
    """Detect metrics from training log output."""
    found: list[dict] = []
    seen_names: set[str] = set()
    for name, pattern, direction in METRIC_PATTERNS:
        matches = re.findall(pattern, log_text, re.IGNORECASE)
        if matches and name not in seen_names:
            seen_names.add(name)
            found.append({
                "name": name,
                "direction": direction,
                "pattern": pattern,
                "last_value": matches[-1],
            })
    return found


def scan_project(project_root: Path) -> dict:
    """Full project scan. Returns suggested values for project.yaml."""
    return {
        "framework": detect_framework(project_root),
        "training_scripts": detect_training_scripts(project_root),
        "config_files": detect_config_files(project_root),
        "model_files": detect_model_files(project_root),
    }
