"""Parse and validate project.yaml — the project descriptor."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetricExtract:
    method: str  # log_grep | json_file | csv_file
    pattern: Optional[str] = None  # regex for log_grep
    file: Optional[str] = None  # path for json_file / csv_file
    key: Optional[str] = None  # json key or csv column


@dataclass
class PrimaryMetric:
    name: str
    direction: str  # minimize | maximize
    extract: MetricExtract
    min_improvement: float = 0.0


@dataclass
class SecondaryMetric:
    name: str
    extract: MetricExtract


@dataclass
class Constraint:
    name: str
    max_value: Optional[float] = None
    min_value: Optional[float] = None


@dataclass
class TrainConfig:
    command_quick: str
    cwd: str = "."
    time_budget_quick: int = 600  # seconds
    command_full: Optional[str] = None
    time_budget_full: Optional[int] = None


@dataclass
class MetricsConfig:
    primary: PrimaryMetric
    secondary: list[SecondaryMetric] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)


@dataclass
class ModifiableConfig:
    files: list[str] = field(default_factory=list)
    read_only: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)


@dataclass
class PreflightConfig:
    max_diff_lines: int = 50
    validate_syntax: bool = True
    validate_configs: bool = True


@dataclass
class WandbConfig:
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None


@dataclass
class TrackingConfig:
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class ResearchConfig:
    enabled: bool = False
    keywords: list[str] = field(default_factory=list)
    arxiv_categories: list[str] = field(default_factory=list)
    openreview_venues: list[str] = field(default_factory=lambda: ["ICLR.cc", "NeurIPS.cc", "ICML.cc"])
    max_papers: int = 20
    max_ideas: int = 10
    cache_ttl_hours: int = 24
    sources: list[str] = field(default_factory=lambda: ["semantic_scholar", "arxiv"])
    auto_after_decide: bool = False
    include_code: bool = True
    time_budget: int = 120  # max seconds for entire research run
    semantic_scholar_api_key: Optional[str] = None  # or env S2_API_KEY


@dataclass
class ProjectConfig:
    """Full parsed project.yaml."""

    name: str
    description: str
    framework: str
    train: TrainConfig
    metrics: MetricsConfig
    modifiable: ModifiableConfig
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    environment: dict[str, Any] = field(default_factory=dict)

    # Resolved at load time
    project_root: Path = field(default_factory=lambda: Path.cwd())


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(r"^(\d+)\s*(s|m|h)$", re.IGNORECASE)


def _parse_time(val: str | int) -> int:
    """Parse a time string like '10m' into seconds."""
    if isinstance(val, (int, float)):
        return int(val)
    m = _TIME_RE.match(str(val).strip())
    if not m:
        raise ValueError(f"Invalid time budget: {val!r}. Use e.g. '5m', '30m', '1h'.")
    num, unit = int(m.group(1)), m.group(2).lower()
    return num * {"s": 1, "m": 60, "h": 3600}[unit]


def _parse_extract(d: dict) -> MetricExtract:
    method = d["method"]
    if method not in ("log_grep", "json_file", "csv_file"):
        raise ValueError(f"Unknown extract method: {method}")
    return MetricExtract(
        method=method,
        pattern=d.get("pattern"),
        file=d.get("file"),
        key=d.get("key"),
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> ProjectConfig:
    """Load and validate a project.yaml file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    project_root = config_path.parent.parent  # .automl-research/project.yaml → project root

    # --- project ---
    proj = raw.get("project", {})
    name = proj.get("name", "Untitled")
    description = proj.get("description", "")
    framework = proj.get("framework", "other")

    # --- train ---
    tr = raw.get("train", {})
    command_quick = tr.get("command_quick") or tr.get("command", "")
    if not command_quick:
        raise ValueError("train.command_quick (or train.command) is required")
    train = TrainConfig(
        command_quick=command_quick.strip(),
        cwd=tr.get("cwd", "."),
        time_budget_quick=_parse_time(tr.get("time_budget_quick", tr.get("time_budget", "10m"))),
        command_full=(tr.get("command_full") or "").strip() or None,
        time_budget_full=_parse_time(tr["time_budget_full"]) if "time_budget_full" in tr else None,
    )

    # --- metrics ---
    met = raw.get("metrics", {})
    prim = met.get("primary", {})
    if not prim:
        raise ValueError("metrics.primary is required")
    primary = PrimaryMetric(
        name=prim["name"],
        direction=prim.get("direction", "minimize"),
        extract=_parse_extract(prim["extract"]),
        min_improvement=float(prim.get("min_improvement", 0.0)),
    )
    secondary = [
        SecondaryMetric(name=s["name"], extract=_parse_extract(s["extract"]))
        for s in met.get("secondary", [])
    ]
    constraints = [
        Constraint(
            name=c["name"],
            max_value=c.get("max_value"),
            min_value=c.get("min_value"),
        )
        for c in met.get("constraints", [])
    ]
    metrics = MetricsConfig(primary=primary, secondary=secondary, constraints=constraints)

    # --- modifiable ---
    mod = raw.get("modifiable", {})
    modifiable = ModifiableConfig(
        files=mod.get("files", []),
        read_only=mod.get("read_only", []),
        forbidden=mod.get("forbidden", []),
    )

    # --- preflight ---
    pf = raw.get("preflight", {})
    preflight = PreflightConfig(
        max_diff_lines=pf.get("max_diff_lines", 50),
        validate_syntax=pf.get("validate_syntax", True),
        validate_configs=pf.get("validate_configs", True),
    )

    # --- tracking ---
    tk = raw.get("tracking", {})
    wb = tk.get("wandb", {})
    tracking = TrackingConfig(
        wandb=WandbConfig(
            enabled=wb.get("enabled", False),
            project=wb.get("project"),
            entity=wb.get("entity"),
        )
    )

    # --- research ---
    res = raw.get("research", {})
    research = ResearchConfig(
        enabled=res.get("enabled", False),
        keywords=res.get("keywords", []),
        arxiv_categories=res.get("arxiv_categories", []),
        openreview_venues=res.get("openreview_venues", ["ICLR.cc", "NeurIPS.cc", "ICML.cc"]),
        max_papers=res.get("max_papers", 20),
        max_ideas=res.get("max_ideas", 10),
        cache_ttl_hours=res.get("cache_ttl_hours", 24),
        sources=res.get("sources", ["semantic_scholar", "arxiv"]),
        auto_after_decide=res.get("auto_after_decide", False),
        include_code=res.get("include_code", True),
        time_budget=res.get("time_budget", 120),
        semantic_scholar_api_key=res.get("semantic_scholar_api_key") or os.environ.get("S2_API_KEY"),
    )

    # --- environment ---
    environment = raw.get("environment", {})

    return ProjectConfig(
        name=name,
        description=description,
        framework=framework,
        train=train,
        metrics=metrics,
        modifiable=modifiable,
        preflight=preflight,
        tracking=tracking,
        research=research,
        environment=environment,
        project_root=project_root,
    )


def find_config() -> Path:
    """Find .automl-research/project.yaml walking up from cwd."""
    d = Path.cwd()
    while True:
        candidate = d / ".automl-research" / "project.yaml"
        if candidate.exists():
            return candidate
        parent = d.parent
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        "No .automl-research/project.yaml found. Run 'automl-research init' first."
    )
