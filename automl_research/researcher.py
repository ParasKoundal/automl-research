"""Research module — discover experiment ideas from published papers.

Searches Semantic Scholar, ArXiv, OpenReview, and Papers With Code.
All API calls use stdlib (urllib) with try/except — research failures
never break the experiment loop.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError

from .config import ProjectConfig, ResearchConfig


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    tldr: str
    authors: list[str]
    year: int
    citation_count: int
    url: str
    source: str  # semantic_scholar | arxiv | openreview
    code_url: str = ""


@dataclass
class ResearchIdea:
    description: str
    paper_title: str
    paper_url: str
    code_url: str
    paper_year: int
    confidence: str  # low | medium | high
    search_reason: str  # why this was found
    citation_count: int = 0


# ---------------------------------------------------------------------------
# HTTP helper (stdlib only, zero deps)
# ---------------------------------------------------------------------------

_USER_AGENT = "automl-research/0.1 (https://github.com/automl-research)"


def _http_get(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> tuple[int, str]:
    """GET request with timeout. Returns (status_code, body)."""
    hdrs = {"User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = Request(url, headers=hdrs)
    try:
        resp = urlopen(req, timeout=timeout)
        return resp.status, resp.read().decode("utf-8", errors="replace")
    except URLError as e:
        if hasattr(e, "code"):
            return e.code, ""
        return 0, ""
    except Exception:
        return 0, ""


# ---------------------------------------------------------------------------
# Paper cache (JSON file, TTL-based)
# ---------------------------------------------------------------------------

class PaperCache:
    """Simple JSON-file cache for API responses."""

    def __init__(self, cache_dir: Path, ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = cache_dir / "cache.json"
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._cache_file.exists():
            try:
                self._data = json.loads(self._cache_file.read_text())
            except Exception:
                self._data = {}

    def _save(self) -> None:
        try:
            self._cache_file.write_text(json.dumps(self._data, indent=2, default=str))
        except Exception:
            pass

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        ts = datetime.fromisoformat(entry["timestamp"])
        if datetime.now() - ts > self.ttl:
            del self._data[key]
            return None
        return entry["value"]

    def set(self, key: str, value: Any) -> None:
        self._data[key] = {"value": value, "timestamp": datetime.now().isoformat()}
        self._save()

    def clear(self) -> None:
        self._data = {}
        self._save()


# ---------------------------------------------------------------------------
# Semantic Scholar API
# ---------------------------------------------------------------------------

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "paperId,title,abstract,tldr,authors,year,citationCount,externalIds,url"


def search_semantic_scholar(
    query: str,
    max_results: int = 20,
    api_key: str | None = None,
    cache: PaperCache | None = None,
    framework: str = "",
) -> list[Paper]:
    """Search Semantic Scholar for papers."""
    cache_key = f"s2:{query}:{max_results}"
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return [Paper(**p) for p in cached]

    # Clean query — strip years/framework filler so S2 focuses on the concept
    clean_q = _clean_query_for_search(query, framework)
    if not clean_q:
        clean_q = query

    params = urlencode({
        "query": clean_q,
        "limit": min(max_results, 100),
        "fields": _S2_FIELDS,
        "fieldsOfStudy": "Computer Science",
    })
    url = f"{_S2_BASE}/paper/search?{params}"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    # Rate limit: 1 req/sec without key, 10/sec with key
    if not api_key:
        time.sleep(1.1)

    status, body = _http_get(url, headers=headers)
    if status == 429:
        # Rate limited — wait and retry once
        time.sleep(5)
        status, body = _http_get(url, headers=headers)
    if status != 200:
        return []

    try:
        data = json.loads(body)
    except Exception:
        return []

    papers = []
    for item in data.get("data", []):
        if not item.get("title"):
            continue
        tldr_text = ""
        if item.get("tldr") and isinstance(item["tldr"], dict):
            tldr_text = item["tldr"].get("text", "")
        papers.append(Paper(
            paper_id=item.get("paperId", ""),
            title=item.get("title", ""),
            abstract=item.get("abstract", "") or "",
            tldr=tldr_text,
            authors=[a.get("name", "") for a in (item.get("authors") or [])[:5]],
            year=item.get("year") or 0,
            citation_count=item.get("citationCount") or 0,
            url=item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
            source="semantic_scholar",
        ))

    if cache and papers:
        cache.set(cache_key, [_paper_to_dict(p) for p in papers])

    return papers


def get_recommendations(
    paper_id: str,
    max_results: int = 10,
    api_key: str | None = None,
) -> list[Paper]:
    """Get papers similar to a given paper via S2 recommendations.

    Discovers the unknown — if a paper on 'cosine annealing' was useful,
    S2 might recommend papers on 'warm restarts', 'SGDR', or 'one-cycle policy'.
    """
    url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}?limit={max_results}&fields={_S2_FIELDS}"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    status, body = _http_get(url, headers=headers)
    if status != 200:
        return []

    try:
        data = json.loads(body)
    except Exception:
        return []

    papers = []
    for item in data.get("recommendedPapers", []):
        if not item.get("title"):
            continue
        tldr_text = ""
        if item.get("tldr") and isinstance(item["tldr"], dict):
            tldr_text = item["tldr"].get("text", "")
        papers.append(Paper(
            paper_id=item.get("paperId", ""),
            title=item.get("title", ""),
            abstract=item.get("abstract", "") or "",
            tldr=tldr_text,
            authors=[a.get("name", "") for a in (item.get("authors") or [])[:5]],
            year=item.get("year") or 0,
            citation_count=item.get("citationCount") or 0,
            url=item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
            source="semantic_scholar",
        ))

    return papers


# ---------------------------------------------------------------------------
# Query cleaning (shared by ArXiv + OpenReview)
# ---------------------------------------------------------------------------

_SEARCH_FILLER = re.compile(
    r"\b(?:state of the art|best practices)\b",
    re.IGNORECASE,
)

_DEFAULT_ML_CATS = ["cs.LG", "cs.AI", "cs.CV", "cs.CL"]


def _clean_query_for_search(query: str, framework: str = "") -> str:
    """Strip years, framework name, and filler words from a search query.

    ArXiv and OpenReview work better with focused technical terms.
    """
    q = re.sub(r"\b\d{4}\b", "", query)  # strip years
    if framework:
        q = re.sub(rf"\b{re.escape(framework)}\b", "", q, flags=re.IGNORECASE)
    cleaned = _SEARCH_FILLER.sub("", q)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # If cleaning left too few words, keep the version with only years/framework stripped
    if len(cleaned.split()) < 2:
        return re.sub(r"\s+", " ", q).strip()
    return cleaned


# ---------------------------------------------------------------------------
# ArXiv API
# ---------------------------------------------------------------------------

_ARXIV_BASE = "https://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(
    query: str,
    categories: list[str] | None = None,
    max_results: int = 20,
    cache: PaperCache | None = None,
    framework: str = "",
) -> list[Paper]:
    """Search ArXiv for papers via the Atom API."""
    cache_key = f"arxiv:{query}:{max_results}"
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return [Paper(**p) for p in cached]

    # Clean query for ArXiv — strip years, framework, filler words
    clean_q = _clean_query_for_search(query, framework)
    if not clean_q:
        clean_q = query  # fallback to original if cleaning removed everything

    # Use ti: (title) + abs: (abstract) for precision — not all: which matches anything
    encoded_q = quote_plus(clean_q)
    search_q = f"(ti:{encoded_q}+OR+abs:{encoded_q})"

    # Always scope to ML categories
    cats = categories or _DEFAULT_ML_CATS
    cat_filter = "+OR+".join(f"cat:{c}" for c in cats)
    search_q = f"{search_q}+AND+({cat_filter})"

    # Build URL manually to avoid urlencode double-encoding the search query
    other_params = urlencode({"start": 0, "max_results": max_results, "sortBy": "relevance"})
    url = f"{_ARXIV_BASE}?search_query={search_q}&{other_params}"

    status, body = _http_get(url, timeout=30)
    if status != 200:
        return []

    try:
        root = ET.fromstring(body)
    except Exception:
        return []

    papers = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title = (entry.findtext("atom:title", "", _ARXIV_NS) or "").strip().replace("\n", " ")
        if not title:
            continue
        abstract = (entry.findtext("atom:summary", "", _ARXIV_NS) or "").strip().replace("\n", " ")
        authors = [a.findtext("atom:name", "", _ARXIV_NS) for a in entry.findall("atom:author", _ARXIV_NS)]

        # Extract year from published date
        published = entry.findtext("atom:published", "", _ARXIV_NS)
        year = int(published[:4]) if published and len(published) >= 4 else 0

        # Extract arxiv ID and URL
        paper_id = entry.findtext("atom:id", "", _ARXIV_NS) or ""
        arxiv_id = paper_id.split("/abs/")[-1] if "/abs/" in paper_id else paper_id

        papers.append(Paper(
            paper_id=arxiv_id,
            title=title,
            abstract=abstract[:1000],
            tldr="",
            authors=authors[:5],
            year=year,
            citation_count=0,
            url=paper_id,
            source="arxiv",
        ))

    # Rate limit: 1 req / 3 sec
    time.sleep(3)

    if cache and papers:
        cache.set(cache_key, [_paper_to_dict(p) for p in papers])

    return papers


def search_arxiv_recent(category: str, max_results: int = 10) -> list[Paper]:
    """Get recent papers from an ArXiv category."""
    params = urlencode({
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{_ARXIV_BASE}?{params}"
    status, body = _http_get(url, timeout=30)
    if status != 200:
        return []

    try:
        root = ET.fromstring(body)
    except Exception:
        return []

    papers = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title = (entry.findtext("atom:title", "", _ARXIV_NS) or "").strip().replace("\n", " ")
        if not title:
            continue
        abstract = (entry.findtext("atom:summary", "", _ARXIV_NS) or "").strip().replace("\n", " ")
        authors = [a.findtext("atom:name", "", _ARXIV_NS) for a in entry.findall("atom:author", _ARXIV_NS)]
        published = entry.findtext("atom:published", "", _ARXIV_NS)
        year = int(published[:4]) if published and len(published) >= 4 else 0
        paper_id = entry.findtext("atom:id", "", _ARXIV_NS) or ""
        arxiv_id = paper_id.split("/abs/")[-1] if "/abs/" in paper_id else paper_id
        papers.append(Paper(
            paper_id=arxiv_id,
            title=title,
            abstract=abstract[:1000],
            tldr="",
            authors=authors[:5],
            year=year,
            citation_count=0,
            url=paper_id,
            source="arxiv",
        ))

    time.sleep(3)
    return papers


# ---------------------------------------------------------------------------
# OpenReview API
# ---------------------------------------------------------------------------

_OPENREVIEW_BASE = "https://api2.openreview.net"


def _get_or_value(content: dict, key: str, default: Any = "") -> Any:
    """OpenReview wraps values in {"value": ...} dicts."""
    v = content.get(key, {})
    if isinstance(v, dict):
        return v.get("value", default)
    return v if v else default


def search_openreview(
    query: str,
    venues: list[str] | None = None,
    max_results: int = 20,
    cache: PaperCache | None = None,
) -> list[Paper]:
    """Search OpenReview for peer-reviewed papers (ICLR/NeurIPS/ICML)."""
    cache_key = f"openreview:{query}:{max_results}"
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return [Paper(**p) for p in cached]

    papers = []
    seen_ids: set[str] = set()  # deduplicate across venue×year iterations

    # Use /notes/search endpoint (semantic search, much better than invitation filter)
    venue_names = {
        "ICLR.cc": ["ICLR 2025", "ICLR 2024", "ICLR 2023"],
        "NeurIPS.cc": ["NeurIPS 2024", "NeurIPS 2023"],
        "ICML.cc": ["ICML 2024", "ICML 2023"],
    }
    target_venues = venues or ["ICLR.cc", "NeurIPS.cc", "ICML.cc"]

    for venue_key in target_venues:
        for venue_name in venue_names.get(venue_key, []):
            params = urlencode({
                "query": query,
                "limit": min(max_results, 25),
                "content.venue": venue_name,
            })
            url = f"{_OPENREVIEW_BASE}/notes/search?{params}"
            status, body = _http_get(url, timeout=30)
            if status != 200:
                continue

            try:
                data = json.loads(body)
            except Exception:
                continue

            for note in data.get("notes", []):
                note_id = note.get("id", "")
                if note_id in seen_ids:
                    continue  # skip duplicate
                seen_ids.add(note_id)

                content = note.get("content", {})
                title = _get_or_value(content, "title")
                if not title:
                    continue
                # Extract year from venue name
                year_match = re.search(r"(\d{4})", venue_name)
                year = int(year_match.group(1)) if year_match else 0
                papers.append(Paper(
                    paper_id=note_id,
                    title=title,
                    abstract=_get_or_value(content, "abstract")[:1000],
                    tldr=_get_or_value(content, "TLDR") or _get_or_value(content, "tldr"),
                    authors=[a for a in _get_or_value(content, "authors", []) if isinstance(a, str)][:5],
                    year=year,
                    citation_count=0,
                    url=f"https://openreview.net/forum?id={note.get('id', '')}",
                    source="openreview",
                ))

            if len(papers) >= max_results:
                break
            time.sleep(1)  # be polite

        if len(papers) >= max_results:
            break

    if cache and papers:
        cache.set(cache_key, [_paper_to_dict(p) for p in papers])

    return papers[:max_results]


# ---------------------------------------------------------------------------
# Papers With Code — code lookup
# ---------------------------------------------------------------------------

_PWC_BASE = "https://paperswithcode.com/api/v1"


_PWC_HEADERS = {"Accept": "application/json"}


def lookup_code(paper_title: str, arxiv_id: str = "") -> str:
    """Find GitHub repo URL for a paper via Papers With Code.

    Returns the most-starred repo URL, or empty string if none found.
    """
    # Try arxiv ID first (more precise)
    if arxiv_id:
        url = f"{_PWC_BASE}/papers/?arxiv_id={quote_plus(arxiv_id)}"
        status, body = _http_get(url, headers=_PWC_HEADERS, timeout=15)
        if status == 200:
            try:
                data = json.loads(body)
                results = data.get("results", [])
                if results:
                    repo = _get_best_repo(results[0].get("id", ""))
                    if repo:
                        return repo
            except Exception:
                pass

    # Fall back to title search
    url = f"{_PWC_BASE}/papers/?q={quote_plus(paper_title[:100])}"
    status, body = _http_get(url, headers=_PWC_HEADERS, timeout=15)
    if status == 200:
        try:
            data = json.loads(body)
            results = data.get("results", [])
            if results:
                return _get_best_repo(results[0].get("id", ""))
        except Exception:
            pass
    return ""


def _get_best_repo(paper_id: str) -> str:
    """Get the most-starred GitHub repo for a paper."""
    if not paper_id:
        return ""
    url = f"{_PWC_BASE}/papers/{paper_id}/repositories/"
    status, body = _http_get(url, headers=_PWC_HEADERS, timeout=15)
    if status == 200:
        try:
            data = json.loads(body)
            repos = data.get("results", [])
            if repos:
                repos.sort(key=lambda r: r.get("stars", 0), reverse=True)
                return repos[0].get("url", "")
        except Exception:
            pass
    return ""


# ---------------------------------------------------------------------------
# Theme extraction (dynamic, no hardcoded categories)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the to is was were be been being have has had do does did will would "
    "shall should can could may might must and but or nor for yet so at by from "
    "in of on with about into through during before after above below between "
    "out off over under up down it its this that these those i me my we our "
    "he she they you your their than then also just more less very too try "
    "tried use used using set change changed get got run ran add added make "
    "made test tested experiment".split()
)


def _extract_themes(descriptions: list[str], top_n: int = 5) -> list[str]:
    """Extract meaningful themes from experiment descriptions using n-grams.

    No predefined category list. Discovers themes from the actual text.
    """
    # Tokenize all descriptions
    words_list = []
    for desc in descriptions:
        tokens = re.findall(r"[a-zA-Z_]+", desc.lower())
        words_list.append([t for t in tokens if t not in _STOP_WORDS and len(t) > 2])

    # Extract bigrams and trigrams
    ngram_counts: dict[str, int] = {}
    for words in words_list:
        for i in range(len(words)):
            # Unigrams (meaningful technical terms)
            if len(words[i]) > 3:
                ngram_counts[words[i]] = ngram_counts.get(words[i], 0) + 1
            # Bigrams
            if i + 1 < len(words):
                bg = f"{words[i]} {words[i+1]}"
                ngram_counts[bg] = ngram_counts.get(bg, 0) + 1
            # Trigrams
            if i + 2 < len(words):
                tg = f"{words[i]} {words[i+1]} {words[i+2]}"
                ngram_counts[tg] = ngram_counts.get(tg, 0) + 1

    # Sort by frequency, prefer multi-word themes
    scored = []
    for phrase, count in ngram_counts.items():
        word_count = len(phrase.split())
        score = count * (1.0 + 0.5 * word_count)  # bonus for multi-word
        scored.append((phrase, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate (remove substrings of longer phrases)
    themes = []
    for phrase, _ in scored:
        if len(themes) >= top_n:
            break
        if not any(phrase in t for t in themes):
            themes.append(phrase)

    return themes


# ---------------------------------------------------------------------------
# Explore/Exploit balance
# ---------------------------------------------------------------------------

def _compute_explore_ratio(summary_text: str, state: dict) -> float:
    """Dynamically balance exploration vs exploitation.

    Returns a float 0.0-1.0 where 1.0 = full exploration.
    """
    total = state.get("total_experiments", 0)

    # Parse recent statuses from summary
    kept_lines = re.findall(r"^- #\d+:.*$", summary_text, re.MULTILINE)
    status_sequence = []
    for line in kept_lines:
        if "What worked" in summary_text.split(line)[0].split("\n")[-5:] if line in summary_text else []:
            status_sequence.append("keep")
        else:
            status_sequence.append("discard")

    # Simpler: count keeps and discards
    n_keep = summary_text.count("What worked")
    kept_items = len(re.findall(r"^- #\d+:.*→", summary_text, re.MULTILINE))
    discarded_section = summary_text.split("What didn't work")[-1] if "What didn't work" in summary_text else ""
    discarded_items = len(re.findall(r"^- #\d+:", discarded_section, re.MULTILINE))

    # Count consecutive discards from state
    last = state.get("last_experiment", {})
    consecutive_discards = state.get("_consecutive_discards", 0)

    ratio = 0.4  # default: 40% explore, 60% exploit

    # EARLY PHASE (< 5 experiments): heavy exploration
    if total < 5:
        ratio = 0.7

    # STUCK (3+ consecutive failures): shift to exploration
    elif consecutive_discards >= 3:
        ratio = 0.8

    # ON A ROLL (recent keeps): heavy exploitation
    elif consecutive_discards == 0 and kept_items >= 2 and last.get("status") == "keep":
        ratio = 0.2

    # DIMINISHING RETURNS (many experiments, few themes)
    if total > 10 and kept_items > 0:
        themes_from_kept = len(set(re.findall(r"#\d+: (.+?)→", summary_text)))
        if themes_from_kept < 3:
            ratio = min(ratio + 0.2, 0.8)

    # LATE PHASE (20+ experiments): more exploitation
    if total > 20 and consecutive_discards < 2:
        ratio = max(ratio - 0.1, 0.2)

    return ratio


# ---------------------------------------------------------------------------
# Context-aware query builder
# ---------------------------------------------------------------------------

def _parse_experiments_from_summary(summary_text: str, section: str) -> list[str]:
    """Extract experiment descriptions from a section of summary.md."""
    descriptions = []
    in_section = False
    for line in summary_text.split("\n"):
        if section.lower() in line.lower():
            in_section = True
            continue
        if in_section and line.startswith("##"):
            break
        if in_section and line.startswith("- #"):
            # Extract description: "- #001: description → metric value"
            match = re.match(r"- #\d+:\s*(.+?)(?:\s*→.*)?$", line)
            if match:
                descriptions.append(match.group(1).strip())
    return descriptions


def _get_paper_ids_from_ideas(ideas_path: Path) -> list[str]:
    """Extract Semantic Scholar paper IDs from ideas.md links."""
    paper_ids = []
    if not ideas_path.exists():
        return paper_ids
    text = ideas_path.read_text()
    # Match semanticscholar.org URLs
    for match in re.finditer(r"semanticscholar\.org/paper/([a-f0-9]+)", text):
        paper_ids.append(match.group(1))
    return paper_ids


def build_context_queries(config: ProjectConfig) -> list[tuple[str, str]]:
    """Analyze experiment history and generate targeted search queries.

    NO HARDCODED CATEGORIES. Extracts themes from actual experiment descriptions.
    Returns list of (query_string, reason) tuples.
    """
    ar_dir = config.project_root / ".automl-research"
    summary_path = ar_dir / "summary.md"
    state_path = ar_dir / "state.json"
    ideas_path = ar_dir / "ideas.md"

    summary = summary_path.read_text() if summary_path.exists() else ""
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    primary_name = config.metrics.primary.name
    framework = config.framework
    queries: list[tuple[str, str]] = []

    ratio = _compute_explore_ratio(summary, state)
    n_total = 8
    n_explore = round(n_total * ratio)
    n_exploit = n_total - n_explore

    # ── EXPLOIT: Deepen what's working ─────────────────────
    kept_descriptions = _parse_experiments_from_summary(summary, "What worked")
    if kept_descriptions:
        themes = _extract_themes(kept_descriptions)
        for theme in themes[:n_exploit]:
            queries.append((
                f"{theme} techniques {framework} improve {primary_name} 2024 2025",
                f"Exploit: '{theme}' is working — finding more research",
            ))

    # S2 recommendations for papers that led to successful experiments
    paper_ids = _get_paper_ids_from_ideas(ideas_path)
    for pid in paper_ids[:2]:
        queries.append((
            f"RECOMMEND:{pid}",
            "Finding papers similar to one that led to a successful experiment",
        ))

    # ── EXPLORE: Discover the completely unknown ───────────
    explore_queries = [
        (f"state of the art {primary_name} {framework} training techniques 2024 2025",
         "Explore: what's new in the field?"),
        (f"novel optimizer {framework} 2024 2025",
         "Explore: discovering unknown optimizers (Shampoo, Lion, Sophia, etc.)"),
        (f"training tricks {framework} {primary_name} improvement",
         "Explore: general training improvements"),
    ]

    # Search for what FAILED — find alternatives
    discarded = _parse_experiments_from_summary(summary, "What didn't work")
    for desc in discarded[:2]:
        explore_queries.append((
            f"alternative to {desc} {framework}",
            f"Finding alternatives to failed: {desc}",
        ))

    # Browse recent top papers in configured ArXiv categories
    for cat in (config.research.arxiv_categories or [])[:2]:
        explore_queries.append((
            f"RECENT:{cat}",
            f"Explore: latest papers in {cat}",
        ))

    # Add configured keywords
    for kw in config.research.keywords:
        explore_queries.append((
            f"{kw} {framework} 2024 2025",
            f"Configured keyword: {kw}",
        ))

    queries.extend(explore_queries[:n_explore])

    return queries


# ---------------------------------------------------------------------------
# Idea extraction from papers
# ---------------------------------------------------------------------------

def extract_ideas(papers: list[Paper], config: ProjectConfig, search_reason: str) -> list[ResearchIdea]:
    """Convert papers into research ideas for the agent to evaluate.

    No regex extraction — present the paper's own TLDR/summary directly.
    The AI agent reading ideas.md is the intelligence layer.
    """
    ideas = []

    for paper in papers:
        desc = _build_idea_description(paper)
        if not desc:
            continue

        # Confidence signal based on citations and recency
        if paper.citation_count > 100 and paper.year >= 2023:
            conf = "high"
        elif paper.citation_count > 20 or paper.year >= 2024:
            conf = "medium"
        else:
            conf = "low"

        ideas.append(ResearchIdea(
            description=desc,
            paper_title=paper.title,
            paper_url=paper.url,
            code_url=paper.code_url,
            paper_year=paper.year,
            confidence=conf,
            search_reason=search_reason,
            citation_count=paper.citation_count,
        ))

    return ideas


def _build_idea_description(paper: Paper) -> str:
    """Present the paper's own summary for the agent to evaluate.

    No regex extraction — the AI agent is the intelligence layer.
    TLDR is preferred (already concise), otherwise first non-boilerplate sentence.
    """
    if paper.tldr:
        return paper.tldr.strip()[:300]

    # First non-boilerplate sentence from abstract
    if paper.abstract:
        for sent in _split_sentences(paper.abstract):
            low = sent.lower().lstrip()
            if not low.startswith(("in this paper", "this paper", "we present a",
                                   "we propose a", "in this work", "this work")):
                return sent.strip()[:300]
        # All sentences were boilerplate — use the second sentence if available
        sents = _split_sentences(paper.abstract)
        if len(sents) > 1:
            return sents[1].strip()[:300]

    return ""


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sents if len(s.strip()) > 20]


# ---------------------------------------------------------------------------
# Local fallback ideas (no API needed)
# ---------------------------------------------------------------------------

def _generate_local_ideas(config: ProjectConfig) -> list[ResearchIdea]:
    """Generate ideas purely from experiment history. No network needed."""
    ar_dir = config.project_root / ".automl-research"
    summary_path = ar_dir / "summary.md"
    summary = summary_path.read_text() if summary_path.exists() else ""

    ideas = []
    kept = _parse_experiments_from_summary(summary, "What worked")
    discarded = _parse_experiments_from_summary(summary, "What didn't work")

    # Pattern: "X worked, try variations"
    for desc in kept[:3]:
        ideas.append(ResearchIdea(
            description=f"Variation: try a more aggressive version of '{desc}'",
            paper_title="", paper_url="", code_url="", paper_year=0,
            confidence="medium", search_reason="Local: variation of what worked",
        ))
        ideas.append(ResearchIdea(
            description=f"Variation: try a more conservative version of '{desc}'",
            paper_title="", paper_url="", code_url="", paper_year=0,
            confidence="low", search_reason="Local: variation of what worked",
        ))

    # Pattern: "X and Y both worked, try combining"
    if len(kept) >= 2:
        for i, a in enumerate(kept):
            for b in kept[i + 1:]:
                ideas.append(ResearchIdea(
                    description=f"Combine: '{a}' + '{b}' in the same experiment",
                    paper_title="", paper_url="", code_url="", paper_year=0,
                    confidence="medium", search_reason="Local: combine what worked",
                ))

    # General suggestions
    general = [
        "Try a completely different optimizer than current",
        "Try a different normalization approach",
        "Try adjusting the loss function",
        "Try data augmentation if applicable",
        "Try different weight initialization",
        "Try gradient clipping or gradient accumulation",
    ]
    for g in general:
        ideas.append(ResearchIdea(
            description=g, paper_title="", paper_url="", code_url="", paper_year=0,
            confidence="low", search_reason="Local: general ML knowledge",
        ))

    return ideas


# ---------------------------------------------------------------------------
# Web search guidance (for the AI agent to execute)
# ---------------------------------------------------------------------------

def generate_search_guidance(config: ProjectConfig) -> list[str]:
    """Generate web search prompts for the AI agent when APIs are unavailable."""
    primary = config.metrics.primary.name
    framework = config.framework
    return [
        f"improve {primary} {framework} 2024 2025",
        f"state of the art training tricks deep learning {framework}",
        f"novel optimizer {framework} 2024",
        f"best practices {primary} optimization",
        f"{framework} training hyperparameter tuning guide",
    ]


# ---------------------------------------------------------------------------
# ideas.md updater
# ---------------------------------------------------------------------------

def _load_existing_ideas(ideas_path: Path) -> set[str]:
    """Load existing paper titles for deduplication."""
    if not ideas_path.exists():
        return set()
    text = ideas_path.read_text()
    existing = set()
    for line in text.split("\n"):
        # Match paper title entries: "- [ ] **Paper Title**" or numbered "1. [ ] **Paper Title**"
        m = re.match(r"(?:\d+\.\s+|- )\[[ x]\]\s+\*\*(.+?)\*\*", line)
        if m:
            existing.add(m.group(1).lower().strip())
    return existing


def update_ideas_md(ideas_path: Path, new_ideas: list[ResearchIdea]) -> int:
    """Append research ideas to ideas.md with rich format for agent consumption."""
    if not new_ideas:
        return 0

    existing = _load_existing_ideas(ideas_path)
    to_add: list[ResearchIdea] = []
    for idea in new_ideas:
        # Deduplicate by paper title (not description, since descriptions are now TLDRs)
        key = idea.paper_title.lower().strip()
        if key and key not in existing:
            to_add.append(idea)
            existing.add(key)
        elif not key:
            # Local ideas have no paper title — dedup by description
            desc_key = idea.description.lower().strip()
            if desc_key not in existing:
                to_add.append(idea)
                existing.add(desc_key)

    if not to_add:
        return 0

    # Group by search_reason
    groups: dict[str, list[ResearchIdea]] = {}
    for idea in to_add:
        groups.setdefault(idea.search_reason, []).append(idea)

    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [f"\n### Research — {date_str}\n"]

    for reason, ideas_in_group in groups.items():
        lines.append(f"**{reason}**\n")
        for i, idea in enumerate(ideas_in_group, 1):
            # Title line with metadata
            meta_parts = []
            if idea.paper_url:
                meta_parts.append(f"[{idea.paper_year}]({idea.paper_url})")
            if idea.citation_count:
                meta_parts.append(f"{idea.citation_count} citations")
            if idea.code_url:
                meta_parts.append(f"[Code]({idea.code_url})")
            meta = " | ".join(meta_parts)

            if idea.paper_title:
                lines.append(f"- [ ] **{idea.paper_title}** ({meta})")
            else:
                lines.append(f"- [ ] **{idea.description}**")

            # TLDR/description as blockquote — the agent reads this to judge relevance
            if idea.description and idea.paper_title:
                lines.append(f"  > {idea.description}")
            lines.append("")

    # Append to ideas.md (preserve existing content)
    current = ideas_path.read_text() if ideas_path.exists() else "# Experiment Ideas\n"

    # Insert before "## Tried Ideas" section if it exists
    tried_marker = "## Tried Ideas"
    if tried_marker in current:
        idx = current.index(tried_marker)
        new_content = current[:idx] + "\n".join(lines) + "\n" + current[idx:]
    else:
        new_content = current.rstrip() + "\n" + "\n".join(lines)

    ideas_path.write_text(new_content)
    return len(to_add)


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def _paper_to_dict(p: Paper) -> dict:
    return {
        "paper_id": p.paper_id, "title": p.title, "abstract": p.abstract,
        "tldr": p.tldr, "authors": p.authors, "year": p.year,
        "citation_count": p.citation_count, "url": p.url, "source": p.source,
        "code_url": p.code_url,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_research(
    config: ProjectConfig,
    keywords: list[str] | None = None,
    deep: bool = False,
    sources: list[str] | None = None,
    max_papers: int | None = None,
    include_code: bool = True,
    dry_run: bool = False,
) -> tuple[list[ResearchIdea], list[Paper]]:
    """Run research: search papers, extract ideas, update ideas.md.

    Every external call is wrapped in try/except — failures never break the loop.

    Returns (ideas, papers).
    """
    rc = config.research
    ar_dir = config.project_root / ".automl-research"
    cache_dir = ar_dir / "cache" / "research"
    cache = PaperCache(cache_dir, ttl_hours=rc.cache_ttl_hours)

    active_sources = sources or rc.sources
    max_per_source = max_papers or rc.max_papers
    api_key = rc.semantic_scholar_api_key
    all_papers: list[Paper] = []
    all_ideas: list[ResearchIdea] = []
    _research_start = time.monotonic()
    _time_budget = rc.time_budget  # seconds (default 120)

    def _over_budget() -> bool:
        return (time.monotonic() - _research_start) >= _time_budget

    # Build queries
    if deep:
        queries = build_context_queries(config)
    elif keywords:
        queries = [(f"{kw} {config.framework} 2024 2025", f"Keyword: {kw}") for kw in keywords]
    elif rc.keywords:
        queries = [(f"{kw} {config.framework} 2024 2025", f"Configured: {kw}") for kw in rc.keywords]
    else:
        # Default broad search
        primary = config.metrics.primary.name
        queries = [
            (f"improve {primary} {config.framework} 2024 2025", "Default search"),
            (f"training techniques {config.framework} state of the art", "Default: SOTA techniques"),
        ]

    # Track which reason found each paper (by normalized title)
    paper_reasons: dict[str, str] = {}

    _query_count = 0
    for query, reason in queries:
        if _over_budget():
            break
        # Rate-limit between queries (ArXiv recommends 3s between requests)
        if _query_count > 0:
            time.sleep(3)
        _query_count += 1

        # Handle special prefixes
        if query.startswith("RECOMMEND:"):
            paper_id = query[len("RECOMMEND:"):]
            if "semantic_scholar" in active_sources:
                try:
                    papers = get_recommendations(paper_id, max_results=5, api_key=api_key)
                    for p in papers:
                        paper_reasons.setdefault(p.title.lower().strip(), reason)
                    all_papers.extend(papers)
                except Exception:
                    pass
            continue

        if query.startswith("RECENT:"):
            category = query[len("RECENT:"):]
            if "arxiv" in active_sources:
                try:
                    papers = search_arxiv_recent(category, max_results=5)
                    for p in papers:
                        paper_reasons.setdefault(p.title.lower().strip(), reason)
                    all_papers.extend(papers)
                except Exception:
                    pass
            continue

        # Regular search across configured sources
        if "semantic_scholar" in active_sources and not _over_budget():
            try:
                papers = search_semantic_scholar(query, max_results=max_per_source, api_key=api_key, cache=cache, framework=config.framework)
                for p in papers:
                    paper_reasons.setdefault(p.title.lower().strip(), reason)
                all_papers.extend(papers)
            except Exception:
                pass

        if "arxiv" in active_sources and not _over_budget():
            try:
                papers = search_arxiv(
                    query, categories=rc.arxiv_categories or None,
                    max_results=max_per_source, cache=cache,
                    framework=config.framework,
                )
                for p in papers:
                    paper_reasons.setdefault(p.title.lower().strip(), reason)
                all_papers.extend(papers)
            except Exception:
                pass

        if "openreview" in active_sources and not _over_budget():
            try:
                or_query = _clean_query_for_search(query, config.framework)
                papers = search_openreview(or_query or query, venues=rc.openreview_venues, max_results=max_per_source, cache=cache)
                for p in papers:
                    paper_reasons.setdefault(p.title.lower().strip(), reason)
                all_papers.extend(papers)
            except Exception:
                pass

    # ── Cross-source dedup ────────────────────────────────
    seen_titles: set[str] = set()
    unique_papers: list[Paper] = []
    for p in all_papers:
        norm_title = p.title.lower().strip()
        if norm_title in seen_titles:
            # Merge info from duplicate into existing
            for existing in unique_papers:
                if existing.title.lower().strip() == norm_title:
                    if p.code_url and not existing.code_url:
                        existing.code_url = p.code_url
                    if p.tldr and not existing.tldr:
                        existing.tldr = p.tldr
                    if p.citation_count > existing.citation_count:
                        existing.citation_count = p.citation_count
                    break
            continue
        seen_titles.add(norm_title)
        unique_papers.append(p)
    all_papers = unique_papers

    # ── Sort: interleave established + frontier papers ────
    # Two pools ensure the agent sees BOTH highly-cited foundational
    # papers AND cutting-edge recent work (not just one or the other).
    by_citations = sorted(all_papers, key=lambda p: p.citation_count, reverse=True)
    by_recency = sorted(all_papers, key=lambda p: (p.year, p.citation_count), reverse=True)
    seen_interleave: set[str] = set()
    interleaved: list[Paper] = []
    i = j = 0
    while i < len(by_citations) or j < len(by_recency):
        # Take from high-citation pool
        while i < len(by_citations):
            t = by_citations[i].title.lower().strip()
            i += 1
            if t not in seen_interleave:
                seen_interleave.add(t)
                interleaved.append(by_citations[i - 1])
                break
        # Take from recent pool
        while j < len(by_recency):
            t = by_recency[j].title.lower().strip()
            j += 1
            if t not in seen_interleave:
                seen_interleave.add(t)
                interleaved.append(by_recency[j - 1])
                break
    all_papers = interleaved

    # ── Extract ideas from deduplicated, sorted papers ────
    for paper in all_papers:
        reason = paper_reasons.get(paper.title.lower().strip(), "Search")
        ideas = extract_ideas([paper], config, reason)
        all_ideas.extend(ideas)

    # Look up code only for papers in final ideas (not all papers — too slow)
    if include_code and rc.include_code:
        idea_titles = {idea.paper_title for idea in all_ideas[:rc.max_ideas]}
        for paper in all_papers:
            if paper.title not in idea_titles or paper.code_url:
                continue
            try:
                arxiv_id = paper.paper_id if paper.source == "arxiv" else ""
                code_url = lookup_code(paper.title, arxiv_id=arxiv_id)
                if code_url:
                    paper.code_url = code_url
                    for idea in all_ideas:
                        if idea.paper_title == paper.title:
                            idea.code_url = code_url
                time.sleep(1)
            except Exception:
                pass

    # If no papers found from any source, fall back to local ideas
    if not all_ideas:
        all_ideas = _generate_local_ideas(config)

    # Cap ideas
    all_ideas = all_ideas[:rc.max_ideas]

    # Update ideas.md (unless dry run)
    if not dry_run:
        ideas_path = ar_dir / "ideas.md"
        update_ideas_md(ideas_path, all_ideas)

    return all_ideas, all_papers
