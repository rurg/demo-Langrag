import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parent.parent
CONTEXT_DIR = ROOT / "data" / "context"
MEMORY_FILE = ROOT / "data" / "memory" / "research_memory.json"
REPORTS_DIR = ROOT / "data" / "reports"


def _load_memory():
    if not MEMORY_FILE.exists():
        return {"entries": []}
    return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))


def _save_memory(store: dict):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


@tool
def web_search(query: str, max_results: int = 4) -> str:
    """Search the web. Returns titles, snippets and URLs."""
    try:
        from duckduckgo_search import DDGS
    except ImportError as e:
        return f"duckduckgo-search not installed: {e}"

    results = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "url": item.get("href", ""),
                })
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return f"No results for: {query}"

    lines = [f"Search results for: {query}", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   URL: {r['url']}")
        lines.append(f"   {r['snippet'][:300]}")
        lines.append("")
    return "\n".join(lines)


@tool
def read_local_file(relative_path: str) -> str:
    """Read a file under data/."""
    safe_root = ROOT / "data"
    path = (ROOT / relative_path).resolve()
    if not str(path).startswith(str(safe_root.resolve())):
        return "Error: path must be under data/"
    if not path.exists():
        return f"File not found: {relative_path}"
    return path.read_text(encoding="utf-8")[:6000]


@tool
def save_to_memory(key: str, content: str, tags: str = "") -> str:
    """Save a note to data/memory/research_memory.json."""
    store = _load_memory()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    store["entries"].append({
        "key": key,
        "content": content[:2000],
        "tags": tag_list,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    })
    _save_memory(store)
    return f"Saved: {key}"


@tool
def recall_from_memory(keyword: str) -> str:
    """Find memory entries by keyword."""
    store = _load_memory()
    kw = keyword.lower()
    matches = [
        e for e in store.get("entries", [])
        if kw in e.get("key", "").lower()
        or kw in e.get("content", "").lower()
        or any(kw in t.lower() for t in e.get("tags", []))
    ]
    if not matches:
        return f"No matches for: {keyword}"
    lines = [f"Memory matches for '{keyword}':", ""]
    for entry in matches[-5:]:
        lines.append(f"- [{entry.get('key')}] {entry.get('content', '')[:400]}")
    return "\n".join(lines)


@tool
def write_report(filename: str, content: str) -> str:
    """Write markdown report to data/reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(filename).name
    if not name.endswith(".md"):
        name += ".md"
    path = REPORTS_DIR / name
    path.write_text(content, encoding="utf-8")
    return str(path.relative_to(ROOT))


RESEARCH_TOOLS = [web_search, read_local_file, save_to_memory, recall_from_memory]
