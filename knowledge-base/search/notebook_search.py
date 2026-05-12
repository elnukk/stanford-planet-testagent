# notebook_search.py
#
# Cell-level retrieval and scoring.
#
# The scoring is intent-gated, meaning NDVI signals only help vegetation queries,
# Orders API signals only help order queries, visualization signals only help
# visualization queries, etc.

import json
import re
from pathlib import Path

from search.metadata_ranker import normalize_text, overlap_score, tokenize


CODE_CELL_BONUS = 4.0
BASE_OVERLAP_WEIGHT = 2.0


SIGNALS = {
    "auth": {
        "terms": [
            "requests.session",
            "session.auth",
            "authorization",
            "api_key",
            "pl_api_key",
            "httpbasicauth",
            "auth.from_key",
            "headers",
            "token",
        ],
        "weight": 6.0,
        "reason": "contains authentication, API key, session, or header setup",
    },
    "orders": {
        "terms": [
            "orders/v2",
            "orders api",
            "create_order",
            "order_request",
            "build_request",
            "client('orders')",
            "ordersclient",
            "order_params",
            "download_order",
            "wait(",
        ],
        "weight": 6.0,
        "reason": "matches Orders API request, creation, delivery, or download logic",
    },
    "data_api": {
        "terms": [
            "data/v1",
            "quick-search",
            "item-types",
            "asset-types",
            "assets",
            "item_type",
            "search_request",
            "data api",
            "activate",
            "download assets",
        ],
        "weight": 6.0,
        "reason": "matches Data API imagery search, item, scene, or asset retrieval logic",
    },
    "vegetation": {
        "terms": [
            "ndvi",
            "normalized difference vegetation index",
            "band_nir",
            "band_red",
            "nir",
            "red",
            "vegetation index",
            "index(sample.nir",
            "sample.nir",
            "sample.red",
        ],
        "weight": 6.0,
        "reason": "matches vegetation-index logic such as NDVI, red band, or NIR band use",
    },
    "visualization": {
        "terms": [
            "imshow",
            "plt.show",
            "show(",
            "matplotlib",
            "rasterio.plot",
            "show_hist",
            "colorbar",
            "fig.savefig",
            "plt.figure",
            "display",
        ],
        "weight": 8.0,
        "reason": "matches raster visualization or plotting logic",
    },
    "imports_setup": {
        "terms": [
            "import requests",
            "import rasterio",
            "import numpy",
            "import numpy as np",
            "import matplotlib",
            "from planet import",
            "planet.session",
        ],
        "weight": 2.0,
        "reason": "contains imports or client setup useful before workflow execution",
    },
}


QUERY_INTENT_TERMS = {
    "auth": ["auth", "authenticate", "authentication", "token", "key", "session", "header", "headers"],
    "orders": ["order", "orders", "delivery", "deliver", "download order", "create order"],
    "data_api": ["data api", "data", "imagery", "image", "scene", "item", "asset", "retrieve", "search"],
    "vegetation": ["ndvi", "vegetation", "index", "nir", "red", "band"],
    "visualization": ["visualize", "visualise", "plot", "display", "show", "raster", "output", "map", "imshow"],
}


def detect_intents(query: str) -> dict[str, bool]:
    """Return which retrieval intents are relevant to the query."""
    normalized_query = normalize_text(query)

    intents = {}
    for intent_name, terms in QUERY_INTENT_TERMS.items():
        intents[intent_name] = any(term in normalized_query for term in terms)

    return intents


def parse_notebook_cells(notebook_path: str) -> list[dict]:
    """Load notebook cells into a simpler list of dictionaries."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    parsed_cells = []

    for index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source", [])

        if isinstance(source, list):
            content = "".join(source)
        else:
            content = str(source)

        parsed_cells.append(
            {
                "cell_index": index,
                "cell_type": cell.get("cell_type", "unknown"),
                "content": content,
            }
        )

    return parsed_cells


def find_matching_terms(text: str, terms: list[str]) -> list[str]:
    """Return signal terms that appear in the cell text."""
    lowered = normalize_text(text)
    matches = []

    for term in terms:
        normalized_term = normalize_text(term)
        if normalized_term and normalized_term in lowered:
            matches.append(term)

    # Keep order but remove duplicates.
    unique_matches = []
    for match in matches:
        if match not in unique_matches:
            unique_matches.append(match)

    return unique_matches


def extract_function_calls(text: str) -> list[str]:
    """Extract simple function/method call names from code text."""
    raw_matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_\.]*?)\s*\(", text)

    ignored = {
        "if",
        "for",
        "while",
        "print",
        "len",
        "list",
        "dict",
        "set",
        "str",
        "int",
        "float",
        "range",
        "enumerate",
    }

    cleaned = []
    for match in raw_matches:
        short_name = match.split(".")[-1]
        if short_name.lower() in ignored:
            continue
        if match not in cleaned:
            cleaned.append(match)

    return cleaned[:5]


def build_reason(cell: dict, query: str, matched_signal_reasons: list[str], matched_terms_by_signal: dict) -> str:
    """Create a human-readable explanation for why a cell was retrieved."""
    reasons = []

    if cell.get("cell_type") == "code":
        reasons.append("code cell that can be reused in a workflow step")
    elif cell.get("cell_type") == "markdown":
        reasons.append("markdown cell that explains a workflow step")

    for signal_reason in matched_signal_reasons:
        signal_terms = matched_terms_by_signal.get(signal_reason, [])
        if signal_terms:
            term_preview = ", ".join(signal_terms[:4])
            reasons.append(f"{signal_reason} ({term_preview})")
        else:
            reasons.append(signal_reason)

    query_tokens = set(tokenize(query))
    content_tokens = set(tokenize(cell.get("content", "")))
    shared_terms = sorted(query_tokens.intersection(content_tokens))
    if shared_terms:
        reasons.append("matches query terms: " + ", ".join(shared_terms[:6]))

    if cell.get("cell_type") == "code":
        calls = extract_function_calls(cell.get("content", ""))
        if calls:
            reasons.append("uses relevant function/method calls: " + ", ".join(calls[:5]))

    if not reasons:
        return "relevant match"

    return "; ".join(reasons) + "."


def score_cell_relevance(cell: dict, query: str) -> tuple[float, str | None]:
    """Score one cell against the query and return score plus explanation."""
    content = cell.get("content", "")
    stripped_content = content.strip()

    if not stripped_content:
        return 0.0, None

    # Ignore synthetic metadata cells. They can help notebook ranking, but they
    # should not be returned as reusable workflow cells.
    if stripped_content.startswith("<!-- PLANET NOTEBOOK METADATA"):
        return 0.0, None

    intents = detect_intents(query)
    score = BASE_OVERLAP_WEIGHT * overlap_score(query, content)

    if cell.get("cell_type") == "code":
        score += CODE_CELL_BONUS

    matched_signal_reasons = []
    matched_terms_by_reason = {}

    # Intent-gated scoring. These strong signals only apply if the query asks for them.
    for intent_name in ["auth", "orders", "data_api", "vegetation", "visualization"]:
        if not intents.get(intent_name, False):
            continue

        signal = SIGNALS[intent_name]
        matched_terms = find_matching_terms(content, signal["terms"])

        if matched_terms:
            score += signal["weight"]
            matched_signal_reasons.append(signal["reason"])
            matched_terms_by_reason[signal["reason"]] = matched_terms

    # Imports/setup are useful but should not dominate final ranking.
    setup_terms = find_matching_terms(content, SIGNALS["imports_setup"]["terms"])
    if setup_terms and cell.get("cell_type") == "code":
        score += SIGNALS["imports_setup"]["weight"]
        matched_signal_reasons.append(SIGNALS["imports_setup"]["reason"])
        matched_terms_by_reason[SIGNALS["imports_setup"]["reason"]] = setup_terms

    lowered = normalize_text(content)

    # Penalize cells that are only setup imports when the query needs an action.
    if cell.get("cell_type") == "code":
        non_comment_lines = [line for line in stripped_content.splitlines() if line.strip() and not line.strip().startswith("#")]
        only_imports = all(line.strip().startswith(("import ", "from ", "%")) for line in non_comment_lines)
        if only_imports:
            score -= 2.0

    # For visualization queries, prefer actual rendering cells over compute-only cells.
    if intents.get("visualization", False):
        if "imshow" in lowered or "plt show" in lowered or "show(" in content.lower():
            score += 4.0
        if "ndvi =" in lowered and "imshow" not in lowered and "show(" not in content.lower():
            score -= 3.0

    # For vegetation queries, prefer cells that actually compute an index.
    if intents.get("vegetation", False):
        has_index_formula = (
            "ndvi =" in lowered
            or "def ndvi" in lowered
            or "index(sample nir" in lowered
            or "sample nir" in lowered and "sample red" in lowered
            or "band_nir" in lowered and "band_red" in lowered and "/" in content
        )
        if has_index_formula:
            score += 5.0

        # Visualization-only NDVI cells are helpful, but less central for "calculate vegetation index".
        if "imshow" in lowered and not has_index_formula:
            score -= 2.0

    # For Orders queries, prefer true Orders API over Tasking API if both appear.
    if intents.get("orders", False):
        if "compute/ops/orders/v2" in content.lower() or "client('orders')" in content.lower() or "create_order" in lowered:
            score += 4.0
        if "tasking" in lowered and "orders/v2" not in lowered and "create_order" not in lowered:
            score -= 3.0

    # For Data API queries, prefer quick-search/search/assets retrieval over plain auth setup.
    if intents.get("data_api", False):
        if "quick-search" in lowered or "search_request" in lowered or "item-types" in lowered or "assets" in lowered:
            score += 4.0
        if "session auth" in lowered and not any(term in lowered for term in ["quick-search", "assets", "item-types"]):
            score -= 1.5

    if score <= 0:
        return 0.0, None

    reason = build_reason(cell, query, matched_signal_reasons, matched_terms_by_reason)
    return score, reason


def search_notebook(notebook_path: str, query: str, top_k: int = 3) -> list[dict]:
    """Return the top cell matches from one notebook."""
    notebook_name = Path(notebook_path).name
    cells = parse_notebook_cells(notebook_path)

    results = []

    for cell in cells:
        score, reason = score_cell_relevance(cell, query)

        if reason is None:
            continue

        results.append(
            {
                "notebook": notebook_name,
                "cell_index": cell["cell_index"],
                "cell_type": cell["cell_type"],
                "content": cell["content"].strip(),
                "reason": reason,
                "score": score,
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]
