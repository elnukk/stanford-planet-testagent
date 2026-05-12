# metadata_ranker.py
#
# Notebook-level ranking utilities.
#
# These functions decide which notebooks are worth searching before the more
# expensive cell-level scorer runs.

import re


STOPWORDS = {
    "a",
    "an",
    "the",
    "that",
    "this",
    "these",
    "those",
    "find",
    "cell",
    "code",
    "calls",
    "call",
    "using",
    "with",
    "for",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "by",
    "from",
}


QUERY_EXPANSIONS = {
    "auth": ["authentication", "api_key", "pl_api_key", "session", "headers", "HTTPBasicAuth"],
    "authenticate": ["authentication", "api_key", "pl_api_key", "session", "headers", "HTTPBasicAuth"],
    "orders": ["Orders API", "orders/v2", "create_order", "order_request", "delivery"],
    "order": ["Orders API", "orders/v2", "create_order", "order_request", "delivery"],
    "data": ["Data API", "data/v1", "quick-search", "item-types", "assets"],
    "imagery": ["image", "scene", "PSScene", "assets", "download"],
    "retrieve": ["search", "quick-search", "download", "assets"],
    "vegetation": ["NDVI", "normalized difference vegetation index", "nir", "red", "band_nir", "band_red"],
    "index": ["NDVI", "normalized difference vegetation index", "nir", "red", "band_nir", "band_red"],
    "visualize": ["imshow", "plt.show", "matplotlib", "rasterio", "plot", "display"],
    "raster": ["rasterio", "imshow", "matplotlib", "GeoTIFF", "plot"],
    "output": ["show", "display", "savefig", "plot"],
}


def normalize_text(text: str) -> str:
    """Normalize text for rough keyword matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_\-/\.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize text and remove very common words."""
    tokens = normalize_text(text).split()
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def expanded_query_text(query: str) -> str:
    """Add predictable workflow synonyms before scoring metadata."""
    tokens = tokenize(query)
    expanded_terms = list(tokens)

    normalized_query = normalize_text(query)

    for key, expansions in QUERY_EXPANSIONS.items():
        if key in normalized_query:
            expanded_terms.extend(expansions)

    return " ".join(expanded_terms)


def overlap_score(query: str, text: str) -> float:
    """Simple weighted token overlap score."""
    query_tokens = tokenize(expanded_query_text(query))
    text_tokens = tokenize(text)

    if not query_tokens or not text_tokens:
        return 0.0

    text_set = set(text_tokens)
    score = 0.0

    for token in query_tokens:
        if token in text_set:
            score += 1.0

    normalized_query = " ".join(tokenize(query))
    normalized_text = " ".join(text_tokens)

    if normalized_query and normalized_query in normalized_text:
        score += 3.0

    return score


def weighted_metadata_score(query: str, filename: str, entry: dict) -> float:
    """Score one notebook metadata entry against the query."""
    score = 0.0

    description = entry.get("description", "")
    apis_used = " ".join(entry.get("apis_used", []))
    planet_product = entry.get("planet_product", "")
    use_case = entry.get("use_case", "")

    score += 2.0 * overlap_score(query, description)
    score += 1.8 * overlap_score(query, apis_used)
    score += 1.3 * overlap_score(query, use_case)
    score += 1.0 * overlap_score(query, planet_product)
    score += 0.8 * overlap_score(query, filename)

    # Direct filename hints are useful in this notebook corpus.
    normalized_filename = normalize_text(filename)
    normalized_query = normalize_text(query)

    if "ndvi" in normalized_query and "ndvi" in normalized_filename:
        score += 8.0
    if "order" in normalized_query and "order" in normalized_filename:
        score += 6.0
    if "data api" in normalized_query and "data" in normalized_filename:
        score += 5.0
    if "visual" in normalized_query and ("visual" in normalized_filename or "raster" in normalized_filename):
        score += 6.0
    if "raster" in normalized_query and "raster" in normalized_filename:
        score += 6.0

    return score


def rank_notebooks(query: str, metadata: dict, top_k: int = 3) -> list[tuple[str, float]]:
    """Return notebook filenames ranked by metadata relevance."""
    scored = []

    for filename, entry in metadata.items():
        if not isinstance(entry, dict):
            continue

        score = weighted_metadata_score(query, filename, entry)
        scored.append((filename, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def expand_query(query: str, results: list[dict]) -> str:
    """
    Refine a query using high-signal terms found in earlier cell matches.

    This is intentionally conservative. It only adds terms that are likely to
    help the next notebook-ranking pass.
    """
    extra_terms = []

    for result in results[:8]:
        text = normalize_text(result.get("content", ""))

        if "create_order" in text:
            extra_terms.append("create_order")
        if "order_request" in text:
            extra_terms.append("order_request")
        if "orders/v2" in text:
            extra_terms.append("orders/v2")
        if "requests.session" in text:
            extra_terms.append("requests.Session")
        if "session.auth" in text:
            extra_terms.append("session.auth")
        if "quick-search" in text:
            extra_terms.append("quick-search")
        if "item-types" in text:
            extra_terms.append("item-types")
        if "assets" in text:
            extra_terms.append("assets")
        if "ndvi" in text:
            extra_terms.append("ndvi")
        if "band_nir" in text:
            extra_terms.append("band_nir")
        if "band_red" in text:
            extra_terms.append("band_red")
        if "imshow" in text:
            extra_terms.append("imshow")
        if "rasterio" in text:
            extra_terms.append("rasterio")
        if "matplotlib" in text:
            extra_terms.append("matplotlib")

    unique_extra_terms = []
    for term in extra_terms:
        if term not in unique_extra_terms and normalize_text(term) not in normalize_text(query):
            unique_extra_terms.append(term)

    if not unique_extra_terms:
        return query

    return query + " " + " ".join(unique_extra_terms[:5])
