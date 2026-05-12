# agentic_search.py
# M1 - Vanesska
#
# Iterative notebook retrieval for workflow-step search.
#
# This file does the high-level retrieval loop:
# 1. Rank notebooks using metadata.
# 2. Search cells inside the best notebooks.
# 3. Check whether the results are strong.
# 4. If weak, expand to more notebooks and refine the query.
# 5. Return the best cells with explanations.

import json
from pathlib import Path

from search.metadata_ranker import expand_query, rank_notebooks
from search.notebook_search import search_notebook


STRONG_SCORE_THRESHOLD = 15.0
MAX_PASSES = 3
FINAL_TOP_K = 5


def load_metadata() -> dict:
    """Load notebook metadata from data/notebooks_metadata.json."""
    metadata_path = Path(__file__).parent / "data" / "notebooks_metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_notebooks(query: str) -> list[dict]:
    """
    Search notebook cells for a workflow-step query.

    This function intentionally does more than one pass. It starts with the
    highest-ranked notebooks, evaluates the quality of cell matches, and expands
    if the first matches are weak.
    """
    base_dir = Path(__file__).parent
    notebooks_dir = base_dir / "notebooks"
    metadata = load_metadata()

    all_results = []
    trace = []
    searched_notebooks = set()
    current_query = query

    for pass_index in range(MAX_PASSES):
        pass_number = pass_index + 1

        # Search more notebooks each pass.
        top_k_notebooks = 3 + pass_index * 3
        ranked_notebooks = rank_notebooks(current_query, metadata, top_k=top_k_notebooks)

        notebooks_to_search = []
        for filename, _ in ranked_notebooks:
            if filename not in searched_notebooks:
                notebooks_to_search.append(filename)
                searched_notebooks.add(filename)

        pass_results = []

        for notebook_filename in notebooks_to_search:
            notebook_path = notebooks_dir / notebook_filename

            if not notebook_path.exists():
                continue

            matches = search_notebook(str(notebook_path), current_query, top_k=4)
            pass_results.extend(matches)

        all_results.extend(pass_results)

        if pass_results:
            best_score = max(result["score"] for result in pass_results)
            average_score = sum(result["score"] for result in pass_results) / len(pass_results)
        else:
            best_score = 0.0
            average_score = 0.0

        if best_score >= STRONG_SCORE_THRESHOLD:
            decision = "strong enough"
        elif pass_number < MAX_PASSES:
            decision = "weak; expanding search"
        else:
            decision = "weak after final pass"

        trace.append(
            {
                "pass": pass_number,
                "query": current_query,
                "searched_notebooks": notebooks_to_search,
                "best_score": round(best_score, 3),
                "average_top_score": round(average_score, 3),
                "decision": decision,
            }
        )

        # If weak, use high-signal terms from what we found to refine the next pass.
        if decision != "strong enough":
            current_query = expand_query(current_query, all_results)
            continue

        # Even if pass 1 is strong, do one refinement pass if possible.
        # This helps find a better exact cell, such as create_order or imshow.
        if pass_number == 1 and pass_number < MAX_PASSES:
            current_query = expand_query(current_query, all_results)
            continue

        break

    # Deduplicate identical notebook/cell matches.
    deduped = {}
    for result in all_results:
        key = (result["notebook"], result["cell_index"])
        if key not in deduped or result["score"] > deduped[key]["score"]:
            deduped[key] = result

    ranked_results = sorted(deduped.values(), key=lambda r: r["score"], reverse=True)

    final_results = []
    for result in ranked_results[:FINAL_TOP_K]:
        public_result = result.copy()
        score = public_result.pop("score")
        public_result["retrieval_notes"] = {
            "hidden_score_used_for_ranking": round(score, 3),
            "trace": trace,
        }
        final_results.append(public_result)

    return final_results


if __name__ == "__main__":
    queries = [
        "retrieve imagery using Data API",
        "authenticate and create an Orders API request",
        "calculate vegetation index",
        "visualize raster output",
    ]

    for q in queries:
        print("\n====================")
        print("QUERY:", q)
        results = search_notebooks(q)
        print(json.dumps(results, indent=2))
