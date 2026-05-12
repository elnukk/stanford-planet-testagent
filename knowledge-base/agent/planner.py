# planner.py
# Workflow Planner
# Input:  structured intake JSON (from intake_bot.py)
# Output: ordered list of steps, each enriched with selected notebook cells and docs
# Passes result to coder.py

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import anthropic
from dotenv import load_dotenv


# Allow imports from knowledge-base/ (agentic_search, search/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_search import search_notebooks
from search.web_search import search_planet_docs

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT_DIR / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

client = anthropic.Anthropic(api_key=api_key)
MODEL = "claude-sonnet-4-6"
HAIKU_MODEL = "claude-haiku-4-5-20251001"


def _sanitize_json_strings(s: str) -> str:
    """Escape control characters and bad backslashes inside JSON string values only."""
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_string:
            next_c = s[i + 1] if i + 1 < len(s) else ''
            if next_c in '"\\/ bfnrtu':
                result.append(c)
                result.append(next_c)
                i += 2
            else:
                result.append('\\\\')
                i += 1
        elif c == '"':
            in_string = not in_string
            result.append(c)
            i += 1
        elif in_string and ord(c) < 0x20:
            escapes = {'\n': '\\n', '\r': '\\r', '\t': '\\t'}
            result.append(escapes.get(c, f'\\u{ord(c):04x}'))
            i += 1
        else:
            result.append(c)
            i += 1
    return ''.join(result)


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_sanitize_json_strings(raw))


def _call_llm(prompt: str, max_retries: int = 5, model: str = MODEL,
              log_cb=None, stage: str = "") -> str:
    delay = 30
    use_thinking = (model == MODEL)
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}])
            if use_thinking:
                kwargs["max_tokens"] = 16000
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}
            else:
                kwargs["max_tokens"] = 4096

            response = client.messages.create(**kwargs)

            raw = ""
            thinking_text = ""
            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    raw = block.text

            if log_cb and thinking_text:
                log_cb("thinking", stage, thinking_text)

            raw = raw.strip()
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
            return raw
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            if isinstance(e, anthropic.APIStatusError) and e.status_code != 529:
                raise
            if attempt < max_retries - 1:
                print(f"[planner] API overloaded/rate-limited, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay = min(delay * 2, 120)
            else:
                raise
        except anthropic.APIError as e:
            raise

        
def _safe_docs_search(query: str) -> dict:
    try:
        return search_planet_docs(query)
    except Exception:
        return {"content": "", "source_url": "", "section": ""}


PRODUCT_BANDS = {
    "PlanetScope": {
        "bands": ["blue", "green", "red", "nir"],
        "notes": "No SWIR bands. Indices requiring SWIR (BSI, NDTI, TI) cannot be computed. Use NDVI or MSAVI instead.",
    },
    "SkySat": {
        "bands": ["blue", "green", "red", "nir"],
        "notes": "No SWIR bands. Same constraints as PlanetScope.",
    },
    "Basemap": {
        "bands": ["blue", "green", "red", "nir"],
        "notes": "No SWIR bands.",
    },
}


def _band_context(product: str) -> str:
    info = PRODUCT_BANDS.get(product)
    if not info:
        return ""
    return (
        f"AVAILABLE BANDS FOR {product}: {', '.join(info['bands'])}.\n"
        f"BAND CONSTRAINTS: {info['notes']}"
    )


# ─── Phase 1: Generate steps ──────────────────────────────────────────────────

def generate_steps(intake: dict, discovery_cells: list, discovery_docs: dict, log_cb=None) -> dict:
    """
    LLM generates workflow title + ordered steps grounded in what search actually found.
    No hardcoded skeleton — steps reflect real notebooks and docs.
    """
    band_context = _band_context(intake.get("planet_product", ""))

    prompt = f"""You are a satellite data workflow planner for Project Centinela, which helps \
conservation organizations analyze Planet satellite data.

Given a user's analysis request and examples of relevant notebooks and API documentation, \
produce an ordered list of workflow steps for a Jupyter notebook.

USER REQUEST:
{json.dumps(intake, indent=2)}

{band_context}

RELEVANT NOTEBOOK CELLS FOUND:
{json.dumps(discovery_cells, indent=2)}

RELEVANT API DOCUMENTATION FOUND:
{json.dumps(discovery_docs, indent=2)}

Produce a JSON object with exactly this structure:
{{
  "workflow_title": "<short descriptive title for this analysis>",
  "steps": [
    {{
      "step_id": <integer starting at 1>,
      "title": "<step title>",
      "category": "<one of: data_retrieval | preprocessing | index_calculation | analysis | visualization>",
      "intent": "<one sentence: what this step does in the context of this specific workflow>",
      "input": "<what this step takes as input>",
      "output": "<what this step produces as output>",
      "retrieval_query": "<specific technical query to find the best notebook cells and docs for this step>",
      "tools_needed": ["<one or both of: notebook_search | web_search>"]
    }}
  ]
}}

Rules:
- Steps must be grounded in what the notebooks and docs show is actually possible.
- Always include authentication and AOI definition — they are required for every workflow.
- If constraints mention cloud masking or shadow removal, include a preprocessing step.
- Only include index calculation steps for indices computable from the available bands listed above. \
Do not generate steps for indices that require bands the product does not have.
- retrieval_query must be specific and technical. Bad: "search imagery". \
Good: "PlanetScope Data API item search filter by date range and geometry".
- Order steps so the output of each feeds the input of the next.
- For tools_needed: use "notebook_search" when the step needs code examples, \
use "web_search" when the step needs API documentation, use both when it needs both. \
Examples: authentication → both, index calculation → notebook_search, \
API endpoint details → web_search.
- Respond with valid JSON only — no markdown fences, no explanation."""

    return _parse_json(_call_llm(prompt, log_cb=log_cb, stage="Generate Steps"))


# ─── Phase 2: Per-step selection ──────────────────────────────────────────────

def select_best_material(step: dict, cells: list, docs: dict, product: str = "") -> dict:
    """
    LLM selects the most relevant cells and docs for a single step.
    Does not forward everything — makes deliberate choices.
    """
    band_context = _band_context(product)

    prompt = f"""You are selecting the best source material for one step in a satellite data workflow.

STEP:
{json.dumps(step, indent=2)}

{band_context}

CANDIDATE NOTEBOOK CELLS:
{json.dumps(cells, indent=2)}

CANDIDATE API DOCUMENTATION:
{json.dumps(docs, indent=2)}

Produce a JSON object with exactly this structure:
{{
  "selected_cells": [
    {{
      "notebook": "<notebook filename>",
      "cell_index": <integer>,
      "content": "<cell content>",
      "reason": "<why this cell is relevant to this step>"
    }}
  ],
  "selected_docs": {{
    "content": "<most relevant portion of the docs>",
    "source_url": "<url>",
    "reason": "<why this doc is relevant to this step>"
  }}
}}

Rules:
- Select at most 2 notebook cells. If none are relevant, return an empty array.
- Reject any cell that uses bands not available for this product (e.g. SWIR bands B11/B12 for PlanetScope). \
Prefer cells that use only the available bands listed above.
- For docs, include only if genuinely relevant. If not relevant, set all fields to empty strings.
- Be selective — the coder will use this material directly to generate code.
- Respond with valid JSON only — no markdown fences, no explanation."""

    return _parse_json(_call_llm(prompt, model=HAIKU_MODEL))


# ─── Main pipeline ────────────────────────────────────────────────────────────

def plan_workflow(intake: dict, log_cb=None) -> dict:
    """
    Full planner pipeline:
      1. Discovery search (broad) to ground step generation
      2. LLM generates ordered steps from intake + discovery results
      3. Per-step targeted retrieval from notebooks and docs
      4. LLM selects best material per step
    Returns enriched plan ready for coder.py.
    """
    # Phase 1: Discovery search
    discovery_query = " ".join(filter(None, [
        intake.get("use_case", ""),
        intake.get("planet_product", ""),
        intake.get("inferred_intent", ""),
    ]))
    print(f"[planner] Discovery search: {discovery_query!r}")
    if log_cb: log_cb("progress", f"Discovery search: {discovery_query!r}")
    discovery_cells = search_notebooks(discovery_query)
    discovery_docs = _safe_docs_search(discovery_query)
    if log_cb: log_cb("progress", f"Found {len(discovery_cells)} notebook cells")

    # Phase 2: Generate steps
    print("[planner] Generating steps...")
    if log_cb: log_cb("progress", "Generating workflow steps...")
    result = generate_steps(intake, discovery_cells, discovery_docs, log_cb=log_cb)
    workflow_title = result.get("workflow_title", "Satellite Analysis Workflow")
    steps = result.get("steps", [])
    print(f"[planner] {len(steps)} steps generated")
    if log_cb: log_cb("progress", f"{len(steps)} steps planned: {workflow_title}")

    # Phase 3 + 4: Per-step retrieval and selection — run all steps in parallel.
    # Each step is independent: notebook search + docs fetch + Haiku selection.
    product = intake.get("planet_product", "")

    if log_cb: log_cb("progress", f"Retrieving material for {len(steps)} steps in parallel...")

    def _process_step(step: dict) -> dict:
        query = step.get("retrieval_query") or step.get("title", "")
        tools = step.get("tools_needed", ["notebook_search", "web_search"])
        print(f"[planner] Step {step['step_id']}: {tools} → {query!r}")

        cells = search_notebooks(query) if "notebook_search" in tools else []
        docs = (
            _safe_docs_search(query)
            if "web_search" in tools
            else {"content": "", "source_url": "", "section": ""}
        )

        selected = select_best_material(step, cells, docs, product=product)

        return {
            **step,
            "selected_cells": selected.get("selected_cells", []),
            "selected_docs": selected.get("selected_docs", {}),
            "provenance": {
                "notebook_cells": [
                    {"notebook": c["notebook"], "cell_index": c["cell_index"]}
                    for c in selected.get("selected_cells", [])
                ],
                "docs_urls": (
                    [selected["selected_docs"]["source_url"]]
                    if selected.get("selected_docs", {}).get("source_url")
                    else []
                ),
            },
        }

    enriched_by_id: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=min(len(steps), 8)) as executor:
        futures = {executor.submit(_process_step, step): step["step_id"] for step in steps}
        for future in as_completed(futures):
            result = future.result()
            enriched_by_id[result["step_id"]] = result
            print(f"[planner] Step {result['step_id']} done")
            if log_cb: log_cb("progress", f"Step {result['step_id']} ready: {result.get('title', '')}")

    # Restore original step order
    enriched_steps = [enriched_by_id[step["step_id"]] for step in steps]

    return {
        "workflow_title": workflow_title,
        "intake": intake,
        "steps": enriched_steps,
    }


if __name__ == "__main__":
    sample_intake = {
        "region": {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": []},
            "description": "Agricultural fields in the Central Valley, California",
        },
        "date_range": {"start": "2024-03-01", "end": "2024-09-30"},
        "temporal_resolution": "biweekly",
        "planet_product": "PlanetScope",
        "use_case": "bare soil detection",
        "user_description": "I want to monitor when fields are tilled before planting season",
        "inferred_intent": "monitor tillage events in agricultural fields using bare soil index",
        "constraints": ["needs cloud masking", "RGB+NIR available"],
    }

    print("Running workflow planner...")
    plan = plan_workflow(sample_intake)
    print(json.dumps(plan, indent=2))
