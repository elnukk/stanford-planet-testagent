# coder.py
# Notebook Assembler
# Input:  enriched plan dict from planner.py
# Output: list of nbformat-compatible cells ready for Convex

import json
import os
import re
import sys
import time
from pathlib import Path
import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT_DIR / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

client = anthropic.Anthropic(api_key=api_key)
MODEL = "claude-sonnet-4-6"


def _sanitize_json_strings(s: str) -> str:
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


def _call_llm(prompt: str, max_retries: int = 5) -> str:
    delay = 30
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
            return raw
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                print(f"[coder] Rate limited, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay = min(delay * 2, 120)
            else:
                raise
        except anthropic.APIError:
            raise


# ─── Phase 1: Collect ─────────────────────────────────────────────────────────

def _collect_cells(plan: dict) -> tuple[list[dict], dict]:
    """Flatten selected_cells from all steps, tagged with step context."""
    intake = plan.get("intake", {})
    cells_with_context = []
    for step in plan.get("steps", []):
        for cell in step.get("selected_cells", []):
            cells_with_context.append({
                "step_id": step["step_id"],
                "step_title": step["title"],
                "step_intent": step.get("intent", ""),
                "notebook": cell.get("notebook", ""),
                "cell_index": cell.get("cell_index"),
                "content": cell.get("content", ""),
                "provenance": step.get("provenance", {}),
            })
    return cells_with_context, intake


# ─── Phase 2: Assemble with LLM ───────────────────────────────────────────────

def _assemble_with_llm(cells: list[dict], intake: dict) -> dict:
    """
    LLM pass: dedup imports, normalize variable names, inject placeholders.
    Returns { imports: str, cells: [{ step_id, source, provenance }] }.
    """
    prompt = f"""You are assembling a runnable Jupyter notebook from source cells collected across \
multiple Planet Labs notebooks. The cells implement a satellite data workflow.

INTAKE (use this to fill in real values for placeholders):
{json.dumps(intake, indent=2)}

SOURCE CELLS (in workflow step order):
{json.dumps(cells, indent=2)}

Your job:

1. IMPORT DEDUPLICATION — Collect every import statement from all cells \
(lines starting with `import` or `from X import Y`). Deduplicate. Emit once as a \
single string in the "imports" field. Do not include imports in the per-step cells.

2. VARIABLE NORMALIZATION — Rename variables so each step's output feeds the next step's \
input. Use descriptive names (e.g. `search_results`, `filtered_items`, `clipped_image`). \
Ensure the final notebook reads top-to-bottom without undefined variable errors.

3. PLACEHOLDER INJECTION — Replace any hardcoded values with these standard placeholders, \
filling in the actual values from intake where shown:
   - Hardcoded coordinates or AOI geometry → `AOI_GEOMETRY` \
(set to: {json.dumps(intake.get("region", {}).get("geometry", {}))})
   - Hardcoded start dates → `DATE_START` (set to: "{intake.get("date_range", {}).get("start", "")}")
   - Hardcoded end dates → `DATE_END` (set to: "{intake.get("date_range", {}).get("end", "")}")
   - Hardcoded API key strings → `os.environ["PL_API_KEY"]`
   - Hardcoded product type strings → `PLANET_PRODUCT` \
(set to: "{intake.get("planet_product", "")}")

4. CODE ONLY — Return only executable Python per cell. Do not include markdown or comments \
explaining what you changed. Do not wrap code in backticks.

Return a JSON object with exactly this structure:
{{
  "imports": "<all deduplicated import statements as a single string, newline-separated>",
  "cells": [
    {{
      "step_id": <integer>,
      "source": "<cleaned Python code for this step, no imports>",
      "provenance": {{ "notebook_cells": [...], "docs_urls": [...] }}
    }}
  ]
}}

Rules:
- One entry in "cells" per step_id. If a step has multiple source cells, merge their code.
- If a step has no source cells, emit a stub: `# TODO: implement {{step title}}`.
- Preserve the step_id order.
- Respond with valid JSON only — no markdown fences, no explanation."""

    return _parse_json(_call_llm(prompt))


# ─── Phase 3: Wrap ────────────────────────────────────────────────────────────

def _build_notebook_cells(plan: dict, assembled: dict) -> list[dict]:
    """Build the final nbformat-compatible cell list."""
    cells = []
    steps_by_id = {s["step_id"]: s for s in plan.get("steps", [])}

    # Title markdown cell
    cells.append({
        "cell_type": "markdown",
        "source": f"# {plan.get('workflow_title', 'Satellite Analysis Workflow')}",
        "metadata": {},
    })

    # Imports code cell
    imports_src = assembled.get("imports", "").strip()
    if imports_src:
        cells.append({
            "cell_type": "code",
            "source": imports_src,
            "metadata": {"step_id": None, "provenance": []},
        })

    # Per-step: markdown header + code cell
    for cell in assembled.get("cells", []):
        step_id = cell.get("step_id")
        step = steps_by_id.get(step_id, {})
        title = step.get("title", f"Step {step_id}")
        intent = step.get("intent", "")

        header = f"## Step {step_id}: {title}"
        if intent:
            header += f"\n\n{intent}"
        cells.append({
            "cell_type": "markdown",
            "source": header,
            "metadata": {"step_id": step_id},
        })

        cells.append({
            "cell_type": "code",
            "source": cell.get("source", "").strip(),
            "metadata": {
                "step_id": step_id,
                "provenance": cell.get("provenance", {}),
            },
        })

    return cells


# ─── Main entry point ─────────────────────────────────────────────────────────

def assemble_notebook(plan: dict) -> list[dict]:
    """
    Full coder pipeline:
      1. Collect — flatten selected_cells from all steps
      2. Assemble — LLM deduplicates imports, normalizes variables, injects placeholders
      3. Wrap — build final nbformat-compatible cell list
    Returns list of notebook cells ready for Convex.
    """
    print("[coder] Collecting cells from enriched plan...")
    cells, intake = _collect_cells(plan)
    print(f"[coder] {len(cells)} source cells collected across {len(plan.get('steps', []))} steps")

    print("[coder] Assembling with LLM...")
    assembled = _assemble_with_llm(cells, intake)
    print(f"[coder] {len(assembled.get('cells', []))} assembled cells returned")

    print("[coder] Building notebook cell list...")
    notebook_cells = _build_notebook_cells(plan, assembled)
    print(f"[coder] Done — {len(notebook_cells)} total cells")

    print("\n" + "=" * 60)
    print("ASSEMBLED NOTEBOOK")
    print("=" * 60)
    for cell in notebook_cells:
        if cell["cell_type"] == "markdown":
            print(f"\n[markdown]\n{cell['source']}")
        else:
            step_id = cell["metadata"].get("step_id")
            label = f"Step {step_id}" if step_id is not None else "imports"
            print(f"\n[code — {label}]\n{cell['source']}")
    print("\n" + "=" * 60 + "\n")

    return notebook_cells


if __name__ == "__main__":
    from planner import plan_workflow

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

    print("\nAssembling notebook...")
    notebook_cells = assemble_notebook(plan)

    print("\n=== Assembled Notebook Cells ===")
    print(json.dumps(notebook_cells, indent=2))
