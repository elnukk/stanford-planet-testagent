# intake_bot.py
# M1 - Brandyn


import json
import os
import re
from pathlib import Path
from google import genai
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# ─── CLIENT SETUP ─────────────────────────────────────────────────────────────
gem_key = os.getenv("GEMINI_API_KEY")
if not gem_key:
    raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=gem_key)
MODEL = "gemini-2.0-flash"

# ─── LAYER 1: FIXED QUESTIONS ─────────────────────────────────────────────────
LAYER_ONE_QUESTIONS = [
    {
        "id": "region",
        "text": "What region are you interested in? (e.g. country name, bounding box, or describe an area)",
    },
    {
        "id": "date_range",
        "text": "What time range? (e.g. 'Jan 2024 to June 2024' or 'last 3 growing seasons')",
    },
    {
        "id": "planet_product",
        "text": "Which Planet product do you have access to? (e.g. PlanetScope, Sentinel-2, Planetary Variables, Basemaps, SkySat)",
    },
]


def load_notebooks_metadata() -> dict:
    """
    Reads notebooks_metadata.json from knowledge-base/data/.
    Falls back to the current directory if not found there.
    """
    search_paths = [
        Path("knowledge-base/data/notebooks_metadata.json"),
        Path("notebooks_metadata.json"),
        Path(__file__).parent / "knowledge-base" / "data" / "notebooks_metadata.json",
        Path(__file__).parent / "notebooks_metadata.json",
    ]
    for path in search_paths:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)

    raise FileNotFoundError(
        "notebooks_metadata.json not found. "
        "Expected at knowledge-base/data/notebooks_metadata.json"
    )


def run_layer_one() -> dict:
    """
    Asks the 3 fixed questions in order.
    Returns: {"region": str, "date_range": str, "planet_product": str}
    """
    print("\n── Layer 1: Core Information ─────────────────────────────")
    answers = {}
    for q in LAYER_ONE_QUESTIONS:
        answer = ""
        while not answer.strip():
            answer = input(f"\n{q['text']}\n> ").strip()
        answers[q["id"]] = answer
    return answers


def generate_layer_two_questions(layer_one: dict, metadata: dict) -> list[str]:
    """
    LLM call: given Layer 1 answers + notebook catalog, generate
    3–5 targeted follow-up questions to disambiguate the use case.
    Returns a list of question strings.
    """
    prompt = f"""You are an assistant helping a user build a satellite data analysis
workflow using Planet APIs. The user has told you:
- Region: {layer_one['region']}
- Time range: {layer_one['date_range']}
- Planet product: {layer_one['planet_product']}

Here is a catalog of available workflows:
{json.dumps(metadata, indent=2)}

Generate between 3 and 5 follow-up questions that will help you confidently
select and parameterize the right workflow for this user.

Rules:
- Each question should eliminate a large portion of the catalog when answered.
- Do not ask for information already provided above.
- Ask only what is necessary — stop when you have enough to fill the full schema.
- Questions should be plain, conversational, and specific.

Respond ONLY with a JSON array of question strings, no explanation, no markdown fences:
["Question 1?", "Question 2?", ...]"""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    raw = response.text.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    questions = json.loads(raw)
    # Enforce max 5 follow-up questions
    return questions[:5]


def run_layer_two(questions: list[str]) -> dict:
    """
    Asks the dynamically generated follow-up questions one at a time.
    Returns: {"question text": "user answer", ...}
    """
    print("\n── Layer 2: Follow-up Questions ──────────────────────────")
    answers = {}
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}]")
        answer = ""
        while not answer.strip():
            answer = input(f"{question}\n> ").strip()
        answers[question] = answer
    return answers


def synthesize_intake_json(layer_one: dict, layer_two: dict) -> dict:
    """
    Final LLM call: turns the full conversation into the structured intake JSON.
    Infers use_case, inferred_intent, and constraints — does not ask the user.
    Returns a dict matching the intake schema.
    """
    conversation_summary = "Layer 1 answers:\n"
    for key, val in layer_one.items():
        conversation_summary += f"  {key}: {val}\n"
    conversation_summary += "\nLayer 2 answers:\n"
    for question, answer in layer_two.items():
        conversation_summary += f"  Q: {question}\n  A: {answer}\n"

    user_description = " | ".join(
        f"{q}: {a}" for q, a in {**layer_one, **layer_two}.items()
    )

    prompt = f"""You are synthesizing a satellite data analysis intake form from a conversation.

Here is the full conversation:
{conversation_summary}

Based solely on what the user said, produce a JSON object with exactly these fields:

{{
  "region": {{
    "type": "Feature",
    "properties": {{}},
    "geometry": {{
      "type": "Polygon",
      "coordinates": []
    }},
    "description": "<human-readable description of the region the user mentioned>"
  }},
  "date_range": {{
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD"
  }},
  "temporal_resolution": "<infer from context: daily | weekly | biweekly | monthly | seasonal | unknown>",
  "planet_product": "<exact product name the user mentioned>",
  "use_case": "<short label for the analytical use case, e.g. 'bare soil detection'>",
  "user_description": "<verbatim or near-verbatim summary of what the user said>",
  "inferred_intent": "<one sentence describing what the user is ultimately trying to achieve>",
  "constraints": ["<constraint 1>", "<constraint 2>"]
}}

Rules:
- For region.geometry.coordinates: leave as empty array [] if the user did not provide a bounding box.
- For date_range: if the user gave relative dates (e.g. "last 3 years"), compute from today.
- For constraints: infer from context (e.g. cloud cover needs, resolution requirements, delivery format).
- user_description should be a faithful condensation of the raw conversation.
- Do NOT invent information not present in the conversation.
- Respond with valid JSON only — no markdown fences, no explanation."""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    raw = response.text.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    intake = json.loads(raw)

    # Always inject the raw user_description from conversation as a fallback
    if not intake.get("user_description"):
        intake["user_description"] = user_description

    return intake


def run_intake() -> dict:
    """
    Main entry point. Runs the full pipeline:
      run_layer_one
        → load_notebooks_metadata
        → generate_layer_two_questions
        → run_layer_two
        → synthesize_intake_json
        → return intake dict
    """
    print("\n══════════════════════════════════════════════════════════")
    print("  Planet Satellite Workflow Intake")
    print("══════════════════════════════════════════════════════════")

    # Layer 1
    layer_one = run_layer_one()

    # Load catalog for follow-up question generation
    print("\nLoading workflow catalog...")
    metadata = load_notebooks_metadata()

    # Layer 2
    print("\nGenerating follow-up questions...")
    questions = generate_layer_two_questions(layer_one, metadata)
    layer_two = run_layer_two(questions)

    # Synthesize
    print("\nSynthesizing intake JSON...")
    intake = synthesize_intake_json(layer_one, layer_two)

    print("\n══════════════════════════════════════════════════════════")
    print("  Intake Complete")
    print("══════════════════════════════════════════════════════════")
    print(json.dumps(intake, indent=2))

    return intake


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_intake()
