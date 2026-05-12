"""CLI entrypoint for the notebook/docs/intake DSPy agent.

This file keeps tool imports lazy so the agent can boot even when:
- API keys have not been added to `.env` yet
- optional dependencies for a specific tool are not installed

Run from the repo root with:
`python3 knowledge-base/dspy_agent.py "find authentication examples for Orders API"`
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import dspy
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


DEFAULT_MODEL = os.getenv("DSPY_MODEL", "claude-sonnet-4-6")


def search_notebooks_tool(query: str) -> list[dict[str, Any]]:
    """Search local Planet notebooks for cells relevant to a workflow step."""
    from agentic_search import search_notebooks

    return search_notebooks(query)


def search_planet_docs_tool(query: str) -> dict[str, Any]:
    """Search docs.planet.com live for API-specific details."""
    from search.web_search import search_planet_docs

    return search_planet_docs(query)


def run_intake_tool() -> dict[str, Any]:
    """Run the interactive two-layer intake flow and return intake JSON."""
    from intake.intake_bot import run_intake

    return run_intake()


def plan_workflow_tool(intake_json: str) -> dict[str, Any]:
    """Plan a complete satellite data workflow from intake JSON.
    Runs discovery search, generates ordered steps, and enriches each step
    with the best notebook cells and API docs via notebook_search and web_search.
    Returns an enriched plan ready for the coder."""
    from agent.planner import plan_workflow

    intake = json.loads(intake_json) if isinstance(intake_json, str) else intake_json
    return plan_workflow(intake)


def code_notebook_tool(enriched_plan_json: str) -> list[dict[str, Any]]:
    """Assemble a runnable Jupyter notebook from an enriched plan produced by plan_workflow_tool.
    Deduplicates imports, normalizes variable names across steps, injects AOI/date/product
    placeholders from the intake, and returns an ordered list of nbformat-compatible cells
    ready to save to Convex."""
    from agent.coder import assemble_notebook

    plan = json.loads(enriched_plan_json) if isinstance(enriched_plan_json, str) else enriched_plan_json
    return assemble_notebook(plan)


class BiodiversityAgentSignature(dspy.Signature):
    """You are a Planet satellite workflow assistant. For every user request:
    1. Call run_intake_tool to collect region, time range, and product from the user.
    2. Call plan_workflow_tool with the intake JSON to generate an enriched step-by-step plan.
    3. Call code_notebook_tool with the enriched plan JSON to assemble notebook cells.
    Return the assembled notebook cells JSON as your response. Do not generate the workflow yourself — always run all three tools in order."""

    user_request = dspy.InputField()
    response = dspy.OutputField()


class BiodiversityWorkflowAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.agent = dspy.ReAct(
            BiodiversityAgentSignature,
            tools=[search_notebooks_tool, search_planet_docs_tool, run_intake_tool, plan_workflow_tool, code_notebook_tool],
        )

    def forward(self, user_request: str):
        return self.agent(user_request=user_request)


def build_agent():
    # Central place to choose the model used by DSPy.
    # If the team changes models later, they should only need to change
    # configuration here rather than rewriting the agent structure.
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set. Add it to the repo-root .env file.")

    lm = dspy.LM(DEFAULT_MODEL, api_key=api_key)
    dspy.configure(lm=lm)
    return BiodiversityWorkflowAgent()


def _truncate_text(value: Any, limit: int = 1200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def print_trajectory(trajectory: dict[str, Any], show_thoughts: bool = False) -> None:
    step = 0
    printed_any = False

    while f"tool_name_{step}" in trajectory:
        printed_any = True
        print(f"\n=== Step {step + 1} ===")

        if show_thoughts:
            thought = trajectory.get(f"thought_{step}", "")
            print("Thought:")
            print(_truncate_text(thought, limit=1600))

        tool_name = trajectory.get(f"tool_name_{step}", "")
        tool_args = trajectory.get(f"tool_args_{step}", {})
        observation = trajectory.get(f"observation_{step}", "")

        print(f"Tool: {tool_name}")
        print("Args:")
        print(_truncate_text(tool_args, limit=1600))
        print("Observation:")
        print(_truncate_text(observation, limit=2000))

        step += 1

    if printed_any:
        print("\n=== End Trajectory ===")


def _extract_notebook_cells(trajectory: dict[str, Any] | None) -> list[dict] | None:
    """Find the last code_notebook_tool observation in the trajectory and parse it as JSON."""
    if not trajectory:
        return None
    step = 0
    last_cells = None
    while f"tool_name_{step}" in trajectory:
        if trajectory.get(f"tool_name_{step}") == "code_notebook_tool":
            obs = trajectory.get(f"observation_{step}", "")
            try:
                parsed = json.loads(obs) if isinstance(obs, str) else obs
                if isinstance(parsed, list):
                    last_cells = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        step += 1
    return last_cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Planet workflow DSPy agent from the CLI.")
    parser.add_argument(
        "user_request",
        nargs="?",
        help="Natural-language request for the agent. If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--show-trajectory",
        action="store_true",
        help="Print the DSPy ReAct tool trajectory after the run.",
    )
    parser.add_argument(
        "--show-thoughts",
        action="store_true",
        help="Include the model's step-by-step thoughts when printing the trajectory.",
    )
    args = parser.parse_args()

    user_request = args.user_request
    if not user_request:
        user_request = input("User request: ").strip()

    if not user_request:
        raise SystemExit("A user request is required.")

    agent = build_agent()
    result = agent(user_request)

    trajectory = getattr(result, "trajectory", None)

    if args.show_trajectory and trajectory:
        print_trajectory(trajectory, show_thoughts=args.show_thoughts)

    # Extract notebook cells from the last code_notebook_tool observation in the trajectory.
    # result.response is a text summary — the actual cells live in the trajectory.
    notebook_cells = _extract_notebook_cells(trajectory)
    if notebook_cells is not None:
        print("\n=== Notebook Cells ===")
        print(json.dumps(notebook_cells, indent=2))
    else:
        response = getattr(result, "response", result)
        print("\n=== Final Response ===")
        print(response)


if __name__ == "__main__":
    main()
