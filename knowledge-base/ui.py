"""Streamlit UI for testing the Planet workflow agent.

Run from the knowledge-base/ directory:
    streamlit run ui.py

Two tabs:
  1. Run Agent  — fill in intake fields, execute plan+code pipeline, view output cells
  2. History    — all past inputs stored in ui_history.json (locally)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT_DIR / ".env")

HISTORY_FILE = Path(__file__).parent / "ui_history.json"


# ── history helpers ────────────────────────────────────────────────────────────

def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_to_history(entry: dict) -> None:
    history = load_history()
    history.insert(0, entry)
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Planet Agent Tester", layout="wide")
st.title("Planet Workflow Agent — Test UI")

tab_run, tab_history = st.tabs(["Run Agent", "History"])


# ── Tab 1: Run Agent ──────────────────────────────────────────────────────────

with tab_run:
    st.subheader("Intake")

    col1, col2 = st.columns(2)

    with col1:
        region_desc = st.text_input(
            "Region",
            placeholder="e.g. Central Valley, California",
        )
        date_start = st.date_input("Date range — start")
        date_end = st.date_input("Date range — end")
        planet_product = st.selectbox(
            "Planet product",
            ["PlanetScope", "SkySat", "Basemap", "Sentinel-2", "Planetary Variables"],
        )

    with col2:
        use_case = st.text_input(
            "Use case",
            placeholder="e.g. bare soil detection",
        )
        user_description = st.text_area(
            "User description",
            placeholder="Describe what you want to analyze...",
            height=80,
        )
        inferred_intent = st.text_input(
            "Inferred intent (optional)",
            placeholder="e.g. monitor tillage events before planting season",
        )
        temporal_resolution = st.selectbox(
            "Temporal resolution",
            ["daily", "weekly", "biweekly", "monthly", "seasonal", "unknown"],
            index=2,
        )
        constraints = st.text_input(
            "Constraints (comma-separated)",
            placeholder="e.g. needs cloud masking, RGB+NIR available",
        )

    run_btn = st.button("Run Agent", type="primary", use_container_width=True)

    if run_btn:
        if not region_desc or not use_case or not user_description:
            st.error("Region, Use case, and User description are required.")
        else:
            intake = {
                "region": {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "Polygon", "coordinates": []},
                    "description": region_desc,
                },
                "date_range": {
                    "start": str(date_start),
                    "end": str(date_end),
                },
                "temporal_resolution": temporal_resolution,
                "planet_product": planet_product,
                "use_case": use_case,
                "user_description": user_description,
                "inferred_intent": inferred_intent or user_description,
                "constraints": [c.strip() for c in constraints.split(",") if c.strip()],
            }

            st.divider()
            st.subheader("Intake JSON sent to agent")
            st.json(intake)

            import dspy
            from dspy_agent import (
                search_notebooks_tool, search_planet_docs_tool,
                plan_workflow_tool, code_notebook_tool,
                BiodiversityAgentSignature, _extract_notebook_cells,
                DEFAULT_MODEL,
            )

            try:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                lm = dspy.LM(DEFAULT_MODEL, api_key=api_key)
                dspy.configure(lm=lm)

                # Provide intake directly so the agent skips the interactive intake tool
                _intake = intake
                def run_intake_tool() -> dict:
                    """Returns the pre-collected intake from the UI form."""
                    return _intake

                agent = dspy.ReAct(
                    BiodiversityAgentSignature,
                    tools=[run_intake_tool, search_notebooks_tool, search_planet_docs_tool,
                           plan_workflow_tool, code_notebook_tool],
                )

                user_request = (
                    f"{intake['user_description']} "
                    f"Region: {intake['region']['description']}. "
                    f"Product: {intake['planet_product']}. "
                    f"Use case: {intake['use_case']}."
                )

                with st.spinner("Running agent (this takes ~60–120 s)..."):
                    result = agent(user_request=user_request)

                trajectory = getattr(result, "trajectory", {}) or {}
                cells = _extract_notebook_cells(trajectory) or []
                workflow_title = intake.get("use_case", "Satellite Analysis Workflow")

                save_to_history({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "intake": intake,
                    "workflow_title": workflow_title,
                    "step_count": 0,
                    "cell_count": len(cells),
                })

                st.success(f"Done — {len(cells)} cells generated")

                # ── Trajectory ────────────────────────────────────────────────
                if trajectory:
                    st.divider()
                    st.subheader("Agent Trajectory")
                    step_idx = 0
                    while f"tool_name_{step_idx}" in trajectory:
                        thought = trajectory.get(f"thought_{step_idx}", "")
                        tool_name = trajectory.get(f"tool_name_{step_idx}", "")
                        tool_args = trajectory.get(f"tool_args_{step_idx}", {})
                        observation = trajectory.get(f"observation_{step_idx}", "")

                        label = f"Step {step_idx + 1}: {tool_name}"
                        with st.expander(label, expanded=False):
                            if thought:
                                st.markdown("**Thought**")
                                st.text(thought[:2000] + ("..." if len(thought) > 2000 else ""))
                            st.markdown(f"**Tool:** `{tool_name}`")
                            if tool_args:
                                args_str = json.dumps(tool_args, indent=2) if not isinstance(tool_args, str) else tool_args
                                st.markdown("**Args**")
                                st.code(args_str[:800] + ("..." if len(args_str) > 800 else ""), language="json")
                            if observation:
                                obs_str = observation if isinstance(observation, str) else json.dumps(observation, indent=2)
                                st.markdown("**Result**")
                                st.text(obs_str[:600] + ("..." if len(obs_str) > 600 else ""))
                        step_idx += 1

                # ── Notebook cells ────────────────────────────────────────────
                if cells:
                    st.divider()
                    st.subheader("Output: Notebook Cells (copy-paste ready)")

                    for i, cell in enumerate(cells):
                        cell_type = cell.get("cell_type", "code")
                        source = cell.get("source", "")
                        if isinstance(source, list):
                            source = "".join(source)

                        label = f"Cell {i + 1} [{cell_type}]"
                        with st.expander(label, expanded=(i < 3)):
                            lang = "python" if cell_type == "code" else "markdown"
                            st.code(source, language=lang)

                    st.divider()
                    st.subheader("Full cells JSON")
                    st.code(json.dumps(cells, indent=2), language="json")

            except Exception as exc:
                st.error(f"Agent error: {exc}")
                st.exception(exc)


# ── Tab 2: History ────────────────────────────────────────────────────────────

with tab_history:
    st.subheader("Past inputs")

    history = load_history()

    if not history:
        st.info("No runs yet. Use the Run Agent tab to get started.")
    else:
        if st.button("Clear history", type="secondary"):
            HISTORY_FILE.write_text("[]")
            st.rerun()

        for i, entry in enumerate(history):
            ts = entry.get("timestamp", "unknown time")
            title = entry.get("workflow_title") or entry["intake"].get("use_case", "run")
            n_cells = entry.get("cell_count", "?")
            n_steps = entry.get("step_count", "?")

            header = f"{ts} — **{title}** ({n_steps} steps, {n_cells} cells)"
            with st.expander(header):
                st.json(entry["intake"])
