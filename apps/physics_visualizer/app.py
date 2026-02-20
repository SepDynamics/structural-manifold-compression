#!/usr/bin/env python3
"""Tripartite Physics Visualizer (Streamlit UI)."""

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifold.router import TripartiteRouter


router = TripartiteRouter()


st.set_page_config(page_title="Tripartite Physics Visualizer", layout="wide")

st.title("Tripartite Physics Visualizer")
st.caption("Live Hazard Tension telemetry with deterministic vs. heuristic resolution")

if "tension_history" not in st.session_state:
    st.session_state.tension_history = []

if "coverage_history" not in st.session_state:
    st.session_state.coverage_history = []

with st.sidebar:
    st.header("Controls")
    llm_endpoint = st.text_input(
        "LLM endpoint",
        value="http://localhost:11434/api/generate",
        help="Ollama or OpenAI compatible generate endpoint",
    )
    hazard_threshold = st.slider(
        "Hazard threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.01,
    )
    coverage_threshold = st.slider(
        "Coverage threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    st.divider()
    st.subheader("Valkey Status")
    valkey_ok = router.wm.ping()
    st.metric("Valkey online", "Yes" if valkey_ok else "No")

query = st.text_input("Enter a query", placeholder="Ask the Tripartite Engine...")
run_query = st.button("Run Query", type="primary", disabled=not query.strip())

if run_query:
    with st.spinner("Processing query..."):
        verified, response, coverage, matched_documents = router.process_query(
            query=query.strip(),
            hazard_threshold=hazard_threshold,
            coverage_threshold=coverage_threshold,
            llm_endpoint=llm_endpoint,
        )

    tension = max(0.0, 1.0 - coverage / 100.0)
    st.session_state.tension_history.append(tension)
    st.session_state.coverage_history.append(coverage)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Coverage %", f"{coverage:.1f}")
    with col2:
        st.metric("Tension", f"{tension:.3f}")
    with col3:
        st.metric("Mode", "Deterministic" if verified else "Heuristic")

    status_color = "green" if verified else "red"
    st.markdown(
        f"<h3 style='color:{status_color}'>"
        f"{'Deterministic Reflex' if verified else 'ADHD Heuristic Generator'}"
        "</h3>",
        unsafe_allow_html=True,
    )

    if response:
        st.text_area("Response", response, height=220)

    if matched_documents:
        st.subheader("Matched Documents")
        st.write(matched_documents)

st.subheader("Hazard Tension Monitor")
if st.session_state.tension_history:
    chart_data = {
        "tension": st.session_state.tension_history,
        "coverage": st.session_state.coverage_history,
    }
    st.line_chart(chart_data, height=220)
else:
    st.info("Run a query to start the telemetry trace.")
