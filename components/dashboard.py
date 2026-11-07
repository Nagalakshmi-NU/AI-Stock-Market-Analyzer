"""Dashboard rendering helpers for each report section."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_section_heading(title: str, description: str) -> None:
    """Render a consistent heading for a dashboard section."""
    st.markdown(f"### {title}")
    st.caption(description)


def render_summary_table(summary_data: dict[str, Any]) -> None:
    """Display a summary table in the dashboard."""
    if not summary_data:
        return
    st.dataframe(pd.DataFrame([summary_data]))
