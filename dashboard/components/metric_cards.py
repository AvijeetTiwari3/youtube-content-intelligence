"""
Metric Cards Component for Streamlit
"""

import streamlit as st


def render_metric(title: str, value: str, delta: str = None, is_positive: bool = True, help_text: str = None):
    """Renders a clean, responsive metric card using native Streamlit metrics."""
    st.metric(
        label=title,
        value=value,
        delta=delta if delta else None,
        delta_color="normal" if is_positive else "inverse",
        help=help_text,
    )
