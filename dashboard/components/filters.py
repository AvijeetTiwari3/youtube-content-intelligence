"""
Sidebar Filter Component for YouTube Content Intelligence
"""

import pandas as pd
import streamlit as st


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🔍 Content Intelligence Filters")

    # Category Filter
    categories = sorted(df["category_name"].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect(
        "Content Categories",
        options=categories,
        default=categories,
    )

    # Virality Filter
    virality_options = ["All Content", "Viral Only (>= 1M views)", "Non-Viral (< 1M views)"]
    selected_virality = st.sidebar.selectbox("Virality Segment", options=virality_options)

    # Publish Hour Filter
    hour_range = st.sidebar.slider("Upload Hour (0 - 23)", min_value=0, max_value=23, value=(0, 23))

    filtered_df = df.copy()

    if selected_categories:
        filtered_df = filtered_df[filtered_df["category_name"].isin(selected_categories)]

    if selected_virality == "Viral Only (>= 1M views)":
        filtered_df = filtered_df[filtered_df["is_viral"] == 1]
    elif selected_virality == "Non-Viral (< 1M views)":
        filtered_df = filtered_df[filtered_df["is_viral"] == 0]

    filtered_df = filtered_df[
        (filtered_df["publish_hour"] >= hour_range[0]) & (filtered_df["publish_hour"] <= hour_range[1])
    ]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"📺 Active Sample: **{len(filtered_df):,}** / {len(df):,} videos")
    return filtered_df
