"""
Creator Channel Intelligence Page
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.components.charts import DARK_LAYOUT, plot_top_creators
from dashboard.components.filters import render_sidebar_filters
from dashboard.components.metric_cards import render_metric


@st.cache_data
def load_data():
    df = pd.read_parquet(ROOT_DIR / "data" / "processed" / "youtube_transformed.parquet")
    return df


def main():
    st.set_page_config(page_title="Creator Intelligence | YouTube Analytics", page_icon="👤", layout="wide")
    raw_df = load_data()
    df = render_sidebar_filters(raw_df)

    st.title("👤 Creator & Channel Intelligence")
    st.caption("Channel retention on trending boards, creator concentration, and audience reach.")

    # Channel Metrics
    unique_channels = df["channel_title"].nunique()
    top_channel_row = df["channel_title"].value_counts().head(1)
    top_channel_name = top_channel_row.index[0] if len(top_channel_row) else "N/A"
    top_channel_count = top_channel_row.iloc[0] if len(top_channel_row) else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric("Unique Trending Channels", f"{unique_channels:,}", "Creator Ecosystem", True)
    with c2:
        render_metric("Top Performing Channel", top_channel_name, f"{top_channel_count} Appearances", True)
    with c3:
        render_metric("Avg Videos / Creator", f"{len(df)/max(unique_channels, 1):.1f}", "Repeat Frequency", True)

    st.markdown("---")

    # Top Creators Bar Chart
    st.plotly_chart(plot_top_creators(df), use_container_width=True)

    # Creator Reach vs Frequency Scatter
    st.subheader("🎯 Creator Reach vs. Frequency Landscape")
    channel_agg = (
        df.groupby("channel_title")
        .agg(
            total_videos=("video_id", "count"),
            median_views=("views", "median"),
            total_views=("views", "sum"),
            median_engagement=("engagement_rate", "median"),
            primary_category=("category_name", "first"),
        )
        .reset_index()
    )
    fig_scatter = px.scatter(
        channel_agg[channel_agg["total_videos"] >= 3],
        x="total_videos",
        y="median_views",
        size="total_views",
        color="primary_category",
        log_y=True,
        hover_name="channel_title",
        title="<b>Creator Output Volume vs. Median Viewership (Bubble Size = Total Views)</b>",
        labels={"total_videos": "Trending Video Count", "median_views": "Median Views (Log Scale)"},
    )
    fig_scatter.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig_scatter, use_container_width=True)


if __name__ == "__main__":
    main()
