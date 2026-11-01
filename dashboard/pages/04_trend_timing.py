"""
Trend Timing & Velocity Analytics Page
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st
from dashboard.components.charts import plot_publish_heatmap, plot_velocity_distribution
from dashboard.components.filters import render_sidebar_filters
from dashboard.components.metric_cards import render_metric


@st.cache_data
def load_data():
    df = pd.read_parquet(ROOT_DIR / "data" / "processed" / "youtube_transformed.parquet")
    pivot_eng = pd.read_parquet(ROOT_DIR / "data" / "processed" / "timing_engagement_pivot.parquet")
    with open(ROOT_DIR / "data" / "processed" / "timing_metrics.json", "r", encoding="utf-8") as f:
        timing_metrics = json.load(f)
    return df, pivot_eng, timing_metrics


def main():
    st.set_page_config(page_title="Trend Timing | YouTube Analytics", page_icon="⏰", layout="wide")
    raw_df, pivot_eng, timing_metrics = load_data()
    df = render_sidebar_filters(raw_df)

    st.title("⏰ Trend Timing & Algorithmic Velocity")
    st.caption("Temporal optimization, upload scheduling windows, and algorithmic pickup duration.")

    # High-level timing KPIs
    vel = timing_metrics["velocity"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Trending within 48h", f"{vel['within_48h_pct']}%", "Fast Traction", True)
    with c2:
        render_metric("Trending within 72h", f"{vel['within_72h_pct']}%", "Three-Day Window", True)
    with c3:
        render_metric("Median Days to Trend", f"{vel['median_days_to_trend']:.0f} Days", "Recommendation Latency", True)
    with c4:
        render_metric("Prime Window", "Thursday - Friday", "14:00 - 17:00 UTC", True)

    st.markdown("---")

    # Publishing Heatmap
    st.plotly_chart(plot_publish_heatmap(pivot_eng), use_container_width=True)

    st.markdown("---")

    # Virality Velocity Distribution
    st.plotly_chart(plot_velocity_distribution(df), use_container_width=True)


if __name__ == "__main__":
    main()
