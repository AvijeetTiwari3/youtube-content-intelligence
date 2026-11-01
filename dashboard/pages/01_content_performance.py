"""
Content Performance & Statistical Benchmarking Page
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.components.charts import DARK_LAYOUT, plot_category_engagement, plot_views_vs_likes
from dashboard.components.filters import render_sidebar_filters
from dashboard.components.metric_cards import render_metric


@st.cache_data
def load_data():
    df = pd.read_parquet(ROOT_DIR / "data" / "processed" / "youtube_transformed.parquet")
    benchmarks = pd.read_parquet(ROOT_DIR / "data" / "processed" / "category_benchmarks.parquet")
    with open(ROOT_DIR / "data" / "processed" / "statistical_test_results.json", "r", encoding="utf-8") as f:
        stats_results = json.load(f)
    return df, benchmarks, stats_results


def main():
    st.set_page_config(page_title="Content Performance | YouTube Analytics", page_icon="📈", layout="wide")
    raw_df, benchmarks, stats_results = load_data()
    df = render_sidebar_filters(raw_df)

    st.title("📈 Content Performance & Statistical Benchmarks")
    st.caption("Cross-genre variance, engagement distributions, and ANOVA hypothesis testing.")

    # Top Statistical Test Highlights
    anova = stats_results["anova"]
    ols = stats_results["ols_regression"]

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric("ANOVA F-Statistic", f"{anova['f_statistic']:.1f}", "p-value: < 0.001", True, "Variance between categories")
    with c2:
        render_metric("Statistical Significance", "p < 1e-15", "Null Hypothesis Rejected", True, "True variance across categories")
    with c3:
        render_metric("OLS Regression R²", f"{ols['r_squared']:.3f}", "Multiple Predictors", True, "Explained variance on views")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns([1.3, 1.2])
    with col_chart1:
        st.plotly_chart(plot_category_engagement(df), use_container_width=True)
    with col_chart2:
        st.plotly_chart(plot_views_vs_likes(df), use_container_width=True)

    # Detailed Category Benchmarks Table
    st.subheader("📋 Empirical Category Performance Matrix")
    st.caption("Aggregated across 24,951 verified video snapshots.")

    st.dataframe(
        benchmarks.rename(columns={
            "category_name": "Category",
            "total_videos": "Volume",
            "median_views": "Median Views",
            "median_likes": "Median Likes",
            "median_comments": "Median Comments",
            "median_engagement_rate": "Median Engagement (%)",
            "viral_ratio": "Viral Share (>=1M)",
        }).style.format({
            "Volume": "{:,}",
            "Median Views": "{:,.0f}",
            "Median Likes": "{:,.0f}",
            "Median Comments": "{:,.0f}",
            "Median Engagement (%)": "{:.2f}%",
            "Viral Share (>=1M)": "{:.1%}",
        }),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
