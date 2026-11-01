"""
Plotly Chart Factory for YouTube Content Intelligence
Publication-grade visualizations configured with YouTube dark theme aesthetics.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(15, 15, 15, 0.7)",
    plot_bgcolor="rgba(15, 15, 15, 0.7)",
    font=dict(color="#F1F1F1", family="Roboto, Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
    yaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
)


def plot_category_engagement(df: pd.DataFrame) -> go.Figure:
    """Median Engagement Rate by Category."""
    cat_summary = (
        df.groupby("category_name")
        .agg(
            median_eng=("engagement_rate", "median"),
            total_videos=("video_id", "count"),
        )
        .sort_values(by="median_eng", ascending=True)
        .reset_index()
    )

    fig = go.Figure(
        go.Bar(
            x=cat_summary["median_eng"],
            y=cat_summary["category_name"],
            orientation="h",
            marker=dict(color="#FF0000"),
            text=[f"{val:.2f}%" for val in cat_summary["median_eng"]],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="<b>Median Engagement Rate (%) by Content Category</b>",
        xaxis_title="Engagement Rate (Likes + Comments / Views %)",
        yaxis_title="Category",
        **DARK_LAYOUT,
    )
    return fig


def plot_views_vs_likes(df: pd.DataFrame) -> go.Figure:
    """Log-Scale Views vs Likes Scatter."""
    sample_df = df.sample(min(2500, len(df)), random_state=42).copy()
    fig = px.scatter(
        sample_df,
        x="views",
        y="likes",
        color="category_name",
        log_x=True,
        log_y=True,
        opacity=0.6,
        hover_name="title",
        hover_data={"views": ":,", "likes": ":,", "engagement_rate": ":.2f%"},
        title="<b>Consumption vs. Endorsement: Views vs. Likes (Log-Log Scale)</b>",
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig


def plot_publish_heatmap(pivot_df: pd.DataFrame) -> go.Figure:
    """Day of Week x Publish Hour Heatmap."""
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Upload Hour (0-23 UTC)", y="Day of Week", color="Median Engagement (%)"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale="Reds",
        aspect="auto",
        title="<b>Optimal Publishing Window: Engagement Rate Heatmap</b>",
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig


def plot_top_creators(df: pd.DataFrame) -> go.Figure:
    """Top Channels Ranked by Trending Appearances."""
    top_channels = (
        df.groupby("channel_title")
        .agg(
            appearances=("video_id", "count"),
            median_views=("views", "median"),
            median_eng=("engagement_rate", "median"),
        )
        .sort_values(by="appearances", ascending=True)
        .tail(15)
        .reset_index()
    )

    fig = go.Figure(
        go.Bar(
            x=top_channels["appearances"],
            y=top_channels["channel_title"],
            orientation="h",
            marker=dict(color="#3EA6FF"),
            text=[f"{a:,} videos" for a in top_channels["appearances"]],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="<b>Top 15 Creators by Frequency of Trending Content</b>",
        xaxis_title="Total Trending Snapshots",
        yaxis_title="Channel",
        **DARK_LAYOUT,
    )
    return fig


def plot_feature_importance(importances: dict) -> go.Figure:
    """Random Forest Virality Predictor Feature Importances."""
    feat_df = pd.DataFrame(
        list(importances.items()), columns=["Feature", "Importance"]
    ).sort_values(by="Importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=feat_df["Importance"],
            y=feat_df["Feature"],
            orientation="h",
            marker=dict(color="#00C853"),
            text=[f"{v:.3f}" for v in feat_df["Importance"]],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="<b>ML Feature Importance: What Drives Video Virality?</b>",
        xaxis_title="Relative Feature Importance (MDI)",
        yaxis_title="Predictor Feature",
        **DARK_LAYOUT,
    )
    return fig


def plot_velocity_distribution(df: pd.DataFrame) -> go.Figure:
    """Trending Velocity: Days from Publish to Trending."""
    vel_counts = (
        df[df["days_to_trend"] <= 14]["days_to_trend"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    vel_counts.columns = ["days_to_trend", "count"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=vel_counts["days_to_trend"],
            y=vel_counts["count"],
            marker=dict(color="#FF7043"),
            text=vel_counts["count"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="<b>Virality Velocity: Distribution of Days from Upload to Trending</b>",
        xaxis_title="Days Elapsed from Publish Date",
        yaxis_title="Total Video Snapshots",
        **DARK_LAYOUT,
    )
    return fig
