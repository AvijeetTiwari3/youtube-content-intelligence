"""
Trend Timing & Velocity Analytics Module
Quantifies upload hour vs. day-of-week engagement lift and trending velocity.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def analyze_trend_timing(
    parquet_path: str = "youtube-content-intelligence/data/processed/youtube_transformed.parquet",
    output_dir: str = "youtube-content-intelligence/data/processed",
) -> dict:
    print("Analyzing Trend Timing and Velocity Dynamics...")
    df = pd.read_parquet(parquet_path)

    # 1. Day of Week Order
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # 2. Publish Hour x Day of Week Engagement Rate Heatmap Pivot
    pivot_engagement = df.pivot_table(
        index="publish_day_name",
        columns="publish_hour",
        values="engagement_rate",
        aggfunc="median",
    ).reindex(day_order).round(2)

    # Publish Hour x Day of Week View Count Pivot
    pivot_views = df.pivot_table(
        index="publish_day_name",
        columns="publish_hour",
        values="views",
        aggfunc="median",
    ).reindex(day_order).round(0)

    # 3. Trending Velocity Distribution (Days from publish to trending)
    # Bucket into 0-1 days (Immediate Virality), 2-3 days (Steady Lift), 4+ days (Slow Burn)
    velocity_counts = df["days_to_trend"].value_counts().sort_index()
    velocity_summary = {
        "within_24h_pct": round(float((df["days_to_trend"] <= 1).mean() * 100), 2),
        "within_48h_pct": round(float((df["days_to_trend"] <= 2).mean() * 100), 2),
        "within_72h_pct": round(float((df["days_to_trend"] <= 3).mean() * 100), 2),
        "median_days_to_trend": float(df["days_to_trend"].median()),
        "mean_days_to_trend": round(float(df["days_to_trend"].mean()), 2),
    }

    # 4. Best Publishing Windows by Engagement
    top_window = (
        df.groupby(["publish_day_name", "publish_hour"])
        .agg(
            median_engagement=("engagement_rate", "median"),
            total_videos=("video_id", "count"),
            viral_rate=("is_viral", "mean"),
        )
        .reset_index()
        .sort_values(by="median_engagement", ascending=False)
        .head(5)
    )

    out_dir = Path(output_dir)
    pivot_engagement.to_parquet(out_dir / "timing_engagement_pivot.parquet")
    pivot_views.to_parquet(out_dir / "timing_views_pivot.parquet")

    results = {
        "velocity": velocity_summary,
        "optimal_windows": top_window.to_dict(orient="records"),
    }

    with open(out_dir / "timing_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Timing & Velocity Analytics Complete!")
    print(f"Median Days to Trend: {velocity_summary['median_days_to_trend']} days")
    print(f"Videos Trending within 48 Hours: {velocity_summary['within_48h_pct']}%")
    return results


if __name__ == "__main__":
    analyze_trend_timing()
