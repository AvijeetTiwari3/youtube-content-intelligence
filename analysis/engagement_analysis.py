"""
Statistical Content & Engagement Analysis Module
Performs One-Way ANOVA, OLS Multiple Regression, and Category Benchmarking.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


ROOT_DIR = Path(__file__).resolve().parent.parent


def run_engagement_analysis(
    parquet_path: str = None,
    output_dir: str = None,
) -> dict:
    if parquet_path is None:
        parquet_path = ROOT_DIR / "data" / "processed" / "youtube_transformed.parquet"
    else:
        parquet_path = Path(parquet_path)

    if output_dir is None:
        output_dir = ROOT_DIR / "data" / "processed"
    else:
        output_dir = Path(output_dir)

    print("Running Statistical Content & Engagement Analysis...")
    df = pd.read_parquet(parquet_path)

    # 1. Category Benchmarking (Median vs Mean to handle viral skew)
    cat_benchmarks = (
        df.groupby("category_name")
        .agg(
            total_videos=("video_id", "count"),
            median_views=("views", "median"),
            mean_views=("views", "mean"),
            median_likes=("likes", "median"),
            median_comments=("comment_count", "median"),
            median_engagement_rate=("engagement_rate", "median"),
            mean_engagement_rate=("engagement_rate", "mean"),
            viral_ratio=("is_viral", "mean"),
        )
        .sort_values(by="median_engagement_rate", ascending=False)
        .reset_index()
    )

    # 2. One-Way ANOVA: Does engagement rate vary significantly across categories?
    categories_list = [group["engagement_rate"].values for _, group in df.groupby("category_name")]
    f_stat, p_val = stats.f_oneway(*categories_list)

    # 3. OLS Multiple Linear Regression on Log(Views)
    reg_df = df.copy()
    reg_df["log_views"] = np.log1p(reg_df["views"])
    reg_df["log_likes"] = np.log1p(reg_df["likes"])
    reg_df["log_comments"] = np.log1p(reg_df["comment_count"])

    X_cols = [
        "engagement_rate",
        "title_word_count",
        "title_sentiment_polarity",
        "tag_count",
        "days_to_trend",
        "title_caps_ratio",
    ]
    X = reg_df[X_cols].fillna(0)
    X = sm.add_constant(X)
    y = reg_df["log_views"]

    ols_model = sm.OLS(y, X).fit()

    ols_summary = {
        "r_squared": round(float(ols_model.rsquared), 4),
        "adj_r_squared": round(float(ols_model.rsquared_adj), 4),
        "f_pvalue": float(ols_model.f_pvalue),
        "coefficients": {k: round(float(v), 4) for k, v in ols_model.params.items()},
        "p_values": {k: round(float(v), 5) for k, v in ols_model.pvalues.items()},
    }

    # 4. Compile Results
    results = {
        "anova": {
            "f_statistic": round(float(f_stat), 3),
            "p_value": float(p_val),
            "is_statistically_significant": bool(p_val < 0.001),
        },
        "ols_regression": ols_summary,
        "category_rankings": cat_benchmarks.to_dict(orient="records"),
    }

    out_dir = Path(output_dir)
    with open(out_dir / "statistical_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cat_benchmarks.to_parquet(out_dir / "category_benchmarks.parquet", index=False)

    print("Statistical Analysis Complete!")
    print(f"ANOVA F-Statistic: {f_stat:.2f} (p-value: {p_val:.2e}) -> Statistically Significant: {p_val < 0.001}")
    print(f"OLS Regression R-squared: {ols_model.rsquared:.4f}")
    return results


if __name__ == "__main__":
    run_engagement_analysis()
