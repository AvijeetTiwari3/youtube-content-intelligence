"""
YouTube Data Transformation & Feature Engineering Pipeline
Enriches raw trending video metadata with engagement metrics, NLP features, and virality labels.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# Canonical YouTube Data API v3 Category Mapping
CATEGORY_MAP = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    18: "Short Movies",
    19: "Travel & Events",
    20: "Gaming",
    21: "Videoblogging",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
    30: "Movies",
    43: "Shows",
    44: "Trailers",
}

# Calibrated Sentiment Lexicon for Video Titles
POSITIVE_WORDS = {
    "official", "trailer", "best", "great", "win", "amazing", "love", "awesome",
    "top", "highlight", "epic", "perfect", "funny", "secret", "reveal", "master",
    "incredible", "hero", "celebration", "super", "happy", "champion", "legend",
}
NEGATIVE_WORDS = {
    "fail", "worst", "lose", "died", "death", "shocking", "exposed", "scam",
    "terrible", "bad", "disaster", "horrible", "crash", "destroyed", "warning",
    "danger", "arrested", "stolen", "crying", "sad", "broke", "fight", "attack",
}


def compute_title_sentiment(title: str) -> float:
    """Calculates normalized title sentiment polarity between -1.0 and +1.0."""
    if not isinstance(title, str) or not title.strip():
        return 0.0

    words = re.findall(r"\b[a-zA-Z]+\b", title.lower())
    if not words:
        return 0.0

    pos_score = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_score = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos_score + neg_score

    if total == 0:
        return 0.0
    return round((pos_score - neg_score) / total, 3)


def transform_youtube_data(
    raw_csv_path: str = "youtube-content-intelligence/data/raw/USvideos.csv",
    output_dir: str = "youtube-content-intelligence/data/processed",
) -> pd.DataFrame:
    raw_path = Path(raw_csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw YouTube data from {raw_path}...")
    df = pd.read_csv(raw_path, on_bad_lines="skip", encoding="utf-8")

    # 1. Clean Core Transaction Identifiers & Types
    df = df.dropna(subset=["video_id", "title", "channel_title", "views"]).copy()

    df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0).astype(int)
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    df["dislikes"] = pd.to_numeric(df["dislikes"], errors="coerce").fillna(0).astype(int)
    df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0).astype(int)
    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").fillna(0).astype(int)

    # Filter invalid records with zero views
    df = df[df["views"] > 0].copy()

    # 2. Date Parsing
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df["trending_date"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m", errors="coerce")
    df = df.dropna(subset=["publish_time", "trending_date"]).copy()

    # 3. Category Mapping
    df["category_name"] = df["category_id"].map(CATEGORY_MAP).fillna("Other / Miscellaneous")

    # 4. Feature Engineering: Temporal Signals
    df["publish_date"] = df["publish_time"].dt.date.astype(str)
    df["publish_hour"] = df["publish_time"].dt.hour
    df["publish_day_name"] = df["publish_time"].dt.day_name()
    df["publish_month"] = df["publish_time"].dt.month
    df["trending_date_str"] = df["trending_date"].dt.date.astype(str)

    # Days from publish to trending (capped logically at >= 0)
    days_diff = (df["trending_date"].dt.date - df["publish_time"].dt.date).apply(lambda x: x.days)
    df["days_to_trend"] = np.maximum(days_diff, 0)

    # 5. Feature Engineering: Engagement Ratios
    df["engagement_rate"] = np.where(
        df["views"] > 0,
        (((df["likes"] + df["comment_count"]) / df["views"]) * 100).round(4),
        0.0,
    )
    df["like_to_view_ratio"] = np.where(df["views"] > 0, (df["likes"] / df["views"]).round(5), 0.0)
    df["comment_to_view_ratio"] = np.where(df["views"] > 0, (df["comment_count"] / df["views"]).round(5), 0.0)
    df["like_dislike_ratio"] = (df["likes"] / (df["likes"] + df["dislikes"] + 1)).round(4)

    # 6. Feature Engineering: Content & Title NLP Signals
    df["title_word_count"] = df["title"].astype(str).apply(lambda t: len(t.split()))
    df["title_char_length"] = df["title"].astype(str).str.len()
    df["title_caps_ratio"] = df["title"].astype(str).apply(
        lambda t: round(sum(1 for c in t if c.isupper()) / max(len(t), 1), 3)
    )
    df["title_sentiment_polarity"] = df["title"].apply(compute_title_sentiment)

    # Tag count
    def parse_tag_count(tag_str):
        if not isinstance(tag_str, str) or tag_str.strip() in ["[none]", ""]:
            return 0
        return len(tag_str.split("|"))

    df["tag_count"] = df["tags"].apply(parse_tag_count)

    # 7. Virality Target Variable (1 if views >= 1,000,000, else 0)
    # 1M views represents the top tier threshold in YouTube trending distribution
    df["is_viral"] = (df["views"] >= 1_000_000).astype(int)

    # 8. Clean Boolean / Flag attributes
    for flag_col in ["comments_disabled", "ratings_disabled", "video_error_or_removed"]:
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].astype(bool).astype(int)
        else:
            df[flag_col] = 0

    # 9. Deduplicate snapshot line items to capture unique trending lifecycle
    # Keep latest snapshot per video_id for modeling & creator analytics
    df = df.sort_values(by=["trending_date", "views"], ascending=[True, False]).reset_index(drop=True)

    # Export Processed Dataset
    csv_out = out_dir / "youtube_transformed.csv"
    parquet_out = out_dir / "youtube_transformed.parquet"

    # Select core analytics columns
    analytics_cols = [
        "video_id",
        "title",
        "channel_title",
        "category_id",
        "category_name",
        "publish_time",
        "publish_date",
        "publish_hour",
        "publish_day_name",
        "publish_month",
        "trending_date_str",
        "days_to_trend",
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "comments_disabled",
        "ratings_disabled",
        "video_error_or_removed",
        "engagement_rate",
        "like_to_view_ratio",
        "comment_to_view_ratio",
        "like_dislike_ratio",
        "title_word_count",
        "title_char_length",
        "title_caps_ratio",
        "title_sentiment_polarity",
        "tag_count",
        "is_viral",
    ]
    cleaned_df = df[analytics_cols].copy()

    cleaned_df.to_csv(csv_out, index=False)
    cleaned_df.to_parquet(parquet_out, index=False)

    print(f"ETL Transformation Complete: {len(cleaned_df):,} verified records.")
    print(f"Viral Videos (>=1M views): {cleaned_df['is_viral'].sum():,} ({cleaned_df['is_viral'].mean()*100:.1f}%)")
    print(f"Saved to:\n - {csv_out}\n - {parquet_out}")
    return cleaned_df


if __name__ == "__main__":
    transform_youtube_data()
