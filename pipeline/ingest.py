"""
YouTube Database Ingestion Pipeline
Loads normalized video entities and performance snapshots into SQLite.
"""

import sqlite3
from pathlib import Path
import pandas as pd


def get_db_connection(db_path: str = "youtube-content-intelligence/database/youtube_trends.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str, schema_path: str):
    schema_sql = Path(schema_path).read_text(encoding="utf-8")
    with get_db_connection(db_path) as conn:
        conn.executescript(schema_sql)
    print(f"Initialized SQLite database at {db_path}")


def ingest_data(parquet_path: str, db_path: str, schema_path: str):
    init_db(db_path, schema_path)

    print(f"Loading transformed data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # 1. Categories Dimension
    categories = (
        df[["category_id", "category_name"]]
        .drop_duplicates(subset=["category_id"])
        .sort_values(by="category_id")
        .reset_index(drop=True)
    )

    # 2. Channels Dimension
    channels = (
        df[["channel_title"]]
        .drop_duplicates(subset=["channel_title"])
        .assign(channel_id=lambda x: "CH_" + x.index.astype(str).str.zfill(5))
        .reset_index(drop=True)
    )

    # 3. Video Snapshots Fact Records
    fact_cols = [
        "video_id",
        "trending_date_str",
        "title",
        "channel_title",
        "category_id",
        "category_name",
        "publish_time",
        "publish_date",
        "publish_hour",
        "publish_day_name",
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "comments_disabled",
        "ratings_disabled",
        "video_error_or_removed",
        "days_to_trend",
        "engagement_rate",
        "like_to_view_ratio",
        "comment_to_view_ratio",
        "title_word_count",
        "title_char_length",
        "title_sentiment_polarity",
        "tag_count",
        "is_viral",
    ]
    fact_df = df[fact_cols].rename(columns={"trending_date_str": "trending_date"}).copy()
    fact_df["publish_time"] = fact_df["publish_time"].astype(str)

    with get_db_connection(db_path) as conn:
        print("Inserting Categories Dimension...")
        categories.to_sql("categories", conn, if_exists="append", index=False)

        print("Inserting Channels Dimension...")
        channels.to_sql("channels", conn, if_exists="append", index=False)

        print("Inserting Video Snapshots...")
        fact_df.to_sql("video_snapshots", conn, if_exists="append", index=False)

    print("Database Ingestion Complete!")
    print(f"Loaded {len(categories)} categories, {len(channels):,} unique channels, {len(fact_df):,} video snapshots.")


if __name__ == "__main__":
    ingest_data(
        parquet_path="youtube-content-intelligence/data/processed/youtube_transformed.parquet",
        db_path="youtube-content-intelligence/database/youtube_trends.db",
        schema_path="youtube-content-intelligence/database/schema.sql",
    )
