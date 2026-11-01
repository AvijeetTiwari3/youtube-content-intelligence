"""
Unit Tests for YouTube Data Pipeline & Database Warehouse
"""

import sqlite3
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_processed_files_exist():
    parquet_path = ROOT_DIR / "data" / "processed" / "youtube_transformed.parquet"
    db_path = ROOT_DIR / "database" / "youtube_trends.db"

    assert parquet_path.exists(), "youtube_transformed.parquet missing"
    assert db_path.exists(), "youtube_trends.db missing"


def test_dataset_integrity():
    parquet_path = ROOT_DIR / "data" / "processed" / "youtube_transformed.parquet"
    df = pd.read_parquet(parquet_path)

    assert len(df) >= 20000, f"Expected >= 20,000 records, got {len(df)}"
    assert df["views"].isnull().sum() == 0, "Null views found"
    assert df["likes"].isnull().sum() == 0, "Null likes found"
    assert (df["views"] > 0).all(), "Zero or negative views found"
    assert (df["days_to_trend"] >= 0).all(), "Negative days_to_trend found"


def test_database_records():
    db_path = ROOT_DIR / "database" / "youtube_trends.db"
    conn = sqlite3.connect(db_path)

    cat_count = pd.read_sql("SELECT COUNT(*) AS c FROM categories;", conn)["c"].iloc[0]
    chan_count = pd.read_sql("SELECT COUNT(*) AS c FROM channels;", conn)["c"].iloc[0]
    snap_count = pd.read_sql("SELECT COUNT(*) AS c FROM video_snapshots;", conn)["c"].iloc[0]
    conn.close()

    assert cat_count == 16, f"Expected 16 categories, got {cat_count}"
    assert chan_count >= 1500, f"Expected >= 1,500 channels, got {chan_count}"
    assert snap_count >= 20000, f"Expected >= 20,000 snapshots, got {snap_count}"


if __name__ == "__main__":
    pytest.main([__file__])
