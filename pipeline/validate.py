"""
Data Governance & Quality Assurance Suite
Executes 10 statistical and relational integrity assertions on YouTube data warehouse.
"""

import sqlite3
import sys
from pathlib import Path
import pandas as pd


class YouTubeDataValidator:
    def __init__(self, db_path: str = "youtube-content-intelligence/database/youtube_trends.db"):
        self.db_path = db_path
        self.passed = 0
        self.failures = []

    def assert_true(self, condition: bool, check_name: str, error_msg: str):
        if condition:
            self.passed += 1
            print(f"  [PASS] {check_name}")
        else:
            self.failures.append(f"{check_name}: {error_msg}")
            print(f"  [FAIL] {check_name}: {error_msg}")

    def run_all_checks(self):
        print(f"\n{'='*65}")
        print(f"Running Data Quality & Governance Assertions: {self.db_path}")
        print(f"{'='*65}")

        conn = sqlite3.connect(self.db_path)

        # 1. Total Record Completeness
        total_rows = pd.read_sql("SELECT COUNT(*) AS c FROM video_snapshots;", conn)["c"].iloc[0]
        self.assert_true(total_rows >= 20000, "1. Volume Check", f"Expected >= 20,000 records, found {total_rows:,}")

        # 2. Non-null Constraints on Core Fields
        null_df = pd.read_sql(
            """
            SELECT
                SUM(CASE WHEN video_id IS NULL THEN 1 ELSE 0 END) AS null_ids,
                SUM(CASE WHEN title IS NULL THEN 1 ELSE 0 END) AS null_titles,
                SUM(CASE WHEN views IS NULL THEN 1 ELSE 0 END) AS null_views,
                SUM(CASE WHEN likes IS NULL THEN 1 ELSE 0 END) AS null_likes
            FROM video_snapshots;
            """,
            conn,
        )
        total_nulls = int(null_df.sum(axis=1).iloc[0])
        self.assert_true(total_nulls == 0, "2. Non-Null Constraints", f"Found {total_nulls} null values in core fields")

        # 3. Positive View Count Domain
        invalid_views = pd.read_sql("SELECT COUNT(*) AS c FROM video_snapshots WHERE views < 1;", conn)["c"].iloc[0]
        self.assert_true(invalid_views == 0, "3. Positive Views Domain", f"Found {invalid_views} records with views < 1")

        # 4. Engagement Counts Non-Negative
        neg_counts = pd.read_sql(
            "SELECT COUNT(*) AS c FROM video_snapshots WHERE likes < 0 OR dislikes < 0 OR comment_count < 0;",
            conn,
        )["c"].iloc[0]
        self.assert_true(neg_counts == 0, "4. Non-Negative Engagements", f"Found {neg_counts} records with negative values")

        # 5. Engagement Rate Logical Bounds
        invalid_eng = pd.read_sql("SELECT COUNT(*) AS c FROM video_snapshots WHERE engagement_rate < 0.0;", conn)["c"].iloc[0]
        self.assert_true(invalid_eng == 0, "5. Engagement Rate >= 0.0%", f"Found {invalid_eng} negative rates")

        # 6. Temporal Consistency (Trending Date >= Publish Date)
        invalid_timing = pd.read_sql("SELECT COUNT(*) AS c FROM video_snapshots WHERE days_to_trend < 0;", conn)["c"].iloc[0]
        self.assert_true(invalid_timing == 0, "6. Trending Velocity >= 0 Days", f"Found {invalid_timing} negative timelines")

        # 7. Sentiment Polarity Bounds [-1.0, +1.0]
        invalid_sentiment = pd.read_sql(
            "SELECT COUNT(*) AS c FROM video_snapshots WHERE title_sentiment_polarity < -1.0 OR title_sentiment_polarity > 1.0;",
            conn,
        )["c"].iloc[0]
        self.assert_true(invalid_sentiment == 0, "7. Sentiment Polarity [-1, +1]", f"Found {invalid_sentiment} out of bounds")

        # 8. Title String Sanity
        invalid_titles = pd.read_sql(
            "SELECT COUNT(*) AS c FROM video_snapshots WHERE title_word_count < 1 OR title_char_length < 1;",
            conn,
        )["c"].iloc[0]
        self.assert_true(invalid_titles == 0, "8. Title Sanity Length Checks", f"Found {invalid_titles} empty title metrics")

        # 9. Binary Virality Target Variable Integrity
        invalid_viral = pd.read_sql("SELECT COUNT(*) AS c FROM video_snapshots WHERE is_viral NOT IN (0, 1);", conn)["c"].iloc[0]
        self.assert_true(invalid_viral == 0, "9. Binary Virality Target {0, 1}", f"Found {invalid_viral} invalid target values")

        # 10. Referential Integrity (Category FK)
        orphan_cats = pd.read_sql(
            """
            SELECT COUNT(*) AS c
            FROM video_snapshots v
            LEFT JOIN categories c ON v.category_id = c.category_id
            WHERE c.category_id IS NULL;
            """,
            conn,
        )["c"].iloc[0]
        self.assert_true(orphan_cats == 0, "10. Foreign Key Integrity (Categories)", f"Found {orphan_cats} orphan category references")

        conn.close()

        print(f"{'='*65}")
        print(f"Validation Summary: {self.passed} Passed, {len(self.failures)} Failed")
        print(f"{'='*65}\n")

        if self.failures:
            print("FAILED CHECKS:")
            for failure in self.failures:
                print(f" - {failure}")
            sys.exit(1)
        else:
            print("All 10 Enterprise Data Governance Checks PASSED successfully!")


if __name__ == "__main__":
    validator = YouTubeDataValidator()
    validator.run_all_checks()
