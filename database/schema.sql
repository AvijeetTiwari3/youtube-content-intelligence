-- Schema for YouTube Content Intelligence & Virality Prediction Platform
-- Relational SQLite Warehouse

DROP TABLE IF EXISTS video_snapshots;
DROP TABLE IF EXISTS channels;
DROP TABLE IF EXISTS categories;

CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS channels (
    channel_id TEXT PRIMARY KEY,
    channel_title TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS video_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    trending_date TEXT NOT NULL,
    title TEXT NOT NULL,
    channel_title TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    category_name TEXT NOT NULL,
    publish_time TEXT NOT NULL,
    publish_date TEXT NOT NULL,
    publish_hour INTEGER NOT NULL,
    publish_day_name TEXT NOT NULL,
    tags TEXT,
    views INTEGER NOT NULL,
    likes INTEGER NOT NULL,
    dislikes INTEGER NOT NULL,
    comment_count INTEGER NOT NULL,
    comments_disabled INTEGER NOT NULL,
    ratings_disabled INTEGER NOT NULL,
    video_error_or_removed INTEGER NOT NULL,
    days_to_trend INTEGER NOT NULL,
    engagement_rate REAL NOT NULL,
    like_to_view_ratio REAL NOT NULL,
    comment_to_view_ratio REAL NOT NULL,
    title_word_count INTEGER NOT NULL,
    title_char_length INTEGER NOT NULL,
    title_sentiment_polarity REAL NOT NULL,
    tag_count INTEGER NOT NULL,
    is_viral INTEGER NOT NULL,
    FOREIGN KEY(category_id) REFERENCES categories(category_id)
);

-- Performance Indexes for fast analytical slicing
CREATE INDEX IF NOT EXISTS idx_videos_category ON video_snapshots(category_id);
CREATE INDEX IF NOT EXISTS idx_videos_channel ON video_snapshots(channel_title);
CREATE INDEX IF NOT EXISTS idx_videos_publish_date ON video_snapshots(publish_date);
CREATE INDEX IF NOT EXISTS idx_videos_views ON video_snapshots(views);
CREATE INDEX IF NOT EXISTS idx_videos_is_viral ON video_snapshots(is_viral);
