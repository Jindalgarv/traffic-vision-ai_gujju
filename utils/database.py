"""SQLite database helper for IntelliTraffic AI evidence review.

Schema (current)
─────────────────
violations
    id                  INTEGER PK AUTOINCREMENT
    violation_id        TEXT UNIQUE          -- human-readable ID e.g. TVAI-20260618-…
    timestamp           TEXT NOT NULL        -- ISO 8601
    image_filename      TEXT
    image_hash          TEXT                 -- SHA-256 of original image bytes (tamper-proof)
    location_name       TEXT                 -- e.g. "Hazratganj Junction"
    gps_coordinates     TEXT                 -- e.g. "26.8467° N, 80.9462° E"
    vehicle_type        TEXT
    violation_type      TEXT
    violations_json     TEXT                 -- JSON list of all violation types on this vehicle
    confidence          REAL
    ocr_plate_number    TEXT
    ocr_confidence      REAL
    ocr_engine          TEXT
    speed_mph           INTEGER
    severity            INTEGER              -- 1-10 severity score
    details             TEXT
    original_image_path TEXT
    annotated_image_path TEXT
    evidence_path       TEXT
    review_status       TEXT NOT NULL DEFAULT 'Pending Review'
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd


DB_PATH = "data/traffic_violations.db"
DEFAULT_REVIEW_STATUS = "Pending Review"


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path), check_same_thread=False)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {str(row[1]) for row in cursor.fetchall()}


# ---------------------------------------------------------------------------
# Init / migration
# ---------------------------------------------------------------------------

def init_db(db_path: str | Path = DB_PATH) -> None:
    """Create or migrate the violations table."""

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            violation_id         TEXT UNIQUE,
            timestamp            TEXT NOT NULL,
            image_filename       TEXT,
            image_hash           TEXT,
            location_name        TEXT,
            gps_coordinates      TEXT,
            vehicle_type         TEXT,
            violation_type       TEXT,
            violations_json      TEXT,
            confidence           REAL,
            ocr_plate_number     TEXT,
            license_plate        TEXT,
            ocr_confidence       REAL,
            ocr_engine           TEXT,
            speed_mph            INTEGER,
            severity             INTEGER,
            details              TEXT,
            original_image_path  TEXT,
            annotated_image_path TEXT,
            evidence_path        TEXT,
            review_status        TEXT NOT NULL DEFAULT 'Pending Review'
        )
        """
    )

    # Lightweight migration for existing databases.
    columns = _table_columns(conn, "violations")
    migrations = {
        "violation_id":         "ALTER TABLE violations ADD COLUMN violation_id TEXT",
        "image_filename":       "ALTER TABLE violations ADD COLUMN image_filename TEXT",
        "image_hash":           "ALTER TABLE violations ADD COLUMN image_hash TEXT",
        "location_name":        "ALTER TABLE violations ADD COLUMN location_name TEXT",
        "gps_coordinates":      "ALTER TABLE violations ADD COLUMN gps_coordinates TEXT",
        "violations_json":      "ALTER TABLE violations ADD COLUMN violations_json TEXT",
        "ocr_plate_number":     "ALTER TABLE violations ADD COLUMN ocr_plate_number TEXT",
        "ocr_confidence":       "ALTER TABLE violations ADD COLUMN ocr_confidence REAL",
        "ocr_engine":           "ALTER TABLE violations ADD COLUMN ocr_engine TEXT",
        "severity":             "ALTER TABLE violations ADD COLUMN severity INTEGER",
        "annotated_image_path": "ALTER TABLE violations ADD COLUMN annotated_image_path TEXT",
        "review_status": (
            "ALTER TABLE violations ADD COLUMN review_status TEXT "
            "NOT NULL DEFAULT 'Pending Review'"
        ),
    }
    for col, sql in migrations.items():
        if col not in columns:
            cursor.execute(sql)

    # Backfill defaults.
    cursor.execute(
        """
        UPDATE violations
        SET review_status = COALESCE(NULLIF(review_status, ''), 'Pending Review')
        WHERE review_status IS NULL OR review_status = ''
        """
    )
    cursor.execute(
        """
        UPDATE violations
        SET ocr_plate_number = COALESCE(NULLIF(ocr_plate_number, ''), license_plate)
        WHERE ocr_plate_number IS NULL OR ocr_plate_number = ''
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

def insert_violation(
    violation_id: str,
    image_filename: str,
    vehicle_type: str,
    violation_type: str,
    confidence: float,
    original_image_path: str,
    annotated_image_path: str,
    ocr_plate_number: str = "N/A",
    evidence_path: str = "",
    review_status: str = DEFAULT_REVIEW_STATUS,
    speed_mph: Optional[int] = None,
    severity: Optional[int] = None,
    details: str = "",
    ocr_confidence: Optional[float] = None,
    ocr_engine: str = "",
    image_hash: str = "",
    location_name: str = "",
    gps_coordinates: str = "",
    violations_json: Optional[List[str]] = None,
    db_path: str | Path = DB_PATH,
) -> int:
    """Insert one violation row and return the new database primary key."""

    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat(timespec="seconds")
    plate = ocr_plate_number or "N/A"
    viol_json = json.dumps(violations_json or [violation_type])

    cursor.execute(
        """
        INSERT INTO violations (
            violation_id,
            timestamp,
            image_filename,
            image_hash,
            location_name,
            gps_coordinates,
            vehicle_type,
            violation_type,
            violations_json,
            confidence,
            ocr_plate_number,
            license_plate,
            ocr_confidence,
            ocr_engine,
            speed_mph,
            severity,
            details,
            original_image_path,
            annotated_image_path,
            evidence_path,
            review_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            violation_id,
            timestamp,
            image_filename,
            image_hash,
            location_name,
            gps_coordinates,
            vehicle_type,
            violation_type,
            viol_json,
            float(confidence),
            plate,
            plate,
            ocr_confidence,
            ocr_engine,
            speed_mph,
            severity,
            details,
            original_image_path,
            annotated_image_path,
            evidence_path,
            review_status,
        ),
    )
    conn.commit()
    new_id = int(cursor.lastrowid)
    conn.close()
    return new_id


# ---------------------------------------------------------------------------
# Update helpers
# ---------------------------------------------------------------------------

def update_evidence_path(
    violation_id: int,
    evidence_path: str,
    db_path: str | Path = DB_PATH,
) -> None:
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE violations SET evidence_path = ? WHERE id = ?",
        (evidence_path, violation_id),
    )
    conn.commit()
    conn.close()


def update_review_status(
    row_id: int,
    review_status: str,
    db_path: str | Path = DB_PATH,
) -> None:
    allowed_statuses = {"Pending Review", "Approved", "Rejected", "Needs Manual Check"}
    if review_status not in allowed_statuses:
        raise ValueError(f"Invalid review status: {review_status}")
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE violations SET review_status = ? WHERE id = ?",
        (review_status, int(row_id)),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_all_violations(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Return all violation rows as a pandas DataFrame (newest first)."""
    init_db(db_path)
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            id, violation_id, timestamp, image_filename,
            image_hash, location_name, gps_coordinates,
            vehicle_type, violation_type, violations_json,
            confidence, ocr_plate_number, ocr_confidence, ocr_engine,
            speed_mph, severity, details,
            original_image_path, annotated_image_path, evidence_path,
            review_status
        FROM violations
        ORDER BY id DESC
        """,
        conn,
    )
    conn.close()
    return df


def get_repeat_offenders(
    db_path: str | Path = DB_PATH,
    min_violations: int = 1,
) -> pd.DataFrame:
    """Return plates with violation counts, sorted descending.

    Args:
        min_violations: Only include plates with at least this many records.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            ocr_plate_number                    AS license_plate,
            COUNT(*)                            AS total_violations,
            MAX(timestamp)                      AS last_seen,
            GROUP_CONCAT(DISTINCT violation_type) AS violation_types,
            ROUND(AVG(COALESCE(severity, 5)), 1) AS avg_severity,
            ROUND(
                COUNT(*) * 10 + AVG(COALESCE(severity, 5)),
                1
            )                                   AS risk_score
        FROM violations
        WHERE ocr_plate_number IS NOT NULL
          AND ocr_plate_number != 'N/A'
          AND ocr_plate_number != ''
        GROUP BY ocr_plate_number
        HAVING COUNT(*) >= ?
        ORDER BY risk_score DESC
        """,
        conn,
        params=(min_violations,),
    )
    conn.close()
    return df


def get_violation_stats(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Return daily counts grouped by violation type for analytics charts."""
    init_db(db_path)
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            DATE(timestamp)  AS date,
            violation_type,
            COUNT(*)         AS count
        FROM violations
        GROUP BY DATE(timestamp), violation_type
        ORDER BY date DESC, count DESC
        """,
        conn,
    )
    conn.close()
    return df


def get_location_stats(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Return violation counts grouped by location."""
    init_db(db_path)
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            COALESCE(NULLIF(location_name, ''), 'Unknown') AS location,
            COUNT(*)                                        AS total_violations,
            COUNT(DISTINCT ocr_plate_number)                AS unique_vehicles
        FROM violations
        GROUP BY location_name
        ORDER BY total_violations DESC
        """,
        conn,
    )
    conn.close()
    return df


def delete_all_violations(db_path: str | Path = DB_PATH) -> None:
    """Delete all violation records from the database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM violations")
    conn.commit()
    conn.close()


def count_violations(db_path: str | Path = DB_PATH) -> int:
    """Return the total number of saved violations."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM violations")
    count = int(cursor.fetchone()[0])
    conn.close()
    return count
