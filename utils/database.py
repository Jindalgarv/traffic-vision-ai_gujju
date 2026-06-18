"""SQLite database helper for TrafficVision AI evidence review."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


DB_PATH = "data/traffic_violations.db"
DEFAULT_REVIEW_STATUS = "Pending Review"


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection."""

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path), check_same_thread=False)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return existing column names for a table."""

    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {str(row[1]) for row in cursor.fetchall()}


def init_db(db_path: str | Path = DB_PATH) -> None:
    """Create or migrate the violations table."""

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            violation_id TEXT UNIQUE,
            timestamp TEXT NOT NULL,
            image_filename TEXT,
            vehicle_type TEXT,
            violation_type TEXT,
            confidence REAL,
            ocr_plate_number TEXT,
            license_plate TEXT,
            ocr_confidence REAL,
            ocr_engine TEXT,
            speed_mph INTEGER,
            details TEXT,
            original_image_path TEXT,
            annotated_image_path TEXT,
            evidence_path TEXT,
            review_status TEXT NOT NULL DEFAULT 'Pending Review'
        )
        """
    )

    # Lightweight migration for users who already ran older versions.
    columns = _table_columns(conn, "violations")
    migrations = {
        "violation_id": "ALTER TABLE violations ADD COLUMN violation_id TEXT",
        "image_filename": "ALTER TABLE violations ADD COLUMN image_filename TEXT",
        "ocr_plate_number": "ALTER TABLE violations ADD COLUMN ocr_plate_number TEXT",
        "ocr_confidence": "ALTER TABLE violations ADD COLUMN ocr_confidence REAL",
        "ocr_engine": "ALTER TABLE violations ADD COLUMN ocr_engine TEXT",
        "annotated_image_path": "ALTER TABLE violations ADD COLUMN annotated_image_path TEXT",
        "review_status": "ALTER TABLE violations ADD COLUMN review_status TEXT NOT NULL DEFAULT 'Pending Review'",
    }
    for column_name, sql in migrations.items():
        if column_name not in columns:
            cursor.execute(sql)

    # Backfill old rows as much as possible.
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
    details: str = "",
    ocr_confidence: Optional[float] = None,
    ocr_engine: str = "",
    db_path: str | Path = DB_PATH,
) -> int:
    """Insert one violation row and return the new database primary key."""

    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat(timespec="seconds")
    plate = ocr_plate_number or "N/A"

    cursor.execute(
        """
        INSERT INTO violations (
            violation_id,
            timestamp,
            image_filename,
            vehicle_type,
            violation_type,
            confidence,
            ocr_plate_number,
            license_plate,
            ocr_confidence,
            ocr_engine,
            speed_mph,
            details,
            original_image_path,
            annotated_image_path,
            evidence_path,
            review_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            violation_id,
            timestamp,
            image_filename,
            vehicle_type,
            violation_type,
            float(confidence),
            plate,
            plate,  # legacy alias used by older UI code
            ocr_confidence,
            ocr_engine,
            speed_mph,
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


def update_evidence_path(
    violation_id: int,
    evidence_path: str,
    db_path: str | Path = DB_PATH,
) -> None:
    """Update crop/evidence path for a violation database primary key."""

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
    """Update human review status for a saved violation."""

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


def get_all_violations(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Return all violation rows as a pandas DataFrame."""

    init_db(db_path)
    conn = get_connection(db_path)
    dataframe = pd.read_sql_query(
        """
        SELECT
            id,
            violation_id,
            timestamp,
            image_filename,
            vehicle_type,
            violation_type,
            confidence,
            ocr_plate_number,
            ocr_confidence,
            ocr_engine,
            speed_mph,
            details,
            original_image_path,
            annotated_image_path,
            evidence_path,
            review_status
        FROM violations
        ORDER BY id DESC
        """,
        conn,
    )
    conn.close()
    return dataframe


def delete_all_violations(db_path: str | Path = DB_PATH) -> None:
    """Delete all violation records from the database."""

    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM violations")
    conn.commit()
    conn.close()


def count_violations(db_path: str | Path = DB_PATH) -> int:
    """Return the number of saved violations."""

    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM violations")
    count = int(cursor.fetchone()[0])
    conn.close()
    return count
