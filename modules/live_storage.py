"""SQLite persistence for live job postings.

Live jobs are stored separately from the historical LinkedIn CSV, so the
historical dataset is never modified. Jobs are deduplicated on ``job_id``
(the Arbeitnow slug, or a hash of the canonical URL when no slug exists).
Unchanged jobs are skipped; changed jobs are updated while keeping their
original ``first_fetched_at`` timestamp.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DB_PATH = Path("data") / "live_jobs.db"

LIVE_JOB_COLUMNS = [
    "job_id", "source", "job_title", "company", "location", "description",
    "description_text", "url", "published_at", "remote", "remote_label",
    "tags", "employment_types", "skills", "data_type", "first_fetched_at",
    "last_updated_at",
]

CONTENT_COLUMNS = [
    "job_title", "company", "location", "description_text", "url",
    "published_at", "remote", "tags", "employment_types", "skills",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect(db_path: str | Path | None = None):
    """Yield a SQLite connection and always release its Windows file handle."""
    connection = sqlite3.connect(str(resolve_db_path(db_path)))
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.close()


def init_db(db_path: str | Path | None = None) -> Path:
    """Create the live jobs schema if it does not exist yet."""
    path = resolve_db_path(db_path)
    with _connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS live_jobs (
                job_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                job_title TEXT NOT NULL,
                company TEXT,
                location TEXT,
                description TEXT,
                description_text TEXT,
                url TEXT,
                published_at TEXT,
                remote INTEGER DEFAULT 0,
                remote_label TEXT,
                tags TEXT,
                employment_types TEXT,
                skills TEXT,
                data_type TEXT DEFAULT 'live',
                first_fetched_at TEXT NOT NULL,
                last_updated_at TEXT NOT NULL,
                content_hash TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_live_jobs_company
                ON live_jobs(company);
            CREATE INDEX IF NOT EXISTS idx_live_jobs_published
                ON live_jobs(published_at);
            CREATE INDEX IF NOT EXISTS idx_live_jobs_remote
                ON live_jobs(remote);

            CREATE TABLE IF NOT EXISTS live_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS job_applications (
                application_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT, job_title TEXT NOT NULL, company TEXT,
                url TEXT, status TEXT NOT NULL, applied_at TEXT,
                notes TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            """
        )
    return path


def _row_content_hash(job: dict[str, Any]) -> str:
    payload = {column: job.get(column) for column in CONTENT_COLUMNS}
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _set_meta(connection: sqlite3.Connection, key: str, value: object) -> None:
    connection.execute(
        """
        INSERT INTO live_meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, "" if value is None else str(value)),
    )


def _get_meta(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM live_meta WHERE key = ?", (key,)
    ).fetchone()
    return None if row is None else row["value"]


def upsert_live_jobs(
    jobs: pd.DataFrame,
    db_path: str | Path | None = None,
) -> dict[str, int]:
    """Insert new live jobs and update changed ones.

    Returns counts for ``inserted``, ``updated``, ``skipped`` and ``total``.
    Duplicate (unchanged) jobs are not downloaded again into the database.
    """
    init_db(db_path)
    stats = {"inserted": 0, "updated": 0, "skipped": 0, "total": 0}

    if jobs is None or jobs.empty:
        with _connect(db_path) as connection:
            _set_meta(connection, "last_refresh_at", _now_iso())
            _set_meta(connection, "last_refresh_total", 0)
            _set_meta(connection, "last_refresh_inserted", 0)
            _set_meta(connection, "last_refresh_updated", 0)
            _set_meta(connection, "last_refresh_skipped", 0)
        return stats

    now = _now_iso()

    with _connect(db_path) as connection:
        for _, row in jobs.iterrows():
            job = row.to_dict()
            job_id = job.get("job_id")
            if not job_id:
                continue

            job["first_fetched_at"] = job.get("fetched_at") or now
            job["last_updated_at"] = now
            content_hash = _row_content_hash(job)
            stats["total"] += 1

            existing = connection.execute(
                "SELECT content_hash FROM live_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()

            if existing is None:
                connection.execute(
                    """
                    INSERT INTO live_jobs (
                        job_id, source, job_title, company, location,
                        description, description_text, url, published_at,
                        remote, remote_label, tags, employment_types, skills,
                        data_type, first_fetched_at, last_updated_at,
                        content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        job.get("source", "Arbeitnow"),
                        job.get("job_title", ""),
                        job.get("company", ""),
                        job.get("location", ""),
                        job.get("description", ""),
                        job.get("description_text", ""),
                        job.get("url", ""),
                        job.get("published_at"),
                        int(bool(job.get("remote", False))),
                        job.get("remote_label", ""),
                        job.get("tags", ""),
                        job.get("employment_types", ""),
                        job.get("skills", ""),
                        job.get("data_type", "live"),
                        job["first_fetched_at"],
                        now,
                        content_hash,
                    ),
                )
                stats["inserted"] += 1
            elif existing["content_hash"] == content_hash:
                connection.execute(
                    "UPDATE live_jobs SET last_updated_at = ? WHERE job_id = ?",
                    (now, job_id),
                )
                stats["skipped"] += 1
            else:
                connection.execute(
                    """
                    UPDATE live_jobs SET
                        source = ?, job_title = ?, company = ?, location = ?,
                        description = ?, description_text = ?, url = ?,
                        published_at = ?, remote = ?, remote_label = ?,
                        tags = ?, employment_types = ?, skills = ?,
                        data_type = ?, last_updated_at = ?, content_hash = ?
                    WHERE job_id = ?
                    """,
                    (
                        job.get("source", "Arbeitnow"),
                        job.get("job_title", ""),
                        job.get("company", ""),
                        job.get("location", ""),
                        job.get("description", ""),
                        job.get("description_text", ""),
                        job.get("url", ""),
                        job.get("published_at"),
                        int(bool(job.get("remote", False))),
                        job.get("remote_label", ""),
                        job.get("tags", ""),
                        job.get("employment_types", ""),
                        job.get("skills", ""),
                        job.get("data_type", "live"),
                        now,
                        content_hash,
                        job_id,
                    ),
                )
                stats["updated"] += 1

        _set_meta(connection, "last_refresh_at", now)
        _set_meta(connection, "last_refresh_total", stats["total"])
        _set_meta(connection, "last_refresh_inserted", stats["inserted"])
        _set_meta(connection, "last_refresh_updated", stats["updated"])
        _set_meta(connection, "last_refresh_skipped", stats["skipped"])
        _set_meta(connection, "last_error", "")

    return stats


def record_refresh_error(
    message: str,
    db_path: str | Path | None = None,
) -> None:
    """Remember the most recent fetch failure for the dashboard."""
    init_db(db_path)
    with _connect(db_path) as connection:
        _set_meta(connection, "last_error", message)
        _set_meta(connection, "last_error_at", _now_iso())


def load_live_jobs(
    db_path: str | Path | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load stored live jobs, newest first.

    Pass ``columns`` to load only the columns needed by a feature (for example,
    omitting the raw HTML ``description`` column on low-memory analytics pages).
    """
    init_db(db_path)
    if columns is not None:
        selected = [col for col in columns if col in LIVE_JOB_COLUMNS]
        if not selected:
            selected = list(LIVE_JOB_COLUMNS)
    else:
        selected = list(LIVE_JOB_COLUMNS)

    select_sql = ", ".join(selected)
    with _connect(db_path) as connection:
        frame = pd.read_sql_query(
            f"""
            SELECT {select_sql}
            FROM live_jobs
            ORDER BY (published_at IS NULL), published_at DESC, job_title ASC
            """,
            connection,
        )

    if frame.empty:
        return pd.DataFrame(columns=selected)

    if "remote" in frame.columns:
        frame["remote"] = frame["remote"].fillna(0).astype(bool)
    return frame


def get_live_jobs_status(db_path: str | Path | None = None) -> dict[str, Any]:
    """Return counts and the latest refresh metadata for the dashboard."""
    path = init_db(db_path)
    with _connect(path) as connection:
        count_row = connection.execute(
            "SELECT COUNT(*) AS total FROM live_jobs"
        ).fetchone()
        meta_rows = connection.execute(
            "SELECT key, value FROM live_meta"
        ).fetchall()

    meta = {row["key"]: row["value"] for row in meta_rows}
    return {
        "db_path": str(path),
        "total_jobs": int(count_row["total"]) if count_row else 0,
        "last_refresh_at": meta.get("last_refresh_at") or None,
        "last_refresh_total": int(meta.get("last_refresh_total") or 0),
        "last_refresh_inserted": int(meta.get("last_refresh_inserted") or 0),
        "last_refresh_updated": int(meta.get("last_refresh_updated") or 0),
        "last_refresh_skipped": int(meta.get("last_refresh_skipped") or 0),
        "last_error": meta.get("last_error") or None,
        "last_error_at": meta.get("last_error_at") or None,
    }


def add_application(application: dict[str, Any], db_path: str | Path | None = None) -> None:
    """Persist one user-managed job application."""
    init_db(db_path)
    now = _now_iso()
    with _connect(db_path) as connection:
        connection.execute(
            """INSERT INTO job_applications
            (job_id, job_title, company, url, status, applied_at, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (application.get("job_id"), application.get("job_title", ""),
             application.get("company", ""), application.get("url", ""),
             application.get("status", "Saved"), application.get("applied_at"),
             application.get("notes", ""), now, now),
        )


def load_applications(db_path: str | Path | None = None) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM job_applications ORDER BY updated_at DESC", connection
        )


def update_application_status(application_id: int, status: str, db_path: str | Path | None = None) -> None:
    init_db(db_path)
    with _connect(db_path) as connection:
        now = _now_iso()
        connection.execute(
            """
            UPDATE job_applications
            SET status = ?,
                updated_at = ?,
                applied_at = CASE
                    WHEN ? = 'Applied' AND applied_at IS NULL THEN ?
                    ELSE applied_at
                END
            WHERE application_id = ?
            """,
            (status, now, status, now, application_id),
        )
