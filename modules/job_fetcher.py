"""Fetch live job postings from the Arbeitnow job board API.

Arbeitnow exposes a free, public JSON API (no API key required):
    https://www.arbeitnow.com/api/job-board-api

The API returns 300 jobs per page and supports pagination with ``?page=N``.
Jobs are ordered by ``created_at`` (newest first) and refreshed roughly hourly
by Arbeitnow.

This module only handles downloading and light validation. Cleaning and
normalization happen in :mod:`modules.live_preprocessing`.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

API_URL = "https://www.arbeitnow.com/api/job-board-api"
DEFAULT_TIMEOUT = 15
DEFAULT_PAGES = 1
MAX_PAGES = 20
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5


class JobFetchError(Exception):
    """Raised when live jobs cannot be fetched from Arbeitnow."""


def _empty_jobs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "slug",
            "company_name",
            "title",
            "description",
            "remote",
            "url",
            "tags",
            "job_types",
            "location",
            "created_at",
        ]
    )


def _request_page(
    session: requests.Session,
    page: int,
    timeout: int,
) -> dict[str, Any]:
    """Request one page and return the decoded payload."""
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = session.get(
                API_URL,
                params={"page": page},
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "CareerLens-JMIS/1.0 (educational project)",
                },
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # 4xx errors will not fix themselves, do not waste retries.
            if status is not None and 400 <= status < 500 and status != 429:
                raise JobFetchError(
                    f"Arbeitnow API returned HTTP {status}."
                ) from exc
            last_error = exc
        except requests.exceptions.RequestException as exc:
            last_error = exc
        else:
            try:
                payload = response.json()
            except ValueError as exc:
                raise JobFetchError(
                    "Arbeitnow API returned invalid JSON."
                ) from exc

            if not isinstance(payload, dict):
                raise JobFetchError(
                    "Unexpected Arbeitnow response: expected a JSON object."
                )
            if "data" not in payload:
                raise JobFetchError(
                    "Unexpected Arbeitnow response: 'data' field is missing."
                )
            if not isinstance(payload["data"], list):
                raise JobFetchError(
                    "Unexpected Arbeitnow response: 'data' is not a list."
                )
            return payload

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    raise JobFetchError(f"Could not reach Arbeitnow API: {last_error}")


def fetch_arbeitnow_jobs(
    pages: int = DEFAULT_PAGES,
    timeout: int = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Download live jobs from Arbeitnow.

    Parameters
    ----------
    pages:
        Number of API pages to download (1-20). Each page holds up to 300 jobs.
    timeout:
        Per-request timeout in seconds.
    session:
        Optional pre-configured ``requests.Session`` (used by tests).

    Returns
    -------
    dict
        ``{"jobs": DataFrame, "pages_fetched": int, "total_received": int,
        "duplicates_removed": int}``. The DataFrame keeps the raw API fields;
        normalization is done separately.

    Raises
    ------
    JobFetchError
        On network errors, HTTP errors, invalid JSON, or unexpected payloads.
    """
    if not isinstance(pages, int) or pages < 1:
        raise ValueError("pages must be a positive integer.")
    pages = min(pages, MAX_PAGES)

    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout must be a positive number.")

    owns_session = session is None
    http = session or requests.Session()
    records: list[dict[str, Any]] = []
    pages_fetched = 0
    next_page = 1

    try:
        for _ in range(pages):
            payload = _request_page(http, next_page, timeout)
            page_rows = [row for row in payload["data"] if isinstance(row, dict)]
            records.extend(page_rows)
            pages_fetched += 1

            links = payload.get("links") or {}
            if not links.get("next") or not page_rows:
                break

            meta = payload.get("meta") or {}
            current = meta.get("current_page")
            next_page = (current + 1) if isinstance(current, int) else next_page + 1
    finally:
        if owns_session:
            http.close()

    if not records:
        return {
            "jobs": _empty_jobs_frame(),
            "pages_fetched": pages_fetched,
            "total_received": 0,
            "duplicates_removed": 0,
        }

    raw_df = pd.DataFrame(records)

    # Drop duplicate jobs inside the same download. A slug is the most stable
    # identifier; fall back to the URL when it is missing.
    before = len(raw_df)
    dedupe_columns = [
        col for col in ("slug", "url") if col in raw_df.columns
    ]
    if dedupe_columns:
        subset = dedupe_columns[0]
        raw_df = raw_df.drop_duplicates(subset=subset, keep="first")
    else:
        raw_df = raw_df.drop_duplicates()

    return {
        "jobs": raw_df.reset_index(drop=True),
        "pages_fetched": pages_fetched,
        "total_received": before,
        "duplicates_removed": before - len(raw_df),
    }
