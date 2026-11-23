"""Normalize live Arbeitnow jobs into the CareerLens job schema.

The historical LinkedIn dataset and the Arbeitnow API do not share the same
columns. This module is the mapping layer between the two: it never modifies
the historical CSV and it never invents values (salary, seniority, etc.) that
the API does not provide.

Canonical live schema
---------------------
``job_id, job_title, company, location, description, description_text, url,
published_at, remote, remote_label, tags, employment_types, skills, source,
data_type, fetched_at, updated_at``
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from datetime import datetime, timezone

import pandas as pd

SOURCE_NAME = "Arbeitnow"
DATA_TYPE = "live"

# A small, transparent skill vocabulary. It is intentionally conservative:
# only skills that can be matched as whole words are reported. The raw tags
# and the original description are always preserved for context.
SKILL_KEYWORDS = [
    # Data / analytics
    "Python", "SQL", "Excel", "Pandas", "NumPy", "Statistics",
    "Machine Learning", "Deep Learning", "Data Analysis", "Data Science",
    "Power BI", "Tableau", "ETL", "Data Warehouse", "NLP",
    # Software engineering
    "Java", "JavaScript", "TypeScript", "C++", "C#", ".NET", "PHP", "Ruby",
    "Go", "Rust", "React", "Angular", "Vue", "Node.js", "Django", "Flask",
    "Spring", "REST", "GraphQL", "Git", "Docker", "Kubernetes", "AWS",
    "Azure", "Google Cloud", "Linux", "Terraform", "CI/CD", "DevOps",
    # Product / design / business
    "Figma", "UX", "UI", "Product Management", "Project Management",
    "Agile", "Scrum", "Jira", "Confluence", "Salesforce", "SAP",
    "SEO", "Marketing", "Sales", "Recruiting", "HR", "Accounting",
    "Finance", "Controlling", "Customer Service", "Communication",
]

GENDER_MARKER_PATTERN = re.compile(
    r"\(\s*(?:m|w|f|d|gn|all genders|any gender|divers)[^)]*\)",
    flags=re.IGNORECASE,
)
TAG_SPLIT_PATTERN = re.compile(r"\s*[|,;]\s*")
WHITESPACE_PATTERN = re.compile(r"\s+")
SCRIPT_STYLE_PATTERN = re.compile(
    r"<(script|style)[^>]*>.*?</\1>", flags=re.IGNORECASE | re.DOTALL
)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+")


def clean_html(raw_html: object) -> str:
    """Convert an HTML job description into readable plain text."""
    if raw_html is None or (isinstance(raw_html, float) and pd.isna(raw_html)):
        return ""

    text = str(raw_html)
    text = SCRIPT_STYLE_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def normalize_title(title: object) -> str:
    """Strip gender markers and normalize whitespace in a job title."""
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return ""

    text = unicodedata.normalize("NFKC", str(title))
    text = GENDER_MARKER_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip(" -–—|")


def normalize_location(location: object) -> str:
    if location is None or (isinstance(location, float) and pd.isna(location)):
        return "Not specified"
    text = WHITESPACE_PATTERN.sub(" ", str(location)).strip()
    return text or "Not specified"


def _split_values(value: object) -> list[str]:
    """Split list/string fields such as tags and job_types safely."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        items = TAG_SPLIT_PATTERN.split(str(value))

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = WHITESPACE_PATTERN.sub(" ", item).strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            cleaned.append(item)
    return cleaned


def _join(values: list[str]) -> str:
    return " | ".join(values)


def parse_created_at(value: object) -> str | None:
    """Parse Arbeitnow's ``created_at`` (Unix seconds) into an ISO string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, (int, float)):
        timestamp = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            timestamp = float(text)
        except ValueError:
            try:
                parsed = pd.to_datetime(text, errors="coerce", utc=True)
            except (ValueError, TypeError):
                return None
            if parsed is pd.NaT or pd.isna(parsed):
                return None
            return parsed.isoformat()

    try:
        return datetime.fromtimestamp(
            timestamp, tz=timezone.utc
        ).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def extract_skills(*texts: str, limit: int = 15) -> list[str]:
    """Find known skills mentioned in the supplied text fields."""
    haystack = " ".join(text for text in texts if text)
    if not haystack:
        return []

    # Normalize separators so "Node.js" and "nodejs" style variants work.
    haystack_lower = haystack.lower()
    found: list[str] = []

    for skill in SKILL_KEYWORDS:
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(skill.lower())
            + r"(?![a-z0-9])"
        )
        if re.search(pattern, haystack_lower):
            found.append(skill)
        if len(found) >= limit:
            break

    return found


def _stable_job_id(slug: object, url: object, title: object, company: object) -> str:
    slug_text = "" if slug is None or (isinstance(slug, float) and pd.isna(slug)) else str(slug).strip()
    if slug_text:
        return slug_text

    url_text = "" if url is None or (isinstance(url, float) and pd.isna(url)) else str(url).strip()
    if url_text:
        return hashlib.sha1(url_text.encode("utf-8")).hexdigest()

    fallback = f"{title}|{company}".lower()
    return hashlib.sha1(fallback.encode("utf-8")).hexdigest()


def normalize_live_jobs(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Map a raw Arbeitnow DataFrame to the canonical live job schema."""
    columns = [
        "job_id", "job_title", "company", "location", "description",
        "description_text", "url", "published_at", "remote", "remote_label",
        "tags", "employment_types", "skills", "source", "data_type",
        "fetched_at", "updated_at",
    ]

    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=columns)

    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []

    for _, record in raw_df.iterrows():
        title_raw = record.get("title")
        company = record.get("company_name")
        location = record.get("location")
        description_raw = record.get("description")
        url = record.get("url")
        slug = record.get("slug")

        title = normalize_title(title_raw)
        description_text = clean_html(description_raw)
        location_text = normalize_location(location)
        company_text = (
            "" if company is None or (isinstance(company, float) and pd.isna(company))
            else str(company).strip()
        )

        tags = _split_values(record.get("tags"))
        employment_types = _split_values(record.get("job_types"))

        skills = extract_skills(
            title,
            " ".join(tags),
            description_text,
        )

        remote_value = record.get("remote")
        if isinstance(remote_value, str):
            remote = remote_value.strip().lower() in {"true", "1", "yes"}
        else:
            remote = bool(remote_value) if pd.notna(remote_value) else False

        rows.append(
            {
                "job_id": _stable_job_id(slug, url, title, company_text),
                "job_title": title or "Untitled job",
                "company": company_text or "Unknown",
                "location": location_text,
                "description": "" if description_raw is None else str(description_raw),
                "description_text": description_text,
                "url": "" if url is None or (isinstance(url, float) and pd.isna(url)) else str(url).strip(),
                "published_at": parse_created_at(record.get("created_at")),
                "remote": remote,
                "remote_label": "Remote" if remote else "On-site",
                "tags": _join(tags),
                "employment_types": _join(employment_types),
                "skills": _join(skills),
                "source": SOURCE_NAME,
                "data_type": DATA_TYPE,
                "fetched_at": now,
                "updated_at": now,
            }
        )

    normalized = pd.DataFrame(rows, columns=columns)
    normalized = normalized.drop_duplicates(subset="job_id", keep="last")
    return normalized.reset_index(drop=True)


def build_search_text(row: pd.Series) -> str:
    """Combine live job fields into one text field for TF-IDF matching."""
    parts = [
        str(row.get("job_title", "")) or "",
        str(row.get("company", "")) or "",
        str(row.get("location", "")) or "",
        str(row.get("tags", "")) or "",
        str(row.get("employment_types", "")) or "",
        str(row.get("skills", "")) or "",
        str(row.get("description_text", "")) or "",
    ]
    return " ".join(part for part in parts if part).strip()


def split_pipe_values(value: object) -> list[str]:
    """Split a ``"a | b"`` stored field back into a list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]
