"""Low-memory resume extraction, forecasting, job matching, and role comparison."""
from __future__ import annotations

import io
import zipfile
from collections import Counter
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from modules.graph_search import CAREER_GRAPH
from modules.knn import recommend_live_jobs
from modules.live_preprocessing import extract_skills, split_pipe_values

MAX_RESUME_BYTES = 5 * 1024 * 1024
MIN_RELIABLE_SALARY_SAMPLES = 5

CORE_ROLES = [
    "Data Analyst",
    "Data Scientist",
    "Software Engineer",
    "Machine Learning Engineer",
    "Web Developer",
]

# Omit bulky raw HTML `description` when loading live jobs for Career Tools.
CAREER_TOOLS_LIVE_COLUMNS = (
    "job_id",
    "job_title",
    "company",
    "location",
    "description_text",
    "url",
    "published_at",
    "remote",
    "remote_label",
    "tags",
    "employment_types",
    "skills",
    "source",
    "data_type",
    "first_fetched_at",
    "last_updated_at",
)

# Compact baseline skill profiles used only to supplement sparse postings and
# to assign sensible Foundation -> Core -> Specialization learning stages.
ROLE_SKILL_PROFILES: dict[str, list[tuple[str, str]]] = {
    "Data Analyst": [
        ("SQL", "Foundation"),
        ("Excel", "Foundation"),
        ("Python", "Foundation"),
        ("Data Analysis", "Core"),
        ("Statistics", "Core"),
        ("Pandas", "Core"),
        ("Power BI", "Core"),
        ("Tableau", "Core"),
        ("ETL", "Specialization"),
        ("Communication", "Foundation"),
    ],
    "Data Scientist": [
        ("Python", "Foundation"),
        ("SQL", "Foundation"),
        ("Statistics", "Foundation"),
        ("Pandas", "Core"),
        ("NumPy", "Core"),
        ("Machine Learning", "Core"),
        ("Data Science", "Core"),
        ("Deep Learning", "Specialization"),
        ("NLP", "Specialization"),
        ("AWS", "Specialization"),
    ],
    "Software Engineer": [
        ("Git", "Foundation"),
        ("Python", "Foundation"),
        ("Java", "Core"),
        ("JavaScript", "Core"),
        ("TypeScript", "Core"),
        ("SQL", "Core"),
        ("REST", "Core"),
        ("Docker", "Core"),
        ("Linux", "Core"),
        ("CI/CD", "Specialization"),
        ("AWS", "Specialization"),
        ("Kubernetes", "Specialization"),
    ],
    "Machine Learning Engineer": [
        ("Python", "Foundation"),
        ("SQL", "Foundation"),
        ("Git", "Foundation"),
        ("NumPy", "Core"),
        ("Pandas", "Core"),
        ("Machine Learning", "Core"),
        ("Deep Learning", "Core"),
        ("Docker", "Core"),
        ("Linux", "Core"),
        ("AWS", "Specialization"),
        ("Kubernetes", "Specialization"),
        ("CI/CD", "Specialization"),
    ],
    "Web Developer": [
        ("JavaScript", "Foundation"),
        ("TypeScript", "Foundation"),
        ("Git", "Foundation"),
        ("React", "Core"),
        ("Node.js", "Core"),
        ("REST", "Core"),
        ("SQL", "Core"),
        ("Vue", "Specialization"),
        ("Angular", "Specialization"),
        ("Docker", "Specialization"),
        ("UI", "Foundation"),
    ],
}

ROLE_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("AI & Machine Learning", ("machine learning", "ml engineer", "ai ", "deep learning", "nlp", "data scientist")),
    ("Data & Analytics", ("data analyst", "analytics", "business intelligence", "bi ", "data engineer", "etl")),
    ("Web & Frontend", ("web developer", "frontend", "front-end", "react", "angular", "vue", "full stack", "fullstack")),
    ("Software & Backend", ("software", "developer", "backend", "back-end", "engineer", "programmer", "java", "python")),
    ("Cloud & DevOps", ("devops", "cloud", "sre", "platform", "kubernetes", "infrastructure")),
    ("Product & Design", ("product", "ux", "ui ", "designer", "design")),
    ("Business & Operations", ("sales", "marketing", "finance", "hr", "recruiter", "operations", "support", "manager")),
]

_STAGE_ORDER = {"Foundation": 0, "Core": 1, "Specialization": 2}


def _graph_prereq_rank() -> dict[str, int]:
    """Derive a lightweight prerequisite depth from the existing CAREER_GRAPH."""
    in_degree = {node: 0 for node in CAREER_GRAPH}
    for edges in CAREER_GRAPH.values():
        for target in edges:
            in_degree[target] = in_degree.get(target, 0) + 1

    ranks: dict[str, int] = {}
    frontier = [node for node, deg in in_degree.items() if deg == 0]
    depth = 0
    while frontier:
        next_frontier = []
        for node in frontier:
            if node not in ranks:
                ranks[node] = depth
            for nxt in CAREER_GRAPH.get(node, {}):
                in_degree[nxt] -= 1
                if in_degree[nxt] <= 0:
                    next_frontier.append(nxt)
        frontier = next_frontier
        depth += 1
    return {k.lower(): v for k, v in ranks.items()}


_PREREQ_RANKS = _graph_prereq_rank()


def classify_job_category(title: object) -> str:
    """Map a job title into a broad category using lightweight substring checks."""
    text = f" {str(title or '').lower()} "
    for category, keywords in ROLE_CATEGORIES:
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse ISO strings or Unix timestamps (seconds or milliseconds) to UTC."""
    if series is None or series.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")

    numeric = pd.to_numeric(series, errors="coerce")
    valid_num = numeric.dropna()
    if len(valid_num) > 0 and len(valid_num) >= len(series.dropna()) * 0.8:
        median_val = float(valid_num.median())
        unit = "ms" if median_val > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)

    return pd.to_datetime(series, errors="coerce", utc=True)


def _find_date_column(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None
    for candidate in ("published_at", "listed_time", "original_listed_time", "posted_date", "created_at", "date"):
        if candidate in df.columns:
            return candidate
    return None


def _filter_mask(df: pd.DataFrame, role_filter: str = "", skill_filter: str = "") -> np.ndarray:
    """Build a boolean mask for role/skill filters without copying df."""
    mask = np.ones(len(df), dtype=bool)
    role_q = str(role_filter or "").strip()
    if role_q and role_q.lower() != "all" and "job_title" in df.columns:
        titles = df["job_title"].fillna("").astype(str)
        # Support either exact category name or title keyword substring.
        category_names = {cat.lower(): kws for cat, kws in ROLE_CATEGORIES}
        if role_q.lower() in category_names:
            kws = category_names[role_q.lower()]
            cat_mask = np.zeros(len(df), dtype=bool)
            lowered = titles.str.lower()
            for kw in kws:
                cat_mask |= lowered.str.contains(kw, regex=False).to_numpy()
            mask &= cat_mask
        else:
            mask &= titles.str.contains(role_q, case=False, regex=False).to_numpy()

    skill_q = str(skill_filter or "").strip()
    if skill_q and skill_q.lower() != "all":
        if "skills" in df.columns:
            mask &= df["skills"].fillna("").astype(str).str.contains(
                skill_q, case=False, regex=False
            ).to_numpy()
        elif "description_text" in df.columns:
            mask &= df["description_text"].fillna("").astype(str).str.contains(
                skill_q, case=False, regex=False
            ).to_numpy()
        elif "description" in df.columns:
            mask &= df["description"].fillna("").astype(str).str.contains(
                skill_q, case=False, regex=False
            ).to_numpy()

    return mask


# ---------------------------------------------------------------------------
# Feature 1: Job Demand Forecasting
# ---------------------------------------------------------------------------

def forecast_demand(
    live_jobs: pd.DataFrame | None,
    historical_jobs: pd.DataFrame | None = None,
    role_filter: str = "",
    skill_filter: str = "",
    granularity: str = "auto",
    horizon: int = 3,
) -> pd.DataFrame:
    """Analyze historical demand over time and apply lightweight OLS regression.

    - Reuses stored live jobs and dated historical jobs without duplicating frames.
    - Refuses to fabricate forecasts when fewer than the minimum required periods
      exist (3 months, 4 weeks, or 7 days), showing actual history instead.
    - Clearly separates actual demand (``actual_jobs``) from projected future
      demand (``forecast_jobs``) via ``record_type``.
    """
    dated_parts: list[pd.Series] = []
    historical_snapshot_count = 0

    if live_jobs is not None and not live_jobs.empty:
        live_date_col = _find_date_column(live_jobs)
        if live_date_col is not None:
            mask = _filter_mask(live_jobs, role_filter, skill_filter)
            if mask.any():
                parsed = _parse_dates(live_jobs.loc[mask, live_date_col]).dropna()
                if not parsed.empty:
                    dated_parts.append(parsed)

    if historical_jobs is not None and not historical_jobs.empty:
        hist_date_col = _find_date_column(historical_jobs)
        mask = _filter_mask(historical_jobs, role_filter, skill_filter)
        matched_hist = int(mask.sum())
        if hist_date_col is not None and matched_hist > 0:
            parsed = _parse_dates(historical_jobs.loc[mask, hist_date_col]).dropna()
            if not parsed.empty:
                dated_parts.append(parsed)
        else:
            historical_snapshot_count = matched_hist

    if not dated_parts:
        empty = pd.DataFrame(columns=["period", "month", "actual_jobs", "forecast_jobs", "record_type"])
        empty.attrs = {
            "forecast_reliable": False,
            "granularity": "M",
            "status_message": "No dated postings matched the selected filters.",
            "slope": None,
            "historical_snapshot_count": historical_snapshot_count,
        }
        return empty

    all_dates = pd.concat(dated_parts, ignore_index=True).dt.tz_localize(None)
    del dated_parts

    chosen_gran = granularity.upper() if granularity and granularity.lower() != "auto" else "AUTO"
    if chosen_gran == "AUTO":
        month_count = all_dates.dt.to_period("M").nunique()
        if month_count >= 2:
            chosen_gran = "M"
        else:
            chosen_gran = "D"

    if chosen_gran not in {"M", "W", "D"}:
        chosen_gran = "M"

    counts = all_dates.dt.to_period(chosen_gran).value_counts().sort_index()
    min_required = {"M": 3, "W": 4, "D": 7}[chosen_gran]
    unit_name = {"M": "months", "W": "weeks", "D": "days"}[chosen_gran]

    labels = counts.index.astype(str).tolist()
    values = counts.to_numpy(dtype=float)

    rows = [
        {
            "period": label,
            "month": label,
            "actual_jobs": int(val),
            "forecast_jobs": pd.NA,
            "record_type": "Actual",
        }
        for label, val in zip(labels, values)
    ]

    reliable = len(counts) >= min_required
    slope_val: float | None = None

    if reliable and horizon > 0:
        x = np.arange(len(values), dtype=float)
        x_mean = float(x.mean())
        y_mean = float(values.mean())
        denom = float(np.sum((x - x_mean) ** 2))
        slope_val = float(np.sum((x - x_mean) * (values - y_mean)) / denom) if denom > 0 else 0.0
        intercept = y_mean - slope_val * x_mean

        last_period = counts.index[-1]
        for step in range(1, int(horizon) + 1):
            future_period = str(last_period + step)
            pred = max(0, int(round(intercept + slope_val * (len(values) - 1 + step))))
            rows.append(
                {
                    "period": future_period,
                    "month": future_period,
                    "actual_jobs": pd.NA,
                    "forecast_jobs": pred,
                    "record_type": "Forecast",
                }
            )
        direction = "upward" if slope_val > 0.25 else ("downward" if slope_val < -0.25 else "stable")
        status_msg = (
            f"OLS linear regression forecast ({horizon} future {unit_name}) based on "
            f"{len(counts)} observed {unit_name}; trend is {direction} "
            f"({slope_val:+.1f} postings/{unit_name[:-1]})."
        )
    else:
        status_msg = (
            f"Showing actual observed demand only ({len(counts)} {unit_name} available). "
            f"At least {min_required} distinct {unit_name} are needed for a reliable forecast."
        )

    result = pd.DataFrame(rows)
    result.attrs = {
        "forecast_reliable": reliable,
        "granularity": chosen_gran,
        "status_message": status_msg,
        "slope": round(slope_val, 3) if slope_val is not None else None,
        "historical_snapshot_count": historical_snapshot_count,
    }
    return result


def demand_summary(
    live_jobs: pd.DataFrame | None,
    historical_jobs: pd.DataFrame | None = None,
    top_n: int = 8,
) -> dict[str, pd.DataFrame]:
    """Summarize top job titles, broad categories, and skills with minimal RAM."""
    title_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()

    if live_jobs is not None and not live_jobs.empty:
        if "job_title" in live_jobs.columns:
            vc = live_jobs["job_title"].dropna().astype(str).value_counts()
            for title, count in vc.items():
                title_counts[title] += int(count)
                category_counts[classify_job_category(title)] += int(count)
        if "skills" in live_jobs.columns:
            for value in live_jobs["skills"].fillna(""):
                for skill in split_pipe_values(value):
                    skill_counts[skill] += 1

    if historical_jobs is not None and not historical_jobs.empty and "job_title" in historical_jobs.columns:
        vc_hist = historical_jobs["job_title"].dropna().astype(str).value_counts()
        for title, count in vc_hist.head(200).items():
            title_counts[title] += int(count)
        for title, count in vc_hist.items():
            category_counts[classify_job_category(title)] += int(count)

    top_titles = pd.DataFrame(title_counts.most_common(top_n), columns=["Job Title", "Postings"])
    top_categories = pd.DataFrame(category_counts.most_common(top_n), columns=["Category", "Postings"])
    top_skills = pd.DataFrame(skill_counts.most_common(top_n), columns=["Skill", "Live Postings"])

    return {
        "titles": top_titles,
        "categories": top_categories,
        "skills": top_skills,
    }


# ---------------------------------------------------------------------------
# Feature 2: Resume Analyzer + Personalized Learning Roadmap
# ---------------------------------------------------------------------------

def extract_resume_text(uploaded_file) -> str:
    """Extract plain text from a PDF or DOCX upload (<= 5 MB); never saves to disk."""
    payload = uploaded_file.getvalue()
    if len(payload) > MAX_RESUME_BYTES:
        del payload
        raise ValueError("Resume must be 5 MB or smaller.")

    name = str(getattr(uploaded_file, "name", "")).lower()
    text = ""
    try:
        if name.endswith(".docx"):
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    xml = archive.read("word/document.xml")
            except (zipfile.BadZipFile, KeyError) as exc:
                raise ValueError("Could not read this DOCX file. Make sure it is a valid Word document.") from exc
            root = ElementTree.fromstring(xml)
            text = " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
        elif name.endswith(".pdf"):
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ValueError("Install pypdf once to analyze PDF resumes.") from exc
            try:
                reader = PdfReader(io.BytesIO(payload))
                text = " ".join(page.extract_text() or "" for page in reader.pages)
            except Exception as exc:
                raise ValueError(f"Could not read this PDF file: {exc}") from exc
        else:
            raise ValueError("Upload a PDF or DOCX resume.")
    finally:
        del payload

    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        raise ValueError("No readable text was found in this resume (scanned image PDFs are not supported).")
    return cleaned


def resume_skills(text: str) -> list[str]:
    """Extract skills from resume text reusing the existing skill vocabulary."""
    return extract_skills(text, limit=30)


def _role_skill_counts(
    role: str,
    live_jobs: pd.DataFrame | None,
    historical_jobs: pd.DataFrame | None = None,
    max_hist_scan: int = 200,
) -> tuple[Counter[str], int, int]:
    """Count demanded skills for a role across live jobs and a bounded historical slice."""
    counts: Counter[str] = Counter()
    live_count = 0
    hist_count = 0
    role_q = str(role or "").strip()

    if live_jobs is not None and not live_jobs.empty:
        if role_q and "job_title" in live_jobs.columns:
            mask = live_jobs["job_title"].fillna("").str.contains(role_q, case=False, regex=False)
            live_count = int(mask.sum())
            skill_series = live_jobs.loc[mask, "skills"] if "skills" in live_jobs.columns else pd.Series(dtype=str)
        else:
            live_count = len(live_jobs)
            skill_series = live_jobs["skills"] if "skills" in live_jobs.columns else pd.Series(dtype=str)

        for value in skill_series.fillna(""):
            for skill in split_pipe_values(value):
                counts[skill] += 1

    if historical_jobs is not None and not historical_jobs.empty and "job_title" in historical_jobs.columns:
        if role_q:
            mask = historical_jobs["job_title"].fillna("").str.contains(role_q, case=False, regex=False)
            hist_count = int(mask.sum())
            if hist_count > 0:
                desc_col = "description" if "description" in historical_jobs.columns else None
                sample_idx = historical_jobs.index[mask][:max_hist_scan]
                titles = historical_jobs.loc[sample_idx, "job_title"].fillna("").astype(str)
                descs = (
                    historical_jobs.loc[sample_idx, desc_col].fillna("").astype(str)
                    if desc_col else pd.Series([""] * len(sample_idx), index=sample_idx)
                )
                for title, desc in zip(titles, descs):
                    for skill in extract_skills(title, desc[:2000], limit=15):
                        counts[skill] += 1

    return counts, live_count, hist_count


def _match_profile_key(role: str) -> str | None:
    role_lower = str(role or "").strip().lower()
    for key in ROLE_SKILL_PROFILES:
        if key.lower() == role_lower or key.lower() in role_lower or role_lower in key.lower():
            return key
    return None


def build_learning_roadmap(
    resume_input: str | list[str] | set[str],
    live_jobs: pd.DataFrame | None,
    role: str,
    historical_jobs: pd.DataFrame | None = None,
    top_n: int = 12,
) -> dict[str, object]:
    """Compare resume skills against target role requirements and build a prioritized roadmap."""
    if isinstance(resume_input, str):
        detected = resume_skills(resume_input)
    else:
        detected = [str(s).strip() for s in resume_input if str(s).strip()]
    user_lower = {s.lower() for s in detected}

    counts, live_count, hist_count = _role_skill_counts(role, live_jobs, historical_jobs)
    total_scanned = live_count + min(hist_count, 200)

    profile_key = _match_profile_key(role)
    profile_map = dict(ROLE_SKILL_PROFILES.get(profile_key, [])) if profile_key else {}

    # Build combined skill list: observed market skills first, supplemented by role profile when sparse.
    ranked_skills: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for skill, freq in counts.most_common(top_n):
        ranked_skills.append((skill, freq, profile_map.get(skill, "Core")))
        seen.add(skill.lower())

    if len(ranked_skills) < 6 and profile_key:
        for skill, stage in ROLE_SKILL_PROFILES[profile_key]:
            if skill.lower() not in seen:
                ranked_skills.append((skill, 0, stage))
                seen.add(skill.lower())
            if len(ranked_skills) >= top_n:
                break

    matched = [skill for skill, _, _ in ranked_skills if skill.lower() in user_lower]
    missing_entries = [entry for entry in ranked_skills if entry[0].lower() not in user_lower]

    # Sort missing skills by: stage (Foundation -> Core -> Specialization),
    # CAREER_GRAPH prerequisite rank, then descending market demand frequency.
    def _sort_key(entry: tuple[str, int, str]) -> tuple[int, int, int, str]:
        skill, freq, stage = entry
        prereq = _PREREQ_RANKS.get(skill.lower(), 5)
        return (_STAGE_ORDER.get(stage, 1), prereq, -freq, skill)

    missing_entries.sort(key=_sort_key)

    roadmap_rows = []
    max_freq = max((freq for _, freq, _ in ranked_skills), default=1) or 1
    for step_idx, (skill, freq, stage) in enumerate(missing_entries, start=1):
        if freq > 0 and (freq / max_freq >= 0.5 or stage == "Foundation"):
            priority = "High"
        elif freq > 0 or stage in {"Foundation", "Core"}:
            priority = "Medium"
        else:
            priority = "Optional"

        if freq > 0 and total_scanned > 0:
            pct = round(min(100.0, (freq / total_scanned) * 100), 1)
            demand_label = f"{freq} postings ({pct}%)"
        else:
            demand_label = "Role baseline"

        if stage == "Foundation":
            focus = f"Build core fundamentals in {skill} with hands-on exercises."
        elif stage == "Core":
            focus = f"Apply {skill} in a portfolio project aligned with {role or 'target role'}."
        else:
            focus = f"Add production/specialized {skill} experience to stand out."

        roadmap_rows.append(
            {
                "Step": step_idx,
                "Skill": skill,
                "Priority": priority,
                "Stage": stage,
                "Market Demand": demand_label,
                "Suggested Focus": focus,
            }
        )

    total_target = len(ranked_skills)
    readiness_pct = round((len(matched) / total_target) * 100, 1) if total_target else 0.0

    sources = []
    if live_count > 0:
        sources.append(f"{live_count} live jobs")
    if hist_count > 0:
        sources.append(f"{hist_count:,} historical jobs (sampled {min(hist_count, 200)})")
    if profile_key and (live_count + hist_count) < 5:
        sources.append(f"{profile_key} baseline profile")
    source_note = " + ".join(sources) if sources else "No matching postings found"

    return {
        "detected_skills": detected,
        "matched_skills": matched,
        "missing_skills": [row["Skill"] for row in roadmap_rows],
        "readiness_pct": readiness_pct,
        "live_matches": live_count,
        "historical_matches": hist_count,
        "data_source": source_note,
        "roadmap_df": pd.DataFrame(roadmap_rows),
    }


def roadmap(
    resume_text: str,
    jobs: pd.DataFrame,
    role: str,
    historical_jobs: pd.DataFrame | None = None,
) -> tuple[list[str], list[str]]:
    """Backward-compatible helper returning (matched_skills, missing_skills)."""
    result = build_learning_roadmap(resume_text, jobs, role, historical_jobs=historical_jobs)
    return list(result["matched_skills"]), list(result["missing_skills"])


# ---------------------------------------------------------------------------
# Feature 3: Resume–Job Matching and Recommendations
# ---------------------------------------------------------------------------

def matching_jobs(
    resume_text: str,
    live_jobs: pd.DataFrame | None,
    historical_jobs: pd.DataFrame | None = None,
    k: int = 5,
    target_role: str = "",
    source_mode: str = "Both",
    extracted_skills: list[str] | None = None,
    max_historical: int = 2000,
) -> pd.DataFrame:
    """Reuse sparse TF-IDF KNN across live jobs and a memory-bounded historical slice."""
    mode = str(source_mode or "Both").strip().lower()
    include_live = mode in {"both", "live", "live only"}
    include_hist = mode in {"both", "historical", "historical only"}

    skills = extracted_skills if extracted_skills is not None else resume_skills(resume_text)
    candidates: list[pd.DataFrame] = []

    if include_live and live_jobs is not None and not live_jobs.empty:
        live_cols = [
            c for c in (
                "job_title", "company", "description_text", "skills", "tags",
                "location", "employment_types", "remote", "remote_label",
                "published_at", "url", "source", "data_type",
            )
            if c in live_jobs.columns
        ]
        live_view = live_jobs.loc[:, live_cols]
        candidates.append(live_view)

    if include_hist and historical_jobs is not None and not historical_jobs.empty:
        # Prioritize role-matching historical records before filling up to max_historical,
        # slicing only required columns to keep RAM minimal.
        hist_indices = historical_jobs.index
        role_q = str(target_role or "").strip()
        if role_q and "job_title" in historical_jobs.columns:
            role_mask = historical_jobs["job_title"].fillna("").str.contains(
                role_q, case=False, regex=False
            )
            matched_idx = hist_indices[role_mask][:max_historical]
            if len(matched_idx) < max_historical:
                remaining = max_historical - len(matched_idx)
                other_idx = hist_indices[~role_mask][:remaining]
                chosen_idx = matched_idx.append(other_idx)
            else:
                chosen_idx = matched_idx
        else:
            chosen_idx = hist_indices[:max_historical]

        needed_cols = [
            c for c in ("job_title", "company", "description", "url", "published_at")
            if c in historical_jobs.columns
        ]
        subset = historical_jobs.loc[chosen_idx, needed_cols]
        n_hist = len(subset)
        history = pd.DataFrame(
            {
                "job_title": subset["job_title"].to_numpy() if "job_title" in subset.columns else [""] * n_hist,
                "company": subset["company"].to_numpy() if "company" in subset.columns else ["Unknown"] * n_hist,
                "description_text": subset["description"].to_numpy() if "description" in subset.columns else [""] * n_hist,
                "skills": [""] * n_hist,
                "tags": [""] * n_hist,
                "location": ["Historical dataset"] * n_hist,
                "employment_types": [""] * n_hist,
                "remote": [False] * n_hist,
                "remote_label": ["Historical"] * n_hist,
                "published_at": subset["published_at"].to_numpy() if "published_at" in subset.columns else [None] * n_hist,
                "url": subset["url"].to_numpy() if "url" in subset.columns else [""] * n_hist,
                "source": ["Historical"] * n_hist,
                "data_type": ["historical"] * n_hist,
            }
        )
        candidates.append(history)

    if not candidates:
        return recommend_live_jobs(pd.DataFrame(), skills="", k=k)

    combined = candidates[0] if len(candidates) == 1 else pd.concat(candidates, ignore_index=True)
    result = recommend_live_jobs(
        combined,
        skills=", ".join(skills),
        preferred_title=target_role,
        extra_profile_text=resume_text,
        k=k,
    )
    del combined, candidates
    return result


# ---------------------------------------------------------------------------
# Feature 4: Career Role Comparison
# ---------------------------------------------------------------------------

def _reliable_salary_stats(
    salaries: pd.Series,
    min_samples: int = MIN_RELIABLE_SALARY_SAMPLES,
) -> dict[str, object]:
    """Return IQR-filtered salary statistics only when enough valid samples exist."""
    numeric = pd.to_numeric(salaries, errors="coerce")
    valid = numeric[np.isfinite(numeric) & (numeric > 0)]
    if len(valid) < min_samples:
        return {"reliable": False, "count": int(len(valid)), "median": None}

    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1
    if iqr > 0:
        inliers = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    else:
        inliers = valid

    if len(inliers) < min_samples:
        return {"reliable": False, "count": int(len(inliers)), "median": None}

    return {
        "reliable": True,
        "count": int(len(inliers)),
        "median": float(inliers.median()),
        "q25": float(inliers.quantile(0.25)),
        "q75": float(inliers.quantile(0.75)),
    }


def compare_career_roles(
    roles: list[str],
    live_jobs: pd.DataFrame | None,
    historical_jobs: pd.DataFrame | None = None,
    resume_skills_list: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare selected career roles across skills, overlap, counts, trends, and salary.

    Reuses existing live and historical DataFrames in place without making copies.
    Returns ``(comparison_df, trends_df)``.
    """
    if not roles:
        return pd.DataFrame(), pd.DataFrame()

    has_resume = bool(resume_skills_list)
    user_skills = list(resume_skills_list or [])
    total_live = len(live_jobs) if live_jobs is not None else 0
    total_hist = len(historical_jobs) if historical_jobs is not None else 0
    total_all = total_live + total_hist

    rows = []
    trend_series: dict[str, pd.Series] = {}

    for role in roles:
        info = build_learning_roadmap(
            user_skills,
            live_jobs,
            role,
            historical_jobs=historical_jobs,
            top_n=8,
        )
        live_count = int(info["live_matches"])
        hist_count = int(info["historical_matches"])
        role_total = live_count + hist_count

        # Top demanded skills = matched + missing in role order (up to 6).
        top_skills = (list(info["matched_skills"]) + list(info["missing_skills"]))[:6]

        # Reliable salary check on historical dataset (Arbeitnow live jobs do not include salary).
        salary_display = "Insufficient reliable data"
        if (
            historical_jobs is not None
            and not historical_jobs.empty
            and "salary" in historical_jobs.columns
            and "job_title" in historical_jobs.columns
            and hist_count > 0
        ):
            role_mask = historical_jobs["job_title"].fillna("").str.contains(
                role, case=False, regex=False
            )
            sal_stats = _reliable_salary_stats(historical_jobs.loc[role_mask, "salary"])
            if sal_stats["reliable"]:
                salary_display = f"${sal_stats['median']:,.0f} (n={sal_stats['count']})"

        # Historical demand trend for this role (reusing forecast_demand on filtered postings).
        role_trend = forecast_demand(live_jobs, historical_jobs, role_filter=role, horizon=0)
        actual_points = role_trend.dropna(subset=["actual_jobs"]) if not role_trend.empty else pd.DataFrame()
        if not actual_points.empty:
            trend_series[role] = actual_points.set_index("period")["actual_jobs"].astype(int)

        if len(actual_points) >= 3:
            slope = role_trend.attrs.get("slope")
            if slope is None:
                vals = actual_points["actual_jobs"].to_numpy(dtype=float)
                slope = float(vals[-1] - vals[0]) / max(1, len(vals) - 1)
            trend_label = "Rising" if slope > 0.25 else ("Declining" if slope < -0.25 else "Stable")
            trend_text = f"{trend_label} ({len(actual_points)} periods)"
        elif role_total > 0 and total_all > 0:
            share = round((role_total / total_all) * 100, 2)
            trend_text = f"Snapshot: {role_total:,} jobs ({share}% share)"
        else:
            trend_text = "No dated trend yet"

        if has_resume:
            total_req = len(info["matched_skills"]) + len(info["missing_skills"])
            overlap_str = f"{info['readiness_pct']:.0f}% ({len(info['matched_skills'])}/{total_req})"
            matched_str = ", ".join(info["matched_skills"][:6]) or "None"
            missing_str = ", ".join(info["missing_skills"][:6]) or "None"
        else:
            overlap_str = "Upload resume"
            matched_str = "Upload resume"
            missing_str = ", ".join(top_skills[:5]) or "None"

        rows.append(
            {
                "Role": role,
                "Live Jobs": live_count,
                "Historical Jobs": hist_count,
                "Total Jobs": role_total,
                "Top Demanded Skills": ", ".join(top_skills) or "N/A",
                "Resume Overlap": overlap_str,
                "Matched Skills": matched_str,
                "Missing Skills": missing_str,
                "Historical Trend": trend_text,
                "Median Salary": salary_display,
            }
        )

    comparison_df = pd.DataFrame(rows)
    if trend_series:
        trends_df = pd.DataFrame(trend_series).fillna(0).astype(int).sort_index()
    else:
        trends_df = pd.DataFrame()

    return comparison_df, trends_df
