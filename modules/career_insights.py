"""Reusable skill-demand, skill-trend, and job-freshness analytics."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from modules.live_preprocessing import extract_skills, split_pipe_values

# Default configurable thresholds for Feature 1.2 (Emerging vs Declining Skills)
DEFAULT_MIN_PERIOD_POSTINGS = 3
DEFAULT_MIN_SKILL_POSTINGS = 3
DEFAULT_MIN_PREV_FOR_DECLINE = 2
DEFAULT_TREND_CHANGE_PCT = 15.0
DEFAULT_MIN_SHARE_DELTA_PP = 1.0

# Default configurable thresholds for Feature 2.2 (Job Expiry & Freshness Tracker)
DEFAULT_FRESH_DAYS = 7
DEFAULT_AGING_DAYS = 30

FRESHNESS_BADGES = {
    "Fresh": "🟢 Fresh",
    "Aging": "🟡 Aging",
    "Stale / possibly outdated": "🟠 Stale / possibly outdated",
    "Expired": "🔴 Expired (source confirmed)",
    "Missing / Invalid Date": "⚪ Missing / Invalid Date",
}

FRESHNESS_SORT_RANK = {
    "Fresh": 0,
    "Aging": 1,
    "Stale / possibly outdated": 2,
    "Expired": 3,
    "Missing / Invalid Date": 4,
}

EXPIRED_STATUS_VALUES = frozenset(
    {"expired", "closed", "removed", "inactive", "filled", "archived", "cancelled"}
)

EXPERIENCE_LEVEL_PATTERNS: dict[str, tuple[str, ...]] = {
    "Entry / Junior": ("entry", "junior", "jr", "intern", "graduate", "associate", "trainee", "werkstudent"),
    "Mid-Level": ("mid", "intermediate", "regular", "professional"),
    "Senior / Lead": ("senior", "sr", "lead", "principal", "staff", "architect"),
    "Manager / Executive": ("manager", "director", "head", "vp", "chief", "executive", "teamlead"),
}


def skill_demand(jobs: pd.DataFrame, limit: int = 12) -> pd.Series:
    skills = Counter(
        skill for value in jobs.get("skills", pd.Series(dtype=str)).fillna("")
        for skill in split_pipe_values(value)
    )
    return pd.Series(skills).sort_values(ascending=False).head(limit) if skills else pd.Series(dtype=int)


def skill_gap(user_skills: str, jobs: pd.DataFrame, role: str = "") -> tuple[list[str], list[str]]:
    """Return possessed and missing in-demand skills from matching live jobs."""
    user = {item.strip().lower() for item in user_skills.replace(";", ",").split(",") if item.strip()}
    candidates = jobs
    if role.strip():
        mask = jobs["job_title"].fillna("").str.contains(role, case=False, regex=False)
        candidates = jobs[mask]
    demand = skill_demand(candidates, 20)
    labels = list(demand.index)
    matched = [skill for skill in labels if skill.lower() in user]
    missing = [skill for skill in labels if skill.lower() not in user]
    return matched, missing


# ---------------------------------------------------------------------------
# Shared Date & Filter Utilities (Low-Memory, No Full DataFrame Copies)
# ---------------------------------------------------------------------------

def parse_job_timestamps(series: pd.Series | None) -> pd.Series:
    """Safely parse ISO date strings or Unix timestamps (seconds or ms) into UTC."""
    if series is None or series.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")

    numeric = pd.to_numeric(series, errors="coerce")
    valid_num = numeric.dropna()
    non_empty = series.dropna()
    if len(valid_num) > 0 and len(valid_num) >= len(non_empty) * 0.8:
        median_val = float(valid_num.median())
        unit = "ms" if median_val > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)

    return pd.to_datetime(series, errors="coerce", utc=True)


def find_date_column(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None
    for col in ("published_at", "listed_time", "original_listed_time", "posted_date", "created_at", "date"):
        if col in df.columns:
            return col
    return None


def _experience_mask(df: pd.DataFrame, experience_filter: str) -> np.ndarray:
    exp_q = str(experience_filter or "").strip()
    if not exp_q or exp_q.lower() == "all":
        return np.ones(len(df), dtype=bool)

    keywords = EXPERIENCE_LEVEL_PATTERNS.get(exp_q, (exp_q.lower(),))
    mask = np.zeros(len(df), dtype=bool)

    if "job_level" in df.columns:
        levels = df["job_level"].astype(str).str.lower()
        for kw in keywords:
            mask |= levels.str.contains(kw, regex=False, na=False).to_numpy()

    if "job_title" in df.columns:
        titles = df["job_title"].fillna("").astype(str).str.lower()
        for kw in keywords:
            mask |= titles.str.contains(kw, regex=False, na=False).to_numpy()

    return mask


def _build_filter_mask(
    df: pd.DataFrame,
    role_filter: str = "",
    location_filter: str = "",
    experience_filter: str = "",
) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    role_q = str(role_filter or "").strip()
    if role_q and role_q.lower() != "all" and "job_title" in df.columns:
        mask &= df["job_title"].fillna("").astype(str).str.contains(
            role_q, case=False, regex=False
        ).to_numpy()

    loc_q = str(location_filter or "").strip()
    if loc_q and loc_q.lower() != "all" and "location" in df.columns:
        mask &= df["location"].fillna("").astype(str).str.contains(
            loc_q, case=False, regex=False
        ).to_numpy()

    if experience_filter and str(experience_filter).strip().lower() != "all":
        mask &= _experience_mask(df, experience_filter)

    return mask


# ---------------------------------------------------------------------------
# Feature 1.2: Emerging vs Declining Skills
# ---------------------------------------------------------------------------

def _collect_dated_skill_records(
    live_jobs: pd.DataFrame | None,
    historical_jobs: pd.DataFrame | None = None,
    role_filter: str = "",
    location_filter: str = "",
    experience_filter: str = "",
    start_date: Any = None,
    end_date: Any = None,
    max_historical_scan: int = 3000,
) -> tuple[list[tuple[pd.Timestamp, set[str]]], dict[str, int]]:
    """Collect (utc_timestamp, unique_skills_in_job) pairs without copying large frames."""
    records: list[tuple[pd.Timestamp, set[str]]] = []
    meta = {
        "live_dated_jobs": 0,
        "historical_dated_jobs": 0,
        "undated_historical_jobs": 0,
        "missing_date_jobs": 0,
    }

    start_ts = pd.to_datetime(start_date, utc=True, errors="coerce") if start_date is not None else pd.NaT
    end_ts = pd.to_datetime(end_date, utc=True, errors="coerce") if end_date is not None else pd.NaT
    if pd.notna(end_ts) and end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    def _process_frame(df: pd.DataFrame, is_historical: bool) -> None:
        date_col = find_date_column(df)
        mask = _build_filter_mask(df, role_filter, location_filter, experience_filter)
        if not mask.any():
            return

        if date_col is None:
            if is_historical:
                meta["undated_historical_jobs"] += int(mask.sum())
            else:
                meta["missing_date_jobs"] += int(mask.sum())
            return

        indices = df.index[mask]
        if is_historical and len(indices) > max_historical_scan:
            indices = indices[:max_historical_scan]

        parsed_dates = parse_job_timestamps(df.loc[indices, date_col])
        invalid_count = int(parsed_dates.isna().sum())
        meta["missing_date_jobs"] += invalid_count

        valid_date_mask = parsed_dates.notna()
        if pd.notna(start_ts):
            valid_date_mask &= parsed_dates >= start_ts
        if pd.notna(end_ts):
            valid_date_mask &= parsed_dates <= end_ts

        if not valid_date_mask.any():
            return

        valid_idx = indices[valid_date_mask.to_numpy()]
        valid_timestamps = parsed_dates.loc[valid_idx].tolist()

        has_skills_col = "skills" in df.columns
        skills_list = df.loc[valid_idx, "skills"].fillna("").astype(str).tolist() if has_skills_col else [""] * len(valid_idx)
        titles_list = df.loc[valid_idx, "job_title"].fillna("").astype(str).tolist() if "job_title" in df.columns else [""] * len(valid_idx)

        desc_col = "description_text" if "description_text" in df.columns else ("description" if "description" in df.columns else None)
        descs_list = df.loc[valid_idx, desc_col].fillna("").astype(str).tolist() if desc_col else [""] * len(valid_idx)

        for ts, raw_skills, title, desc in zip(valid_timestamps, skills_list, titles_list, descs_list):
            if raw_skills.strip():
                job_skills = set(split_pipe_values(raw_skills))
            else:
                job_skills = set(extract_skills(title, desc[:1500], limit=15))
            records.append((ts, job_skills))
            if is_historical:
                meta["historical_dated_jobs"] += 1
            else:
                meta["live_dated_jobs"] += 1

    if live_jobs is not None and not live_jobs.empty:
        _process_frame(live_jobs, is_historical=False)

    if historical_jobs is not None and not historical_jobs.empty:
        _process_frame(historical_jobs, is_historical=True)

    return records, meta


def classify_skill_trend(
    prev_count: int,
    curr_count: int,
    prev_total_jobs: int,
    curr_total_jobs: int,
    min_skill_postings: int = DEFAULT_MIN_SKILL_POSTINGS,
    min_prev_for_decline: int = DEFAULT_MIN_PREV_FOR_DECLINE,
    change_threshold_pct: float = DEFAULT_TREND_CHANGE_PCT,
    min_share_delta_pp: float = DEFAULT_MIN_SHARE_DELTA_PP,
) -> dict[str, Any]:
    """Classify a single skill's trend using total-postings-normalized share.

    Normalizing by ``prev_total_jobs`` and ``curr_total_jobs`` ensures a skill
    is never labeled as declining merely because total job volume decreased.
    """
    total_mentions = int(prev_count) + int(curr_count)
    prev_share = (prev_count / prev_total_jobs * 100.0) if prev_total_jobs > 0 else 0.0
    curr_share = (curr_count / curr_total_jobs * 100.0) if curr_total_jobs > 0 else 0.0
    share_delta_pp = curr_share - prev_share

    if prev_share > 0:
        pct_change: float | None = ((curr_share - prev_share) / prev_share) * 100.0
        pct_display = f"{pct_change:+.1f}%"
    elif curr_share > 0:
        pct_change = None
        pct_display = "New (0% in prev)"
    else:
        pct_change = 0.0
        pct_display = "0.0%"

    if total_mentions < min_skill_postings:
        classification = "Insufficient sample"
    elif (
        share_delta_pp >= min_share_delta_pp
        and (
            (pct_change is not None and pct_change >= change_threshold_pct)
            or (prev_count == 0 and curr_count >= min_skill_postings)
        )
    ):
        classification = "Emerging"
    elif (
        prev_count >= min_prev_for_decline
        and share_delta_pp <= -min_share_delta_pp
        and pct_change is not None
        and pct_change <= -change_threshold_pct
    ):
        classification = "Declining"
    else:
        classification = "Stable"

    return {
        "previous_count": int(prev_count),
        "current_count": int(curr_count),
        "previous_share_pct": round(prev_share, 2),
        "current_share_pct": round(curr_share, 2),
        "share_delta_pp": round(share_delta_pp, 2),
        "pct_change": round(pct_change, 1) if pct_change is not None else pd.NA,
        "pct_change_display": pct_display,
        "trend": classification,
        "postings_with_skill": total_mentions,
    }


def analyze_skill_trends(
    live_jobs: pd.DataFrame | None,
    historical_jobs: pd.DataFrame | None = None,
    period_mode: str = "auto",
    role_filter: str = "",
    location_filter: str = "",
    experience_filter: str = "",
    start_date: Any = None,
    end_date: Any = None,
    min_period_postings: int = DEFAULT_MIN_PERIOD_POSTINGS,
    min_skill_postings: int = DEFAULT_MIN_SKILL_POSTINGS,
    min_prev_for_decline: int = DEFAULT_MIN_PREV_FOR_DECLINE,
    change_threshold_pct: float = DEFAULT_TREND_CHANGE_PCT,
    min_share_delta_pp: float = DEFAULT_MIN_SHARE_DELTA_PP,
) -> dict[str, Any]:
    """Identify Emerging, Stable, and Declining skills across two time periods.

    Returns a dictionary containing:
    - ``available`` (bool): whether sufficient dated postings exist
    - ``status_message`` (str): clear explanation of periods, logic, or limitations
    - ``coverage_warning`` (str | None): notice when dataset spans a short window
    - ``trends_df`` (pd.DataFrame): classified skills table
    - ``emerging_df``, ``stable_df``, ``declining_df`` (pd.DataFrame)
    - ``timeline_df`` (pd.DataFrame): period x skill share (%) for trend charts
    - ``period_info`` (dict): exact dates and job counts for previous vs current periods
    """
    empty_df = pd.DataFrame(
        columns=[
            "Skill", "Current Demand", "Previous Demand",
            "Current Share (%)", "Previous Share (%)", "Share Change (pp)",
            "Percentage Change", "Percentage Change (%)",
            "Trend Classification", "Skill Postings", "Total Postings Analyzed",
        ]
    )

    records, meta = _collect_dated_skill_records(
        live_jobs,
        historical_jobs=historical_jobs,
        role_filter=role_filter,
        location_filter=location_filter,
        experience_filter=experience_filter,
        start_date=start_date,
        end_date=end_date,
    )

    if len(records) < min_period_postings * 2:
        undated_note = (
            f" ({meta['undated_historical_jobs']:,} historical CSV records lack posting dates and were not used to invent trends.)"
            if meta["undated_historical_jobs"] > 0 else ""
        )
        return {
            "available": False,
            "status_message": (
                f"Trend analysis is unavailable: only {len(records)} dated postings matched the filters, "
                f"and at least {min_period_postings * 2} dated postings across 2 time periods are required.{undated_note}"
            ),
            "coverage_warning": None,
            "trends_df": empty_df,
            "emerging_df": empty_df,
            "stable_df": empty_df,
            "declining_df": empty_df,
            "timeline_df": pd.DataFrame(),
            "period_info": meta,
        }

    records.sort(key=lambda item: item[0])
    timestamps = pd.Series([item[0] for item in records]).dt.tz_convert("UTC").dt.tz_localize(None)

    mode = str(period_mode or "auto").strip().lower()
    months = timestamps.dt.to_period("M")
    weeks = timestamps.dt.to_period("W")
    days = timestamps.dt.to_period("D")

    chosen_gran = "M"
    if mode in {"month", "monthly", "m"}:
        chosen_gran = "M"
        buckets = months
    elif mode in {"week", "weekly", "w"}:
        chosen_gran = "W"
        buckets = weeks
    elif mode in {"day", "daily", "d"}:
        chosen_gran = "D"
        buckets = days
    else:
        # Auto mode: prefer monthly if >= 2 months have enough jobs, else weekly, else daily window.
        m_vc = months.value_counts()
        w_vc = weeks.value_counts()
        if len(m_vc) >= 2 and int((m_vc >= min_period_postings).sum()) >= 2:
            chosen_gran = "M"
            buckets = months
        elif (
            len(w_vc) >= 2
            and int((w_vc >= min_period_postings).sum()) >= 2
            and (timestamps.max() - timestamps.min()).total_seconds() >= 13 * 86400
        ):
            # Avoid comparing partial calendar weeks when fewer than
            # 14 calendar dates are represented in the dataset.
            chosen_gran = "W"
            buckets = weeks
        else:
            chosen_gran = "D"
            buckets = days

    unique_buckets = sorted(buckets.unique())
    if len(unique_buckets) < 2:
        unit_label = {"M": "months", "W": "weeks", "D": "days"}[chosen_gran]
        return {
            "available": False,
            "status_message": (
                f"Insufficient historical date coverage for {unit_label} comparison: "
                f"all {len(records)} dated postings fall within a single period ({unique_buckets[0] if unique_buckets else 'N/A'}). "
                "Switch granularity to Auto/Daily or collect postings across additional periods."
            ),
            "coverage_warning": None,
            "trends_df": empty_df,
            "emerging_df": empty_df,
            "stable_df": empty_df,
            "declining_df": empty_df,
            "timeline_df": pd.DataFrame(),
            "period_info": meta,
        }

    # Split buckets into Previous Period vs Current (Recent) Period.
    # First try comparing the latest bucket against the previous bucket; if either has
    # too few postings, split the distinct sorted buckets into earlier half vs recent half.
    latest_bucket = unique_buckets[-1]
    prev_bucket = unique_buckets[-2]
    latest_mask = (buckets == latest_bucket).to_numpy()
    prev_mask = (buckets == prev_bucket).to_numpy()

    if int(latest_mask.sum()) >= min_period_postings and int(prev_mask.sum()) >= min_period_postings:
        curr_mask = latest_mask
        prev_period_label = str(prev_bucket)
        curr_period_label = str(latest_bucket)
    else:
        mid = max(1, len(unique_buckets) // 2)
        prev_set = set(unique_buckets[:mid])
        curr_set = set(unique_buckets[mid:])
        prev_mask = buckets.isin(prev_set).to_numpy()
        curr_mask = buckets.isin(curr_set).to_numpy()
        prev_period_label = f"{unique_buckets[0]} to {unique_buckets[mid - 1]}"
        curr_period_label = f"{unique_buckets[mid]} to {unique_buckets[-1]}"

    prev_total = int(prev_mask.sum())
    curr_total = int(curr_mask.sum())

    if prev_total < min_period_postings or curr_total < min_period_postings:
        return {
            "available": False,
            "status_message": (
                f"Insufficient postings per period: Previous period ({prev_period_label}) has {prev_total} jobs "
                f"and Current period ({curr_period_label}) has {curr_total} jobs "
                f"(minimum {min_period_postings} required in each period)."
            ),
            "coverage_warning": None,
            "trends_df": empty_df,
            "emerging_df": empty_df,
            "stable_df": empty_df,
            "declining_df": empty_df,
            "timeline_df": pd.DataFrame(),
            "period_info": meta,
        }

    prev_counts: Counter[str] = Counter()
    curr_counts: Counter[str] = Counter()
    bucket_skill_counts: dict[str, Counter[str]] = {str(b): Counter() for b in unique_buckets}
    bucket_totals: Counter[str] = Counter()

    for idx, (_, job_skills) in enumerate(records):
        b_label = str(buckets.iloc[idx])
        bucket_totals[b_label] += 1
        for skill in job_skills:
            bucket_skill_counts[b_label][skill] += 1
        if prev_mask[idx]:
            for skill in job_skills:
                prev_counts[skill] += 1
        elif curr_mask[idx]:
            for skill in job_skills:
                curr_counts[skill] += 1

    all_skills = sorted(set(prev_counts.keys()) | set(curr_counts.keys()))
    total_analyzed = prev_total + curr_total
    rows: list[dict[str, Any]] = []
    low_sample_skills = 0

    for skill in all_skills:
        p_cnt = prev_counts.get(skill, 0)
        c_cnt = curr_counts.get(skill, 0)
        res = classify_skill_trend(
            prev_count=p_cnt,
            curr_count=c_cnt,
            prev_total_jobs=prev_total,
            curr_total_jobs=curr_total,
            min_skill_postings=min_skill_postings,
            min_prev_for_decline=min_prev_for_decline,
            change_threshold_pct=change_threshold_pct,
            min_share_delta_pp=min_share_delta_pp,
        )
        if res["trend"] == "Insufficient sample":
            low_sample_skills += 1
            continue

        rows.append(
            {
                "Skill": skill,
                "Current Demand": f"{c_cnt} jobs ({res['current_share_pct']:.1f}%)",
                "Previous Demand": f"{p_cnt} jobs ({res['previous_share_pct']:.1f}%)",
                "Current Share (%)": res["current_share_pct"],
                "Previous Share (%)": res["previous_share_pct"],
                "Share Change (pp)": res["share_delta_pp"],
                "Percentage Change": res["pct_change_display"],
                "Percentage Change (%)": res["pct_change"],
                "Trend Classification": res["trend"],
                "Skill Postings": res["postings_with_skill"],
                "Total Postings Analyzed": total_analyzed,
            }
        )

    trends_df = pd.DataFrame(rows) if rows else empty_df.copy()
    if not trends_df.empty:
        trends_df = trends_df.sort_values(
            by=["Share Change (pp)", "Current Share (%)"],
            ascending=[False, False],
        ).reset_index(drop=True)

    emerging_df = trends_df[trends_df["Trend Classification"] == "Emerging"].reset_index(drop=True) if not trends_df.empty else empty_df.copy()
    stable_df = trends_df[trends_df["Trend Classification"] == "Stable"].reset_index(drop=True) if not trends_df.empty else empty_df.copy()
    declining_df = (
        trends_df[trends_df["Trend Classification"] == "Declining"]
        .sort_values(by=["Share Change (pp)", "Previous Share (%)"], ascending=[True, False])
        .reset_index(drop=True)
        if not trends_df.empty else empty_df.copy()
    )

    # Build period x skill share (%) timeline for top skills
    top_timeline_skills = (
        trends_df.sort_values("Skill Postings", ascending=False)["Skill"].head(10).tolist()
        if not trends_df.empty else []
    )
    timeline_rows = []
    for b_obj in unique_buckets:
        b_str = str(b_obj)
        b_tot = max(1, bucket_totals[b_str])
        row_dict: dict[str, Any] = {"Period": b_str}
        for sk in top_timeline_skills:
            row_dict[sk] = round(bucket_skill_counts[b_str].get(sk, 0) / b_tot * 100.0, 2)
        timeline_rows.append(row_dict)
    timeline_df = pd.DataFrame(timeline_rows).set_index("Period") if timeline_rows and top_timeline_skills else pd.DataFrame()

    span_days = int((timestamps.max() - timestamps.min()).total_seconds() // 86400) + 1
    coverage_warning = None
    if chosen_gran == "D" or span_days < 45:
        coverage_warning = (
            f"Limited date coverage notice: available dated postings span {span_days} day(s) "
            f"({timestamps.min().date()} to {timestamps.max().date()}). "
            "Results reflect short-term trends within the available dataset, not global or multi-month market shifts."
        )
    if meta["undated_historical_jobs"] > 0:
        extra_note = (
            f" {meta['undated_historical_jobs']:,} historical CSV postings have no timestamp column "
            "and were excluded from period comparison rather than inventing dates."
        )
        coverage_warning = (coverage_warning + extra_note) if coverage_warning else extra_note.strip()

    period_info = {
        **meta,
        "granularity": chosen_gran,
        "previous_period": prev_period_label,
        "current_period": curr_period_label,
        "previous_total_jobs": prev_total,
        "current_total_jobs": curr_total,
        "total_analyzed_jobs": total_analyzed,
        "low_sample_excluded": low_sample_skills,
        "span_days": span_days,
    }

    status_msg = (
        f"Compared Current Period ({curr_period_label}: {curr_total} postings) against "
        f"Previous Period ({prev_period_label}: {prev_total} postings) using normalized posting share (%). "
        f"Thresholds: ±{change_threshold_pct:.0f}% relative share change, ±{min_share_delta_pp:.1f} pp minimum share delta, "
        f"and ≥{min_skill_postings} postings per skill ({low_sample_skills} low-sample skills excluded)."
    )

    return {
        "available": True,
        "status_message": status_msg,
        "coverage_warning": coverage_warning,
        "trends_df": trends_df,
        "emerging_df": emerging_df,
        "stable_df": stable_df,
        "declining_df": declining_df,
        "timeline_df": timeline_df,
        "period_info": period_info,
    }


# ---------------------------------------------------------------------------
# Feature 2.2: Job Expiry & Freshness Tracker
# ---------------------------------------------------------------------------

def _is_explicitly_expired(
    status_val: Any,
    expiry_ts: pd.Timestamp | None,
    closed_ts: pd.Timestamp | None,
    ref_ts: pd.Timestamp,
) -> bool:
    """Return True ONLY when the source explicitly confirms a job is expired or closed."""
    if status_val is not None and not (isinstance(status_val, float) and pd.isna(status_val)):
        status_text = str(status_val).strip().lower()
        if status_text in EXPIRED_STATUS_VALUES:
            return True

    if closed_ts is not None and pd.notna(closed_ts) and closed_ts <= ref_ts:
        return True

    if expiry_ts is not None and pd.notna(expiry_ts) and expiry_ts <= ref_ts:
        return True

    return False


def classify_job_freshness(
    published_at: Any,
    reference_time: datetime | pd.Timestamp | str | None = None,
    fresh_days: int = DEFAULT_FRESH_DAYS,
    aging_days: int = DEFAULT_AGING_DAYS,
    status: Any = None,
    expiry_at: Any = None,
    closed_at: Any = None,
) -> dict[str, Any]:
    """Classify a single job's freshness without conflating stale age with confirmed expiry."""
    if fresh_days < 0:
        raise ValueError("fresh_days must be non-negative.")
    if aging_days < fresh_days:
        raise ValueError("aging_days must be greater than or equal to fresh_days.")

    if reference_time is None:
        ref_ts = pd.Timestamp(datetime.now(timezone.utc))
    else:
        ref_ts = pd.to_datetime(reference_time, utc=True, errors="coerce")
        if pd.isna(ref_ts):
            ref_ts = pd.Timestamp(datetime.now(timezone.utc))

    pub_series = parse_job_timestamps(pd.Series([published_at]))
    pub_ts = pub_series.iloc[0] if not pub_series.empty else pd.NaT

    exp_ts = parse_job_timestamps(pd.Series([expiry_at])).iloc[0] if expiry_at is not None else pd.NaT
    cls_ts = parse_job_timestamps(pd.Series([closed_at])).iloc[0] if closed_at is not None else pd.NaT

    explicit_expired = _is_explicitly_expired(status, exp_ts, cls_ts, ref_ts)

    if pd.isna(pub_ts):
        category = "Expired" if explicit_expired else "Missing / Invalid Date"
        return {
            "age_days": pd.NA,
            "freshness_category": category,
            "freshness_badge": FRESHNESS_BADGES[category],
            "published_date": "Missing / Invalid",
        }

    delta_days = float((ref_ts - pub_ts).total_seconds() / 86400.0)
    # Reject timestamps more than ~2 hours in the future as invalid/future dates
    if delta_days < -0.08:
        return {
            "age_days": pd.NA,
            "freshness_category": "Missing / Invalid Date",
            "freshness_badge": FRESHNESS_BADGES["Missing / Invalid Date"],
            "published_date": f"{pub_ts.strftime('%Y-%m-%d %H:%M UTC')} (Future)",
        }

    age_days = max(0.0, round(delta_days, 1))
    if explicit_expired:
        category = "Expired"
    elif age_days <= fresh_days:
        category = "Fresh"
    elif age_days <= aging_days:
        category = "Aging"
    else:
        category = "Stale / possibly outdated"

    return {
        "age_days": age_days,
        "freshness_category": category,
        "freshness_badge": FRESHNESS_BADGES[category],
        "published_date": pub_ts.strftime("%Y-%m-%d %H:%M UTC"),
    }


def evaluate_jobs_freshness(
    live_jobs: pd.DataFrame | None,
    historical_jobs: pd.DataFrame | None = None,
    fresh_days: int = DEFAULT_FRESH_DAYS,
    aging_days: int = DEFAULT_AGING_DAYS,
    reference_time: datetime | pd.Timestamp | str | None = None,
    source_mode: str = "Live only",
    freshness_filter: str = "All",
    keyword: str = "",
    sort_by: str = "newest",
    max_historical_rows: int = 1000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Evaluate job age and freshness categories across stored live and/or historical jobs.

    Preserves all older/stale postings in storage for historical analytics and
    never marks a job as ``Expired`` unless the source explicitly confirms expiry.
    """
    if fresh_days < 0:
        raise ValueError("fresh_days must be non-negative.")
    if aging_days < fresh_days:
        raise ValueError("aging_days must be greater than or equal to fresh_days.")

    if reference_time is None:
        ref_ts = pd.Timestamp(datetime.now(timezone.utc))
    else:
        ref_ts = pd.to_datetime(reference_time, utc=True, errors="coerce")
        if pd.isna(ref_ts):
            ref_ts = pd.Timestamp(datetime.now(timezone.utc))

    mode = str(source_mode or "Live only").strip().lower()
    include_live = mode in {"live", "live only", "both", "all"}
    include_hist = mode in {"historical", "historical only", "both", "all"}

    frames: list[pd.DataFrame] = []

    def _build_source_view(df: pd.DataFrame, default_source: str, limit_rows: int | None = None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        view = df.iloc[:limit_rows] if (limit_rows is not None and len(df) > limit_rows) else df
        n = len(view)
        idx = view.index

        date_col = find_date_column(view)
        pub_raw = view[date_col] if date_col is not None else pd.Series([None] * n, index=idx)
        pub_ts = parse_job_timestamps(pub_raw)

        status_col = "job_status" if "job_status" in view.columns else ("status" if "status" in view.columns else None)
        if status_col:
            explicit_expired = view[status_col].fillna("").astype(str).str.strip().str.lower().isin(EXPIRED_STATUS_VALUES).to_numpy()
        else:
            explicit_expired = np.zeros(n, dtype=bool)

        if "closed_time" in view.columns:
            cls_ts = parse_job_timestamps(view["closed_time"])
            explicit_expired |= (cls_ts.notna() & (cls_ts <= ref_ts)).to_numpy()

        if "expiry" in view.columns:
            exp_ts = parse_job_timestamps(view["expiry"])
            explicit_expired |= (exp_ts.notna() & (exp_ts <= ref_ts)).to_numpy()

        delta_days = ((ref_ts - pub_ts).dt.total_seconds() / 86400.0).to_numpy(dtype=float)
        valid_date = pub_ts.notna().to_numpy() & np.isfinite(delta_days) & (delta_days >= -0.08)
        age_vals = np.where(valid_date, np.maximum(0.0, np.round(delta_days, 1)), np.nan)

        categories = np.full(n, "Missing / Invalid Date", dtype=object)
        fresh_mask = valid_date & (~explicit_expired) & (age_vals <= fresh_days)
        aging_mask = valid_date & (~explicit_expired) & (age_vals > fresh_days) & (age_vals <= aging_days)
        stale_mask = valid_date & (~explicit_expired) & (age_vals > aging_days)

        categories[fresh_mask] = "Fresh"
        categories[aging_mask] = "Aging"
        categories[stale_mask] = "Stale / possibly outdated"
        categories[explicit_expired] = "Expired"

        pub_formatted = pub_ts.dt.strftime("%Y-%m-%d %H:%M UTC").fillna("Missing / Invalid").to_numpy(dtype=object)
        future_mask = pub_ts.notna().to_numpy() & (delta_days < -0.08)
        if future_mask.any():
            pub_formatted[future_mask] = "Invalid (Future Date)"

        if "last_updated_at" in view.columns:
            last_chk = parse_job_timestamps(view["last_updated_at"]).dt.strftime("%Y-%m-%d %H:%M UTC").fillna("Not recorded")
        elif "first_fetched_at" in view.columns:
            last_chk = parse_job_timestamps(view["first_fetched_at"]).dt.strftime("%Y-%m-%d %H:%M UTC").fillna("Not recorded")
        elif "fetched_at" in view.columns:
            last_chk = parse_job_timestamps(view["fetched_at"]).dt.strftime("%Y-%m-%d %H:%M UTC").fillna("Not recorded")
        else:
            last_chk = pd.Series(["Historical snapshot"] * n, index=idx)

        badges = [FRESHNESS_BADGES.get(cat, str(cat)) for cat in categories]
        ranks = [FRESHNESS_SORT_RANK.get(cat, 9) for cat in categories]

        return pd.DataFrame(
            {
                "job_id": view["job_id"].to_numpy() if "job_id" in view.columns else [f"{default_source}-{i}" for i in range(n)],
                "source": view["source"].to_numpy() if "source" in view.columns else [default_source] * n,
                "job_title": view["job_title"].fillna("Untitled").to_numpy() if "job_title" in view.columns else ["Untitled"] * n,
                "company": view["company"].fillna("Unknown").to_numpy() if "company" in view.columns else ["Unknown"] * n,
                "location": view["location"].fillna("Not specified").to_numpy() if "location" in view.columns else ["Not specified"] * n,
                "published_date": pub_formatted,
                "age_days": pd.Series(age_vals, dtype="Float64").to_numpy(),
                "freshness_category": categories,
                "freshness_badge": badges,
                "last_checked": last_chk.to_numpy(),
                "url": view["url"].fillna("").to_numpy() if "url" in view.columns else [""] * n,
                "_sort_rank": ranks,
                "_sort_age": np.where(np.isnan(age_vals), 1e12, age_vals),
            }
        )

    if include_live and live_jobs is not None and not live_jobs.empty:
        frames.append(_build_source_view(live_jobs, default_source="Arbeitnow"))

    undated_historical_extra = 0
    if include_hist and historical_jobs is not None and not historical_jobs.empty:
        hist_date_col = find_date_column(historical_jobs)
        frames.append(_build_source_view(historical_jobs, default_source="Historical CSV", limit_rows=max_historical_rows))
        if hist_date_col is None and len(historical_jobs) > max_historical_rows:
            undated_historical_extra = len(historical_jobs) - max_historical_rows

    if not frames:
        empty = pd.DataFrame(
            columns=[
                "job_id", "source", "job_title", "company", "location",
                "published_date", "age_days", "freshness_category",
                "freshness_badge", "last_checked", "url",
            ]
        )
        return empty, {
            "total": 0,
            "Fresh": 0,
            "Aging": 0,
            "Stale / possibly outdated": 0,
            "Expired": 0,
            "Missing / Invalid Date": 0,
            "undated_historical_omitted": 0,
        }

    combined = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    del frames

    # Summary counts before category/keyword display filters
    vc = combined["freshness_category"].value_counts()
    summary = {
        "total": int(len(combined)) + undated_historical_extra,
        "Fresh": int(vc.get("Fresh", 0)),
        "Aging": int(vc.get("Aging", 0)),
        "Stale / possibly outdated": int(vc.get("Stale / possibly outdated", 0)),
        "Expired": int(vc.get("Expired", 0)),
        "Missing / Invalid Date": int(vc.get("Missing / Invalid Date", 0)) + undated_historical_extra,
        "undated_historical_omitted": undated_historical_extra,
    }

    mask = np.ones(len(combined), dtype=bool)
    cat_q = str(freshness_filter or "All").strip()
    if cat_q and cat_q.lower() != "all":
        mask &= (combined["freshness_category"] == cat_q).to_numpy()

    kw_q = str(keyword or "").strip()
    if kw_q:
        search_col = (
            combined["job_title"].astype(str)
            + " "
            + combined["company"].astype(str)
            + " "
            + combined["location"].astype(str)
        )
        mask &= search_col.str.contains(kw_q, case=False, regex=False).to_numpy()

    filtered = combined if mask.all() else combined.loc[mask]

    sort_key = str(sort_by or "newest").strip().lower()
    if sort_key in {"freshness", "category"}:
        filtered = filtered.sort_values(by=["_sort_rank", "_sort_age", "job_title"], ascending=[True, True, True])
    elif sort_key in {"oldest", "age_desc"}:
        # Sort valid older postings first, keeping missing dates at the bottom
        valid_order = filtered["_sort_age"] >= 1e11
        filtered = filtered.assign(_missing=valid_order).sort_values(
            by=["_missing", "_sort_age", "job_title"], ascending=[True, False, True]
        ).drop(columns=["_missing"])
    else:
        filtered = filtered.sort_values(by=["_sort_age", "_sort_rank", "job_title"], ascending=[True, True, True])

    output_cols = [
        "job_id", "source", "job_title", "company", "location",
        "published_date", "age_days", "freshness_category",
        "freshness_badge", "last_checked", "url",
    ]
    return filtered[output_cols].reset_index(drop=True), summary
