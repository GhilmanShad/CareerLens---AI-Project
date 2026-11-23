import re

import numpy as np
import pandas as pd

STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have how i if in into is it its
    of on or our that the their them they this to was were what when where
    which who will with you your we us not no can will job role work working
    company team teams new all any more most other than then there these
    those very also about over under using use used within across per
    """.split()
)

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9+#.]*")
SKILL_SPLIT_PATTERN = re.compile(r"\s*[,;|]\s*")


def standardize_series(series):
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)

    if not np.isfinite(std) or std == 0:
        return values - mean

    return (values - mean) / std


def get_knn_recommendations(df, query_salary, k=3):
    if "salary" not in df.columns:
        return pd.DataFrame()

    if not np.isfinite(query_salary) or query_salary <= 0:
        return pd.DataFrame()

    if not isinstance(k, (int, np.integer)) or k < 1:
        return pd.DataFrame()

    salaries = pd.to_numeric(df["salary"], errors="coerce")
    valid_mask = np.isfinite(salaries) & (salaries > 0)
    valid_salaries = salaries.loc[valid_mask]

    if valid_salaries.empty:
        return pd.DataFrame(columns=df.columns)

    mean = valid_salaries.mean()
    std = valid_salaries.std(ddof=0)

    if not np.isfinite(std) or std == 0:
        std = 1.0

    # Manual one-dimensional Euclidean distance (slice only top-K rows to save RAM)
    query_scaled = (query_salary - mean) / std
    distances = np.abs(((valid_salaries - mean) / std) - query_scaled)
    top_distances = distances.nsmallest(min(k, len(distances)))

    result = df.loc[top_distances.index].copy()
    result["distance"] = top_distances
    return result


def predict_job_level(nearest_jobs):
    if "job_level" not in nearest_jobs.columns:
        return None

    labels = nearest_jobs["job_level"].dropna()
    labels = labels.astype(str).str.strip()
    labels = labels[labels != ""]

    if labels.empty:
        return None

    # Majority vote; alphabetical tie-breaking
    counts = labels.value_counts()
    winners = counts[counts == counts.max()].index.tolist()

    return sorted(winners)[0]


# ---------------------------------------------------------------------------
# Text KNN (TF-IDF + cosine similarity) for live job recommendations
# ---------------------------------------------------------------------------
# The historical salary KNN above is kept unchanged. The functions below extend
# the same from-scratch KNN module with a text-based neighbour search:
#
# 1. Build a TF-IDF matrix for all candidate jobs (title, tags, skills,
#    description) using a manual tokenizer and IDF calculation.
# 2. Transform the user profile into the same vocabulary space.
# 3. Rank jobs by cosine similarity (L2-normalized dot product) and take top K.
#
# No scikit-learn is used, preserving the academic "implement it yourself"
# requirement of the project.

class SparseTFIDFMatrix:
    """Lightweight pure-NumPy sparse TF-IDF matrix (COO float32/int32).

    Avoids dense (n_docs x n_vocab) float64 allocations that spike RAM by
    hundreds of megabytes on a few thousand job descriptions.
    """

    def __init__(self, rows, cols, data, shape):
        self.rows = rows
        self.cols = cols
        self.data = data
        self.shape = shape

    @property
    def size(self):
        return int(self.data.size)

    def __matmul__(self, vector):
        n_rows = self.shape[0]
        if n_rows == 0 or self.data.size == 0 or vector.size == 0:
            return np.zeros(n_rows, dtype=np.float32)
        weights = self.data * vector[self.cols]
        return np.bincount(self.rows, weights=weights, minlength=n_rows).astype(np.float32)

    def toarray(self):
        dense = np.zeros(self.shape, dtype=np.float32)
        if self.data.size:
            dense[self.rows, self.cols] = self.data
        return dense


def tokenize_text(text):
    """Lowercase tokenization used by the TF-IDF vectorizer."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []

    tokens = TOKEN_PATTERN.findall(str(text).lower())
    return [
        token for token in tokens
        if len(token) >= 2 and token not in STOPWORDS
    ]


def build_tfidf(documents):
    """Build a sparse L2-normalized TF-IDF matrix from an iterable of documents.

    Returns ``(matrix, vocabulary, idf)`` where ``matrix`` is a
    :class:`SparseTFIDFMatrix` supporting ``matrix @ query_vector``.
    The IDF formula is the smoothed ``log((1 + n) / (1 + df)) + 1``.
    """
    vocab_index = {}
    doc_freq = []
    rows_list = []
    cols_list = []
    tf_list = []
    n_docs = 0

    for row_index, document in enumerate(documents):
        n_docs += 1
        tokens = tokenize_text(document)
        if not tokens:
            continue
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        inv_total = 1.0 / len(tokens)
        for token, count in counts.items():
            col = vocab_index.get(token)
            if col is None:
                col = len(vocab_index)
                vocab_index[token] = col
                doc_freq.append(1)
            else:
                doc_freq[col] += 1
            rows_list.append(row_index)
            cols_list.append(col)
            tf_list.append(count * inv_total)

    if not vocab_index or n_docs == 0:
        empty = SparseTFIDFMatrix(
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            (n_docs, 0),
        )
        return empty, {}, np.zeros(0, dtype=np.float32)

    df_arr = np.asarray(doc_freq, dtype=np.float32)
    idf = (np.log((1.0 + n_docs) / (1.0 + df_arr)) + 1.0).astype(np.float32)

    rows = np.asarray(rows_list, dtype=np.int32)
    cols = np.asarray(cols_list, dtype=np.int32)
    data = (np.asarray(tf_list, dtype=np.float32) * idf[cols]).astype(np.float32)
    del rows_list, cols_list, tf_list, doc_freq, df_arr

    row_sq = np.bincount(rows, weights=data * data, minlength=n_docs)
    norms = np.sqrt(row_sq).astype(np.float32)
    norms[norms == 0] = 1.0
    data /= norms[rows]

    matrix = SparseTFIDFMatrix(rows, cols, data, (n_docs, len(vocab_index)))
    return matrix, vocab_index, idf


def transform_profile_text(text, vocabulary, idf):
    """Vectorize a user profile using an existing TF-IDF vocabulary."""
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    tokens = tokenize_text(text)
    if not tokens or not vocabulary:
        return vector

    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = float(len(tokens))

    for token, count in counts.items():
        index = vocabulary.get(token)
        if index is not None:
            vector[index] = (count / total) * idf[index]

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _split_skill_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [
        item.strip() for item in SKILL_SPLIT_PATTERN.split(str(value))
        if item.strip()
    ]


def _empty_recommendations():
    return pd.DataFrame(
        columns=[
            "source", "job_title", "company", "location", "remote_label",
            "skills", "published_at", "url", "match_score", "matched_skills",
            "missing_skills", "match_reason",
        ]
    )


def _iter_job_documents(candidates):
    """Yield weighted search text per job without calling iterrows()."""
    n = len(candidates)
    titles = candidates["job_title"].fillna("").astype(str).tolist() if "job_title" in candidates.columns else [""] * n
    skills = candidates["skills"].fillna("").astype(str).tolist() if "skills" in candidates.columns else [""] * n
    tags = candidates["tags"].fillna("").astype(str).tolist() if "tags" in candidates.columns else [""] * n
    if "description_text" in candidates.columns:
        descriptions = candidates["description_text"].fillna("").astype(str).tolist()
    elif "description" in candidates.columns:
        descriptions = candidates["description"].fillna("").astype(str).tolist()
    else:
        descriptions = [""] * n

    for title, skill, tag, desc in zip(titles, skills, tags, descriptions):
        # Title and skills carry more weight; cap description length to keep
        # sparse tokenization fast and bounded on very long postings.
        yield f"{title} {title} {skill} {skill} {tag} {desc[:2500]}"


def recommend_live_jobs(
    jobs,
    skills="",
    preferred_title="",
    experience_level="",
    location="",
    employment_type="",
    remote_preference="Any",
    k=5,
    extra_profile_text="",
):
    """Return the top-K jobs for a user profile using sparse TF-IDF KNN.

    Hard preference filters (location, employment type, remote) are applied
    before the neighbour search without copying the full DataFrame. The match
    score is cosine similarity between the TF-IDF profile vector and each
    job's TF-IDF vector, expressed as a percentage (relevance score, not a
    hiring probability).
    """
    if jobs is None or jobs.empty:
        return _empty_recommendations()

    try:
        k = int(k)
    except (TypeError, ValueError):
        return _empty_recommendations()
    if k < 1:
        return _empty_recommendations()

    mask = np.ones(len(jobs), dtype=bool)

    location_query = str(location or "").strip()
    if location_query and "location" in jobs.columns:
        mask &= jobs["location"].fillna("").str.contains(
            location_query, case=False, regex=False
        ).to_numpy()

    employment_query = str(employment_type or "").strip()
    if employment_query and "employment_types" in jobs.columns:
        mask &= jobs["employment_types"].fillna("").str.contains(
            employment_query, case=False, regex=False
        ).to_numpy()

    remote_query = str(remote_preference or "Any").strip().lower()
    if remote_query == "remote" and "remote" in jobs.columns:
        mask &= jobs["remote"].fillna(False).astype(bool).to_numpy()
    elif remote_query in {"on-site", "onsite", "office"} and "remote" in jobs.columns:
        mask &= ~jobs["remote"].fillna(False).astype(bool).to_numpy()

    if not mask.any():
        return _empty_recommendations()

    candidates = jobs if mask.all() else jobs.loc[mask]

    # Compose the profile: skills and target title are repeated to emphasize
    # them, while softer preferences contribute less weight.
    profile_text = " ".join(
        part for part in [
            str(skills or ""), str(skills or ""), str(skills or ""),
            str(preferred_title or ""), str(preferred_title or ""),
            str(preferred_title or ""),
            str(experience_level or ""),
            str(location_query),
            str(employment_query),
            str(extra_profile_text or "")[:1500],
        ]
        if part.strip()
    )

    if not tokenize_text(profile_text):
        return _empty_recommendations()

    matrix, vocabulary, idf = build_tfidf(_iter_job_documents(candidates))
    if not vocabulary:
        return _empty_recommendations()

    query_vector = transform_profile_text(profile_text, vocabulary, idf)
    scores = matrix @ query_vector
    del matrix

    top_n = min(k, len(candidates))
    if top_n < len(scores):
        top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
        order = top_idx[np.argsort(-scores[top_idx])]
    else:
        order = np.argsort(-scores)

    profile_skills = _split_skill_list(skills)
    profile_skills_lower = {s.lower() for s in profile_skills}
    profile_title_tokens = set(tokenize_text(preferred_title))

    # Lazy import only if any top-K row lacks pre-extracted skills (e.g., historical rows)
    extract_fn = None

    results = []
    for position in order:
        score = float(scores[position])
        if score <= 0:
            continue

        row = candidates.iloc[int(position)]
        job_skills = _split_skill_list(row.get("skills"))
        if not job_skills:
            if extract_fn is None:
                from modules.live_preprocessing import extract_skills as extract_fn
            desc_val = row.get("description_text") or row.get("description") or ""
            job_skills = extract_fn(str(row.get("job_title", "") or ""), str(desc_val))

        job_tags = _split_skill_list(row.get("tags"))

        matched_skills = [
            skill for skill in job_skills
            if skill.lower() in profile_skills_lower
            or any(
                ps.lower() == skill.lower() or ps.lower() in skill.lower()
                for ps in profile_skills
            )
        ]
        matched_lower = {s.lower() for s in matched_skills}
        missing_skills = [
            skill for skill in job_skills
            if skill.lower() not in matched_lower
        ]

        title_tokens = set(tokenize_text(row.get("job_title")))
        title_overlap = sorted(profile_title_tokens & title_tokens)
        if matched_skills and title_overlap:
            reason = "Matching skills and job title"
        elif matched_skills:
            reason = "Matching skills"
        elif title_overlap:
            reason = "Related job title"
        else:
            reason = "Similar job description text"

        raw_source = str(row.get("source", "") or row.get("data_type", "") or "").strip()
        if raw_source.lower() == "historical":
            source_label = "Historical"
        elif raw_source:
            source_label = "Live" if raw_source.lower() in {"live", "arbeitnow"} else raw_source
        else:
            source_label = "Historical" if str(row.get("remote_label", "")).lower() == "historical" else "Live"

        results.append(
            {
                "source": source_label,
                "job_title": row.get("job_title", ""),
                "company": row.get("company", ""),
                "location": row.get("location", ""),
                "remote_label": row.get("remote_label", ""),
                "skills": " | ".join(job_skills),
                "published_at": row.get("published_at"),
                "url": row.get("url", ""),
                "match_score": round(score * 100, 1),
                "matched_skills": " | ".join(matched_skills),
                "missing_skills": " | ".join(missing_skills),
                "match_reason": reason,
                "job_tags": " | ".join(job_tags),
            }
        )

    if not results:
        return _empty_recommendations()

    return pd.DataFrame(results)
