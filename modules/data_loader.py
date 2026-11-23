import pandas as pd
import os


def prepare_dataset(df):
    """Make common CSV column names compatible with the existing dashboard."""
    aliases = {
        "title": "job_title", "job_title": "job_title", "position": "job_title",
        "company_name": "company", "employer_name": "company", "organization": "company",
        "job_description": "description", "details": "description",
        "annual_salary": "salary", "salary_usd": "salary", "pay": "salary",
        "level": "job_level", "experience_level": "job_level",
        "published_at": "published_at", "posted_date": "published_at",
        "listed_time": "published_at", "original_listed_time": "published_at",
        "created_at": "published_at", "date": "published_at",
        "expiry": "expiry", "closed_time": "closed_time",
        "status": "job_status", "job_status": "job_status",
        "location": "location", "job_posting_url": "url", "url": "url",
    }
    renamed = {}
    reserved = set(df.columns)
    for column in df.columns:
        target = aliases.get(column.lower().strip())
        if target and target not in reserved:
            renamed[column] = target
            reserved.add(target)
    if renamed:
        df = df.rename(columns=renamed)
    if "job_title" not in df.columns:
        raise ValueError("CSV needs a job_title, title, or position column.")
    for column, default in {"company": "Unknown", "description": "", "salary": pd.NA,
                            "job_level": "Unknown"}.items():
        if column not in df.columns:
            df[column] = default
    # Compact numeric and low-cardinality string columns to reduce RAM footprint.
    if "salary" in df.columns:
        df["salary"] = pd.to_numeric(df["salary"], errors="coerce").astype("float32")
    if "job_level" in df.columns and df["job_level"].dtype == object:
        if df["job_level"].nunique(dropna=True) <= 50:
            df["job_level"] = df["job_level"].astype("category")
    return df


def load_and_inspect_data(filepath="data/job_postings.csv", nrows=None, usecols=None, verbose=False):
    if not os.path.exists(filepath):
        return "Error: Dataset file nahi mili. Please check the path."

    read_kwargs = {"low_memory": True}
    if nrows is not None:
        read_kwargs["nrows"] = nrows
    if usecols is not None:
        read_kwargs["usecols"] = usecols

    df = pd.read_csv(filepath, **read_kwargs)

    if verbose:
        print("--- Dataset Shape ---")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        print("\n--- Columns & Data Types ---")
        print(df.dtypes)
        print("\n--- Missing Values ---")
        print(df.isnull().sum())

    try:
        return prepare_dataset(df)
    except ValueError as exc:
        return str(exc)


if __name__ == "__main__":
    df = load_and_inspect_data(verbose=True)
