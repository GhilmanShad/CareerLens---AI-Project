import pandas as pd


def clean_data(df, inplace=False, remove_duplicates=False):
    # Work on a copy unless in-place modification is requested.
    if not inplace:
        df = df.copy()

    # Remove records without a valid job title.
    df.dropna(subset=["job_title"], inplace=True)

    # Fill missing company names.
    if "company" in df.columns:
        df["company"] = df["company"].fillna("Unknown")

    # Convert salaries to numeric values.
    # Preserve missing salaries instead of inventing values.
    if "salary" in df.columns:
        salaries = pd.to_numeric(df["salary"], errors="coerce")
        df["salary"] = salaries.where(salaries > 0).astype("float32")

    # Optionally remove exact duplicate records.
    # This does not deduplicate by job title or company alone.
    if remove_duplicates:
        df.drop_duplicates(inplace=True)

    return df
