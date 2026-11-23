import pandas as pd
import numpy as np


def clean_data(df, inplace=False):
    # Avoid duplicating huge DataFrames twice; mutate in place when requested.
    if inplace:
        df.dropna(subset=['job_title'], inplace=True)
    else:
        df = df.loc[df['job_title'].notna()].copy()

    # Preserve records with missing company names
    if 'company' in df.columns:
        df['company'] = df['company'].fillna('Unknown')

    # Convert salaries to compact float32 numeric values
    if 'salary' in df.columns:
        salaries = pd.to_numeric(df['salary'], errors='coerce')
        df['salary'] = salaries.where(salaries > 0).astype('float32')

    # Do not fill missing salaries with the median.
    # Do not remove duplicates without a reliable job ID.

    return df
