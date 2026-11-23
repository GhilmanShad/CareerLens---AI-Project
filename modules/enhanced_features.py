
import re
import pandas as pd

from modules.decision_tree_features import extract_features


DESCRIPTION_KEYWORDS = [
    "entry level",
    "junior",
    "senior",
    "associate",
    "internship",
    "intern",
    "director",
    "executive",
    "years of experience",
    "leadership",
]


def extract_enhanced_features(df):
    # Preserve the nine original features.
    features = extract_features(df)

    descriptions = (
        df["description"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    # Add ten description-based binary features.
    for keyword in DESCRIPTION_KEYWORDS:
        pattern = (
            r"(?<!\w)"
            + re.escape(keyword)
            + r"(?!\w)"
        )

        features[f"description_{keyword.replace(' ', '_')}"] = (
            descriptions.str.contains(
                pattern,
                regex=True
            ).astype(int)
        )

    return features
