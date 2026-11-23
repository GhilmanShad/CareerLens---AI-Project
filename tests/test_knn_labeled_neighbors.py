import unittest

import pandas as pd

from modules.knn import get_knn_recommendations, predict_job_level


class KNNLabeledNeighborsTests(unittest.TestCase):

    def test_prediction_uses_labeled_neighbors(self):
        df = pd.DataFrame({
            "salary": [80000, 80000, 80000, 81000, 82000, 83000],
            "job_level": [
                None, None, None,
                "Entry level", "Entry level", "Mid-Senior level"
            ],
        })

        recommendations = get_knn_recommendations(df, 80000, k=3)

        levels = df["job_level"].astype("string").str.strip()
        valid_levels = (
            levels.notna()
            & levels.ne("")
            & ~levels.str.lower().isin(
                ["none", "nan", "null", "unknown"]
            )
        ).fillna(False)

        labeled_neighbors = get_knn_recommendations(
            df.loc[valid_levels], 80000, k=3
        )

        prediction = predict_job_level(labeled_neighbors)

        self.assertEqual(len(recommendations), 3)
        self.assertTrue(recommendations["job_level"].isna().all())
        self.assertEqual(len(labeled_neighbors), 3)
        self.assertEqual(prediction, "Entry level")


if __name__ == "__main__":
    unittest.main()
