import unittest
import pandas as pd

from modules.career_insights import evaluate_jobs_freshness


class TestFreshnessSorting(unittest.TestCase):

    def test_precise_chronological_sorting(self):
        reference = "2026-11-23T12:00:00Z"

        jobs = pd.DataFrame([
            {
                "job_id": "older",
                "job_title": "Z Older Job",
                "published_at": "2026-11-23T10:00:00Z",
            },
            {
                "job_id": "newer",
                "job_title": "A Newer Job",
                "published_at": "2026-11-23T10:10:00Z",
            },
            {
                "job_id": "undated",
                "job_title": "Missing Date Job",
                "published_at": None,
            },
        ])

        for sort_mode, expected in [
            ("newest", ["newer", "older", "undated"]),
            ("oldest", ["older", "newer", "undated"]),
        ]:
            with self.subTest(sort_mode=sort_mode):
                result, _ = evaluate_jobs_freshness(
                    jobs,
                    reference_time=reference,
                    sort_by=sort_mode,
                )

                self.assertEqual(
                    result["job_id"].tolist(),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
