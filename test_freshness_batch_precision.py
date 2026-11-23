import unittest

import pandas as pd

from modules.career_insights import evaluate_jobs_freshness


class TestFreshnessBatchPrecision(unittest.TestCase):

    def test_batch_classification_uses_precise_age(self):
        reference = "2026-11-23T12:00:00Z"

        jobs = pd.DataFrame([
            {
                "job_id": "j1",
                "job_title": "Exactly 7 Days",
                "published_at": "2026-11-16T12:00:00Z",
            },
            {
                "job_id": "j2",
                "job_title": "7 Days Plus 10 Minutes",
                "published_at": "2026-11-16T11:50:00Z",
            },
            {
                "job_id": "j3",
                "job_title": "Exactly 30 Days",
                "published_at": "2026-10-24T12:00:00Z",
            },
            {
                "job_id": "j4",
                "job_title": "30 Days Plus 10 Minutes",
                "published_at": "2026-10-24T11:50:00Z",
            },
        ])

        result, summary = evaluate_jobs_freshness(
            jobs,
            reference_time=reference,
            fresh_days=7,
            aging_days=30,
        )

        categories = result.set_index("job_id")["freshness_category"]

        self.assertEqual(categories["j1"], "Fresh")
        self.assertEqual(categories["j2"], "Aging")
        self.assertEqual(categories["j3"], "Aging")
        self.assertEqual(
            categories["j4"],
            "Stale / possibly outdated",
        )

        self.assertEqual(summary["Fresh"], 1)
        self.assertEqual(summary["Aging"], 2)
        self.assertEqual(summary["Stale / possibly outdated"], 1)


if __name__ == "__main__":
    unittest.main()
