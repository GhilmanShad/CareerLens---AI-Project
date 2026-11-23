import unittest

from modules.career_insights import classify_job_freshness


class TestFreshnessPrecision(unittest.TestCase):

    def test_exact_freshness_boundaries(self):
        reference = "2026-11-23T12:00:00Z"

        cases = [
            ("2026-11-16T12:00:00Z", "Fresh"),
            ("2026-11-16T11:50:00Z", "Aging"),
            ("2026-10-24T12:00:00Z", "Aging"),
            ("2026-10-24T11:50:00Z", "Stale / possibly outdated"),
        ]

        for published, expected in cases:
            with self.subTest(published=published):
                result = classify_job_freshness(
                    published,
                    reference_time=reference,
                    fresh_days=7,
                    aging_days=30,
                )
                self.assertEqual(
                    result["freshness_category"],
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
