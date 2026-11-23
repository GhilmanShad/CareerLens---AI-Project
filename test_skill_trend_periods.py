import unittest

import pandas as pd

from modules.career_insights import analyze_skill_trends


class TestSkillTrendPeriodSelection(unittest.TestCase):

    def test_auto_avoids_incomplete_calendar_weeks(self):
        rows = []

        for day, count in [
            ("2026-11-20", 30),
            ("2026-11-21", 30),
            ("2026-11-22", 31),
            ("2026-11-23", 209),
        ]:
            rows.extend(
                {
                    "job_title": "Data Analyst",
                    "published_at": f"{day}T12:00:00Z",
                    "skills": "Python | SQL",
                }
                for _ in range(count)
            )

        jobs = pd.DataFrame(rows)

        auto_result = analyze_skill_trends(
            jobs,
            period_mode="auto",
        )

        weekly_result = analyze_skill_trends(
            jobs,
            period_mode="week",
        )

        self.assertTrue(auto_result["available"])
        self.assertEqual(
            auto_result["period_info"]["granularity"], "D"
        )

        self.assertTrue(weekly_result["available"])
        self.assertEqual(
            weekly_result["period_info"]["granularity"], "W"
        )


if __name__ == "__main__":
    unittest.main()
