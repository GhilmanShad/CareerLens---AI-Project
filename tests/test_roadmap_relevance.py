import unittest

import pandas as pd

from modules.resume_tools import build_learning_roadmap


class RoadmapRelevanceTests(unittest.TestCase):

    def test_data_analyst_baseline_takes_priority(self):
        jobs = pd.DataFrame({
            "job_title": ["Data Analyst"] * 3,
            "skills": [
                "Python | SQL | Go | .NET",
                "Python | SQL | Go | C#",
                "Python | SQL | Statistics | Communication",
            ],
        })

        result = build_learning_roadmap(
            ["Python", "SQL"],
            jobs,
            "Data Analyst",
        )

        roadmap = result["roadmap_df"]
        skills = roadmap["Skill"].tolist()

        # Essential Data Analyst skills must be included.
        self.assertIn("Excel", skills)
        self.assertIn("Pandas", skills)
        self.assertIn("Power BI", skills)

        # Unrelated technologies must not become core requirements.
        for skill in ("Go", ".NET", "C#"):
            if skill in skills:
                row = roadmap.loc[roadmap["Skill"] == skill].iloc[0]
                self.assertEqual(row["Priority"], "Optional")

        # Readiness should reflect the recognized role baseline.
        self.assertAlmostEqual(result["readiness_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()
