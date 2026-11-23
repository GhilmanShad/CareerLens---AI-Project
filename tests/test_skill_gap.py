import unittest

import pandas as pd

from modules.career_insights import skill_gap


class TestSkillGap(unittest.TestCase):

    def setUp(self):
        self.jobs = pd.DataFrame([
            {
                "job_title": "Data Analyst",
                "skills": "Python | SQL | Statistics | Power BI",
            },
            {
                "job_title": "Senior Data Analyst",
                "skills": "Python | SQL | Tableau",
            },
            {
                "job_title": "Backend Developer",
                "skills": "Java | Docker | Git",
            },
        ])

    def test_matches_existing_skills(self):
        matched, missing = skill_gap(
            "Python, SQL", self.jobs, "Data Analyst"
        )
        self.assertIn("Python", matched)
        self.assertIn("SQL", matched)
        self.assertNotIn("Python", missing)

    def test_identifies_missing_skills(self):
        matched, missing = skill_gap(
            "Python, SQL", self.jobs, "Data Analyst"
        )
        self.assertIn("Statistics", missing)
        self.assertIn("Power BI", missing)
        self.assertIn("Tableau", missing)

    def test_filters_by_target_role(self):
        matched, missing = skill_gap(
            "Python", self.jobs, "Data Analyst"
        )
        self.assertNotIn("Java", matched + missing)
        self.assertNotIn("Docker", matched + missing)

    def test_no_matching_jobs(self):
        matched, missing = skill_gap(
            "Python", self.jobs, "Cybersecurity Analyst"
        )
        self.assertEqual(matched, [])
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
