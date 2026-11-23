import tempfile
import unittest
from pathlib import Path

from modules.live_storage import (
    add_application,
    load_applications,
    update_application_status,
)


class ApplicationDateTests(unittest.TestCase):

    def test_applied_date_is_recorded_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "applications.db"

            add_application(
                {
                    "job_id": "test-job-001",
                    "job_title": "Data Analyst",
                    "company": "Test Company",
                },
                db_path=db_path,
            )

            applications = load_applications(db_path=db_path)
            application_id = int(applications.iloc[0]["application_id"])

            self.assertEqual(applications.iloc[0]["status"], "Saved")
            self.assertIsNone(applications.iloc[0]["applied_at"])

            update_application_status(
                application_id, "Applied", db_path=db_path
            )

            applications = load_applications(db_path=db_path)
            applied_at = applications.iloc[0]["applied_at"]

            self.assertEqual(applications.iloc[0]["status"], "Applied")
            self.assertIsNotNone(applied_at)
            self.assertTrue(str(applied_at).strip())

            update_application_status(
                application_id, "Interview", db_path=db_path
            )

            applications = load_applications(db_path=db_path)

            self.assertEqual(applications.iloc[0]["status"], "Interview")
            self.assertEqual(applications.iloc[0]["applied_at"], applied_at)


if __name__ == "__main__":
    unittest.main()
