import io
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from modules.career_insights import (
    analyze_skill_trends,
    classify_job_freshness,
    classify_skill_trend,
    evaluate_jobs_freshness,
    skill_demand,
    skill_gap,
)
from modules.data_loader import prepare_dataset
from modules.decision_tree import build_tree, predict
from modules.decision_tree_features import extract_features
from modules.graph_search import CAREER_GRAPH, a_star, bfs, calculate_path_cost, dfs
from modules.kmeans import run_kmeans
from modules.knn import (
    SparseTFIDFMatrix,
    build_tfidf,
    get_knn_recommendations,
    predict_job_level,
    recommend_live_jobs,
)
from modules.live_preprocessing import normalize_live_jobs
from modules.live_storage import (
    add_application,
    load_applications,
    load_live_jobs,
    update_application_status,
    upsert_live_jobs,
)
from modules.outlier_detection import detect_iqr_outliers, detect_zscore_outliers
from modules.preprocessing import clean_data
from modules.resume_tools import (
    CAREER_TOOLS_LIVE_COLUMNS,
    build_learning_roadmap,
    compare_career_roles,
    demand_summary,
    extract_resume_text,
    forecast_demand,
    matching_jobs,
    resume_skills,
    roadmap,
)


RAW_JOB = {
    "slug": "python-role",
    "company_name": "Example",
    "title": "Python Developer (m/w/d)",
    "description": "<p>Python, SQL and Docker required.</p>",
    "remote": True,
    "url": "https://example.test/job",
    "tags": ["Python", "Backend"],
    "job_types": ["Full Time"],
    "location": "Berlin",
    "created_at": 1700000000,
}


class LiveFeaturesTests(unittest.TestCase):
    def test_normalization_storage_and_deduplication(self):
        jobs = normalize_live_jobs(pd.DataFrame([RAW_JOB]))
        self.assertEqual(jobs.iloc[0]["job_title"], "Python Developer")
        self.assertIn("Python", jobs.iloc[0]["skills"])
        with tempfile.TemporaryDirectory() as folder:
            path = f"{folder}/live.db"
            self.assertEqual(upsert_live_jobs(jobs, path)["inserted"], 1)
            first_seen = load_live_jobs(path).iloc[0]["last_updated_at"]
            self.assertEqual(upsert_live_jobs(jobs, path)["skipped"], 1)
            second_seen = load_live_jobs(path).iloc[0]["last_updated_at"]
            self.assertGreaterEqual(second_seen, first_seen)
            self.assertEqual(len(load_live_jobs(path)), 1)
            selective = load_live_jobs(path, columns=CAREER_TOOLS_LIVE_COLUMNS)
            self.assertNotIn("description", selective.columns)
            self.assertIn("description_text", selective.columns)

    def test_sparse_tfidf_and_recommendation(self):
        matrix, vocab, idf = build_tfidf(["Python SQL Docker", "Java Spring SQL"])
        self.assertIsInstance(matrix, SparseTFIDFMatrix)
        self.assertGreater(len(vocab), 0)
        self.assertEqual(len(idf), len(vocab))

        jobs = normalize_live_jobs(pd.DataFrame([RAW_JOB]))
        recs = recommend_live_jobs(jobs, skills="Python")
        self.assertFalse(recs.empty)
        self.assertIn("Python", recs.iloc[0]["matched_skills"])
        self.assertIn("SQL", recs.iloc[0]["missing_skills"])
        self.assertTrue(recommend_live_jobs(jobs).empty)

    def test_skill_gap_and_application_tracker(self):
        jobs = normalize_live_jobs(pd.DataFrame([RAW_JOB]))
        matched, missing = skill_gap("Python", jobs, "Developer")
        self.assertIn("Python", matched)
        self.assertIn("SQL", missing)
        self.assertFalse(skill_demand(jobs).empty)
        with tempfile.TemporaryDirectory() as folder:
            path = f"{folder}/applications.db"
            add_application({"job_title": "Python Developer", "company": "Example"}, path)
            application = load_applications(path).iloc[0]
            update_application_status(int(application["application_id"]), "Applied", path)
            self.assertEqual(load_applications(path).iloc[0]["status"], "Applied")

    def test_forecast_demand_insufficient_and_sufficient_periods(self):
        single_job = normalize_live_jobs(pd.DataFrame([RAW_JOB]))
        trend_short = forecast_demand(single_job, granularity="M")
        self.assertIn("actual_jobs", trend_short.columns)
        self.assertFalse(trend_short.attrs.get("forecast_reliable"))
        self.assertTrue(trend_short["forecast_jobs"].isna().all())

        multi_jobs = pd.DataFrame({
            "job_title": ["Data Analyst"] * 10,
            "skills": ["Python | SQL"] * 10,
            "published_at": [
                "2026-01-10T00:00:00Z",
                "2026-02-10T00:00:00Z", "2026-02-15T00:00:00Z",
                "2026-03-10T00:00:00Z", "2026-03-12T00:00:00Z", "2026-03-18T00:00:00Z",
                "2026-04-05T00:00:00Z", "2026-04-10T00:00:00Z", "2026-04-15T00:00:00Z", "2026-04-20T00:00:00Z",
            ],
        })
        trend_full = forecast_demand(multi_jobs, role_filter="Data Analyst", granularity="M", horizon=2)
        self.assertTrue(trend_full.attrs.get("forecast_reliable"))
        actual_rows = trend_full[trend_full["record_type"] == "Actual"]
        forecast_rows = trend_full[trend_full["record_type"] == "Forecast"]
        self.assertEqual(len(actual_rows), 4)
        self.assertEqual(len(forecast_rows), 2)
        self.assertTrue(forecast_rows["forecast_jobs"].notna().all())

        summary = demand_summary(multi_jobs)
        self.assertFalse(summary["titles"].empty)
        self.assertFalse(summary["categories"].empty)

    def test_resume_roadmap_and_matching_jobs(self):
        jobs = normalize_live_jobs(pd.DataFrame([RAW_JOB]))
        self.assertIn("Python", resume_skills("Python and SQL developer"))
        matched, missing = roadmap("Python", jobs, "Developer")
        self.assertIn("Python", matched)
        self.assertIn("SQL", missing)

        detailed = build_learning_roadmap("Python", jobs, "Data Scientist")
        self.assertIn("Python", detailed["matched_skills"])
        self.assertFalse(detailed["roadmap_df"].empty)
        self.assertIn("Priority", detailed["roadmap_df"].columns)

        historical = pd.DataFrame({
            "job_title": ["Python Data Analyst"],
            "company": ["Past Co"],
            "description": ["Python SQL Tableau reporting and statistics"],
        })
        matches = matching_jobs("Python SQL", jobs, historical, k=5, target_role="Data Analyst", source_mode="Both")
        self.assertFalse(matches.empty)
        self.assertIn("missing_skills", matches.columns)
        self.assertIn("source", matches.columns)
        self.assertEqual(set(matches["source"].unique()), {"Live", "Historical"})

    def test_role_comparison_and_salary_reliability(self):
        live = normalize_live_jobs(pd.DataFrame([RAW_JOB]))
        hist = pd.DataFrame({
            "job_title": ["Data Analyst"] * 6 + ["Data Scientist"] * 2,
            "company": ["Co"] * 8,
            "salary": [70000, 75000, 80000, 82000, 85000, 90000, 120000, 130000],
            "description": ["Python SQL Excel Power BI"] * 6 + ["Python Machine Learning SQL"] * 2,
        })
        comp_df, _ = compare_career_roles(
            ["Data Analyst", "Data Scientist"],
            live,
            historical_jobs=hist,
            resume_skills_list=["Python", "SQL"],
        )
        self.assertEqual(len(comp_df), 2)
        da_row = comp_df[comp_df["Role"] == "Data Analyst"].iloc[0]
        ds_row = comp_df[comp_df["Role"] == "Data Scientist"].iloc[0]
        self.assertIn("$", da_row["Median Salary"])
        self.assertEqual(ds_row["Median Salary"], "Insufficient reliable data")
        self.assertIn("%", da_row["Resume Overlap"])

    def test_docx_and_pdf_resume_extraction(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.writestr("word/document.xml", '<w:document xmlns:w="w"><w:t>Python SQL</w:t></w:document>')

        class DocxUpload:
            name = "resume.docx"
            def getvalue(self):
                return content.getvalue()

        self.assertEqual(extract_resume_text(DocxUpload()), "Python SQL")

        class TooLargeUpload:
            name = "huge.pdf"
            def getvalue(self):
                return b"x" * (5 * 1024 * 1024 + 10)

        with self.assertRaises(ValueError):
            extract_resume_text(TooLargeUpload())

    def test_emerging_vs_declining_skills_and_normalization(self):
        # 1. Verify volume normalization: if total jobs drop from 20 to 10 and skill count
        # drops from 10 (50%) to 5 (50%), it must be classified as Stable (0% share change), NOT Declining!
        norm_check = classify_skill_trend(prev_count=10, curr_count=5, prev_total_jobs=20, curr_total_jobs=10)
        self.assertEqual(norm_check["trend"], "Stable")
        self.assertEqual(norm_check["pct_change"], 0.0)

        # 2. Verify low-sample protection and insufficient previous coverage protection
        low_sample = classify_skill_trend(prev_count=1, curr_count=1, prev_total_jobs=10, curr_total_jobs=10, min_skill_postings=3)
        self.assertEqual(low_sample["trend"], "Insufficient sample")

        # Only 1 mention in previous period -> must NOT be labeled Declining even if 0 in current
        no_false_decline = classify_skill_trend(
            prev_count=1, curr_count=2, prev_total_jobs=5, curr_total_jobs=20,
            min_skill_postings=3, min_prev_for_decline=2,
        )
        self.assertNotEqual(no_false_decline["trend"], "Declining")

        # 3. Synthetic multi-month dataset with clear Emerging, Stable, and Declining skills
        prev_rows = [
            {"job_title": "Data Engineer", "location": "Berlin", "published_at": "2026-01-15T10:00:00Z", "skills": "Python | SQL | PHP"}
            for _ in range(8)
        ] + [
            {"job_title": "Data Engineer", "location": "Berlin", "published_at": "2026-01-20T10:00:00Z", "skills": "Python | Docker"}
            for _ in range(2)
        ]
        curr_rows = [
            {"job_title": "Data Engineer", "location": "Berlin", "published_at": "2026-02-15T10:00:00Z", "skills": "Python | Docker | Kubernetes"}
            for _ in range(8)
        ] + [
            {"job_title": "Data Engineer", "location": "Berlin", "published_at": "2026-02-20T10:00:00Z", "skills": "Python | SQL"}
            for _ in range(2)
        ]
        synthetic_df = pd.DataFrame(prev_rows + curr_rows)
        res = analyze_skill_trends(synthetic_df, period_mode="month", min_skill_postings=3)
        self.assertTrue(res["available"])
        trends = res["trends_df"].set_index("Skill")["Trend Classification"].to_dict()
        self.assertEqual(trends.get("Docker"), "Emerging")
        self.assertEqual(trends.get("Kubernetes"), "Emerging")
        self.assertEqual(trends.get("Python"), "Stable")
        self.assertEqual(trends.get("SQL"), "Declining")
        self.assertEqual(trends.get("PHP"), "Declining")
        self.assertFalse(res["timeline_df"].empty)

        # 4. Insufficient historical dates -> unavailable with honest message
        undated_df = pd.DataFrame({"job_title": ["Analyst"] * 10, "description": ["Python SQL"] * 10})
        unavailable_res = analyze_skill_trends(None, historical_jobs=undated_df)
        self.assertFalse(unavailable_res["available"])
        self.assertTrue(unavailable_res["trends_df"].empty)

    def test_job_freshness_categories_boundaries_and_stale_preservation(self):
        ref = "2026-11-23T12:00:00Z"

        # Exact boundary checks (fresh_days=7, aging_days=30)
        self.assertEqual(
            classify_job_freshness("2026-11-23T12:00:00Z", reference_time=ref, fresh_days=7, aging_days=30)["freshness_category"],
            "Fresh",
        )
        self.assertEqual(
            classify_job_freshness("2026-11-16T12:00:00Z", reference_time=ref, fresh_days=7, aging_days=30)["freshness_category"],
            "Fresh",
        )
        self.assertEqual(
            classify_job_freshness("2026-11-10T12:00:00Z", reference_time=ref, fresh_days=7, aging_days=30)["freshness_category"],
            "Aging",
        )
        self.assertEqual(
            classify_job_freshness("2026-10-24T12:00:00Z", reference_time=ref, fresh_days=7, aging_days=30)["freshness_category"],
            "Aging",
        )
        # 60 days old WITHOUT explicit source expiry -> must be 'Stale / possibly outdated', NEVER 'Expired'
        stale_check = classify_job_freshness("2026-09-20T12:00:00Z", reference_time=ref, fresh_days=7, aging_days=30)
        self.assertEqual(stale_check["freshness_category"], "Stale / possibly outdated")
        self.assertNotEqual(stale_check["freshness_category"], "Expired")

        # Explicit source confirmation (status='expired' or expiry timestamp in the past) -> 'Expired'
        exp_status = classify_job_freshness("2026-11-20T12:00:00Z", reference_time=ref, status="expired")
        self.assertEqual(exp_status["freshness_category"], "Expired")
        exp_time = classify_job_freshness("2026-11-01T12:00:00Z", reference_time=ref, expiry_at="2026-11-15T00:00:00Z")
        self.assertEqual(exp_time["freshness_category"], "Expired")

        # Missing, malformed, and future posting dates -> 'Missing / Invalid Date'
        self.assertEqual(classify_job_freshness(None, reference_time=ref)["freshness_category"], "Missing / Invalid Date")
        self.assertEqual(classify_job_freshness("not-a-date", reference_time=ref)["freshness_category"], "Missing / Invalid Date")
        self.assertEqual(classify_job_freshness("2026-12-05T12:00:00Z", reference_time=ref)["freshness_category"], "Missing / Invalid Date")

        # Evaluate batch DataFrame and verify stale jobs remain preserved in SQLite for historical analytics
        batch_df = pd.DataFrame([
            {"job_id": "j1", "job_title": "Fresh Role", "company": "A", "location": "Berlin", "published_at": "2026-11-22T12:00:00Z", "skills": "Python"},
            {"job_id": "j2", "job_title": "Aging Role", "company": "B", "location": "Munich", "published_at": "2026-11-05T12:00:00Z", "skills": "SQL"},
            {"job_id": "j3", "job_title": "Stale Role", "company": "C", "location": "Hamburg", "published_at": "2026-08-01T12:00:00Z", "skills": "Python"},
            {"job_id": "j4", "job_title": "Invalid Date Role", "company": "D", "location": "Berlin", "published_at": None, "skills": "Docker"},
        ])
        evaluated, summary = evaluate_jobs_freshness(batch_df, reference_time=ref, fresh_days=7, aging_days=30)
        self.assertEqual(summary["Fresh"], 1)
        self.assertEqual(summary["Aging"], 1)
        self.assertEqual(summary["Stale / possibly outdated"], 1)
        self.assertEqual(summary["Missing / Invalid Date"], 1)
        self.assertEqual(summary["Expired"], 0)
        self.assertEqual(len(evaluated), 4)
        # Stale job is still included in historical skill demand analytics
        self.assertEqual(int(skill_demand(batch_df)["Python"]), 2)

    def test_existing_live_db_integration(self):
        # Test with the existing project SQLite database (data/live_jobs.db) if populated
        stored = load_live_jobs(columns=CAREER_TOOLS_LIVE_COLUMNS)
        if not stored.empty:
            fresh_df, summary = evaluate_jobs_freshness(stored)
            self.assertEqual(summary["total"], len(stored))
            self.assertFalse(fresh_df.empty)
            trend_res = analyze_skill_trends(stored, period_mode="auto")
            self.assertIn("available", trend_res)

    def test_existing_core_ml_and_graph_modules(self):
        raw_df = pd.DataFrame({
            "title": ["Junior Python Developer", "Senior Data Scientist", "Data Analyst", "Lead Engineer", "Intern Analyst"],
            "company_name": ["A", "B", "C", "D", "E"],
            "annual_salary": [60000, 140000, 85000, 160000, 40000],
            "job_description": ["Python SQL", "Python Machine Learning", "SQL Excel", "Java AWS", "Excel"],
            "experience_level": ["Entry", "Senior", "Mid", "Senior", "Entry"],
        })
        df = clean_data(prepare_dataset(raw_df), inplace=True)
        self.assertEqual(str(df["salary"].dtype), "float32")

        knn_res = get_knn_recommendations(df, 80000, k=3)
        self.assertEqual(len(knn_res), 3)
        self.assertIsNotNone(predict_job_level(knn_res))

        clustered, centroids = run_kmeans(df[["salary"]], column="salary", k=2)
        self.assertEqual(len(clustered), 5)
        self.assertEqual(len(centroids), 2)

        feats = extract_features(df[["job_title", "salary", "job_level"]])
        tree = build_tree(feats.to_numpy(), df["job_level"].astype(str).to_numpy(), max_depth=3)
        pred = predict(tree, feats.iloc[0].to_numpy())
        self.assertIn(pred, {"Entry", "Mid", "Senior"})

        _, z_count = detect_zscore_outliers(df, threshold=2.0)
        self.assertGreaterEqual(z_count, 0)
        _, iqr_stats = detect_iqr_outliers(df)
        self.assertIn("Median", iqr_stats)

        bfs_path, _ = bfs(CAREER_GRAPH, "Python", "Data Scientist")
        dfs_path, _ = dfs(CAREER_GRAPH, "Python", "Data Scientist")
        a_path, a_cost = a_star(CAREER_GRAPH, "Python", "Data Scientist")
        self.assertTrue(bfs_path and dfs_path and a_path)
        self.assertEqual(a_cost, calculate_path_cost(CAREER_GRAPH, a_path))


if __name__ == "__main__":
    unittest.main()
