# CareerLens

CareerLens combines the historical job-market analysis project with a live Arbeitnow job feed.

## Run

The easiest way on Windows is to double-click **Run CareerLens.bat** in the project folder.

To try another dataset, use **Choose a local CSV dataset** in the sidebar and then **Load selected CSV**. It accepts common names such as `title`/`position`, `company_name`, `job_description`, and `annual_salary`; a title column is required.

```powershell
py -3.14 -m pip install -r requirements.txt
py -3.14 -m streamlit run app.py
```

Open **Live Job Feed** and select **Refresh Live Jobs**. Arbeitnow needs no API key. Jobs are cleaned, deduplicated and stored in `data/live_jobs.db`; previously saved results remain visible if the API is unavailable.

Use the filters to browse stored jobs. Enter skills and a preferred role under **KNN job recommendations** for TF-IDF/cosine-similarity matches. The score is relevance only, not a hiring probability.

The live page also shows title, location, and skill-demand trends, identifies skills missing for a selected target role, and lets you save a posting. Use **Application Tracker** to update saved applications from Saved through Applied, Interview, Offer, or Rejected. Tracker data is stored in the same local SQLite database.

**Career Tools** provides lightweight demand trends, PDF/DOCX resume skill extraction, a learning roadmap, live-job matching, and role comparison. Resume uploads are analyzed only when requested and are not stored. PDF support requires `pypdf` (included in `requirements.txt`).

## Tests

```powershell
py -3.14 -m unittest test_live_features.py
```

The tests are offline and cover normalization, SQLite persistence, KNN recommendations, skill-gap analysis, and application-status updates.
