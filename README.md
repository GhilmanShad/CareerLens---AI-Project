
# CareerLens
### Job Market Intelligence & Career Recommendation System

CareerLens is an AI-based job market analysis and career recommendation system developed as a semester project.

It combines historical job-market analysis, machine learning algorithms, live job listings, career insights, and an interactive Streamlit dashboard.

## Features

### Historical Job Market Analysis
- Explore job titles, companies, salaries, and experience levels.
- Analyze salary distributions and job-market trends.
- Identify salary outliers.
- Visualize job-market statistics.

### AI Algorithms
The project implements three main machine learning algorithms:

**1. Decision Tree Classification**
- Predicts job experience levels.
- Uses salary and engineered job-related features.
- Implements a custom Decision Tree classifier.

**2. K-Nearest Neighbors (KNN)**
- Finds jobs with similar salaries.
- Predicts job levels using neighboring records.
- Supports configurable K values.

**3. K-Means Clustering**
- Groups jobs based on salary similarity.
- Supports configurable cluster counts.
- Provides visualizations of salary clusters.

### Live Job Market Features
- Fetch live job postings from Arbeitnow.
- Clean, normalize, and deduplicate job records.
- Store job listings in a local SQLite database.
- Recommend jobs using TF-IDF and cosine similarity.
- Analyze skill demand and identify skill gaps.
- Track job applications.

### Career Tools
- Extract skills from PDF and DOCX resumes.
- Compare job roles.
- Generate learning roadmaps.
- Match resumes with available job opportunities.
- Explore career-related insights.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn (evaluation and data splitting)
- NetworkX
- SQLite
- Requests
- pypdf

The primary Decision Tree, KNN, and K-Means implementations are located in the project's `modules/` directory.

## Dataset

The historical dataset is stored locally at:

`data/job_postings.csv`

It contains 123,849 records before preprocessing.

Main columns:
- `job_title`
- `company`
- `salary`
- `description`
- `job_level`

Preprocessing includes:
- Handling missing job titles and company names.
- Converting salaries to numeric values.
- Excluding invalid salaries where required.
- Optional duplicate removal.
- Feature engineering for classification.
- Optional IQR-based outlier filtering for clustering.

The dataset source and licensing information should be documented once verified.

## Model Evaluation

### Classification Results

Both classification models were evaluated using 5,350 test records from 26,747 eligible records.

| Metric | Decision Tree | KNN |
|---|---:|---:|
| Accuracy | 64.65% | 54.11% |
| Macro F1-score | 0.49 | 0.29 |
| Weighted F1-score | 0.60 | 0.53 |

The Decision Tree achieved higher overall accuracy than salary-based KNN.

However, the models use different input features, so the results do not represent a controlled comparison using identical feature sets.

The Decision Tree also performs poorly on the Associate class, indicating that overall accuracy alone does not capture all classification limitations.

### K-Means Clustering Results

K-Means was evaluated using salary data with IQR outlier filtering.

- Records after preprocessing and filtering: 33,717
- Evaluation sample: 5,000
- K values evaluated: 2–8
- Best K by Silhouette Score: 2
- Best Silhouette Score: 0.6201

The Davies–Bouldin Index was lowest for K = 8 (0.4758), showing that different clustering metrics may favor different configurations.

Detailed results are available in:

- `reports/decision_tree_evaluation.txt`
- `reports/knn_evaluation.txt`
- `reports/kmeans_evaluation.txt`

## Installation and Setup

### Ubuntu / Linux

Clone the repository:

```bash
git clone https://github.com/atiq-dehraj/CareerLens---AI-Lab-project.git
cd CareerLens---AI-Lab-project
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

### Windows

Clone or download the repository.

Install the dependencies:

```powershell
py -m pip install -r requirements.txt
```

Run the application:

```powershell
py -m streamlit run app.py
```

Alternatively, use `Run CareerLens.bat` if the required Python environment is configured.

## Using CareerLens

1. Start the Streamlit application.
2. Load the default historical dataset or upload a compatible CSV.
3. Explore job-market statistics and AI analysis features.
4. Use the live job feed to retrieve current postings.
5. Explore career recommendations and skill-gap analysis.
6. Track saved job applications.

Uploaded CSV datasets can use common aliases such as `title`, `position`, `company_name`, `annual_salary`, and `job_description`. A job-title column is required.

The live feed uses Arbeitnow without requiring an API key. Retrieved jobs are stored in `data/live_jobs.db`, allowing previously saved results to remain available when the API is unavailable.

Live-job recommendation scores represent text relevance, not hiring probabilities.

## Running Evaluations

Decision Tree:

```bash
python evaluate_models.py
```

KNN:

```bash
python evaluate_knn.py
```

K-Means:

```bash
python evaluate_kmeans.py
```

## Automated Tests

Run the automated test suite:

```bash
python -m unittest discover -v
```

The existing tests cover data normalization, persistence, recommendations, career tools, and other core features.

## Project Structure

```text
CareerLens/
├── app.py
├── modules/
│   ├── preprocessing.py
│   ├── data_loader.py
│   ├── decision_tree.py
│   ├── decision_tree_features.py
│   ├── enhanced_features.py
│   ├── knn.py
│   ├── kmeans.py
│   ├── career_insights.py
│   ├── graph_search.py
│   ├── live_api.py
│   ├── live_preprocessing.py
│   ├── live_storage.py
│   └── resume_tools.py
├── data/
│   └── job_postings.csv
├── reports/
│   ├── decision_tree_evaluation.txt
│   ├── knn_evaluation.txt
│   └── kmeans_evaluation.txt
├── evaluate_models.py
├── evaluate_knn.py
├── evaluate_kmeans.py
├── test_live_features.py
├── requirements.txt
└── README.md
```

## Limitations

- Historical job-level predictions are imperfect, particularly for underrepresented classes.
- Salary-only KNN does not consider all factors influencing job seniority.
- Salary-based K-Means identifies numerical salary groups, not necessarily meaningful occupational categories.
- Live job availability depends on the external API.
- The historical dataset's provenance and license still require verification.

## Project Repository

https://github.com/atiq-dehraj/CareerLens---AI-Lab-project
