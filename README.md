# CareerLens
## Job Market Intelligence & Career Recommendation System

CareerLens is an Artificial Intelligence semester project that combines
machine learning, historical job-market analysis, live job listings,
career recommendations, and an interactive Streamlit web application.

> **Model update (27 November 2026):** The selected experience classifier is now V6, a fourth algorithm family (linear SVM). Its validation accuracy is 83.54% and its previously evaluated held-out retest accuracy is 83.11%. The original manually implemented Decision Tree, KNN and K-Means are preserved. Sections below describing their original experiments remain historical evidence; see [V6 evaluation and deployment](#v6-evaluation-and-deployment) for the selected model. Full updated Streamlit regression testing is pending on the original installation.

## 1. Problem Statement

Job seekers often struggle to understand market demand, salary trends,
required skills, experience levels, and suitable career opportunities.

CareerLens addresses these challenges by combining historical and live
job-market data with AI algorithms to support informed career decisions
through a centralized, interactive platform.

## 2. Project Objectives

- Analyze historical and current job-market data.
- Predict job experience levels using machine learning.
- Recommend jobs based on salary and skill similarity.
- Identify salary patterns, clusters, and outliers.
- Analyze market skill demand and individual skill gaps.
- Generate personalized career development roadmaps.
- Support resume analysis and job matching.
- Integrate AI algorithms into an interactive web application.

## 3. Application Features

### Historical Job Market Analysis

- Interactive market overview.
- Job title, company, salary, and experience-level analysis.
- Salary distribution visualization.
- Z-score and IQR-based outlier detection.

### Artificial Intelligence Algorithms

#### Linear SVM (selected V6)

- Uses job title, description and optional company name; no salary required.
- Combines word and character TF-IDF, experience cues and company/title patterns.
- Trained on 60,295 original and 15,000 supplementary records (sample weight 0.10).
- Open **Experience Classifier (SVM)** and submit the job information; the trained model is loaded locally and cached.

#### Decision Tree Classification

- Custom Decision Tree classifier.
- Predicts job experience levels.
- Supports salary-based and engineered-feature classification.
- Uses salary, job-title keywords, and job-description keywords.

#### K-Nearest Neighbors (KNN)

- Custom salary-based nearest-neighbor algorithm.
- Identifies similar job opportunities.
- Predicts experience levels using neighboring records.
- Supports configurable K values.

#### K-Means Clustering

- Custom salary-based clustering algorithm.
- Groups jobs according to salary similarity.
- Supports configurable cluster counts.
- Provides cluster visualization and centroid analysis.

### Career Path Finder

- Breadth-First Search (BFS).
- Depth-First Search (DFS).
- A* Search.
- Graph-based career transition analysis.

### Live Job Market

- Retrieves live job postings from Arbeitnow.
- Normalizes and deduplicates retrieved records.
- Stores job listings in a local SQLite database.
- Recommends jobs using TF-IDF and cosine similarity.
- Analyzes skill demand and skill gaps.
- Supports job application tracking.

### Career Intelligence Tools

1. Job Demand Forecasting.
2. Resume Analyzer and Learning Roadmap.
3. Resume-Job Matching.
4. Career Role Comparison.
5. Emerging vs Declining Skills Analysis.
6. Job Expiry and Freshness Tracker.

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and AI implementation |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Scikit-learn | Evaluation, data splitting and text processing |
| NetworkX | Career graph representation |
| SQLite | Local database and application tracking |
| Requests | Live API integration |
| pypdf | PDF resume text extraction |
| zipfile and ElementTree | DOCX resume text extraction |
| Git and GitHub | Version control and collaboration |
| unittest | Automated software testing |

The primary Decision Tree, KNN, and K-Means algorithms are implemented
within the project's modules rather than using ready-made scikit-learn
estimators.

The project has been tested locally on Ubuntu using Python 3.14.4.

## 5. Dataset

### Historical Dataset

The historical dataset contains 123,849 job records before
application-level preprocessing.

The local dataset structure matches the following published dataset:

- Dataset: LinkedIn Job Postings (2023-2024)
- Platform: Kaggle
- Publisher: Arsh Kon
- Source: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

The exact downloaded dataset version and applicable licensing terms
should be confirmed against the original Kaggle download.

### Main Dataset Attributes

| Attribute | Description |
|---|---|
| job_title | Title of the advertised position |
| company | Organization offering the position |
| salary | Normalized annual salary in USD |
| description | Job description and requirements |
| job_level | Experience-level classification target |

### Classification Target Classes

- Associate
- Director
- Entry level
- Executive
- Internship
- Mid-Senior level

### Live Dataset

Live job postings are retrieved from the Arbeitnow Job Board API.

API endpoint:
https://www.arbeitnow.com/api/job-board-api

Live job data supports job recommendations, skill analysis,
career intelligence, and application tracking.

## 6. Data Preprocessing

The preprocessing workflow includes:

- Removing records with missing job titles.
- Handling missing company names.
- Converting salary values into numeric format.
- Selecting valid positive USD-normalized salaries.
- Removing duplicate records during model evaluation.
- Handling missing classification targets.
- Selecting relevant dataset attributes.
- Extracting binary features from job titles and descriptions.
- Applying IQR-based salary outlier filtering where required.
- Splitting eligible records into training and testing datasets.

### Feature Engineering

The enhanced Decision Tree uses 19 input features:

- One numerical annual salary feature.
- Eight binary job-title keyword features.
- Ten binary job-description keyword features.

The binary features represent the presence or absence of
experience-related terms.

Feature scaling is not required for the salary-only KNN and K-Means
configurations because they operate on a single numerical feature.
Decision Trees are also generally insensitive to feature scaling.

## 7. Exploratory Data Analysis

The application provides interactive exploratory analysis through:

- Job experience-level distributions.
- Company and job-title analysis.
- Salary distribution analysis.
- Salary outlier identification.
- Salary cluster visualization.
- Skill frequency and demand analysis.
- Career role comparisons.
- Live job-market trend visualization.

These visualizations help identify patterns in job availability,
salary distributions, experience requirements, and skill demand.

## 8. Machine Learning Model Evaluation

### Classification Evaluation Methodology

All three evaluated classification configurations use the same
eligible dataset and stratified training/testing split.

| Parameter | Value |
|---|---:|
| Eligible records | 26,747 |
| Training records | 21,397 |
| Testing records | 5,350 |
| Training proportion | 80% |
| Testing proportion | 20% |
| Random seed | 42 |
| Target classes | 6 |
| KNN neighbors | 5 |
| Decision Tree maximum depth | 7 |

### Classification Results

| Metric | Salary-only KNN | Salary-only Decision Tree | Enhanced Decision Tree |
|---|---:|---:|---:|
| Accuracy | 54.11% | 59.46% | 64.65% |
| Macro Precision | 0.31 | 0.23 | 0.64 |
| Macro Recall | 0.29 | 0.25 | 0.47 |
| Macro F1-score | 0.29 | 0.22 | 0.49 |
| Weighted F1-score | 0.53 | 0.53 | 0.60 |

The majority-class baseline accuracy is approximately 46.36%.

### Best Original Classification Configuration

The Enhanced Decision Tree was selected among the original salary-dependent experiments. V6 is now the selected experience classifier; see the V6 section for its separate evaluation.

It achieves:

- Accuracy of 64.65%.
- Macro F1-score of 0.49.
- Weighted F1-score of 0.60.
- Approximately 10.54 percentage points higher accuracy than
  salary-only KNN.
- Approximately 5.20 percentage points higher accuracy than
  the salary-only Decision Tree.

The enhanced configuration benefits from additional information
extracted from job titles and descriptions.

The salary-only Decision Tree and salary-only KNN provide a
same-feature algorithm comparison. Comparing the enhanced Decision
Tree against these configurations also demonstrates the contribution
of feature engineering.

Although the enhanced model performs best overall, its performance
on the Associate class remains weak. Classification accuracy alone
does not fully describe performance across imbalanced classes.

### K-Means Clustering Evaluation

K-Means is evaluated separately because it is an unsupervised
learning algorithm.

The dataset is split before IQR filtering. Outlier thresholds are
calculated using training data and then applied to both training
and testing records.

| Parameter | Value |
|---|---:|
| Valid salary records | 34,623 |
| Training records before IQR filtering | 27,698 |
| Testing records before IQR filtering | 6,925 |
| Training records after IQR filtering | 26,964 |
| Testing records after IQR filtering | 6,753 |
| Training evaluation sample | 5,000 |
| K values evaluated | 2-8 |
| Selected K | 2 |

The optimal K is selected using the training silhouette score.

### Final Clustering Results

| Metric | Result |
|---|---:|
| Selected K | 2 |
| Training sample silhouette score | 0.6223 |
| Independent test silhouette score | 0.6122 |
| Independent test Davies-Bouldin index | 0.5356 |

### Salary Cluster Distribution

| Cluster | Training Centroid | Test Records |
|---|---:|---:|
| Lower salary cluster | $60,456.47 | 4,358 |
| Higher salary cluster | $143,128.70 | 2,395 |

The held-out test results indicate that the learned salary clusters
generalize reasonably well to unseen, in-range salary records.

Clustering metrics are not directly comparable with supervised
classification accuracy or F1-scores.

### Detailed Evaluation Reports

- [Decision Tree Evaluation](reports/decision_tree_evaluation.txt)
- [KNN Evaluation](reports/knn_evaluation.txt)
- [K-Means Evaluation](reports/kmeans_evaluation.txt)

## 9. Installation and Setup

### Ubuntu / Linux

Clone the official project repository:

```bash
git clone https://github.com/atiq-dehraj/CareerLens---AI-Lab-project.git
cd CareerLens---AI-Lab-project
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

### Dataset Preparation

The original and processed datasets are excluded from GitHub
because of their large size.

1. Download the historical dataset from:

   https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

2. Extract the downloaded dataset into the project's archive directory.

The expected structure is:

```text
archive/
├── postings.csv
├── companies/
│   ├── companies.csv
│   ├── company_industries.csv
│   ├── company_specialities.csv
│   └── employee_counts.csv
├── jobs/
│   ├── benefits.csv
│   ├── job_industries.csv
│   ├── job_skills.csv
│   └── salaries.csv
└── mappings/
    ├── industries.csv
    └── skills.csv
```

Create the data directory and prepare the historical dataset:

```bash
mkdir -p data
python prepare_data.py
```

The processed dataset should be generated at:

```text
data/job_postings.csv
```

### Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

The default Streamlit address is:

http://localhost:8501

### Windows

Clone or download the repository and open PowerShell
inside the project directory.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download and extract the dataset into the archive directory
as described above.

```powershell
New-Item -ItemType Directory -Force data
python prepare_data.py
python -m streamlit run app.py
```

Alternatively, use Run CareerLens.bat if the required
Python environment is configured.

## 10. Using CareerLens

1. Start the Streamlit application.
2. Load the default historical dataset or upload a compatible CSV.
3. Explore the market overview and salary analysis.
4. Generate KNN-based job recommendations.
5. Explore salary clusters using K-Means.
6. Predict experience levels using the Decision Tree.
7. Explore career paths using graph search algorithms.
8. Retrieve live job postings.
9. Analyze resumes and generate learning roadmaps.
10. Compare career roles and market skills.
11. Track job freshness and saved applications.

Uploaded CSV files may use common aliases such as title,
position, company_name, annual_salary, and job_description.

A job-title column is required.

The Arbeitnow API does not require an API key.
Retrieved job listings are stored locally in data/live_jobs.db,
allowing previously saved records to remain available when
the API is unavailable.

## 11. Running Model Evaluations

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

## 12. Automated Testing

Run the automated test suite:

```bash
python -m unittest discover -v
```

The latest recorded regression test run completed successfully.

| Test Result | Count |
|---|---:|
| Total tests | 22 |
| Passed | 22 |
| Failed | 0 |

The tests cover areas including:

- Live job data normalization.
- Database persistence.
- Job recommendations.
- Application date handling.
- Skill-gap analysis.
- Career roadmap relevance.
- KNN labeled-neighbor selection.
- Skill trend period selection.
- Job freshness classification.
- Job freshness precision and sorting.

## 13. Project Structure

```text
CareerLens/
├── app.py
├── prepare_data.py
├── evaluate_models.py
├── evaluate_knn.py
├── evaluate_kmeans.py
├── modules/
│   ├── career_insights.py
│   ├── data_loader.py
│   ├── decision_tree.py
│   ├── decision_tree_features.py
│   ├── enhanced_features.py
│   ├── graph_search.py
│   ├── job_fetcher.py
│   ├── kmeans.py
│   ├── knn.py
│   ├── live_api.py
│   ├── live_preprocessing.py
│   ├── live_storage.py
│   ├── outlier_detection.py
│   ├── preprocessing.py
│   └── resume_tools.py
├── archive/                         # Local original dataset
├── data/                            # Local processed data
│   ├── job_postings.csv
│   └── live_jobs.db
├── reports/
│   ├── decision_tree_evaluation.txt
│   ├── knn_evaluation.txt
│   └── kmeans_evaluation.txt
├── tests/
│   ├── __init__.py
│   ├── test_application_dates.py
│   ├── test_knn_labeled_neighbors.py
│   ├── test_roadmap_relevance.py
│   └── test_skill_gap.py
├── test_live_features.py
├── test_freshness_precision.py
├── test_freshness_batch_precision.py
├── test_freshness_sorting.py
├── test_skill_trend_periods.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 14. Project Limitations

- Classification performance varies across experience-level classes.
- The Associate class remains difficult to predict accurately.
- Salary-only KNN cannot capture all factors influencing job seniority.
- Salary clusters represent numerical salary groups rather than
  definitive occupational categories.
- Historical job-market data may not reflect current market conditions.
- Live job availability depends on an external API.
- Reliable forecasting requires sufficient historical time coverage.
- Short-term skill trends should not be interpreted as established
  long-term market trends.
- Resume matching scores represent text relevance rather than
  hiring probabilities.
- Skill extraction depends on available text and recognized keywords.
- Career recommendations provide decision support rather than
  guaranteed employment outcomes.
- Exact historical dataset provenance and licensing require
  final confirmation.

## 15. Group Members and Contributions

### Ghilman Shad Ali

- Registration Number: 24P-3101
- GitHub: https://github.com/GhilmanShad
- Primary Role: AI Development and Quality Assurance

Responsibilities:

- Machine learning development and evaluation.
- Feature engineering and comparative model analysis.
- Career recommendation and intelligence features.
- Application debugging and functional improvements.
- Functional and regression testing.
- Technical documentation and project report preparation.

### Atiq-ur-Rehman

- Registration Number: 24P-3002
- GitHub: https://github.com/atiq-dehraj
- Primary Role: Application Development and System Integration

Responsibilities:

- Project architecture and module organization.
- Streamlit application development.
- Integration of AI algorithms into the application.
- Live job API and database integration.
- Data visualization and interactive features.
- GitHub repository management and collaborative development.

### Mohammad Ramzan

- Registration Number: 24P-3051
- GitHub: https://github.com/ramzanbhutto
- Primary Role: Dataset Preparation and Initial Development

Responsibilities:

- Initial dataset exploration and auditing.
- Dataset filtering and relevant attribute selection.
- Identification of missing values and duplicate records.
- Preliminary exploratory data analysis.
- Dataset organization and preprocessing preparation.
- Initial project planning and requirements analysis.

## 16. Project Repository

Official GitHub repository:

https://github.com/atiq-dehraj/CareerLens---AI-Lab-project

## 17. Academic Purpose

CareerLens was developed as a three-member Artificial Intelligence
semester project to demonstrate the practical application of
machine learning, data preprocessing, exploratory data analysis,
model evaluation, graph search, recommendation techniques,
and interactive web application development.


## V6 evaluation and deployment

The selected frozen V6 model is packaged in `models/experience_svm_v6.joblib`.
The original 123,849-posting historical dataset is unchanged; the additional 15,000 postings are used only for SVM training. The supplementary data comes from https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024 (selected Associate and Mid-Senior rows; sample labels not independently verified).

| Metric | Validation (15,173) | Held-out retest (18,972) |
|---|---:|---:|
| Accuracy | 83.54% | 83.11% |
| Macro precision | 80.16% | 80.20% |
| Macro recall | 76.33% | 76.81% |
| Macro F1 | 78.05% | 78.30% |
| Weighted F1 | 83.21% | 82.78% |
| Associate recall | 50.65% | 51.12% |

Description-prefix groups were split using seeds 42 and 43. Features and company patterns were fitted on original training data only. The test partition was excluded from V6 training but was evaluated in earlier experiments. This is not an untouched external test; the 85% goal was not met. Earlier Decision Tree/KNN results were measured on a different population and cannot be treated as a controlled algorithm comparison. Company-specific patterns may generalize less well to new employers; near-duplicates and noisy labels can remain.

Install and verify in a project virtual environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -v
python -m streamlit run app.py
```

Use only the trusted bundled model. Tested model runtime: Python 3.13.5, scikit-learn 1.8.0, numpy 2.3.5, scipy 1.17.0, pandas 2.2.3, joblib 1.5.3. Ubuntu Python 3.14 environment compatibility and end-to-end UI are not yet verified. The source update preserves `modules/resume_tools.py` from your original project. In the execution container, Streamlit and the original resume module were unavailable; 14 tests passed (including 4 new classifier tests), two original test modules could not be imported, and the UI smoke test was skipped. Run the complete local suite before merging.

To reproduce the frozen retest (not retrain):

```bash
python final_test.py --original archive/postings.csv --model models/experience_svm_v6.joblib --output reports/v6_test.json
```

Training scripts (`cache_features.py`, `fit_cached.py`, `run_v4.py`, `optimize_experience_v3.py`, `compare_supplement.py`) document the experimental pipeline; exact V6 retraining requires the original postings, supplementary sample ZIP, and V4/V5 checkpoints from the prior optimization packages. See `reports/v6_validation.json` and `reports/v6_test.json` for full results and the confusion matrix. No raw data is included in the update.
