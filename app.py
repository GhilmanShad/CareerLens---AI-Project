import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

# Modules Imports
from modules.data_loader import load_and_inspect_data, prepare_dataset
from modules.preprocessing import clean_data
from modules.outlier_detection import detect_zscore_outliers, detect_iqr_outliers
from modules.knn import get_knn_recommendations, predict_job_level, recommend_live_jobs
from modules.kmeans import run_kmeans  
from modules.decision_tree import build_tree, predict
from modules.decision_tree_features import extract_features
from modules.graph_search import bfs, dfs, a_star, CAREER_GRAPH, calculate_path_cost
from modules.job_fetcher import fetch_arbeitnow_jobs, JobFetchError
from modules.live_preprocessing import normalize_live_jobs, split_pipe_values
from modules.live_storage import (
    add_application,
    load_live_jobs,
    load_applications,
    get_live_jobs_status,
    record_refresh_error,
    upsert_live_jobs,
    update_application_status,
)
from modules.career_insights import (
    DEFAULT_AGING_DAYS,
    DEFAULT_FRESH_DAYS,
    DEFAULT_MIN_SKILL_POSTINGS,
    DEFAULT_TREND_CHANGE_PCT,
    EXPERIENCE_LEVEL_PATTERNS,
    analyze_skill_trends,
    evaluate_jobs_freshness,
    skill_demand,
    skill_gap,
)
from modules.resume_tools import (
    CAREER_TOOLS_LIVE_COLUMNS,
    CORE_ROLES,
    ROLE_CATEGORIES,
    build_learning_roadmap,
    compare_career_roles,
    demand_summary,
    extract_resume_text,
    forecast_demand,
    matching_jobs,
    resume_skills,
)


def calculate_skill_demand(descriptions):
    skills = [
        'Python', 'SQL', 'Excel',
        'Pandas', 'Machine Learning', 'Statistics'
    ]

    counts = []
    valid = descriptions.dropna()
    for skill in skills:
        counts.append(
            int(valid.str.contains(
                rf'\b{skill}\b',
                case=False,
                regex=True,
                na=False
            ).sum())
        )

    total = len(descriptions)

    demands = [
        round(count / total * 100, 1) if total else 0
        for count in counts
    ]

    top_skill = (
        skills[counts.index(max(counts))]
        if any(counts) else "N/A"
    )

    return skills, demands, top_skill

# Page configuration
st.set_page_config(page_title="CareerLens", page_icon="📊", layout="wide")

# --- CUSTOM CSS FOR UI (FORCE LIGHT MODE) ---
st.markdown("""
<style>
    /* Global Background & Fonts */
    .stApp {
        background-color: #f8f9fa;
    }
    html, body, [class*="css"] {
        color: #111827 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #002b22; /* Dark Green */
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h1 * {
        color: #00e6b8 !important;
        -webkit-text-fill-color: #00e6b8 !important;
        font-size: 38px !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        line-height: 1.15 !important;
        margin-bottom: 4px !important;
    }
    .sidebar-brand-box {
        margin-top: -28px;
        padding-bottom: 14px;
        margin-bottom: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.18);
    }
    .sidebar-brand-title {
        font-size: 38px !important;
        font-weight: 800 !important;
        color: #00e6b8 !important;
        -webkit-text-fill-color: #00e6b8 !important;
        letter-spacing: -0.6px;
        line-height: 1.1;
        margin: 0 0 6px 0;
    }
    .sidebar-brand-subtitle {
        color: #d1fae5 !important;
        -webkit-text-fill-color: #d1fae5 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        margin: 0;
        opacity: 0.9;
    }
    
    /* White Cards */
    .custom-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 15px;
    }
    .custom-card h4 {
        margin: 0;
        font-size: 14px;
        color: #4b5563 !important;
        font-weight: 600;
    }
    .custom-card h2 {
        margin: 5px 0 0 0;
        font-size: 28px;
        color: #00796b !important;
        font-weight: 700;
    }
    
    /* Chart/Table Boxes */
    .chart-box {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    .chart-box h4 {
        color: #374151 !important;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Improve widget label and helper-text contrast */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCheckbox"] label p,
    [data-testid="stCaptionContainer"] p {
        color: #374151 !important;
        opacity: 1 !important;
    }

    [data-testid="stWidgetLabel"] {
        color: #374151 !important;
        opacity: 1 !important;
    }

    /* FIX STREAMLIT WIDGETS (INPUTS, DROPDOWNS, BUTTONS) */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    [data-testid="stTextInputRootElement"],
    [data-testid="stNumberInputContainer"],
    .react-aria-ComboBox [role="group"] {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
    }

    [data-testid="stTextInputRootElement"] input,
    [data-testid="stNumberInputContainer"] input,
    .react-aria-ComboBox input,
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input {
        background-color: transparent !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #111827 !important;
        opacity: 1 !important;
    }

    [data-testid="stTextInputRootElement"] input::placeholder,
    [data-testid="stNumberInputContainer"] input::placeholder,
    .react-aria-ComboBox input::placeholder {
        color: #6b7280 !important;
        -webkit-text-fill-color: #6b7280 !important;
        opacity: 1 !important;
    }

    /* Dropdown arrow */
    .react-aria-ComboBox button {
        color: #374151 !important;
    }
    .react-aria-ComboBox button svg {
        color: #374151 !important;
        fill: #374151 !important;
    }

    /* Dropdown menu */
    [role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12) !important;
    }
    [role="listbox"] [role="option"],
    [role="listbox"] [role="option"] div {
        background-color: transparent !important;
        color: #111827 !important;
    }
    [role="listbox"] [role="option"][data-focused="true"] {
        background-color: #f3f4f6 !important;
    }
    [role="listbox"] [role="option"][aria-selected="true"] {
        background-color: #e6f4f1 !important;
    }

    /* Number input stepper buttons */
    [data-testid="stNumberInput"] button {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
    }

    /* Metric result text: keep it readable on the light dashboard. */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        padding: 14px !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #111827 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #111827 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        font-weight: 700 !important;
    }
    .stButton>button {
        background-color: #00796b !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #004d40 !important;
    }
    
    /* Table Styling */
    [data-testid="stDataFrame"] th {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        font-weight: 600 !important;
    }

    /* Tabs (Streamlit 1.67 data-testid="stTab"): dark black text on light background */
    [data-testid="stTab"],
    [data-testid="stTab"] *,
    [data-testid="stTabs"] [role="tab"],
    [data-testid="stTabs"] [role="tab"] *,
    [data-testid="stTabs"] button,
    [data-testid="stTabs"] button *,
    button[data-baseweb="tab"],
    button[data-baseweb="tab"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    [data-testid="stTab"][aria-selected="true"],
    [data-testid="stTab"][aria-selected="true"] *,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] *,
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] * {
        color: #00796b !important;
        -webkit-text-fill-color: #00796b !important;
        font-weight: 700 !important;
    }

    /* Selectbox & Multiselect selected text and tags */
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }
    span[data-baseweb="tag"] {
        background-color: #00796b !important;
    }
    span[data-baseweb="tag"] span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* File Uploader Dropzone (dark box): keep all inner text, button & icon bright white */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] *,
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploaderDropzone"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        border: 1px solid #9ca3af !important;
    }
    /* Uploaded file name item shown below the dropzone in main area */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }
        /* Decision Tree result text contrast */
    .decision-tree-result-text,
    .decision-tree-result-text p {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand-box">'
        '<div class="sidebar-brand-title">CareerLens</div>'
        '<div class="sidebar-brand-subtitle">Job Market Intelligence</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    # Naya "Live Job Feed" option add kiya gaya hai
    menu = st.radio("Navigation", 
        ["Market Overview", "Salary & Outliers", "KNN Recommendations", "K-Means Clustering", "Decision Tree", "Career Path Finder", "Live Job Feed", "Career Tools", "Application Tracker"],
        label_visibility="collapsed")
    
    st.markdown("---")
    uploaded_dataset = st.file_uploader("Choose a local CSV dataset", type=["csv"])
    load_label = "Load selected CSV" if uploaded_dataset else "Load default dataset"
    if st.button(load_label):
        with st.spinner("Loading..."):
            try:
                df = (prepare_dataset(pd.read_csv(uploaded_dataset, low_memory=True)) if uploaded_dataset
                      else load_and_inspect_data("data/job_postings.csv"))
                if isinstance(df, str):
                    st.error(df)
                else:
                    st.session_state.data = clean_data(df, inplace=True)
                    st.session_state.pop("skill_demand_cache", None)
                    st.session_state.pop("resume_match_cache", None)
                    st.success(f"Loaded {len(st.session_state.data):,} records.")
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(f"Could not load this CSV: {exc}")

st.title(menu)

if st.session_state.data is None and menu not in {"Live Job Feed", "Career Tools", "Application Tracker"}:
    st.info("👈 Please click 'Load Dataset' in the sidebar to start.")
else:
    # Agar Live Job Feed select nahi hai tabhi data load karein
    if menu not in {"Live Job Feed", "Career Tools", "Application Tracker"}:
        df = st.session_state.data

    # ==========================================
    # 1. MARKET OVERVIEW
    # ==========================================
    if menu == "Market Overview":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Dataset summary, top skills and job titles</p><br>", unsafe_allow_html=True)
        
        # Cache lightweight summary in session_state instead of hashing a 450MB Series.
        cache_key = (id(df), len(df))
        cached_skills = st.session_state.get("skill_demand_cache")
        if cached_skills and cached_skills[0] == cache_key:
            skills, demands, top_skill = cached_skills[1]
        else:
            with st.spinner("Analyzing job skills..."):
                skills, demands, top_skill = calculate_skill_demand(df['description'])
            st.session_state.skill_demand_cache = (cache_key, (skills, demands, top_skill))
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f'<div class="custom-card"><h4>Jobs analysed</h4><h2>{df.shape[0]:,}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(
                f'<div class="custom-card"><h4>Top tracked skill</h4>'
                f'<h2>{top_skill}</h2></div>',
                unsafe_allow_html=True
            )
        with col3: st.markdown(f'<div class="custom-card"><h4>Median salary</h4><h2>${df["salary"].median():,.0f}</h2></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="custom-card"><h4>Job categories</h4><h2>{df["job_title"].nunique()}</h2></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns([1.5, 1])
        with col_chart1:
            st.markdown('<div class="chart-box"><h4>Skill demand (% of postings)</h4>', unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            fig1.patch.set_facecolor('white')
            ax1.barh(skills[::-1], demands[::-1], color='#00796b', height=0.6)
            ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False); ax1.spines['left'].set_visible(False); ax1.spines['bottom'].set_visible(False)
            for i, v in enumerate(demands[::-1]): ax1.text(v + 2, i, f"{v}%", va='center', color='#374151', fontsize=10)
            ax1.set_xticks([]); ax1.tick_params(axis='y', colors='#374151', labelsize=11)
            st.pyplot(fig1)
            plt.close(fig1)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_chart2:
            st.markdown('<div class="chart-box"><h4>Top job titles</h4>', unsafe_allow_html=True)
            top_titles = df['job_title'].value_counts().head(5).reset_index()
            top_titles.columns = ['Title', 'Postings']
            st.dataframe(top_titles, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 2. SALARY & OUTLIERS
    # ==========================================
    elif menu == "Salary & Outliers":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Z-score and IQR outlier detection side by side</p><br>", unsafe_allow_html=True)
        
        col_input1, col_input2, col_input3, col_input4 = st.columns(4)
        with col_input1: st.text_input("Salary field", "salary", disabled=True)
        with col_input2: st.text_input("Method", "Z-score + IQR", disabled=True)
        with col_input3: z_thresh = st.number_input(
    "Z threshold", min_value=0.5, value=3.0, step=0.5
)
        with col_input4: st.text_input("Currency / unit", "USD / year", disabled=True)
        
        if st.button("Run Analysis"):
            outliers_z, count_z = detect_zscore_outliers(df, threshold=z_thresh)
            outliers_iqr, stats_iqr = detect_iqr_outliers(df)
            col_metric1, col_metric2, col_metric3 = st.columns(3)

            with col_metric1:
                st.metric("Valid Salary Records", df["salary"].notna().sum())

            with col_metric2:
                st.metric("Z-score Outliers", count_z)

            with col_metric3:
                st.metric("IQR Outliers", len(outliers_iqr))

            # Salary distribution and IQR outlier boundaries
            valid_salaries = pd.to_numeric(
                df["salary"], errors="coerce"
            )
            valid_salaries = valid_salaries[
                np.isfinite(valid_salaries) & (valid_salaries > 0)
            ]

            if not valid_salaries.empty and stats_iqr:
                st.subheader("Salary Distribution & IQR Boundaries")

                fig, ax = plt.subplots(figsize=(10, 4))
                # Limit the chart range to improve readability.
                # Outlier calculations still use all valid salaries.
                display_limit = valid_salaries.quantile(0.99)
                display_salaries = valid_salaries[
                    valid_salaries <= display_limit
                ]

                ax.hist(
                    display_salaries,
                    bins=50,
                    color="#00796b",
                    alpha=0.75,
                    edgecolor="white"
                )
                ax.set_xlim(0, display_limit)

                ax.axvline(
                    stats_iqr["Lower Fence"],
                    color="#d97706",
                    linestyle="--",
                    label="IQR Lower Fence"
                )
                ax.axvline(
                    stats_iqr["Upper Fence"],
                    color="#be123c",
                    linestyle="--",
                    label="IQR Upper Fence"
                )

                ax.set_xlabel("Annual Salary (USD)")
                ax.set_ylabel("Number of Job Postings")
                ax.legend()
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                st.pyplot(fig)
                plt.close(fig)



    # ==========================================
    # 3. KNN RECOMMENDATIONS
    # ==========================================
    elif menu == "KNN Recommendations":
        st.caption("Find job postings with salaries closest to your target.")

        col1, col2 = st.columns(2)

        with col1:
            target_salary = st.number_input(
                "Target Annual Salary (USD)",
                min_value=1,
                value=80000,
                step=5000
            )

        with col2:
            k_neighbors = st.number_input(
                "Number of Neighbors (K)",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            )

        if st.button("Find Recommendations"):
            nearest_jobs = get_knn_recommendations(
                df, target_salary, k=int(k_neighbors)
            )

            if nearest_jobs.empty:
                st.warning("No matching jobs with valid salaries were found.")
            else:
                # Use labeled neighbors for job-level prediction.
                labeled_df = df.copy()

                if "job_level" in labeled_df.columns:
                    levels = labeled_df["job_level"].astype("string").str.strip()
                    valid_levels = (
                        levels.notna()
                        & levels.ne("")
                        & ~levels.str.lower().isin(["none", "nan", "null", "unknown"])
                    ).fillna(False)
                    labeled_df = labeled_df.loc[valid_levels]
                else:
                    labeled_df = labeled_df.iloc[0:0]

                labeled_neighbors = get_knn_recommendations(
                    labeled_df, target_salary, k=int(k_neighbors)
                )
                predicted_level = predict_job_level(labeled_neighbors)

                st.success(
                    f"Found {len(nearest_jobs)} nearest job postings."
                )

                if predicted_level is not None:
                    st.metric("Predicted Job Level", predicted_level)
                else:
                    st.info("Job level unavailable for these recommendations.")

                display_columns = [
                    col for col in
                    ["job_title", "company", "salary", "job_level", "distance"]
                    if col in nearest_jobs.columns
                ]

                st.dataframe(
                    nearest_jobs[display_columns],
                    hide_index=True,
                    use_container_width=True
                )

    # ==========================================
    # 4. K-MEANS CLUSTERING
    # ==========================================
    elif menu == "K-Means Clustering":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Group postings by selected numeric features</p><br>", unsafe_allow_html=True)
        
        st.caption(
            "One-dimensional K-Means: groups jobs by annual salary. "
            "Job index is used only for visualization."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.text_input("Clustering feature", "salary", disabled=True)

        with col2:
            k_clusters = st.number_input(
                "K (clusters)", min_value=2, max_value=8, value=3
            )

        with col3:
            st.text_input("Max iterations", "100", disabled=True)

        apply_iqr = st.checkbox(
            "Exclude salary outliers using IQR",
            value=True
        )

        if st.button("Run K-Means"):
            # Slice only the salary column so we do not copy large text descriptions.
            salaries = pd.to_numeric(df["salary"], errors="coerce")
            valid_mask = np.isfinite(salaries) & (salaries > 0)

            if apply_iqr and valid_mask.any():
                valid = salaries.loc[valid_mask]
                q1 = valid.quantile(0.25)
                q3 = valid.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                valid_mask &= (salaries >= lower) & (salaries <= upper)

            clustering_df = pd.DataFrame({"salary": salaries.loc[valid_mask]})

            clustered_df, centroids = run_kmeans(
                clustering_df, column="salary", k=k_clusters
            )

            st.caption(
                f"Clustered {len(clustered_df):,} postings · "
                f"IQR filtering {'enabled' if apply_iqr else 'disabled'}"
            )
            
            col_chart1, col_chart2 = st.columns([1.5, 1])
            with col_chart1:
                st.subheader("Cluster plot")
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('white')
                colors = ['#00796b', '#d97706', '#4f46e5', '#be123c', '#10b981']
                
                for i in range(k_clusters):
                    cluster_data = clustered_df[clustered_df['cluster'] == i]
                    ax.scatter(cluster_data.index, cluster_data['salary'], c=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.6)
                
                for i, centroid in enumerate(centroids):
                    cluster_data = clustered_df[
                        clustered_df["cluster"] == i
                    ]

                    if cluster_data.empty:
                        continue

                    centroid_x = np.mean(
                        cluster_data.index.to_numpy(dtype=float)
                    )

                    ax.scatter(
                        centroid_x,
                        centroid[0],
                        c=colors[i % len(colors)],
                        marker="X",
                        s=200,
                        edgecolors="black",
                        zorder=5
                    )
                
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.set_xlabel("Job Index", color='#374151')
                ax.set_ylabel("Annual Salary (USD)", color='#374151')
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                
            with col_chart2:
                st.subheader("Centroids and sizes")
                summary = clustered_df.groupby('cluster')['salary'].agg(['count', 'mean']).reset_index()
                summary.columns = ['Cluster', 'Size', 'Avg Salary']
                summary['Avg Salary'] = summary['Avg Salary'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(summary, hide_index=True, use_container_width=True)

    # ==========================================
    # ==========================================
    # 5. DECISION TREE
    # ==========================================
    elif menu == "Decision Tree":
        st.caption("Predict job seniority using salary and job-title features")

        col1, col2 = st.columns(2)

        with col1:
            job_title = st.text_input(
                "Job title",
                value="Senior Python Developer"
            )

        with col2:
            test_salary = st.number_input(
                "Annual salary (USD)",
                min_value=1,
                max_value=1000000,
                value=95000,
                step=5000
            )

        depth = st.slider("Maximum tree depth", 1, 10, 3)

        if st.button("Train & Predict"):
            # Slice only required columns to avoid copying huge job descriptions.
            salaries = pd.to_numeric(df["salary"], errors="coerce")
            valid_mask = (
                df["job_level"].notna()
                & np.isfinite(salaries)
                & (salaries > 0)
            )

            if int(valid_mask.sum()) < 2:
                st.error("Not enough valid training records.")
            else:
                valid_sal = salaries.loc[valid_mask]
                q1 = valid_sal.quantile(0.25)
                q3 = valid_sal.quantile(0.75)
                iqr = q3 - q1

                inlier_mask = valid_mask & salaries.between(
                    q1 - 1.5 * iqr,
                    q3 + 1.5 * iqr
                )
                training_df = df.loc[inlier_mask, ["job_title", "salary", "job_level"]]

                features = extract_features(training_df)
                X = features.to_numpy()
                y = training_df["job_level"].to_numpy()

                with st.spinner("Training Decision Tree..."):
                    tree = build_tree(X, y, max_depth=depth)

                sample = pd.DataFrame({
                    "job_title": [job_title],
                    "salary": [test_salary]
                })

                sample_features = extract_features(sample)
                sample_features = sample_features[features.columns]
                sample_values = sample_features.iloc[0].to_numpy()

                prediction = predict(tree, sample_values)

                st.success(f"Predicted job level: {prediction}")
                st.caption(
                    f"Trained on {len(training_df):,} records "
                    f"using {X.shape[1]} features"
                )

                col_tree, col_pred = st.columns([2, 1])

                with col_tree:
                    st.subheader("Decision path")

                    current_node = tree

                    while current_node.value is None:
                        feature_idx = current_node.feature
                        feature_name = features.columns[feature_idx]
                        threshold = current_node.threshold
                        value = sample_values[feature_idx]

                        if value <= threshold:
                            st.markdown(
                                f'<p class="decision-tree-result-text">'
                                f'{feature_name}: {value:,.0f} '
                                f'≤ {threshold:,.0f} → Left</p>',
                                unsafe_allow_html=True
                            )
                            current_node = current_node.left
                        else:
                            st.markdown(
                                f'<p class="decision-tree-result-text">'
                                f'{feature_name}: {value:,.0f} '
                                f'&gt; {threshold:,.0f} → Right</p>',
                                unsafe_allow_html=True
                            )
                            current_node = current_node.right

                    st.markdown(
                        f'<p class="decision-tree-result-text">'
                        f'Leaf prediction: <strong>{current_node.value}</strong></p>',
                        unsafe_allow_html=True
                    )

                with col_pred:
                    st.subheader("Prediction")
                    st.markdown(
                        '<p class="decision-tree-result-text"><strong>Job level</strong></p>',
                        unsafe_allow_html=True
                    )
                    st.info(prediction)
    # ==========================================
    # 6. CAREER PATH FINDER
    # ==========================================
    elif menu == "Career Path Finder":
        st.markdown("<p style='color: #6b7280; font-size: 16px; margin-top: -10px;'>Compare BFS, DFS and A* on the same start and goal</p><br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        all_nodes = list(CAREER_GRAPH.keys())
        with col1: start_node = st.selectbox("Current skills", all_nodes, index=0)
        with col2: goal_node = st.selectbox("Target role", all_nodes, index=len(all_nodes)-1)
        with col3: st.text_input("Algorithm", "Compare BFS, DFS, A*", disabled=True)
        with col4: st.text_input("Cost unit", "Learning hours", disabled=True)
        
        if st.button("Find Path"):
            st.markdown('<div class="chart-box"><h4>Career graph (highlighted: A* route)</h4>', unsafe_allow_html=True)
            
            G = nx.DiGraph()
            for node, edges in CAREER_GRAPH.items():
                for neighbor, weight in edges.items():
                    G.add_edge(node, neighbor, weight=weight)
                    
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('white')
            
            pos = nx.spring_layout(G, k=1.5, seed=42)
            
            nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='#00796b', ax=ax)
            node_labels = {
                "Machine Learning": "Machine\nLearning",
                "Data Scientist": "Data\nScientist",
                "Visualization": "Visualization",
            }

            labels = {node: node_labels.get(node, node) for node in G.nodes()}

            nx.draw_networkx_labels(
                G, pos,
                labels=labels,
                font_color='white',
                font_size=8,
                font_weight='bold',
                ax=ax
            )
            a_path, a_cost = a_star(CAREER_GRAPH, start_node, goal_node)

            highlighted_edges = set(zip(a_path, a_path[1:])) if a_path else set()

            edge_colors = [
                '#ef4444' if edge in highlighted_edges else '#9ca3af'
                for edge in G.edges()
            ]

            edge_widths = [
                4 if edge in highlighted_edges else 2
                for edge in G.edges()
            ]

            nx.draw_networkx_edges(
                G, pos,
                arrowstyle='->',
                arrowsize=20,
                edge_color=edge_colors,
                width=edge_widths,
                ax=ax
            )
            
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-box"><h4>Algorithm comparison</h4>', unsafe_allow_html=True)
            bfs_path, bfs_cost = bfs(CAREER_GRAPH, start_node, goal_node)
            dfs_path, dfs_cost = dfs(CAREER_GRAPH, start_node, goal_node)
            a_path, a_cost = a_star(CAREER_GRAPH, start_node, goal_node)
            
            algo_df = pd.DataFrame({
                "Algorithm": ["BFS", "DFS", "A*"],
                "Path": [" → ".join(bfs_path) if bfs_path else "None", " → ".join(dfs_path) if dfs_path else "None", " → ".join(a_path) if a_path else "None"],
                "Edges": [len(bfs_path)-1 if bfs_path else 0, len(dfs_path)-1 if dfs_path else 0, len(a_path)-1 if a_path else 0],
                "Cost": [f"{calculate_path_cost(CAREER_GRAPH, path)*40} h" if path else "N/A" for path in (bfs_path, dfs_path, a_path)]
            })
            st.dataframe(algo_df, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 7. LIVE JOB FEED
    # ==========================================
    elif menu == "Live Job Feed":
        st.header("Live Job Market Feed")
        st.caption("Live postings from Arbeitnow. No API key is required; saved jobs remain available offline.")
        status = get_live_jobs_status()
        if st.button("Refresh Live Jobs", type="primary"):
            try:
                with st.spinner("Fetching latest jobs..."):
                    result = fetch_arbeitnow_jobs()
                    stats = upsert_live_jobs(normalize_live_jobs(result["jobs"]))
                st.success(f"Fetched {result['total_received']}; inserted {stats['inserted']}, updated {stats['updated']}, unchanged {stats['skipped']}.")
                status = get_live_jobs_status()
            except (JobFetchError, ValueError) as exc:
                record_refresh_error(str(exc))
                st.error(f"Refresh failed: {exc}. Stored jobs are still available below.")

        st.caption(f"Stored jobs: {status['total_jobs']} | Last refresh: {status['last_refresh_at'] or 'Never'}")
        jobs = load_live_jobs(columns=CAREER_TOOLS_LIVE_COLUMNS)
        if jobs.empty:
            st.info("No saved live jobs yet. Use Refresh Live Jobs to load them.")
        else:
            f1, f2, f3, f4 = st.columns(4)
            keyword = f1.text_input("Keyword")
            location_filter = f2.text_input("Location")
            remote_filter = f3.selectbox("Work mode", ["All", "Remote", "On-site"])
            freshness_quick = f4.selectbox(
                "Freshness filter",
                ["All", "Fresh", "Aging", "Stale / possibly outdated", "Missing / Invalid Date"],
            )
            mask = np.ones(len(jobs), dtype=bool)
            if keyword:
                text = jobs[["job_title", "company", "skills", "tags"]].fillna("").agg(" ".join, axis=1)
                mask &= text.str.contains(keyword, case=False, regex=False).to_numpy()
            if location_filter:
                mask &= jobs["location"].fillna("").str.contains(location_filter, case=False, regex=False).to_numpy()
            if remote_filter != "All":
                mask &= (jobs["remote"] == (remote_filter == "Remote")).to_numpy()
            visible = jobs if mask.all() else jobs.loc[mask]

            fresh_table, fresh_summary = evaluate_jobs_freshness(
                visible,
                fresh_days=st.session_state.get("cfg_fresh_days", DEFAULT_FRESH_DAYS),
                aging_days=st.session_state.get("cfg_aging_days", DEFAULT_AGING_DAYS),
                source_mode="Live only",
                freshness_filter=freshness_quick,
                sort_by="newest",
            )
            if freshness_quick != "All" and not fresh_table.empty:
                visible = visible[visible["job_id"].isin(set(fresh_table["job_id"]))]
            elif freshness_quick != "All" and fresh_table.empty:
                visible = visible.iloc[0:0]

            st.caption(
                f"Feed Freshness — 🟢 Fresh: {fresh_summary['Fresh']} | "
                f"🟡 Aging: {fresh_summary['Aging']} | "
                f"🟠 Stale / possibly outdated: {fresh_summary['Stale / possibly outdated']} | "
                f"⚪ Missing/Invalid date: {fresh_summary['Missing / Invalid Date']}"
            )
            feed_display = fresh_table[[
                "job_title", "company", "location", "freshness_badge",
                "age_days", "published_date", "last_checked", "url",
            ]].rename(columns={
                "job_title": "Job Title",
                "company": "Company",
                "location": "Location",
                "freshness_badge": "Freshness",
                "age_days": "Age (days)",
                "published_date": "Published Date",
                "last_checked": "Last Checked",
                "url": "URL",
            })
            st.dataframe(
                feed_display,
                hide_index=True,
                use_container_width=True,
                column_config={"URL": st.column_config.LinkColumn("Job Link")},
            )

            st.subheader("KNN job recommendations")
            p1, p2, p3, p4 = st.columns(4)
            skills = p1.text_input("Your skills", placeholder="Python, SQL, Tableau")
            title = p2.text_input("Preferred role", placeholder="Data Analyst")
            preference = p3.selectbox("Remote preference", ["Any", "Remote", "On-site"])
            recommendation_k = p4.number_input("Matches (K)", min_value=1, max_value=20, value=5)
            p5, p6 = st.columns(2)
            preferred_location = p5.text_input("Preferred location", placeholder="Berlin")
            employment_type = p6.text_input("Employment type", placeholder="Full Time")
            if st.button("Find matching jobs"):
                recommendations = recommend_live_jobs(
                    visible, skills=skills, preferred_title=title,
                    location=preferred_location, employment_type=employment_type,
                    remote_preference=preference, k=recommendation_k,
                )
                if recommendations.empty:
                    st.warning("No matches found. Add skills or broaden filters.")
                else:
                    st.dataframe(recommendations, hide_index=True, use_container_width=True)
                    st.caption("Match score is text similarity, not a hiring probability.")

            st.subheader("Live market snapshot")
            a1, a2, a3 = st.columns(3)
            a1.bar_chart(visible["job_title"].value_counts().head(10))
            a2.bar_chart(visible["location"].value_counts().head(10))
            a3.bar_chart(skill_demand(visible))

            st.subheader("Skill gap analyzer")
            gap_skills = st.text_input("Skills you already have", key="gap_skills", placeholder="Python, SQL")
            gap_role = st.text_input("Target role for gap analysis", key="gap_role", placeholder="Data Analyst")
            if st.button("Analyze skill gap"):
                if gap_role.strip():
                    role_mask = visible["job_title"].fillna("").str.contains(
                        gap_role.strip(), case=False, regex=False
                    )
                    matching_jobs = visible.loc[role_mask]
                else:
                    matching_jobs = visible

                sample_size = len(matching_jobs)
                st.caption(f"Analyzed {sample_size} matching job posting(s).")

                if sample_size == 0:
                    st.info("No matching jobs found. Try another role or clear the job filters.")
                else:
                    if sample_size < 5:
                        st.warning(
                            "Limited data: fewer than 5 matching job postings. "
                            "Skill recommendations may not represent the wider job market."
                        )

                    matched, missing = skill_gap(gap_skills, matching_jobs)
                    st.success(
                        "You already match: "
                        + (", ".join(matched) or "No common listed skills yet.")
                    )
                    st.warning(
                        "Skills mentioned in postings to consider learning: "
                        + (", ".join(missing[:10]) or "No additional skills found.")
                    )
                    st.caption(
                        "Skills are ranked by keyword frequency in matching job postings, "
                        "not by verified importance or mandatory requirements."
                    )

            st.subheader("Save a job application")
            selectable = visible[["job_id", "job_title", "company", "url"]].copy()
            if selectable.empty:
                st.info("Clear filters to save a job application.")
            else:
                selectable["label"] = selectable["job_title"] + " — " + selectable["company"]
                selected = st.selectbox("Live job", selectable["label"], key="application_job")
                if st.button("Save application"):
                    job = selectable.loc[selectable["label"] == selected].iloc[0]
                    add_application(job.to_dict())
                    st.success("Application saved. Manage its status in Application Tracker.")

    elif menu == "Application Tracker":
        st.header("Job Application Tracker")
        applications = load_applications()
        if applications.empty:
            st.info("Save a live job from Live Job Feed to start tracking applications.")
        else:
            counts = applications["status"].value_counts()
            st.bar_chart(counts)
            st.dataframe(applications[["job_title", "company", "status", "applied_at", "url", "notes"]], hide_index=True, use_container_width=True)
            selected_id = st.selectbox("Application to update", applications["application_id"])
            new_status = st.selectbox("New status", ["Saved", "Applied", "Interview", "Offer", "Rejected"])
            if st.button("Update application status"):
                update_application_status(int(selected_id), new_status)
                st.success("Application status updated.")

    elif menu == "Career Tools":
        st.caption(
            "Low-memory career intelligence: reuses stored SQLite live jobs and any loaded historical dataset "
            "without making extra API requests or duplicating large DataFrames."
        )
        live_jobs = load_live_jobs(columns=CAREER_TOOLS_LIVE_COLUMNS)
        historical = st.session_state.get("data")

        c1, c2, c3 = st.columns(3)
        c1.metric("Stored Live Jobs", f"{len(live_jobs):,}")
        c2.metric("Loaded Historical Jobs", f"{len(historical):,}" if historical is not None else "Not loaded")
        c3.metric(
            "Active Resume",
            st.session_state.get("resume_name", "None uploaded"),
        )

        tab_forecast, tab_resume, tab_match, tab_compare, tab_skill_trends, tab_freshness = st.tabs([
            "1. Job Demand Forecasting",
            "2. Resume Analyzer & Roadmap",
            "3. Resume–Job Match",
            "4. Career Role Comparison",
            "5. Emerging vs Declining Skills",
            "6. Job Expiry & Freshness",
        ])

        # --------------------------------------------------
        # TAB 1: JOB DEMAND FORECASTING
        # --------------------------------------------------
        with tab_forecast:
            st.subheader("Job Demand Forecasting & Market Trends")
            category_options = ["All"] + CORE_ROLES + [cat for cat, _ in ROLE_CATEGORIES]
            top_live_skills = list(skill_demand(live_jobs, 15).index) if not live_jobs.empty else []
            skill_options = ["All"] + (top_live_skills or ["Python", "SQL", "JavaScript", "Docker", "AWS", "Machine Learning"])

            fc1, fc2, fc3, fc4 = st.columns(4)
            role_filter = fc1.selectbox("Role or Category filter", category_options, key="fc_role")
            skill_filter = fc2.selectbox("Skill filter", skill_options, key="fc_skill")
            gran_label = fc3.selectbox("Time granularity", ["Auto", "Monthly (M)", "Weekly (W)", "Daily (D)"], key="fc_gran")
            horizon = fc4.number_input("Forecast horizon (periods)", min_value=1, max_value=6, value=3, key="fc_horizon")

            gran_map = {"Auto": "auto", "Monthly (M)": "M", "Weekly (W)": "W", "Daily (D)": "D"}
            trend = forecast_demand(
                live_jobs,
                historical_jobs=historical,
                role_filter="" if role_filter == "All" else role_filter,
                skill_filter="" if skill_filter == "All" else skill_filter,
                granularity=gran_map[gran_label],
                horizon=int(horizon),
            )

            hist_snap = trend.attrs.get("historical_snapshot_count", 0)
            if hist_snap > 0:
                st.caption(
                    f"Historical CSV snapshot matching filter: **{hist_snap:,} postings** "
                    "(default historical CSV has no timestamp column, so time-series uses dated postings)."
                )

            if trend.empty:
                st.info("No dated job postings match the current filters. Refresh Live Jobs or broaden filters.")
            else:
                st.info(trend.attrs.get("status_message", ""))
                actual_df = trend[trend["record_type"] == "Actual"]
                forecast_df = trend[trend["record_type"] == "Forecast"]

                fig_fc, ax_fc = plt.subplots(figsize=(9, 3.6))
                fig_fc.patch.set_facecolor("white")

                x_actual = list(range(len(actual_df)))
                y_actual = actual_df["actual_jobs"].astype(float).tolist()
                ax_fc.plot(
                    x_actual,
                    y_actual,
                    marker="o",
                    linewidth=2.2,
                    color="#00796b",
                    label="Actual Demand (Observed)",
                )

                if not forecast_df.empty and trend["forecast_jobs"].notna().any():
                    x_forecast = list(range(len(actual_df) - 1, len(actual_df) + len(forecast_df)))
                    y_forecast = [y_actual[-1]] + forecast_df["forecast_jobs"].astype(float).tolist()
                    ax_fc.plot(
                        x_forecast,
                        y_forecast,
                        marker="s",
                        linestyle="--",
                        linewidth=2.2,
                        color="#d97706",
                        label="Forecasted Demand (OLS Regression)",
                    )

                all_labels = trend["period"].tolist()
                ax_fc.set_xticks(range(len(all_labels)))
                ax_fc.set_xticklabels(all_labels, rotation=25, ha="right", fontsize=9)
                ax_fc.set_ylabel("Job Postings")
                ax_fc.spines["top"].set_visible(False)
                ax_fc.spines["right"].set_visible(False)
                ax_fc.legend()
                st.pyplot(fig_fc)
                plt.close(fig_fc)

                st.dataframe(
                    trend[["period", "record_type", "actual_jobs", "forecast_jobs"]].rename(
                        columns={
                            "period": "Period",
                            "record_type": "Type",
                            "actual_jobs": "Actual Postings",
                            "forecast_jobs": "Forecasted Postings",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            st.markdown("#### Demand Breakdown by Title, Category, and Skill")
            summaries = demand_summary(live_jobs, historical_jobs=historical, top_n=8)
            s1, s2, s3 = st.columns(3)
            with s1:
                st.caption("Top Job Titles")
                st.dataframe(summaries["titles"], hide_index=True, use_container_width=True)
            with s2:
                st.caption("Top Role Categories")
                st.dataframe(summaries["categories"], hide_index=True, use_container_width=True)
            with s3:
                st.caption("Top In-Demand Skills (Live)")
                st.dataframe(summaries["skills"], hide_index=True, use_container_width=True)

        # --------------------------------------------------
        # TAB 2: RESUME ANALYZER + LEARNING ROADMAP
        # --------------------------------------------------
        with tab_resume:
            st.subheader("Resume Analyzer & Personalized Learning Roadmap")
            ru1, ru2 = st.columns([2, 1])
            resume_file = ru1.file_uploader(
                "Upload resume (PDF or DOCX, max 5 MB — processed in memory only)",
                type=["pdf", "docx"],
            )
            role_choice = ru2.selectbox("Target role", CORE_ROLES + ["Custom..."], key="ra_role")
            if role_choice == "Custom...":
                target_role = st.text_input("Enter custom target role", value="Data Engineer", key="ra_custom_role")
            else:
                target_role = role_choice

            b1, b2 = st.columns([1, 1])
            if b1.button("Analyze Resume"):
                if resume_file is None:
                    st.warning("Please upload a PDF or DOCX resume first.")
                else:
                    try:
                        extracted_text = extract_resume_text(resume_file)
                        st.session_state.resume_text = extracted_text
                        st.session_state.resume_skills = resume_skills(extracted_text)
                        st.session_state.resume_name = resume_file.name
                        st.session_state.pop("resume_match_cache", None)
                        st.success(f"Analyzed '{resume_file.name}' ({len(st.session_state.resume_skills)} skills detected).")
                    except (ValueError, KeyError) as exc:
                        st.error(str(exc))

            if b2.button("Clear Resume from Memory"):
                for key in ("resume_text", "resume_skills", "resume_name", "resume_match_cache"):
                    st.session_state.pop(key, None)
                st.info("Cleared resume text and cached matches from session memory.")

            if st.session_state.get("resume_text"):
                current_skills = st.session_state.get("resume_skills", [])
                plan = build_learning_roadmap(
                    current_skills,
                    live_jobs,
                    target_role,
                    historical_jobs=historical,
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Detected Resume Skills", len(current_skills))
                m2.metric(f"{target_role} Readiness", f"{plan['readiness_pct']:.0f}%")
                m3.metric("Role Postings Analyzed", f"{plan['live_matches'] + plan['historical_matches']:,}")
                st.caption(f"Requirement source: {plan['data_source']}")

                st.write("**Detected skills in your resume:** " + (", ".join(current_skills) or "None detected"))
                st.success("**Role-ready (matched) skills:** " + (", ".join(plan["matched_skills"]) or "None yet"))
                st.warning("**Missing skills for target role:** " + (", ".join(plan["missing_skills"]) or "None — you cover all listed skills!"))

                if not plan["roadmap_df"].empty:
                    st.markdown("#### Prioritized Learning Roadmap")
                    st.dataframe(plan["roadmap_df"], hide_index=True, use_container_width=True)
            else:
                st.info("Upload a PDF or DOCX resume and click **Analyze Resume** to generate your personalized roadmap.")

        # --------------------------------------------------
        # TAB 3: RESUME–JOB MATCHING & RECOMMENDATIONS
        # --------------------------------------------------
        with tab_match:
            st.subheader("Resume–Job Matching & Recommendations (Sparse TF-IDF KNN)")
            st.caption(
                "Reuses the project's from-scratch TF-IDF + cosine-similarity KNN with sparse matrix operations. "
                "No live API requests are triggered during matching."
            )
            if not st.session_state.get("resume_text"):
                st.info("Upload and analyze your resume in Tab 2 first to unlock job recommendations.")
            elif live_jobs.empty and historical is None:
                st.warning("No job datasets are available yet. Refresh Live Jobs or load the historical CSV in the sidebar.")
            else:
                mc1, mc2, mc3 = st.columns(3)
                source_mode = mc1.selectbox(
                    "Job source",
                    ["Both", "Live only", "Historical only"],
                    index=0,
                    key="rm_source",
                )
                match_role = mc2.selectbox(
                    "Target role focus",
                    [target_role] + [r for r in CORE_ROLES if r != target_role] + ["Any"],
                    index=0,
                    key="rm_role",
                )
                top_k = mc3.number_input("Top recommendations (K)", min_value=1, max_value=15, value=5, key="rm_k")

                effective_role = "" if match_role == "Any" else match_role
                cache_sig = (
                    st.session_state.get("resume_name", ""),
                    len(st.session_state.get("resume_text", "")),
                    source_mode,
                    effective_role,
                    int(top_k),
                    len(live_jobs),
                    len(historical) if historical is not None else 0,
                )

                run_match = st.button("Recommend Matching Jobs")
                cached_match = st.session_state.get("resume_match_cache")

                if run_match or (cached_match and cached_match[0] == cache_sig):
                    if run_match or not cached_match or cached_match[0] != cache_sig:
                        with st.spinner("Computing sparse TF-IDF similarity..."):
                            matches = matching_jobs(
                                st.session_state.resume_text,
                                live_jobs,
                                historical_jobs=historical,
                                k=int(top_k),
                                target_role=effective_role,
                                source_mode=source_mode,
                                extracted_skills=st.session_state.get("resume_skills"),
                            )
                        st.session_state.resume_match_cache = (cache_sig, matches)
                    else:
                        matches = cached_match[1]

                    if matches.empty:
                        st.warning("No matching jobs found for the selected source and filters.")
                    else:
                        display_cols = [
                            c for c in (
                                "source", "job_title", "company", "location",
                                "match_score", "matched_skills", "missing_skills",
                                "match_reason", "url",
                            )
                            if c in matches.columns
                        ]
                        st.dataframe(
                            matches[display_cols],
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "url": st.column_config.LinkColumn("Job Link"),
                                "match_score": st.column_config.NumberColumn("Match Score (%)", format="%.1f%%"),
                            },
                        )
                        st.caption("Match score reflects text & skill similarity, not hiring probability.")

        # --------------------------------------------------
        # TAB 4: CAREER ROLE COMPARISON
        # --------------------------------------------------
        with tab_compare:
            st.subheader("Career Role Comparison")
            st.caption(
                "Compares in-demand skills, resume skill overlap, missing skills, available job counts, "
                "historical demand trends, and IQR-validated salary data (shown only when >= 5 valid records exist)."
            )
            selected_roles = st.multiselect(
                "Select career roles to compare",
                CORE_ROLES,
                default=["Data Analyst", "Data Scientist", "Software Engineer"],
                key="rc_roles",
            )
            if not selected_roles:
                st.info("Select at least one role above to compare.")
            else:
                comparison_df, role_trends_df = compare_career_roles(
                    selected_roles,
                    live_jobs,
                    historical_jobs=historical,
                    resume_skills_list=st.session_state.get("resume_skills"),
                )
                st.dataframe(comparison_df, hide_index=True, use_container_width=True)

                rc_col1, rc_col2 = st.columns(2)
                with rc_col1:
                    st.caption("Available Job Counts by Role")
                    counts_chart_df = comparison_df.set_index("Role")[["Live Jobs", "Historical Jobs"]]
                    st.bar_chart(counts_chart_df)
                with rc_col2:
                    st.caption("Dated Postings Trend by Role")
                    if not role_trends_df.empty:
                        st.line_chart(role_trends_df)
                    else:
                        st.info("No dated postings matched the selected roles for a time-series comparison.")

        # --------------------------------------------------
        # TAB 5: EMERGING VS DECLINING SKILLS (Feature 1.2)
        # --------------------------------------------------
        with tab_skill_trends:
            st.subheader("Emerging vs Declining Skills (Trends Within Available Dataset)")
            st.caption(
                "Compares skill demand share (% of unique job postings per period) across recent vs previous periods "
                "so overall job-volume fluctuations do not falsely label a skill as declining."
            )

            st_col1, st_col2, st_col3, st_col4 = st.columns(4)
            st_role = st_col1.selectbox(
                "Role filter",
                ["All"] + CORE_ROLES,
                key="st_role_filter",
            )
            st_loc = st_col2.text_input("Location filter", placeholder="e.g. Berlin, Remote", key="st_loc_filter")
            st_exp = st_col3.selectbox(
                "Experience level",
                ["All"] + list(EXPERIENCE_LEVEL_PATTERNS.keys()),
                key="st_exp_filter",
            )
            st_period_label = st_col4.selectbox(
                "Comparison period mode",
                ["Auto (Best Available)", "Monthly", "Weekly", "Daily"],
                key="st_period_mode",
            )

            st_cfg1, st_cfg2, st_cfg3, st_cfg4 = st.columns(4)
            st_min_mentions = st_cfg1.number_input(
                "Min postings per skill",
                min_value=2,
                max_value=20,
                value=DEFAULT_MIN_SKILL_POSTINGS,
                key="st_min_mentions",
            )
            st_thresh_pct = st_cfg2.number_input(
                "Change threshold (%)",
                min_value=5.0,
                max_value=100.0,
                value=DEFAULT_TREND_CHANGE_PCT,
                step=5.0,
                key="st_thresh_pct",
            )
            st_start_date = st_cfg3.text_input("Start date (YYYY-MM-DD, optional)", value="", key="st_start_dt")
            st_end_date = st_cfg4.text_input("End date (YYYY-MM-DD, optional)", value="", key="st_end_dt")

            period_mode_map = {
                "Auto (Best Available)": "auto",
                "Monthly": "month",
                "Weekly": "week",
                "Daily": "day",
            }

            trend_res = analyze_skill_trends(
                live_jobs,
                historical_jobs=historical,
                period_mode=period_mode_map[st_period_label],
                role_filter="" if st_role == "All" else st_role,
                location_filter=st_loc,
                experience_filter="" if st_exp == "All" else st_exp,
                start_date=st_start_date.strip() or None,
                end_date=st_end_date.strip() or None,
                min_skill_postings=int(st_min_mentions),
                change_threshold_pct=float(st_thresh_pct),
            )

            if trend_res.get("coverage_warning"):
                st.warning(trend_res["coverage_warning"])

            if not trend_res["available"]:
                st.info(trend_res["status_message"])
            else:
                st.info(trend_res["status_message"])
                p_info = trend_res["period_info"]
                tm1, tm2, tm3, tm4 = st.columns(4)
                tm1.metric("Emerging Skills", len(trend_res["emerging_df"]))
                tm2.metric("Stable Skills", len(trend_res["stable_df"]))
                tm3.metric("Declining Skills", len(trend_res["declining_df"]))
                tm4.metric("Dated Postings Analyzed", f"{p_info['total_analyzed_jobs']:,}")

                ch_col1, ch_col2 = st.columns(2)
                with ch_col1:
                    st.markdown("#### Emerging Skills (Share Increase)")
                    em_df = trend_res["emerging_df"].head(10)
                    if em_df.empty:
                        st.caption("No skills met the Emerging threshold for the selected filters.")
                    else:
                        fig_em, ax_em = plt.subplots(figsize=(6.5, 3.4))
                        fig_em.patch.set_facecolor("white")
                        ax_em.barh(
                            em_df["Skill"][::-1],
                            em_df["Share Change (pp)"][::-1],
                            color="#00796b",
                            height=0.6,
                        )
                        ax_em.set_xlabel("Demand Share Change (+ percentage points)")
                        ax_em.spines["top"].set_visible(False)
                        ax_em.spines["right"].set_visible(False)
                        st.pyplot(fig_em)
                        plt.close(fig_em)

                with ch_col2:
                    st.markdown("#### Declining Skills (Share Decrease)")
                    dec_df = trend_res["declining_df"].head(10)
                    if dec_df.empty:
                        st.caption("No skills met the Declining threshold with sufficient prior coverage.")
                    else:
                        fig_dec, ax_dec = plt.subplots(figsize=(6.5, 3.4))
                        fig_dec.patch.set_facecolor("white")
                        ax_dec.barh(
                            dec_df["Skill"][::-1],
                            dec_df["Share Change (pp)"][::-1],
                            color="#be123c",
                            height=0.6,
                        )
                        ax_dec.set_xlabel("Demand Share Change (- percentage points)")
                        ax_dec.spines["top"].set_visible(False)
                        ax_dec.spines["right"].set_visible(False)
                        st.pyplot(fig_dec)
                        plt.close(fig_dec)

                timeline_df = trend_res["timeline_df"]
                if not timeline_df.empty:
                    st.markdown("#### Skill Demand Share Over Time (% of Period Postings)")
                    avail_skills = list(timeline_df.columns)
                    chosen_timeline_skills = st.multiselect(
                        "Skills to plot over time",
                        avail_skills,
                        default=avail_skills[:5],
                        key="st_timeline_skills",
                    )
                    if chosen_timeline_skills:
                        st.line_chart(timeline_df[chosen_timeline_skills])

                st.markdown("#### Skill Trend Classification Table (Within Available Dataset)")
                class_filter = st.selectbox(
                    "Filter table by classification",
                    ["All", "Emerging", "Stable", "Declining"],
                    key="st_class_filter",
                )
                table_df = trend_res["trends_df"]
                if class_filter != "All" and not table_df.empty:
                    table_df = table_df[table_df["Trend Classification"] == class_filter]

                display_trend_cols = [
                    "Skill",
                    "Current Demand",
                    "Previous Demand",
                    "Share Change (pp)",
                    "Percentage Change",
                    "Trend Classification",
                    "Skill Postings",
                    "Total Postings Analyzed",
                ]
                st.dataframe(table_df[display_trend_cols], hide_index=True, use_container_width=True)

        # --------------------------------------------------
        # TAB 6: JOB EXPIRY & FRESHNESS TRACKER (Feature 2.2)
        # --------------------------------------------------
        with tab_freshness:
            st.subheader("Job Expiry & Freshness Tracker")
            st.caption(
                "Tracks job posting age and freshness without deleting older jobs. "
                "A stale job is labeled 'Stale / possibly outdated' and preserved for historical analytics; "
                "jobs are only marked 'Expired' when the source explicitly confirms expiry or closure."
            )

            fr_c1, fr_c2, fr_c3, fr_c4 = st.columns(4)
            cfg_fresh = fr_c1.number_input(
                "Fresh threshold (max days)",
                min_value=1,
                max_value=60,
                value=int(st.session_state.get("cfg_fresh_days", DEFAULT_FRESH_DAYS)),
                key="fr_fresh_days",
            )
            cfg_aging = fr_c2.number_input(
                "Aging threshold (max days)",
                min_value=int(cfg_fresh),
                max_value=365,
                value=max(int(cfg_fresh), int(st.session_state.get("cfg_aging_days", DEFAULT_AGING_DAYS))),
                key="fr_aging_days",
            )
            st.session_state.cfg_fresh_days = int(cfg_fresh)
            st.session_state.cfg_aging_days = int(cfg_aging)

            fr_source = fr_c3.selectbox(
                "Job source",
                ["Live only", "Both", "Historical only"],
                index=0,
                key="fr_source_mode",
            )
            fr_cat = fr_c4.selectbox(
                "Freshness category filter",
                ["All", "Fresh", "Aging", "Stale / possibly outdated", "Expired", "Missing / Invalid Date"],
                index=0,
                key="fr_cat_filter",
            )

            fr_s1, fr_s2 = st.columns([2, 1])
            fr_kw = fr_s1.text_input("Filter by job title, company, or location", value="", key="fr_keyword")
            fr_sort_label = fr_s2.selectbox(
                "Sort postings by",
                ["Newest posting first", "Freshness category", "Oldest posting first"],
                index=0,
                key="fr_sort_by",
            )
            sort_map = {
                "Newest posting first": "newest",
                "Freshness category": "freshness",
                "Oldest posting first": "oldest",
            }

            fresh_df, f_sum = evaluate_jobs_freshness(
                live_jobs,
                historical_jobs=historical,
                fresh_days=int(cfg_fresh),
                aging_days=int(cfg_aging),
                source_mode=fr_source,
                freshness_filter=fr_cat,
                keyword=fr_kw,
                sort_by=sort_map[fr_sort_label],
            )

            fm1, fm2, fm3, fm4, fm5 = st.columns(5)
            fm1.metric(f"🟢 Fresh (0–{cfg_fresh}d)", f"{f_sum['Fresh']:,}")
            fm2.metric(f"🟡 Aging ({cfg_fresh + 1}–{cfg_aging}d)", f"{f_sum['Aging']:,}")
            fm3.metric(f"🟠 Stale (>{cfg_aging}d)", f"{f_sum['Stale / possibly outdated']:,}")
            fm4.metric("🔴 Expired (Confirmed)", f"{f_sum['Expired']:,}")
            fm5.metric("⚪ Missing / Invalid Date", f"{f_sum['Missing / Invalid Date']:,}")

            if f_sum.get("undated_historical_omitted", 0) > 0:
                st.caption(
                    f"Note: The default historical CSV has no posting timestamp column. "
                    f"All {f_sum['Missing / Invalid Date']:,} undated historical records are counted under "
                    f"'Missing / Invalid Date', while table display is capped at 1,000 historical rows to keep RAM usage minimal."
                )

            if fresh_df.empty:
                st.info("No postings match the selected freshness filters.")
            else:
                display_fresh = fresh_df[[
                    "job_title", "company", "location", "published_date",
                    "age_days", "freshness_badge", "last_checked", "source", "url",
                ]].rename(columns={
                    "job_title": "Job Title",
                    "company": "Company",
                    "location": "Location",
                    "published_date": "Published Date",
                    "age_days": "Age (Days)",
                    "freshness_badge": "Freshness Category",
                    "last_checked": "Last Checked / Fetched",
                    "source": "Source",
                    "url": "URL",
                })
                st.dataframe(
                    display_fresh,
                    hide_index=True,
                    use_container_width=True,
                    column_config={"URL": st.column_config.LinkColumn("Job Link")},
                )
