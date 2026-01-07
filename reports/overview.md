# Yelp Rating Risk Monitor

YELP RATING ALERT SYSTEM


0. Overview

This project builds an end-to-end early-warning system that identifies
restaurants on Yelp that are at risk of a meaningful rating decline in the
near future. It integrates:

1. Data preparation and feature engineering from raw Yelp reviews
2. An optimized machine learning model that predicts future rating drops
3. LLM-based root cause analysis for qualitative insights
4. Interactive dashboards for risk monitoring and operational decisions

All components run in a reproducible four-step pipeline, each implemented
in its own Jupyter notebook and accompanied by a structured text report.


======================================================================
1. Data Preparation (Step 1 – 1_data_prep.ipynb)
======================================================================

1.1 Raw data sources

We use two files from the Yelp Academic Dataset:

- yelp_academic_dataset_business.json
- yelp_academic_dataset_review.json

We filter the dataset to restaurant businesses located in Philadelphia.
The processed dataset contains ~100,000 reviews from ~5,300 restaurants
spanning 2005–2022.

1.2 Cleaning and preprocessing

Key cleaning steps include:

- Removing reviews with missing ratings, text, or invalid timestamps
- Normalizing date formats
- Merging business attributes (name, stars_business, location)
- Restricting to valid restaurant categories

Outputs:

- data/processed/restaurants.csv
- data/processed/reviews_initial.csv
- data/processed/reviews_clean.csv

1.3 Sentiment & time-series feature engineering

We compute:

- text_length
- sentiment score via VADER
- 7-day and 14-day rolling averages: rating_7d, rating_14d, sentiment_7d, sentiment_14d
- rating_volatility, rating_momentum, sentiment_momentum

The resulting time-series structured file:

- data/processed/reviews_features.csv

Serves as input to Step 2 (model training).


======================================================================
2. Optimized Rating Drop Prediction Model
(Step 2 – 2_model_random_forest_OPTIMIZED.ipynb)
======================================================================

2.1 Business-driven label definition (no leakage)

A rating drop event is defined as:

- rating_7d_future: 7-day average rating shifted 7 days forward
- label_rating_drop = 1 if rating_7d_future ≤ rating_7d − 0.3

This detects meaningful, sustained declines in customer experience.

To avoid label leakage:

- All future-looking columns (rating_7d_future, future_rating_14d, earlier
  rating_drop fields) are explicitly excluded from model inputs.
- Only historically observable numeric features are used.


2.2 Feature set (29 numeric features)

After excluding leakage-prone future features, we construct a fully numeric,
deployable feature matrix with **29 features**, including:

• Review-level
  - stars_review, text_length, sentiment
• Business-level rolling features
  - stars_business, rating_7d, rating_14d, sentiment_7d, sentiment_14d
  - rating_volatility, rating_momentum, sentiment_momentum
• Interaction and stability indicators
  - rating_sentiment_interaction, momentum_volatility
  - negative_ratio_7d, review_density
  - rating_range_7d, sentiment_range_7d
• Time indicators & second-order dynamics
  - day_of_week, month, is_weekend, is_holiday_season
  - rating_acceleration, sentiment_acceleration


2.3 Train–test split & SMOTE imbalance handling

After removing rows with missing feature/label values:

- Train shape: 60731 rows
- Test shape: 15183 rows

Class balance before SMOTE:
- label 0: 47349
- label 1: 13382
- minority/majority ratio = **0.283**

Because **0.283 < 0.300**, SMOTE is applied:

- sampling_strategy=0.30
- Balanced train size: 61553 rows
- Final class ratio: 0.769 / 0.231

This dynamic rule ensures SMOTE is used **only when needed**.


2.4 Benchmark models

We benchmarked:

1. Time-series logistic regression
2. Logistic regression on all numeric features
3. Gradient Boosting classifier
4. Optimized Random Forest (final)

The optimized Random Forest achieved the strongest combination of recall,
precision, AUC, and F1.


2.5 Final model performance (test set)

The optimized Random Forest achieves the following performance on the
held-out test set:

- ROC AUC: 0.753
- Accuracy: 0.703 (70.3%)
- Precision: 0.396 (39.6%)
- Recall: 0.661 (66.1%)
- F1-Score: 0.495
- Optimal Threshold: 0.250
- Features: 29 (rigorously leakage-free)

Performance interpretation:

ROC AUC of 0.753 indicates strong discriminative ability - the model can
effectively rank restaurants by their risk of rating decline. This represents
a 50% improvement over random guessing (AUC 0.50).

The recall of 66% means the system successfully identifies two-thirds of
restaurants that will experience rating drops in the next 7 days. This is
appropriate for an early-warning system where the cost of missing a true
drop (restaurant loses customers without intervention) exceeds the cost of
a false alarm (unnecessary review of a stable restaurant).

The precision of 40% indicates that among flagged restaurants, 40% will
actually experience drops. While this creates some false positives, it is
acceptable given:
1. The high cost of false negatives in this business context
2. The relatively low cost of reviewing flagged restaurants
3. The early-warning nature of the system (proactive rather than reactive)

These metrics represent true deployable performance with zero data leakage.
All 29 features use only historical and current data available at prediction
time. The model can be deployed in production with confidence that actual
performance will match these evaluation metrics.



2.6 Threshold tuning for alert decisions

We search thresholds from 0.0 to 1.0 and select the one maximizing F1.
The optimal threshold is 0.250, which:

- Improves recall (catch more true at-risk businesses)
- Keeps precision at a practical level
- Reflects early-warning priorities (better to alert early than miss events)


2.7 Model interpretability with SHAP

We use SHAP (TreeExplainer) to interpret the Random Forest:

Top global drivers include:

- rating_7d, sentiment_7d
- negative_ratio_7d
- rating_momentum, sentiment_momentum
- review_density
- volatility and range features

These insights help explain *why* a business is trending toward risk.

Saved artifacts include:

- rating_drop_model_optimized.pkl
- model_summary_optimized.csv
- feature_list_optimized.txt
- model_comparison_summary.csv
- SHAP & threshold optimization figures
- reviews_features_optimized.csv


======================================================================
3. LLM Root Cause Analysis
(Step 3 – 3_llm_root_cause_analysis.ipynb)
======================================================================

3.1 Purpose

While the ML model answers *which* restaurants are at risk, the LLM pipeline
answers *why*. Using a local Llama 3.2 model via Ollama, we extract structured,
interpretable categories from negative reviews.

3.2 Input and filtering

- Input: reviews_features_optimized.csv
- Select negative reviews (stars_review ≤ 3)
- Sample ~20 reviews for analysis per run

Extracted fields include:

- primary_issue
- category flags (food_quality, service_speed, etc.)
- severity (low/medium/high)
- explanation text

Results saved to:

- results/llm_complaint_analysis.csv


3.3 Aggregate insights

Typical outputs include:

- Food Quality: ~most common
- Staff Behavior: frequent theme
- Pricing / Portion Size: notable
- Severity split: meaningful distribution

These insights help operations teams determine *what to fix first*.


======================================================================
4. Dashboards and Reporting
(Step 4 – 4_dashboard_and_report_OPTIMIZED.ipynb)
======================================================================

4.1 Inputs

- reviews_features_optimized.csv
- rating_drop_model_optimized.pkl
- feature_list_optimized.txt
- model_summary_optimized.csv
- llm_complaint_analysis.csv (optional)

4.2 Risk scoring

We compute:

- risk_score = P(rating_drop | features)
- risk_flag = 1 if risk_score ≥ **0.290**

Business-level aggregation:

- avg_risk_score
- max_risk_score
- latest_rating
- n_reviews

Used to rank top high-risk restaurants.


4.3 Dashboards (Plotly HTML)

Generated dashboards:

1. **dashboard_overview.html**
   - Average rating vs average risk trends over time

2. **dashboard_alerts.html**
   - Top high-risk businesses (ranked by avg_risk_score)

3. **dashboard_llm_complaints.html**
   - Category + severity distribution of negative reviews


======================================================================
5. Final Pipeline Integration
(Step 5 – 5_final_project.ipynb)
======================================================================

This notebook orchestrates the full system end-to-end:

1. Data preparation
2. Model training + evaluation
3. LLM extraction
4. Dashboard generation

It also documents all major output files:

- Cleaned datasets
- Final feature-engineered dataset
- Optimized model + summary
- Risk dashboards
- SHAP and threshold figures
- LLM results
- Final project report (this file)


======================================================================
6. Limitations & Future Work
======================================================================

- Label threshold (0.3 stars) may be tuned using business impact metrics
- Apply time-based cross-validation for stronger temporal guarantees
- Expand LLM module to generate prescriptive insights
- Add causal modeling or sequence models for richer forecasting
- Extend system to multiple cities or cuisines


======================================================================
7. Conclusion
======================================================================
The Yelp Rating Alert System integrates data engineering, predictive
modeling, and LLM-based analytics into a coherent early-warning platform:

- Step 1 prepares high-quality review and time-series features from raw
  Yelp data covering 99,997 reviews across 5,393 Philadelphia restaurants.

- Step 2 delivers an optimized Random Forest model with honest, deployable
  performance (ROC AUC 0.753, F1 0.495, Recall 66%) under a business-driven
  rating drop definition. Rigorous feature selection ensures zero data
  leakage and true production readiness.

- Step 3 uses a local LLM (Llama 3.2) to translate unstructured review text
  into structured complaint categories and severities at zero cost.

- Step 4 surfaces the results via interactive dashboards and business
  intelligence reports, enabling operations teams to focus on highest-risk
  restaurants and understand underlying issues.

The system demonstrates how classical machine learning, time-series
engineering, and modern LLMs can be combined with rigorous methodology
to build practical, deployable risk monitoring systems. The emphasis on
data leakage prevention and honest performance evaluation ensures the
system will perform as expected in production deployment.
