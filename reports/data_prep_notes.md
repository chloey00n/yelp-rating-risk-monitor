# Data Preparation Notes

DATA PREPARATION REPORT
======================================================================

Notebook: 1_data_prep.ipynb
Purpose: Prepare, clean, and engineer foundational review-level data for all
subsequent components of the Rating Drop Early-Warning System.

======================================================================
1. OVERVIEW
----------------------------------------------------------------------

This notebook performs all preliminary data engineering steps required to
enable the predictive model, the LLM complaint analysis module, and the
business intelligence dashboard. The main objective is to transform the raw
Yelp dataset into a structured, reliable, and analysis-ready dataset with
review-level, business-level, temporal, and sentiment features.

This data pipeline serves as the backbone for the entire project.

======================================================================
2. DATA SOURCES
----------------------------------------------------------------------

The notebook uses two files from the Yelp Academic Dataset:

1. yelp_academic_dataset_business.json
2. yelp_academic_dataset_review.json

Filtering is applied to include only:

- Businesses located in Philadelphia
- Businesses categorized as restaurants
- All associated customer reviews for these restaurants

After filtering, the dataset includes:

- Total reviews: ~100,000+
- Restaurant businesses: ~5,300+
- Time coverage: 2005–2022

======================================================================
3. DATA CLEANING STEPS
----------------------------------------------------------------------

Key preprocessing operations performed:

1. Load JSON files line-by-line into pandas DataFrames.
2. Standardize all date formats using pandas datetime.
3. Remove entries with missing text, missing ratings, or invalid timestamps.
4. Filter to restaurant-related categories and Philadelphia geographic area.
5. Merge business information into the review dataset.
6. Ensure all text fields are UTF-8 encoded and normalized.

These operations ensure all downstream steps work with clean and
consistent data.

======================================================================
4. SENTIMENT ANALYSIS (VADER)
----------------------------------------------------------------------

The notebook applies VADER sentiment scoring to each review text.
Generated fields:

- sentiment: compound polarity score in [-1.0, 1.0]
- pos / neu / neg probabilities

VADER is chosen due to:

- Its strong performance on social-media-style text
- Zero cost
- No need for GPU or complex preprocessing

This sentiment score becomes a major input for both the optimized
Random Forest model and the LLM complaint analysis stage.

======================================================================
5. BASE FEATURE ENGINEERING
----------------------------------------------------------------------

Rolling-window features are generated to capture short-term trends:

- rating_7d: 7-day average star rating
- sentiment_7d: 7-day average VADER sentiment
- review_count_7d: number of reviews in past 7 days
- sentiment_momentum: week-over-week sentiment change
- rating_momentum: week-over-week rating change
- rating_volatility: rolling 7-day standard deviation
- sentiment_volatility: rolling 7-day standard deviation

These features form the minimum required signal for the
“future rating drop” prediction task.

======================================================================
6. OUTPUT FILES
----------------------------------------------------------------------

This notebook produces two key deliverables:

1. data/processed/reviews_clean.csv
   - Cleaned review-level dataset
   - Standardized, merged, filtered, sentiment-annotated

2. data/processed/reviews_features.csv
   - Feature-engineered dataset
   - Contains rolling metrics used in both the baseline and optimized
     Random Forest models

These files are direct inputs to:

- Optimized model training (Notebook 2)
- LLM root cause analysis (Notebook 3)
- Dashboard + Final Report generator (Notebook 4)

======================================================================
7. PROJECT IMPACT
----------------------------------------------------------------------

This data foundation enables:

- High-quality machine learning predictions
- Reliable complaint analysis using LLMs
- Accurate business-level risk signals
- Scalable dashboards and reporting

Without this preparation notebook, the entire pipeline would be unable to
produce consistent or meaningful outputs.
