# Dashboard Notes

======================================================================
DASHBOARD & BUSINESS INSIGHTS SUMMARY (OPTIMIZED MODEL)
======================================================================

Model Performance (Optimized Random Forest):
  ROC AUC   : 0.753
  Accuracy  : 0.703
  Precision : 0.396
  Recall    : 0.661
  F1-Score  : 0.495
  Threshold : 0.250
  Features  : 29 (leakage-free)

Performance Notes:
- Model uses only historical/current data (no future information)
- 66% recall ensures most rating drops are caught early
- 40% precision is acceptable for early-warning use case
- All metrics represent true deployable performance

Overall System Statistics:
  Businesses monitored : 4094
  Reviews analyzed     : 89878
  High-risk businesses : 1190

Top 10 high-risk businesses:
  - Xi'an Cuisine | avg_risk=0.617, latest_rating=3.00, reviews=6
  - Amasi Restaurant and Hookah | avg_risk=0.603, latest_rating=3.50, reviews=1
  - Magic Noodles | avg_risk=0.600, latest_rating=3.50, reviews=4
  - The Blockley | avg_risk=0.586, latest_rating=3.50, reviews=1
  - Andy's Chicken | avg_risk=0.576, latest_rating=4.00, reviews=6
  - Mr Wish | avg_risk=0.575, latest_rating=4.00, reviews=9
  - Fred's Water Ice | avg_risk=0.575, latest_rating=4.00, reviews=4
  - Founded Coffee & Pizza | avg_risk=0.572, latest_rating=4.00, reviews=7
  - The French Bakery | avg_risk=0.569, latest_rating=4.00, reviews=7
  - Nigiyaka Na | avg_risk=0.565, latest_rating=3.50, reviews=3

LLM Complaint Analysis (sampled negative reviews):
  Reviews analyzed by LLM : 19
  Top primary issues:
    • the bouncer did not ask for ID (1 reviews,  5.3%)
    • They skimp out on the grilled salmon (1 reviews,  5.3%)
    • Rude staff (1 reviews,  5.3%)
    • Terrible food, The meat was not even lamb... (1 reviews,  5.3%)
    • not very gluten free-friendly (1 reviews,  5.3%)
