# ZeroPlastic India Intelligence Dashboard

Data-driven analytics for ZeroPlastic India — a sustainable home & personal care startup eliminating plastic packaging across shampoo bars, dishwash blocks, laundry sheets and more.

## Quick Deploy to Streamlit Cloud

1. **Unzip** this file — you get a flat folder with 4 files + `.streamlit/` folder
2. **Create a new GitHub repository** and upload all files (including the `.streamlit/` folder)
3. Go to **share.streamlit.io** → New app → select your repo → Main file: `app.py` → Deploy

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## File Structure (flat — no nested sub-folders)

```
app.py                       ← Single-file Streamlit app (all 5 pages)
zeroplastic_dataset.csv      ← 2,000 row synthetic Indian consumer survey dataset
requirements.txt             ← Dependencies
.streamlit/config.toml       ← Theme config (ZeroPlastic teal)
README.md                    ← This file
```

## Dashboard Pages

- **📊 Descriptive** — Demographics, eco-spend distribution, format interest rates, city × intent heatmap, discovery channels, Ayurveda affinity, aspiration gap
- **🔍 Diagnostic** — Eco-skeptic paradox, psychographic correlations, trust chain analysis, habit strength vs conversion, chi-square significance tests, hard water impact
- **🤖 Predictive** — Classification (RF + LR + Gradient Boosting + XGBoost): accuracy/precision/recall/F1/ROC-AUC + confusion matrix + feature importance · Clustering (K-Means): elbow/silhouette/PCA/radar · Association Rules (Apriori): support/confidence/lift · Regression (RF + Ridge): eco-spend prediction
- **🎯 Prescriptive** — Segment action cards, budget allocation, conversion funnel, seasonal campaign calendar
- **📥 Upload & Score** — Upload new survey CSV → auto-score with buy probability + predicted eco-spend + cluster + marketing action → download results

## Dataset

`zeroplastic_dataset.csv` contains 2,000 synthetic Indian consumer respondents across 5 seeded personas:
- **Eco Warrior** (12%) — Core identity eco-buyer, highest LTV, subscribes readily
- **Mindful Parent** (22%) — Safety-first family buyer, Amazon + Nykaa channel
- **Eco Skeptic** (18%) — High eco-concern but greenwashing distrust, needs proof
- **Urban Renter** (28%) — Convenience-first, low habit lock, quick-commerce native
- **Price Pragmatist** (20%) — Value-driven, Tier 2/3, needs per-wash cost story

## Tech Stack

| Component | Library |
|-----------|---------|
| App framework | Streamlit ≥ 1.32 |
| Data processing | pandas ≥ 2.0, numpy ≥ 1.24 |
| Machine learning | scikit-learn ≥ 1.3 |
| Visualisation | Plotly ≥ 5.18 |
| Association rules | mlxtend ≥ 0.22 |
| Statistics | scipy ≥ 1.10 |
| Gradient boosting | xgboost ≥ 1.7 (optional, falls back gracefully) |
