# 📊 Exploratory Module – Understand Your Data Before You Model It

Welcome to the **Exploratory module** of the `data-science-hack-functions` library — a toolkit built for fast, functional, and **human-readable data understanding**.

If you’ve ever:
- Started training a model only to find out your data is garbage
- Got 95% accuracy but didn’t realize one feature was a constant
- Skipped EDA because it was “too manual”...

This module is for you.

---

## 🔍 What This Module Helps You Do

The exploratory module helps you **understand the shape and quality of your dataset** before you train any model.

It answers:
- Which columns are mostly missing?
- Are there constant or low-cardinality features?
- Which features are skewed or heavy-tailed?
- Where are the outliers?
- How clean is this dataset, really?

With a single function call, you get:
- Scrollable HTML tables for **Jupyter-friendly summaries**
- Descriptive statistics for **numeric and categorical** data
- Entropy and cardinality scores for better feature analysis
- Z-score-based outlier detection
- Optional return of all outputs as DataFrames

---

## 🧱 Functions in This Module

### 🔧 `summary_dataframe(df, detailing=True, correlation_matrix=True)`

Generate a **complete overview of your entire DataFrame**, with automatic detection of:
- Constant columns
- Missing values and their percentage
- Data types and unique values
- Optional: skewness, kurtosis, entropy, Z-score outliers
- Optional: correlation matrix

> Perfect for: **initial EDA**, model preprocessing, and feature pruning.

---

### 🔧 `summary_column(df, column_name='target', plots=['histogram'])`

Zoom in on one column — numerical or categorical — and explore:
- Summary stats: variance, IQR, mode, skewness, etc.
- Visuals: histogram/bar chart for top values
- Missing values trend over time (if time column provided)
- Entropy for categorical features

> Perfect for: **target column analysis**, understanding categorical distributions, or checking for class imbalance.

---

## 📸 Visual Output

Each function optionally produces clean visual output:

- 📊 **HTML Tables** for interactive scrolling
- 📈 **Histograms, bar charts, and correlation plots**
- 🎯 **Z-score outliers and entropy per feature**
- 🧩 **Optional return of all data as pandas DataFrames**

---

## 🧪 Real-World Example

```python
from data_science_hack_functions.exploratory import summary_dataframe, summary_column

# Summarize the whole dataset
summary_dataframe(df, detailing=True, correlation_matrix=True)

# Analyze a single column with histogram and entropy
summary_column(df, column_name='age', detailing=True, plots=['histogram'])
