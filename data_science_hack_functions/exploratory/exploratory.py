
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew, entropy
from tabulate import tabulate
from IPython.display import display, HTML

def summary_dataframe(df: pd.DataFrame, verbose: bool = True, return_dataframes: bool = False,
                      detailing: bool = False, correlation_matrix: bool = False, fast_mode: bool = False):
    """
    Generates a detailed summary report of an entire DataFrame for exploratory data analysis (EDA).

    This function provides a transparent, column-by-column overview of your dataset, including
    data types, missing value patterns, uniqueness, cardinality, and optionally deeper
    statistical insights like skewness, kurtosis, entropy, and correlation structure.

    It helps you understand the shape and quality of your data before modeling, and supports
    structured auditing via optional DataFrame outputs for downstream usage.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset to analyze. Must be a rectangular DataFrame with rows and columns.

    verbose : bool, default=True
        If True, prints human-readable summaries using styled tables, markdown, or HTML (in notebooks).
        If `fast_mode=True`, this will be ignored (no prints will be shown).

    return_dataframes : bool, default=False
        If True, returns the core summary tables as pandas DataFrames for programmatic use.

    detailing : bool, default=False
        If True, enables additional statistics:
        - For numeric: skewness, kurtosis, and z-score-based outlier counts
        - For categorical: entropy (how evenly distributed the values are)
        - Also shows duplicate row and column analysis

    correlation_matrix : bool, default=False
        If True, computes a Pearson correlation matrix for all numeric features.

    fast_mode : bool, default=False
        If True, disables all visual and computationally expensive diagnostics:
        - Skips entropy, skewness, kurtosis, outliers, and duplicate detection
        - Skips correlation matrix computation
        - Skips verbose display/logging
        Use this mode when profiling large datasets (e.g., 1M+ rows) or in batch workflows.

    Returns
    -------
    tuple of pd.DataFrame, optional
        Only returned if `return_dataframes=True`. Includes:
        - summary : Core metadata (dtype, missing %, unique %, etc.)
        - desc_numeric : Descriptive stats for all numeric columns
        - desc_categorical : Descriptive stats for categorical/object columns
        - correlation_matrix : Numeric correlation matrix (only if `correlation_matrix=True`)

    Raises
    ------
    ValueError
        If the input DataFrame is empty or not valid.

    Examples
    --------
    >>> summary_dataframe(df, detailing=True, correlation_matrix=True)

    >>> summary, num_stats, cat_stats = summary_dataframe(df, return_dataframes=True, detailing=True)

    Notes
    -----
    - The function is non-destructive: it reads from the input DataFrame without modifying it.
    - If you're working with extremely large datasets, set `fast_mode=True` to avoid slow diagnostics.
    - When `verbose=True`, this function uses IPython’s HTML renderer for a notebook-friendly display.

    See Also
    --------
    summary_column : Analyze a single column with detailed metrics and plots
    preprocess_dataframe : Prepare a dataset for modeling through scaling, encoding, and imputation
    preprocess_column : Clean a single column manually (e.g., outlier handling, transformation)

    User Guide
    ----------
    🧠 When Should You Use This?
    - At the start of a project to assess **data readiness**.
    - Before feature engineering to identify **columns to drop, fix, or transform**.
    - During EDA or notebook exploration to communicate **data quality**.
    - In automated pipelines where you need **programmatic summary outputs**.

    📌 What You'll Learn:
    - Which columns have high missingness, low variance, or high cardinality
    - How many numeric/categorical features exist
    - Skewness or entropy in features (if detailing=True)
    - Whether your dataset has duplicated rows or columns
    - Correlation patterns among numeric features (optional)

    ⚙️ Recommended Usage Patterns:

    1. **Full EDA diagnostic (notebooks):**
       >>> summary_dataframe(df, detailing=True, correlation_matrix=True)

    2. **For dashboards or programmatic reporting:**
       >>> summary, num_stats, cat_stats = summary_dataframe(df, return_dataframes=True)

    3. **Batch analysis or large files:**
       >>> summary_dataframe(df, fast_mode=True)

    4. **Minimal quick check (CLI or scripts):**
       >>> summary_dataframe(df, detailing=False, verbose=True)

    💡 Tips:
    - Use with `preprocess_dataframe()` to act on low-quality features you identify here.
    - `entropy` close to 0 → one category dominates (low information)
    - High skew/kurtosis → consider log or robust transformations
    - Z-score outliers >10 → column likely needs clipping or scaling
    - Use `fast_mode=True` when processing high-volume datasets or in production loops.
    - Use `detailing=False` for a quick overview of the dataset without deep stats.
    - Use `correlation_matrix=False` to skip the correlation matrix.
    - Use `return_dataframes=True` to export summaries to reports or ML audit logs
    """

    if fast_mode:
        detailing = False
        correlation_matrix = False
        verbose = False

    if df.empty:
        raise ValueError("The provided DataFrame is empty. Provide a valid dataset.")

    total_rows = df.shape[0]
    numeric_df = df.select_dtypes(include=["number"])
    categorical_df = df.select_dtypes(include=["object", "category"])

    summary = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.values,
        "Total Values": df.count().values,
        "Missing Values": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / total_rows * 100).round(2),
        "Unique Values": df.nunique().values,
        "Unique %": (df.nunique().values / total_rows * 100).round(2)
    })

    summary["Constant Column"] = summary["Unique Values"] == 1
    summary["Cardinality Category"] = summary["Unique Values"].apply(
        lambda x: "Low" if x <= 10 else "Medium" if x <= 100 else "High"
    )

    if detailing:
        duplicate_rows = df.duplicated().sum()
        try:
            duplicate_columns = df.T.duplicated().sum()
        except Exception:
            duplicate_columns = "Too large to compute"

        if not numeric_df.empty:
            desc_numeric = numeric_df.describe().transpose()
            desc_numeric["Skewness"] = numeric_df.apply(lambda x: skew(x.dropna()), axis=0)
            desc_numeric["Kurtosis"] = numeric_df.apply(lambda x: kurtosis(x.dropna()), axis=0)
            desc_numeric["Z-score Outliers"] = numeric_df.apply(
                lambda x: (np.abs((x - x.mean()) / x.std()) > 3).sum(), axis=0
            )
        else:
            desc_numeric = None

        if not categorical_df.empty:
            desc_categorical = categorical_df.describe().transpose()
            desc_categorical["Entropy"] = categorical_df.apply(
                lambda x: entropy(x.value_counts(normalize=True), base=2) if x.nunique() > 1 else 0
            )
        else:
            desc_categorical = None
    else:
        duplicate_rows = None
        duplicate_columns = None
        desc_numeric = numeric_df.describe().transpose() if not numeric_df.empty else None
        desc_categorical = categorical_df.describe().transpose() if not categorical_df.empty else None

    corr_matrix = numeric_df.corr() if (not numeric_df.empty and correlation_matrix) else None

    if verbose:
        def show_df(df_, title):
            if df_ is not None:
                html = df_.to_html(classes='scroll-table', escape=False)
                display(HTML(f"<h3>{title}</h3>" + html))

        display(HTML("""
        <style>
        .scroll-table {
            display: block;
            max-height: 400px;
            overflow-y: auto;
            overflow-x: auto;
            border: 1px solid #ccc;
            padding: 10px;
            font-size: 14px;
        }
        </style>
        """))

        show_df(summary, "Summary Statistics")
        show_df(desc_numeric, "Descriptive Statistics (Numerical)")
        show_df(desc_categorical, "Descriptive Statistics (Categorical)")
        show_df(corr_matrix, "Correlation Matrix")

        if detailing:
            print(f"\nTotal Duplicate Rows: {duplicate_rows}")
            print(f"Total Duplicate Columns: {duplicate_columns}")

    if return_dataframes:
        return summary, desc_numeric, desc_categorical, corr_matrix



def summary_column(df: pd.DataFrame, column_name: str, top_n: int = 10,
                   verbose: bool = True, return_dataframes: bool = False,
                   detailing: bool = True, time_column: str = None,
                   plots: list = None, fast_mode: bool = False):
    """
    Generates a detailed, human-readable summary of a single column in a DataFrame.

    This function helps you understand the nature of a specific column by providing
    summary metrics such as missingness, uniqueness, cardinality, entropy, skewness,
    and optional visualizations. It adapts intelligently to both numeric and categorical data
    and can highlight distribution issues, outliers, or missing trends over time.

    It is ideal for exploratory data analysis (EDA), column-wise diagnostics, and
    auditing feature quality before preprocessing or modeling.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column to summarize.

    column_name : str
        The name of the column to analyze.

    top_n : int, default=10
        Number of most frequent values to display in the frequency table for categorical features.

    verbose : bool, default=True
        If True, prints all summaries in formatted tables with headers. If `fast_mode=True`, this is ignored.

    return_dataframes : bool, default=False
        If True, returns:
        - summary_table : key metrics (missing %, unique %, etc.)
        - desc_stats : descriptive stats (mean, std, IQR, etc.)
        - freq_dist : frequency counts of top values

    detailing : bool, default=True
        If True, enables additional diagnostics:
        - For numeric: skewness, kurtosis, z-score outliers
        - For categorical: entropy
        - Also enables visualizations if `plots` is specified

    time_column : str, optional
        If specified, enables a missing-value trend chart over time using this datetime column.
        Useful for temporal datasets and time-series analysis.

    plots : list of str, optional
        List of plots to display:
        - "histogram": for numeric distribution
        - "bar": for top category counts
        - "missing_trend": for missing rate over time (requires `time_column`)

    fast_mode : bool, default=False
        If True, skips all optional visualizations, skew/entropy calculations, and pretty print tables.
        Recommended when analyzing very large datasets or when integrating into production pipelines.

    Returns
    -------
    tuple of pd.DataFrame, optional
        If `return_dataframes=True`, returns:
        - summary_table : base profile of the column
        - desc_stats : statistical or categorical description
        - freq_dist : top-N frequency breakdown

    Raises
    ------
    ValueError
        If the specified column does not exist in the DataFrame.

    Examples
    --------
    >>> summary_column(df, "salary", detailing=True, plots=["histogram"])

    >>> col_stats, desc, top_vals = summary_column(
            df,
            "product_category",
            detailing=True,
            plots=["bar"],
            return_dataframes=True
        )

    >>> summary_column(df, "discount", fast_mode=True)

    Notes
    -----
    - The function detects whether the column is numeric or categorical and adapts its metrics accordingly.
    - Outlier detection (z-score) is only applied to numeric features with sufficient variance.
    - Plots are automatically skipped when `fast_mode=True`.

    See Also
    --------
    summary_dataframe : Summarizes all columns of a DataFrame at once.
    preprocess_column : Cleans and transforms a single column based on rules.
    preprocess_dataframe : End-to-end preprocessing pipeline for the entire DataFrame.

    User Guide
    ----------
    🧠 When Should You Use This?
    - You want to audit or explore one column in depth.
    - You're deciding how to impute, encode, or drop a specific column.
    - You want to visualize category frequency or numeric distribution interactively.
    - You're building an automated column-report pipeline (with return_dataframes=True).

    ⚙️ Recommended Use Cases

    1. **Understand a numeric column with outliers:**
       >>> summary_column(df, "loan_amount", detailing=True, plots=["histogram"])

    2. **Explore a categorical feature for feature engineering:**
       >>> summary_column(df, "device_type", top_n=5, detailing=True, plots=["bar"])

    3. **Check for seasonal missingness (e.g., sensors or logs):**
       >>> summary_column(df, "temperature", time_column="timestamp", plots=["missing_trend"])

    4. **Automation or fast analysis at scale:**
       >>> summary_column(df, "user_age", fast_mode=True)

    💡 Tips:
    - Use with `preprocess_column()` to apply encoding or transformation after diagnosis.
    - If `entropy` is very low, the column may have little signal or be constant.
    - use 'fast_mode'=True` for large datasets to skip slow diagnostics.
    - Use `detailing=False` for a quick overview of the column without deep stats.
    - For many zero-variance columns, use `summary_dataframe()` for batch detection.
    - Always use `return_dataframes=True` if building custom reports or logging stats.
    """


    if fast_mode:
        detailing = False
        plots = []

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    plots = plots or []

    column_data = df[column_name]
    data_type = column_data.dtype
    total_count = len(column_data)
    missing_values = column_data.isnull().sum()
    unique_values = column_data.nunique()
    non_missing_values = total_count - missing_values

    desc_stats = column_data.describe(include="all").to_frame()
    additional_stats = {}
    

    if np.issubdtype(data_type, np.number):
        additional_stats["Variance"] = column_data.var()
        additional_stats["IQR"] = column_data.quantile(0.75) - column_data.quantile(0.25)
        additional_stats["Mode"] = column_data.mode().values[0] if not column_data.mode().empty else np.nan
        additional_stats["Min"] = column_data.min()
        additional_stats["Max"] = column_data.max()

        if detailing:
            if non_missing_values > 1:
                additional_stats["Skewness"] = skew(column_data.dropna())
                additional_stats["Kurtosis"] = kurtosis(column_data.dropna())
            else:
                additional_stats["Skewness"] = np.nan
                additional_stats["Kurtosis"] = np.nan

            mean = column_data.mean()
            std = column_data.std()
            additional_stats["Z-score Outlier Count"] = ((np.abs((column_data - mean) / std) > 3).sum()) if std > 0 else 0

    elif data_type == "object" or data_type.name == "category":
        additional_stats["Mode"] = column_data.mode().values[0] if not column_data.mode().empty else "N/A"
        if detailing and unique_values < 10000:
            value_probs = column_data.value_counts(normalize=True)
            additional_stats["Entropy"] = entropy(value_probs, base=2) if unique_values > 1 else 0

    # ✅ Value Counts (Always computed)
    freq_dist = column_data.value_counts(dropna=False).reset_index().head(top_n)
    freq_dist.columns = ["Value", "Count"]
    freq_dist["Percentage"] = (freq_dist["Count"] / total_count * 100).round(2).astype(str) + " %"

    # Summary Table
    summary_table = pd.DataFrame([
        ["Data Type", data_type],
        ["Total Values", total_count],
        ["Non-Missing Values", non_missing_values],
        ["Missing Values", missing_values],
        ["Missing %", round((missing_values / total_count * 100), 2) if total_count > 0 else 0],
        ["Unique Values", unique_values],
    ] + list(additional_stats.items()), columns=["Metric", "Value"])

    if verbose:
        print("\n" + "=" * 100)
        print(f"Analysis for Column: {column_name}")
        print("=" * 100)

        print("\nSummary Statistics:")
        print(tabulate(summary_table, headers="keys", tablefmt="fancy_grid", showindex=False))

        print("\nDescriptive Statistics:")
        print(tabulate(desc_stats, headers="keys", tablefmt="fancy_grid"))

        if not freq_dist.empty:
            print(f"\nTop {top_n} Value Counts:")
            print(tabulate(freq_dist, headers="keys", tablefmt="fancy_grid"))

    # 🔍 Plots (only if detailing=True)
    if detailing:
        if np.issubdtype(data_type, np.number) and "histogram" in plots:
            plt.figure(figsize=(10, 4))
            column_data.hist(bins=30, edgecolor='black')
            plt.title(f"Histogram of {column_name}")
            plt.xlabel(column_name)
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        elif (data_type == "object" or data_type.name == "category") and "bar" in plots:
            if not freq_dist.empty:
                plt.figure(figsize=(10, 4))
                freq_dist.plot(kind="bar", x="Value", y="Count", legend=False)
                plt.title(f"Top {top_n} Categories in {column_name}")
                plt.xticks(rotation=45, ha='right')
                plt.ylabel("Count")
                plt.tight_layout()
                plt.show()

        if time_column and time_column in df.columns and "missing_trend" in plots:
            if pd.api.types.is_datetime64_any_dtype(df[time_column]):
                missing_series = df.set_index(time_column)[column_name].isnull().resample("W").mean()
                missing_series.plot(figsize=(10, 3), title=f"Missing Rate Over Time for {column_name}")
                plt.ylabel("Missing %")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    if return_dataframes:
        return summary_table, desc_stats, freq_dist
