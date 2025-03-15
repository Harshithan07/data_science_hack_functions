import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from tabulate import tabulate

def summary_dataframe(df: pd.DataFrame, verbose: bool = True, return_dataframes: bool = False):
    """
    Generate a comprehensive summary for an entire DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    verbose (bool): If True, prints the summary.
    return_dataframes (bool): If True, returns the summary DataFrames.

    Returns:
    tuple (optional):
        - summary (pd.DataFrame): Summary statistics of all columns.
        - desc_numeric (pd.DataFrame): Detailed numerical statistics.
        - desc_categorical (pd.DataFrame): Detailed categorical statistics.
        - correlation_matrix (pd.DataFrame): Correlation matrix (if applicable).
    """

    if df.empty:
        raise ValueError("The provided DataFrame is empty. Provide a valid dataset.")

    total_rows = df.shape[0]
    numeric_df = df.select_dtypes(include=["number"])
    categorical_df = df.select_dtypes(include=["object"])

    # Basic Summary Table
    summary = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.values,
        "Total Values": df.count().values,
        "Missing Values": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / total_rows * 100).round(2),
        "Unique Values": df.nunique().values,
        "Unique %": (df.nunique().values / total_rows * 100).round(2)
    })

    # Identify Constant Columns
    summary["Constant Column"] = summary["Unique Values"] == 1

    # Feature Cardinality Categorization
    summary["Cardinality Category"] = summary["Unique Values"].apply(
        lambda x: "Low" if x <= 10 else "Medium" if x <= 100 else "High"
    )

    # Detect Duplicates
    duplicate_rows = df.duplicated().sum()
    duplicate_columns = df.T.duplicated().sum()

    # Numerical Descriptive Statistics
    if not numeric_df.empty:
        desc_numeric = numeric_df.describe().transpose()
        desc_numeric["Skewness"] = numeric_df.apply(lambda x: skew(x.dropna()), axis=0)
        desc_numeric["Kurtosis"] = numeric_df.apply(lambda x: kurtosis(x.dropna()), axis=0)
        desc_numeric["Z-score Outliers"] = numeric_df.apply(lambda x: (np.abs((x - x.mean()) / x.std()) > 3).sum(), axis=0)
    else:
        desc_numeric = None

    # Categorical Descriptive Statistics
    if not categorical_df.empty:
        desc_categorical = categorical_df.describe().transpose()
        desc_categorical["Entropy"] = categorical_df.apply(
            lambda x: entropy(x.value_counts(normalize=True), base=2) if x.nunique() > 1 else 0
        )
    else:
        desc_categorical = None

    # Correlation Matrix
    correlation_matrix = numeric_df.corr() if not numeric_df.empty else None

    # Print Output
    if verbose:
        print("\nSummary Statistics:")
        print(tabulate(summary, headers="keys", tablefmt="fancy_grid", showindex=False))

        if desc_numeric is not None:
            print("\nDescriptive Statistics (Numerical Data):")
            print(tabulate(desc_numeric, headers="keys", tablefmt="fancy_grid"))

        if desc_categorical is not None:
            print("\nDescriptive Statistics (Categorical Data):")
            print(tabulate(desc_categorical, headers="keys", tablefmt="fancy_grid"))

        if correlation_matrix is not None:
            print("\nCorrelation Matrix:")
            print(tabulate(correlation_matrix, headers="keys", tablefmt="fancy_grid"))

        print(f"\nTotal Duplicate Rows: {duplicate_rows}")
        print(f"Total Duplicate Columns: {duplicate_columns}")

    if return_dataframes:
        return summary, desc_numeric, desc_categorical, correlation_matrix


def summary_column(df: pd.DataFrame, column_name: str, top_n: int = 10, verbose: bool = True, return_dataframes: bool = False):
    """
    Generate an enhanced summary for a single column, including descriptive statistics,
    frequency distribution, skewness, kurtosis, and entropy.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): Name of the column to analyze.
    top_n (int): Number of unique values to display in value counts.
    verbose (bool): If True, prints the summary.
    return_dataframes (bool): If True, returns the summary DataFrames.

    Returns:
    tuple (optional):
        - summary_table (pd.DataFrame): Summary statistics of the column.
        - desc_stats (pd.DataFrame): Descriptive statistics.
        - freq_dist (pd.DataFrame): Frequency distribution (if applicable).
    """
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    column_data = df[column_name]
    data_type = column_data.dtype
    total_count = len(column_data)
    missing_values = column_data.isnull().sum()
    unique_values = column_data.nunique()
    non_missing_values = total_count - missing_values

    # Basic Descriptive Statistics
    desc_stats = column_data.describe(include="all").to_frame()

    # Additional Statistics for Numerical Data
    additional_stats = {}
    if np.issubdtype(column_data.dtype, np.number):
        additional_stats["Variance"] = column_data.var()
        additional_stats["Skewness"] = skew(column_data.dropna()) if non_missing_values > 1 else np.nan
        additional_stats["Kurtosis"] = kurtosis(column_data.dropna()) if non_missing_values > 1 else np.nan
        additional_stats["IQR"] = column_data.quantile(0.75) - column_data.quantile(0.25)
        additional_stats["Mode"] = column_data.mode().values[0] if not column_data.mode().empty else np.nan
        additional_stats["Min"] = column_data.min()
        additional_stats["Max"] = column_data.max()
        additional_stats["Z-score Outlier Count"] = ((np.abs((column_data - column_data.mean()) / column_data.std()) > 3).sum())

    # Additional Statistics for Categorical Data
    elif data_type == "object":
        additional_stats["Mode"] = column_data.mode().values[0] if not column_data.mode().empty else "N/A"
        additional_stats["Entropy"] = entropy(column_data.value_counts(normalize=True), base=2) if unique_values > 1 else 0

    # Frequency Distribution for Categorical / Low-Cardinality Numeric Data
    freq_dist = None
    if data_type == "object" or unique_values < 30:
        freq_dist = column_data.value_counts(dropna=False).head(top_n).reset_index()
        freq_dist.columns = ["Value", "Count"]
        freq_dist["Percentage"] = (freq_dist["Count"] / total_count * 100).round(2).astype(str) + " %"

    # Creating Summary Table
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

        if freq_dist is not None:
            print(f"\nTop {top_n} Value Counts:")
            print(tabulate(freq_dist, headers="keys", tablefmt="fancy_grid"))

        print("\n" + "=" * 100)

    if return_dataframes:
        return summary_table, desc_stats, freq_dist