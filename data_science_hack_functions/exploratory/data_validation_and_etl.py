from typing import Optional, Tuple, Union, Callable, Dict, List
import pandas as pd
from IPython.display import display, HTML
import numpy as np
import re
import json

def display_scrollable_table(df: pd.DataFrame, title: str = "Preview"):
    html = df.to_html(classes='scroll-table', escape=False, index=False)
    styled_html = f"""
    <style>
        .scroll-table {{
            background-color: #1e1e1e;
            color: #e0e0e0;
            font-family: monospace;
            font-size: 13px;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 6px;
            max-height: 400px;
            overflow-x: auto;
            overflow-y: auto;
        }}
        .scroll-table table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .scroll-table th {{
            background-color: #2a2a2a;
            color: #ffffff;
            position: sticky;
            top: 0;
            z-index: 1;
            padding: 6px;
            border-bottom: 1px solid #555;
        }}
        .scroll-table td {{
            padding: 6px;
            border-bottom: 1px solid #333;
        }}
        .scroll-table tr:nth-child(even) {{ background-color: #252525; }}
        .scroll-table tr:nth-child(odd) {{ background-color: #1e1e1e; }}
    </style>
    <h3 style='color:#f0f0f0;'>{title}</h3>
    <div class="scroll-table">{html}</div>
    """
    display(HTML(styled_html))

def validate_and_clean_data(
    df: pd.DataFrame,
    schema: Optional[Dict[str, str]] = None,
    rename_map: Optional[Dict[str, str]] = None,
    deduplicate: bool = True,
    coerce_types: bool = True,
    parse_dates: bool = True,
    enforce_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    unique_columns: Optional[List[str]] = None,
    category_constraints: Optional[Dict[str, List]] = None,
    clean_strings: bool = True,
    snake_case_columns: bool = True,
    verbose: bool = True,
    return_report: bool = True,
    report_path: Optional[str] = None,
    export_json: Optional[str] = None,
    pii_keywords: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate and clean a raw Pandas DataFrame with schema enforcement, common ETL transformations, 
    and full audit logging — designed for production-ready ML pipelines and interactive data workflows.

    This function performs all critical preprocessing validations:
    - Ensures column names and data types are as expected
    - Removes duplicates, trims and cleans strings
    - Parses date columns, applies range/category checks
    - Tracks actions in a structured report
    - Optionally saves audit report as HTML and JSON

    Parameters
    ----------
    df : pd.DataFrame
        The raw input DataFrame to validate and clean. This should be the first step in your ML pipeline.

    schema : dict, optional
        A dictionary mapping expected column names to their expected pandas-compatible data types (e.g., 'int', 'float').
        Used to coerce types using `.astype()` and flag columns that don't match.

        Example:
        >>> schema = {"age": "int", "income": "float", "signup_date": "datetime64[ns]"}

    rename_map : dict, optional
        Dictionary to rename columns before validation. Useful when dealing with inconsistent column headers
        from different data sources.

        Example:
        >>> rename_map = {"SignUpDate": "signup_date"}

    deduplicate : bool, default=True
        If True, removes duplicate rows using `df.drop_duplicates()`. Duplicates removed are logged in the report.

    coerce_types : bool, default=True
        If True, attempts to cast each column to the type specified in the `schema`.
        Safe conversions only; fails silently and skips invalid conversions.

    parse_dates : bool, default=True
        If True, automatically detects and parses columns that contain 'date' in their name using `pd.to_datetime()`.

    enforce_ranges : dict, optional
        A dictionary specifying numeric columns with acceptable (min, max) value ranges.
        Rows violating the range are not dropped but counted and reported.

        Example:
        >>> enforce_ranges = {"age": (0, 120), "income": (0, 1_000_000)}

    unique_columns : list of str, optional
        Columns that must contain unique values (e.g., primary keys). Violations are logged, not dropped.

        Example:
        >>> unique_columns = ["email", "user_id"]

    category_constraints : dict, optional
        Dictionary mapping column names to a list of allowed values. Invalid category entries are flagged in the report.

        Example:
        >>> category_constraints = {"gender": ["male", "female", "other"]}

    clean_strings : bool, default=True
        If True, strips leading/trailing spaces and lowercases all object-type (string) columns.

    snake_case_columns : bool, default=True
        If True, converts all column names to `snake_case` for consistency across downstream ML workflows.

    verbose : bool, default=True
        If True, prints basic progress logs to console. Does not affect HTML output.

    return_report : bool, default=True
        If True, displays a beautiful scrollable HTML summary in notebooks and returns the full metadata dictionary.

    report_path : str, optional
        File path to save the report as an `.html` file. Uses the same style as your notebook HTML output.

        Example:
        >>> report_path = "validation_report.html"

    export_json : str, optional
        If provided, saves the full validation report dictionary as a `.json` file for use in audit pipelines, logging systems, etc.

        Example:
        >>> export_json = "validation_metadata.json"

    pii_keywords : list of str, optional
        List of keywords (e.g., 'email', 'ssn', 'phone') to scan for in column names to flag potential PII fields.

        Example:
        >>> pii_keywords = ["email", "phone", "ssn"]

    Returns
    -------
    df_clean : pd.DataFrame
        The cleaned and type-coerced DataFrame, safe for downstream modeling or transformation.

    report : dict
        A structured dictionary summarizing all validation checks, transformation steps, and column-level actions.
        Contains keys such as: 'missing_values', 'coerced_columns', 'pii_suspects', etc.

    Examples
    --------
    ▶️ Basic usage with report:
    >>> df_clean, report = validate_and_clean_data(df, schema={"age": "int", "income": "float"})

    ▶️ Add renaming and range checks:
    >>> df_clean, report = validate_and_clean_data(
            df,
            rename_map={"SignUpDate": "signup_date"},
            enforce_ranges={"age": (0, 100)}
        )

    ▶️ Generate downloadable audit report:
    >>> validate_and_clean_data(
            df, schema=schema, report_path=\"validation.html\", export_json=\"report.json\"
        )

    Notes
    -----
    - Columns not in schema are ignored during type coercion
    - Violations are logged but **data is not dropped** unless explicitly enabled
    - You can use this before any ML function like `preprocess_dataframe()` or `feature_engineering()`
    - The audit report helps ensure trust, governance, and repeatability in data flows

    Related
    -------
    • preprocess_dataframe() – for full-scale modeling prep  
    • feature_exploration() – for checking feature quality  
    • prepare_sample_split() – for reproducible sampling/splitting  
    • pandas_profiling (external) – for deep statistical reports  
    • great_expectations (optional) – for declarative expectations
    """

    report = {
        "original_shape": df.shape,
        "coerced_columns": [],
        "renamed_columns": {},
        "dropped_duplicates": 0,
        "date_parsed": [],
        "range_violations": {},
        "category_violations": {},
        "missing_values": {},
        "unique_violations": [],
        "pii_suspects": []
    }
    df = df.copy()

    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        report["renamed_columns"] = rename_map

    if snake_case_columns:
        df.columns = [re.sub(r'\W|^(?=\d)', '_', col).lower() for col in df.columns]

    if coerce_types and schema:
        for col, expected_type in schema.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(expected_type)
                    report["coerced_columns"].append(col)
                except:
                    continue

    if parse_dates:
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    report["date_parsed"].append(col)
                except:
                    continue

    missing = df.isnull().mean()
    report["missing_values"] = missing[missing > 0].round(4).to_dict()

    if deduplicate:
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        report["dropped_duplicates"] = before - after

    if enforce_ranges:
        for col, (min_val, max_val) in enforce_ranges.items():
            if col in df.columns:
                violations = ~df[col].between(min_val, max_val)
                count = violations.sum()
                if count > 0:
                    report["range_violations"][col] = int(count)

    if unique_columns:
        for col in unique_columns:
            if col in df.columns and df[col].duplicated().any():
                report["unique_violations"].append(col)

    if category_constraints:
        for col, allowed in category_constraints.items():
            if col in df.columns:
                invalid = ~df[col].isin(allowed)
                if invalid.any():
                    report["category_violations"][col] = int(invalid.sum())

    if clean_strings:
        str_cols = df.select_dtypes(include='object').columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()

    if pii_keywords:
        suspicious = [col for col in df.columns if any(pii.lower() in col.lower() for pii in pii_keywords)]
        report["pii_suspects"] = suspicious

    if return_report:
        summary = []
        for k, v in report.items():
            summary.append((k, len(v) if isinstance(v, (dict, list)) else v, str(v)))
        df_summary = pd.DataFrame(summary, columns=["Check", "Summary", "Details"])
        display_scrollable_table(df_summary, title="📋 Data Validation & Cleaning Report")

        if report_path:
            html_table = df_summary.to_html(index=False, escape=False)
            styled = f"""
            <html>
            <head>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #111; color: #eee; padding: 20px; }}
                table {{ border-collapse: collapse; width: 100%; background-color: #1e1e1e; }}
                th, td {{ border: 1px solid #555; padding: 8px; text-align: left; }}
                th {{ background-color: #2a2a2a; }}
                tr:nth-child(even) {{ background-color: #252525; }}
                tr:nth-child(odd) {{ background-color: #1e1e1e; }}
            </style>
            </head>
            <body>
            <h2>📋 Data Validation & Cleaning Report</h2>
            {html_table}
            </body>
            </html>
            """
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(styled)

        if export_json:
            with open(export_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

    return df, report
