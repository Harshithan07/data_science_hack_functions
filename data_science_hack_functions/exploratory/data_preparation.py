from typing import Optional, Tuple, Union, Callable, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
from tabulate import tabulate

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

def prepare_sample_split(
    df: pd.DataFrame,
    target: Optional[str] = None,
    sample_size: Optional[int] = None,
    split: bool = False,
    test_size: float = 0.2,
    stratify: bool = False,
    random_seed: int = 42,
    return_indices: bool = False,
    return_metadata: bool = False,
    verbose: bool = False,
    log_callback: Optional[Callable[[str], None]] = None
) -> Union[pd.DataFrame, Tuple, Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
    """
    Sample and/or split your dataset for training workflows with optional stratification, reproducibility, 
    metadata tracking, and notebook-friendly summaries.

    This function is ideal for:
    - Efficiently working with large datasets by subsampling
    - Creating reproducible train/test splits with or without class balance
    - Getting audit-ready metadata (row counts, strategy used, seed)
    - Easily integrating into notebook-based data science pipelines

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset from which sampling and/or splitting is performed. Each row is an observation.

    target : str, optional
        Column name to use for stratification (classification problems).
        Required only if `stratify=True`. Can be ignored for regression tasks or unsupervised workflows.

        Example:
        >>> target='label'

    sample_size : int, optional
        Number of rows to sample from the dataset before splitting.
        If None, the entire dataset is used.

        Example:
        >>> sample_size=10000

    split : bool, default=False
        Whether to split the sampled/full dataset into train and test sets.

        If True, you get `(train_df, test_df)` or `(train_df, test_df, metadata)`.
        If False, only a single DataFrame is returned.

    test_size : float, default=0.2
        Proportion of test set if `split=True`.

        Example:
        >>> test_size=0.3  # 70/30 split

    stratify : bool, default=False
        If True, ensures that train/test (or sample) retains class balance by stratifying using the `target`.

        Recommended for classification tasks.

    random_seed : int, default=42
        Sets the random state for both sampling and splitting to ensure full reproducibility.

        Example:
        >>> random_seed=123

    return_indices : bool, default=False
        If True, returns the list of original indices of the sampled DataFrame.

        Example:
        >>> df_sampled, idx = prepare_sample_split(df, sample_size=5000, return_indices=True)

    return_metadata : bool, default=False
        If True, returns a detailed dictionary summarizing:
        - number of rows before/after sampling
        - stratification status
        - train/test split sizes (if split=True)
        - random seed used

        Metadata is also displayed in scrollable HTML format when using notebooks.

    verbose : bool, default=False
        If True, prints internal logging messages about the sampling/splitting process.

    log_callback : callable, optional
        Optional custom logging function (e.g., `logger.info`) to capture messages in logs or dashboards.

        Example:
        >>> prepare_sample_split(df, log_callback=my_logger)

    Returns
    -------
    pd.DataFrame or tuple
        - If `split=False`: returns a sampled `pd.DataFrame`, or with indices/metadata if requested.
        - If `split=True`: returns `(train_df, test_df)` or `(train_df, test_df, metadata)`.

    Examples
    --------
    ▶️ Basic sampling:
    >>> df_sampled = prepare_sample_split(df, sample_size=5000)

    ▶️ Sampling + splitting:
    >>> train_df, test_df = prepare_sample_split(df, sample_size=10000, split=True)

    ▶️ Stratified split with metadata:
    >>> train_df, test_df, meta = prepare_sample_split(
            df, target='label', stratify=True, split=True, return_metadata=True
        )

    ▶️ Sample with reproducibility:
    >>> df_sampled = prepare_sample_split(df, sample_size=3000, random_seed=123)

    ▶️ With logging hook:
    >>> prepare_sample_split(df, log_callback=lambda msg: print(f"[LOG]: {msg}"))

    Notes
    -----
    - Internally uses `sklearn.model_selection.train_test_split()` for splitting and stratification.
    - Always resets index to prevent downstream issues with row alignment.
    - Stratification only works with classification-style discrete targets.

    Related
    -------
    • feature_exploration() — run after sampling to analyze feature quality  
    • evaluate_classification_model() — use after train/test split for performance metrics  
    • feature_engineering() — to transform features post sampling/split  
    • preprocess_dataframe() — clean the data before modeling
    """

    def log(msg):
        if verbose:
            print(msg)
        if log_callback:
            log_callback(msg)

    if stratify and not target:
        raise ValueError("Stratification requires the 'target' column.")
    if target and target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the DataFrame.")

    metadata = {
        "original_rows": len(df),
        "random_seed": random_seed,
        "stratified": stratify,
        "split": split,
        "test_size": test_size,
        "sample_size_requested": sample_size,
        "final_rows": len(df)
    }

    df_sampled = df.copy()
    if sample_size and sample_size < len(df):
        log(f"Sampling {sample_size} rows from {len(df)}...")
        if stratify and target:
            _, df_sampled = train_test_split(
                df, train_size=sample_size, stratify=df[target], random_state=random_seed
            )
        else:
            df_sampled = df.sample(n=sample_size, random_state=random_seed)
        df_sampled = df_sampled.reset_index(drop=False)
        metadata["final_rows"] = len(df_sampled)

    if split:
        stratifier = df_sampled[target] if stratify and target else None
        train_df, test_df = train_test_split(
            df_sampled, test_size=test_size, stratify=stratifier, random_state=random_seed
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        metadata["train_rows"] = len(train_df)
        metadata["test_rows"] = len(test_df)

        log(f"Split complete: {len(train_df)} train rows, {len(test_df)} test rows.")

        if return_metadata:
            display_scrollable_table(pd.DataFrame(list(metadata.items()), columns=["Step", "Value"]), title="📦 Sample & Split Metadata")
            return (train_df, test_df, metadata)
        return (train_df, test_df)

    df_sampled = df_sampled.reset_index(drop=True)

    if return_indices:
        return df_sampled, df_sampled.index.to_list()

    if return_metadata:
        display_scrollable_table(pd.DataFrame(list(metadata.items()), columns=["Step", "Value"]), title="📦 Sample Metadata")
        return df_sampled, metadata

    return df_sampled
