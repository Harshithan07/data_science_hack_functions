import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from tabulate import tabulate
from IPython.display import display, HTML


class _PreprocessingSteps(dict):
    """Custom object to display preprocessing steps nicely in notebooks or as plain text."""

    def _repr_html_(self):
        rows = []
        for key, value in self.items():
            summary = f"{len(value)} items" if isinstance(value, list) and value else "[✔]"
            details = ', '.join(map(str, value)) if isinstance(value, list) else str(value)
            rows.append((key, summary, details))

        table_html = tabulate(rows, headers=["Step", "Summary", "Details"], tablefmt="unsafehtml")
        return f"<h3>🧠 Preprocessing Steps Summary</h3>{table_html}"

    def __str__(self):
        rows = []
        for key, value in self.items():
            summary = f"{len(value)} items" if isinstance(value, list) and value else "[✔]"
            details = ', '.join(map(str, value)) if isinstance(value, list) else str(value)
            rows.append((key, summary, details))
        return tabulate(rows, headers=["Step", "Summary", "Details"], tablefmt="fancy_grid")


def _display_scrollable_preview(df: pd.DataFrame, title: str = "Preview"):
    """Internal helper to show a styled, scrollable DataFrame compatible with dark themes like Colab's."""
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
        .scroll-table tr:nth-child(even) {{
            background-color: #252525;
        }}
        .scroll-table tr:nth-child(odd) {{
            background-color: #1e1e1e;
        }}
    </style>
    <h3 style='color:#f0f0f0;'>{title}</h3>
    <div class="scroll-table">{html}</div>
    """
    display(HTML(styled_html))



def preprocess_dataframe(
    df: pd.DataFrame,
    impute: bool = True,
    numeric_method: str = 'mean',
    categorical_method: str = 'mode',
    drop_missing_thresh: float = 0.3,
    encode: str = 'onehot',
    scale: str = 'standard',
    drop_constant: bool = True,
    max_cardinality: Optional[int] = 100,
    return_steps: bool = False,
    preview: bool = False,
    verbose: bool = True,
    fast_mode: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Preprocess a tabular dataset for machine learning using a streamlined, transparent pipeline.

    This function performs automatic and configurable preprocessing on a pandas DataFrame, 
    applying common transformations such as missing value imputation, categorical encoding, 
    numeric feature scaling, and filtering of low-information or problematic features.

    It is designed to make raw datasets "model-ready" with minimal manual effort,
    while still allowing for full control over each preprocessing step.

    Parameters
    ----------
    df : pd.DataFrame
        The raw input dataset to preprocess. Must be a standard, rectangular DataFrame.

    impute : bool, default=True
        If True, fills missing values using the specified `numeric_method` and `categorical_method`.

    numeric_method : {'mean', 'median', 'constant'}, default='mean'
        Strategy to fill missing values in numeric columns:
        - 'mean': use column average
        - 'median': use column median
        - 'constant': fill with 0

    categorical_method : {'mode', 'constant', 'drop'}, default='mode'
        Strategy to fill missing values in categorical columns:
        - 'mode': fill with most frequent category
        - 'constant': fill with "missing"
        - 'drop': skip imputation for categoricals

    drop_missing_thresh : float, default=0.3
        Drop columns with more than this proportion of missing values (e.g., 0.3 = 30%).

    encode : {'onehot', 'ordinal', None}, default='onehot'
        How to encode categorical columns:
        - 'onehot': expand categories into binary columns (ideal for tree-based and linear models)
        - 'ordinal': convert categories into integers (compact but model-sensitive)
        - None: skip encoding

    scale : {'standard', 'minmax', 'robust', None}, default='standard'
        Scaling method for numeric features:
        - 'standard': zero mean, unit variance (default for most ML models)
        - 'minmax': scales features to [0, 1]
        - 'robust': scales using median and IQR (more resistant to outliers)
        - None: skip scaling

    drop_constant : bool, default=True
        If True, drops columns where all values are the same — these provide no predictive value.

    max_cardinality : int or None, default=100
        If set, drops categorical columns with more than this many unique values.
        Useful to eliminate IDs or high-entropy features that aren't generalizable.

    return_steps : bool, default=False
        If True, returns a second object (`steps`) summarizing the preprocessing pipeline steps.
        This is useful for auditing, documentation, or reproducing pipelines.

    preview : bool, default=False
        If True, displays a styled scrollable HTML preview of the top rows of the transformed DataFrame.
        Intended for use inside notebooks (e.g., Jupyter, Colab).

    verbose : bool, default=True
        If True, prints human-readable logs describing each transformation as it is applied.

    fast_mode : bool, default=False
        If True, disables logging and previews to optimize performance for large datasets (1M+ rows).
        Recommended for production or batch processing pipelines.

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, dict]
        - The transformed DataFrame, ready for use in ML models.
        - If `return_steps=True`, also returns a dictionary-like object listing:
            - dropped columns (due to high missingness, constant value, or high cardinality)
            - columns encoded, scaled, or imputed
            - final column names

    Examples
    --------
    >>> df_clean = preprocess_dataframe(df)

    >>> df_clean, steps = preprocess_dataframe(
            df,
            encode="ordinal",
            scale="robust",
            drop_missing_thresh=0.25,
            return_steps=True,
            preview=True
        )
    >>> print(steps)

    Notes
    -----
    - This function does not modify the input DataFrame (works on a copy).
    - Designed to be flexible enough for prototyping, reproducible for experiments,
      and fast enough for production workloads.
    - Use `fast_mode=True` when processing high-volume datasets or in production loops.
    - For column-wise control, see `preprocess_column()`.

    See Also
    --------
    preprocess_column : Clean and transform a single Series with similar options.

    User Guide
    ----------
    🧭 When Should You Use This Function?
    - You're starting with raw, messy tabular data that has missing values, mixed data types, or irrelevant columns.
    - You want to transform the dataset into a clean, numerical form ready for modeling — without hardcoding pipelines manually.
    - You're preparing data for machine learning algorithms (scikit-learn, XGBoost, LightGBM, etc.) and need a reproducible cleaning strategy.
    - You're in an experimentation phase and want fast iteration with logs, or you're moving toward deployment and need performance mode.

    ⚙️ Recommended Configurations (Use-Case Based)

    1. **General-purpose ML modeling (balanced tabular data):**
       → Works well with logistic regression, SVMs, and shallow neural networks.
       >>> preprocess_dataframe(df, encode="onehot", scale="standard")

       *Why?* One-hot encoding ensures categorical variables are treated independently. Standard scaling helps models converge.

    2. **Tree-based models (RandomForest, XGBoost, LightGBM):**
       → These models handle ordinal input well and don’t require scaling.
       >>> preprocess_dataframe(df, encode="ordinal", scale=None)

       *Why?* One-hot can add noise or bloat tree-based models. Ordinal + no scaling is faster and sufficient.

    3. **High-cardinality datasets or sparse data (e.g., recommender systems):**
       >>> preprocess_dataframe(df, encode=None, max_cardinality=50)

       *Why?* Skip encoding and limit high-cardinality columns to avoid exploding the feature space.

    4. **Production or big data batch preprocessing:**
       → Great for pipelines where performance > visuals.
       >>> preprocess_dataframe(df, fast_mode=True)

       *Why?* Disables all logging and display overhead — ideal for 1M+ rows.

    5. **Auditing or debugging preprocessing behavior:**
       → You want to know exactly what was changed and why.
       >>> df_clean, steps = preprocess_dataframe(df, return_steps=True, verbose=True)
       >>> print(steps)

       *Why?* `steps` logs dropped columns, encoded features, and final output — great for versioning and reproducibility.


    💡 Tips:
    - Use `preview=True` only in notebooks to visualize output cleanly.
    - Set `max_cardinality=None` to retain all categorical columns, even high-card ones.
    - `drop_constant=True` removes junk features automatically.
    - Use `fast_mode=True` when processing high-volume datasets or in production loops.
    - Use `preprocess_column()` for detailed tuning on a single feature.

    Related:
    --------
    • preprocess_column(): Clean and transform a single Series with similar options.
    • summary_column(): View deep stats for a single column before preprocessing
    • summary_dataframe(): View deep stats for a DataFrame before preprocessing
    
    """

    df = df.copy()
    steps = {}

    # Fast mode: heavy preprocessing skipped
    if fast_mode:
        preview = False
        verbose = False

    # Drop high-missing columns
    missing_ratio = df.isnull().mean()
    high_missing = missing_ratio[missing_ratio > drop_missing_thresh].index.tolist()
    if high_missing and verbose:
        print(f"[Drop] {len(high_missing)} columns dropped for >{int(drop_missing_thresh * 100)}% missing.")
    df.drop(columns=high_missing, inplace=True)
    steps['dropped_high_missing'] = high_missing

    # Drop constant columns
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if drop_constant and constant_cols:
        if verbose:
            print(f"[Drop] {len(constant_cols)} constant columns dropped.")
        df.drop(columns=constant_cols, inplace=True)
    steps['dropped_constant'] = constant_cols

    # Drop high-cardinality categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    high_card_cols = [col for col in cat_cols if df[col].nunique() > max_cardinality] if max_cardinality else []
    if high_card_cols:
        if verbose:
            print(f"[Drop] {len(high_card_cols)} high-cardinality columns dropped.")
        df.drop(columns=high_card_cols, inplace=True)
    steps['dropped_high_cardinality'] = high_card_cols

    # Refresh column types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    transformers = []

    # Imputation
    if impute:
        if num_cols:
            transformers.append(('imputer_num', SimpleImputer(strategy=numeric_method), num_cols))
            if verbose:
                print(f"[Impute] Numeric → {numeric_method} on {len(num_cols)} cols")
        if cat_cols and categorical_method != 'drop':
            strategy = 'most_frequent' if categorical_method == 'mode' else 'constant'
            transformers.append(('imputer_cat', SimpleImputer(strategy=strategy), cat_cols))
            if verbose:
                print(f"[Impute] Categorical → {categorical_method} on {len(cat_cols)} cols")
    elif verbose:
        print("[Impute] Skipped")

    # Encoding
    if encode and cat_cols:
        if encode == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        elif encode == 'ordinal':
            encoder = OrdinalEncoder()
        else:
            raise ValueError("Invalid encoding method")
        transformers.append(('encoder', encoder, cat_cols))
        if verbose:
            print(f"[Encode] {encode} encoding applied")
    elif verbose:
        print("[Encode] Skipped")

    # Scaling
    if scale and num_cols:
        if scale == 'standard':
            scaler = StandardScaler()
        elif scale == 'minmax':
            scaler = MinMaxScaler()
        elif scale == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scale method")
        transformers.append(('scaler', scaler, num_cols))
        if verbose:
            print(f"[Scale] {scale} scaling applied")
    elif verbose:
        print("[Scale] Skipped")

    # Apply ColumnTransformer
    if transformers:
        pipeline = ColumnTransformer(transformers, remainder='drop')
        df_transformed = pipeline.fit_transform(df)

        # Generate readable column names
        feature_names = []
        for name, transformer, cols in pipeline.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names = transformer.get_feature_names_out(cols)
                except:
                    names = [f"{name}_{col}" for col in cols]
            else:
                names = [f"{col}" for col in cols]
            feature_names.extend(names)

        df = pd.DataFrame(df_transformed, columns=feature_names)
        steps['final_columns'] = feature_names

        if verbose:
            print(f"[Transform] Final shape: {df.shape}")
    elif verbose:
        print("[Transform] No transformations applied")

    if preview:
        _display_scrollable_preview(df.head(10), title="🧼 Preprocessed DataFrame (Top 10 Rows)")

    return (df, _PreprocessingSteps(steps)) if return_steps else df



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable, Optional, Tuple
from scipy.stats import zscore, iqr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
from IPython.display import display, HTML
import warnings

class _ColumnSteps(dict):
    def _repr_html_(self):
        rows = [f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in self.items()]
        return f"<table style='font-family:monospace;color:#eee;background:#222;padding:8px;'>{''.join(rows)}</table>"
    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.items()])


def preprocess_column(
    col: pd.Series,
    impute: Optional[Union[str, float, int]] = None,
    scale: Optional[str] = None,
    encode: Optional[str] = None,
    cap_outliers: Optional[str] = None,
    cap_quantiles: Tuple[float, float] = (0.01, 0.99),
    custom_map: Optional[Callable] = None,
    strategy: str = "manual",
    inplace: bool = False,
    preview: bool = False,
    plot: bool = False,
    verbose: bool = True,
    fast_mode = False
) -> Union[pd.Series, Tuple[pd.Series, dict]]:
    """
    Preprocess a single column of data with smart transformations for modeling.

    This function allows targeted preprocessing of a single pandas Series — such as a feature column
    from a DataFrame — including missing value imputation, outlier treatment, encoding, scaling, 
    custom transformations, and optional inspection via plots or previews.

    It is useful for exploratory analysis, fine-tuned feature engineering, or inspecting
    column-specific cleaning steps outside of full-pipeline automation.

    Parameters
    ----------
    col : pd.Series
        The input column to clean. Can be numeric or categorical.

    impute : {'mean', 'median', 'mode', 'constant'} or scalar, optional
        Strategy to fill missing values:
        - 'mean', 'median': numeric only
        - 'mode': most frequent value (works for both types)
        - 'constant': fill with 0 or "missing"
        - scalar: fill with a specific value (e.g., 0 or 'unknown')

    scale : {'standard', 'minmax', 'robust'}, optional
        If provided, applies scaling (numeric only):
        - 'standard': zero mean, unit variance
        - 'minmax': rescale to [0, 1]
        - 'robust': scale based on median and IQR (for outliers)

    encode : {'ordinal', 'onehot'}, optional
        If provided, applies encoding (categorical only):
        - 'ordinal': converts categories to integer codes
        - 'onehot': returns a new DataFrame with binary indicator columns

    cap_outliers : {'zscore', 'iqr'}, optional
        Method for outlier detection and clipping (numeric only):
        - 'zscore': clips values with |Z| > 3
        - 'iqr': clips values outside 1.5 * IQR from Q1/Q3

    cap_quantiles : tuple of float, default=(0.01, 0.99)
        Lower and upper quantiles to cap values (a form of Winsorization).
        Applied only for numeric columns.

    custom_map : callable, optional
        A custom function applied to every non-null element (e.g., `np.log1p`, `str.lower`).
        Useful for transformations like scaling, mapping, or normalization.

    strategy : {'manual', 'auto'}, default='manual'
        - 'manual': use the specified arguments only
        - 'auto': infer reasonable defaults based on column dtype and missing values

    inplace : bool, default=False
        If True, modifies the original Series inside a DataFrame. Otherwise works on a copy.

    preview : bool, default=False
        If True, prints a small sample (head) of the cleaned column.

    plot : bool, default=False
        If True, shows a histogram or bar plot (before/after if possible) to visualize the distribution.

    verbose : bool, default=True
        If True, logs steps taken (e.g., imputed with mean, encoded with ordinal).

    fast_mode : bool, default=False
        If True, disables plotting and previews for faster execution on large data.

    Returns
    -------
    pd.Series or Tuple[pd.Series, dict]
        - Cleaned Series (or one-hot encoded DataFrame if applicable).
        - If used in unpacking (`col_clean, steps = ...`), also returns a dictionary
          detailing the preprocessing actions taken.

    Examples
    --------
    >>> col_clean = preprocess_column(df["age"], impute="mean", scale="standard")[0]
    
    >>> cat_col, steps = preprocess_column(
            df["gender"],
            impute="mode",
            encode="ordinal",
            preview=True,
            verbose=True
        )
    >>> print(steps)

    User Guide
    ----------
    🧭 When Should You Use This?
    - You want fine-grained control over a **single column** in your dataset — e.g., apply different transformations for different features.
    - You’re doing **exploratory data analysis** and want to inspect the impact of transformations before applying them in batch.
    - You’re manually curating a feature set for a model and want to test encoding, scaling, or outlier treatment interactively.
    - You’re building a **custom preprocessing function** per column for pipelines.

    ⚙️ Recommended Workflows (Based on Column Type)

    1. **For numeric columns (e.g., age, income, price):**
       Apply standard cleaning + scaling + clipping:
       >>> col_clean, steps = preprocess_column(
               df["income"],
               impute="mean",
               scale="standard",
               cap_outliers="zscore",
               cap_quantiles=(0.01, 0.99),
               preview=True,
               plot=True
           )

       *Why?* Numeric data benefits from scaling and outlier handling for better model convergence.

    2. **For categorical columns (e.g., gender, city):**
       Apply imputation and ordinal encoding:
       >>> col_clean = preprocess_column(
               df["gender"],
               impute="mode",
               encode="ordinal"
           )[0]

       *Why?* Many models expect numeric input; ordinal encoding works well for tree models.

    3. **When exploring transformation effects:**
       Use custom mapping functions or log transforms:
       >>> preprocess_column(df["price"], custom_map=np.log1p, plot=True)

       *Why?* Helps reduce skew in price-like data, which improves linear model performance.

    4. **Quick automation:**
       Let the function auto-decide how to handle the column:
       >>> preprocess_column(df["feature_x"], strategy="auto")

       *Why?* Saves time when you’re cleaning many features quickly.

    🔍 Tips:
    - Use `preview=True` to view how the column looks after cleaning.
    - Use `plot=True` to visualize distributions before/after transformations.
    - Use `steps` output to document what happened to the column — especially useful in notebooks or reports.
    - Use `fast_mode=True` when processing high-volume datasets or in production loops.
    - One-hot encoding returns a DataFrame (not Series) — plan how you store or merge it back.

    Related:
    --------
    • preprocess_dataframe(): Clean an entire DataFrame in one call
    • summary_column(): View deep stats for a single column before preprocessing
    • summary_dataframe(): View deep stats for a DataFrame before preprocessing

    """
    col = col.copy()
    original_dtype = col.dtype
    steps = _ColumnSteps()
    steps["original_dtype"] = str(original_dtype)

    # Fastmode
    if fast_mode:
      preview = False
      verbose = False
      plot = False
    
    # Strategy
    if strategy == "auto":
        if pd.api.types.is_numeric_dtype(col):
            impute = impute or "mean"
            scale = scale or "standard"
        elif pd.api.types.is_categorical_dtype(col) or col.dtype == object:
            impute = impute or "mode"
            encode = encode or "ordinal"

    # Imputation
    if impute is not None:
        if impute == "mean":
            fill_value = col.mean()
        elif impute == "median":
            fill_value = col.median()
        elif impute == "mode":
            fill_value = col.mode().iloc[0] if not col.mode().empty else None
        elif impute == "constant":
            fill_value = 0
        elif isinstance(impute, (int, float, str)):
            fill_value = impute
        else:
            raise ValueError("Invalid impute method")
        col.fillna(fill_value, inplace=True)
        steps["imputed"] = f"{impute} → {fill_value}"

    # Custom map
    if custom_map:
        col = col.apply(lambda x: custom_map(x) if pd.notnull(x) else x)
        steps["custom_map"] = custom_map.__name__ if hasattr(custom_map, '__name__') else 'lambda'

    # Outlier Capping
    if cap_outliers == "zscore":
        zs = zscore(col.dropna())
        mask = np.abs(zs) > 3
        capped_vals = col[~mask]
        q1, q2 = capped_vals.min(), capped_vals.max()
        col = np.clip(col, q1, q2)
        steps["outlier_cap"] = f"zscore > 3 → clipped to [{q1:.2f}, {q2:.2f}]"

    elif cap_outliers == "iqr":
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr_val = q3 - q1
        lower = q1 - 1.5 * iqr_val
        upper = q3 + 1.5 * iqr_val
        col = np.clip(col, lower, upper)
        steps["outlier_cap"] = f"IQR → clipped to [{lower:.2f}, {upper:.2f}]"

    # Capping (Winsorization)
    if cap_quantiles and pd.api.types.is_numeric_dtype(col):
        low, high = col.quantile(cap_quantiles[0]), col.quantile(cap_quantiles[1])
        col = np.clip(col, low, high)
        steps["winsorize"] = f"Capped at {cap_quantiles[0]*100:.0f}–{cap_quantiles[1]*100:.0f}% quantiles"
    elif cap_quantiles and not pd.api.types.is_numeric_dtype(col):
      if verbose:
        warnings.warn("Winsorization skipped: column is not numeric.")

    # Encoding (Categorical)
    if encode and (col.dtype == "object" or col.dtype.name == "category"):
        if encode == "ordinal":
            col = pd.Series(OrdinalEncoder().fit_transform(col.values.reshape(-1, 1)).flatten(), index=col.index)
            steps["encoded"] = "ordinal"
        elif encode == "onehot":
            encoded_df = pd.get_dummies(col, prefix=col.name)
            steps["encoded"] = f"onehot → {len(encoded_df.columns)} cols"
            return encoded_df, steps
        else:
            raise ValueError("Invalid encoding method")

    # Scaling
    if scale and pd.api.types.is_numeric_dtype(col):
        scaler = {"standard": StandardScaler(),
                  "minmax": MinMaxScaler(),
                  "robust": RobustScaler()}.get(scale)

        if scaler:
            col_vals = col.values.reshape(-1, 1)
            scaled = scaler.fit_transform(col_vals).flatten()
            col = pd.Series(scaled, index=col.index)
            steps["scaled"] = scale
        else:
            raise ValueError("Invalid scale method")

    if preview:
        display(HTML(col.head(10).to_frame().to_html()))
    if plot:
        plt.figure(figsize=(8, 3))
        if pd.api.types.is_numeric_dtype(col):
            plt.hist(col.dropna(), bins=30, color="#66b3ff", edgecolor="k")
            plt.title("Numeric Column Distribution")
        else:
            col.value_counts().head(20).plot(kind="bar", color="#66b3ff", edgecolor="k")
            plt.title("Categorical Column Frequency")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    steps["final_dtype"] = str(col.dtype)
    return col, steps