import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error
from typing import Dict, List, Union, Callable, Optional, Any
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from sklearn.utils import resample
import inspect

console = Console()

def run_nested_cv_regression(
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    model_dict: Optional[Dict[str, BaseEstimator]] = None,
    param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    scoring_list: List[Union[str, Callable]] = ['r2'],
    outer_splits: int = 3,
    inner_splits: int = 3,
    random_state: int = 42,
    use_scaling: Optional[Callable[[str], bool]] = lambda name: name not in ['random_forest', 'gboost'],
    preprocessor: Optional[Pipeline] = None,
    search_method: str = 'grid',
    n_iter: int = 10,
    sample_frac: Optional[float] = None,
    max_samples: Optional[int] = None,
    fast_mode: bool = False,
    verbose: bool = True,
    return_results: bool = False,
    print_style: str = 'tabulate',
    show_plots: bool = True,
    normalize: bool = False
) -> Optional[Dict[str, Any]]:

    """
    🔁 Nested Cross-Validation for Regression Models with Hyperparameter Tuning and Scoring Diagnostics

    Evaluate multiple regression models using nested cross-validation with internal hyperparameter tuning. 
    Supports both GridSearchCV and RandomizedSearchCV, optional scaling, rich or tabulate logging styles, 
    metric customization, sampling controls, and visual diagnostics. Useful for benchmarking, model selection, 
    and automated experimentation.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape (n_samples, n_features).

    y : pd.Series or np.ndarray
        Regression target values of shape (n_samples,).

    model_dict : dict of str → estimator
        Dictionary mapping model names to scikit-learn-compatible regressors.

    param_grids : dict of str → dict
        Dictionary mapping model names to hyperparameter grids (sklearn format).

    scoring_list : list of str or callables, default=['r2']
        List of scoring metrics to evaluate (e.g., 'r2', 'rmse', 'mae', or custom functions).

    outer_splits : int, default=3
        Number of outer CV folds used for unbiased performance estimation.

    inner_splits : int, default=3
        Number of inner CV folds used for hyperparameter tuning.

    random_state : int, default=42
        Seed for reproducibility across cross-validation and sampling.

    use_scaling : callable, default=lambda name: name not in ['random_forest', 'gboost']
        Function to determine whether a model should use standard scaling.

    preprocessor : sklearn-compatible transformer or Pipeline, optional
        Custom preprocessing steps. Overrides automatic scaling if provided.

    search_method : {'grid', 'random'}, default='grid'
        - 'grid': exhaustive grid search.
        - 'random': randomized search using `n_iter` samples from the grid.

    n_iter : int, default=10
        Number of iterations for random search (used only when `search_method='random'`).

    sample_frac : float, optional
        If provided, randomly samples a fraction of the data (e.g., 0.1 = 10%).

    max_samples : int, optional
        If provided, limits total training size to this number of rows.

    fast_mode : bool, default=False
        If True, reduces CV folds and skips visuals/logs for faster experimentation.

    verbose : bool, default=True
        Whether to print progress, best parameters, and summary tables.

    return_results : bool, default=False
        If True, returns a dictionary with detailed results and summary plots.

    print_style : {'tabulate', 'rich'}, default='tabulate'
        Formatting style for printed summaries (console-friendly or rich-colored).

    show_plots : bool, default=True
        If True, generates metric comparison plots using matplotlib/seaborn.

    normalize : bool, default=False
        If True, normalizes metric scales for side-by-side comparison across metrics.

    Returns
    -------
    dict, optional
        Only if `return_results=True`:
        - 'summary': pd.DataFrame summarizing all models and metrics
        - 'results': full nested CV scores and best params
        - 'best_params': dictionary of best params for each model/metric
        - 'figures': matplotlib figure objects (if plots were generated)

    Raises
    ------
    ValueError
        If required inputs (X, y, model_dict, param_grids) are missing.

    Examples
    --------
    >>> models = {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor()
        }

    >>> grids = {
            'linear': {},
            'rf': {'n_estimators': [50, 100], 'max_depth': [3, 5]}
        }

    >>> run_nested_cv_regression(
            X=X, y=y,
            model_dict=models,
            param_grids=grids,
            scoring_list=['r2', 'rmse', 'mae'],
            search_method='random',
            n_iter=5,
            fast_mode=True,
            show_plots=False
        )

    Notes
    -----
    🔍 Scoring Options:
        • Standard metrics: 'r2', 'adjusted_r2', 'mae', 'rmse', 'rmsle', 'mape'
        • Custom scorers: pass a callable with (y_true, y_pred) → float
        • Negative scores (e.g., RMSE) are automatically converted to positive.

    📈 Visualization:
        - Score plots (e.g., R², explained variance) are grouped separately from error plots (e.g., RMSE, MAE).
        - Use `normalize=True` to enable relative comparison across all metrics in a single chart.

    ⚡ Tips:
        - Use `sample_frac` or `max_samples` for speed on large datasets.
        - `fast_mode=True` is useful for test runs or CI/CD pipelines.
        - Use `return_results=True` to integrate output into reports or notebooks.

    See Also
    --------
    - run_nested_cv_classification : For binary/multiclass classification
    - evaluate_regression_model : For detailed regression diagnostics
    - GridSearchCV, RandomizedSearchCV : Sklearn tuning tools
    """

    if X is None or y is None or model_dict is None or param_grids is None:
        console.print("[bold red]ERROR:[/bold red] Missing required parameters: 'X', 'y', 'model_dict', 'param_grids'.")
        doc = inspect.getdoc(run_nested_cv_regression)
        console.print(f"\n[bold cyan]Docstring:[/bold cyan]\n\n{doc}")
        return

    # Sampling logic
    if sample_frac:
        X, y = resample(X, y, replace=False, n_samples=int(len(X) * sample_frac), random_state=random_state)
    elif max_samples and len(X) > max_samples:
        X, y = resample(X, y, replace=False, n_samples=max_samples, random_state=random_state)

    y = np.ravel(y)
    outer_cv = KFold(n_splits=2 if fast_mode else outer_splits, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=2 if fast_mode else inner_splits, shuffle=True, random_state=random_state)

    results, summary_rows = {}, []

    for model_name, model in model_dict.items():
        pretty_model_name = model_name.replace("_", " ").upper()
        if not fast_mode:
            print(f"\n{'='*60}\n Running Nested CV for: {pretty_model_name}\n{'='*60}")

        steps = [('scaler', StandardScaler())] if use_scaling(model_name) and preprocessor is None else []
        if preprocessor:
            steps = preprocessor.steps.copy()
        steps.append(('clf', model))
        pipe = Pipeline(steps)

        param_grid = {f'clf__{k}': v for k, v in param_grids[model_name].items()}
        results[model_name] = {}

        for scoring in scoring_list:
            if isinstance(scoring, str):
                scoring_label = scoring
                postprocess = None

                if scoring == 'rmse':
                    scorer = 'neg_mean_squared_error'
                    postprocess = lambda s: np.sqrt(-s)
                elif scoring == 'rmsle':
                    scorer = 'neg_mean_squared_log_error'
                    postprocess = lambda s: np.sqrt(-s)
                elif scoring == 'mae':
                    scorer = 'neg_mean_absolute_error'
                    postprocess = lambda s: -s
                elif scoring == 'mape':
                    scorer = make_scorer(lambda yt, yp: mean_absolute_percentage_error(yt, yp), greater_is_better=False)
                    postprocess = lambda s: -s
                elif scoring == 'adjusted_r2':
                    scorer = make_scorer(lambda yt, yp: 1 - ((1 - r2_score(yt, yp)) * (len(yt) - 1) / (len(yt) - X.shape[1] - 1)))
                else:
                    scorer = scoring
            else:
                scoring_label = scoring.__name__
                scorer = scoring
                postprocess = None

            if not fast_mode:
                print(f"\n→ Scoring Metric: {scoring_label.upper()}")

            if search_method == 'grid':
                grid = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv, scoring=scorer, n_jobs=-1)
            else:
                grid = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=n_iter,
                                          cv=inner_cv, scoring=scorer, n_jobs=-1, random_state=random_state)

            raw_scores = cross_val_score(grid, X, y, cv=outer_cv, scoring=scorer, n_jobs=-1)
            scores = postprocess(raw_scores) if postprocess else raw_scores

            grid.fit(X, y)
            best_params = dict(sorted(grid.best_params_.items()))
            mean_score = scores.mean()
            std_score = scores.std()

            if not fast_mode and verbose:
                if print_style == 'rich':
                    score_table = Table(show_header=True, header_style="bold magenta")
                    score_table.add_column("Mean Score")
                    score_table.add_column("Std Dev")
                    score_table.add_row(f"{mean_score:.4f}", f"{std_score:.4f}")
                    console.print(score_table)

                    console.print("[bold green]Best Hyperparameters:[/bold green]")
                    for k in sorted(best_params):
                        console.print(f"   [cyan]- {k}: {best_params[k]}")
                else:
                    print(tabulate([[f"{mean_score:.4f}", f"{std_score:.4f}"]], headers=["Mean Score", "Std Dev"], tablefmt="pretty"))
                    print("Best Hyperparameters:")
                    for k in sorted(best_params):
                        print(f"   - {k}: {best_params[k]}")

            results[model_name][scoring_label] = {
                'score_mean': mean_score,
                'score_std': std_score,
                'best_params': best_params
            }

            summary_rows.append({
                'Model': model_name,
                'Metric': scoring_label,
                'Mean Score': mean_score,
                'Std Dev': std_score
            })

    summary_df = pd.DataFrame(summary_rows)

    if not fast_mode:
        print(f"\n{'='*60}\n Final Model Performance Summary\n{'='*60}")
        if print_style == 'rich':
            table = Table(title="Final Model Performance Summary", header_style="bold magenta")
            table.add_column("Model")
            table.add_column("Metric")
            table.add_column("Mean Score", justify="right")
            table.add_column("Std Dev", justify="right")
            for _, row in summary_df.iterrows():
                table.add_row(
                    row['Model'],
                    row['Metric'],
                    f"{row['Mean Score']:.4f}",
                    f"{row['Std Dev']:.4f}"
                )
            console.print(table)
        else:
            print(tabulate(summary_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    if show_plots and not fast_mode:
        print("\n Generating Model Comparison Visuals...")
        plot_df = summary_df.pivot(index='Model', columns='Metric', values='Mean Score')

        if normalize:
            plot_df_norm = plot_df.copy()
            for metric in plot_df_norm.columns:
                col = plot_df_norm[metric]
                if metric in {'rmse', 'mae', 'mape', 'rmsle'}:
                    col = -col
                min_, max_ = col.min(), col.max()
                plot_df_norm[metric] = (col - min_) / (max_ - min_ + 1e-8)

            fig_norm, ax = plt.subplots(figsize=(10, 6))
            plot_df_norm.plot(kind='bar', ax=ax, rot=0)
            ax.set_title("Normalized Model Comparison (All Metrics)")
            ax.set_ylabel("Normalized Score")
            ax.set_xlabel("Model")
            ax.grid(axis='y')
            ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
        else:
            error_metrics = {'mae', 'rmse', 'rmsle', 'mape'}
            score_metrics = set(plot_df.columns) - error_metrics

            sns.set(style="whitegrid")
            palette = sns.color_palette("Set2", n_colors=len(plot_df))

            if score_metrics:
                fig_score, ax = plt.subplots(figsize=(10, 6))
                plot_df[sorted(score_metrics)].plot(kind='bar', ax=ax, color=palette, rot=0)
                ax.set_title("Model Comparison (Score Metrics)")
                ax.set_ylabel("Score")
                ax.set_xlabel("Model")
                ax.grid(axis='y')
                ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.show()

            if error_metrics & set(plot_df.columns):
                fig_error, ax = plt.subplots(figsize=(10, 6))
                plot_df[sorted(error_metrics & set(plot_df.columns))].plot(kind='bar', ax=ax, color=palette, rot=0)
                ax.set_title("Model Comparison (Error Metrics)")
                ax.set_ylabel("Error")
                ax.set_xlabel("Model")
                ax.grid(axis='y')
                ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.show()

    if return_results:
        return {
            'summary': summary_df,
            'results': results,
            'best_params': {
                model: {metric: results[model][metric]['best_params'] for metric in results[model]}
                for model in results
            }
        }
