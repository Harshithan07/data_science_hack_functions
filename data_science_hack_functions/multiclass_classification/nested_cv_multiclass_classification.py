import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target
from typing import Dict, List, Union, Callable, Optional, Any
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from sklearn.utils import resample
import inspect

console = Console()

def run_nested_cv_multiclass_classification(
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    model_dict: Optional[Dict[str, BaseEstimator]] = None,
    param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    scoring_list: List[Union[str, Callable]] = ['accuracy'],
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
    show_plots: bool = True
) -> Optional[Dict[str, Any]]:

    """
    Run nested cross-validation for multiclass classification models with internal tuning, scaling, and visualization.

    This function supports nested cross-validation for evaluating multiple classifiers on multiclass targets. 
    It performs hyperparameter tuning using either GridSearchCV or RandomizedSearchCV within inner folds and evaluates 
    model performance on outer folds. Additional features include smart sub-sampling, optional standardization, and 
    visual summary generation using matplotlib/seaborn.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (n_samples, n_features).

    y : pd.Series or np.ndarray
        Multiclass classification target variable (n_samples,).

    model_dict : dict of str → estimator
        Dictionary mapping model names to scikit-learn-compatible classifiers.

    param_grids : dict of str → dict
        Dictionary mapping model names to their hyperparameter grids.

    scoring_list : list of str or callable, default=['accuracy']
        List of scoring metrics for evaluation (e.g., ['accuracy', 'f1_macro', 'recall_macro']).

    outer_splits : int, default=3
        Number of outer CV folds (used to estimate generalization performance).

    inner_splits : int, default=3
        Number of inner CV folds (used for hyperparameter tuning).

    random_state : int, default=42
        Random seed for reproducibility across resampling and CV.

    use_scaling : callable, default=lambda name: name not in ['random_forest', 'gboost']
        Function that returns True/False depending on whether a model should use `StandardScaler`.

    preprocessor : sklearn-compatible transformer, optional
        Custom preprocessing pipeline. If provided, overrides automatic scaling logic.

    search_method : {'grid', 'random'}, default='grid'
        Search strategy to use:
            - 'grid': exhaustive search using GridSearchCV
            - 'random': random search using RandomizedSearchCV

    n_iter : int, default=10
        Number of parameter combinations to try (only used for random search).

    sample_frac : float, optional
        If provided, samples this fraction (e.g., 0.1 = 10%) of total data.

    max_samples : int, optional
        If provided, caps total sample count to this number.

    fast_mode : bool, default=False
        If True, reduces CV splits to 2 and skips plots/logs for speed.

    verbose : bool, default=True
        Whether to print intermediate logs and tuning results.

    return_results : bool, default=False
        If True, returns dictionary of performance summaries, best parameters, and figures.

    print_style : {'tabulate', 'rich'}, default='tabulate'
        Controls formatting of logs — rich formatting vs plain ASCII tables.

    show_plots : bool, default=True
        If False, disables visualization of model comparisons.

    Returns
    -------
    dict, optional
        If `return_results=True`, returns:
        - 'summary': DataFrame with final evaluation of all models/metrics
        - 'results': Nested dictionary of raw scores and best params per model/metric
        - 'best_params': Extracted best hyperparameters per model and metric
        - 'figures': Optional matplotlib plots (if show_plots is True)

    Raises
    ------
    ValueError
        If any required argument (`X`, `y`, `model_dict`, `param_grids`) is missing.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier

    >>> models = {
    ...     'logistic': LogisticRegression(max_iter=200),
    ...     'rf': RandomForestClassifier()
    ... }

    >>> grids = {
    ...     'logistic': {'C': [0.1, 1, 10]},
    ...     'rf': {'n_estimators': [50, 100]}
    ... }

    >>> results = run_nested_cv_multiclass_classification(
    ...     X=X, y=y,
    ...     model_dict=models,
    ...     param_grids=grids,
    ...     scoring_list=['accuracy', 'f1_macro'],
    ...     search_method='grid',
    ...     show_plots=True,
    ...     return_results=True
    ... )

    >>> print(results['summary'])

    Notes
    -----
    🔁 What is Nested Cross-Validation?
        Nested CV is used to provide an unbiased estimate of model performance while tuning hyperparameters.
        The outer loop evaluates model generalization, while the inner loop tunes parameters.

    📊 Supported Metrics:
        - accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, etc.
        - Custom scoring functions are also supported.

    ⚙️ Recommended Settings:
        - Use grid search for small, well-bounded hyperparameter grids.
        - Use random search with `n_iter` for wide or expensive search spaces.
        - Set `fast_mode=True` for quick experiments or when plotting/logging isn’t needed.

    📉 Memory-Saving Tips:
        - Use `sample_frac` or `max_samples` for faster computation.
        - Avoid plotting in low-memory environments by disabling `show_plots`.

    📦 Outputs:
        - Compact table of results across models/metrics
        - Visual comparisons of bounded (0–1) and unbounded metrics
        - Structured dictionary for pipeline integration or downstream analysis

    See Also
    --------
    - GridSearchCV, RandomizedSearchCV (sklearn)
    - evaluate_multiclass_classification_model: For detailed model diagnostics post-training
    """


    if X is None or y is None or model_dict is None or param_grids is None:
        console.print("[bold red]ERROR:[/bold red] Missing required parameters: 'X', 'y', 'model_dict', 'param_grids'.")
        doc = inspect.getdoc(run_nested_cv_multiclass_classification)
        console.print(f"\n[bold cyan]Docstring:[/bold cyan]\n\n{doc}")
        return

    if type_of_target(y) != 'multiclass':
        raise ValueError("This function only supports multiclass classification targets.")

    if sample_frac:
        X, y = resample(X, y, replace=False, n_samples=int(len(X) * sample_frac), random_state=random_state)
    elif max_samples and len(X) > max_samples:
        X, y = resample(X, y, replace=False, n_samples=max_samples, random_state=random_state)

    y = pd.Series(y).astype(str)

    outer_cv = StratifiedKFold(n_splits=2 if fast_mode else outer_splits, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=2 if fast_mode else inner_splits, shuffle=True, random_state=random_state)

    results, summary_rows = {}, []

    for model_name, model in model_dict.items():
        if not fast_mode:
            console.print(f"[bold blue]\nModel: {model_name.upper()}[/bold blue]")

        steps = [('scaler', StandardScaler())] if use_scaling(model_name) and preprocessor is None else []
        if preprocessor:
            steps = preprocessor.steps.copy()
        steps.append(('clf', model))
        pipe = Pipeline(steps)

        param_grid = {f'clf__{k}': v for k, v in param_grids[model_name].items()}
        results[model_name] = {}

        for scoring in scoring_list:
            scoring_label = scoring.upper() if isinstance(scoring, str) else 'CUSTOM SCORER'
            if not fast_mode:
                console.print(f"\n→ Scoring Metric: {scoring_label}")

            if search_method == 'grid':
                grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring=scoring, n_jobs=-1)
            else:
                grid = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=n_iter,
                                          cv=inner_cv, scoring=scoring, n_jobs=-1, random_state=random_state)

            nested_scores = cross_val_score(grid, X, y, cv=outer_cv, scoring=scoring, n_jobs=-1)
            grid.fit(X, y)
            best_params = dict(sorted(grid.best_params_.items()))
            mean_score = nested_scores.mean()
            std_score = nested_scores.std()

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

            results[model_name][scoring] = {
                'score_mean': mean_score,
                'score_std': std_score,
                'best_params': best_params
            }

            summary_rows.append({
                'Model': model_name,
                'Metric': scoring,
                'Mean Score': mean_score,
                'Std Dev': std_score
            })

    summary_df = pd.DataFrame(summary_rows)

    if not fast_mode and verbose:
        console.print("\n[bold yellow]Final Model Performance Summary[/bold yellow]")
        if print_style == 'rich':
            table = Table(title="Summary", header_style="bold magenta")
            table.add_column("Model")
            table.add_column("Metric")
            table.add_column("Mean Score", justify="right")
            table.add_column("Std Dev", justify="right")
            for _, row in summary_df.iterrows():
                table.add_row(row['Model'], row['Metric'], f"{row['Mean Score']:.4f}", f"{row['Std Dev']:.4f}")
            console.print(table)
        else:
            print(tabulate(summary_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    if not fast_mode and show_plots:
        plot_df = summary_df.pivot(index='Model', columns='Metric', values='Mean Score')
        bounded_metrics = {'accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'balanced_accuracy'}
        all_metrics = set(plot_df.columns)
        bounded = sorted(list(all_metrics.intersection(bounded_metrics)))
        unbounded = sorted(list(all_metrics.difference(bounded_metrics)))

        palette = sns.color_palette("Set2", n_colors=len(plot_df))
        if bounded:
            fig_bounded, ax = plt.subplots(figsize=(10, 6))
            plot_df[bounded].plot(kind='bar', ax=ax, color=palette, rot=0)
            ax.set_title("Model Comparison (Bounded Metrics 0–1)")
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.show()

        if unbounded:
            fig_unbounded, axes = plt.subplots(1, len(unbounded), figsize=(6 * len(unbounded), 6), sharey=True)
            if len(unbounded) == 1:
                axes = [axes]
            for ax, metric in zip(axes, unbounded):
                plot_df[metric].sort_values().plot(kind='barh', ax=ax, color=palette)
                ax.set_title(f"Model Comparison ({metric})")
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
