import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from typing import Dict, List, Union, Callable, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from sklearn.utils import resample
import inspect

console = Console()

def run_nested_cv_classification(
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
    search_method: str = 'grid',  # or 'random'
    n_iter: int = 10,  # for random search
    sample_frac: Optional[float] = None,
    max_samples: Optional[int] = None,
    fast_mode: bool = False,
    verbose: bool = True,
    return_results: bool = False,
    print_style: str = 'tabulate'
) -> Optional[Dict[str, Any]]:

    """
    Run nested cross-validation for multiple classification models with optional tuning strategies and scaling.

    This function evaluates one or more classification models using nested cross-validation — 
    an approach that combines robust performance estimation with internal hyperparameter tuning. 
    It supports both GridSearchCV and RandomizedSearchCV, allows sub-sampling for large datasets, 
    and offers a simplified or silent fast mode for production or large-scale runs.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (n_samples, n_features).
    y : pd.Series or np.ndarray
        Binary classification target variable (n_samples,).

    model_dict : dict of str → estimator
        Dictionary mapping model names to scikit-learn classifiers.

    param_grids : dict of str → dict
        Dictionary mapping model names to hyperparameter grids in sklearn format.

    scoring_list : list of str or callables, default=['accuracy']
        List of scoring metrics to evaluate during nested CV (e.g., ['accuracy', 'roc_auc']).

    outer_splits : int, default=3
        Number of folds in the outer CV loop (used for final performance estimation).

    inner_splits : int, default=3
        Number of folds in the inner CV loop (used for hyperparameter tuning).

    search_method : {'grid', 'random'}, default='grid'
        Search strategy to use for tuning:
        - 'grid': exhaustive parameter search (GridSearchCV)
        - 'random': random search with `n_iter` (RandomizedSearchCV)

    n_iter : int, optional
        Only used when `search_method='random'`. Number of parameter settings sampled.

    max_samples : int, optional
        If provided, limits the number of rows used for fitting to this count (e.g., 100_000).

    sample_frac : float, optional
        If provided, randomly samples this fraction of rows (e.g., 0.1 for 10%).

    use_scaling : callable, default=lambda name: name not in ['random_forest', 'gboost']
        Function to determine if a given model should use `StandardScaler`.

    preprocessor : sklearn-compatible transformer, optional
        If provided, replaces built-in scaling logic with a user-defined transformer or pipeline.

    verbose : bool, default=True
        If True, prints progress, hyperparameter tables, and detailed summaries.

    fast_mode : bool, default=False
        If True, disables visual output and reduces outer CV folds to speed up runtime.

    return_results : bool, default=False
        If True, returns a dictionary with summaries, best parameters, and matplotlib figures.

    print_style : {'tabulate', 'rich'}, default='tabulate'
        Controls how summary tables are printed (colorful or ASCII).

    Returns
    -------
    dict, optional
        Only if `return_results=True`. Includes:
        - 'summary': pd.DataFrame of all model-metric combinations
        - 'results': detailed CV results and best parameters
        - 'best_params': dict of best params for each model and metric
        - 'figures': optional comparison plots (if fast_mode=False)

    Raises
    ------
    ValueError
        If any of the required inputs (`X`, `y`, `model_dict`, `param_grids`) are missing.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier

    >>> models = {
            'logistic': LogisticRegression(),
            'rf': RandomForestClassifier()
        }

    >>> grids = {
            'logistic': {'C': [0.1, 1, 10]},
            'rf': {'n_estimators': [50, 100]}
        }

    >>> results = run_nested_cv_classification(
            X=X, y=y,
            model_dict=models,
            param_grids=grids,
            scoring_list=['accuracy', 'f1'],
            fast_mode=True,
            search_method='random',
            n_iter=5,
            max_samples=10000,
            return_results=True
        )

    >>> print(results['summary'])

    User Guide
    ----------
    🧪 What is Nested CV?
        - Nested cross-validation helps prevent overfitting in model selection by using an inner loop 
          for hyperparameter tuning and an outer loop for estimating generalization performance.

    📌 Common Use Cases:
        - Benchmarking many classifiers fairly.
        - Avoiding bias from tuning and testing on the same split.
        - Handling imbalanced or high-cardinality feature spaces.

    ⚙️ When to Use Which Search Strategy:
        - Use `'grid'` for small hyperparameter spaces where you want exact control.
        - Use `'random'` when the space is large or you're optimizing speed vs. accuracy.

    ⚡ When to Use `fast_mode=True`:
        - During large-scale experiments (e.g., 10M+ rows).
        - If visual plots or verbose logging are unnecessary.
        - In production or headless environments.

    🔍 Interpreting the Output:
        - Mean/Std Dev: Represents model stability across folds.
        - Best Params: Optimal hyperparameters found by inner CV.
        - Figures: Comparison plots for accuracy, F1, AUC, etc.

    📉 Scaling Rules:
        - By default, tree-based models skip scaling.
        - You can override this using `use_scaling()` or provide a custom `preprocessor`.

    🚫 Memory Tips for Big Data:
        - Use `max_samples` or `sample_frac` to downsample safely.
        - Enable `fast_mode` to avoid expensive operations like seaborn/Matplotlib rendering.

    See Also
    --------
    - GridSearchCV, RandomizedSearchCV (from sklearn)
    - evaluate_classification_model() for post-training model diagnostics
  
    """

    if X is None or y is None or model_dict is None or param_grids is None:
        console.print("[bold red]ERROR:[/bold red] Missing required parameters: 'X', 'y', 'model_dict', 'param_grids'.")
        doc = inspect.getdoc(run_nested_cv_classification)
        console.print(f"\n[bold cyan]Docstring:[/bold cyan]\n\n{doc}")
        return

    # Apply sampling if needed
    if sample_frac:
        X, y = resample(X, y, replace=False, n_samples=int(len(X) * sample_frac), random_state=random_state)
    elif max_samples and len(X) > max_samples:
        X, y = resample(X, y, replace=False, n_samples=max_samples, random_state=random_state)

    y = np.ravel(y)
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
            if search_method == 'grid':
                grid = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv, scoring=scoring, n_jobs=-1)
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
                    print(tabulate([[f"{mean_score:.4f}", f"{std_score:.4f}"]],
                                   headers=["Mean Score", "Std Dev"], tablefmt="pretty"))
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

    if not fast_mode:
        plot_df = summary_df.pivot(index='Model', columns='Metric', values='Mean Score')
        bounded_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision', 'balanced_accuracy'}
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
