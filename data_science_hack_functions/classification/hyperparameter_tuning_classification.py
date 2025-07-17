import numpy as np
import pandas as pd
import optuna
import time
import inspect
from typing import Any, Callable, Dict, Union, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import get_scorer
from tabulate import tabulate
from rich.console import Console
from sklearn.pipeline import make_pipeline


console = Console()


def hyperparameter_tuning_classification(
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    model_class: Optional[Callable[..., Any]] = None,
    param_grid: Optional[Dict[str, Callable[[optuna.Trial], Any]]] = None,
    scoring: Union[str, Callable] = 'accuracy',
    n_trials: int = 50,
    cv_folds: int = 5,
    stratified: bool = True,
    direction: str = 'maximize',
    verbose: bool = True,
    return_model: bool = True,
    random_state: int = 42,
    use_fraction: Optional[float] = None,
    use_n_samples: Optional[int] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    🔧 Hyperparameter Tuning for Classification (via Optuna)

    Optimize hyperparameters of any binary classification model using Optuna’s efficient 
    sampling. This function supports cross-validation, custom scoring, stratified sampling, 
    reproducibility, and returns the best trial summary and optionally the best fitted model.

    Ideal for automated model selection pipelines, leaderboard tuning, or experimentation 
    in research and production ML workflows.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix with shape (n_samples, n_features).

    y : pd.Series or np.ndarray
        Target vector with binary labels (0 or 1).

    model_class : callable
        A scikit-learn-style classifier class (e.g., `RandomForestClassifier`, `LogisticRegression`).
        Not an instance – must be the class itself.

    param_grid : dict
        Dictionary mapping hyperparameter names to Optuna sampling functions.
        Example:
            {
                "C": lambda t: t.suggest_float("C", 0.01, 10, log=True),
                "penalty": lambda t: t.suggest_categorical("penalty", ["l1", "l2"])
            }

    scoring : str or callable, default='accuracy'
        Metric to optimize during cross-validation. Supports:
        - Any string from sklearn (e.g., 'f1', 'roc_auc', 'log_loss')
        - A custom callable with signature: scorer(estimator, X_val, y_val)

    n_trials : int, default=50
        Number of optimization trials to run.

    cv_folds : int, default=5
        Number of folds in cross-validation to evaluate each hyperparameter setting.

    stratified : bool, default=True
        If True, use StratifiedKFold for cross-validation (preserves class balance).

    direction : {'maximize', 'minimize'}, default='maximize'
        Whether to maximize or minimize the scoring function.

    verbose : bool, default=True
        If True, prints trial progress, parameter table, and best result summary.

    return_model : bool, default=True
        If True, fits and returns the best model using the entire dataset.

    random_state : int, default=42
        Random seed for reproducible folds and results.

    use_fraction : float or None, optional
        If provided, samples a random fraction (e.g., 0.1 = 10%) of the dataset.

    use_n_samples : int or None, optional
        If provided, uses only the first N rows of the data.

    fast_mode : bool, default=False
        If True, reduces `n_trials` to 10, disables logs, and optimizes speed.
        Use for quick tests or large-scale experiments.

    Returns
    -------
    dict
        A dictionary containing:
        - 'best_score' : float
        - 'best_params' : dict
        - 'study' : optuna.Study
        - 'best_model' : fitted model (if return_model=True)

    Raises
    ------
    ValueError
        If required arguments are missing or incompatible.

    User Guide
    ----------
    🧠 When to Use:
    - You're comparing models or trying to find optimal settings for one.
    - You want to replace GridSearchCV with faster, smarter search.
    - You want control over cross-validation, scoring, and sampling.
    - You’re tuning on large datasets and want quick feedback via fast_mode or subsampling.

    📌 Core Concepts:

    • Optuna Trials:
        Each trial samples a different set of parameters using Optuna’s intelligent search strategy
        (Tree Parzen Estimator) and evaluates them using cross-validation.

    • param_grid:
        Define hyperparameter ranges via lambdas — much more flexible than grid search.
        Supports `suggest_float`, `suggest_int`, `suggest_categorical`, and log-scale sampling.

    • scoring:
        Choose a string (e.g., 'f1', 'accuracy', 'roc_auc') or define a custom scoring function
        that returns a float. Examples:
            - 'neg_log_loss': minimizes log loss (automatically handled)
            - custom_cost(estimator, X, y): returns cost/loss based on predictions

    • StratifiedKFold:
        Recommended for binary classification, especially if classes are imbalanced.

    • fast_mode:
        Use fast_mode=True to reduce trials to 10, turn off prints, and return results quickly.
        Ideal for initial tests or iterative tuning on large data.

    • Sampling Subsets:
        - `use_fraction=0.1`: randomly sample 10% of the data
        - `use_n_samples=5000`: use only the first 5000 rows
        - You can pass both; `use_n_samples` is applied after `use_fraction`

    🧪 Example Usage:
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> results = hyperparameter_tuning_classification(
            X=X, y=y,
            model_class=RandomForestClassifier,
            param_grid={
                "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 300),
                "max_depth": lambda t: t.suggest_int("max_depth", 3, 15),
                "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10),
            },
            scoring='f1',
            n_trials=50,
            stratified=True,
            fast_mode=False,
            return_model=True
        )

    >>> print(results["best_params"])
    >>> print(results["best_score"])
    >>> best_model = results["best_model"]

    💡 Tips:
    - Use fewer trials with small data or fast models (e.g., LogisticRegression).
    - Tune for `log_loss` when probabilistic accuracy matters (e.g., fraud detection).
    - Always use `return_model=True` in production workflows to get the fitted model.
    - Set `fast_mode=True` when running in a loop or with large datasets.
    - Pair with `evaluate_classification_model()` for post-tuning performance analysis.

    See Also
    --------
    evaluate_classification_model : Full evaluation suite for binary classifiers.
    GridSearchCV : Traditional brute-force alternative (slower, less efficient).
    Optuna : https://optuna.org
    """
    if any(param is None for param in [X, y, model_class, param_grid]):
        console.print("[bold red]\nERROR:[/bold red] Missing required arguments: 'X', 'y', 'model_class', and 'param_grid'.")
        doc = inspect.getdoc(hyperparameter_tuning_classification)
        console.print(f"\n[bold cyan]Docstring:[/bold cyan]\n\n{doc}")
        return

    if fast_mode:
        verbose = False
        n_trials = min(n_trials, 10)

    if use_fraction is not None:
        idx = np.random.choice(len(X), int(len(X) * use_fraction), replace=False)
        X = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        y = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
    elif use_n_samples is not None:
        idx = np.random.choice(len(X), min(use_n_samples, len(X)), replace=False)
        X = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        y = y.iloc[idx] if hasattr(y, "iloc") else y[idx]

    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    elif callable(scoring):
        scorer = scoring
    else:
        raise ValueError("'scoring' must be a string or a callable taking (estimator, X, y).")

    if verbose:
        console.rule("[bold blue]🔍 Optuna Hyperparameter Tuning")
        console.print(f"[bold]Model        :[/bold] {model_class.__name__}")
        console.print(f"[bold]Metric       :[/bold] {scoring if isinstance(scoring, str) else 'custom'}")
        console.print(f"[bold]Trials       :[/bold] {n_trials}")
        console.print(f"[bold]CV Folds     :[/bold] {cv_folds}")
        console.print(f"[bold]Stratified   :[/bold] {str(stratified)}")
        console.print(f"[bold]Fast Mode    :[/bold] {'✅' if fast_mode else '❌'}")
        console.rule()

    def objective(trial):
        params = {k: v(trial) for k, v in param_grid.items()}
        model = model_class(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state) if stratified else KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            X_test  = X.iloc[test_idx]  if hasattr(X, "iloc") else X[test_idx]
            y_test  = y.iloc[test_idx]  if hasattr(y, "iloc") else y[test_idx]
            try:
                model.fit(X_train, y_train)
                score = scorer(model, X_test, y_test)
                scores.append(score)
            except Exception as e:
                return float('-inf') if direction == 'maximize' else float('inf')
        return np.mean(scores)

    start_time = time.time()
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - start_time

    best_params = study.best_params
    best_score = study.best_value

    if verbose:
        console.rule("[bold green]✅ Best Results")
        console.print(f"[bold]Best Score:[/bold] {best_score:.5f}")
        console.print("\n[bold]Best Hyperparameters:[/bold]")
        console.print(tabulate(best_params.items(), headers=["Hyperparameter", "Value"], tablefmt="fancy_grid"))
        console.print(f"[bold]Elapsed Time:[/bold] {elapsed:.2f} seconds")

    output = {
        'best_score': best_score,
        'best_params': best_params,
        'study': study
    }

    if return_model:
        best_model = model_class(**best_params)
        best_model.fit(X, y)
        output['best_model'] = best_model

    return output