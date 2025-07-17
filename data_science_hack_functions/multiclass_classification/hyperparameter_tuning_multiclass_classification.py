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

console = Console()

def hyperparameter_tuning_multiclass_classification(
    X: Union[pd.DataFrame, np.ndarray] = None,
    y: Union[pd.Series, np.ndarray] = None,
    model_class: Callable[..., Any] = None,
    param_grid: Dict[str, Callable[[optuna.Trial], Any]] = None,
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
    🎯 Hyperparameter Tuning for Multiclass Classification (via Optuna)

    Optimize hyperparameters of any multiclass classification model using Optuna’s efficient 
    search algorithm. This function supports cross-validation (with stratification), custom 
    or built-in scoring metrics, reproducibility, and flexible data sampling. Ideal for 
    experimentation, model leaderboard tuning, and automated pipelines.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape (n_samples, n_features).

    y : pd.Series or np.ndarray
        Target vector with multiclass labels (3+ classes supported).

    model_class : callable
        A scikit-learn-style classifier class (e.g., `RandomForestClassifier`, `LogisticRegression`).
        Should be the class itself, not an instantiated object.

    param_grid : dict
        Dictionary mapping hyperparameter names to Optuna search functions.
        Example:
            {
                "C": lambda t: t.suggest_float("C", 0.01, 10, log=True),
                "solver": lambda t: t.suggest_categorical("solver", ["lbfgs", "saga"])
            }

    scoring : str or callable, default='accuracy'
        Evaluation metric to optimize. Can be:
        - Any valid sklearn scorer string (e.g., 'f1_macro', 'balanced_accuracy', 'neg_log_loss')
        - A custom scoring function with signature: scorer(estimator, X_val, y_val)

    n_trials : int, default=50
        Number of Optuna trials to run.

    cv_folds : int, default=5
        Number of folds for cross-validation during evaluation.

    stratified : bool, default=True
        Whether to use StratifiedKFold (preserves class distribution across folds).
        Set to False to use standard KFold.

    direction : {'maximize', 'minimize'}, default='maximize'
        Whether to maximize or minimize the scoring function.

    verbose : bool, default=True
        If True, prints Optuna progress, trial summary, and best results.

    return_model : bool, default=True
        If True, fits and returns the best model using the entire dataset.

    random_state : int, default=42
        Seed for reproducibility (used in CV splitting and Optuna sampler).

    use_fraction : float or None, optional
        Randomly sample a fraction of the dataset (e.g., 0.1 = 10%).

    use_n_samples : int or None, optional
        Limit dataset to first N rows (after fraction sampling, if both are set).

    fast_mode : bool, default=False
        If True:
            - Limits trials to 10
            - Disables verbose logs
            - Useful for quick experimentation

    Returns
    -------
    dict
        A dictionary containing:
        - 'best_score' : float
        - 'best_params' : dict
        - 'study' : optuna.study.Study
        - 'best_model' : trained model (if return_model=True)

    Raises
    ------
    ValueError
        If required parameters are missing or scoring is misconfigured.

    User Guide
    ----------
    🎯 When to Use:
    - You're tuning multiclass classification models (3+ classes).
    - You want intelligent hyperparameter search using Optuna instead of GridSearchCV.
    - You require macro/micro scoring for imbalanced datasets.
    - You need flexible sampling and control over reproducibility.

    📌 Core Concepts:

    • Multiclass Scoring:
        Use `'f1_macro'`, `'balanced_accuracy'`, or other sklearn scoring strings.
        You can also define a custom scorer that returns a float (e.g., macro recall).

    • Optuna Trials:
        Each trial samples hyperparameters and evaluates them via cross-validation.
        Optuna uses Tree Parzen Estimator (TPE) by default for efficient search.

    • param_grid:
        A dictionary of Optuna sampling functions. Example:
            {
                'max_depth': lambda t: t.suggest_int('max_depth', 3, 10),
                'min_samples_split': lambda t: t.suggest_int('min_samples_split', 2, 20)
            }

    • StratifiedKFold:
        Recommended when class distribution is skewed. Ensures representative folds.

    • fast_mode:
        Enables a faster tuning mode by reducing trials and silencing logs. Great for early tests.

    • Sampling Subsets:
        - `use_fraction=0.1`: randomly samples 10% of the data
        - `use_n_samples=5000`: takes only first 5000 rows (after fraction)
        - Combine both for rapid prototyping

    🧪 Example Usage:
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> results = hyperparameter_tuning_multiclass_classification(
            X=X, y=y,
            model_class=RandomForestClassifier,
            param_grid={
                "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 300),
                "max_depth": lambda t: t.suggest_int("max_depth", 3, 15),
                "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10),
            },
            scoring='f1_macro',
            fast_mode=False,
            return_model=True
        )

    >>> print(results["best_params"])
    >>> print(results["best_score"])
    >>> best_model = results["best_model"]

    💡 Tips:
    - Use `f1_macro` or `balanced_accuracy` when class distribution is uneven.
    - Set `fast_mode=True` when running many experiments in parallel.
    - Combine with `evaluate_multiclass_classification()` to analyze final performance.
    - Avoid using accuracy alone for imbalanced multiclass tasks — prefer macro metrics.

    See Also
    --------
    evaluate_multiclass_classification : Full evaluation utility for multiclass models.
    GridSearchCV : Slower, less flexible alternative.
    Optuna Docs : https://optuna.org
    """


    if any(p is None for p in [X, y, model_class, param_grid]):
        console.print("[bold red]ERROR:[/bold red] Missing required arguments: 'X', 'y', 'model_class', and 'param_grid'.")
        doc = inspect.getdoc(hyperparameter_tuning_multiclass_classification)
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
        raise ValueError("'scoring' must be a string or a callable.")

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
            X_test = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
            y_test = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]
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



