import numpy as np
import pandas as pd
import optuna
import time
import inspect
from typing import Any, Callable, Dict, Union, Optional
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from tabulate import tabulate
from rich.console import Console

console = Console()

def hyperparameter_tuning_regression(
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    model_class: Optional[Callable[..., Any]] = None,
    param_grid: Optional[Dict[str, Callable[[optuna.Trial], Any]]] = None,
    scoring: Union[str, Callable] = 'r2',
    n_trials: int = 50,
    cv_folds: int = 5,
    direction: str = 'maximize',
    verbose: bool = True,
    return_model: bool = True,
    random_state: int = 42,
    use_fraction: Optional[float] = None,
    use_n_samples: Optional[int] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:

    """
    🔧 Hyperparameter Tuning for Regression (via Optuna)

    Optimize hyperparameters of any regression model using Optuna’s efficient 
    search. This function supports K-fold cross-validation, flexible scoring,
    sampling controls, reproducibility, and optionally returns the best fitted model.

    Ideal for leaderboard-style tuning, pipeline integration, and experimentation
    across research or production ML environments.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix with shape (n_samples, n_features).

    y : pd.Series or np.ndarray
        Target vector of regression outputs.

    model_class : callable
        A scikit-learn-style regressor class (e.g., `RandomForestRegressor`, `SVR`).
        Not an instance – must be the class itself.

    param_grid : dict
        Dictionary mapping hyperparameter names to Optuna sampling functions.
        Example:
            {
                "alpha": lambda t: t.suggest_float("alpha", 0.001, 10, log=True),
                "fit_intercept": lambda t: t.suggest_categorical("fit_intercept", [True, False])
            }

    scoring : str or callable, default='r2'
        Scoring metric to optimize. Supports:
        - Any sklearn string (e.g., 'neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error')
        - A custom scoring function with signature: scorer(estimator, X_val, y_val)

    n_trials : int, default=50
        Number of Optuna optimization trials.

    cv_folds : int, default=5
        Number of folds in K-Fold cross-validation for each trial.

    direction : {'maximize', 'minimize'}, default='maximize'
        Optimization goal — e.g., maximize R², or minimize RMSE.

    verbose : bool, default=True
        If True, logs model name, metric, best results, and parameter table.

    return_model : bool, default=True
        If True, trains and returns the best model on the entire dataset.

    random_state : int, default=42
        Seed for reproducible splits and optimization behavior.

    use_fraction : float or None, optional
        If set, randomly samples a fraction of the data (e.g., 0.2 = 20%).

    use_n_samples : int or None, optional
        If set, samples up to a fixed number of rows (e.g., 10000). Applied after `use_fraction`.

    fast_mode : bool, default=False
        If True, reduces number of trials to 10, disables print logs, and speeds up execution.

    Returns
    -------
    dict
        Contains:
        - 'best_score' : float
        - 'best_params' : dict
        - 'study' : optuna.Study
        - 'best_model' : fitted model (if return_model=True)

    Raises
    ------
    ValueError
        If scoring is neither a valid string nor a callable function.

    User Guide
    ----------
    🧠 When to Use:
    - You're optimizing regression models across different parameter sets.
    - You want to avoid grid search overhead with smarter sampling.
    - You’re working with large datasets or want reproducible trials.
    - You prefer flexible, callable-based metric optimization.

    📌 Key Components:

    • param_grid:
        Define hyperparameter ranges using Optuna’s trial suggestions — more expressive than traditional grids.
        Examples: `suggest_float`, `suggest_int`, `suggest_categorical`, `suggest_loguniform`.

    • scoring:
        Use built-in sklearn scorers like 'r2', 'neg_root_mean_squared_error', or define your own function.

    • Subsampling:
        Use `use_fraction` and/or `use_n_samples` to downsample large datasets while tuning.

    • fast_mode:
        Cuts down `n_trials` to 10 and disables verbose printing. Great for initial testing.

    🧪 Example Usage:
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> results = hyperparameter_tuning_regression(
            X=X, y=y,
            model_class=GradientBoostingRegressor,
            param_grid={
                "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 300),
                "max_depth": lambda t: t.suggest_int("max_depth", 3, 10),
                "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            },
            scoring='neg_root_mean_squared_error',
            fast_mode=True,
            return_model=True
        )

    >>> print(results["best_params"])
    >>> print(results["best_score"])
    >>> model = results["best_model"]

    💡 Tips:
    - Use log-scale for continuous values (e.g., learning rates, regularization strength).
    - Prefer `neg_root_mean_squared_error` or `neg_mean_absolute_error` for cost-aware regression.
    - Use `r2` when model interpretability or variance explanation is the goal.

    See Also
    --------
    evaluate_regression_model : For full regression metrics and plots after training.
    Optuna : https://optuna.org
    GridSearchCV : Classic alternative using exhaustive search.
    """

    if any(param is None for param in [X, y, model_class, param_grid]):
        console.print("[bold red]\nERROR:[/bold red] Missing required arguments: 'X', 'y', 'model_class', and 'param_grid'.")
        doc = inspect.getdoc(hyperparameter_tuning_regression)
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
        console.print(f"[bold]Fast Mode    :[/bold] {'✅' if fast_mode else '❌'}")
        console.rule()

    def objective(trial):
        params = {k: v(trial) for k, v in param_grid.items()}
        model = model_class(**params)
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, test_idx in cv.split(X):
            X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            X_test = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
            y_test = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]
            try:
                model.fit(X_train, y_train)
                score = scorer(model, X_test, y_test)
                scores.append(score)
            except Exception:
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
