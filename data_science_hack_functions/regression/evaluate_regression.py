import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from rich.console import Console
import inspect

console = Console()

def evaluate_regression_model(
    model=None,
    X_train=None, y_train=None,
    X_test=None, y_test=None,
    cv=5,
    validation_params=None,
    scoring_curve='r2',
    verbose=True,
    return_dict=False,
    return_model_only=False,
    export_model=False,
    extra_plots=None,
    custom_metrics=None,
    log_transform=False,
    fast_mode=False,
    sample_fraction=None,
    sample_size=None
):
    
    """
    Evaluate a regression model with metrics, diagnostics, visualization, and export handling.

    This function evaluates a scikit-learn-compatible regressor using a variety of performance metrics.
    It supports trained and untrained models, optionally fits them, and generates plots to analyze
    model performance, including learning curves, residuals, and error distributions.

    Parameters
    ----------
    model : estimator object, optional
        A scikit-learn-compatible regressor. If unfitted, the function will train it using provided data.

    X_train : array-like, optional
        Training feature matrix. Required if model is not already fitted.

    y_train : array-like, optional
        Training target vector.

    X_test : array-like, optional
        Testing feature matrix.

    y_test : array-like, optional
        Testing target vector.

    cv : int, default=5
        Number of cross-validation folds used for learning and validation curves.

    validation_params : dict, optional
        Dictionary of hyperparameter names to lists of values for validation curve plotting.

    scoring_curve : str, default='r2'
        Scoring metric used for the learning and validation curves.

    verbose : bool, default=True
        If True, prints all metrics and visualizations.

    return_dict : bool, default=False
        If True, returns computed metrics in a dictionary.

    return_model_only : bool, default=False
        If True, returns only the trained model, suppressing all metrics and plots.

    export_model : bool, default=False
        If True, returns a tuple of (metrics_dict, trained_model).

    extra_plots : list of str, optional
        Diagnostic visualizations to generate. Supported values:
            - 'pred_vs_actual': Predicted vs Actual scatterplot
            - 'residuals': Residuals vs Fitted plot
            - 'error_dist': Histogram of prediction errors
            - 'qq': Q-Q plot of residuals
            - 'feature_importance': Feature importance bar plot (if supported)
            - 'learning': Learning curve plot
            - 'validation': Validation curve(s) for specified parameters

    custom_metrics : dict, optional
        Dictionary of user-defined metric functions with format: {name: callable(y_true, y_pred)}.

    log_transform : bool, default=False
        If True, log1p-transform both `y_test` and predictions before metric computation.

    fast_mode : bool, default=False
        If True, disables visual output and reduces verbosity for speed. Ideal for batch evaluations.

    sample_fraction : float, optional
        If set, uses a random subset of the test data by fraction (e.g., 0.1 = 10%).

    sample_size : int, optional
        If set, uses only the specified number of rows from the test data.

    Returns
    -------
    dict, estimator, or tuple
        - Dictionary of metrics if `return_dict=True`
        - Trained model if `return_model_only=True`
        - (metrics_dict, model) if `export_model=True`

    Raises
    ------
    ValueError
        If both `sample_fraction` and `sample_size` are specified simultaneously.

    Examples
    --------
    >>> evaluate_regression_model(
            model=LinearRegression(),
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            extra_plots=['pred_vs_actual', 'residuals', 'learning'],
            return_dict=True
        )

    >>> # Save and export model
    >>> metrics, trained_model = evaluate_regression_model(
            model=RandomForestRegressor(),
            X_train=X, y_train=y,
            X_test=Xt, y_test=yt,
            export_model=True
        )

    User Guide
    ----------
    🧠 When to Use This Function:
        • You want a comprehensive evaluation of a regression model with minimal boilerplate.
        • You need insight into prediction accuracy, residuals, and learning behavior.
        • You're experimenting with different models and want standardized reports.
        • You want to benchmark models quickly on subsets of large datasets.
        • You wish to export trained models alongside their evaluation scores.

    📊 Core Metrics Explained:
        • R² (R-squared):
            → Proportion of variance in target explained by the model.
            → Values closer to 1 indicate better performance.

        • Adjusted R²:
            → Penalized R² for the number of predictors used. More robust to overfitting.

        • RMSE (Root Mean Squared Error):
            → Measures average magnitude of error. Sensitive to outliers.

        • MAE (Mean Absolute Error):
            → Measures average absolute deviation. Robust and interpretable.

        • MAPE (Mean Absolute Percentage Error):
            → Scales MAE by the magnitude of the true value. Not defined for zero targets.

        • RMSLE (Root Mean Squared Log Error):
            → Useful when dealing with targets across several orders of magnitude. Ignores underestimation penalties.

        • Custom Metrics:
            → Pass any function with signature `func(y_true, y_pred)` for domain-specific scoring.

    📈 Diagnostic Visualizations (enabled via `extra_plots`):
        • Predicted vs Actual:
            → Shows how closely predictions align with ground truth.

        • Residual Plot:
            → Detects heteroskedasticity or model bias across fitted values.

        • Error Distribution:
            → Histogram of residuals, useful for checking skew.

        • Q-Q Plot:
            → Checks if residuals follow normal distribution (key assumption in linear models).

        • Feature Importance:
            → For tree models, displays relative contribution of each feature.

        • Learning Curve:
            → Shows training vs validation performance across increasing sample sizes.

        • Validation Curve:
            → Visualizes sensitivity to specific hyperparameters.

    ⚙ Runtime & Usability Tips:
        • fast_mode=True:
            → Skips all visual output and logging. Best for CI jobs or looped experimentation.

        • return_model_only=True:
            → Returns the fitted model only (no metrics or plots). Handy for pipelines.

        • export_model=True:
            → Returns a tuple of (metrics_dict, trained_model) for downstream use.

        • sample_fraction / sample_size:
            → Great for fast prototyping or large dataset downsampling.

    See Also
    --------
    - sklearn.metrics : Reference for all built-in scoring functions
    - run_nested_cv_regression : For comparing multiple regression models
    - summary_dataframe(), preprocess_dataframe() : For EDA & preprocessing utilities
    """

    if all(arg is None for arg in [model, X_train, y_train, X_test, y_test]):
        console.print(inspect.getdoc(evaluate_regression_model))
        return

    if sample_fraction is not None and sample_size is not None:
        raise ValueError("Specify only one of sample_fraction or sample_size.")

    start_time = time.time()
    extra_plots = extra_plots or []
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    if sample_fraction:
        if not fast_mode and verbose:
            print(f"[Sampling] Using {sample_fraction * 100:.1f}% of the test set.")
        X_test = X_test.sample(frac=sample_fraction, random_state=42)
        y_test = y_test.loc[X_test.index]
    elif sample_size:
        if not fast_mode and verbose:
            print(f"[Sampling] Using {sample_size} rows of the test set.")
        X_test = X_test.sample(n=sample_size, random_state=42)
        y_test = y_test.loc[X_test.index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if log_transform:
        y_test = np.log1p(y_test)
        y_pred = np.log1p(y_pred)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    try:
        if (y_test < 0).any() or (y_pred < 0).any():
            rmsle = np.nan
            if verbose and not fast_mode:
                print("Warning: Negative values detected. Skipping RMSLE computation.")
        else:
            rmsle = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred)))
    except Exception as e:
        rmsle = np.nan
        if verbose and not fast_mode:
            print(f"RMSLE computation failed: {e}")

    n, p = X_test.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    if verbose and not fast_mode:
        print("\nTest Set Regression Metrics:")
        print(f"R²           : {r2:.4f}")
        print(f"Adjusted R²  : {adj_r2:.4f}")
        print(f"RMSE         : {rmse:.4f}")
        print(f"MAE          : {mae:.4f}")
        print(f"MAPE         : {mape:.4f}")
        print(f"RMSLE        : {rmsle:.4f}")

    if custom_metrics and not fast_mode:
        for name, func in custom_metrics.items():
            val = func(y_test, y_pred)
            if verbose:
                print(f"{name:<12}: {val:.4f}")

    if not fast_mode:
        if 'pred_vs_actual' in extra_plots:
            plt.figure(figsize=(6, 6))
            sns.scatterplot(x=y_test, y=y_pred)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Predicted vs Actual")
            plt.tight_layout()
            plt.show()

        if 'residuals' in extra_plots:
            residuals = y_test - y_pred
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=y_pred, y=residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Predicted")
            plt.ylabel("Residual")
            plt.title("Residual Plot")
            plt.tight_layout()
            plt.show()

        if 'error_dist' in extra_plots:
            errors = y_test - y_pred
            plt.figure(figsize=(8, 5))
            sns.histplot(errors, kde=True)
            plt.title("Error Distribution")
            plt.xlabel("Prediction Error")
            plt.tight_layout()
            plt.show()

        if 'qq' in extra_plots:
            residuals = y_test - y_pred
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title("Q-Q Plot of Residuals")
            plt.tight_layout()
            plt.show()

        if hasattr(model, 'feature_importances_') and 'feature_importance' in extra_plots:
            plt.figure(figsize=(8, 5))
            feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
            feat_imp.sort_values().plot(kind='barh')
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.show()

        if 'learning' in extra_plots:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=cv, scoring=scoring_curve
            )
            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train", marker='o')
            plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation", marker='s')
            plt.xlabel("Training Size")
            plt.ylabel(scoring_curve)
            plt.title("Learning Curve")
            plt.legend()
            plt.tight_layout()
            plt.show()

        if validation_params and 'validation' in extra_plots:
            for param_name, param_range in validation_params.items():
                print(f"\nValidation Curve for: {param_name}")
                pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
                train_scores, val_scores = validation_curve(
                    pipe, X_train, y_train,
                    param_name=f'clf__{param_name}',
                    param_range=param_range,
                    scoring=scoring_curve,
                    cv=cv, n_jobs=-1
                )
                plt.figure(figsize=(8, 5))
                plt.plot(param_range, np.mean(train_scores, axis=1), label="Train", marker='o')
                plt.plot(param_range, np.mean(val_scores, axis=1), label="Validation", marker='s')
                plt.xlabel(param_name)
                plt.ylabel(scoring_curve)
                plt.title(f"Validation Curve: {param_name}")
                plt.legend()
                plt.tight_layout()
                plt.show()

    if return_model_only:
        return model

    if export_model:
        return {
            'r2': r2,
            'adj_r2': adj_r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'rmsle': rmsle,
            'custom_metrics': {
                name: func(y_test, y_pred) for name, func in (custom_metrics or {}).items()
            },
            'runtime_secs': time.time() - start_time
        }, model

    if return_dict:
        return {
            'r2': r2,
            'adj_r2': adj_r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'rmsle': rmsle,
            'custom_metrics': {
                name: func(y_test, y_pred) for name, func in (custom_metrics or {}).items()
            },
            'runtime_secs': time.time() - start_time
        }
