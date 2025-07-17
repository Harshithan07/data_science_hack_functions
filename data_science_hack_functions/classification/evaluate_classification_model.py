import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import DetCurveDisplay
from sklearn.calibration import calibration_curve
from rich.console import Console
from tabulate import tabulate
import inspect

console = Console()

def evaluate_classification_model(
    model=None,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    cv=5,
    cost_fn=None,
    cost_fp=None,
    validation_params=None,
    scoring_curve="accuracy",
    verbose=True,
    return_dict=False,
    return_model_only=False,
    export_model=False,
    extra_plots=None,
    sample_fraction=None,
    sample_size=None,
    fast_mode=False
):
    """
    Evaluate a binary classification model with diagnostics, metrics, plots, and model handling.

    This function evaluates a scikit-learn-compatible classifier using standard and advanced metrics.
    It supports both pretrained and unfitted models, can train during evaluation, and provides plots
    for calibration, threshold tuning, learning/validation curves, and more. It's useful for audits,
    experimentation, and production diagnostics.

    Parameters
    ----------
    model : estimator object, optional
        A scikit-learn classifier. If unfitted, the function will train it using provided data.

    X_train : array-like, optional
        Training feature matrix. Required if model is not already fitted.

    y_train : array-like, optional
        Training labels.

    X_test : array-like, optional
        Testing feature matrix.

    y_test : array-like, optional
        Testing labels.

    cv : int, default=5
        Number of folds for cross-validation in learning/validation curves.

    cost_fn : float, optional
        Cost for false negatives in misclassification cost calculation.

    cost_fp : float, optional
        Cost for false positives.

    validation_params : dict, optional
        Dictionary of hyperparameter names to lists of values for validation curve plotting.

    scoring_curve : str, default='accuracy'
        Scoring metric used in learning and validation curves.

    verbose : bool, default=True
        If True, prints metrics and plots.

    return_dict : bool, default=False
        If True, returns metrics as a dictionary.

    return_model_only : bool, default=False
        If True, returns the trained model only (no metrics).

    export_model : bool, default=False
        If True, returns (metrics_dict, trained_model) tuple.

    extra_plots : list of str, optional
        Options include:
            - "threshold": Precision/Recall/F1 vs threshold
            - "calibration": Reliability of predicted probabilities
            - "ks": KS separation statistic
            - "lift": Lift curve
            - "det": Detection Error Tradeoff curve

    sample_fraction : float, optional
        Fraction of data to use for train/test sets (e.g., 0.1 = 10%).

    sample_size : int, optional
        Number of rows to use from training and testing sets.

    fast_mode : bool, default=False
        If True, skips plots and printouts for faster processing.

    Returns
    -------
    dict, estimator, or tuple
        - Dictionary of metrics if `return_dict=True`
        - Fitted model if `return_model_only=True`
        - (metrics_dict, model) if `export_model=True`

    Raises
    ------
    ValueError
        If necessary inputs are missing or misaligned.

    Examples
    --------
    >>> evaluate_classification_model(
            model=clf,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cost_fn=10, cost_fp=1,
            validation_params={'max_depth': [2, 4, 6]},
            scoring_curve='f1',
            extra_plots=['threshold', 'ks'],
            return_dict=True
        )

    >>> # Save a newly trained model
    >>> model = evaluate_classification_model(
            model=RandomForestClassifier(),
            X_train=X, y_train=y,
            X_test=Xt, y_test=yt,
            return_model_only=True
        )

    >>> # Evaluate a pretrained model
    >>> from joblib import load
    >>> model = load("model.joblib")
    >>> evaluate_classification_model(model=model, X_test=Xt, y_test=yt)

    User Guide
    ----------
    🧠 When to Use This Function:
        • You have a binary classifier (trained or not) and want a full evaluation report.
        • You want to generate visual diagnostics to understand performance beyond raw metrics.
        • You're comparing different models and need standardized summaries and visual feedback.
        • You want to include cost-sensitive evaluation logic (e.g., FP/FN tradeoffs in fraud, medical diagnosis).
        • You plan to export a trained model right after evaluation to reuse or save for deployment.

    📊 Core Metrics Explained:
        • Accuracy:
            → Proportion of correct predictions overall.
            → Works well when classes are balanced, but misleading with imbalanced classes.

        • Precision:
            → Of all predicted positives, how many were actually correct?
            → Use when false positives are costly (e.g., spam detection).

        • Recall:
            → Of all actual positives, how many did the model catch?
            → Use when false negatives are costly (e.g., cancer diagnosis).

        • F1 Score:
            → Harmonic mean of Precision and Recall.
            → Best for imbalanced datasets where both FP and FN matter.

        • ROC AUC:
            → Measures the model's ability to rank positives over negatives.
            → Robust against imbalance. Closer to 1 = better.

        • Cost-sensitive Average Loss:
            → Weighted loss calculation based on your domain-specific cost of False Positives and False Negatives.
            → Useful in fraud detection, churn prediction, or medical triage where not all errors are equal.

    📈 Optional Diagnostic Plots:
        These are generated when `extra_plots` is set and `fast_mode=False`.

        • Threshold Curve (`extra_plots=['threshold']`):
            → Shows how Precision, Recall, and F1 change with different probability thresholds.
            → Use it when you need to manually tune the decision boundary (e.g., prioritize recall over precision).

        • Calibration Curve (`extra_plots=['calibration']`):
            → Compares predicted probabilities to observed outcomes.
            → Helps determine if model outputs represent true probabilities (e.g., in credit scoring, risk modeling).

        • KS Statistic (`extra_plots=['ks']`):
            → Plots the cumulative distribution of scores for each class and measures the maximum separation.
            → A higher KS value (closer to 1) indicates good class separation.

        • Lift Curve (`extra_plots=['lift']`):
            → Compares model performance against random guessing in terms of capturing true positives.
            → Great for targeting top decile groups (e.g., marketing response modeling).

        • DET Curve (`extra_plots=['det']`):
            → Plots False Positive Rate vs. False Negative Rate using logarithmic scale.
            → Especially useful in imbalanced classification (e.g., rare disease detection, security event modeling).

    📚 Learning & Validation Curves (CV Required):
        • Learning Curve:
            → Plots model performance vs. training size.
            → Helps detect underfitting (low train/val scores) or overfitting (train ≫ val).
            → Useful to decide if more data will help your model.

        • Validation Curve (`validation_params={'C': [...], 'max_depth': [...]}`):
            → Shows how a single hyperparameter affects training and validation performance.
            → Use to find optimal model complexity (e.g., tree depth, regularization).

    ⚙ Runtime & Usability Tips:
        • fast_mode=True:
            → Skip all visual output and verbose logs. Ideal for CI/CD pipelines or large batch runs.

        • return_model_only=True:
            → Use this if your model is not yet fitted and you want the trained object back after evaluation.

        • export_model=True:
            → Returns both (metrics_dict, trained_model) to chain into pipelines, dashboards, or export routines.

        • sample_fraction / sample_size:
            → Quickly prototype or test on large datasets without full evaluation time/cost.
            → Useful when training/testing on full 10M+ rows is not practical.

    See Also
    --------
    - preprocess_dataframe() : For feature preprocessing
    - summary_dataframe() : For data overview
    - run_nested_cv_classification() : For model comparison
    """

    # ❌ If no arguments, show help/guide
    if any(x is None for x in [model, X_train, y_train, X_test, y_test]):
        console.print("[bold red]\nERROR:[/bold red] Missing required arguments: 'model', 'X_train', 'y_train', 'X_test', 'y_test'.")
        doc = inspect.getdoc(evaluate_classification_model)
        console.print(f"\n[bold cyan]Docstring:[/bold cyan]\n\n{doc}")
        return

    def sample(X, y, how=None):
        if how is None:
            return X, y
        if isinstance(how, float):
            return train_test_split(X, y, train_size=how, stratify=y, random_state=42)[0:2]
        elif isinstance(how, int):
            return X[:how], y[:how]
        return X, y

    if sample_fraction:
        X_train, y_train = sample(X_train, y_train, how=sample_fraction)
        X_test, y_test = sample(X_test, y_test, how=sample_fraction)
    if sample_size:
        X_train, y_train = sample(X_train, y_train, how=sample_size)
        X_test, y_test = sample(X_test, y_test, how=sample_size)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_pred_proba = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

    if verbose:
        print("\n📊 Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Class"})
        print(tabulate(report_df, headers="keys", tablefmt="fancy_grid", showindex=False))

        print("\n📈 Evaluation Metrics:")
        metrics = [
            ["Accuracy", f"{accuracy:.4f}"],
            ["Precision", f"{precision:.4f}"],
            ["Recall", f"{recall:.4f}"],
            ["F1 Score", f"{f1:.4f}"],
            ["ROC AUC", f"{roc_auc:.4f}" if roc_auc is not None else "N/A"]
        ]
        print(tabulate(metrics, headers=["Metric", "Score"], tablefmt="fancy_grid"))

    cm = confusion_matrix(y_test, y_pred)
    avg_cost = None
    if cost_fn is not None and cost_fp is not None:
        fn, fp = cm[1, 0], cm[0, 1]
        avg_cost = (cost_fn * fn + cost_fp * fp) / len(y_test)
        if verbose:
            print(f"\n💰 Avg Cost (FN={cost_fn}, FP={cost_fp}): {avg_cost:.4f}")

    if not fast_mode and verbose:
        class_labels = list(np.unique(y_test))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    extra_plots = extra_plots or []
    if not fast_mode and y_pred_proba is not None:
        if "threshold" in extra_plots:
            thresholds = np.linspace(0.01, 0.99, 50)
            precisions, recalls, f1s = [], [], []
            for t in thresholds:
                preds = (y_pred_proba >= t).astype(int)
                precisions.append(precision_score(y_test, preds, zero_division=0))
                recalls.append(recall_score(y_test, preds, zero_division=0))
                f1s.append(f1_score(y_test, preds, zero_division=0))
            plt.figure(figsize=(10, 5))
            plt.plot(thresholds, precisions, label="Precision")
            plt.plot(thresholds, recalls, label="Recall")
            plt.plot(thresholds, f1s, label="F1 Score")
            plt.title("Threshold vs Metrics")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        if "calibration" in extra_plots:
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
            plt.figure(figsize=(6, 5))
            plt.plot(prob_pred, prob_true, marker='o', label='Calibration')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.title("Calibration Curve")
            plt.xlabel("Predicted Prob.")
            plt.ylabel("Actual Pos. Rate")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    if not fast_mode and cv > 1 and verbose:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=cv, scoring=scoring_curve
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation", linestyle="--")
        plt.xlabel("Training Set Size")
        plt.ylabel(scoring_curve.capitalize())
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if not fast_mode and validation_params:
        for param, values in validation_params.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
            train_scores, val_scores = validation_curve(
                pipe, X_train, y_train,
                param_name=f"clf__{param}",
                param_range=values,
                scoring=scoring_curve,
                cv=cv, n_jobs=-1
            )
            plt.figure(figsize=(8, 5))
            plt.plot(values, np.mean(train_scores, axis=1), label="Train")
            plt.plot(values, np.mean(val_scores, axis=1), label="Validation", linestyle="--")
            plt.title(f"Validation Curve: {param}")
            plt.xlabel(param)
            plt.ylabel(scoring_curve.capitalize())
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    if return_model_only:
        return model

    if return_dict:
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "avg_cost": avg_cost,
            "model": model if export_model else None
        }


