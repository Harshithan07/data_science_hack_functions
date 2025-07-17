import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from rich.console import Console
from tabulate import tabulate
from typing import Optional, Union

console = Console()

def evaluate_multiclass_classification(
    model=None,
    X_train=None, y_train=None,
    X_test=None, y_test=None,
    cv: int = 5,
    validation_params: Optional[dict] = None,
    scoring_curve: str = "accuracy",
    verbose: bool = True,
    return_dict: bool = False,
    return_model_only: bool = False,
    export_model: bool = False,
    extra_plots: Optional[list] = None,
    sample_fraction: Optional[float] = None,
    sample_size: Optional[int] = None,
    fast_mode: bool = False
):
    """
    Evaluate a multiclass classification model with diagnostics, metrics, plots, and model handling.

    This function evaluates a scikit-learn-compatible classifier across multiple classes using both
    standard and advanced metrics. It supports unfitted or pretrained models, can train during evaluation,
    and provides visual diagnostics such as learning curves, calibration plots, and confusion matrices.
    It is ideal for model auditing, experiment evaluation, and production diagnostics in multiclass settings.

    Parameters
    ----------
    model : estimator object, optional
        A scikit-learn-compatible classifier. If not fitted, it will be trained using X_train and y_train.

    X_train : array-like, optional
        Training feature matrix. Required if the model is not already fitted.

    y_train : array-like, optional
        Training labels.

    X_test : array-like, optional
        Testing feature matrix.

    y_test : array-like, optional
        Testing labels.

    cv : int, default=5
        Number of cross-validation folds used for learning and validation curve plotting.

    validation_params : dict, optional
        Dictionary of hyperparameter names mapped to value lists (e.g., {'max_depth': [2, 4, 6]}).
        Used for plotting validation curves.

    scoring_curve : str, default='accuracy'
        Scoring metric used during learning and validation curve generation.

    verbose : bool, default=True
        If True, prints key metrics, classification reports, and shows plots.

    return_dict : bool, default=False
        If True, returns a dictionary containing evaluation metrics and confusion matrix.

    return_model_only : bool, default=False
        If True, returns only the trained model. Ignores all metric outputs.

    export_model : bool, default=False
        If True and return_dict=True, includes the trained model in the output dictionary.

    extra_plots : list of str, optional
        Additional diagnostic plots to generate. Options:
            - "calibration": Reliability plots per class (One-vs-Rest format)

    sample_fraction : float, optional
        Fraction of the test set to use during evaluation (e.g., 0.1 = 10%).

    sample_size : int, optional
        Absolute number of rows to use from the test set.

    fast_mode : bool, default=False
        If True, disables all visualizations and print outputs for fast evaluation.

    Returns
    -------
    dict, estimator, or tuple
        - Dictionary of metrics if `return_dict=True`
        - Fitted model if `return_model_only=True`
        - (metrics_dict, model) if `return_dict=True` and `export_model=True`

    Raises
    ------
    ValueError
        If required inputs are missing or invalid combinations of sample parameters are specified.

    Examples
    --------
    >>> evaluate_multiclass_classification(
            model=clf,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            validation_params={'max_depth': [2, 4, 6]},
            scoring_curve='f1_macro',
            extra_plots=['calibration'],
            return_dict=True
        )

    >>> # Return only the trained model
    >>> clf = evaluate_multiclass_classification(
            model=RandomForestClassifier(),
            X_train=X, y_train=y,
            X_test=Xt, y_test=yt,
            return_model_only=True
        )

    >>> # Use pretrained model for diagnostics
    >>> from joblib import load
    >>> model = load("clf.joblib")
    >>> evaluate_multiclass_classification(model=model, X_test=Xt, y_test=yt)

    User Guide
    ----------
    🧠 When to Use This Function:
        • You have a multiclass classifier (trained or not) and want a complete diagnostic report.
        • You need reliable evaluation across multiple classes using macro-averaged metrics.
        • You want visual feedback to compare models, understand performance, or debug issues.
        • You want exportable results for dashboards, model cards, or batch audit pipelines.

    📊 Core Metrics Explained:
        • Accuracy:
            → Overall proportion of correct predictions.
            → Suitable for balanced class distributions.

        • Macro Precision:
            → Average precision across all classes.
            → Treats all classes equally, regardless of support.

        • Macro Recall:
            → Average recall across all classes.
            → Useful when missing any class matters equally.

        • Macro F1 Score:
            → Harmonic mean of macro precision and recall.
            → Good for imbalanced multiclass classification.

        • ROC AUC (OVR):
            → One-vs-Rest AUC computed across all classes.
            → Reflects model's ability to separate each class from the rest.

    📈 Optional Diagnostic Plots:
        • Calibration Curve (`extra_plots=['calibration']`)
            → For each class, compares predicted probabilities vs. true outcomes in One-vs-Rest fashion.
            → Helps assess if predicted probabilities are well-calibrated.

        • Learning Curve:
            → Plots training/validation performance as a function of training set size.
            → Helps detect underfitting, overfitting, and whether more data may help.

        • Validation Curve (`validation_params={'max_depth': [...]}`)
            → Shows how model performance changes with different values of one hyperparameter.
            → Useful for tuning complexity (e.g., tree depth, number of estimators).

    ⚠ Binary-Only Features Removed:
        • Threshold tuning curves, KS statistic, DET curve, lift curve, and cost-sensitive loss are not supported
          in multiclass context due to their reliance on binary decision boundaries.

    ⚙ Runtime & Usability Tips:
        • fast_mode=True:
            → Skips all visual output and logs. Great for large-scale evaluations or scripts.

        • return_model_only=True:
            → Quickly fit and retrieve a model from the evaluation process.

        • export_model=True:
            → Use in pipelines to return both evaluation metrics and model object in one step.

        • sample_fraction / sample_size:
            → Useful when working with large datasets. Enables quick prototyping or evaluation subsets.

    See Also
    --------
    - preprocess_dataframe() : For feature cleaning and encoding
    - run_nested_cv_classification() : For nested cross-validation and model comparison
    - summary_dataframe(), summary_column() : For EDA and column profiling
    - evaluate_classification_model() : Binary classification version
    """

    if any(x is None for x in [model, X_train, y_train, X_test, y_test]):
        raise ValueError("Missing one or more required inputs: model, X_train, y_train, X_test, y_test.")

    if sample_fraction and sample_size:
        raise ValueError("Specify only one of sample_fraction or sample_size.")

    # Sampling
    if sample_fraction:
        if not fast_mode and verbose:
            print(f"[Sampling] Using {sample_fraction*100:.1f}% of test data.")
        X_test = X_test.sample(frac=sample_fraction, random_state=42)
        y_test = y_test.loc[X_test.index]
    elif sample_size:
        if not fast_mode and verbose:
            print(f"[Sampling] Using {sample_size} rows from test data.")
        X_test = X_test.sample(n=sample_size, random_state=42)
        y_test = y_test.loc[X_test.index]

    extra_plots = extra_plots or []
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_pred_proba = model.decision_function(X_test)

    # Metrics (macro-average)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    roc_auc = None
    if y_pred_proba is not None and len(np.unique(y_test)) > 2:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        except:
            roc_auc = None

    if verbose and not fast_mode:
        print("\n📊 Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Class"})
        print(tabulate(report_df, headers="keys", tablefmt="fancy_grid", showindex=False))

        print("\n📈 Evaluation Metrics:")
        metrics = [
            ["Accuracy", f"{accuracy:.4f}"],
            ["Macro Precision", f"{precision:.4f}"],
            ["Macro Recall", f"{recall:.4f}"],
            ["Macro F1 Score", f"{f1:.4f}"],
            ["ROC AUC (OVR)", f"{roc_auc:.4f}" if roc_auc is not None else "N/A"]
        ]
        print(tabulate(metrics, headers=["Metric", "Score"], tablefmt="fancy_grid"))

    # Confusion Matrix
    if not fast_mode and verbose:
        cm = confusion_matrix(y_test, y_pred)
        labels = np.unique(y_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    # Calibration Plot
    if "calibration" in extra_plots and not fast_mode and y_pred_proba is not None:
        for i, class_label in enumerate(np.unique(y_test)):
            prob_true, prob_pred = calibration_curve((y_test == class_label).astype(int), y_pred_proba[:, i], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label=f"Class {class_label}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title("Calibration Curves (One-vs-Rest)")
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Fraction")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Learning Curve
    if not fast_mode and cv > 1 and verbose:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=cv, scoring=scoring_curve
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train", marker='o')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation", marker='s')
        plt.xlabel("Training Size")
        plt.ylabel(scoring_curve)
        plt.title("Learning Curve")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # Validation Curve
    if validation_params and not fast_mode:
        for param, values in validation_params.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
            train_scores, val_scores = validation_curve(
                pipe, X_train, y_train,
                param_name=f"clf__{param}",
                param_range=values,
                scoring=scoring_curve,
                cv=cv,
                n_jobs=-1
            )
            plt.figure(figsize=(8, 5))
            plt.plot(values, np.mean(train_scores, axis=1), label="Train", marker='o')
            plt.plot(values, np.mean(val_scores, axis=1), label="Validation", marker='s')
            plt.title(f"Validation Curve: {param}")
            plt.xlabel(param)
            plt.ylabel(scoring_curve)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # Return section
    if return_model_only:
        return model

    if return_dict:
        return {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'roc_auc_macro': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': model if export_model else None
        }






