import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from hadhari.preprocessing.preprocessor import preprocess_texts

SAVE_DIR = "artifacts"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_cross_validation(
    pipeline: Pipeline,
    X: Sequence[str],
    y: Sequence[Any],
    *,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Perform Stratified K-Fold cross-validation and return aggregated metrics."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }
    scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    return {
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "precision_mean": float(np.mean(scores["test_precision"])),
        "precision_std": float(np.std(scores["test_precision"])),
        "recall_mean": float(np.mean(scores["test_recall"])),
        "recall_std": float(np.std(scores["test_recall"])),
        "f1_mean": float(np.mean(scores["test_f1"])),
        "f1_std": float(np.std(scores["test_f1"])),
    }


def train(
    X: pl.Series | Sequence[str],
    y: pl.Series | Sequence[Any],
    *,
    model: BaseEstimator | None = None,
    vectorizer: Any | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    save_model: bool = True,
    verbose: bool = True,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train a text classification pipeline with evaluation and optional artifact saving.

    Returns
    -------
    tuple[Pipeline, dict[str, Any]]
        The trained scikit-learn pipeline and structured evaluation metrics.

    """
    X_list = X.to_list() if isinstance(X, pl.Series) else list(X)
    y_list = y.to_list() if isinstance(y, pl.Series) else list(y)

    if vectorizer is None:
        TOKEN_PATTERN = r"(?u)\b\w+\b"  # noqa: S105
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), token_pattern=TOKEN_PATTERN)

    if model is None:
        model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline([
        ("preprocessor", FunctionTransformer(preprocess_texts)),
        ("vectorizer", vectorizer),
        ("classifier", model),
    ])

    # Stratified K-Fold Cross Validation
    cv_results = evaluate_cross_validation(
        pipeline, X_list, y_list, cv_folds=cv_folds, random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_list,
        y_list,
        test_size=test_size,
        stratify=y_list,
        random_state=random_state,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "dataset_size": len(X_list),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": random_state,
        "test_accuracy": float(accuracy),
        "cv_folds": cv_folds,
        "cv_metrics": cv_results,
        "classification_report": report,
    }

    if verbose:
        logger.info("Test Accuracy: %.4f", accuracy)
        logger.info("5-Fold CV Accuracy: %.4f ± %.4f", cv_results["accuracy_mean"], cv_results["accuracy_std"])
        logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

    if save_model:
        accuracy_percentage = int(accuracy * 100)
        model_name = model.__class__.__name__.lower()
        vectorizer_name = vectorizer.__class__.__name__.lower()
        dataset_size = len(X_train)
        base_path = (
            Path(__file__).resolve().parent
            / SAVE_DIR
            / model_name
            / f"{vectorizer_name}_{dataset_size}_{accuracy_percentage}"
        )
        base_path.parent.mkdir(parents=True, exist_ok=True)
        model_path = base_path.with_suffix(".joblib")
        metrics_path = base_path.with_suffix(".json")

        joblib.dump(pipeline, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if verbose:
            logger.info("Model saved to %s", model_path)
            logger.info("Metrics saved to %s", metrics_path)

    return pipeline, metrics
