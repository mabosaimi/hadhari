import logging
from collections.abc import Sequence
from pathlib import Path

import joblib
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing.preprocessor import clean_text

SAVE_DIR = "artifacts"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train(
    X: pl.Series | Sequence[str],  # noqa: N803
    y: pl.Series | Sequence[str],
    *,
    model: BaseEstimator | None = None,
    vectorizer: TransformerMixin | None = None,
    test_size: float = 0.2,
    random_state: int | None = None,
    save_model: bool = True,
    verbose: bool = True,
) -> Pipeline:
    """Train a text classification pipeline using a vectorizer and classifier.

    Parameters
    ----------
    X : pl.Series | list[str]
        Input text
    y : pl.Series | list[str]
        Target labels
    model : BaseEstimator | None
        scikit-learn compatible classifier (default: LogisticRegression)
    vectorizer : TransformerMixin | None
        Text vectorizer (default: TfidfVectorizer)
    test_size : float
        Proportion for test split
    random_state : int | None
        Seed for reproducibility
    save_model : bool
        Whether to save the model (default: True)
    verbose : bool
        Whether to print metrics

    Returns
    -------
    Pipeline
        Trained scikit-learn pipeline

    """
    X_clean = X.map_elements(clean_text)  # noqa: N806
    y_list = y.to_list() if isinstance(y, pl.Series) else list(y)

    if vectorizer is None:
        TOKEN_PATTERN = r"(?u)\b\w+\b"  # noqa: N806, S105
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), token_pattern=TOKEN_PATTERN)

    if model is None:
        model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X_clean,
        y_list,
        test_size=test_size,
        stratify=y_list,
        random_state=random_state,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    if verbose:
        logger.info("Accuracy: %s", accuracy)
        logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

    if save_model:
        accuracy_percentage = int(accuracy * 100)
        model_name = model.__class__.__name__.lower()
        vectorizer_name = vectorizer.__class__.__name__.lower()
        dataset_size = len(X_train)
        save_path = (
            Path(__file__).resolve().parent
            / SAVE_DIR
            / model_name
            / f"{vectorizer_name}_{dataset_size}_{accuracy_percentage}.joblib"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)

        if verbose:
            logger.info("Model saved to %s", save_path)

    return pipeline
