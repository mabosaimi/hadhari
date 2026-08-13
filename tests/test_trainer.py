from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import hadhari.models.trainer
from hadhari.models.trainer import evaluate_cross_validation, train
from hadhari.preprocessing.preprocessor import preprocess_texts


def test_train_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Set SAVE_DIR to tmp_path for testing
    monkeypatch.setattr(hadhari.models.trainer, "SAVE_DIR", str(tmp_path))

    X = ["عرض خاص اشتري الان", "مرحبا كيف حالك اليوم", "احصل على خصم 50 بالمئة", "السلام عليكم ورحمة الله"] * 5
    y = [1, 0, 1, 0] * 5

    pipeline, metrics = train(
        X,
        y,
        test_size=0.2,
        random_state=42,
        cv_folds=2,
        save_model=True,
        verbose=False,
    )

    assert pipeline is not None
    assert "test_accuracy" in metrics
    assert "cv_metrics" in metrics
    assert metrics["cv_folds"] == 2

    # Verify prediction on new input
    preds = pipeline.predict(["اشتري الان وحصل على خصم"])
    assert len(preds) == 1


def test_train_save_model_false() -> None:
    X = ["رسالة 1", "رسالة 2", "رسالة 3", "رسالة 4"] * 3
    y = [0, 1, 0, 1] * 3

    pipeline, metrics = train(
        X,
        y,
        test_size=0.2,
        random_state=42,
        cv_folds=2,
        save_model=False,
        verbose=False,
    )

    assert pipeline is not None
    assert metrics["dataset_size"] == 12


def test_evaluate_cross_validation() -> None:
    pipeline = Pipeline([
        ("preprocessor", FunctionTransformer(preprocess_texts)),
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression()),
    ])

    X = ["رسالة سبام جديدة", "مرحبا صديقي", "عرض لفترة محدودة", "كيف الحال"] * 4
    y = [1, 0, 1, 0] * 4

    cv_res = evaluate_cross_validation(pipeline, X, y, cv_folds=2, random_state=42)
    assert "accuracy_mean" in cv_res
    assert "accuracy_std" in cv_res
    assert 0.0 <= cv_res["accuracy_mean"] <= 1.0
