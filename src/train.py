import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

import hadhari
from hadhari.data.loader import load_messages
from hadhari.data.repository import LocalFileMessageRepository
from hadhari.models.trainer import build_model, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hadhari Spam Detection Model Training Entrypoint")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to local Parquet/CSV dataset snapshot")
    parser.add_argument(
        "--export-snapshot",
        type=str,
        default=None,
        help="Export loaded Firestore dataset to local Parquet file (gitignored)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (default: 0.2)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--no-save", action="store_true", help="Do not save model and evaluation artifacts")
    parser.add_argument("--model", type=str, default="logreg", help="Model type: logreg | linearsvc (default: logreg)")
    parser.add_argument("--max-features", type=int, default=1000, help="TF-IDF max features (default: 1000)")
    parser.add_argument("--ngram-range", type=int, nargs=2, default=[1, 2], help="TF-IDF ngram range (default: 1 2)")
    parser.add_argument("--experiment", type=str, default="hadhari", help="MLFlow experiment name (default: hadhari)")
    parser.add_argument("--run-name", type=str, default=None, help="MLFlow run name (default: auto-generated)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLFlow tracking")

    args = parser.parse_args()

    if args.dataset_path:
        logger.info("Loading dataset from local snapshot: %s", args.dataset_path)
        repo = LocalFileMessageRepository(args.dataset_path)
        df = load_messages(repository=repo, validated_only=True)
    else:
        logger.info("Loading dataset from Firestore...")
        df = load_messages(validated_only=True)

    logger.info("Loaded %d validated messages.", len(df))

    if args.export_snapshot:
        export_path = Path(args.export_snapshot)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(export_path)
        logger.info("Exported private dataset snapshot to %s", export_path)

    X = df["raw_message"]
    y = df["label"]

    TOKEN_PATTERN = r"(?u)\b\w+\b"  # noqa: S105
    model = build_model(args.model)
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range),
        token_pattern=TOKEN_PATTERN,
    )

    pipeline, metrics = train(
        X,
        y,
        model=model,
        vectorizer=vectorizer,
        test_size=args.test_size,
        random_state=args.seed,
        cv_folds=args.cv_folds,
        save_model=not args.no_save,
        verbose=True,
    )

    logger.info("Training complete. Test Accuracy: %.2f%%", metrics["test_accuracy"] * 100)

    if not args.no_mlflow:
        _log_mlflow_run(args, pipeline, metrics)


def _log_mlflow_run(args: argparse.Namespace, pipeline: Any, metrics: dict[str, Any]) -> None:
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or f"{args.model}-{args.max_features}f-{args.ngram_range[0]}-{args.ngram_range[1]}ng"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("hadhari_version", hadhari.__version__)
        mlflow.set_tag("sklearn_version", sklearn.__version__)

        if args.dataset_path:
            dataset_hash = hashlib.sha256(Path(args.dataset_path).read_bytes()).hexdigest()
            mlflow.set_tag("dataset_sha256", dataset_hash)
            mlflow.set_tag("dataset_path", args.dataset_path)

        mlflow.log_params({
            "model_type": args.model,
            "max_features": args.max_features,
            "ngram_range": str(tuple(args.ngram_range)),
            "random_state": args.seed,
            "test_size": args.test_size,
            "cv_folds": args.cv_folds,
            "dataset_size": metrics["dataset_size"],
            "train_size": metrics["train_size"],
        })

        flat_metrics = {
            "test_accuracy": metrics["test_accuracy"],
        }
        flat_metrics.update(metrics["cv_metrics"])
        mlflow.log_metrics(flat_metrics)

        mlflow.sklearn.log_model(  # pyright: ignore[reportPrivateImportUsage]
            pipeline,
            "pipeline",
            serialization_format="cloudpickle",
        )

    logger.info("MLFlow run logged to experiment %r as %r", args.experiment, run_name)


if __name__ == "__main__":
    main()
