import argparse
import logging
from pathlib import Path

from hadhari.data.loader import load_messages
from hadhari.data.repository import LocalFileMessageRepository
from hadhari.models.trainer import train

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

    _pipeline, metrics = train(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        cv_folds=args.cv_folds,
        save_model=not args.no_save,
        verbose=True,
    )

    logger.info("Training complete. Test Accuracy: %.2f%%", metrics["test_accuracy"] * 100)


if __name__ == "__main__":
    main()
