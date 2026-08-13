from pathlib import Path
from typing import Protocol

import polars as pl

from db.firestore import get_db


class MessageRepository(Protocol):
    """Abstract repository interface for loading dataset messages."""

    def load_messages(self, *, validated_only: bool = True) -> pl.DataFrame:
        """Load messages into a Polars DataFrame with columns ['id', 'raw_message', 'label']."""
        ...


class FirestoreMessageRepository:
    """Firestore implementation of MessageRepository."""

    def load_messages(self, *, validated_only: bool = True) -> pl.DataFrame:
        db = get_db()
        messages_ref = db.collection("messages")
        docs = messages_ref.stream()

        messages = []
        for doc in docs:
            data = doc.to_dict()
            is_validated = data.get("validated")

            if validated_only and not is_validated:
                continue

            message = {
                "id": doc.id,
                "raw_message": data.get("raw_message"),
                "label": data.get("prediction"),
            }

            if not validated_only:
                message["validated"] = is_validated

            messages.append(message)

        return pl.DataFrame(messages)


class LocalFileMessageRepository:
    """Local file (Parquet/CSV) implementation of MessageRepository for reproducible training."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load_messages(self, *, validated_only: bool = True) -> pl.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        if self.file_path.suffix == ".parquet":
            df = pl.read_parquet(self.file_path)
        elif self.file_path.suffix in {".csv", ".tsv"}:
            df = pl.read_csv(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")

        if validated_only and "validated" in df.columns:
            df = df.filter(pl.col("validated") == True)  # noqa: E712

        return df
