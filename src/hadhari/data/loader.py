import polars as pl

from hadhari.data.repository import FirestoreMessageRepository, MessageRepository


def load_messages(
    repository: MessageRepository | None = None,
    *,
    validated_only: bool = True,
) -> pl.DataFrame:
    """Load messages using the specified repository or defaulting to Firestore."""
    repo = repository if repository is not None else FirestoreMessageRepository()
    df = repo.load_messages(validated_only=validated_only)
    return df.unique(subset=["raw_message"], maintain_order=True)
