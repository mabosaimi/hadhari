from pathlib import Path

import polars as pl
import pytest

from data.repository import LocalFileMessageRepository, MessageRepository


def test_local_file_repository_parquet(tmp_path: Path) -> None:
    parquet_path = tmp_path / "test_data.parquet"
    data = pl.DataFrame({
        "id": ["1", "2", "3"],
        "raw_message": ["رسالة 1", "رسالة 2", "رسالة 3"],
        "label": [0, 1, 0],
        "validated": [True, True, False],
    })
    data.write_parquet(parquet_path)

    repo: MessageRepository = LocalFileMessageRepository(parquet_path)

    df_validated = repo.load_messages(validated_only=True)
    assert len(df_validated) == 2
    assert df_validated["id"].to_list() == ["1", "2"]

    df_all = repo.load_messages(validated_only=False)
    assert len(df_all) == 3


def test_local_file_repository_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_data.csv"
    data = pl.DataFrame({
        "id": ["1", "2"],
        "raw_message": ["مرحبا", "اعلان"],
        "label": [0, 1],
        "validated": [True, True],
    })
    data.write_csv(csv_path)

    repo = LocalFileMessageRepository(csv_path)
    df = repo.load_messages()
    assert len(df) == 2
    assert df["label"].to_list() == [0, 1]


def test_local_file_repository_not_found(tmp_path: Path) -> None:
    repo = LocalFileMessageRepository(tmp_path / "missing.parquet")
    with pytest.raises(FileNotFoundError):
        repo.load_messages()
