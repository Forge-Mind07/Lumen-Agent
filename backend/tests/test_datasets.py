from pathlib import Path

from app.config import Settings
from app.datasets import register_dataset
from app.storage import Storage


def test_registers_and_profiles_csv(settings: Settings, storage: Storage) -> None:
    fixture = Path(__file__).parent / "fixtures" / "sales.csv"
    with fixture.open("rb") as source:
        dataset = register_dataset(
            filename="sales.csv", source=source, storage=storage, settings=settings
        )

    assert dataset.status == "ready"
    assert dataset.row_count == 6
    assert dataset.column_count == 5
    assert dataset.profile is not None
    assert {column.name for column in dataset.profile.columns} == {
        "order_date",
        "region",
        "product",
        "revenue",
        "units",
    }
    revenue = next(column for column in dataset.profile.columns if column.name == "revenue")
    assert revenue.minimum == 800
    assert revenue.maximum == 1600
    assert Path(dataset.original_path).exists()
    assert Path(dataset.duckdb_path).exists()
