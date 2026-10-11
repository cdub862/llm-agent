import json

import pytest
from pydantic import ValidationError

from llm_agent.schemas import EntryInput
from llm_agent.storage import add_entry


def test_add_entry_creates_monthly_file(tmp_path):
    entry = EntryInput(
        date="2026-10-11",
        description="Tortilla cheese salami",
        calories=300,
        category="snacks",
        tags=["cheese", "meat"],
    )

    result = add_entry(entry, data_dir=tmp_path)

    path = tmp_path / "2026-10.json"

    assert path.exists()
    assert result["id"]
    assert result["calories"] == 300

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["entries"] == [result]


def test_add_entry_appends_to_existing_month(tmp_path):
    for description in ["Oatmeal", "Sandwich"]:
        entry = EntryInput(
            date="2026-10-11",
            description=description,
            calories=300,
            category="grains",
            tags=[],
        )

        add_entry(entry, data_dir=tmp_path)

    data = json.loads((tmp_path / "2026-10.json").read_text(encoding="utf-8"))

    assert len(data["entries"]) == 2
    assert data["entries"][0]["id"] != data["entries"][1]["id"]


def test_add_entry_separates_months(tmp_path):
    for date in ["2026-09-30", "2026-10-01"]:
        entry = EntryInput(
            date=date,
            description="Oatmeal",
            calories=300,
            category="grains",
            tags=[],
        )

        add_entry(entry, data_dir=tmp_path)

    assert (tmp_path / "2026-09.json").exists()
    assert (tmp_path / "2026-10.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("date", "invalid"),
        ("description", ""),
        ("calories", -100),
        ("calories", "300"),
        ("category", "unknown"),
        ("tags", "breakfast"),
    ],
)
def test_add_entry_rejects_invalid_data(tmp_path, field, value):
    arguments = {
        "date": "2026-10-11",
        "description": "Oatmeal",
        "calories": 300,
        "category": "grains",
        "tags": ["breakfast"],
    }

    arguments[field] = value

    with pytest.raises(ValidationError):
        entry = EntryInput(**arguments)
        add_entry(entry, data_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []
