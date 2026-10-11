import json
from datetime import date

import pytest
from pydantic import ValidationError

from llm_agent.schemas import EntryInput, EntryUpdate
from llm_agent.storage import (
    add_entry,
    delete_entry,
    get_calorie_summary,
    list_entries,
    update_entry,
)


@pytest.fixture
def sample_entries(tmp_path):
    records = []

    for entry_date, description in [
        ("2026-09-30", "Oatmeal"),
        ("2026-10-01", "Sandwich"),
        ("2026-10-05", "Pizza"),
        ("2026-10-11", "Salad"),
    ]:
        entry = EntryInput(
            date=entry_date,
            description=description,
            calories=300,
            category="other",
            tags=[],
        )

        records.append(add_entry(entry, data_dir=tmp_path))

    return records


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


def test_list_entries_empty_directory(tmp_path):
    assert list_entries(data_dir=tmp_path) == []


def test_list_entries_all_months(tmp_path, sample_entries):
    result = list_entries(data_dir=tmp_path)

    assert result == sample_entries


def test_list_entries_date_range(tmp_path, sample_entries):
    result = list_entries(
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 5),
        data_dir=tmp_path,
    )

    assert [entry["description"] for entry in result] == [
        "Sandwich",
        "Pizza",
    ]


def test_list_entries_start_date_only(tmp_path, sample_entries):
    result = list_entries(
        start_date=date(2026, 10, 5),
        data_dir=tmp_path,
    )

    assert [entry["description"] for entry in result] == [
        "Pizza",
        "Salad",
    ]


def test_list_entries_end_date_only(tmp_path, sample_entries):
    result = list_entries(
        end_date=date(2026, 10, 1),
        data_dir=tmp_path,
    )

    assert [entry["description"] for entry in result] == [
        "Oatmeal",
        "Sandwich",
    ]


def test_list_entries_invalid_date_range(tmp_path):
    with pytest.raises(ValueError, match="start_date"):
        list_entries(
            start_date=date(2026, 10, 11),
            end_date=date(2026, 10, 1),
            data_dir=tmp_path,
        )


def test_update_entry_changes_only_requested_fields(tmp_path):
    original = add_entry(
        EntryInput(
            date="2026-10-11",
            description="Oatmeal",
            calories=300,
            category="grains",
            tags=["breakfast"],
        ),
        data_dir=tmp_path,
    )

    updated = update_entry(
        original["id"],
        EntryUpdate(calories=450),
        data_dir=tmp_path,
    )

    assert updated["id"] == original["id"]
    assert updated["calories"] == 450
    assert updated["description"] == "Oatmeal"
    assert updated["category"] == "grains"
    assert updated["tags"] == ["breakfast"]

    assert list_entries(data_dir=tmp_path) == [updated]


def test_update_entry_moves_between_months(tmp_path):
    original = add_entry(
        EntryInput(
            date="2026-09-30",
            description="Pizza",
            calories=800,
            category="fast_food",
            tags=[],
        ),
        data_dir=tmp_path,
    )

    updated = update_entry(
        original["id"],
        EntryUpdate(date="2026-10-01"),
        data_dir=tmp_path,
    )

    assert updated["id"] == original["id"]
    assert updated["date"] == "2026-10-01"

    september = json.loads((tmp_path / "2026-09.json").read_text(encoding="utf-8"))
    october = json.loads((tmp_path / "2026-10.json").read_text(encoding="utf-8"))

    assert september["entries"] == []
    assert october["entries"] == [updated]


def test_update_entry_not_found(tmp_path):
    with pytest.raises(KeyError, match="Entry not found"):
        update_entry(
            "unknown-id",
            EntryUpdate(calories=500),
            data_dir=tmp_path,
        )


def test_update_entry_rejects_invalid_values():
    with pytest.raises(ValidationError):
        EntryUpdate(calories=-100)


def test_update_entry_rejects_empty_changes(tmp_path):
    with pytest.raises(ValueError, match="No fields"):
        update_entry(
            "some-id",
            EntryUpdate(),
            data_dir=tmp_path,
        )


def test_update_entry_rejects_null_fields(tmp_path):
    with pytest.raises(ValueError, match="cannot be null"):
        update_entry(
            "some-id",
            EntryUpdate(description=None),
            data_dir=tmp_path,
        )


def test_delete_entry(tmp_path):
    entry = add_entry(
        EntryInput(
            date="2026-10-11",
            description="Oatmeal",
            calories=300,
            category="grains",
            tags=["breakfast"],
        ),
        data_dir=tmp_path,
    )

    deleted = delete_entry(entry["id"], data_dir=tmp_path)

    assert deleted == entry
    assert list_entries(data_dir=tmp_path) == []

    data = json.loads((tmp_path / "2026-10.json").read_text(encoding="utf-8"))

    assert data["entries"] == []


def test_delete_entry_preserves_other_entries(tmp_path):
    first = add_entry(
        EntryInput(
            date="2026-10-11",
            description="Oatmeal",
            calories=300,
            category="grains",
            tags=[],
        ),
        data_dir=tmp_path,
    )

    second = add_entry(
        EntryInput(
            date="2026-10-11",
            description="Pizza",
            calories=800,
            category="fast_food",
            tags=[],
        ),
        data_dir=tmp_path,
    )

    delete_entry(first["id"], data_dir=tmp_path)

    assert list_entries(data_dir=tmp_path) == [second]


def test_delete_entry_not_found(tmp_path):
    with pytest.raises(KeyError, match="Entry not found"):
        delete_entry("unknown-id", data_dir=tmp_path)


def test_delete_entry_does_not_modify_other_months(tmp_path):
    september = add_entry(
        EntryInput(
            date="2026-09-30",
            description="Oatmeal",
            calories=300,
            category="grains",
            tags=[],
        ),
        data_dir=tmp_path,
    )

    october = add_entry(
        EntryInput(
            date="2026-10-11",
            description="Pizza",
            calories=800,
            category="fast_food",
            tags=[],
        ),
        data_dir=tmp_path,
    )

    delete_entry(october["id"], data_dir=tmp_path)

    assert list_entries(data_dir=tmp_path) == [september]


def test_calorie_summary_across_months(tmp_path, sample_entries):
    summary = get_calorie_summary(
        start_date=date(2026, 9, 30),
        end_date=date(2026, 10, 5),
        data_dir=tmp_path,
    )

    assert summary["total_calories"] == 900
    assert summary["average_daily_calories"] == 150
    assert summary["by_category"] == {"other": 900}

    assert summary["daily_totals"] == {
        "2026-09-30": 300,
        "2026-10-01": 300,
        "2026-10-02": 0,
        "2026-10-03": 0,
        "2026-10-04": 0,
        "2026-10-05": 300,
    }


def test_calorie_summary_seven_day_average(tmp_path, sample_entries):
    summary = get_calorie_summary(
        start_date=date(2026, 10, 5),
        end_date=date(2026, 10, 11),
        data_dir=tmp_path,
    )

    assert summary["total_calories"] == 600
    assert summary["average_daily_calories"] == 85.71


def test_calorie_summary_by_category(tmp_path):
    for description, calories, category in [
        ("Oatmeal", 300, "grains"),
        ("Pizza", 800, "fast_food"),
        ("Rice", 200, "grains"),
    ]:
        add_entry(
            EntryInput(
                date="2026-10-11",
                description=description,
                calories=calories,
                category=category,
                tags=[],
            ),
            data_dir=tmp_path,
        )

    summary = get_calorie_summary(
        start_date=date(2026, 10, 11),
        end_date=date(2026, 10, 11),
        data_dir=tmp_path,
    )

    assert summary["total_calories"] == 1300
    assert summary["by_category"] == {
        "fast_food": 800,
        "grains": 500,
    }


def test_calorie_summary_empty_range(tmp_path):
    summary = get_calorie_summary(
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 3),
        data_dir=tmp_path,
    )

    assert summary["total_calories"] == 0
    assert summary["average_daily_calories"] == 0
    assert summary["by_category"] == {}
    assert len(summary["daily_totals"]) == 3
    assert all(calories == 0 for calories in summary["daily_totals"].values())


def test_calorie_summary_invalid_range(tmp_path):
    with pytest.raises(ValueError, match="start_date"):
        get_calorie_summary(
            start_date=date(2026, 10, 11),
            end_date=date(2026, 10, 1),
            data_dir=tmp_path,
        )
