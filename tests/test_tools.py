import json

import pytest
from pydantic import ValidationError

from llm_agent.schemas import (
    CalorieSummaryInput,
    DeleteEntryToolInput,
    EntryInput,
    ListEntriesInput,
    UpdateEntryToolInput,
)
from llm_agent.tools import (
    ADD_ENTRY_TOOL,
    HANDLERS,
    LIST_ENTRIES_TOOL,
    TOOLS,
    ListEntriesInput,
    add_entry_tool,
    delete_entry_tool,
    get_calorie_summary_tool,
    list_entries_tool,
    update_entry_tool,
)


@pytest.fixture
def calorie_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    first = add_entry_tool(
        date="2026-10-10",
        description="Oatmeal",
        calories=300,
        category="grains",
        tags=["breakfast"],
    )

    second = add_entry_tool(
        date="2026-10-11",
        description="Pizza",
        calories=800,
        category="fast_food",
        tags=["restaurant"],
    )

    return first, second


def test_add_entry_tool_schema():
    assert ADD_ENTRY_TOOL["function"]["name"] == "add_entry"
    assert ADD_ENTRY_TOOL["function"]["parameters"] == EntryInput.model_json_schema()


def test_add_entry_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = add_entry_tool(
        date="2026-10-11",
        description="Tortilla cheese salami",
        calories=300,
        category="snacks",
        tags=["cheese", "meat"],
    )

    path = tmp_path / "data" / "2026-10.json"

    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["entries"] == [result]
    assert result["id"]
    assert result["date"] == "2026-10-11"
    assert result["description"] == "Tortilla cheese salami"
    assert result["calories"] == 300
    assert result["category"] == "snacks"
    assert result["tags"] == ["cheese", "meat"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("date", "invalid-date"),
        ("description", ""),
        ("calories", -100),
        ("calories", "300"),
        ("category", "breakfast"),
        ("tags", "cheese"),
    ],
)
def test_add_entry_tool_rejects_invalid_data(tmp_path, monkeypatch, field, value):
    monkeypatch.chdir(tmp_path)

    arguments = {
        "date": "2026-10-11",
        "description": "Oatmeal",
        "calories": 300,
        "category": "grains",
        "tags": ["breakfast"],
    }

    arguments[field] = value

    with pytest.raises(ValidationError):
        add_entry_tool(**arguments)

    assert not (tmp_path / "data").exists()


def test_add_entry_tool_rejects_unknown_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError):
        add_entry_tool(
            date="2026-10-11",
            description="Oatmeal",
            calories=300,
            category="grains",
            tags=[],
            unknown_field=True,
        )

    assert not (tmp_path / "data").exists()


def test_list_entries_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    add_entry_tool(
        date="2026-10-11",
        description="Oatmeal",
        calories=300,
        category="grains",
        tags=["breakfast"],
    )

    add_entry_tool(
        date="2026-10-10",
        description="Pizza",
        calories=800,
        category="fast_food",
        tags=[],
    )

    result = list_entries_tool(
        start_date="2026-10-11",
        end_date="2026-10-11",
    )

    assert len(result) == 1
    assert result[0]["description"] == "Oatmeal"


def test_list_entries_tool_schema():
    assert (
        LIST_ENTRIES_TOOL["function"]["parameters"]
        == ListEntriesInput.model_json_schema()
    )


def test_list_entries_tool_invalid_date():
    with pytest.raises(ValidationError):
        list_entries_tool(start_date="invalid")


def test_all_tool_schemas():
    expected = {
        "add_entry": EntryInput,
        "list_entries": ListEntriesInput,
        "update_entry": UpdateEntryToolInput,
        "delete_entry": DeleteEntryToolInput,
        "get_calorie_summary": CalorieSummaryInput,
    }

    assert len(TOOLS) == len(expected)
    assert set(HANDLERS) == set(expected)

    for tool in TOOLS:
        function = tool["function"]

        assert function["parameters"] == expected[function["name"]].model_json_schema()


def test_list_entries_tool(calorie_entries):
    result = list_entries_tool(
        start_date="2026-10-11",
        end_date="2026-10-11",
    )

    assert len(result) == 1
    assert result[0]["description"] == "Pizza"


def test_update_entry_tool(calorie_entries):
    first, _ = calorie_entries

    updated = update_entry_tool(
        entry_id=first["id"],
        calories=450,
    )

    assert updated["id"] == first["id"]
    assert updated["calories"] == 450
    assert updated["description"] == "Oatmeal"

    records = list_entries_tool()
    assert records[0]["calories"] == 450


def test_delete_entry_tool(calorie_entries):
    first, second = calorie_entries

    deleted = delete_entry_tool(entry_id=first["id"])

    assert deleted == first
    assert list_entries_tool() == [second]


def test_calorie_summary_tool(calorie_entries):
    result = get_calorie_summary_tool(
        start_date="2026-10-09",
        end_date="2026-10-11",
    )

    assert result["total_calories"] == 1100
    assert result["average_daily_calories"] == 366.67
    assert result["by_category"] == {
        "fast_food": 800,
        "grains": 300,
    }
    assert result["daily_totals"]["2026-10-09"] == 0


def test_update_entry_tool_rejects_missing_id():
    with pytest.raises(ValidationError):
        update_entry_tool(calories=450)


def test_delete_entry_tool_rejects_missing_id():
    with pytest.raises(ValidationError):
        delete_entry_tool()


def test_calorie_summary_tool_rejects_invalid_date():
    with pytest.raises(ValidationError):
        get_calorie_summary_tool(
            start_date="invalid",
            end_date="2026-10-11",
        )
