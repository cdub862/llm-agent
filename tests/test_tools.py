import json

import pytest
from pydantic import ValidationError

from llm_agent.schemas import EntryInput
from llm_agent.tools import ADD_ENTRY_TOOL, add_entry_tool


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
