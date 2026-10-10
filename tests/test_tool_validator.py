import pytest
from jsonschema import ValidationError

from llm_agent.tool_validator import validate_tool_call


@pytest.fixture
def tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "add_entry",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "format": "date",
                        },
                        "description": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "calories": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "category": {
                            "type": "string",
                            "enum": ["grains", "snacks", "other"],
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "date",
                        "description",
                        "calories",
                        "category",
                        "tags",
                    ],
                    "additionalProperties": False,
                },
            },
        }
    ]


@pytest.fixture
def valid_call():
    return {
        "name": "add_entry",
        "arguments": {
            "date": "2026-10-10",
            "description": "Oatmeal",
            "calories": 300,
            "category": "grains",
            "tags": ["breakfast"],
        },
    }


def test_valid_tool_call(tools, valid_call):
    result = validate_tool_call(valid_call, tools)

    assert result == valid_call


def test_unknown_tool(tools, valid_call):
    valid_call["name"] = "delete_everything"

    with pytest.raises(ValueError, match="Unknown tool"):
        validate_tool_call(valid_call, tools)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("date", "not-a-date"),
        ("description", ""),
        ("calories", -100),
        ("calories", "300"),
        ("category", "snack"),
        ("tags", "breakfast"),
    ],
)
def test_invalid_arguments(tools, valid_call, field, value):
    valid_call["arguments"][field] = value

    with pytest.raises(ValidationError):
        validate_tool_call(valid_call, tools)


def test_missing_required_field(tools, valid_call):
    del valid_call["arguments"]["calories"]

    with pytest.raises(ValidationError):
        validate_tool_call(valid_call, tools)


def test_unexpected_field(tools, valid_call):
    valid_call["arguments"]["unknown"] = True

    with pytest.raises(ValidationError):
        validate_tool_call(valid_call, tools)
