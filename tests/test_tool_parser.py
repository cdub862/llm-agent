import json

import pytest

from llm_agent.tool_parser import parse_tool_call, to_openai_tool_call


def test_parse_valid_tool_call():
    content = """
    <tool_call>
    {
        "name": "add_entry",
        "arguments": {
            "date": "2026-10-10",
            "description": "Cheeseburger with fries",
            "calories": 850,
            "category": "fast_food",
            "tags": ["beef", "fried", "restaurant"]
        }
    }
    </tool_call>
    """

    result = parse_tool_call(content)

    assert result == {
        "name": "add_entry",
        "arguments": {
            "date": "2026-10-10",
            "description": "Cheeseburger with fries",
            "calories": 850,
            "category": "fast_food",
            "tags": ["beef", "fried", "restaurant"],
        },
    }


def test_parse_plain_text():
    assert parse_tool_call("Hello, how can I help?") is None


def test_parse_invalid_json():
    content = '<tool_call>{"name": invalid}</tool_call>'

    with pytest.raises(json.JSONDecodeError):
        parse_tool_call(content)


@pytest.mark.parametrize(
    "content",
    [
        "<tool_call>[]</tool_call>",
        '<tool_call>{"name": 123, "arguments": {}}</tool_call>',
        '<tool_call>{"name": "add_entry", "arguments": []}</tool_call>',
        '<tool_call>{"name": "add_entry"}</tool_call>',
    ],
)
def test_parse_invalid_types(content):
    with pytest.raises(TypeError):
        parse_tool_call(content)


@pytest.mark.parametrize("name", ["", " ", "\t"])
def test_parse_empty_function_name(name):
    content = f"<tool_call>{json.dumps({'name': name, 'arguments': {}})}</tool_call>"

    with pytest.raises(ValueError):
        parse_tool_call(content)


def test_convert_to_openai_tool_call():
    parsed = {
        "name": "add_entry",
        "arguments": {
            "date": "2026-10-10",
            "description": "Oatmeal",
            "calories": 300,
            "category": "grains",
            "tags": ["breakfast"],
        },
    }

    result = to_openai_tool_call(parsed)

    assert result["id"].startswith("call_")
    assert result["type"] == "function"
    assert result["function"]["name"] == "add_entry"
    assert json.loads(result["function"]["arguments"]) == parsed["arguments"]
