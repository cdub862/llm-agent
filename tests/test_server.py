import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from llm_agent.config import MODEL_ID
from llm_agent.server import app


def test_chat_completion_with_tool_call():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_entry",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calories": {"type": "integer", "minimum": 0},
                    },
                    "required": ["calories"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    mock_result = {
        "content": (
            '<tool_call>{"name": "add_entry", '
            '"arguments": {"calories": 300}}</tool_call>'
        ),
        "prompt_tokens": 20,
        "completion_tokens": 15,
        "finish_reason": "stop",
    }

    with patch(
        "llm_agent.server.generate",
        return_value=mock_result,
    ) as mock_generate:
        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Log 300 calories"}],
                "tools": tools,
            },
        )

    assert response.status_code == 200

    result = response.json()
    choice = result["choices"][0]

    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None

    tool_call = choice["message"]["tool_calls"][0]

    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "add_entry"
    assert json.loads(tool_call["function"]["arguments"]) == {"calories": 300}

    assert result["usage"]["total_tokens"] == 35
    mock_generate.assert_called_once()


def test_chat_completion_rejects_invalid_tool_arguments():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_entry",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calories": {"type": "integer", "minimum": 0},
                    },
                    "required": ["calories"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    mock_result = {
        "content": (
            '<tool_call>{"name": "add_entry", '
            '"arguments": {"calories": -100}}</tool_call>'
        ),
        "prompt_tokens": 20,
        "completion_tokens": 15,
        "finish_reason": "stop",
    }

    with patch(
        "llm_agent.server.generate",
        return_value=mock_result,
    ):
        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Log -100 calories"}],
                "tools": tools,
            },
        )

    assert response.status_code == 502
    assert "Invalid model-generated tool call" in response.json()["detail"]
