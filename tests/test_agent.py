import json
from unittest.mock import Mock

from llm_agent.agent import run_agent

from tests.helpers import to_namespace


def test_agent_executes_tool_and_returns_answer():
    tool_call_data = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "get_daily_total",
            "arguments": '{"date": "2026-10-10"}',
        },
    }

    tool_call = to_namespace(tool_call_data)
    tool_call.model_dump = Mock(return_value=tool_call_data)

    client = Mock()
    client.chat.completions.create.side_effect = [
        to_namespace(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [tool_call],
                        }
                    }
                ]
            }
        ),
        to_namespace(
            {
                "choices": [
                    {
                        "message": {
                            "content": "You logged 300 calories.",
                            "tool_calls": None,
                        }
                    }
                ]
            }
        ),
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_daily_total",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                    },
                    "required": ["date"],
                },
            },
        }
    ]

    handler = Mock(return_value={"total": 300})

    result = run_agent(
        client=client,
        messages=[{"role": "user", "content": "How many calories today?"}],
        tools=tools,
        handlers={"get_daily_total": handler},
    )

    assert result == "You logged 300 calories."
    handler.assert_called_once_with(date="2026-10-10")
    assert client.chat.completions.create.call_count == 2

    second_request = client.chat.completions.create.call_args_list[1].kwargs

    assert second_request["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": json.dumps({"total": 300}),
    }
