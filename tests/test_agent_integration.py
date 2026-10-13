import json
from unittest.mock import Mock

from openai.types.chat import ChatCompletion

from llm_agent.agent import run_agent
from llm_agent.storage import list_entries
from llm_agent.tools import HANDLERS, TOOLS


def make_completion(content=None, tool_name=None, arguments=None):
    message = {
        "role": "assistant",
        "content": content,
    }

    if tool_name:
        message["tool_calls"] = [
            {
                "id": f"call_{tool_name}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            }
        ]

    return ChatCompletion.model_validate(
        {
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_name else "stop",
                    "message": message,
                }
            ],
        }
    )


def test_agent_adds_entry_and_calculates_total(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    client = Mock()

    client.chat.completions.create.side_effect = [
        make_completion(
            tool_name="add_entry",
            arguments={
                "date": "2026-10-11",
                "description": "Oatmeal",
                "calories": 300,
                "category": "grains",
                "tags": ["breakfast"],
            },
        ),
        make_completion(
            tool_name="get_calorie_summary",
            arguments={
                "start_date": "2026-10-11",
                "end_date": "2026-10-11",
            },
        ),
        make_completion(
            content="You logged 300 calories on October 11.",
        ),
    ]

    answer = run_agent(
        client=client,
        messages=[
            {
                "role": "user",
                "content": (
                    "Log 300 calories of oatmeal on October 11, 2026, "
                    "and tell me my total for that day."
                ),
            }
        ],
        tools=TOOLS,
        handlers=HANDLERS,
        max_steps=5,
    )

    assert answer == "You logged 300 calories on October 11."
    assert client.chat.completions.create.call_count == 3

    entries = list_entries(data_dir=tmp_path / "data")

    assert len(entries) == 1
    assert entries[0]["description"] == "Oatmeal"
    assert entries[0]["calories"] == 300

    final_call = client.chat.completions.create.call_args_list[-1]
    history = final_call.kwargs["messages"]

    summary = json.loads(history[-1]["content"])

    assert summary["total_calories"] == 300
    assert history[-1]["tool_call_id"] == "call_get_calorie_summary"
