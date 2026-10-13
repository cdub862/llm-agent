import json
from unittest.mock import Mock

import pytest
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


def test_agent_rejects_invalid_tool_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    client = Mock()
    client.chat.completions.create.side_effect = [
        make_completion(
            tool_name="add_entry",
            arguments={
                "date": "2026-10-11",
                "description": "Oatmeal",
                "calories": -100,
                "category": "grains",
                "tags": ["breakfast"],
            },
        ),
        make_completion(
            content="I cannot log negative calories.",
        ),
    ]

    answer = run_agent(
        client=client,
        messages=[
            {
                "role": "user",
                "content": "Log -100 calories of oatmeal.",
            }
        ],
        tools=TOOLS,
        handlers=HANDLERS,
        max_steps=5,
    )

    assert answer == "I cannot log negative calories."
    assert client.chat.completions.create.call_count == 2

    assert not (tmp_path / "data" / "2026-10.json").exists()

    second_call = client.chat.completions.create.call_args_list[1]
    history = second_call.kwargs["messages"]

    error_result = json.loads(history[-1]["content"])

    assert error_result["ok"] is False
    assert error_result["error"]["type"] == "invalid_tool_arguments"


def test_agent_recovers_from_invalid_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    client = Mock()
    client.chat.completions.create.side_effect = [
        make_completion(
            tool_name="add_entry",
            arguments={
                "date": "2026-10-11",
                "description": "Oatmeal",
                "calories": -100,
                "category": "grains",
                "tags": [],
            },
        ),
        make_completion(
            tool_name="add_entry",
            arguments={
                "date": "2026-10-11",
                "description": "Oatmeal",
                "calories": 300,
                "category": "grains",
                "tags": [],
            },
        ),
        make_completion(
            content="Successfully logged 300 calories.",
        ),
    ]

    answer = run_agent(
        client=client,
        messages=[
            {
                "role": "user",
                "content": "Log 300 calories of oatmeal.",
            }
        ],
        tools=TOOLS,
        handlers=HANDLERS,
        max_steps=5,
    )

    assert answer == "Successfully logged 300 calories."
    assert client.chat.completions.create.call_count == 3

    entries = list_entries(data_dir=tmp_path / "data")

    assert len(entries) == 1
    assert entries[0]["calories"] == 300

    final_call = client.chat.completions.create.call_args_list[-1]
    history = final_call.kwargs["messages"]

    tool_results = [message for message in history if message["role"] == "tool"]

    assert len(tool_results) == 2

    first_result = json.loads(tool_results[0]["content"])
    second_result = json.loads(tool_results[1]["content"])

    assert first_result["ok"] is False
    assert first_result["error"]["type"] == "invalid_tool_arguments"
    assert second_result["calories"] == 300


def test_agent_stops_after_max_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    client = Mock()

    # Qwen repeatedly generates the same invalid tool call.
    client.chat.completions.create.return_value = make_completion(
        tool_name="add_entry",
        arguments={
            "date": "2026-10-11",
            "description": "Oatmeal",
            "calories": -100,
            "category": "grains",
            "tags": [],
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Agent exceeded maximum steps",
    ):
        run_agent(
            client=client,
            messages=[
                {
                    "role": "user",
                    "content": "Log 300 calories of oatmeal.",
                }
            ],
            tools=TOOLS,
            handlers=HANDLERS,
            max_steps=3,
        )

    assert client.chat.completions.create.call_count == 3

    # No invalid entries should have been persisted.
    assert not (tmp_path / "data" / "2026-10.json").exists()

    # The last model request should contain the previous errors.
    final_request = client.chat.completions.create.call_args_list[-1]
    history = final_request.kwargs["messages"]

    tool_results = [message for message in history if message["role"] == "tool"]

    assert len(tool_results) == 3

    for result in tool_results:
        content = json.loads(result["content"])
        assert content["ok"] is False
        assert content["error"]["type"] == "invalid_tool_arguments"


def test_agent_rejects_nonpositive_max_steps():
    client = Mock()

    with pytest.raises(
        ValueError,
        match="max_steps must be positive",
    ):
        run_agent(
            client=client,
            messages=[],
            tools=TOOLS,
            handlers=HANDLERS,
            max_steps=0,
        )

    client.chat.completions.create.assert_not_called()
