from llm_agent.inference import prepare_model_messages


def test_prepare_model_messages():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_daily_total",
                        "arguments": '{"date": "2026-10-10"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"total_calories": 300}',
        },
    ]

    result = prepare_model_messages(messages)

    assert result[0]["tool_calls"][0]["function"]["arguments"] == {"date": "2026-10-10"}

    assert result[1] == messages[1]

    # Original messages must remain unchanged.
    assert (
        messages[0]["tool_calls"][0]["function"]["arguments"]
        == '{"date": "2026-10-10"}'
    )
