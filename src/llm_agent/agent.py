import json
from collections.abc import Callable

from openai import OpenAI

from llm_agent.config import MODEL_ID
from llm_agent.tool_validator import validate_tool_call


def run_agent(
    client: OpenAI,
    messages: list[dict],
    tools: list[dict],
    handlers: dict[str, Callable],
    max_steps: int = 5,
) -> str:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    history = [message.copy() for message in messages]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=history,
            tools=tools,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""

        history.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [call.model_dump() for call in message.tool_calls],
            }
        )

        for call in message.tool_calls:
            parsed = {
                "name": call.function.name,
                "arguments": json.loads(call.function.arguments),
            }

            validate_tool_call(parsed, tools)

            handler = handlers.get(parsed["name"])

            if handler is None:
                raise ValueError(f"No handler registered for {parsed['name']}")

            result = handler(**parsed["arguments"])

            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )

    raise RuntimeError("Agent exceeded maximum steps")
