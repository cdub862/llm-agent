from openai import OpenAI

from llm_agent.agent import run_agent


def get_daily_total(date: str) -> dict:
    return {"date": date, "total_calories": 300}


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_daily_total",
            "description": "Get the total logged calories for a date.",
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

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
)

answer = run_agent(
    client=client,
    messages=[
        {
            "role": "user",
            "content": "How many calories did I log on 2026-10-10?",
        }
    ],
    tools=tools,
    handlers={"get_daily_total": get_daily_total},
)

print(answer)
